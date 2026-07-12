from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def number(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def metrics(rows: pd.DataFrame, pnl_col: str) -> dict[str, object]:
    if rows.empty:
        return {"trades": 0, "win_rate": None, "profit_factor": None, "net_pnl": 0.0, "max_drawdown": 0.0}
    pnl = pd.to_numeric(rows[pnl_col], errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(rows)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else None),
        "net_pnl": float(pnl.sum()),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def risk_for(row: pd.Series) -> float:
    width = number(row.get("entry_width"))
    debit = number(row.get("entry_debit"))
    credit = number(row.get("entry_credit"))
    direction = str(row.get("direction") or "")
    if direction in {"Bull Call", "Bear Put"} and math.isfinite(debit) and debit > 0:
        return debit * 100.0
    if math.isfinite(width) and math.isfinite(credit) and width > credit:
        return (width - credit) * 100.0
    return 0.0


def accepted_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    if "exact_evaluated" in rows:
        rows = rows[rows["exact_evaluated"].map(truthy)]
    if "decision_pass" in rows:
        rows = rows[rows["decision_pass"].map(truthy)]
    elif "replay_guard_pass" in rows:
        rows = rows[rows["replay_guard_pass"].map(truthy)]
    rows = rows.copy()
    rows["entry_date"] = pd.to_datetime(rows["asof"], errors="coerce")
    rows["exit_date"] = pd.to_datetime(rows.get("exit_day"), errors="coerce")
    rows["exit_date"] = rows["exit_date"].fillna(pd.to_datetime(rows.get("expiry"), errors="coerce"))
    rows["risk_1x"] = rows.apply(risk_for, axis=1)
    rows["pnl_1x"] = pd.to_numeric(rows["pnl_1x"], errors="coerce")
    rows = rows.dropna(subset=["entry_date", "exit_date", "pnl_1x"])
    return rows.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


def portfolio_replay(
    rows: pd.DataFrame,
    account_value: float,
    aggregate_risk_budget: float,
    risk_per_trade_pct: float,
    max_contracts: int,
    max_ticker_share: float,
    max_sector_share: float,
) -> pd.DataFrame:
    accepted: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        one_risk = number(row.get("risk_1x"), 0.0)
        if one_risk <= 0:
            continue
        day = row["entry_date"]
        active = [item for item in accepted if item["exit_date"] >= day]
        active_risk = sum(float(item["allocated_risk"]) for item in active)
        available = max(0.0, aggregate_risk_budget - active_risk) if aggregate_risk_budget > 0 else math.inf
        per_trade_cap = account_value * risk_per_trade_pct
        contracts = min(max_contracts, int(min(available, per_trade_cap) // one_risk))
        if contracts < 1:
            continue
        ticker = str(row.get("ticker") or "UNKNOWN")
        sector = str(row.get("sector") or "UNKNOWN")
        while contracts > 0:
            proposed = one_risk * contracts
            ticker_risk = sum(float(item["allocated_risk"]) for item in active if item["ticker"] == ticker) + proposed
            sector_risk = sum(float(item["allocated_risk"]) for item in active if item["sector"] == sector) + proposed
            if ticker_risk / account_value <= max_ticker_share and sector_risk / account_value <= max_sector_share:
                break
            contracts -= 1
        if contracts < 1:
            continue
        record = row.to_dict()
        record["contracts"] = contracts
        record["allocated_risk"] = one_risk * contracts
        record["pnl_risk_sized"] = number(row.get("pnl_1x"), 0.0) * contracts
        accepted.append(record)
    return pd.DataFrame(accepted)


def monthly(rows: pd.DataFrame, pnl_col: str, target: float) -> list[dict[str, object]]:
    if rows.empty:
        return []
    work = rows.copy()
    work["month"] = work["entry_date"].dt.to_period("M").astype(str)
    result: list[dict[str, object]] = []
    for month, part in work.groupby("month", sort=True):
        block = metrics(part, pnl_col)
        block.update({"month": month, "target": target, "target_met": float(block["net_pnl"]) >= target})
        result.append(block)
    return result


def fmt_pf(value: object) -> str:
    if value is None:
        return "n/a"
    return "inf" if value == math.inf else f"{float(value):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio-capacity validation for the accepted Codex Daily V4 replay ledger.")
    parser.add_argument("--replay-detail", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--monthly-target", type=float, default=10_000.0)
    parser.add_argument("--account-value", type=float, required=True)
    parser.add_argument("--aggregate-risk-budget", type=float, default=0.0, help="Optional aggregate cap; zero matches V4's default of no configured aggregate cap.")
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument("--max-contracts", type=int, default=20)
    parser.add_argument("--max-ticker-share", type=float, default=0.20)
    parser.add_argument("--max-sector-share", type=float, default=0.40)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = accepted_rows(args.replay_detail)
    sized = portfolio_replay(base, args.account_value, args.aggregate_risk_budget, args.risk_per_trade_pct, args.max_contracts, args.max_ticker_share, args.max_sector_share)
    scenarios: dict[str, dict[str, object]] = {}
    for multiple in (1, 2, 3):
        col = f"pnl_{multiple}x"
        base[col] = base["pnl_1x"] * multiple
        scenarios[f"fixed_{multiple}x"] = metrics(base, col)
        scenarios[f"fixed_{multiple}x"]["monthly"] = monthly(base, col, args.monthly_target)

    if not sized.empty:
        entry_cost = sized.apply(
            lambda row: max(number(row.get("entry_credit"), 0.0), number(row.get("entry_debit"), 0.0)) * 100.0 * int(row["contracts"]),
            axis=1,
        )
        for label, stress in (("base", 0.0), ("worse_fill_5pct", 0.05), ("worse_fill_10pct", 0.10)):
            col = f"pnl_risk_sized_{label}"
            sized[col] = sized["pnl_risk_sized"] - entry_cost * stress
            scenarios[f"risk_sized_{label}"] = metrics(sized, col)
            scenarios[f"risk_sized_{label}"]["monthly"] = monthly(sized, col, args.monthly_target)
    else:
        scenarios["risk_sized_base"] = metrics(sized, "pnl_risk_sized")

    positive_months = [m["net_pnl"] for m in scenarios["risk_sized_base"].get("monthly", []) if m["net_pnl"] > 0]
    median_positive = float(pd.Series(positive_months).median()) if positive_months else 0.0
    payload = {
        "source": str(args.replay_detail),
        "policy": {
            "monthly_target": args.monthly_target,
            "account_value": args.account_value,
            "aggregate_risk_budget": args.aggregate_risk_budget,
            "risk_per_trade_pct": args.risk_per_trade_pct,
            "max_contracts": args.max_contracts,
            "max_ticker_share": args.max_ticker_share,
            "max_sector_share": args.max_sector_share,
            "holding_period": "entry date through actual replay exit date",
        },
        "accepted_trades": int(len(base)),
        "risk_sized_trades": int(len(sized)),
        "scenarios": scenarios,
        "feasibility": {
            "months_meeting_target": sum(bool(m["target_met"]) for m in scenarios["risk_sized_base"].get("monthly", [])),
            "months_observed": len(scenarios["risk_sized_base"].get("monthly", [])),
            "median_positive_month_pnl": median_positive,
            "target_supported": bool(scenarios["risk_sized_base"].get("monthly")) and all(bool(m["target_met"]) for m in scenarios["risk_sized_base"]["monthly"]),
        },
    }
    (args.out_dir / "portfolio_capacity_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    base.to_csv(args.out_dir / "accepted_trades_fixed_size.csv", index=False)
    sized.to_csv(args.out_dir / "accepted_trades_risk_sized.csv", index=False)

    lines = ["# Codex Daily V4 Portfolio-Capacity Validation", "", "This is a read-only validation layer. It does not change V4 selection or approval rules.", ""]
    aggregate_text = "not configured" if args.aggregate_risk_budget <= 0 else f"${args.aggregate_risk_budget:,.0f}"
    lines += ["## Capacity verdict", "", f"- Accepted replay trades: {len(base)}", f"- Risk-sized trades admitted: {len(sized)}", f"- Monthly target: ${args.monthly_target:,.0f}", f"- Account value: ${args.account_value:,.0f}", f"- Per-ticket max-loss cap: {args.risk_per_trade_pct:.1%} of account", f"- Aggregate risk budget: {aggregate_text}", f"- Months meeting target: {payload['feasibility']['months_meeting_target']}/{payload['feasibility']['months_observed']}", f"- Target supported by replay: {'YES' if payload['feasibility']['target_supported'] else 'NO'}", ""]
    lines += ["## Scenario summary", "", "| Scenario | Trades | Win | PF | Net P/L | Max DD |", "|---|---:|---:|---:|---:|---:|"]
    for name, block in scenarios.items():
        win = block.get("win_rate")
        win_text = f"{float(win):.1%}" if win is not None else "n/a"
        lines.append(f"| {name} | {block.get('trades', 0)} | {win_text} | {fmt_pf(block.get('profit_factor'))} | ${float(block.get('net_pnl', 0)):,.0f} | ${float(block.get('max_drawdown', 0)):,.0f} |")
    lines += ["", "## Risk-sized monthly results", "", "| Month | Trades | Win | PF | Net P/L | Max DD | Target met |", "|---|---:|---:|---:|---:|---:|---:|"]
    for block in scenarios["risk_sized_base"].get("monthly", []):
        win = block.get("win_rate")
        lines.append(f"| {block['month']} | {block['trades']} | {float(win):.1%} | {fmt_pf(block.get('profit_factor'))} | ${float(block['net_pnl']):,.0f} | ${float(block['max_drawdown']):,.0f} | {'YES' if block['target_met'] else 'NO'} |")
    lines += ["", "## Interpretation", "", "- Fixed-size scenarios show mechanical scaling only; they do not claim liquidity supports that size.", "- Risk-sized results enforce overlapping capital, per-trade risk, ticker concentration, and sector concentration.", "- A monthly target is supported only when every observed risk-sized month reaches it; one annualized daily board is never treated as a forecast.", ""]
    (args.out_dir / "PORTFOLIO_CAPACITY_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["feasibility"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
