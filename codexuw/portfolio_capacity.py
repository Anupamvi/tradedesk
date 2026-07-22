from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .credit_policy import assess_credit_spread
from .debit_policy import assess_debit_spread


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
    equity = pd.concat([pd.Series([0.0]), pnl.cumsum()], ignore_index=True)
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
    if "replay_guard_pass" in rows:
        rows = rows[rows["replay_guard_pass"].map(truthy)]
    rows = rows.copy()
    rows["entry_date"] = pd.to_datetime(rows["asof"], errors="coerce")
    rows["exit_date"] = pd.to_datetime(rows.get("exit_day"), errors="coerce")
    rows["exit_date"] = rows["exit_date"].fillna(pd.to_datetime(rows.get("expiry"), errors="coerce"))
    rows["risk_1x"] = rows.apply(risk_for, axis=1)
    rows["pnl_1x"] = pd.to_numeric(rows["pnl_1x"], errors="coerce")
    rows = rows.dropna(subset=["entry_date", "exit_date", "pnl_1x"])
    return rows.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


def actionable_rows(path: Path) -> pd.DataFrame:
    """Return the as-of-safe V4 Medium/High book, not the broader qualified pool."""
    raw = pd.read_csv(path, low_memory=False)
    exact_mask = pd.Series(True, index=raw.index)
    for column in ("exact_evaluated", "replay_guard_pass"):
        if column in raw.columns:
            exact_mask &= raw[column].map(truthy)
    prior_pool = raw[exact_mask].copy()
    decision_mask = exact_mask.copy()
    if "decision_pass" in raw.columns:
        decision_mask &= raw["decision_pass"].map(truthy)
    candidates = raw[decision_mask].copy()
    for frame in (prior_pool, candidates):
        frame["entry_date"] = pd.to_datetime(frame["asof"], errors="coerce")
        frame["exit_date"] = pd.to_datetime(frame.get("exit_day"), errors="coerce")
        frame["exit_date"] = frame["exit_date"].fillna(pd.to_datetime(frame.get("expiry"), errors="coerce"))
        frame["pnl_1x"] = pd.to_numeric(frame["pnl_1x"], errors="coerce")
    selected: list[dict[str, object]] = []
    for _, row in candidates.dropna(subset=["entry_date", "exit_date", "pnl_1x"]).iterrows():
        direction = str(row.get("direction") or "")
        if direction in {"Bull Call", "Bear Put"}:
            entry_policy_pass, _ = assess_debit_spread(row, live=False)
        elif direction in {"Bull Put", "Bear Call"}:
            entry_policy_pass, _ = assess_credit_spread(row, live=False)
        else:
            entry_policy_pass = False
        if not entry_policy_pass:
            continue
        prior = prior_pool[
            (prior_pool["exit_date"] < row["entry_date"])
            & prior_pool["direction"].astype(str).eq(direction)
            & prior_pool["regime"].astype(str).eq(str(row.get("regime") or ""))
        ]
        pnl = pd.to_numeric(prior["pnl_1x"], errors="coerce").dropna()
        sample = int(len(pnl))
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0.0)
        average_pnl = float(pnl.mean()) if sample else 0.0
        if sample < 12 or profit_factor < 1.25 or average_pnl <= 0:
            continue
        record = row.to_dict()
        record["prior_edge_sample_size"] = sample
        record["prior_edge_profit_factor"] = profit_factor
        record["prior_edge_avg_pnl"] = average_pnl
        record["risk_1x"] = risk_for(row)
        selected.append(record)
    if not selected:
        return pd.DataFrame(columns=[*candidates.columns, "prior_edge_sample_size", "prior_edge_profit_factor", "prior_edge_avg_pnl", "risk_1x"])
    return pd.DataFrame(selected).sort_values(["entry_date", "ticker"], kind="stable").reset_index(drop=True)


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


def build_portfolio_capacity_payload(
    base: pd.DataFrame,
    *,
    source: str,
    monthly_target: float,
    account_value: float,
    aggregate_risk_budget: float = 0.0,
    risk_per_trade_pct: float = 0.02,
    max_contracts: int = 20,
    max_ticker_share: float = 0.20,
    max_sector_share: float = 0.40,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    fixed = base.copy()
    sized = (
        portfolio_replay(
            fixed,
            account_value,
            aggregate_risk_budget,
            risk_per_trade_pct,
            max_contracts,
            max_ticker_share,
            max_sector_share,
        )
        if account_value > 0
        else pd.DataFrame()
    )
    scenarios: dict[str, dict[str, object]] = {}
    for multiple in (1, 2, 3):
        col = f"pnl_{multiple}x"
        fixed[col] = fixed["pnl_1x"] * multiple
        scenarios[f"fixed_{multiple}x"] = metrics(fixed, col)
        scenarios[f"fixed_{multiple}x"]["monthly"] = monthly(fixed, col, monthly_target)

    if not sized.empty:
        entry_cost = sized.apply(
            lambda row: max(number(row.get("entry_credit"), 0.0), number(row.get("entry_debit"), 0.0))
            * 100.0
            * int(row["contracts"]),
            axis=1,
        )
        for label, stress in (("base", 0.0), ("worse_fill_5pct", 0.05), ("worse_fill_10pct", 0.10)):
            col = f"pnl_risk_sized_{label}"
            sized[col] = sized["pnl_risk_sized"] - entry_cost * stress
            scenarios[f"risk_sized_{label}"] = metrics(sized, col)
            scenarios[f"risk_sized_{label}"]["monthly"] = monthly(sized, col, monthly_target)
    else:
        scenarios["risk_sized_base"] = metrics(sized, "pnl_risk_sized")
        scenarios["risk_sized_worse_fill_5pct"] = metrics(sized, "pnl_risk_sized")
        scenarios["risk_sized_worse_fill_10pct"] = metrics(sized, "pnl_risk_sized")

    risk_months = scenarios["risk_sized_base"].get("monthly", [])
    positive_months = [month["net_pnl"] for month in risk_months if month["net_pnl"] > 0]
    median_positive = float(pd.Series(positive_months).median()) if positive_months else 0.0
    payload: dict[str, object] = {
        "source": source,
        "policy": {
            "monthly_target": monthly_target,
            "account_value": account_value,
            "aggregate_risk_budget": aggregate_risk_budget,
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_contracts": max_contracts,
            "max_ticker_share": max_ticker_share,
            "max_sector_share": max_sector_share,
            "holding_period": "entry date through actual replay exit date",
        },
        "accepted_trades": int(len(fixed)),
        "risk_sized_trades": int(len(sized)),
        "scenarios": scenarios,
        "feasibility": {
            "status": "evaluated" if account_value > 0 else "account_value_required",
            "reason": "" if account_value > 0 else "set --validation-account-value to evaluate overlapping risk-sized capacity",
            "months_meeting_target": sum(bool(month["target_met"]) for month in risk_months),
            "months_observed": len(risk_months),
            "median_positive_month_pnl": median_positive,
            "target_supported": bool(risk_months) and all(bool(month["target_met"]) for month in risk_months),
        },
    }
    return payload, fixed, sized


def write_portfolio_capacity_outputs(
    *,
    replay_detail: Path,
    out_dir: Path,
    monthly_target: float,
    account_value: float,
    aggregate_risk_budget: float = 0.0,
    risk_per_trade_pct: float = 0.02,
    max_contracts: int = 20,
    max_ticker_share: float = 0.20,
    max_sector_share: float = 0.40,
    asof: object | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = actionable_rows(replay_detail)
    if asof is not None:
        cutoff = pd.to_datetime(asof, errors="coerce")
        if not pd.isna(cutoff):
            base = base[base["entry_date"] <= cutoff].copy()
    payload, fixed, sized = build_portfolio_capacity_payload(
        base,
        source=str(replay_detail),
        monthly_target=monthly_target,
        account_value=account_value,
        aggregate_risk_budget=aggregate_risk_budget,
        risk_per_trade_pct=risk_per_trade_pct,
        max_contracts=max_contracts,
        max_ticker_share=max_ticker_share,
        max_sector_share=max_sector_share,
    )
    summary_path = out_dir / "portfolio_capacity_summary.json"
    fixed_path = out_dir / "accepted_trades_fixed_size.csv"
    sized_path = out_dir / "accepted_trades_risk_sized.csv"
    report_path = out_dir / "PORTFOLIO_CAPACITY_VALIDATION.md"
    payload["artifacts"] = {
        "summary": str(summary_path),
        "fixed_size": str(fixed_path),
        "risk_sized": str(sized_path),
        "report": str(report_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    fixed.to_csv(fixed_path, index=False)
    sized.to_csv(sized_path, index=False)

    scenarios = payload["scenarios"]
    feasibility = payload["feasibility"]
    aggregate_text = "not configured" if aggregate_risk_budget <= 0 else f"${aggregate_risk_budget:,.0f}"
    lines = [
        "# Codex Daily V4 Portfolio-Capacity Validation",
        "",
        "This read-only layer uses actual overlapping entry/exit dates; it does not change candidate approval.",
        "",
        "## Capacity verdict",
        "",
        f"- Accepted replay trades: {len(fixed)}",
        f"- Risk-sized trades admitted: {len(sized)}",
        f"- Monthly target: ${monthly_target:,.0f}",
        f"- Account value: {'not configured' if account_value <= 0 else '$' + format(account_value, ',.0f')}",
        f"- Per-ticket max-loss cap: {risk_per_trade_pct:.1%} of account",
        f"- Aggregate risk budget: {aggregate_text}",
        f"- Months meeting target: {feasibility['months_meeting_target']}/{feasibility['months_observed']}",
        f"- Target supported by replay: {'YES' if feasibility['target_supported'] else 'NO'}",
        f"- Capacity status: {feasibility['status']}",
        "",
        "## Scenario summary",
        "",
        "| Scenario | Trades | Win | PF | Net P/L | Max DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in scenarios.items():
        win = block.get("win_rate")
        win_text = f"{float(win):.1%}" if win is not None else "n/a"
        lines.append(
            f"| {name} | {block.get('trades', 0)} | {win_text} | {fmt_pf(block.get('profit_factor'))} | "
            f"${float(block.get('net_pnl', 0)):,.0f} | ${float(block.get('max_drawdown', 0)):,.0f} |"
        )
    lines += [
        "",
        "## Risk-sized monthly results",
        "",
        "| Month | Trades | Win | PF | Net P/L | Max DD | Target met |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for block in scenarios["risk_sized_base"].get("monthly", []):
        win = block.get("win_rate")
        lines.append(
            f"| {block['month']} | {block['trades']} | {float(win):.1%} | {fmt_pf(block.get('profit_factor'))} | "
            f"${float(block['net_pnl']):,.0f} | ${float(block['max_drawdown']):,.0f} | "
            f"{'YES' if block['target_met'] else 'NO'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Fixed-size scenarios are mechanical scaling only and do not claim that liquidity supports the size.",
        "- Risk-sized results enforce overlapping capital, per-trade risk, ticker concentration, and sector concentration.",
        "- A monthly target is supported only when every observed risk-sized month reaches it.",
        "",
    ]
    report_path.write_text("\\n".join(lines), encoding="utf-8")
    return payload


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

    base = actionable_rows(args.replay_detail)
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
