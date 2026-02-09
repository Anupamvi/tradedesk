#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

DEFAULT_SHEET_CACHE_CSV = r"c:\uw_root\out\playbook\source_sheet_latest.csv"
DEFAULT_SHEET_REALIZED_CSV = r"c:\uw_root\out\playbook\cleaned_realized_trades_from_sheet.csv"


def norm_col(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def parse_float(x) -> float:
    if x is None:
        return math.nan
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return math.nan
    s = str(x).strip()
    if not s:
        return math.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "")
    if s.startswith("-"):
        neg = True
    try:
        v = float(s)
        return -abs(v) if neg else v
    except Exception:
        return math.nan


def find_col(columns: List[str], aliases: List[str]) -> Optional[str]:
    mapped = {c: norm_col(c) for c in columns}
    for alias in aliases:
        a = norm_col(alias)
        for raw, n in mapped.items():
            if n == a:
                return raw
    return None


def compute_profit_factor(pnl: pd.Series) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return math.inf if gp > 0 else math.nan
    return gp / gl


def compute_max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    c = pnl.cumsum()
    dd = c - c.cummax()
    return float(dd.min())


def longest_streak(flags: List[bool], target: bool) -> int:
    best = 0
    cur = 0
    for f in flags:
        if bool(f) == target:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def trailing_loss_streak(pnl: pd.Series) -> int:
    k = 0
    for v in reversed(pnl.tolist()):
        if v < 0:
            k += 1
        else:
            break
    return k


def load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def load_realized_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    date_col = find_col(cols, ["date", "trade_date", "close_date", "closed_at"])
    pnl_col = find_col(cols, ["realized_pnl", "pnl", "profit_loss", "net_pnl", "amount"])
    strategy_col = find_col(cols, ["strategy", "strategy_type", "setup"])
    symbol_col = find_col(cols, ["symbol", "ticker", "underlying", "underlying_symbol"])

    if not date_col or not pnl_col:
        raise ValueError(f"Missing required columns in {path}. Need date + realized pnl.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["realized_pnl"] = df[pnl_col].map(parse_float)
    out["strategy"] = df[strategy_col].astype(str).str.strip() if strategy_col else "Unknown"
    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper() if symbol_col else "UNKNOWN"
    out["symbol"] = out["symbol"].replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    out["strategy"] = out["strategy"].replace({"": "Unknown", "nan": "Unknown"})

    out = out.dropna(subset=["date", "realized_pnl"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["weekday"] = out["date"].dt.day_name()
    out["win"] = out["realized_pnl"] > 0
    return out


def build_realized_from_sheet_csv_url(
    sheet_csv_url: str,
    raw_cache_csv: Path,
    realized_out_csv: Path,
) -> Dict[str, float]:
    """
    Pull Google Sheet CSV export URL, parse manual options log rows into standardized realized trades,
    and persist both raw cache + standardized realized CSV.
    """
    raw_cache_csv.parent.mkdir(parents=True, exist_ok=True)
    realized_out_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(sheet_csv_url, low_memory=False)
    raw_df.to_csv(raw_cache_csv, index=False)

    try:
        from analyze_trading_year import standardize_manual_options_log_df  # local module
    except Exception as e:
        raise RuntimeError(f"Failed to import manual log parser from analyze_trading_year.py: {e}") from e

    std_df, meta = standardize_manual_options_log_df(raw_df, source_name=sheet_csv_url)
    if std_df.empty:
        raise RuntimeError("Sheet CSV parsed but no usable realized rows were extracted.")
    std_df.to_csv(realized_out_csv, index=False)
    return meta or {}


def infer_strategy_from_text(strategy: str, description: str, side: str) -> str:
    s = f"{strategy} {description} {side}".upper()
    has_put = "PUT" in s or bool(re.search(r"\bP\b", s))
    has_call = "CALL" in s or bool(re.search(r"\bC\b", s))
    is_short = "SHORT" in s or "SELL" in s or "WRITE" in s
    is_long = "LONG" in s or "BUY" in s
    if is_short and has_put:
        return "Short Put Option"
    if is_short and has_call:
        return "Short Call Option"
    if is_long and has_put:
        return "Long Put Option"
    if is_long and has_call:
        return "Long Call Option"
    if is_short:
        return "Short"
    if is_long:
        return "Long"
    return "Unknown"


def load_open_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    symbol_col = find_col(cols, ["symbol", "ticker", "underlying"])
    strategy_col = find_col(cols, ["strategy", "strategy_type", "position_type"])
    side_col = find_col(cols, ["side", "action", "direction"])
    desc_col = find_col(cols, ["description", "instrument", "name"])
    expiry_col = find_col(cols, ["expiry", "expiration", "exp_date", "expiration_date"])
    risk_col = find_col(
        cols,
        [
            "max_loss",
            "risk",
            "risk_amount",
            "max_risk",
            "buying_power_effect",
            "bp_effect",
            "margin_requirement",
        ],
    )

    out = pd.DataFrame(index=df.index)
    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper() if symbol_col else "UNKNOWN"
    out["strategy_raw"] = df[strategy_col].astype(str).str.strip() if strategy_col else ""
    out["description"] = df[desc_col].astype(str).str.strip() if desc_col else ""
    out["side_raw"] = df[side_col].astype(str).str.strip() if side_col else ""
    out["expiry"] = pd.to_datetime(df[expiry_col], errors="coerce") if expiry_col else pd.NaT
    out["risk"] = df[risk_col].map(parse_float).abs() if risk_col else math.nan
    out["strategy"] = [
        infer_strategy_from_text(a, b, c) for a, b, c in zip(out["strategy_raw"], out["description"], out["side_raw"])
    ]
    out["symbol"] = out["symbol"].replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    return out


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_md_table(df: pd.DataFrame, n: int = 30) -> str:
    if df.empty:
        return "_none_"
    return df.head(n).to_markdown(index=False)


def run_daily(
    trades: pd.DataFrame,
    out_dir: Path,
    cfg: Dict,
    open_positions_csv: Optional[Path],
    lookback_trades: int,
) -> Dict:
    pb = cfg.get("playbook", {})
    daily_cfg = pb.get("daily", {})
    risk_cfg = pb.get("risk_limits", {})

    daily_loss_stop_cash = float(daily_cfg.get("daily_loss_stop_cash", 1000.0))
    streak_yellow = int(daily_cfg.get("loss_streak_yellow", 3))
    streak_red = int(daily_cfg.get("loss_streak_red", 4))
    min_pf_yellow = float(daily_cfg.get("rolling_pf_yellow_floor", 1.0))
    short_put_limit = float(risk_cfg.get("short_put_max_share", 0.35))
    symbol_limit = float(risk_cfg.get("single_symbol_max_share", 0.10))
    expiry_limit = float(risk_cfg.get("single_expiry_max_share_short_put", 0.25))

    x = trades.sort_values("date").reset_index(drop=True)
    recent = x.tail(max(1, int(lookback_trades))).copy()
    last_day = x["date"].dt.date.max()
    day_df = x[x["date"].dt.date == last_day].copy()

    day_pnl = float(day_df["realized_pnl"].sum()) if not day_df.empty else 0.0
    recent_pf = compute_profit_factor(recent["realized_pnl"])
    recent_net = float(recent["realized_pnl"].sum())
    cur_loss_streak = trailing_loss_streak(x["realized_pnl"])

    status = "GREEN"
    severity = 0
    triggers: List[str] = []
    actions: List[str] = []

    def escalate(level: str, trigger: str, action: str) -> None:
        nonlocal status, severity
        rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}[level]
        if rank > severity:
            severity = rank
            status = level
        triggers.append(trigger)
        actions.append(action)

    if day_pnl <= -abs(daily_loss_stop_cash):
        escalate(
            "RED",
            f"Daily realized P/L {day_pnl:,.2f} <= stop {-abs(daily_loss_stop_cash):,.2f}",
            "Stop opening new risk today; only reduce or hedge open risk.",
        )
    if cur_loss_streak >= streak_red:
        escalate(
            "RED",
            f"Trailing loss streak is {cur_loss_streak} (red threshold {streak_red})",
            "Cut size by 50% and skip the next 2 non-core setups.",
        )
    elif cur_loss_streak >= streak_yellow:
        escalate(
            "YELLOW",
            f"Trailing loss streak is {cur_loss_streak} (yellow threshold {streak_yellow})",
            "Reduce new position size by 25% until streak resets.",
        )
    if np.isfinite(recent_pf) and recent_pf < min_pf_yellow:
        escalate(
            "YELLOW",
            f"Recent {len(recent)}-trade PF is {recent_pf:.2f} < {min_pf_yellow:.2f}",
            "Restrict entries to top-conviction setups only this week.",
        )

    pos_summary: Dict[str, float] = {}
    symbol_risk_table = pd.DataFrame()
    expiry_risk_table = pd.DataFrame()
    if open_positions_csv and open_positions_csv.exists():
        pos = load_open_positions(open_positions_csv)
        pos = pos[pos["risk"].notna() & (pos["risk"] > 0)].copy()
        if not pos.empty:
            total_risk = float(pos["risk"].sum())
            short_put_risk = float(pos.loc[pos["strategy"] == "Short Put Option", "risk"].sum())
            short_put_share = short_put_risk / total_risk if total_risk > 0 else 0.0
            pos_summary = {
                "total_open_risk": total_risk,
                "short_put_risk": short_put_risk,
                "short_put_risk_share": short_put_share,
            }

            symbol_risk_table = (
                pos.groupby("symbol", dropna=False)["risk"]
                .sum()
                .reset_index(name="risk")
                .sort_values("risk", ascending=False)
                .reset_index(drop=True)
            )
            symbol_risk_table["risk_share"] = symbol_risk_table["risk"] / max(1e-9, total_risk)

            if short_put_share > short_put_limit:
                escalate(
                    "RED",
                    f"Short-put risk share {short_put_share:.1%} > limit {short_put_limit:.1%}",
                    "Do not add short puts; trim/roll largest short-put exposure first.",
                )

            sym_breach = symbol_risk_table[symbol_risk_table["risk_share"] > symbol_limit]
            if not sym_breach.empty:
                names = ", ".join(sym_breach["symbol"].head(5).tolist())
                escalate(
                    "YELLOW",
                    f"Single-symbol risk concentration above {symbol_limit:.1%}: {names}",
                    "Reduce concentrated ticker risk before adding same-ticker trades.",
                )

            short_put_pos = pos[pos["strategy"] == "Short Put Option"].copy()
            short_put_pos = short_put_pos[short_put_pos["expiry"].notna()].copy()
            if not short_put_pos.empty and short_put_risk > 0:
                expiry_risk_table = (
                    short_put_pos.groupby(short_put_pos["expiry"].dt.date)["risk"]
                    .sum()
                    .reset_index(name="risk")
                    .sort_values("risk", ascending=False)
                    .reset_index(drop=True)
                )
                expiry_risk_table["risk_share"] = expiry_risk_table["risk"] / max(1e-9, short_put_risk)
                exp_breach = expiry_risk_table[expiry_risk_table["risk_share"] > expiry_limit]
                if not exp_breach.empty:
                    exp = str(exp_breach.iloc[0, 0])
                    share = float(exp_breach.iloc[0]["risk_share"])
                    escalate(
                        "YELLOW",
                        f"Short-put expiry concentration {share:.1%} on {exp} > {expiry_limit:.1%}",
                        "Spread short-put expiries; avoid stacking same expiration risk.",
                    )

    payload = {
        "mode": "daily",
        "status": status,
        "as_of_trade_date": str(last_day),
        "day_realized_pnl": day_pnl,
        "recent_trade_count": int(len(recent)),
        "recent_net_pnl": recent_net,
        "recent_profit_factor": recent_pf,
        "trailing_loss_streak": int(cur_loss_streak),
        "thresholds": {
            "daily_loss_stop_cash": daily_loss_stop_cash,
            "loss_streak_yellow": streak_yellow,
            "loss_streak_red": streak_red,
            "rolling_pf_yellow_floor": min_pf_yellow,
            "short_put_max_share": short_put_limit,
            "single_symbol_max_share": symbol_limit,
            "single_expiry_max_share_short_put": expiry_limit,
        },
        "triggers": triggers,
        "actions": actions,
        "open_position_risk_summary": pos_summary,
    }
    write_json(out_dir / "daily_risk_monitor.json", payload)

    lines = [
        "# Daily Risk Monitor",
        f"Status: **{status}**",
        "",
        "## Snapshot",
        f"- As-of trade date: {last_day}",
        f"- Day realized P/L: {day_pnl:,.2f}",
        f"- Recent trades checked: {len(recent)}",
        f"- Recent net P/L: {recent_net:,.2f}",
        f"- Recent PF: {recent_pf:.2f}" if np.isfinite(recent_pf) else "- Recent PF: n/a",
        f"- Trailing loss streak: {cur_loss_streak}",
        "",
        "## Triggers",
    ]
    if triggers:
        lines.extend([f"- {t}" for t in triggers])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Actions")
    if actions:
        lines.extend([f"- {a}" for a in actions])
    else:
        lines.append("- Continue normal sizing under current limits.")
    lines.append("")
    if pos_summary:
        lines.append("## Open Risk")
        lines.append(
            f"- Total open risk: {pos_summary.get('total_open_risk', 0.0):,.2f} | "
            f"Short put share: {pos_summary.get('short_put_risk_share', 0.0):.1%}"
        )
        lines.append("")
        lines.append("### Symbol Concentration")
        lines.append(to_md_table(symbol_risk_table[["symbol", "risk", "risk_share"]], n=20))
        lines.append("")
        lines.append("### Short Put Expiry Concentration")
        if not expiry_risk_table.empty:
            lines.append(to_md_table(expiry_risk_table, n=20))
        else:
            lines.append("_none_")
        lines.append("")
    write_text(out_dir / "daily_risk_monitor.md", "\n".join(lines))
    return payload


def aggregate_edge(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            trades=("realized_pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("realized_pnl", "sum"),
            avg_pnl=("realized_pnl", "mean"),
            profit_factor=("realized_pnl", compute_profit_factor),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
        .reset_index(drop=True)
    )
    return g


def run_weekly(trades: pd.DataFrame, out_dir: Path, cfg: Dict, as_of: Optional[pd.Timestamp]) -> Dict:
    pb = cfg.get("playbook", {})
    wk = pb.get("weekly", {})
    strategy_window = int(wk.get("strategy_window_trades", 20))
    strategy_min_trades = int(wk.get("strategy_min_trades", 20))
    strategy_min_pf = float(wk.get("strategy_min_pf", 1.0))
    symbol_window = int(wk.get("symbol_window_trades", 20))
    symbol_min_trades = int(wk.get("symbol_min_trades", 5))
    symbol_pause_net = float(wk.get("symbol_pause_if_net_below", 0.0))
    combo_min_trades = int(wk.get("combo_min_trades", 3))
    combo_pause_net = float(wk.get("combo_pause_if_net_below", -1000.0))

    x = trades.sort_values("date").reset_index(drop=True)
    if as_of is not None:
        x = x[x["date"] <= as_of].copy()

    strategy_rows = []
    for strategy, g in x.groupby("strategy", dropna=False):
        t = g.tail(max(1, strategy_window)).copy()
        trades_n = int(len(t))
        pf = compute_profit_factor(t["realized_pnl"])
        net = float(t["realized_pnl"].sum())
        wr = float((t["realized_pnl"] > 0).mean())
        action = "KEEP"
        reason = ""
        if trades_n >= strategy_min_trades and np.isfinite(pf) and pf < strategy_min_pf:
            action = "PAUSE"
            reason = f"PF {pf:.2f} < {strategy_min_pf:.2f}"
        strategy_rows.append(
            {
                "strategy": strategy,
                "window_trades": trades_n,
                "window_win_rate": wr,
                "window_net_pnl": net,
                "window_profit_factor": pf,
                "action": action,
                "reason": reason,
            }
        )
    strategy_actions = pd.DataFrame(strategy_rows).sort_values(["action", "window_net_pnl"], ascending=[True, False])

    symbol_rows = []
    for symbol, g in x.groupby("symbol", dropna=False):
        t = g.tail(max(1, symbol_window)).copy()
        trades_n = int(len(t))
        net = float(t["realized_pnl"].sum())
        pf = compute_profit_factor(t["realized_pnl"])
        wr = float((t["realized_pnl"] > 0).mean())
        action = "KEEP"
        reason = ""
        if trades_n >= symbol_min_trades and net <= symbol_pause_net:
            action = "PAUSE"
            reason = f"Net {net:,.2f} <= {symbol_pause_net:,.2f}"
        symbol_rows.append(
            {
                "symbol": symbol,
                "window_trades": trades_n,
                "window_win_rate": wr,
                "window_net_pnl": net,
                "window_profit_factor": pf,
                "action": action,
                "reason": reason,
            }
        )
    symbol_actions = pd.DataFrame(symbol_rows).sort_values(["action", "window_net_pnl"], ascending=[True, False])

    combo_stats = aggregate_edge(x, ["strategy", "symbol"])
    combo_actions = combo_stats.copy()
    if not combo_actions.empty:
        combo_actions["action"] = "KEEP"
        combo_actions["reason"] = ""
        bad = (combo_actions["trades"] >= combo_min_trades) & (combo_actions["net_pnl"] <= combo_pause_net)
        combo_actions.loc[bad, "action"] = "PAUSE"
        combo_actions.loc[bad, "reason"] = combo_actions.loc[bad, "net_pnl"].map(
            lambda v: f"net_pnl {v:,.2f} <= {combo_pause_net:,.2f}"
        )
        combo_actions = combo_actions.sort_values(["action", "net_pnl"], ascending=[True, False]).reset_index(drop=True)

    status = "GREEN"
    if (strategy_actions["action"] == "PAUSE").any() or (symbol_actions["action"] == "PAUSE").any():
        status = "YELLOW"
    if not combo_actions.empty and (combo_actions["action"] == "PAUSE").any():
        status = "RED"

    payload = {
        "mode": "weekly",
        "status": status,
        "as_of_date": str(x["date"].max().date()) if not x.empty else "",
        "strategy_pauses": int((strategy_actions["action"] == "PAUSE").sum()),
        "symbol_pauses": int((symbol_actions["action"] == "PAUSE").sum()),
        "combo_pauses": int((combo_actions["action"] == "PAUSE").sum()) if not combo_actions.empty else 0,
        "thresholds": {
            "strategy_window_trades": strategy_window,
            "strategy_min_trades": strategy_min_trades,
            "strategy_min_pf": strategy_min_pf,
            "symbol_window_trades": symbol_window,
            "symbol_min_trades": symbol_min_trades,
            "symbol_pause_if_net_below": symbol_pause_net,
            "combo_min_trades": combo_min_trades,
            "combo_pause_if_net_below": combo_pause_net,
        },
    }
    write_json(out_dir / "weekly_edge_report.json", payload)
    strategy_actions.to_csv(out_dir / "weekly_strategy_actions.csv", index=False)
    symbol_actions.to_csv(out_dir / "weekly_symbol_actions.csv", index=False)
    if not combo_actions.empty:
        combo_actions.to_csv(out_dir / "weekly_combo_actions.csv", index=False)

    lines = [
        "# Weekly Edge Report",
        f"Status: **{status}**",
        "",
        "## Pause Summary",
        f"- Strategy pauses: {payload['strategy_pauses']}",
        f"- Symbol pauses: {payload['symbol_pauses']}",
        f"- Strategy-symbol combo pauses: {payload['combo_pauses']}",
        "",
        "## Strategy Actions",
        to_md_table(strategy_actions, n=50),
        "",
        "## Symbol Actions",
        to_md_table(symbol_actions, n=50),
        "",
    ]
    if not combo_actions.empty:
        lines.extend(
            [
                "## Strategy-Symbol Actions",
                to_md_table(combo_actions, n=100),
                "",
            ]
        )
    write_text(out_dir / "weekly_edge_report.md", "\n".join(lines))
    return payload


def run_monthly(trades: pd.DataFrame, out_dir: Path, cfg: Dict) -> Dict:
    pb = cfg.get("playbook", {})
    mo = pb.get("monthly", {})
    target_monthly = float(mo.get("target_monthly_profit", 20000.0))
    long_call_kill = float(mo.get("long_call_monthly_kill_switch", -1500.0))

    x = trades.sort_values("date").reset_index(drop=True)
    monthly = (
        x.groupby("month", dropna=False)
        .agg(
            trades=("realized_pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("realized_pnl", "sum"),
            avg_pnl=("realized_pnl", "mean"),
            profit_factor=("realized_pnl", compute_profit_factor),
        )
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )
    monthly["rolling_3m_net"] = monthly["net_pnl"].rolling(3, min_periods=1).mean()
    monthly["rolling_3m_pf"] = monthly["profit_factor"].rolling(3, min_periods=1).mean()

    avg_monthly = float(monthly["net_pnl"].mean()) if not monthly.empty else math.nan
    med_monthly = float(monthly["net_pnl"].median()) if not monthly.empty else math.nan
    last_3m_avg = float(monthly["net_pnl"].tail(3).mean()) if not monthly.empty else math.nan
    monthly_gap = target_monthly - avg_monthly if np.isfinite(avg_monthly) else math.nan
    multiplier = target_monthly / avg_monthly if np.isfinite(avg_monthly) and avg_monthly > 0 else math.nan

    by_strategy = aggregate_edge(x, ["strategy"])
    by_symbol = aggregate_edge(x, ["symbol"])
    by_strategy_month = (
        x.groupby(["month", "strategy"], dropna=False)["realized_pnl"]
        .sum()
        .reset_index(name="net_pnl")
        .sort_values(["month", "net_pnl"], ascending=[True, False])
        .reset_index(drop=True)
    )

    long_call_month = (
        x[x["strategy"] == "Long Call Option"]
        .groupby("month", dropna=False)["realized_pnl"]
        .sum()
        .reset_index(name="long_call_net")
        .sort_values("month")
    )
    long_call_breach = long_call_month[long_call_month["long_call_net"] <= long_call_kill].copy()

    top_losses = x.sort_values("realized_pnl").head(20).copy()

    payload = {
        "mode": "monthly",
        "as_of_date": str(x["date"].max().date()) if not x.empty else "",
        "months_analyzed": int(len(monthly)),
        "avg_monthly_net_pnl": avg_monthly,
        "median_monthly_net_pnl": med_monthly,
        "last_3m_avg_net_pnl": last_3m_avg,
        "target_monthly_profit": target_monthly,
        "gap_to_target": monthly_gap,
        "required_multiplier_to_target": multiplier,
        "long_call_kill_switch_threshold": long_call_kill,
        "long_call_breach_months": long_call_breach["month"].tolist(),
    }
    write_json(out_dir / "monthly_longitudinal_review.json", payload)
    monthly.to_csv(out_dir / "monthly_pnl_trend.csv", index=False)
    by_strategy.to_csv(out_dir / "monthly_strategy_edge.csv", index=False)
    by_symbol.to_csv(out_dir / "monthly_symbol_edge.csv", index=False)
    by_strategy_month.to_csv(out_dir / "monthly_strategy_month_matrix.csv", index=False)
    long_call_month.to_csv(out_dir / "monthly_long_call_net.csv", index=False)
    top_losses.to_csv(out_dir / "monthly_top_losses.csv", index=False)

    lines = [
        "# Monthly Longitudinal Review",
        "",
        "## Performance Baseline",
        f"- Months analyzed: {payload['months_analyzed']}",
        f"- Average monthly net P/L: {avg_monthly:,.2f}" if np.isfinite(avg_monthly) else "- Average monthly net P/L: n/a",
        f"- Median monthly net P/L: {med_monthly:,.2f}" if np.isfinite(med_monthly) else "- Median monthly net P/L: n/a",
        f"- Last 3-month avg net P/L: {last_3m_avg:,.2f}" if np.isfinite(last_3m_avg) else "- Last 3-month avg net P/L: n/a",
        f"- Target monthly net P/L: {target_monthly:,.2f}",
        f"- Gap to target: {monthly_gap:,.2f}" if np.isfinite(monthly_gap) else "- Gap to target: n/a",
        f"- Required multiplier to target: {multiplier:.2f}x" if np.isfinite(multiplier) else "- Required multiplier to target: n/a",
        "",
        "## Long Call Kill-Switch Check",
        f"- Threshold: monthly long-call net <= {long_call_kill:,.2f}",
    ]
    if not long_call_breach.empty:
        lines.append("- Breach months: " + ", ".join(long_call_breach["month"].tolist()))
    else:
        lines.append("- Breach months: none")
    lines.extend(
        [
            "",
            "## Monthly Net Trend",
            to_md_table(monthly, n=36),
            "",
            "## Strategy Edge",
            to_md_table(by_strategy, n=30),
            "",
            "## Symbol Edge",
            to_md_table(by_symbol, n=30),
            "",
            "## Largest Losses",
            to_md_table(top_losses[["date", "symbol", "strategy", "realized_pnl"]], n=20),
            "",
        ]
    )
    write_text(out_dir / "monthly_longitudinal_review.md", "\n".join(lines))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Trade playbook pipeline: daily risk, weekly edge, monthly review.")
    ap.add_argument("mode", choices=["daily", "weekly", "monthly", "all"])
    ap.add_argument(
        "--realized-csv",
        default=r"c:\uw_root\out\trade_performance_review_manual_options_full\cleaned_realized_trades.csv",
        help="CSV with realized trades (date, strategy, symbol, realized_pnl).",
    )
    ap.add_argument(
        "--sheet-csv-url",
        default="",
        help="Optional Google Sheet CSV export URL. If provided, pipeline pulls this URL and auto-builds realized CSV.",
    )
    ap.add_argument(
        "--sheet-cache-csv",
        default=DEFAULT_SHEET_CACHE_CSV,
        help="Where to cache pulled raw sheet CSV when --sheet-csv-url is used.",
    )
    ap.add_argument(
        "--sheet-realized-csv",
        default=DEFAULT_SHEET_REALIZED_CSV,
        help="Where to write standardized realized CSV when --sheet-csv-url is used.",
    )
    ap.add_argument("--open-positions-csv", default="", help="Optional open positions CSV for daily risk concentration checks.")
    ap.add_argument("--config", default=r"c:\uw_root\rulebook_config.yaml", help="YAML config path.")
    ap.add_argument("--out-dir", default=r"c:\uw_root\out\playbook", help="Output directory.")
    ap.add_argument("--lookback-trades", type=int, default=30, help="Recent-trade lookback for daily checks.")
    ap.add_argument("--start-date", default="", help="Optional filter start date YYYY-MM-DD.")
    ap.add_argument("--end-date", default="", help="Optional filter end date YYYY-MM-DD.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_meta: Dict[str, float] = {}
    if args.sheet_csv_url:
        raw_cache_csv = (
            out_dir / "source_sheet_latest.csv"
            if args.sheet_cache_csv == DEFAULT_SHEET_CACHE_CSV
            else Path(args.sheet_cache_csv).resolve()
        )
        realized_csv = (
            out_dir / "cleaned_realized_trades_from_sheet.csv"
            if args.sheet_realized_csv == DEFAULT_SHEET_REALIZED_CSV
            else Path(args.sheet_realized_csv).resolve()
        )
        sheet_meta = build_realized_from_sheet_csv_url(args.sheet_csv_url, raw_cache_csv, realized_csv)
    else:
        realized_csv = Path(args.realized_csv).resolve()
        if not realized_csv.exists():
            raise FileNotFoundError(f"Missing realized CSV: {realized_csv}")

    cfg = load_config(Path(args.config).resolve())
    trades = load_realized_trades(realized_csv)
    if args.start_date:
        sd = dt.date.fromisoformat(args.start_date)
        trades = trades[trades["date"].dt.date >= sd].copy()
    if args.end_date:
        ed = dt.date.fromisoformat(args.end_date)
        trades = trades[trades["date"].dt.date <= ed].copy()
    if trades.empty:
        raise RuntimeError("No trades after filters.")

    daily = None
    weekly = None
    monthly = None
    open_positions_csv = Path(args.open_positions_csv).resolve() if args.open_positions_csv else None

    if args.mode in {"daily", "all"}:
        daily = run_daily(trades, out_dir, cfg, open_positions_csv, args.lookback_trades)
    if args.mode in {"weekly", "all"}:
        weekly = run_weekly(trades, out_dir, cfg, as_of=None)
    if args.mode in {"monthly", "all"}:
        monthly = run_monthly(trades, out_dir, cfg)

    if args.mode == "all":
        summary = {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "source_realized_csv": str(realized_csv),
            "source_sheet_csv_url": args.sheet_csv_url or "",
            "daily_status": daily.get("status") if daily else "",
            "weekly_status": weekly.get("status") if weekly else "",
            "monthly_target_gap": monthly.get("gap_to_target") if monthly else math.nan,
        }
        if sheet_meta:
            summary["sheet_parser_meta"] = sheet_meta
        write_json(out_dir / "playbook_run_summary.json", summary)
        lines = [
            "# Trade Playbook Run",
            "",
            f"- Source realized CSV: `{realized_csv}`",
            (f"- Source sheet URL: `{args.sheet_csv_url}`" if args.sheet_csv_url else "- Source sheet URL: n/a"),
            f"- Daily status: **{summary['daily_status']}**",
            f"- Weekly status: **{summary['weekly_status']}**",
            (
                f"- Gap to target monthly P/L: {summary['monthly_target_gap']:,.2f}"
                if np.isfinite(summary["monthly_target_gap"])
                else "- Gap to target monthly P/L: n/a"
            ),
            "",
            "## Files",
            "- `daily_risk_monitor.md`",
            "- `weekly_edge_report.md`",
            "- `monthly_longitudinal_review.md`",
            "- `playbook_run_summary.json`",
        ]
        write_text(out_dir / "playbook_run_summary.md", "\n".join(lines))

    print(f"Mode: {args.mode}")
    print(f"Source: {realized_csv}")
    if args.sheet_csv_url:
        print(f"Pulled sheet URL: {args.sheet_csv_url}")
    print(f"Rows loaded: {len(trades)}")
    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
