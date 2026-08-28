"""Exact historical ORATS debit-spread replay.

Signals are formed after session T using only data through T. The option entry
uses the T+1 historical chain, and the exact same expiration/strikes are
liquidated from the historical exit chain. No current-IV reconstruction and no
perfect-midpoint assumption are permitted.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from corat.config import PROJECT_ROOT
from corat.clock import today_new_york
from corat.history import _match_setup
from corat.models import Bar, SetupSignal
from corat.options import choose_debit_spread
from corat.orats import OratsClient
from corat.scoring import build_stock_plan
from corat.store import canonical_json, sha256_bytes, sha256_file, utc_now, write_json, write_text
from corat.technical import bars_from_dailies, technical_snapshot


def _resolve(config: Mapping[str, Any], key: str) -> Path:
    path = Path(str(config[key]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_for_leg(
    rows: Iterable[Mapping[str, Any]],
    expiration: str,
    strike: float,
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("expirDate") or "")[:10] != expiration:
            continue
        candidate = _number(row.get("strike"))
        if candidate is not None and abs(candidate - strike) < 1e-6:
            return row
    return None


def conservative_exit_credit(
    rows: Sequence[Mapping[str, Any]],
    direction: str,
    expiration: str,
    long_strike: float,
    short_strike: float,
) -> Optional[float]:
    long_row = _row_for_leg(rows, expiration, long_strike)
    short_row = _row_for_leg(rows, expiration, short_strike)
    if long_row is None or short_row is None:
        return None
    call = direction == "BULLISH"
    prefix = "call" if call else "put"
    long_bid = _number(long_row.get(prefix + "BidPrice"))
    long_ask = _number(long_row.get(prefix + "AskPrice"))
    short_bid = _number(short_row.get(prefix + "BidPrice"))
    short_ask = _number(short_row.get(prefix + "AskPrice"))
    if None in (long_bid, long_ask, short_bid, short_ask):
        return None
    assert long_bid is not None and long_ask is not None and short_bid is not None and short_ask is not None
    if long_ask < long_bid or short_ask < short_bid:
        return None
    natural = long_bid - short_ask
    midpoint = (long_bid + long_ask) / 2.0 - (short_bid + short_ask) / 2.0
    return natural + 0.25 * (midpoint - natural)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(row["pnl_dollars"]) for row in rows]
    returns = [float(row["return_on_max_loss"]) for row in rows]
    winners = [value for value in values if value > 0]
    losers = [value for value in values if value < 0]
    gains = sum(winners)
    losses = abs(sum(losers))
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return {
        "n": len(rows),
        "win_rate": sum(1 for value in values if value > 0) / float(len(values)) if values else None,
        "expectancy_dollars": statistics.mean(values) if values else None,
        "median_pnl_dollars": statistics.median(values) if values else None,
        "average_return_on_max_loss": statistics.mean(returns) if returns else None,
        "average_winner_dollars": statistics.mean(winners) if winners else None,
        "average_loser_dollars": statistics.mean(losers) if losers else None,
        "profit_factor": gains / losses if losses > 0 else (float("inf") if gains > 0 else None),
        "total_pnl_dollars": sum(values),
        "max_drawdown_dollars": drawdown,
    }


def _signal_indices(
    setup_name: str,
    ticker_bars: Sequence[Bar],
    spy_bars: Sequence[Bar],
    start_date: str,
    end_date: str,
    holding_sessions: int,
    spacing_sessions: int,
) -> List[int]:
    spy_by_date = {bar.date: bar for bar in spy_bars}
    dates = sorted(spy_by_date)
    signals = []
    last = -1000
    for index in range(60, len(ticker_bars) - holding_sessions - 2):
        if index - last < spacing_sessions:
            continue
        signal_date = ticker_bars[index].date
        if signal_date < start_date or signal_date > end_date:
            continue
        spy_history = [spy_by_date[key] for key in dates if key <= signal_date]
        if len(spy_history) < 61:
            continue
        if _match_setup(setup_name, ticker_bars[: index + 1], spy_history):
            signals.append(index)
            last = index
    return signals


def render_option_replay(report: Mapping[str, Any]) -> str:
    train = report["train_metrics"]
    test = report["test_metrics"]
    lines = [
        "# CORAT Exact ORATS Option Replay — {} {}".format(report["ticker"], report["setup"]),
        "",
        "Status: **EXACT-CHAIN RESEARCH DIAGNOSTIC — NO PRODUCTION PROMOTION**",
        "",
        "Signal T uses price data through T; entry uses the next session's historical ORATS chain; exit uses the exact same expiration and strikes from the historical exit chain. Entry is midpoint shifted 25% toward the debit natural; exit is natural shifted only 25% toward midpoint. Commissions are included.",
        "",
        "Split date: {}  ".format(report["split_date"]),
        "Signals found / completed / missed: {} / {} / {}  ".format(report["signals_found"], report["completed"], report["missed"]),
        "",
        "| Split | N | Win | EV $ | Avg return/risk | PF | Total $ | Max DD $ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Train | {} | {} | {} | {} | {} | {} | {} |".format(train["n"], _pct(train["win_rate"]), _money(train["expectancy_dollars"]), _pct(train["average_return_on_max_loss"]), _num(train["profit_factor"]), _money(train["total_pnl_dollars"]), _money(train["max_drawdown_dollars"])),
        "| Test | {} | {} | {} | {} | {} | {} | {} |".format(test["n"], _pct(test["win_rate"]), _money(test["expectancy_dollars"]), _pct(test["average_return_on_max_loss"]), _num(test["profit_factor"]), _money(test["total_pnl_dollars"]), _money(test["max_drawdown_dollars"])),
        "",
        "Execution-evidence gate: **{}**  ".format("PASS" if report["execution_evidence_gate"] else "FAIL"),
        "Production promotion: **FALSE** — this replay does not include catalyst, regime, sector, live-parity, correlated-book, or prospective-shadow evidence.",
        "",
        "## Completed trades",
        "",
        "| Signal | Entry | Exit | Expiration | Legs | Debit | Exit credit | P/L | Return/risk | Split |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["trades"]:
        lines.append("| {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} |".format(row["signal_date"], row["entry_date"], row["exit_date"], row["expiration"], row["long_strike"], row["short_strike"], _money(row["entry_debit"]), _money(row["exit_credit"]), _money(row["pnl_dollars"]), _pct(row["return_on_max_loss"]), row["split"]))
    return "\n".join(lines) + "\n"


def _money(value: Any) -> str:
    return "DATA UNAVAILABLE" if value is None else "${:,.2f}".format(float(value))


def _pct(value: Any) -> str:
    return "DATA UNAVAILABLE" if value is None else "{:.1%}".format(float(value))


def _num(value: Any) -> str:
    if value is None:
        return "DATA UNAVAILABLE"
    if math.isinf(float(value)):
        return "inf"
    return "{:.2f}".format(float(value))


def run_option_replay(
    config: Mapping[str, Any],
    token: str,
    ticker: str,
    setup_name: str,
    direction: str,
    start_date: str,
    end_date: str,
    split_date: str,
    holding_sessions: int = 10,
    max_signals: int = 40,
    offline: bool = False,
    refresh: bool = False,
    max_requests: Optional[int] = None,
) -> Dict[str, Any]:
    if direction not in {"BULLISH", "BEARISH"}:
        raise ValueError("direction must be BULLISH or BEARISH")
    start_day = date.fromisoformat(start_date)
    end_day = date.fromisoformat(end_date)
    split_day = date.fromisoformat(split_date)
    if not start_day <= split_day <= end_day:
        raise ValueError("option replay split date must fall inside the replay window")
    if end_day > date.fromisoformat(today_new_york()):
        raise ValueError("option replay end date cannot be in the future")
    if holding_sessions <= 0 or max_signals <= 0:
        raise ValueError("holding sessions and max signals must be positive")
    orats_cfg = config["orats"]
    client = OratsClient(
        token,
        str(orats_cfg["base_url"]),
        _resolve(config, "cache_root"),
        _resolve(config, "state_root"),
        float(orats_cfg["request_timeout_seconds"]),
        int(max_requests or max(int(orats_cfg["max_requests_per_run"]), max_signals * 2 + 5)),
        int(orats_cfg["monthly_request_cap"]),
        int(orats_cfg["requests_per_minute"]),
        offline,
        refresh,
    )
    price_start = (date.fromisoformat(start_date) - timedelta(days=400)).isoformat()
    prices = client.fetch_dailies([ticker, "SPY"], price_start, end_date, 2)
    bars = bars_from_dailies(prices.rows)
    ticker_bars = bars.get(ticker.upper(), [])
    spy_bars = bars.get("SPY", [])
    if len(ticker_bars) < 100 or len(spy_bars) < 100:
        raise ValueError("insufficient ORATS daily history for option replay")
    indices = _signal_indices(
        setup_name,
        ticker_bars,
        spy_bars,
        start_date,
        end_date,
        holding_sessions,
        int(config["history"]["signal_spacing_sessions"]),
    )[-max_signals:]
    traces = list(prices.traces)
    errors = list(prices.errors)
    trades = []
    missed = []
    liquidity = config["liquidity"]
    for index in indices:
        signal_date = ticker_bars[index].date
        entry_index = index + 1
        exit_index = entry_index + holding_sessions
        if exit_index >= len(ticker_bars):
            missed.append({"signal_date": signal_date, "reason": "missing forward session"})
            continue
        entry_date = ticker_bars[entry_index].date
        exit_date = ticker_bars[exit_index].date
        signal_snapshot = technical_snapshot(ticker.upper(), ticker_bars[: index + 1], signal_date)
        if signal_snapshot is None:
            missed.append({"signal_date": signal_date, "reason": "missing signal snapshot"})
            continue
        signal = SetupSignal(setup_name, direction, 1.0, True, "frozen replay setup", "next-session entry", "frozen structural invalidation")
        plan = build_stock_plan(signal_snapshot, signal, None, float(config["risk"]["normal_risk_pct"]))
        if plan is None:
            missed.append({"signal_date": signal_date, "reason": "missing risk plan"})
            continue
        entry_chain = client.fetch_chain(ticker, entry_date, int(orats_cfg["min_dte"]), int(orats_cfg["max_dte"]))
        traces.extend(entry_chain.traces)
        errors.extend(entry_chain.errors)
        structure = choose_debit_spread(
            entry_chain.rows,
            direction,
            plan.target_1,
            holding_sessions,
            int(liquidity["minimum_option_open_interest"]),
            int(liquidity["minimum_option_volume"]),
            float(liquidity["maximum_option_spread_pct"]),
        )
        if not structure.valid or len(structure.legs) != 2 or structure.expected_entry is None or structure.maximum_loss is None:
            missed.append({"signal_date": signal_date, "reason": "; ".join(structure.reasons) or "no valid entry structure"})
            continue
        exit_chain = client.fetch_historical_chain_full(ticker, exit_date, max_dte=120)
        traces.extend(exit_chain.traces)
        errors.extend(exit_chain.errors)
        exit_credit = conservative_exit_credit(
            list(exit_chain.rows),
            direction,
            structure.expiration,
            structure.legs[0].strike,
            structure.legs[1].strike,
        )
        if exit_credit is None:
            missed.append({"signal_date": signal_date, "reason": "exact exit legs unavailable"})
            continue
        commissions = 4.0 * 0.65
        pnl = (exit_credit - structure.expected_entry) * 100.0 - commissions
        trades.append(
            {
                "ticker": ticker.upper(),
                "setup": setup_name,
                "direction": direction,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "expiration": structure.expiration,
                "long_strike": structure.legs[0].strike,
                "short_strike": structure.legs[1].strike,
                "entry_debit": structure.expected_entry,
                "entry_natural": structure.natural_entry,
                "exit_credit": exit_credit,
                "maximum_loss": structure.maximum_loss,
                "pnl_dollars": pnl,
                "return_on_max_loss": pnl / structure.maximum_loss,
                "commissions_dollars": commissions,
                "split": "TRAIN" if signal_date <= split_date else "TEST",
            }
        )
    train_rows = [row for row in trades if row["split"] == "TRAIN"]
    test_rows = [row for row in trades if row["split"] == "TEST"]
    train_metrics = _metrics(train_rows)
    test_metrics = _metrics(test_rows)
    execution_gate = bool(
        test_metrics["n"] >= 100
        and (test_metrics["expectancy_dollars"] or 0) > 0
        and (test_metrics["profit_factor"] or 0) > 1.0
        and len({row["entry_date"] for row in test_rows}) >= 40
    )
    report: Dict[str, Any] = {
        "schema_version": "corat.option_replay.v1",
        "status": "EXACT_CHAIN_RESEARCH_DIAGNOSTIC_NO_PRODUCTION_PROMOTION",
        "ticker": ticker.upper(),
        "setup": setup_name,
        "direction": direction,
        "start_date": start_date,
        "end_date": end_date,
        "split_date": split_date,
        "holding_sessions": holding_sessions,
        "signals_found": len(indices),
        "completed": len(trades),
        "missed": len(missed),
        "missed_rows": missed,
        "trades": trades,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "execution_evidence_gate": execution_gate,
        "production_promotion": False,
        "generated_at_utc": utc_now(),
        "source_traces": [trace.to_dict() for trace in traces],
        "source_errors": sorted(set(errors)),
        "orats_usage": client.usage(),
    }
    digest = sha256_bytes(canonical_json({"ticker": ticker, "setup": setup_name, "trades": trades, "generated": report["generated_at_utc"]}).encode("utf-8"))[:12]
    run_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), digest)
    out = _resolve(config, "output_root") / "option_replays" / ticker.upper() / run_id
    out.mkdir(parents=True, exist_ok=False)
    json_path = out / "option_replay.json"
    md_path = out / "option_replay.md"
    write_json(json_path, report)
    write_text(md_path, render_option_replay(report))
    manifest_path = out / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "corat.option_replay_manifest.v1",
            "run_id": run_id,
            "outputs": {str(json_path): sha256_file(json_path), str(md_path): sha256_file(md_path)},
            "production_promotion": False,
            "order_submission_surface": False,
        },
    )
    report["artifacts"] = {"run_dir": str(out), "report": str(md_path), "json": str(json_path), "manifest": str(manifest_path)}
    return report
