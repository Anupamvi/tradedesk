from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from uwos.wheel_vendor.occ import parse_occ_symbol
from uwos.fresh_wheel_schwab import (
    OptionContract,
    REPLAY_BLOCKED_CSP,
    UniverseRow,
    WheelConfig,
    action_confidence,
    build_universe,
    bs_delta,
    coerce_date,
    find_export,
    limit_for_sell,
    normalize_symbol,
    pick_csp,
    read_csv_export,
    safe_float,
    safe_int,
)


@dataclass
class ReplaySignal:
    signal_date: dt.date
    ticker: str
    action: str
    confidence: float
    spot: float
    quality_score: float
    flow_score: float
    option_symbol: str
    expiry: dt.date
    strike: float
    entry_credit: float
    dte: int
    alert_price: float | None
    entry_date: dt.date | None = None
    exit_date: dt.date | None = None
    exit_mark: float | None = None
    exit_reason: str = ""
    pnl_per_contract: float | None = None
    credit_capture_pct: float | None = None
    return_on_cash_pct: float | None = None
    hit_50pct_target: bool | None = None
    outcome_status: str = "pending"


@dataclass
class ReplayMetrics:
    sessions: int
    signals: int
    scored: int
    target_hits: int
    profitable: int
    precision_50pct: float | None
    profitable_rate: float | None
    avg_return_on_cash_pct: float | None
    avg_pnl_per_contract: float | None
    strong_signals: int
    strong_scored: int
    strong_target_hits: int
    strong_precision_50pct: float | None
    strong_avg_return_on_cash_pct: float | None
    strong_avg_pnl_per_contract: float | None


class ReplayDataCache:
    def __init__(self) -> None:
        self._closes: dict[Path, dict[str, float]] = {}
        self._option_marks: dict[Path, dict[str, float]] = {}

    def screener_closes(self, folder: Path) -> dict[str, float]:
        if folder not in self._closes:
            self._closes[folder] = load_screener_closes(folder)
        return self._closes[folder]

    def option_marks(self, folder: Path) -> dict[str, float]:
        if folder not in self._option_marks:
            self._option_marks[folder] = option_mark_by_symbol(folder)
        return self._option_marks[folder]


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def has_replay_inputs(path: Path) -> bool:
    return bool(
        (list(path.glob("stock-screener-*.csv")) or list(path.glob("stock-screener-*.zip")))
        and (list(path.glob("hot-chains-*.csv")) or list(path.glob("hot-chains-*.zip")))
    )


def iter_sessions(data_root: Path, start: dt.date | None, end: dt.date | None) -> list[tuple[dt.date, Path]]:
    sessions: list[tuple[dt.date, Path]] = []
    for path in data_root.iterdir():
        if not path.is_dir():
            continue
        if len(path.name) != 10:
            continue
        day = coerce_date(path.name)
        if day is None:
            continue
        if path.name != day.isoformat():
            continue
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        if has_replay_inputs(path):
            sessions.append((day, path))
    return sorted(sessions)


def load_screener_closes(base_dir: Path) -> dict[str, float]:
    path = find_export(base_dir, "stock-screener-")
    df = read_csv_export(path)
    if "ticker" not in df.columns or "close" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = normalize_symbol(str(row.get("ticker") or ""))
        close = safe_float(row.get("close"))
        if ticker and math.isfinite(close) and close > 0:
            out[ticker] = close
    return out


def _chain_underlying_close(row: pd.Series, fallback_spot: float) -> float:
    for col in ("close.1", "underlying_price", "stock_price"):
        if col in row.index:
            value = safe_float(row.get(col))
            if math.isfinite(value) and value > 0:
                return value
    return fallback_spot


def load_uw_option_contracts(base_dir: Path, asof: dt.date, tickers: Iterable[str], config: WheelConfig) -> list[OptionContract]:
    wanted = {normalize_symbol(ticker) for ticker in tickers if str(ticker).strip()}
    if not wanted:
        return []
    path = find_export(base_dir, "hot-chains-")
    df = read_csv_export(path)
    if "option_symbol" not in df.columns:
        return []
    parsed = df["option_symbol"].map(parse_occ_symbol)
    df = df.assign(
        _root=parsed.map(lambda item: normalize_symbol(item.root) if item else ""),
        _expiry=parsed.map(lambda item: item.expiry if item else None),
        _right=parsed.map(lambda item: item.right if item else ""),
        _strike=parsed.map(lambda item: item.strike if item else math.nan),
    )
    df = df[df["_root"].isin(wanted)].copy()
    if df.empty:
        return []
    contracts: list[OptionContract] = []
    for _, row in df.iterrows():
        expiry = row.get("_expiry")
        if not isinstance(expiry, dt.date):
            continue
        dte = (expiry - asof).days
        if dte <= 0:
            continue
        right = str(row.get("_right") or "").upper()
        strike = safe_float(row.get("_strike"))
        bid = safe_float(row.get("bid"), 0.0)
        ask = safe_float(row.get("ask"), 0.0)
        last = safe_float(row.get("close"), 0.0)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif last > 0:
            mid = last
        else:
            mid = 0.0
        spread_pct = (ask - bid) / mid if mid > 0 and ask >= bid else 999.0
        underlying = _chain_underlying_close(row, 0.0)
        iv = safe_float(row.get("iv"), math.nan)
        delta = bs_delta(underlying, strike, dte, iv, right == "C", config.risk_free_rate)
        contracts.append(
            OptionContract(
                symbol=str(row.get("option_symbol") or "").strip().upper(),
                expiry=expiry,
                dte=dte,
                right=right,
                strike=strike,
                bid=bid,
                ask=ask,
                mark=last if last > 0 else mid,
                mid=mid,
                delta=delta,
                iv=iv,
                open_interest=safe_int(row.get("open_interest")),
                volume=safe_int(row.get("volume")),
                spread_pct=spread_pct,
            )
        )
    return contracts


def option_mark_by_symbol(base_dir: Path) -> dict[str, float]:
    try:
        path = find_export(base_dir, "hot-chains-")
    except FileNotFoundError:
        return {}
    df = read_csv_export(path)
    if "option_symbol" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        symbol = str(row.get("option_symbol") or "").strip().upper()
        if not symbol:
            continue
        bid = safe_float(row.get("bid"), 0.0)
        ask = safe_float(row.get("ask"), 0.0)
        last = safe_float(row.get("close"), 0.0)
        if bid > 0 and ask > 0:
            mark = (bid + ask) / 2.0
        elif last > 0:
            mark = last
        else:
            continue
        out[symbol] = mark
    return out


def is_strong_signal(signal: ReplaySignal) -> bool:
    if signal.action != "OPEN_CSP" or signal.strike <= 0 or signal.entry_credit <= 0:
        return False
    premium_yield = signal.entry_credit / signal.strike * 100.0
    return signal.confidence >= 85.0 and premium_yield >= 1.0 and signal.entry_credit * 100.0 >= 300.0


def build_signal(row: UniverseRow, put: OptionContract, asof: dt.date) -> ReplaySignal | None:
    if row.close <= 0:
        return None
    discount_pct = (row.close - put.strike) / row.close * 100.0
    confidence = action_confidence(row, put, 12.0 + min(discount_pct, 8.0))
    entry_credit = limit_for_sell(put)
    if confidence >= 78.0 and discount_pct >= 3.0:
        action = "OPEN_CSP"
        alert_price = None
        entry_date = asof
    else:
        action = "SET_CSP_ALERT"
        alert_price = round(min(row.close * 0.97, put.strike * 1.02), 2)
        entry_date = None
    if row.ticker in REPLAY_BLOCKED_CSP:
        return None
    if confidence < 67.0:
        return None
    return ReplaySignal(
        signal_date=asof,
        ticker=row.ticker,
        action=action,
        confidence=confidence,
        spot=row.close,
        quality_score=row.quality_score,
        flow_score=row.flow_score,
        option_symbol=put.symbol,
        expiry=put.expiry,
        strike=put.strike,
        entry_credit=entry_credit,
        dte=put.dte,
        alert_price=alert_price,
        entry_date=entry_date,
    )


def generate_signals_for_session(base_dir: Path, asof: dt.date, config: WheelConfig) -> list[ReplaySignal]:
    universe = build_universe(base_dir, config)
    contracts = load_uw_option_contracts(base_dir, asof, [row.ticker for row in universe], config)
    puts_by_ticker: dict[str, list[OptionContract]] = {}
    for contract in contracts:
        if contract.right != "P":
            continue
        parsed = parse_occ_symbol(contract.symbol)
        if parsed is None:
            continue
        puts_by_ticker.setdefault(normalize_symbol(parsed.root), []).append(contract)
    signals: list[ReplaySignal] = []
    for row in universe:
        put = pick_csp(row, puts_by_ticker.get(row.ticker, []), row.close, asof, config)
        if put is None:
            continue
        signal = build_signal(row, put, asof)
        if signal is not None:
            signals.append(signal)
    return sorted(signals, key=lambda item: item.confidence, reverse=True)


def score_signal(
    signal: ReplaySignal,
    sessions: list[tuple[dt.date, Path]],
    start_index: int,
    *,
    alert_window_days: int,
    management_window_days: int,
    cache: ReplayDataCache | None = None,
) -> ReplaySignal:
    cache = cache or ReplayDataCache()
    entry_date = signal.entry_date
    entry_credit = signal.entry_credit
    if signal.action == "SET_CSP_ALERT":
        trigger_deadline_index = min(len(sessions) - 1, start_index + alert_window_days)
        for idx in range(start_index + 1, trigger_deadline_index + 1):
            day, folder = sessions[idx]
            close = cache.screener_closes(folder).get(signal.ticker)
            if close is None or signal.alert_price is None or close > signal.alert_price:
                continue
            mark = cache.option_marks(folder).get(signal.option_symbol)
            if mark is None:
                continue
            entry_date = day
            entry_credit = mark
            break
        if entry_date is None:
            signal.outcome_status = "alert_not_triggered"
            return signal

    assert entry_date is not None
    target_mark = entry_credit * 0.50
    exit_deadline = min(signal.expiry, entry_date + dt.timedelta(days=management_window_days * 2))
    last_mark: tuple[dt.date, float] | None = None
    for idx in range(start_index + 1, len(sessions)):
        day, folder = sessions[idx]
        if day > exit_deadline:
            break
        mark = cache.option_marks(folder).get(signal.option_symbol)
        if mark is None:
            continue
        last_mark = (day, mark)
        if mark <= target_mark:
            signal.entry_date = entry_date
            signal.entry_credit = round(entry_credit, 4)
            signal.exit_date = day
            signal.exit_mark = round(mark, 4)
            signal.exit_reason = "hit_50pct_target"
            signal.pnl_per_contract = round((entry_credit - mark) * 100.0, 2)
            signal.credit_capture_pct = round((entry_credit - mark) / entry_credit * 100.0, 2) if entry_credit > 0 else None
            signal.return_on_cash_pct = round((entry_credit - mark) / signal.strike * 100.0, 4) if signal.strike > 0 else None
            signal.hit_50pct_target = True
            signal.outcome_status = "scored"
            return signal

    if last_mark is None:
        signal.entry_date = entry_date
        signal.entry_credit = round(entry_credit, 4)
        signal.outcome_status = "no_exit_quote"
        return signal
    day, mark = last_mark
    signal.entry_date = entry_date
    signal.entry_credit = round(entry_credit, 4)
    signal.exit_date = day
    signal.exit_mark = round(mark, 4)
    signal.exit_reason = "horizon_mark"
    signal.pnl_per_contract = round((entry_credit - mark) * 100.0, 2)
    signal.credit_capture_pct = round((entry_credit - mark) / entry_credit * 100.0, 2) if entry_credit > 0 else None
    signal.return_on_cash_pct = round((entry_credit - mark) / signal.strike * 100.0, 4) if signal.strike > 0 else None
    signal.hit_50pct_target = bool(mark <= target_mark)
    signal.outcome_status = "scored"
    return signal


def calculate_metrics(outcomes: list[ReplaySignal], sessions: int) -> ReplayMetrics:
    scored = [item for item in outcomes if item.outcome_status == "scored" and item.pnl_per_contract is not None]
    strong = [item for item in outcomes if is_strong_signal(item)]
    strong_scored = [item for item in scored if is_strong_signal(item)]
    target_hits = sum(1 for item in scored if item.hit_50pct_target)
    profitable = sum(1 for item in scored if (item.pnl_per_contract or 0.0) > 0)
    returns = [item.return_on_cash_pct for item in scored if item.return_on_cash_pct is not None]
    pnls = [item.pnl_per_contract for item in scored if item.pnl_per_contract is not None]
    strong_hits = sum(1 for item in strong_scored if item.hit_50pct_target)
    strong_returns = [item.return_on_cash_pct for item in strong_scored if item.return_on_cash_pct is not None]
    strong_pnls = [item.pnl_per_contract for item in strong_scored if item.pnl_per_contract is not None]
    return ReplayMetrics(
        sessions=sessions,
        signals=len(outcomes),
        scored=len(scored),
        target_hits=target_hits,
        profitable=profitable,
        precision_50pct=target_hits / len(scored) if scored else None,
        profitable_rate=profitable / len(scored) if scored else None,
        avg_return_on_cash_pct=sum(returns) / len(returns) if returns else None,
        avg_pnl_per_contract=sum(pnls) / len(pnls) if pnls else None,
        strong_signals=len(strong),
        strong_scored=len(strong_scored),
        strong_target_hits=strong_hits,
        strong_precision_50pct=strong_hits / len(strong_scored) if strong_scored else None,
        strong_avg_return_on_cash_pct=sum(strong_returns) / len(strong_returns) if strong_returns else None,
        strong_avg_pnl_per_contract=sum(strong_pnls) / len(strong_pnls) if strong_pnls else None,
    )


def run_replay(
    *,
    data_root: Path,
    start: dt.date | None,
    end: dt.date | None,
    out_dir: Path,
    config: WheelConfig,
    max_signals_per_session: int,
    alert_window_days: int,
    management_window_days: int,
) -> tuple[list[ReplaySignal], ReplayMetrics, dict[str, Path]]:
    sessions = iter_sessions(data_root, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[ReplaySignal] = []
    blocked_until: dict[str, dt.date] = {}
    cache = ReplayDataCache()
    for idx, (day, folder) in enumerate(sessions):
        if idx >= len(sessions) - 1:
            continue
        try:
            signals = generate_signals_for_session(folder, day, config)
        except Exception:
            continue
        emitted = 0
        for signal in signals:
            if emitted >= max_signals_per_session:
                break
            if blocked_until.get(signal.ticker, dt.date.min) > day:
                continue
            scored = score_signal(
                signal,
                sessions,
                idx,
                alert_window_days=alert_window_days,
                management_window_days=management_window_days,
                cache=cache,
            )
            outcomes.append(scored)
            blocked_until[signal.ticker] = day + dt.timedelta(days=management_window_days)
            emitted += 1
    metrics = calculate_metrics(outcomes, len(sessions))
    outputs = write_replay_outputs(out_dir, start, end, outcomes, metrics, config)
    return outcomes, metrics, outputs


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.1f}%"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def write_replay_outputs(
    out_dir: Path,
    start: dt.date | None,
    end: dt.date | None,
    outcomes: list[ReplaySignal],
    metrics: ReplayMetrics,
    config: WheelConfig,
) -> dict[str, Path]:
    suffix = f"{start.isoformat() if start else 'start'}_{end.isoformat() if end else 'end'}"
    report_path = out_dir / f"fresh-wheel-replay-{suffix}.md"
    outcomes_csv = out_dir / f"fresh-wheel-replay-outcomes-{suffix}.csv"
    manifest_path = out_dir / f"fresh-wheel-replay-manifest-{suffix}.json"

    fieldnames = list(asdict(outcomes[0]).keys()) if outcomes else list(ReplaySignal.__dataclass_fields__.keys())
    with outcomes_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in outcomes:
            data = asdict(item)
            for key in ("signal_date", "expiry", "entry_date", "exit_date"):
                value = data.get(key)
                data[key] = value.isoformat() if isinstance(value, dt.date) else ""
            writer.writerow(data)

    lines = [
        f"# Fresh Wheel Replay ({suffix})",
        "",
        "## Scope",
        "",
        "- Strategy under test: fresh CSP selector from `uwos.fresh_wheel_schwab`.",
        "- Historical quote source: local UW `hot-chains` snapshots.",
        "- Live Schwab chains are intentionally not used in historical replay.",
        f"- Sessions loaded: `{metrics.sessions}`",
        f"- Signals emitted: `{metrics.signals}`",
        f"- Scored executions: `{metrics.scored}`",
        "",
        "## Metrics",
        "",
        f"- 50% credit-capture hits: `{metrics.target_hits}`",
        f"- Precision at 50% target: `{_fmt_pct(metrics.precision_50pct)}`",
        f"- Profitable mark rate: `{_fmt_pct(metrics.profitable_rate)}`",
        f"- Average return on secured cash: `{_fmt_num(metrics.avg_return_on_cash_pct)}%`",
        f"- Average PnL per contract: `${_fmt_num(metrics.avg_pnl_per_contract)}`",
        f"- Strong ex-ante signals: `{metrics.strong_signals}`",
        f"- Strong scored executions: `{metrics.strong_scored}`",
        f"- Strong 50% credit-capture hits: `{metrics.strong_target_hits}`",
        f"- Strong precision at 50% target: `{_fmt_pct(metrics.strong_precision_50pct)}`",
        f"- Strong average return on secured cash: `{_fmt_num(metrics.strong_avg_return_on_cash_pct)}%`",
        f"- Strong average PnL per contract: `${_fmt_num(metrics.strong_avg_pnl_per_contract)}`",
        "",
        "## Notes",
        "",
        "- This replay is a calibration gate, not a production guarantee.",
        "- It only scores contracts that remain visible in later UW hot-chain snapshots; missing later quotes are marked explicitly.",
        "- Overlapping same-symbol signals are suppressed for the management window to reduce sample inflation.",
        "- Replay-blocked CSP names are excluded from signal generation.",
        "",
    ]
    scored = [item for item in outcomes if item.outcome_status == "scored"]
    if scored:
        weekly: dict[tuple[int, int], list[ReplaySignal]] = {}
        for item in scored:
            iso = item.signal_date.isocalendar()
            weekly.setdefault((iso.year, iso.week), []).append(item)
        lines.extend(
            [
                "## Strong-Signal Weekly Cadence",
                "",
                "| Week | Scored | Strong Scored | Strong Hit 50% | Strong Avg PnL | Strong Tickers |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for year_week in sorted(weekly):
            rows = weekly[year_week]
            strong_rows = [item for item in rows if is_strong_signal(item)]
            strong_hits = sum(1 for item in strong_rows if item.hit_50pct_target)
            strong_pnls = [item.pnl_per_contract for item in strong_rows if item.pnl_per_contract is not None]
            avg_strong_pnl = sum(strong_pnls) / len(strong_pnls) if strong_pnls else None
            tickers = ", ".join(item.ticker for item in strong_rows) if strong_rows else "-"
            lines.append(
                f"| {year_week[0]}-W{year_week[1]:02d} | {len(rows)} | {len(strong_rows)} | {strong_hits} | "
                f"${_fmt_num(avg_strong_pnl)} | {tickers} |"
            )
        lines.append("")
    if scored:
        lines.extend(["## Top Scored Rows", "", "| Date | Ticker | Action | Credit | Exit | PnL | Hit 50% |", "| --- | --- | --- | ---: | ---: | ---: | --- |"])
        for item in sorted(scored, key=lambda row: row.signal_date, reverse=True)[:20]:
            lines.append(
                f"| {item.signal_date.isoformat()} | {item.ticker} | {item.action} | {item.entry_credit:.2f} | "
                f"{item.exit_mark if item.exit_mark is not None else 0.0:.2f} | {item.pnl_per_contract or 0.0:.2f} | "
                f"{'yes' if item.hit_50pct_target else 'no'} |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "strategy": "fresh_wheel_schwab_csp_selector",
        "historical_quote_source": "local_uw_hot_chains",
        "schwab_live_used_for_replay": False,
        "outputs": {
            "report": str(report_path),
            "outcomes_csv": str(outcomes_csv),
        },
        "metrics": asdict(metrics),
        "config": asdict(config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": report_path, "outcomes_csv": outcomes_csv, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical replay for the fresh wheel CSP selector using local UW snapshots.")
    parser.add_argument("--data-root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-04-10")
    parser.add_argument("--out-dir", default="/Users/anuppamvi/uw_root/tradedesk/out/fresh_wheel_replay")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--max-signals-per-session", type=int, default=3)
    parser.add_argument("--alert-window-days", type=int, default=5)
    parser.add_argument("--management-window-days", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WheelConfig(max_symbols=args.max_symbols)
    _, metrics, outputs = run_replay(
        data_root=Path(args.data_root).expanduser().resolve(),
        start=parse_date(args.start),
        end=parse_date(args.end),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        config=config,
        max_signals_per_session=args.max_signals_per_session,
        alert_window_days=args.alert_window_days,
        management_window_days=args.management_window_days,
    )
    print(f"Report:   {outputs['report']}")
    print(f"Outcomes: {outputs['outcomes_csv']}")
    print(f"Manifest: {outputs['manifest']}")
    print(f"Signals:  {metrics.signals}")
    print(f"Scored:   {metrics.scored}")
    print(f"Hit50:    {metrics.target_hits}")
    print(f"Precision@50: {_fmt_pct(metrics.precision_50pct)}")


if __name__ == "__main__":
    main()
