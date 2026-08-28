"""Frozen price-setup walk-forward diagnostics.

This intentionally does not reconstruct historical option prices. It is a
first-stage underlying-edge test. Option promotion requires a separate exact
historical-chain replay and is therefore always false in this report.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from corat.config import PROJECT_ROOT, load_universe, supporting_tickers
from corat.clock import today_new_york
from corat.constants import DATA_UNAVAILABLE
from corat.history import analyze_analogues
from corat.orats import OratsClient
from corat.store import canonical_json, sha256_bytes, sha256_file, utc_now, write_json, write_text
from corat.technical import bars_from_dailies


SETUPS = (
    ("TREND PULLBACK", "BULLISH"),
    ("BREAKOUT + CONFIRMATION", "BULLISH"),
    ("RELATIVE-STRENGTH LEADER", "BULLISH"),
    ("EMERGING SECTOR ROTATION", "BULLISH"),
    ("OVERSOLD REVERSAL", "BULLISH"),
    ("FAILED BREAKOUT / TREND BREAKDOWN", "BEARISH"),
)


def _resolve(config: Mapping[str, Any], key: str) -> Path:
    path = Path(str(config[key]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _metric(stats: Mapping[str, Any], key: str) -> str:
    value = stats.get(key)
    if value is None:
        return DATA_UNAVAILABLE
    if key in {"expectancy", "win_rate", "max_drawdown"}:
        return "{:.2%}".format(float(value))
    return "{:.3f}".format(float(value))


def render_backtest(report: Mapping[str, Any]) -> str:
    lines = [
        "# CORAT Frozen Price-Setup Backtest — through {}".format(report.get("as_of")),
        "",
        "Status: **PRICE ANALOG DIAGNOSTIC — NO OPTION OR PRODUCTION PROMOTION**",
        "",
        "Chronological split: train signals through {}; test signals from {}. The rules are frozen across both periods. Forward returns are direction-adjusted and signals are spaced to reduce overlap.".format(report.get("train_end"), report.get("test_start")),
        "",
        "| Ticker | Setup | Train N | Train EV | Test N | Test win | Test EV | Test PF | Test DD | Eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        train = row["train"]
        test = row["test"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["ticker"], row["setup"], train["sample_size"], _metric(train, "expectancy"),
                test["sample_size"], _metric(test, "win_rate"), _metric(test, "expectancy"),
                _metric(test, "profit_factor"), _metric(test, "max_drawdown"), row["underlying_research_eligible"],
            )
        )
    lines.extend(
        [
            "",
            "Promotion rules used here require at least 40 independent test observations, positive test expectancy, profit factor above 1, and no missing price evidence. Passing only makes an underlying setup research-eligible. It does not validate an option structure, live fills, or profitability.",
            "",
            "Exact historical option-chain replay: **NOT RUN / DATA UNAVAILABLE IN THIS DIAGNOSTIC**.",
            "",
        ]
    )
    return "\n".join(lines)


def run_backtest(
    config: Mapping[str, Any],
    token: str,
    as_of: str,
    tickers: Optional[Sequence[str]] = None,
    split_date: Optional[str] = None,
    offline: bool = False,
    refresh: bool = False,
    max_requests: Optional[int] = None,
) -> Dict[str, Any]:
    end_day = date.fromisoformat(as_of)
    if end_day > date.fromisoformat(today_new_york()):
        raise ValueError("CORAT backtest end date cannot be in the future")
    start_day = end_day - timedelta(days=int(config["lookback_calendar_days"]))
    split = date.fromisoformat(split_date) if split_date else start_day + (end_day - start_day) * 2 // 3
    if not start_day < split < end_day:
        raise ValueError("backtest split date must fall inside the configured history window")
    test_start = (split + timedelta(days=1)).isoformat()
    selected = load_universe(config, tickers=tickers)
    candidates = [item for item in selected if item.kind in {"equity", "benchmark", "sector_etf"}]
    names = supporting_tickers(config, candidates)
    orats_cfg = config["orats"]
    client = OratsClient(
        token,
        str(orats_cfg["base_url"]),
        _resolve(config, "cache_root"),
        _resolve(config, "state_root"),
        float(orats_cfg["request_timeout_seconds"]),
        int(max_requests or orats_cfg["max_requests_per_run"]),
        int(orats_cfg["monthly_request_cap"]),
        int(orats_cfg["requests_per_minute"]),
        offline,
        refresh,
    )
    fetched = client.fetch_dailies(names, start_day.isoformat(), as_of, int(orats_cfg["batch_size"]))
    bars = bars_from_dailies(fetched.rows)
    history_cfg = config["history"]
    rows = []
    for item in candidates:
        ticker_bars = bars.get(item.ticker, [])
        spy_bars = bars.get("SPY", [])
        if len(ticker_bars) < 100 or len(spy_bars) < 100:
            continue
        for setup, direction in SETUPS:
            common = dict(
                setup_name=setup,
                direction=direction,
                bars=ticker_bars,
                spy_bars=spy_bars,
                horizons=[int(value) for value in history_cfg["forward_horizons"]],
                primary_horizon=int(history_cfg["primary_horizon_sessions"]),
                minimum_sample=int(history_cfg["minimum_analog_sample"]),
                maximum_sample=int(history_cfg["maximum_analog_sample"]),
                signal_spacing=int(history_cfg["signal_spacing_sessions"]),
            )
            train = analyze_analogues(as_of=split.isoformat(), signal_end_date=split.isoformat(), **common)
            test = analyze_analogues(as_of=as_of, signal_start_date=test_start, **common)
            eligible = bool(
                test.sample_size >= 40
                and test.expectancy is not None and test.expectancy > 0
                and test.profit_factor is not None and test.profit_factor > 1.0
            )
            if train.sample_size or test.sample_size:
                rows.append(
                    {
                        "ticker": item.ticker,
                        "setup": setup,
                        "direction": direction,
                        "train": train.to_dict(),
                        "test": test.to_dict(),
                        "underlying_research_eligible": eligible,
                        "option_promotion": False,
                    }
                )
    rows.sort(key=lambda row: ((row["test"].get("expectancy") or -999), row["test"].get("sample_size") or 0), reverse=True)
    report: Dict[str, Any] = {
        "schema_version": "corat.backtest.v1",
        "status": "PRICE_ANALOG_DIAGNOSTIC_NO_OPTION_PROMOTION",
        "as_of": as_of,
        "start": start_day.isoformat(),
        "train_end": split.isoformat(),
        "test_start": test_start,
        "generated_at_utc": utc_now(),
        "rows": rows,
        "source_traces": [trace.to_dict() for trace in fetched.traces],
        "source_errors": fetched.errors,
        "orats_usage": client.usage(),
    }
    digest = sha256_bytes(canonical_json({"as_of": as_of, "rows": rows, "generated": report["generated_at_utc"]}).encode("utf-8"))[:12]
    run_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), digest)
    out = _resolve(config, "output_root") / "backtests" / as_of / run_id
    out.mkdir(parents=True, exist_ok=False)
    json_path = out / "backtest.json"
    md_path = out / "backtest.md"
    write_json(json_path, report)
    write_text(md_path, render_backtest(report))
    manifest = {
        "schema_version": "corat.backtest_manifest.v1",
        "run_id": run_id,
        "outputs": {str(json_path): sha256_file(json_path), str(md_path): sha256_file(md_path)},
        "option_promotion": False,
        "order_submission_surface": False,
    }
    manifest_path = out / "manifest.json"
    write_json(manifest_path, manifest)
    report["artifacts"] = {"run_dir": str(out), "report": str(md_path), "json": str(json_path), "manifest": str(manifest_path)}
    return report
