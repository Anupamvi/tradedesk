"""CORAT command-line interface."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from corat import __version__
from corat.backtest import run_backtest
from corat.clock import today_new_york
from corat.config import PROJECT_ROOT, load_config, load_universe
from corat.context import context_template
from corat.full_replay import (
    authorize_replay,
    build_replay_plan,
    local_orats_usage,
    run_full_replay,
)
from corat.ledger import (
    ledger_path,
    ledger_summary,
    read_events,
    record_plan,
    record_trade_event,
    render_open_trade_review,
    review_open_trades,
    trade_states,
)
from corat.orats import OratsClient
from corat.option_replay import run_option_replay
from corat.pipeline import compare_runs, render_delta, run_scan
from corat.research import build_auto_context
from corat.secrets import orats_token
from corat.store import read_json, write_json, write_text

def _tickers(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    values = [value.strip().upper() for value in text.split(",") if value.strip()]
    return values or None


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corat",
        description="Evidence-first ORATS-backed swing research. No order submission.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def scan_arguments(target: argparse.ArgumentParser, include_tickers: bool = True) -> None:
        target.add_argument("--date", default=today_new_york(), help="decision date YYYY-MM-DD")
        if include_tickers:
            target.add_argument("--tickers", default=None, help="comma-separated subset")
        target.add_argument("--config", type=Path, default=None)
        target.add_argument("--context", type=Path, default=None, help="corat.context.v1 JSON evidence")
        target.add_argument("--offline", action="store_true", help="never call ORATS; require cached inputs")
        target.add_argument("--refresh", action="store_true", help="refresh matching ORATS caches")
        target.add_argument("--max-requests", type=int, default=None)
        target.add_argument("--portfolio-nav", type=float, default=None)
        target.add_argument("--output-root", type=Path, default=None)
        target.add_argument("--validation", action="store_true", help="label this run as a validation smoke")
        target.add_argument("--no-schwab", action="store_true", help="disable optional read-only Schwab quotes")
        target.add_argument("--schwab-env", type=Path, default=None, help="Schwab dotenv path; values are never copied")

    run = sub.add_parser("run", help="run the research pipeline")
    scan_arguments(run)
    full = sub.add_parser("full-scan", help="scan the configured full universe")
    scan_arguments(full, include_tickers=False)
    full.add_argument("--no-auto-research", action="store_true", help="skip the automatic discovery/news-enrichment pass")
    full.add_argument("--research-limit", type=int, default=None, help="number of leading discovery names to enrich")
    analyze = sub.add_parser("analyze", help="run the complete framework on one ticker")
    analyze.add_argument("ticker")
    scan_arguments(analyze, include_tickers=False)
    delta = sub.add_parser("delta-scan", help="run and compare with a prior immutable run")
    scan_arguments(delta)
    delta.add_argument("--previous", default=None, help="prior run.json, latest.json, run directory, or YYYY-MM-DD")

    backtest = sub.add_parser("backtest", help="run frozen price-setup train/test diagnostics")
    backtest.add_argument("--date", default=today_new_york())
    backtest.add_argument("--split-date", default=None)
    backtest.add_argument("--tickers", default=None)
    backtest.add_argument("--config", type=Path, default=None)
    backtest.add_argument("--offline", action="store_true")
    backtest.add_argument("--refresh", action="store_true")
    backtest.add_argument("--max-requests", type=int, default=None)
    backtest.add_argument("--output-root", type=Path, default=None)

    replay = sub.add_parser("option-replay", help="replay exact historical ORATS debit spreads")
    replay.add_argument("ticker")
    replay.add_argument("--setup", required=True, choices=[
        "TREND PULLBACK", "BREAKOUT + CONFIRMATION", "RELATIVE-STRENGTH LEADER",
        "EMERGING SECTOR ROTATION", "OVERSOLD REVERSAL", "FAILED BREAKOUT / TREND BREAKDOWN",
    ])
    replay.add_argument("--direction", choices=["BULLISH", "BEARISH"], required=True)
    replay.add_argument("--start", required=True)
    replay.add_argument("--end", required=True)
    replay.add_argument("--split-date", required=True)
    replay.add_argument("--holding-sessions", type=int, default=10)
    replay.add_argument("--max-signals", type=int, default=40)
    replay.add_argument("--config", type=Path, default=None)
    replay.add_argument("--offline", action="store_true")
    replay.add_argument("--refresh", action="store_true")
    replay.add_argument("--max-requests", type=int, default=None)
    replay.add_argument("--output-root", type=Path, default=None)

    full_replay = sub.add_parser(
        "full-replay",
        help="plan a frozen full-pipeline walk-forward replay; does not run unless --execute is supplied",
    )
    full_replay.add_argument("--start", required=True)
    full_replay.add_argument("--end", required=True)
    full_replay.add_argument("--train-end", required=True)
    full_replay.add_argument("--validation-end", required=True)
    full_replay.add_argument("--tickers", default=None, help="comma-separated subset; omit for historical dynamic discovery")
    full_replay.add_argument("--spacing-sessions", type=int, default=1)
    full_replay.add_argument("--assumed-triggers-per-date", type=int, default=8)
    full_replay.add_argument("--assumed-option-trades-per-date", type=int, default=4)
    full_replay.add_argument("--max-trades-per-date", type=int, default=0, help="optional user cap; zero means no cap")
    full_replay.add_argument("--max-open-positions", type=int, default=0, help="optional user cap; zero means no cap")
    full_replay.add_argument("--initial-nav", type=float, default=100000.0)
    full_replay.add_argument("--risk-pct", type=float, default=None)
    full_replay.add_argument("--minimum-test-trades", type=int, default=40)
    full_replay.add_argument("--request-budget", type=int, default=None, help="hard ORATS request cap; required for online execution")
    full_replay.add_argument("--monthly-reserve", type=int, default=None, help="ORATS requests that must remain unused; required for online execution")
    full_replay.add_argument("--confirmed-remaining", type=int, default=None, help="current remaining requests shown by ORATS; required for online execution")
    full_replay.add_argument("--offline", action="store_true", help="execute from existing caches only; cannot make ORATS requests")
    full_replay.add_argument("--execute", action="store_true", help="explicitly start the replay; omission is plan-only")
    full_replay.add_argument("--config", type=Path, default=None)
    full_replay.add_argument("--output-root", type=Path, default=None)

    doctor = sub.add_parser("doctor", help="verify configuration and optional ORATS entitlement")
    doctor.add_argument("--config", type=Path, default=None)
    doctor.add_argument("--online", action="store_true")

    template = sub.add_parser("context-template", help="write an evidence-ingestion template")
    template.add_argument("--date", default=today_new_york())
    template.add_argument("--tickers", required=True)
    template.add_argument("--output", type=Path, default=None)

    plan = sub.add_parser("record-plan", help="copy one immutable candidate into the evidence ledger")
    plan.add_argument("--run", type=Path, required=True)
    plan.add_argument("--ticker", required=True)
    plan.add_argument("--trade-id", default=None)
    plan.add_argument("--scaling-plan", type=Path, default=None, help="predefined scaling JSON; cannot be added after entry")
    plan.add_argument("--config", type=Path, default=None)

    event = sub.add_parser("record-event", help="record submitted/fill/open/close evidence")
    event.add_argument("trade_id")
    event.add_argument("status", choices=["SUBMITTED","FILLED","OPEN","REDUCED","CLOSED","CANCELED","EXPIRED","REVIEW"])
    event.add_argument("--price", type=float, default=None)
    event.add_argument("--quantity", type=int, default=None, help="current total position quantity after this event")
    event.add_argument("--realized-pnl", type=float, default=None)
    event.add_argument("--mfe", type=float, default=None)
    event.add_argument("--mae", type=float, default=None)
    event.add_argument("--reason", default="")
    event.add_argument("--review-horizon-sessions", type=int, choices=[5,10,20], default=None)
    event.add_argument("--config", type=Path, default=None)

    ledger = sub.add_parser("ledger-report", help="summarize persisted recommendation and outcome evidence")
    ledger.add_argument("--config", type=Path, default=None)

    review = sub.add_parser("review-open-trades", help="reevaluate original theses for ledger open trades")
    scan_arguments(review, include_tickers=False)
    return parser


def _config(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    output = getattr(args, "output_root", None)
    if output is not None:
        overrides["output_root"] = str(output.expanduser().resolve())
    return load_config(args.config, overrides=overrides)


def _context(args: argparse.Namespace) -> Optional[Path]:
    if getattr(args, "context", None) is not None:
        return args.context
    candidate = PROJECT_ROOT / "inputs" / "context" / (str(args.date) + ".json")
    return candidate if candidate.is_file() else None


def _require_token() -> str:
    token = orats_token(PROJECT_ROOT)
    if not token:
        raise RuntimeError("ORATS_TOKEN missing. Put it in {}/.env or export ORATS_TOKEN; never paste it into a report.".format(PROJECT_ROOT))
    return token


def _print_run(result: Mapping[str, Any]) -> int:
    print("corat=ok")
    print("posture={}".format(result["posture"]))
    print("as_of={}".format(result["as_of"]))
    print("target_trades={}".format(result["diagnostics"].get("target_trades", 0)))
    print("option_target_trades={}".format(result["diagnostics"].get("option_target_trades", 0)))
    print("stock_target_trades={}".format(result["diagnostics"].get("stock_target_trades", 0)))
    print("candidates={}".format(len(result["candidates"])))
    print("orats_requests={}".format(result["orats_usage"]["run_requests"]))
    print("report={}".format(result["artifacts"]["report"]))
    print("manifest={}".format(result["artifacts"]["manifest"]))
    return 0


def _scan_result(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    token: str,
    names: Optional[Sequence[str]],
    context_path: Optional[Path],
    posture: str,
    use_schwab: bool,
) -> Dict[str, Any]:
    return run_scan(
        config,
        token,
        args.date,
        tickers=names,
        context_path=context_path,
        offline=args.offline,
        refresh=args.refresh,
        max_requests=args.max_requests,
        portfolio_nav=args.portfolio_nav,
        posture=posture,
        use_schwab=use_schwab,
        schwab_env_path=args.schwab_env,
    )


def _run_command(args: argparse.Namespace, names: Optional[Sequence[str]]) -> int:
    config = _config(args)
    result = _scan_result(
        args,
        config,
        _require_token(),
        names,
        _context(args),
        "VALIDATION_SMOKE_RESEARCH_ONLY" if args.validation else "RESEARCH_ONLY",
        not args.no_schwab,
    )
    return _print_run(result)


def _full_scan_command(args: argparse.Namespace) -> int:
    if args.offline or args.no_auto_research:
        return _run_command(args, None)
    config = _config(args)
    token = _require_token()
    discovery_config = copy.deepcopy(config)
    research_limit = int(args.research_limit or config.get("max_enriched_candidates") or 12)
    discovery_config["max_final_ideas"] = max(
        int(discovery_config.get("max_final_ideas") or 10),
        research_limit,
    )
    existing_context = _context(args)
    discovery = _scan_result(
        args,
        discovery_config,
        token,
        None,
        existing_context,
        "DISCOVERY_ONLY_NOT_FINAL",
        False,
    )
    auto_context_path = Path(discovery["artifacts"]["run_dir"]) / "auto-context.json"
    auto_context = build_auto_context(
        args.date,
        discovery["candidates"],
        auto_context_path,
        existing_path=existing_context,
        maximum_tickers=research_limit,
        lookback_days=int((config.get("research") or {}).get("lookback_days") or 21),
        timeout_seconds=float((config.get("research") or {}).get("request_timeout_seconds") or 15),
    )
    result = _scan_result(
        args,
        config,
        token,
        None,
        auto_context_path,
        "VALIDATION_SMOKE_RESEARCH_ONLY" if args.validation else "RESEARCH_ONLY",
        not args.no_schwab,
    )
    metadata = auto_context.get("research_metadata") or {}
    print("discovery_report={}".format(discovery["artifacts"]["report"]))
    print("auto_context={}".format(auto_context_path))
    print("researched_tickers={}".format(len(metadata.get("researched_tickers") or [])))
    print("research_evidence={}".format(metadata.get("evidence_rows_added_or_seen") or 0))
    return _print_run(result)


def _output_root(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["output_root"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _state_root(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["state_root"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "run.json"
    payload = read_json(resolved, {})
    if payload.get("run_path"):
        resolved = Path(payload["run_path"])
    return resolved


def _resolve_previous(config: Mapping[str, Any], current_date: str, value: Optional[str]) -> Path:
    output = _output_root(config)
    if value:
        direct = Path(value).expanduser()
        if direct.exists():
            if direct.is_dir():
                direct = direct / "run.json"
            payload = read_json(direct, {})
            if payload.get("run_path"):
                return Path(payload["run_path"])
            return direct
        day = output / value / "latest.json"
        payload = read_json(day, {})
        if payload.get("run_path"):
            return Path(payload["run_path"])
        raise RuntimeError("previous run not found: {}".format(value))
    dates = sorted(
        path.name for path in output.iterdir()
        if path.is_dir() and len(path.name) == 10 and path.name < current_date
    ) if output.is_dir() else []
    if not dates:
        raise RuntimeError("no earlier CORAT run is available")
    payload = read_json(output / dates[-1] / "latest.json", {})
    if not payload.get("run_path"):
        raise RuntimeError("latest pointer for {} is invalid".format(dates[-1]))
    return Path(payload["run_path"])


def _delta_command(args: argparse.Namespace) -> int:
    config = _config(args)
    previous_path = _resolve_previous(config, args.date, args.previous)
    previous = read_json(previous_path)
    if not isinstance(previous, dict):
        raise RuntimeError("previous run is unreadable: {}".format(previous_path))
    token = _require_token()
    current = run_scan(
        config,
        token,
        args.date,
        tickers=_tickers(args.tickers),
        context_path=_context(args),
        offline=args.offline,
        refresh=args.refresh,
        max_requests=args.max_requests,
        portfolio_nav=args.portfolio_nav,
        posture="VALIDATION_SMOKE_RESEARCH_ONLY" if args.validation else "RESEARCH_ONLY",
        use_schwab=not args.no_schwab,
        schwab_env_path=args.schwab_env,
    )
    delta = compare_runs(previous, current)
    run_dir = Path(current["artifacts"]["run_dir"])
    json_path = run_dir / "delta.json"
    report_path = run_dir / "delta.md"
    write_json(json_path, delta)
    write_text(report_path, render_delta(delta))
    print("corat_delta=ok")
    print("changes={}".format(len(delta["changes"])))
    print("report={}".format(report_path))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    token = orats_token(PROJECT_ROOT)
    discovery = config.get("discovery") or {}
    statuses = {
        "python": sys.version.split()[0],
        "project_root": str(PROJECT_ROOT),
        "config": str(config.get("_config_path")),
        "configured_seed_count": len(load_universe(config)),
        "dynamic_orats_discovery": bool(discovery.get("dynamic_orats_universe")),
        "dynamic_equity_target": int(discovery.get("maximum_equities") or 0),
        "orats_token": "AVAILABLE" if token else "DATA UNAVAILABLE",
        "order_submission_surface": False,
    }
    code = 0 if token else 2
    if args.online and token:
        _, cache_root, state_root = (
            Path(str(config[key])) if Path(str(config[key])).is_absolute() else PROJECT_ROOT / str(config[key])
            for key in ("output_root", "cache_root", "state_root")
        )
        cfg = config["orats"]
        client = OratsClient(
            token, str(cfg["base_url"]), cache_root, state_root,
            float(cfg["request_timeout_seconds"]), 2, int(cfg["monthly_request_cap"]),
            int(cfg["requests_per_minute"]), False, True,
        )
        probe = client.fetch_asof("cores", ["SPY"], today_new_york(), 1)
        statuses["orats_online"] = "AVAILABLE" if probe.rows else "DATA UNAVAILABLE"
        statuses["orats_rows"] = len(probe.rows)
        statuses["orats_errors"] = probe.errors
        if not probe.rows:
            code = 2
    for key, value in statuses.items():
        print("{}={}".format(key, value))
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _base_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "context-template":
            output = args.output or (PROJECT_ROOT / "inputs" / "context" / (args.date + ".json"))
            write_json(output, context_template(args.date, _tickers(args.tickers) or []))
            print("context_template={}".format(output.expanduser().resolve()))
            return 0
        if args.command == "record-plan":
            config = _config(args)
            scaling = read_json(args.scaling_plan, {}) if args.scaling_plan else {}
            event = record_plan(
                ledger_path(_state_root(config)),
                _run_file(args.run),
                args.ticker,
                trade_id=args.trade_id,
                predefined_scaling=scaling,
            )
            print("trade_id={}".format(event["trade_id"]))
            print("status={}".format(event["status"]))
            return 0
        if args.command == "record-event":
            config = _config(args)
            event = record_trade_event(
                ledger_path(_state_root(config)),
                args.trade_id,
                args.status,
                price=args.price,
                quantity=args.quantity,
                realized_pnl=args.realized_pnl,
                mfe=args.mfe,
                mae=args.mae,
                reason=args.reason,
                review_horizon_sessions=args.review_horizon_sessions,
            )
            print("trade_id={}".format(event["trade_id"]))
            print("status={}".format(event["status"]))
            return 0
        if args.command == "ledger-report":
            config = _config(args)
            path = ledger_path(_state_root(config))
            summary = ledger_summary(read_events(path))
            output = _state_root(config) / "ledger_summary.json"
            write_json(output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            print("summary={}".format(output))
            return 0
        if args.command == "review-open-trades":
            config = _config(args)
            events = read_events(ledger_path(_state_root(config)))
            open_names = sorted({state["ticker"] for state in trade_states(events) if state["status"] in {"FILLED","OPEN","REDUCED"}})
            if not open_names:
                review = review_open_trades(events, {"as_of":args.date,"candidates":[]})
                output = _state_root(config) / ("open_trade_review_{}.md".format(args.date))
                write_text(output, render_open_trade_review(review))
                print("open_trades=0")
                print("report={}".format(output))
                return 0
            current = run_scan(
                config,
                _require_token(),
                args.date,
                tickers=open_names,
                context_path=_context(args),
                offline=args.offline,
                refresh=args.refresh,
                max_requests=args.max_requests,
                portfolio_nav=args.portfolio_nav,
                posture="OPEN_TRADE_REVIEW_RESEARCH_ONLY",
                use_schwab=not args.no_schwab,
                schwab_env_path=args.schwab_env,
            )
            review = review_open_trades(events, current)
            run_dir = Path(current["artifacts"]["run_dir"])
            json_path = run_dir / "open_trade_review.json"
            report_path = run_dir / "open_trade_review.md"
            write_json(json_path, review)
            write_text(report_path, render_open_trade_review(review))
            print("open_trades={}".format(len(review["reviews"])))
            print("report={}".format(report_path))
            return 0
        if args.command == "backtest":
            config = _config(args)
            report = run_backtest(
                config,
                _require_token(),
                args.date,
                tickers=_tickers(args.tickers),
                split_date=args.split_date,
                offline=args.offline,
                refresh=args.refresh,
                max_requests=args.max_requests,
            )
            print("corat_backtest=ok")
            print("rows={}".format(len(report["rows"])))
            print("report={}".format(report["artifacts"]["report"]))
            return 0
        if args.command == "option-replay":
            config = _config(args)
            report = run_option_replay(
                config,
                _require_token(),
                args.ticker,
                args.setup,
                args.direction,
                args.start,
                args.end,
                args.split_date,
                holding_sessions=args.holding_sessions,
                max_signals=args.max_signals,
                offline=args.offline,
                refresh=args.refresh,
                max_requests=args.max_requests,
            )
            print("corat_option_replay=ok")
            print("signals={}".format(report["signals_found"]))
            print("completed={}".format(report["completed"]))
            print("promotion={}".format(report["production_promotion"]))
            print("report={}".format(report["artifacts"]["report"]))
            return 0
        if args.command == "full-replay":
            config = _config(args)
            names = _tickers(args.tickers)
            plan = build_replay_plan(
                config,
                args.start,
                args.end,
                args.train_end,
                args.validation_end,
                tickers=names,
                spacing_sessions=args.spacing_sessions,
                assumed_triggers_per_date=args.assumed_triggers_per_date,
                assumed_option_trades_per_date=args.assumed_option_trades_per_date,
                max_trades_per_date=args.max_trades_per_date,
                initial_nav=args.initial_nav,
                risk_pct=args.risk_pct,
                max_open_positions=args.max_open_positions,
                minimum_test_trades=args.minimum_test_trades,
            )
            if not args.execute:
                print("corat_full_replay=plan_only_not_started")
                print("orats_requests=0")
                print(json.dumps(plan, indent=2, sort_keys=True))
                return 0
            # Authorize before reading the token or creating the client. The
            # library runner repeats this guard for non-CLI callers.
            authorize_replay(
                plan,
                local_orats_usage(config),
                True,
                args.offline,
                args.request_budget,
                args.monthly_reserve,
                args.confirmed_remaining,
            )
            token = "OFFLINE_CACHE_ONLY" if args.offline else _require_token()
            report = run_full_replay(
                config,
                token,
                plan,
                execute=True,
                offline=args.offline,
                request_budget=args.request_budget,
                monthly_reserve=args.monthly_reserve,
                confirmed_remaining=args.confirmed_remaining,
                initial_nav=args.initial_nav,
                risk_pct=args.risk_pct,
                max_open_positions=args.max_open_positions,
                minimum_test_trades=args.minimum_test_trades,
            )
            print("corat_full_replay=completed_research_only")
            print("completed={}".format(report["completed"]))
            print("test_n={}".format((report["metrics"]["by_split"].get("TEST") or {}).get("n", 0)))
            print("production_promotion={}".format(report["production_promotion"]))
            print("orats_requests={}".format(report["orats_usage"]["run_requests"]))
            print("report={}".format(report["artifacts"]["report"]))
            return 0
        if args.command == "analyze":
            return _run_command(args, [str(args.ticker).upper()])
        if args.command == "full-scan":
            return _full_scan_command(args)
        if args.command == "run":
            return _run_command(args, _tickers(args.tickers))
        if args.command == "delta-scan":
            return _delta_command(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print("corat_error={}".format(exc), file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
