#!/usr/bin/env python3
"""Build a combined goal-readiness packet for the options recommendation stack.

This script is intentionally separate from the individual pipelines.  It reads
Options Pattern proof artifacts, runs deterministic V3/V4/Trade Desk guardrail
checks, and reports whether the active confidence goal can honestly be closed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codexuw.daily_v3 import parse_args as parse_v3_args
from codexuw.daily_v4 import (
    apply_v4_professional_dispositions,
    build_v4_opportunity_board,
    build_v4_swing_target_tickets,
    parse_args as parse_v4_args,
    run_v4_daily,
)
from codexuw.engine import assign_trade_statuses, select_final_trades
from codexuw.opportunity import build_opportunity_board, build_target_ticket_board
from codexuw.validation import select_systematic_date_folders
from uwos.trade_desk import build_recommendations


GOAL_MAJOR_REQUIRED_TICKERS = (
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOG",
    "GOOGL",
    "PLTR",
    "AMD",
    "MU",
    "META",
    "HOOD",
    "NOW",
)

ORDER_ENTRY_REQUIRED_FIELDS = (
    "ticker",
    "trade_legs",
    "expiration_date",
    "target_debit_credit",
    "max_risk_per_contract",
    "expected_R",
    "expected_R_per_day",
    "validation_profit_factor",
    "baselines_beaten_names",
    "risk_label",
)


@dataclass
class ProofRow:
    area: str
    status: str
    confidence_score: float
    evidence: str
    blocker: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined options recommendation goal-readiness evidence.")
    parser.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument(
        "--options-pattern-matrix-dir",
        default="",
        help="Directory containing confidence_summary.csv and portfolio_trade_rows.csv. Defaults to latest goal matrix.",
    )
    parser.add_argument("--out-dir", default="/tmp/options_recommendation_goal_acceptance")
    parser.add_argument("--as-of", default="", help="Optional yyyy-mm-dd upper bound for source-complete coverage.")
    parser.add_argument(
        "--codexdaily-proof-dir",
        default="",
        help="Optional checkpoint/resume CodexDaily historical proof directory. When present, V3/V4 coverage is read from here before root/out.",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return next(reader, {}) or {}
    except Exception:
        return {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _status_pass(status: str) -> bool:
    return status.upper() == "PASS"


def _status_is_complete(status: str) -> bool:
    return status.upper() in {"PASS", "WARN"}


def _latest_options_pattern_matrix(root: Path) -> Path:
    preferred = root / "out" / "options_pattern_pipeline_v1" / "goal_uncapped_current_v1"
    if (preferred / "confidence_summary.csv").exists():
        return preferred
    base = root / "out" / "options_pattern_pipeline_v1"
    candidates = [p for p in base.glob("*/confidence_summary.csv") if p.is_file()]
    if not candidates:
        return preferred
    return max((p.parent for p in candidates), key=lambda p: (p / "confidence_summary.csv").stat().st_mtime)


def _option_pattern_rows(matrix_dir: Path) -> list[ProofRow]:
    confidence = _read_first_csv_row(matrix_dir / "confidence_summary.csv")
    portfolio = _read_first_csv_row(matrix_dir / "portfolio_acceptance_summary.csv")
    if not confidence:
        return [
            ProofRow(
                "options_pattern_matrix",
                "FAIL",
                0.0,
                f"missing confidence_summary.csv under {matrix_dir}",
                "run uncapped Options Pattern matrix first",
            )
        ]

    coverage_ok = confidence.get("coverage_status") == "EXACT_FULL_SCOPE"
    portfolio_ok = confidence.get("portfolio_status") == "PASS"
    gate_fail = int(_to_float(confidence.get("gate_fail_trade_count"), 999999))
    trade_count = int(_to_float(confidence.get("trade_count"), 0))
    avg_r = _to_float(confidence.get("avg_expected_R"), -999)
    score = _to_float(confidence.get("overall_confidence_score"), 0.0)
    blockers = str(confidence.get("blockers") or "").strip()
    scope_only_blocker = blockers in {"", "n/a", "codexdaily_v3_v4_and_trade_desk_not_in_this_matrix_scope"}
    missing_required = str(confidence.get("missing_required_ticker_examples") or "").strip().lower()
    required_ok = missing_required in {"", "n/a", "none"}
    ok = coverage_ok and portfolio_ok and gate_fail == 0 and trade_count > 0 and avg_r > 0 and score >= 7.0 and scope_only_blocker and required_ok
    evidence = (
        f"coverage={confidence.get('coverage_status')}; portfolio={confidence.get('portfolio_status')}; "
        f"trades={trade_count}; auto={confidence.get('auto_trade_count')}; "
        f"target_ready={confidence.get('target_ready_trade_count')}; avg_R={confidence.get('avg_expected_R')}; "
        f"score={confidence.get('overall_confidence_score')}; blockers={blockers or 'n/a'}"
    )
    rows = [
        ProofRow(
            "options_pattern_matrix",
            "PASS" if ok else "FAIL",
            min(10.0, max(0.0, score)),
            evidence,
            "" if ok else "Options Pattern matrix did not meet exact full-scope positive-expectancy gate.",
        )
    ]

    rows.append(_option_pattern_order_entry_row(matrix_dir, portfolio))
    return rows


def _option_pattern_order_entry_row(matrix_dir: Path, portfolio: dict[str, str]) -> ProofRow:
    matrix_rows = _read_csv_rows(matrix_dir / "goal_acceptance_matrix.csv")
    generated_rows: list[dict[str, Any]] = []
    if matrix_rows:
        try:
            from scripts import options_pattern_goal_matrix as matrix

            generated_rows = matrix.build_portfolio_trade_rows(matrix_rows)
        except Exception:
            generated_rows = []
    path = matrix_dir / "portfolio_trade_rows.csv"
    if not generated_rows and not path.exists():
        return ProofRow("options_pattern_order_entry", "FAIL", 0.0, f"missing {path}", "missing portfolio trade rows")
    checked = 0
    failures: list[str] = []
    tickers: set[str] = set()
    source_rows: list[Mapping[str, Any]]
    source_label = "current_builder_from_goal_matrix" if generated_rows else "portfolio_trade_rows_csv"
    source_rows = generated_rows if generated_rows else _read_csv_rows(path)
    for row in source_rows:
        checked += 1
        ticker = str(row.get("ticker") or "").upper().strip()
        if ticker:
            tickers.add(ticker)
        missing = [field for field in ORDER_ENTRY_REQUIRED_FIELDS if not str(row.get(field) or "").strip()]
        legs = str(row.get("trade_legs") or "")
        if "Buy " not in legs and "Sell " not in legs:
            missing.append("plain_language_buy_sell_legs")
        if str(row.get("portfolio_gate_status") or "").upper() == "FAIL":
            missing.append("portfolio_gate_status")
        if missing and len(failures) < 5:
            failures.append(f"{ticker or checked}:{','.join(missing)}")
    major_seen = sorted(set(GOAL_MAJOR_REQUIRED_TICKERS) & tickers)
    ok = checked > 0 and not failures
    evidence = (
        f"source={source_label}; rows_checked={checked}; gate_pass={portfolio.get('gate_pass_trade_count')}; "
        f"gate_fail={portfolio.get('gate_fail_trade_count')}; major_tickers_seen={','.join(major_seen) or 'n/a'}"
    )
    return ProofRow(
        "options_pattern_order_entry",
        "PASS" if ok else "FAIL",
        10.0 if ok else 4.0,
        evidence,
        "" if ok else "; ".join(failures),
    )


def _credit_candidate(**overrides: Any) -> dict[str, Any]:
    row = {
        "ticker": "AAA",
        "sector": "Technology",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "strategy_kind": "Credit",
        "expiry": dt.date(2026, 5, 29),
        "dte": 10,
        "hard_rejects": "",
        "penalties": "",
        "credit": 1.30,
        "mid_credit": 1.30,
        "natural_credit": 1.20,
        "required_entry": 1.20,
        "target_entry": 1.20,
        "credit_pct_width": 0.26,
        "spread_width": 5.0,
        "max_profit": 130.0,
        "max_loss": 100.0,
        "breakeven": 98.7,
        "distance_pct": 0.05,
        "expected_move_ratio": 0.85,
        "iv30d": 0.24,
        "combined_flow_bias": 0.12,
        "score": 8.0,
        "confidence": "High",
        "live_status": "PASS",
        "quote_width_pct": 0.08,
        "short_oi": 1000,
        "short_volume": 500,
        "long_oi": 1000,
        "long_volume": 500,
        "short_strike": 100.0,
        "long_strike": 95.0,
        "short_leg": "AAA260529P00100000",
        "long_leg": "AAA260529P00095000",
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "replay_ev_verdict": "acceptable",
        "edge_verdict": "acceptable",
        "edge_sample_size": 10,
        "edge_win_rate": 0.65,
        "edge_avg_pnl": 45.0,
        "confirmation_score": 8.0,
        "catalyst_status": "supportive",
    }
    row.update(overrides)
    return row


def _v3_functional_row() -> ProofRow:
    try:
        default_args = parse_v3_args(["run", "--date", "2026-06-09"])
        scored = pd.DataFrame([
            _credit_candidate(ticker="AAA", score=8.0),
            _credit_candidate(ticker="BBB", score=7.8),
        ])
        executable = assign_trade_statuses(scored)
        final = select_final_trades(
            executable,
            regime={"sizing_stance": "normal"},
            risk_budget=5_000,
            recent_performance={"status": "unavailable"},
            max_final_trades=0,
            risk_config={"risk_mandate": "target-growth", "max_contracts_per_trade": 1},
        )
        board = build_opportunity_board(
            scored=executable,
            final=final,
            watchlist=pd.DataFrame(),
            portfolio={"status": "ok", "cash": 25_000},
        )
        tickets = build_target_ticket_board(board)
        negative = assign_trade_statuses(
            pd.DataFrame(
                [
                    _credit_candidate(
                        ticker="NOW",
                        credit=1.67,
                        required_entry=1.40,
                        target_entry=1.40,
                        replay_ev_verdict="acceptable_secondary_income",
                        edge_verdict="acceptable_secondary_income",
                        edge_sample_size=17,
                        edge_win_rate=0.5294117647,
                        edge_avg_pnl=-33.897,
                        confirmation_score=9.0,
                    )
                ]
            )
        )
        discovery_ok = int(default_args.max_tickers) == 0 and int(default_args.max_candidates) == 0
        visible_ok = int(default_args.max_final_trades) == 0 and set(final["ticker"]) == {"AAA", "BBB"}
        target_ok = set(final["ticker"].astype(str).str.upper()) <= set(tickets["Ticker"].astype(str).str.upper())
        negative_status = str(negative.iloc[0].get("trade_status") or "")
        negative_ok = negative_status not in {"Execute", "Watch"} and "negative_edge_avg_pnl" in str(negative.iloc[0].get("trade_status_reason") or "")
        ok = discovery_ok and visible_ok and target_ok and negative_ok
        evidence = (
            f"default_max_tickers={default_args.max_tickers}; default_max_candidates={default_args.max_candidates}; "
            f"default_max_final_trades={default_args.max_final_trades}; selected={','.join(final['ticker'].astype(str))}; "
            f"target_rows={len(tickets)}; NOW_negative_status={negative_status}"
        )
        return ProofRow("codexdaily_v3_functional_gates", "PASS" if ok else "FAIL", 8.0 if ok else 4.0, evidence, "" if ok else "V3 discovery, visibility, or negative-edge gate failed")
    except Exception as exc:
        return ProofRow("codexdaily_v3_functional_gates", "FAIL", 0.0, f"exception={exc}", "V3 functional check crashed")


def _v4_functional_row() -> ProofRow:
    try:
        args = parse_v4_args(["run", "--date", "2026-06-09", "--max-final-trades", "1"])
        no_v3_wrapper = "run_v3_daily" not in run_v4_daily.__code__.co_names
        scored = pd.DataFrame(
            [
                _credit_candidate(
                    ticker="NOW",
                    credit=1.67,
                    mid_credit=1.67,
                    natural_credit=1.60,
                    required_entry=1.40,
                    target_entry=1.40,
                    replay_ev_verdict="acceptable_secondary_income",
                    edge_verdict="thin_sample",
                    edge_sample_size=17,
                    edge_win_rate=0.5294117647,
                    edge_avg_pnl=-33.897,
                    confirmation_score=9.0,
                    score=7.8,
                )
            ]
        )
        adjusted = apply_v4_professional_dispositions(scored)
        board = build_v4_opportunity_board(adjusted, top_flow=pd.DataFrame())
        tickets = build_v4_swing_target_tickets(
            scored=adjusted,
            board=pd.DataFrame(),
            regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
            top_flow=pd.DataFrame([{"rank": 1, "ticker": "NOW", "net_premium": 2_000_000, "flow_direction": "bullish"}]),
        )
        now_statuses = board[board["Ticker"].eq("NOW")]["Status"].astype(str)
        negative_ok = adjusted.iloc[0]["trade_status"] == "Avoid" and tickets.empty and not now_statuses.str.contains("WORK LIMIT|ENTER|SCOUT", regex=True).any()
        discovery_ok = int(args.max_tickers) == 0 and int(args.max_candidates) == 0
        ok = discovery_ok and no_v3_wrapper and negative_ok
        evidence = (
            f"default_max_tickers={args.max_tickers}; default_max_candidates={args.max_candidates}; "
            f"max_final_trades_arg={args.max_final_trades}; no_v3_wrapper={no_v3_wrapper}; "
            f"NOW_status={adjusted.iloc[0]['trade_status']}; target_tickets={len(tickets)}"
        )
        return ProofRow("codexdaily_v4_functional_gates", "PASS" if ok else "FAIL", 8.0 if ok else 4.0, evidence, "" if ok else "V4 discovery, independence, or negative-edge gate failed")
    except Exception as exc:
        return ProofRow("codexdaily_v4_functional_gates", "FAIL", 0.0, f"exception={exc}", "V4 functional check crashed")


def _trade_desk_row() -> ProofRow:
    result = {
        "as_of": "2026-06-03T16:00:00Z",
        "positions": [
            {
                "symbol": "NOW   260717P00120000",
                "underlying": "NOW",
                "asset_type": "OPTION",
                "put_call": "PUT",
                "strike": 120.0,
                "expiry": "2026-07-17",
                "qty": -1,
                "avg_cost": 6.31,
                "entry_date": "2026-06-01",
                "live_quote": {"ask": 7.70, "bid": 7.50},
                "greeks": {"delta": -0.49},
                "underlying_quote": {"last": 120.37},
                "computed": {"dte": 44, "unrealized_pnl": -250.0, "theta_pnl_per_day": 11.0},
            },
            {
                "symbol": "NOW   260717P00110000",
                "underlying": "NOW",
                "asset_type": "OPTION",
                "put_call": "PUT",
                "strike": 110.0,
                "expiry": "2026-07-17",
                "qty": 1,
                "avg_cost": 3.31,
                "entry_date": "2026-06-01",
                "live_quote": {"ask": 3.05, "bid": 2.85},
                "greeks": {"delta": -0.34},
                "underlying_quote": {"last": 120.37},
                "computed": {"dte": 44, "unrealized_pnl": 70.0, "theta_pnl_per_day": -5.0},
            },
        ],
    }
    try:
        rows = build_recommendations(result)
    except Exception as exc:
        return ProofRow("trade_desk_management", "FAIL", 0.0, f"exception={exc}", "Trade Desk check crashed")
    if not rows:
        return ProofRow("trade_desk_management", "FAIL", 0.0, "no recommendation rows", "NOW-like spread was not reviewed")
    row = rows[0]
    action = str(row.get("action") or "")
    position = str(row.get("position") or "")
    guidance = str(row.get("order_guidance") or "")
    ok = action in {"ROLL", "CLOSE"} and "120" in position and "110" in position and "close/roll debit" in guidance
    evidence = f"action={action}; category={row.get('category')}; order_guidance={guidance}; position={position}"
    return ProofRow("trade_desk_management", "PASS" if ok else "FAIL", 8.0 if ok else 4.0, evidence, "" if ok else "Trade Desk did not force NOW-like spread to roll/close as one spread")


def _source_dates(root: Path, as_of: dt.date | None) -> list[str]:
    folders = select_systematic_date_folders(root, as_of=as_of, latest_n=10000)
    dates = []
    for folder in folders:
        try:
            dates.append(folder.name[:10])
        except Exception:
            continue
    return sorted(set(dates))


def _codexdaily_manifest_path(root: Path, pipeline: str, day: str, proof_dir: Path | None = None) -> tuple[Path, str]:
    prefix = f"codexdaily_{pipeline}"
    filename = f"{prefix}_manifest_{day}.json"
    candidates: list[tuple[Path, str]] = []
    if proof_dir is not None:
        candidates.extend(
            [
                (proof_dir / pipeline / f"{prefix}_{day}" / filename, "proof_dir"),
                (proof_dir / f"{prefix}_{day}" / filename, "proof_dir"),
                (proof_dir / pipeline / day / filename, "proof_dir"),
            ]
        )
    candidates.append((root / "out" / f"{prefix}_{day}" / filename, "root_out"))
    for path, source in candidates:
        if path.exists():
            return path, source
    return candidates[0] if candidates else (root / "out" / f"{prefix}_{day}" / filename, "root_out")


def _flag_int(tokens: list[str], flag: str) -> int | None:
    if flag not in tokens:
        return None
    idx = tokens.index(flag)
    if idx + 1 >= len(tokens):
        return None
    try:
        return int(float(tokens[idx + 1]))
    except ValueError:
        return None


def _legacy_proof_scope_from_commands(rows: list[dict[str, Any]]) -> tuple[str, str]:
    caps: set[str] = set()
    saw_command = False
    for row in rows:
        command = str(row.get("command") or "").strip()
        if not command:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        saw_command = True
        for flag, label in [
            ("--bot-max-rows", "bot_max_rows"),
            ("--max-tickers", "max_tickers"),
            ("--max-candidates", "max_candidates"),
        ]:
            value = _flag_int(tokens, flag)
            if value is not None and value > 0:
                caps.add(f"{label}={value}")
    if caps:
        return "CAPPED", ",".join(sorted(caps))
    if saw_command:
        return "FULL", "legacy checkpoint commands show uncapped discovery/candidate settings"
    return "UNKNOWN", "no proof-scope metadata or command rows found"


def _codexdaily_proof_scope(proof_dir: Path | None) -> tuple[str, str]:
    if proof_dir is None:
        return "", ""
    checkpoint = _read_json(proof_dir / "codexdaily_historical_proof_checkpoint.json")
    if checkpoint:
        status = str(checkpoint.get("proof_scope_status") or "").upper()
        note = str(checkpoint.get("proof_scope_notes") or "")
        if status:
            return status, note
        rows = checkpoint.get("rows")
        if isinstance(rows, list):
            return _legacy_proof_scope_from_commands(rows)
    summary_rows = _read_csv_rows(proof_dir / "codexdaily_historical_proof_summary.csv")
    if summary_rows:
        statuses = {str(row.get("proof_scope_status") or "").upper() for row in summary_rows if row.get("proof_scope_status")}
        notes = sorted({str(row.get("proof_scope_note") or "") for row in summary_rows if row.get("proof_scope_note")})
        if statuses:
            if statuses == {"FULL"}:
                return "FULL", notes[0] if notes else ""
            return "CAPPED", ";".join(notes) or ",".join(sorted(statuses))
        return _legacy_proof_scope_from_commands(summary_rows)
    return "UNKNOWN", "proof directory has no checkpoint/summary scope metadata"


def _manifest_coverage_row(root: Path, pipeline: str, dates: list[str], proof_dir: Path | None = None) -> ProofRow:
    existing = []
    missing = []
    missing_policy = []
    capped = []
    source_counts: dict[str, int] = {}
    proof_scope_status, proof_scope_note = _codexdaily_proof_scope(proof_dir)
    for day in dates:
        manifest_path, source = _codexdaily_manifest_path(root, pipeline, day, proof_dir)
        manifest = _read_json(manifest_path)
        if not manifest:
            missing.append(day)
            continue
        existing.append(day)
        source_counts[source] = source_counts.get(source, 0) + 1
        policy = manifest.get("visible_signal_policy") or {}
        if not policy:
            missing_policy.append(day)
        elif policy.get("active_execute_cap") not in {None, ""}:
            capped.append(day)
    if not dates:
        return ProofRow(f"codexdaily_{pipeline}_historical_coverage", "FAIL", 0.0, "source_complete_dates=0", "no source-complete dates found")
    proof_scope_problem = proof_scope_status not in {"", "FULL"}
    if not missing and not missing_policy and not capped and not proof_scope_problem:
        status = "PASS"
        score = 8.0
        blocker = ""
    elif existing:
        status = "PARTIAL"
        score = 6.5
        blocker = "rerun current-code CodexDaily historical proof uncapped" if proof_scope_problem else "regenerate current-code manifests for missing/stale dates"
    else:
        status = "FAIL"
        score = 2.0
        blocker = "no current-code historical manifests found"
    evidence = (
        f"source_complete_dates={len(dates)}; manifests={len(existing)}; missing={len(missing)}; "
        f"missing_visible_policy={len(missing_policy)}; capped={len(capped)}; "
        f"proof_scope={proof_scope_status or 'n/a'}; proof_scope_note={proof_scope_note or 'n/a'}; "
        f"manifest_sources={','.join(f'{key}:{value}' for key, value in sorted(source_counts.items())) or 'n/a'}; "
        f"missing_examples={','.join(missing[:5]) or 'n/a'}; stale_policy_examples={','.join(missing_policy[:5]) or 'n/a'}"
    )
    return ProofRow(f"codexdaily_{pipeline}_historical_coverage", status, score, evidence, blocker)


def _combined_row(rows: list[ProofRow]) -> ProofRow:
    hard_failures = [row for row in rows if row.status == "FAIL"]
    partials = [row for row in rows if row.status == "PARTIAL"]
    complete_rows = [row for row in rows if _status_is_complete(row.status)]
    functional_rows = [
        row for row in rows if row.area in {"options_pattern_matrix", "options_pattern_order_entry", "codexdaily_v3_functional_gates", "codexdaily_v4_functional_gates", "trade_desk_management"}
    ]
    functional_score = min((row.confidence_score for row in functional_rows), default=0.0)
    if hard_failures:
        status = "FAIL"
        score = min(6.0, functional_score)
    elif partials:
        status = "PARTIAL"
        score = min(6.8, functional_score)
    else:
        status = "PASS"
        score = min((row.confidence_score for row in complete_rows), default=8.0)
    blockers = [row.area for row in hard_failures + partials]
    evidence = f"functional_confidence_score={functional_score:.1f}; complete_rows={len(complete_rows)}/{len(rows)}; can_mark_goal_complete={'yes' if status == 'PASS' else 'no'}"
    return ProofRow("combined_goal_verdict", status, round(score, 2), evidence, ",".join(blockers))


def _write_outputs(out_dir: Path, rows: list[ProofRow]) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "goal_acceptance_summary.csv"
    json_path = out_dir / "goal_acceptance_summary.json"
    md_path = out_dir / "goal_acceptance_report.md"
    fieldnames = ["area", "status", "confidence_score", "evidence", "blocker"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    combined = next((row for row in rows if row.area == "combined_goal_verdict"), rows[-1])
    lines = [
        "# Options Recommendation Goal Acceptance",
        "",
        "## Verdict",
        "",
        f"- status: {combined.status}",
        f"- confidence_score: {combined.confidence_score}",
        f"- can_mark_goal_complete: {'yes' if combined.status == 'PASS' else 'no'}",
        f"- blockers: {combined.blocker or 'n/a'}",
        "",
        "## Proof Rows",
        "",
        "| Area | Status | Score | Evidence | Blocker |",
        "|:--|:--|--:|:--|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row.area} | {row.status} | {row.confidence_score:.2f} | "
            f"{_md_cell(row.evidence)} | {_md_cell(row.blocker or 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PASS means the combined goal can be closed. PARTIAL means the functional gates pass but current-code historical coverage is incomplete. FAIL means at least one functional gate is broken.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def _md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:900]


def build_rows(root: Path, matrix_dir: Path, as_of: dt.date | None = None, codexdaily_proof_dir: Path | None = None) -> list[ProofRow]:
    rows: list[ProofRow] = []
    rows.extend(_option_pattern_rows(matrix_dir))
    rows.append(_v3_functional_row())
    rows.append(_v4_functional_row())
    rows.append(_trade_desk_row())
    dates = _source_dates(root, as_of)
    rows.append(_manifest_coverage_row(root, "v3", dates, codexdaily_proof_dir))
    rows.append(_manifest_coverage_row(root, "v4", dates, codexdaily_proof_dir))
    rows.append(_combined_row(rows))
    return rows


def _parse_date(value: str) -> dt.date | None:
    if not value:
        return None
    return dt.date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    matrix_dir = Path(args.options_pattern_matrix_dir).expanduser().resolve() if args.options_pattern_matrix_dir else _latest_options_pattern_matrix(root)
    out_dir = Path(args.out_dir).expanduser().resolve()
    codexdaily_proof_dir = Path(args.codexdaily_proof_dir).expanduser().resolve() if args.codexdaily_proof_dir else None
    rows = build_rows(root, matrix_dir, _parse_date(args.as_of), codexdaily_proof_dir)
    csv_path, json_path, md_path = _write_outputs(out_dir, rows)
    combined = next(row for row in rows if row.area == "combined_goal_verdict")
    print(md_path)
    print(f"status={combined.status} confidence_score={combined.confidence_score} can_mark_goal_complete={'yes' if combined.status == 'PASS' else 'no'}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    return 0 if combined.status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
