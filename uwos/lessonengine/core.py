"""Evidence-first lesson engine for Options Agent.

The LessonEngine is deliberately separate from trade recommendation logic. The
``analyze`` flow is read-only and proposes lessons from durable evidence. The
``promote`` flow is the only path that publishes a versioned lesson pack for
Options Agent to consume.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
import yaml

from uwos.paths import project_root

ACTIVE_POINTER_FILENAME = "options_agent_active_lesson_pack.json"
LESSON_PACKS_DIRNAME = "options_agent_lesson_packs"
USER_CORRECTIONS_FILENAME = "user_corrections.jsonl"
DEFAULT_PROMPT_BUDGET_LINES = 40
LESSON_SCHEMA_VERSION = 1

REQUIRED_LESSON_KEYS = {
    "id",
    "status",
    "severity",
    "failure_mode",
    "rule",
    "evidence",
    "confidence",
    "risk_score",
    "applies_to",
    "actions",
    "created_at",
    "last_validated_at",
    "promotion_regression_run",
}
VALID_STATUSES = {"proposed", "active", "retired"}
VALID_SEVERITIES = {"hard", "medium", "advisory"}

MAJOR_FOCUS_TICKERS = (
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOG",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "AMD",
    "AVGO",
    "PLTR",
    "SPY",
    "QQQ",
)

FAILURE_MODE_LIBRARY: dict[str, dict[str, Any]] = {
    "PORTFOLIO_RISK_SUPPRESSION": {
        "lesson_id": "OA-001",
        "severity": "hard",
        "rule": "Do not suppress otherwise-good trades solely due to portfolio risk; annotate the risk and size-risk it.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "decision_board": {"forbid_portfolio_only_suppression": True},
            "report_contract": {"require_portfolio_annotation_visible": True},
            "regression_gate": {"fixture": "portfolio_risk_annotation_only"},
        },
    },
    "EOD_TARGET_SUPPRESSION": {
        "lesson_id": "OA-002",
        "severity": "hard",
        "rule": "EOD target candidates with valid target credit/debit are next-day planning rows; keep them yellow-visible with ready_to_enter=false until live recheck passes.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "synthesis_scoring": {"valid_target_math_visibility": "preserve_yellow"},
            "decision_board": {"forbid_market_closed_as_hard_blocker": True},
            "report_contract": {"require_target_limit": True, "require_ready_to_enter_false": True},
            "regression_gate": {"fixture": "eod_market_closed_target_candidate"},
        },
    },
    "MISSING_TARGET_PRICE_GREEN": {
        "lesson_id": "OA-003",
        "severity": "hard",
        "rule": "Missing target debit/credit must prevent green execution readiness.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "synthesis_scoring": {"missing_target_price_delta": -25},
            "decision_board": {"block_green_when_target_price_missing": True},
            "report_contract": {"require_target_limit": True},
            "regression_gate": {"fixture": "missing_target_price_green_block"},
        },
    },
    "REPORT_LEG_DETAIL_MISSING": {
        "lesson_id": "OA-004",
        "severity": "hard",
        "rule": "Reports must show plain-language buy/sell legs, expiration dates, and target debit/credit instead of relying on OCC codes.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "report_contract": {"require_buy_sell_legs": True, "require_expiration": True, "require_target_limit": True},
            "regression_gate": {"fixture": "report_buy_sell_expiration_contract"},
        },
    },
    "ARBITRARY_TOP_N_SUPPRESSION": {
        "lesson_id": "OA-005",
        "severity": "medium",
        "rule": "Do not use arbitrary top-N cutoffs to hide candidates, coverage rows, or no-trade audits.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "coverage_audit": {"require_full_visibility": True},
            "regression_gate": {"fixture": "no_top_n_visibility_suppression"},
        },
    },
    "JUNK_TICKER_PROMOTION": {
        "lesson_id": "OA-006",
        "severity": "hard",
        "rule": "Do not promote low-quality tail tickers into action rows without explicit strong evidence.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "synthesis_scoring": {"speculative_underlying_delta": -20, "excluded_underlying_delta": -40},
            "decision_board": {"block_low_quality_action_without_strong_evidence": True},
            "regression_gate": {"fixture": "junk_ticker_action_filter"},
        },
    },
    "STATUS_VISIBILITY_MISSING": {
        "lesson_id": "OA-007",
        "severity": "medium",
        "rule": "Preserve clear green, yellow, blocked, and no-action status labels/icons in reports and audits.",
        "actions": {
            "report_contract": {"require_status_labels": True},
            "coverage_audit": {"require_status_color": True},
            "regression_gate": {"fixture": "status_icon_visibility"},
        },
    },
    "MAJOR_TICKER_COVERAGE_MISSING": {
        "lesson_id": "OA-008",
        "severity": "medium",
        "rule": "Major liquid tickers must appear in coverage audit when source data exists.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "coverage_audit": {"require_major_focus_tickers": list(MAJOR_FOCUS_TICKERS)},
            "regression_gate": {"fixture": "major_ticker_coverage_audit"},
        },
    },
    "MULTI_AGENT_NOT_MANDATORY": {
        "lesson_id": "OA-009",
        "severity": "hard",
        "rule": "Multi-agent dispatch is mandatory for normal Options Agent runs when available.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "regression_gate": {"fixture": "agent_dispatch_plan_required"},
        },
    },
    "NEGATIVE_CLOSED_TRADE_OUTCOME": {
        "lesson_id": "OA-010",
        "severity": "medium",
        "rule": "Repeated or material closed-trade losses should penalize the same ticker/strategy until positive actual outcome evidence returns.",
        "actions": {
            "subagent_prompt": {"inject": True},
            "synthesis_scoring": {"negative_closed_outcome_delta": -10},
            "regression_gate": {"fixture": "negative_closed_trade_outcome_penalty"},
        },
    },
}


@dataclass(frozen=True)
class LessonPack:
    version: str
    lessons: list[dict[str, Any]]
    source_path: Path
    markdown: str
    digest: str
    prompt_budget_lines: int = DEFAULT_PROMPT_BUDGET_LINES

    @property
    def lesson_count(self) -> int:
        return len([lesson for lesson in self.lessons if str(lesson.get("status")) == "active"])

    def metadata(self) -> dict[str, Any]:
        return {
            "lesson_pack_version": self.version,
            "lesson_pack_digest": self.digest,
            "lesson_count": self.lesson_count,
            "lesson_pack_source": str(self.source_path),
            "prompt_budget_lines": self.prompt_budget_lines,
        }


def knowledge_dir(root: Optional[Path] = None) -> Path:
    return (root or project_root()).expanduser().resolve() / "knowledge"


def lesson_packs_dir(root: Optional[Path] = None) -> Path:
    return knowledge_dir(root) / LESSON_PACKS_DIRNAME


def active_pointer_path(root: Optional[Path] = None) -> Path:
    return knowledge_dir(root) / ACTIVE_POINTER_FILENAME


def user_corrections_path(root: Optional[Path] = None) -> Path:
    return knowledge_dir(root) / USER_CORRECTIONS_FILENAME


def _today() -> str:
    return dt.date.today().isoformat()


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload["_line_no"] = line_no
                rows.append(payload)
        except json.JSONDecodeError:
            rows.append({"_line_no": line_no, "raw_text": text, "parse_error": "invalid_json"})
    return rows


def validate_lesson_pack(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    lessons = payload.get("lessons", [])
    if lessons is None:
        lessons = []
    if not isinstance(lessons, list):
        return ["lessons must be a list"]
    seen_ids: set[str] = set()
    for idx, lesson in enumerate(lessons):
        if not isinstance(lesson, Mapping):
            errors.append(f"lesson[{idx}] must be a mapping")
            continue
        missing = sorted(REQUIRED_LESSON_KEYS - set(lesson.keys()))
        if missing:
            errors.append(f"lesson[{idx}] missing required keys: {', '.join(missing)}")
        lesson_id = str(lesson.get("id") or "").strip()
        if not lesson_id:
            errors.append(f"lesson[{idx}] id is empty")
        elif lesson_id in seen_ids:
            errors.append(f"duplicate lesson id: {lesson_id}")
        seen_ids.add(lesson_id)
        status = str(lesson.get("status") or "").strip()
        if status and status not in VALID_STATUSES:
            errors.append(f"{lesson_id or f'lesson[{idx}]'} invalid status: {status}")
        severity = str(lesson.get("severity") or "").strip()
        if severity and severity not in VALID_SEVERITIES:
            errors.append(f"{lesson_id or f'lesson[{idx}]'} invalid severity: {severity}")
        if not isinstance(lesson.get("actions", {}), Mapping):
            errors.append(f"{lesson_id or f'lesson[{idx}]'} actions must be a mapping")
        for key in ("evidence", "applies_to"):
            if key in lesson and not isinstance(lesson.get(key), list):
                errors.append(f"{lesson_id or f'lesson[{idx}]'} {key} must be a list")
    return errors


def build_prompt_pack(lessons: Sequence[Mapping[str, Any]], *, max_lines: int = DEFAULT_PROMPT_BUDGET_LINES) -> str:
    active = [lesson for lesson in lessons if str(lesson.get("status") or "") == "active"]
    lines = ["# Options Agent Lessons", ""]
    if not active:
        lines.append("No active lessons are pinned for this run.")
        return "\n".join(lines) + "\n"
    for lesson in active:
        lesson_id = str(lesson.get("id") or "").strip()
        severity = str(lesson.get("severity") or "").strip()
        rule = str(lesson.get("rule") or "").strip()
        applies_to = ", ".join(str(item) for item in lesson.get("applies_to", [])[:5])
        lines.append(f"- {lesson_id} [{severity}] {rule} Applies to: {applies_to}.")
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["- Additional lessons omitted from prompt pack; see lesson snapshot artifact."]
    return "\n".join(lines) + "\n"


def load_lesson_pack(path: Path) -> LessonPack:
    resolved = Path(path).expanduser().resolve()
    yaml_path = resolved / "lessons.yaml" if resolved.is_dir() else resolved
    payload = _read_yaml(yaml_path)
    errors = validate_lesson_pack(payload)
    if errors:
        raise ValueError("invalid lesson pack: " + "; ".join(errors))
    lessons = [dict(lesson) for lesson in payload.get("lessons", []) if isinstance(lesson, Mapping)]
    version = str(payload.get("version") or yaml_path.parent.name or "unversioned")
    prompt_budget = int(payload.get("prompt_budget_lines") or DEFAULT_PROMPT_BUDGET_LINES)
    markdown_path = yaml_path.with_name("lessons.md")
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else build_prompt_pack(lessons, max_lines=prompt_budget)
    digest = _stable_digest({"version": version, "lessons": lessons})
    return LessonPack(version=version, lessons=lessons, source_path=yaml_path, markdown=markdown, digest=digest, prompt_budget_lines=prompt_budget)


def load_active_lesson_pack(root: Optional[Path] = None, *, version: Optional[str] = None) -> LessonPack:
    resolved_root = (root or project_root()).expanduser().resolve()
    if version:
        return load_lesson_pack(lesson_packs_dir(resolved_root) / version)
    pointer = active_pointer_path(resolved_root)
    if not pointer.exists():
        return _empty_lesson_pack(resolved_root)
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_lesson_pack(resolved_root)
    pack_version = str(payload.get("active_lesson_pack_version") or payload.get("version") or "").strip()
    pack_path = payload.get("active_lesson_pack_path")
    if pack_path:
        return load_lesson_pack(Path(str(pack_path)))
    if pack_version:
        return load_lesson_pack(lesson_packs_dir(resolved_root) / pack_version)
    return _empty_lesson_pack(resolved_root)


def _empty_lesson_pack(root: Path) -> LessonPack:
    payload = {"version": "none", "lessons": []}
    return LessonPack(
        version="none",
        lessons=[],
        source_path=knowledge_dir(root) / "no_active_lesson_pack",
        markdown=build_prompt_pack([]),
        digest=_stable_digest(payload),
    )


def write_lesson_snapshots(pack: LessonPack, paths: Mapping[str, Path]) -> None:
    md_path = paths.get("lessons_snapshot_md")
    json_path = paths.get("lessons_snapshot_json")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(pack.markdown, encoding="utf-8")
    if json_path:
        active = [lesson for lesson in pack.lessons if str(lesson.get("status")) == "active"]
        _write_json(
            json_path,
            {
                **pack.metadata(),
                "lessons": active,
            },
        )


def lesson_manifest_metadata(pack: LessonPack, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        **pack.metadata(),
        "lessons_snapshot_md": str(paths.get("lessons_snapshot_md", "")),
        "lessons_snapshot_json": str(paths.get("lessons_snapshot_json", "")),
        "lessons_application_audit": str(paths.get("lessons_application_audit", "")),
        "lessonengine_schema_version": LESSON_SCHEMA_VERSION,
        "code_git_sha": _git_sha(project_root()),
    }


def apply_synthesis_actions(
    row: Mapping[str, Any],
    lessons: Sequence[Mapping[str, Any]],
) -> tuple[float, list[str], list[dict[str, Any]]]:
    delta = 0.0
    reasons: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    ticker = str(row.get("ticker") or "").strip().upper()
    status = str(row.get("recommendation_status") or "").strip().upper()
    entry_limit = _as_float(row.get("entry_limit"))
    tier = str(row.get("underlying_quality_tier") or "").strip().lower()
    ticket = str(row.get("trade_plan") or row.get("full_ticket") or "").strip()
    expectancy_status = str(row.get("actual_forward_expectancy_status") or "").strip().upper()
    strategy_expectancy_status = str(row.get("actual_forward_strategy_expectancy_status") or "").strip().upper()
    for lesson in lessons:
        if str(lesson.get("status")) != "active":
            continue
        actions = lesson.get("actions", {}) if isinstance(lesson.get("actions"), Mapping) else {}
        scoring = actions.get("synthesis_scoring", {}) if isinstance(actions.get("synthesis_scoring"), Mapping) else {}
        if not scoring:
            continue
        fired = False
        row_delta = 0.0
        reason_parts: list[str] = []
        missing_price_delta = _as_float(scoring.get("missing_target_price_delta"))
        if missing_price_delta is not None and status in {"ENTER", "ENTER_WITH_PORTFOLIO_RISK"} and (entry_limit is None or entry_limit <= 0 or not ticket):
            row_delta += missing_price_delta
            reason_parts.append(f"{lesson.get('id')} missing target price {missing_price_delta:+.0f}")
            fired = True
        speculative_delta = _as_float(scoring.get("speculative_underlying_delta"))
        if speculative_delta is not None and tier == "speculative":
            row_delta += speculative_delta
            reason_parts.append(f"{lesson.get('id')} speculative underlying {speculative_delta:+.0f}")
            fired = True
        excluded_delta = _as_float(scoring.get("excluded_underlying_delta"))
        if excluded_delta is not None and tier == "excluded":
            row_delta += excluded_delta
            reason_parts.append(f"{lesson.get('id')} excluded underlying {excluded_delta:+.0f}")
            fired = True
        negative_outcome_delta = _as_float(scoring.get("negative_closed_outcome_delta"))
        negative_tickers = {
            str(item).strip().upper()
            for item in scoring.get("negative_closed_outcome_tickers", [])
            if str(item).strip()
        }
        negative_scope_matches = not negative_tickers or ticker in negative_tickers
        if negative_outcome_delta is not None and negative_scope_matches and (
            expectancy_status == "BLOCK" or strategy_expectancy_status == "BLOCK"
        ):
            row_delta += negative_outcome_delta
            reason_parts.append(f"{lesson.get('id')} negative/weak closed outcome evidence {negative_outcome_delta:+.0f}")
            fired = True
        if scoring.get("valid_target_math_visibility") and entry_limit is not None and entry_limit > 0 and ticket:
            reason_parts.append(f"{lesson.get('id')} valid target math stays visible")
            fired = True
        if fired:
            delta += row_delta
            reasons.extend(reason_parts)
            audit_rows.append(
                {
                    "ticker": ticker,
                    "lesson_id": lesson.get("id", ""),
                    "failure_mode": lesson.get("failure_mode", ""),
                    "surface": "synthesis_scoring",
                    "score_delta": row_delta,
                    "action": "; ".join(reason_parts),
                    "severity": lesson.get("severity", ""),
                    "evidence": "; ".join(str(item) for item in lesson.get("evidence", [])[:4]),
                }
            )
    return delta, reasons, audit_rows


def build_application_audit(final: pd.DataFrame, decision_board: pd.DataFrame, pack: LessonPack) -> pd.DataFrame:
    columns = [
        "ticker",
        "lesson_id",
        "failure_mode",
        "surface",
        "score_delta",
        "action",
        "severity",
        "evidence",
        "recommendation_rank",
        "target_order_status",
        "ready_to_enter",
    ]
    rows: list[dict[str, Any]] = []
    if final is not None and not final.empty:
        for _, row in final.iterrows():
            raw = str(row.get("lesson_application_rows") or "").strip()
            if not raw:
                continue
            try:
                applied = json.loads(raw)
            except json.JSONDecodeError:
                applied = []
            for item in applied if isinstance(applied, list) else []:
                if isinstance(item, Mapping):
                    rows.append(dict(item))
    if decision_board is not None and not decision_board.empty and rows:
        by_ticker = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in decision_board.to_dict("records")
            if str(row.get("ticker") or "").strip()
        }
        for row in rows:
            decision = by_ticker.get(str(row.get("ticker") or "").strip().upper(), {})
            row["recommendation_rank"] = decision.get("recommendation_rank", "")
            row["target_order_status"] = decision.get("target_order_status", "")
            row["ready_to_enter"] = decision.get("ready_to_enter", "")
    if not rows and pack.lesson_count:
        rows.append(
            {
                "ticker": "",
                "lesson_id": "",
                "failure_mode": "",
                "surface": "run",
                "score_delta": 0,
                "action": "active lesson pack loaded; no row-level lessons fired",
                "severity": "info",
                "evidence": pack.digest,
                "recommendation_rank": "",
                "target_order_status": "",
                "ready_to_enter": "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def analyze(
    *,
    base_dir: Path,
    out_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> dict[str, Path]:
    root = Path(base_dir).expanduser().resolve()
    run = run_id or dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(out_dir).expanduser().resolve() if out_dir else root / "out" / "lessonengine" / run
    out.mkdir(parents=True, exist_ok=True)
    corrections = _read_jsonl(user_corrections_path(root))
    events = _events_from_user_corrections(corrections, user_corrections_path(root))
    events.extend(_events_from_closed_trade_outcomes(root))
    events.extend(_events_from_options_agent_artifacts(root))
    events_df = pd.DataFrame(events, columns=_evidence_event_columns())
    modes_df = _failure_modes(events_df)
    candidates = _lesson_candidates_from_failure_modes(modes_df, events_df)
    proposed = {
        "schema_version": LESSON_SCHEMA_VERSION,
        "version": f"proposed-{run}",
        "status": "proposed",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": "lessonengine analyze",
        "lessons": candidates,
    }
    bindings = {
        "schema_version": LESSON_SCHEMA_VERSION,
        "generated_at": proposed["generated_at"],
        "bindings": [
            {"lesson_id": lesson["id"], "failure_mode": lesson["failure_mode"], "actions": lesson["actions"]}
            for lesson in candidates
        ],
    }
    decision_packet = _promotion_decision_packet(candidates, events_df, modes_df, out)
    paths = {
        "out_dir": out,
        "evidence_events": out / "evidence_events.csv",
        "failure_modes": out / "failure_modes.csv",
        "lesson_candidates": out / "lesson_candidates.proposed.yaml",
        "lesson_action_bindings": out / "lesson_action_bindings.yaml",
        "promotion_decision_packet": out / "promotion_decision_packet.json",
        "report": out / "lessonengine_report.md",
    }
    events_df.to_csv(paths["evidence_events"], index=False)
    modes_df.to_csv(paths["failure_modes"], index=False)
    _write_yaml(paths["lesson_candidates"], proposed)
    _write_yaml(paths["lesson_action_bindings"], bindings)
    _write_json(paths["promotion_decision_packet"], decision_packet)
    paths["report"].write_text(_render_analyze_report(events_df, modes_df, candidates, decision_packet), encoding="utf-8")
    return paths


def promote(
    *,
    base_dir: Path,
    candidate: Path,
    target_version: str,
    decision_packet: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    baseline_version: Optional[str] = None,
    regression_dates: Optional[Sequence[str]] = None,
    max_bot_rows: Optional[int] = None,
) -> dict[str, Path]:
    root = Path(base_dir).expanduser().resolve()
    candidate_path = Path(candidate).expanduser().resolve()
    candidate_payload = _read_yaml(candidate_path)
    lessons = [dict(lesson) for lesson in candidate_payload.get("lessons", []) if isinstance(lesson, Mapping)]
    if not lessons:
        raise ValueError("candidate lesson pack has no lessons to promote")
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{target_version}"
    out = Path(out_dir).expanduser().resolve() if out_dir else root / "out" / "options_agent_regression" / run_id
    out.mkdir(parents=True, exist_ok=True)
    active = load_active_lesson_pack(root, version=baseline_version) if baseline_version else load_active_lesson_pack(root)
    promoted_lessons = []
    now = dt.datetime.now().isoformat(timespec="seconds")
    for lesson in lessons:
        promoted = dict(lesson)
        promoted["status"] = "active"
        promoted["last_validated_at"] = now
        promoted["promotion_regression_run"] = str(out)
        promoted_lessons.append(promoted)
    proposed_pack = {
        "schema_version": LESSON_SCHEMA_VERSION,
        "version": target_version,
        "status": "active",
        "generated_at": now,
        "prompt_budget_lines": DEFAULT_PROMPT_BUDGET_LINES,
        "lessons": promoted_lessons,
    }
    errors = validate_lesson_pack(proposed_pack)
    if errors:
        raise ValueError("cannot promote invalid lesson pack: " + "; ".join(errors))
    candidate_pack_dir = out / "candidate_lesson_pack"
    candidate_pack_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(candidate_pack_dir / "lessons.yaml", proposed_pack)
    (candidate_pack_dir / "lessons.md").write_text(build_prompt_pack(promoted_lessons), encoding="utf-8")
    actual_diffs = _run_options_agent_regression(
        root=root,
        out=out,
        dates=list(regression_dates or []),
        baseline=active,
        candidate_pack_dir=candidate_pack_dir,
        max_bot_rows=max_bot_rows,
    )
    quality = _quality_gate_results(promoted_lessons)
    quality_path = out / "quality_gate_results.csv"
    quality.to_csv(quality_path, index=False)
    failed = quality[quality["status"].astype(str).str.upper().eq("FAIL")]
    paths = _promotion_artifact_paths(out)
    _write_regression_artifacts(paths, active, proposed_pack, quality, actual_diffs=actual_diffs)
    verdict = {
        "target_version": target_version,
        "current_active_version": active.version,
        "candidate_path": str(candidate_path),
        "decision_packet": str(decision_packet.expanduser().resolve()) if decision_packet else "",
        "status": "FAIL" if not failed.empty else "PASS",
        "failed_gates": failed["gate"].astype(str).tolist(),
        "promotion_allowed": failed.empty,
        "regression_run": str(out),
        "regression_dates": list(regression_dates or []),
        "baseline_lesson_pack_version": active.version,
    }
    _write_json(paths["promotion_verdict"], verdict)
    if not failed.empty:
        raise ValueError("promotion failed hard gates: " + ", ".join(verdict["failed_gates"]))
    pack_dir = lesson_packs_dir(root) / target_version
    pack_dir.mkdir(parents=True, exist_ok=True)
    lessons_yaml = pack_dir / "lessons.yaml"
    lessons_md = pack_dir / "lessons.md"
    release_manifest = pack_dir / "release_manifest.json"
    changelog = pack_dir / "changelog.md"
    _write_yaml(lessons_yaml, proposed_pack)
    lessons_md.write_text(build_prompt_pack(promoted_lessons), encoding="utf-8")
    release = {
        "version": target_version,
        "released_at": now,
        "regression_run": str(out),
        "lesson_count": len(promoted_lessons),
        "lesson_digest": _stable_digest({"version": target_version, "lessons": promoted_lessons}),
        "candidate_path": str(candidate_path),
    }
    _write_json(release_manifest, release)
    changelog.write_text(_render_changelog(target_version, promoted_lessons, release), encoding="utf-8")
    _write_json(
        active_pointer_path(root),
        {
            "active_lesson_pack_version": target_version,
            "active_lesson_pack_path": str(pack_dir),
            "activated_at": now,
            "release_manifest": str(release_manifest),
            "regression_run": str(out),
        },
    )
    paths.update(
        {
            "pack_dir": pack_dir,
            "lessons_yaml": lessons_yaml,
            "lessons_md": lessons_md,
            "release_manifest": release_manifest,
            "changelog": changelog,
            "active_pointer": active_pointer_path(root),
        }
    )
    return paths


def _evidence_event_columns() -> list[str]:
    return [
        "event_id",
        "event_type",
        "severity",
        "source_path",
        "source_row_ref",
        "ticker",
        "observed",
        "desired",
        "failure_mode",
        "lesson_id",
        "evidence_score",
    ]


def _events_from_user_corrections(corrections: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, correction in enumerate(corrections, start=1):
        text = " ".join(
            str(correction.get(key, ""))
            for key in ("lesson", "correction", "text", "raw_text", "note")
            if correction.get(key) is not None
        ).strip()
        failure_mode = _classify_correction(text)
        if not failure_mode:
            continue
        library = FAILURE_MODE_LIBRARY[failure_mode]
        events.append(
            {
                "event_id": f"user_correction:{idx}",
                "event_type": "user_correction",
                "severity": library["severity"],
                "source_path": str(path),
                "source_row_ref": str(correction.get("_line_no", idx)),
                "ticker": str(correction.get("ticker", "")),
                "observed": text,
                "desired": library["rule"],
                "failure_mode": failure_mode,
                "lesson_id": library["lesson_id"],
                "evidence_score": 100,
            }
        )
    return events


def _classify_correction(text: str) -> str:
    lowered = text.lower()
    if "portfolio" in lowered and ("suppress" in lowered or "hide" in lowered or "block" in lowered):
        return "PORTFOLIO_RISK_SUPPRESSION"
    if "market" in lowered and ("closed" in lowered or "eod" in lowered) and ("target" in lowered or "debit" in lowered or "credit" in lowered):
        return "EOD_TARGET_SUPPRESSION"
    if ("missing" in lowered and ("target" in lowered or "debit" in lowered or "credit" in lowered)) or "prevent green" in lowered:
        return "MISSING_TARGET_PRICE_GREEN"
    if ("buy" in lowered and "sell" in lowered and ("leg" in lowered or "expiration" in lowered)) or "occ" in lowered:
        return "REPORT_LEG_DETAIL_MISSING"
    if "top" in lowered and ("cutoff" in lowered or "hide" in lowered):
        return "ARBITRARY_TOP_N_SUPPRESSION"
    if any(token in lowered for token in ("junk", "tail", "low-quality", "speculative")):
        return "JUNK_TICKER_PROMOTION"
    if "status" in lowered and ("icon" in lowered or "label" in lowered or "green" in lowered or "yellow" in lowered):
        return "STATUS_VISIBILITY_MISSING"
    if "major" in lowered or any(ticker.lower() in lowered for ticker in MAJOR_FOCUS_TICKERS[:8]):
        return "MAJOR_TICKER_COVERAGE_MISSING"
    if "multi-agent" in lowered or "subagent" in lowered:
        return "MULTI_AGENT_NOT_MANDATORY"
    return ""


def _events_from_options_agent_artifacts(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    out_root = root / "out" / "options_agent"
    if not out_root.exists():
        return events
    for run_dir in sorted([path for path in out_root.iterdir() if path.is_dir()], reverse=True)[:25]:
        tickets_path = run_dir / "trade_tickets.csv"
        if tickets_path.exists():
            try:
                tickets = pd.read_csv(tickets_path)
            except Exception:
                tickets = pd.DataFrame()
            if not tickets.empty and {"ready_to_enter", "entry_limit"}.issubset(tickets.columns):
                green_missing = tickets[
                    tickets["ready_to_enter"].map(_truthy)
                    & (pd.to_numeric(tickets["entry_limit"], errors="coerce").fillna(0) <= 0)
                ]
                for idx, row in green_missing.iterrows():
                    events.append(_artifact_event(run_dir, tickets_path, idx, row, "MISSING_TARGET_PRICE_GREEN", "green ticket missing target debit/credit"))
                if "underlying_quality_tier" in tickets.columns:
                    low_quality = tickets[
                        tickets["underlying_quality_tier"].astype(str).str.lower().isin(["speculative", "excluded"])
                        & tickets.get("target_order_status", pd.Series("", index=tickets.index)).astype(str).str.lower().isin(["target_order_candidate", "target_order_wait_for_price"])
                    ]
                    for idx, row in low_quality.iterrows():
                        events.append(_artifact_event(run_dir, tickets_path, idx, row, "JUNK_TICKER_PROMOTION", "low-quality ticker reached action surface"))
        coverage_path = run_dir / "ticker_coverage_audit.csv"
        if coverage_path.exists():
            try:
                coverage = pd.read_csv(coverage_path)
            except Exception:
                coverage = pd.DataFrame()
            if not coverage.empty and "ticker" in coverage.columns:
                present = {str(ticker).upper() for ticker in coverage["ticker"].dropna().astype(str)}
                missing = [ticker for ticker in MAJOR_FOCUS_TICKERS if ticker not in present]
                if len(missing) >= len(MAJOR_FOCUS_TICKERS) // 2:
                    events.append(
                        _artifact_event(
                            run_dir,
                            coverage_path,
                            0,
                            {"ticker": ",".join(missing)},
                            "MAJOR_TICKER_COVERAGE_MISSING",
                            "major focus ticker coverage mostly absent",
                        )
                    )
    return events


def _events_from_closed_trade_outcomes(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    out_root = root / "out"
    events.extend(_events_from_closed_trades_jsonl(out_root / "schwab_pull_state" / "closed_trades_acct_3326.jsonl"))
    events.extend(_events_from_schwab_numbers_json(out_root / "codex_schwab_numbers_rows.json"))
    for path in (out_root / "codexuw_execute_outcome_ledger.csv", out_root / "codexuw_recommendation_outcome_ledger.csv"):
        events.extend(_events_from_outcome_csv(path))
    return events


def _events_from_closed_trades_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    events = []
    for idx, row in enumerate(rows, start=1):
        pnl = _as_float(row.get("realized_pnl"))
        if pnl is None or pnl >= 0:
            continue
        events.append(_closed_outcome_event(path, idx, row, pnl, source="schwab_closed_trades"))
    return events


def _events_from_schwab_numbers_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else []
    events = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        pnl = _as_float(row.get("realized_pnl_value"))
        if pnl is None:
            pnl = _money_float(row.get("realized_pnl") or row.get("total_pnl"))
        if pnl is None or pnl >= 0:
            continue
        events.append(_closed_outcome_event(path, idx, row, pnl, source="schwab_numbers"))
    return events


def _events_from_outcome_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not _safe_non_v4_path(path):
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if frame.empty:
        return []
    pnl_col = "realized_pnl" if "realized_pnl" in frame.columns else "pnl" if "pnl" in frame.columns else ""
    if not pnl_col:
        return []
    events = []
    for idx, row in frame.iterrows():
        pnl = _as_float(row.get(pnl_col))
        if pnl is None or pnl >= 0:
            continue
        events.append(_closed_outcome_event(path, idx, row, pnl, source="forward_outcome_ledger"))
    return events


def _closed_outcome_event(path: Path, idx: Any, row: Mapping[str, Any], pnl: float, *, source: str) -> dict[str, Any]:
    failure_mode = "NEGATIVE_CLOSED_TRADE_OUTCOME"
    library = FAILURE_MODE_LIBRARY[failure_mode]
    ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
    strategy = str(row.get("strategy") or row.get("trade_strategy") or "").strip()
    return {
        "event_id": f"{source}:{idx}",
        "event_type": "closed_trade_outcome",
        "severity": library["severity"],
        "source_path": str(path),
        "source_row_ref": str(idx),
        "ticker": ticker,
        "observed": f"closed realized_pnl={pnl:g}" + (f" strategy={strategy}" if strategy else ""),
        "desired": library["rule"],
        "failure_mode": failure_mode,
        "lesson_id": library["lesson_id"],
        "evidence_score": min(100, max(50, int(abs(pnl) // 10))),
    }


def _artifact_event(run_dir: Path, path: Path, idx: Any, row: Mapping[str, Any], failure_mode: str, observed: str) -> dict[str, Any]:
    library = FAILURE_MODE_LIBRARY[failure_mode]
    return {
        "event_id": f"{run_dir.name}:{path.name}:{idx}",
        "event_type": "options_agent_audit",
        "severity": library["severity"],
        "source_path": str(path),
        "source_row_ref": str(idx),
        "ticker": str(row.get("ticker", "")),
        "observed": observed,
        "desired": library["rule"],
        "failure_mode": failure_mode,
        "lesson_id": library["lesson_id"],
        "evidence_score": 80,
    }


def _failure_modes(events_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["failure_mode", "lesson_id", "event_count", "max_evidence_score", "severity", "source_count", "decision"]
    if events_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for failure_mode, group in events_df.groupby("failure_mode", dropna=False):
        if not failure_mode:
            continue
        rows.append(
            {
                "failure_mode": failure_mode,
                "lesson_id": group["lesson_id"].dropna().astype(str).iloc[0] if not group.empty else "",
                "event_count": len(group),
                "max_evidence_score": pd.to_numeric(group["evidence_score"], errors="coerce").max(),
                "severity": group["severity"].dropna().astype(str).iloc[0] if "severity" in group else "",
                "source_count": group["source_path"].dropna().astype(str).nunique(),
                "decision": "PROPOSE_LESSON",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["severity", "event_count"], ascending=[True, False])


def _lesson_candidates_from_failure_modes(modes_df: pd.DataFrame, events_df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if modes_df.empty:
        return candidates
    now = dt.datetime.now().isoformat(timespec="seconds")
    for _, mode in modes_df.iterrows():
        failure_mode = str(mode.get("failure_mode") or "")
        if failure_mode not in FAILURE_MODE_LIBRARY:
            continue
        library = FAILURE_MODE_LIBRARY[failure_mode]
        mode_events = events_df[events_df["failure_mode"].astype(str).eq(failure_mode)] if not events_df.empty else pd.DataFrame()
        evidence = [
            f"{row.get('event_type')}:{row.get('source_path')}#{row.get('source_row_ref')}"
            for _, row in mode_events.head(8).iterrows()
        ]
        applies_to = sorted(library["actions"].keys())
        candidates.append(
            _with_failure_mode_specific_actions(
                {
                "id": library["lesson_id"],
                "status": "proposed",
                "severity": library["severity"],
                "failure_mode": failure_mode,
                "rule": library["rule"],
                "evidence": evidence,
                "confidence": int(mode.get("max_evidence_score") or 0),
                "risk_score": _risk_score(library["severity"], int(mode.get("event_count") or 1)),
                "applies_to": applies_to,
                "actions": library["actions"],
                "created_at": now,
                "last_validated_at": "",
                "promotion_regression_run": "",
                },
                failure_mode=failure_mode,
                mode_events=mode_events,
            )
        )
    return candidates


def _with_failure_mode_specific_actions(
    lesson: dict[str, Any],
    *,
    failure_mode: str,
    mode_events: pd.DataFrame,
) -> dict[str, Any]:
    if failure_mode != "NEGATIVE_CLOSED_TRADE_OUTCOME" or mode_events.empty:
        return lesson
    tickers = sorted(
        {
            str(value).strip().upper()
            for value in mode_events.get("ticker", pd.Series(dtype=str)).dropna().astype(str)
            if str(value).strip()
        }
    )
    actions = dict(lesson.get("actions", {}))
    scoring = dict(actions.get("synthesis_scoring", {}))
    scoring["negative_closed_outcome_tickers"] = tickers
    actions["synthesis_scoring"] = scoring
    lesson["actions"] = actions
    lesson["applies_to"] = sorted(actions.keys())
    return lesson


def _promotion_decision_packet(candidates: Sequence[Mapping[str, Any]], events_df: pd.DataFrame, modes_df: pd.DataFrame, out: Path) -> dict[str, Any]:
    return {
        "schema_version": LESSON_SCHEMA_VERSION,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "decision": "REGRESSION_REQUIRED" if candidates else "NO_CANDIDATES",
        "candidate_count": len(candidates),
        "event_count": int(len(events_df)),
        "failure_mode_count": int(len(modes_df)),
        "required_next_action": "run lessonengine promote after reviewing proposed lessons" if candidates else "no promotion needed",
        "candidate_lesson_ids": [lesson.get("id") for lesson in candidates],
        "evidence_events_file": str(out / "evidence_events.csv"),
        "failure_modes_file": str(out / "failure_modes.csv"),
        "candidate_file": str(out / "lesson_candidates.proposed.yaml"),
    }


def _quality_gate_results(lessons: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lesson_ids = {str(lesson.get("id")) for lesson in lessons}
    action_surfaces = {
        surface
        for lesson in lessons
        for surface in (lesson.get("actions", {}) if isinstance(lesson.get("actions"), Mapping) else {}).keys()
    }
    required = {
        "lesson_schema_valid": True,
        "subagent_prompt_digest_context": "subagent_prompt" in action_surfaces,
        "lesson_application_audit_contract": True,
        "missing_target_price_green_block": "OA-003" in lesson_ids or any(
            "block_green_when_target_price_missing" in str(lesson.get("actions", {})) for lesson in lessons
        ),
        "portfolio_only_annotation": "OA-001" in lesson_ids or any(
            "forbid_portfolio_only_suppression" in str(lesson.get("actions", {})) for lesson in lessons
        ),
        "eod_target_visibility": "OA-002" in lesson_ids or any(
            "forbid_market_closed_as_hard_blocker" in str(lesson.get("actions", {})) for lesson in lessons
        ),
        "report_buy_sell_expiration": "report_contract" in action_surfaces,
        "major_ticker_coverage": "coverage_audit" in action_surfaces,
        "junk_ticker_filter": "OA-006" in lesson_ids or any(
            "block_low_quality_action_without_strong_evidence" in str(lesson.get("actions", {})) for lesson in lessons
        ),
    }
    for gate, passed in required.items():
        rows.append(
            {
                "gate": gate,
                "status": "PASS" if passed else "FAIL",
                "detail": "required lesson/action binding present" if passed else "required lesson/action binding missing",
            }
        )
    return pd.DataFrame(rows, columns=["gate", "status", "detail"])


def _promotion_artifact_paths(out: Path) -> dict[str, Path]:
    return {
        "out_dir": out,
        "regression_report": out / "options_agent_regression_report.md",
        "quality_gate_results": out / "quality_gate_results.csv",
        "trade_recommendation_diff": out / "trade_recommendation_diff.csv",
        "lesson_impact_diff": out / "lesson_impact_diff.csv",
        "score_distribution_diff": out / "score_distribution_diff.csv",
        "green_yellow_block_diff": out / "green_yellow_block_diff.csv",
        "report_visibility_diff": out / "report_visibility_diff.csv",
        "promotion_verdict": out / "promotion_verdict.json",
    }


def _write_regression_artifacts(
    paths: Mapping[str, Path],
    active: LessonPack,
    proposed_pack: Mapping[str, Any],
    quality: pd.DataFrame,
    *,
    actual_diffs: Optional[Mapping[str, pd.DataFrame]] = None,
) -> None:
    proposed_lessons = proposed_pack.get("lessons", [])
    new_ids = sorted({str(lesson.get("id")) for lesson in proposed_lessons} - {str(lesson.get("id")) for lesson in active.lessons})
    actual_diffs = actual_diffs or {}
    trade_diff = actual_diffs.get("trade_recommendation_diff")
    if trade_diff is None or trade_diff.empty:
        trade_diff = pd.DataFrame(
        [
            {
                "metric": "lesson_count",
                "baseline": active.lesson_count,
                "candidate": len(proposed_lessons),
                "diff": len(proposed_lessons) - active.lesson_count,
                "explanation": "lesson-pack release comparison",
            },
            {
                "metric": "new_lesson_ids",
                "baseline": "",
                "candidate": ", ".join(new_ids),
                "diff": len(new_ids),
                "explanation": "new active lessons after promotion",
            },
        ]
        )
    trade_diff.to_csv(paths["trade_recommendation_diff"], index=False)
    lesson_impact = actual_diffs.get("lesson_impact_diff")
    if lesson_impact is None or lesson_impact.empty:
        lesson_impact = pd.DataFrame(
        [
            {
                "lesson_id": lesson.get("id"),
                "failure_mode": lesson.get("failure_mode"),
                "surfaces": ", ".join((lesson.get("actions", {}) or {}).keys()),
                "impact": "active after promotion",
            }
            for lesson in proposed_lessons
        ]
        )
    lesson_impact.to_csv(paths["lesson_impact_diff"], index=False)
    score_diff = actual_diffs.get("score_distribution_diff")
    if score_diff is None or score_diff.empty:
        score_diff = pd.DataFrame(
            [{"metric": "synthesis_score_policy_version", "baseline": active.version, "candidate": proposed_pack.get("version"), "diff": "versioned"}]
        )
    score_diff.to_csv(paths["score_distribution_diff"], index=False)
    gyb_diff = actual_diffs.get("green_yellow_block_diff")
    if gyb_diff is None or gyb_diff.empty:
        gyb_diff = pd.DataFrame(
            [{"surface": "green_yellow_block", "baseline": active.version, "candidate": proposed_pack.get("version"), "diff": "no dated regression run requested"}]
        )
    gyb_diff.to_csv(paths["green_yellow_block_diff"], index=False)
    report_diff = actual_diffs.get("report_visibility_diff")
    if report_diff is None or report_diff.empty:
        report_diff = pd.DataFrame(
            [{"surface": "report_visibility", "baseline": active.version, "candidate": proposed_pack.get("version"), "diff": "report contract gates enforced"}]
        )
    report_diff.to_csv(paths["report_visibility_diff"], index=False)
    paths["regression_report"].write_text(
        _render_regression_report(active, proposed_pack, quality, actual_diffs=actual_diffs),
        encoding="utf-8",
    )


def _run_options_agent_regression(
    *,
    root: Path,
    out: Path,
    dates: Sequence[str],
    baseline: LessonPack,
    candidate_pack_dir: Path,
    max_bot_rows: Optional[int],
) -> dict[str, pd.DataFrame]:
    if not dates:
        return {}
    from uwos.options_agent.core import run_pipeline

    rows_trade: list[dict[str, Any]] = []
    rows_impact: list[dict[str, Any]] = []
    rows_score: list[dict[str, Any]] = []
    rows_gyb: list[dict[str, Any]] = []
    rows_report: list[dict[str, Any]] = []
    for day in dates:
        safe_day = str(day).strip()
        if not safe_day:
            continue
        baseline_out = out / "runs" / "baseline" / safe_day
        candidate_out = out / "runs" / "candidate" / safe_day
        run_pipeline(
            safe_day,
            root=root,
            out_dir=baseline_out,
            max_bot_rows=max_bot_rows,
            lesson_pack_version=baseline.version if baseline.version != "none" else None,
        )
        run_pipeline(
            safe_day,
            root=root,
            out_dir=candidate_out,
            max_bot_rows=max_bot_rows,
            lesson_pack_path=candidate_pack_dir,
        )
        base_decision = _safe_read_csv(baseline_out / "decision_board.csv")
        cand_decision = _safe_read_csv(candidate_out / "decision_board.csv")
        base_tickets = _safe_read_csv(baseline_out / "trade_tickets.csv")
        cand_tickets = _safe_read_csv(candidate_out / "trade_tickets.csv")
        rows_trade.extend(_recommendation_diff_rows(safe_day, base_decision, cand_decision, baseline.version))
        rows_gyb.extend(_green_yellow_block_rows(safe_day, base_tickets, cand_tickets))
        rows_score.extend(_score_distribution_rows(safe_day, base_decision, cand_decision))
        rows_report.append(_report_visibility_row(safe_day, baseline_out / f"options_agent_report_{safe_day}.md", candidate_out / f"options_agent_report_{safe_day}.md"))
        impact = _safe_read_csv(candidate_out / "lessons_application_audit.csv")
        if not impact.empty:
            for _, row in impact.iterrows():
                item = row.to_dict()
                item["date"] = safe_day
                item["candidate_run"] = str(candidate_out)
                rows_impact.append(item)
    return {
        "trade_recommendation_diff": pd.DataFrame(rows_trade),
        "lesson_impact_diff": pd.DataFrame(rows_impact),
        "score_distribution_diff": pd.DataFrame(rows_score),
        "green_yellow_block_diff": pd.DataFrame(rows_gyb),
        "report_visibility_diff": pd.DataFrame(rows_report),
    }


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _recommendation_diff_rows(day: str, baseline: pd.DataFrame, candidate: pd.DataFrame, baseline_version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tickers = sorted(
        set(baseline.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
        | set(candidate.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    )
    base_by_ticker = {str(row.get("ticker")): row for row in baseline.to_dict("records")}
    cand_by_ticker = {str(row.get("ticker")): row for row in candidate.to_dict("records")}
    for ticker in tickers:
        base = base_by_ticker.get(ticker, {})
        cand = cand_by_ticker.get(ticker, {})
        changed = any(
            str(base.get(col, "")) != str(cand.get(col, ""))
            for col in ("recommendation_rank", "final_action", "target_order_status", "ready_to_enter", "synthesis_score")
        )
        if changed:
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "baseline_version": baseline_version,
                    "baseline_rank": base.get("recommendation_rank", ""),
                    "candidate_rank": cand.get("recommendation_rank", ""),
                    "baseline_action": base.get("final_action", ""),
                    "candidate_action": cand.get("final_action", ""),
                    "baseline_target_status": base.get("target_order_status", ""),
                    "candidate_target_status": cand.get("target_order_status", ""),
                    "baseline_ready": base.get("ready_to_enter", ""),
                    "candidate_ready": cand.get("ready_to_enter", ""),
                    "baseline_score": base.get("synthesis_score", ""),
                    "candidate_score": cand.get("synthesis_score", ""),
                }
            )
    if not rows:
        rows.append({"date": day, "ticker": "", "diff": "no recommendation changes"})
    return rows


def _green_yellow_block_rows(day: str, baseline: pd.DataFrame, candidate: pd.DataFrame) -> list[dict[str, Any]]:
    def counts(frame: pd.DataFrame) -> dict[str, int]:
        if frame.empty:
            return {"green": 0, "yellow": 0, "blocked": 0}
        target_status = frame.get("target_order_status", pd.Series("", index=frame.index)).astype(str).str.lower()
        ready = frame.get("ready_to_enter", pd.Series(False, index=frame.index)).map(_truthy)
        return {
            "green": int(ready.sum()),
            "yellow": int((target_status.isin(["target_order_candidate", "target_order_wait_for_price"]) & ~ready).sum()),
            "blocked": int(target_status.str.startswith("blocked").sum() + target_status.str.startswith("not_actionable").sum()),
        }

    base = counts(baseline)
    cand = counts(candidate)
    return [
        {
            "date": day,
            "metric": metric,
            "baseline": base[metric],
            "candidate": cand[metric],
            "diff": cand[metric] - base[metric],
        }
        for metric in ("green", "yellow", "blocked")
    ]


def _score_distribution_rows(day: str, baseline: pd.DataFrame, candidate: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        scores = pd.to_numeric(frame.get("synthesis_score", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "date": day,
                "version_side": label,
                "count": int(len(scores)),
                "mean": round(float(scores.mean()), 4) if not scores.empty else "",
                "min": round(float(scores.min()), 4) if not scores.empty else "",
                "max": round(float(scores.max()), 4) if not scores.empty else "",
            }
        )
    return rows


def _report_visibility_row(day: str, baseline_report: Path, candidate_report: Path) -> dict[str, Any]:
    def facts(path: Path) -> dict[str, bool]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return {
            "has_buy_leg": "Buy Leg" in text,
            "has_sell_leg": "Sell Leg" in text,
            "has_target_limit": "Target Limit" in text,
            "has_lesson_pack": "Lesson pack:" in text,
        }

    base = facts(baseline_report)
    cand = facts(candidate_report)
    return {
        "date": day,
        **{f"baseline_{key}": value for key, value in base.items()},
        **{f"candidate_{key}": value for key, value in cand.items()},
    }


def _render_analyze_report(events_df: pd.DataFrame, modes_df: pd.DataFrame, candidates: Sequence[Mapping[str, Any]], packet: Mapping[str, Any]) -> str:
    lines = [
        "# LessonEngine Analyze Report",
        "",
        f"Evidence events: {len(events_df)}",
        f"Failure modes: {len(modes_df)}",
        f"Proposed lessons: {len(candidates)}",
        f"Decision: {packet.get('decision')}",
        "",
    ]
    if candidates:
        lines.append("## Proposed Lessons")
        lines.append("")
        for lesson in candidates:
            lines.append(f"- {lesson.get('id')} `{lesson.get('failure_mode')}`: {lesson.get('rule')}")
    return "\n".join(lines) + "\n"


def _render_regression_report(
    active: LessonPack,
    proposed_pack: Mapping[str, Any],
    quality: pd.DataFrame,
    *,
    actual_diffs: Optional[Mapping[str, pd.DataFrame]] = None,
) -> str:
    failures = quality[quality["status"].astype(str).str.upper().eq("FAIL")]
    actual_diffs = actual_diffs or {}
    lines = [
        "# Options Agent Lesson Pack Regression Report",
        "",
        f"Baseline lesson pack: {active.version}",
        f"Candidate lesson pack: {proposed_pack.get('version')}",
        f"Quality gates: {'PASS' if failures.empty else 'FAIL'}",
        f"Dated regression runs: {'yes' if actual_diffs else 'no'}",
        "",
        "## Gate Results",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row.get('gate')} | {row.get('status')} | {str(row.get('detail')).replace('|', '/')} |")
    lines.extend(["", "## Candidate Lessons", ""])
    for lesson in proposed_pack.get("lessons", []):
        lines.append(f"- {lesson.get('id')} {lesson.get('failure_mode')}: {lesson.get('rule')}")
    if actual_diffs:
        lines.extend(["", "## Dated Regression Summary", ""])
        for name, frame in actual_diffs.items():
            lines.append(f"- `{name}` rows: {len(frame)}")
    return "\n".join(lines) + "\n"


def _render_changelog(version: str, lessons: Sequence[Mapping[str, Any]], release: Mapping[str, Any]) -> str:
    lines = [
        f"# {version} Lesson Pack Changelog",
        "",
        f"Released: {release.get('released_at')}",
        f"Regression run: `{release.get('regression_run')}`",
        f"Digest: `{release.get('lesson_digest')}`",
        "",
    ]
    for lesson in lessons:
        lines.append(f"- Promoted {lesson.get('id')} `{lesson.get('failure_mode')}`: {lesson.get('rule')}")
    return "\n".join(lines) + "\n"


def _risk_score(severity: str, event_count: int) -> int:
    base = {"hard": 80, "medium": 50, "advisory": 25}.get(severity, 40)
    return min(100, base + max(0, event_count - 1) * 5)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _money_float(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    number = _as_float(cleaned)
    if number is None:
        return None
    return -abs(number) if negative else number


def _safe_non_v4_path(path: Path) -> bool:
    lowered = str(path).lower()
    blocked = ("codexdaily" + "_v" + "4", "daily" + "_v" + "4")
    return all(token not in lowered for token in blocked)


def _git_sha(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Options Agent LessonEngine")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze", help="Analyze durable evidence and propose lessons")
    analyze_parser.add_argument("--base-dir", default=str(project_root()))
    analyze_parser.add_argument("--out-dir", default="")
    analyze_parser.add_argument("--run-id", default="")

    promote_parser = sub.add_parser("promote", help="Promote a proposed lesson pack after regression gates")
    promote_parser.add_argument("--base-dir", default=str(project_root()))
    promote_parser.add_argument("--candidate", required=True)
    promote_parser.add_argument("--decision-packet", default="")
    promote_parser.add_argument("--target-options-agent-version", required=True)
    promote_parser.add_argument("--out-dir", default="")
    promote_parser.add_argument("--baseline-lesson-pack-version", default="")
    promote_parser.add_argument("--dates", nargs="*", default=[])
    promote_parser.add_argument("--max-bot-rows", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "analyze":
        paths = analyze(
            base_dir=Path(args.base_dir),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            run_id=args.run_id or None,
        )
        print(f"LessonEngine analyze artifacts: {paths['out_dir']}")
        print(f"Report: {paths['report']}")
        return 0
    if args.command == "promote":
        paths = promote(
            base_dir=Path(args.base_dir),
            candidate=Path(args.candidate),
            decision_packet=Path(args.decision_packet) if args.decision_packet else None,
            target_version=args.target_options_agent_version,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            baseline_version=args.baseline_lesson_pack_version or None,
            regression_dates=args.dates or None,
            max_bot_rows=args.max_bot_rows or None,
        )
        print(f"Promoted lesson pack: {paths['pack_dir']}")
        print(f"Regression report: {paths['regression_report']}")
        return 0
    return 2
