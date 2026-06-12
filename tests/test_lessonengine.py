import json
from pathlib import Path

import pandas as pd
import pytest

from uwos.lessonengine import core as lessonengine
from uwos.options_agent import core as options_core


def _write_seed_corrections(root: Path) -> None:
    knowledge = root / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "user_corrections.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Do not suppress good trades solely due to portfolio risk; annotate and size-risk them.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "EOD files are next-day planning inputs; show target credit/debit and mark pending live recheck instead of hiding candidates because market is closed.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Missing target debit/credit must prevent green execution readiness.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Reports must show buy sell legs and expiration dates, not only OCC codes.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Do not use arbitrary top-N cutoffs to hide candidates or audits.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Do not promote junk low-quality tail tickers into action rows without strong evidence.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Preserve status icons and labels for green yellow blocked and no-action rows.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Major liquid tickers such as AAPL NVDA MSFT GOOG GOOGL META AMZN TSLA AMD AVGO PLTR SPY QQQ must appear in coverage audit.",
                    }
                ),
                json.dumps(
                    {
                        "date": "2026-06-01",
                        "source": "test",
                        "lesson": "Multi-agent dispatch is mandatory for normal Options Agent runs.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_lesson_pack_schema_rejects_malformed_lessons() -> None:
    errors = lessonengine.validate_lesson_pack({"version": "bad", "lessons": [{"id": "OA-BAD"}]})
    assert errors
    assert any("missing required keys" in error for error in errors)


def test_lessonengine_analyze_bubbles_corrections_to_proposed_lessons(tmp_path: Path) -> None:
    _write_seed_corrections(tmp_path)
    paths = lessonengine.analyze(base_dir=tmp_path, run_id="test")

    events = pd.read_csv(paths["evidence_events"])
    modes = pd.read_csv(paths["failure_modes"])
    proposed = lessonengine._read_yaml(paths["lesson_candidates"])
    packet = json.loads(paths["promotion_decision_packet"].read_text(encoding="utf-8"))

    assert len(events) == 9
    assert set(modes["failure_mode"]) >= {"EOD_TARGET_SUPPRESSION", "MISSING_TARGET_PRICE_GREEN"}
    assert {lesson["id"] for lesson in proposed["lessons"]} >= {"OA-001", "OA-002", "OA-003", "OA-009"}
    assert all(lesson["status"] == "proposed" for lesson in proposed["lessons"])
    assert packet["decision"] == "REGRESSION_REQUIRED"


def test_lessonengine_analyze_bubbles_closed_trade_losses(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "user_corrections.jsonl").write_text("", encoding="utf-8")
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps({"ticker": "NVDA", "strategy": "long_call", "realized_pnl": -175.0}) + "\n",
        encoding="utf-8",
    )

    paths = lessonengine.analyze(base_dir=tmp_path, run_id="closed-loss")
    events = pd.read_csv(paths["evidence_events"])
    proposed = lessonengine._read_yaml(paths["lesson_candidates"])

    assert "closed_trade_outcome" in events["event_type"].tolist()
    assert "NEGATIVE_CLOSED_TRADE_OUTCOME" in events["failure_mode"].tolist()
    assert {lesson["id"] for lesson in proposed["lessons"]} == {"OA-010"}


def test_promote_publishes_active_pack_only_after_gates_pass(tmp_path: Path) -> None:
    _write_seed_corrections(tmp_path)
    analyze_paths = lessonengine.analyze(base_dir=tmp_path, run_id="promote")
    promote_paths = lessonengine.promote(
        base_dir=tmp_path,
        candidate=analyze_paths["lesson_candidates"],
        decision_packet=analyze_paths["promotion_decision_packet"],
        target_version="options-agent-v5",
        out_dir=tmp_path / "out" / "options_agent_regression" / "promote",
    )

    pointer = json.loads((tmp_path / "knowledge" / "options_agent_active_lesson_pack.json").read_text(encoding="utf-8"))
    verdict = json.loads(promote_paths["promotion_verdict"].read_text(encoding="utf-8"))
    pack = lessonengine.load_active_lesson_pack(tmp_path)

    assert pointer["active_lesson_pack_version"] == "options-agent-v5"
    assert verdict["promotion_allowed"] is True
    assert pack.lesson_count == 9
    assert "OA-002" in pack.markdown


def test_promote_refuses_missing_hard_gate_bindings(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.yaml"
    lesson = dict(lessonengine.FAILURE_MODE_LIBRARY["EOD_TARGET_SUPPRESSION"])
    candidate_payload = {
        "version": "bad-proposed",
        "lessons": [
            {
                "id": lesson["lesson_id"],
                "status": "proposed",
                "severity": lesson["severity"],
                "failure_mode": "EOD_TARGET_SUPPRESSION",
                "rule": lesson["rule"],
                "evidence": ["test"],
                "confidence": 100,
                "risk_score": 80,
                "applies_to": ["subagent_prompt"],
                "actions": {"subagent_prompt": {"inject": True}},
                "created_at": "2026-06-01T00:00:00",
                "last_validated_at": "",
                "promotion_regression_run": "",
            }
        ],
    }
    lessonengine._write_yaml(candidate, candidate_payload)

    with pytest.raises(ValueError, match="promotion failed hard gates"):
        lessonengine.promote(base_dir=tmp_path, candidate=candidate, target_version="options-agent-v5")

    assert not (tmp_path / "knowledge" / "options_agent_active_lesson_pack.json").exists()


def test_options_agent_dispatch_plan_injects_lesson_digest_into_every_subagent(tmp_path: Path) -> None:
    pack = lessonengine.LessonPack(
        version="options-agent-v5",
        lessons=[
            {
                "id": "OA-002",
                "status": "active",
                "severity": "hard",
                "failure_mode": "EOD_TARGET_SUPPRESSION",
                "rule": "Keep valid EOD targets yellow-visible.",
                "evidence": ["test"],
                "confidence": 100,
                "risk_score": 80,
                "applies_to": ["subagent_prompt"],
                "actions": {"subagent_prompt": {"inject": True}},
                "created_at": "2026-06-01T00:00:00",
                "last_validated_at": "2026-06-01T00:00:00",
                "promotion_regression_run": "/tmp/regression",
            }
        ],
        source_path=Path("/tmp/lessons.yaml"),
        markdown="# Options Agent Lessons\n\n- OA-002 [hard] Keep valid EOD targets yellow-visible.\n",
        digest="sha256:test",
    )
    candidates = pd.DataFrame([{"ticker": "WMT", "bias": "bullish", "score": 80, "flow_reason": "test"}])
    tasks = options_core.build_research_tasks(candidates, {"regime": "risk_on"}, pd.DataFrame(), top_trades=1, lesson_pack=pack)
    paths = options_core.output_paths("2026-06-01", root=tmp_path, out_dir=tmp_path / "out" / "options_agent")
    plan = options_core.build_agent_dispatch_plan(
        tasks,
        "2026-06-01",
        paths,
        lesson_pack=pack,
    )

    assert tasks["lesson_pack_digest"] == "sha256:test"
    assert plan["common_context"]["lesson_pack_digest"] == "sha256:test"
    assert plan["subagent_tasks"]
    assert all(task["lesson_pack_digest"] == "sha256:test" for task in plan["subagent_tasks"])
    assert all("OA-002" in task["prompt"] for task in plan["subagent_tasks"])


def test_synthesis_scoring_records_lesson_effects_and_application_audit() -> None:
    pack = lessonengine.LessonPack(
        version="options-agent-v5",
        lessons=[
            {
                "id": "OA-006",
                "status": "active",
                "severity": "hard",
                "failure_mode": "JUNK_TICKER_PROMOTION",
                "rule": "Penalize speculative action rows.",
                "evidence": ["test"],
                "confidence": 100,
                "risk_score": 80,
                "applies_to": ["synthesis_scoring"],
                "actions": {"synthesis_scoring": {"speculative_underlying_delta": -20}},
                "created_at": "2026-06-01T00:00:00",
                "last_validated_at": "2026-06-01T00:00:00",
                "promotion_regression_run": "/tmp/regression",
            }
        ],
        source_path=Path("/tmp/lessons.yaml"),
        markdown="# Options Agent Lessons\n\n- OA-006 [hard] Penalize speculative action rows.\n",
        digest="sha256:test",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "JUNK",
                "score": 80.0,
                "signal_premium": 1_000_000,
                "recommendation_status": options_core.RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "underlying_quality_tier": "speculative",
                "live_validation_status": "PASS",
                "full_ticket": "SELL 1 JUNK 2026-06-19 10 Put / BUY 1 JUNK 2026-06-19 9 Put @ 0.50 CREDIT",
                "entry_limit": 0.5,
            }
        ]
    )
    ranked = options_core.apply_synthesis_ranking(final, pd.DataFrame(), top_trades=1, lesson_pack=pack)
    audit = lessonengine.build_application_audit(ranked, pd.DataFrame(), pack)

    assert ranked["lesson_score_delta"].tolist() == [-20.0]
    assert ranked["lesson_ids_applied"].tolist() == ["OA-006"]
    assert audit["lesson_id"].tolist() == ["OA-006"]
