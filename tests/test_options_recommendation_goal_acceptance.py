from __future__ import annotations

import json

from scripts import options_recommendation_goal_acceptance as goal


def test_combined_row_stays_partial_when_historical_coverage_is_incomplete() -> None:
    rows = [
        goal.ProofRow("options_pattern_matrix", "PASS", 10.0, "ok"),
        goal.ProofRow("options_pattern_order_entry", "PASS", 10.0, "ok"),
        goal.ProofRow("codexdaily_v3_functional_gates", "PASS", 8.0, "ok"),
        goal.ProofRow("codexdaily_v4_functional_gates", "PASS", 8.0, "ok"),
        goal.ProofRow("trade_desk_management", "PASS", 8.0, "ok"),
        goal.ProofRow("codexdaily_v3_historical_coverage", "PARTIAL", 6.5, "missing"),
    ]

    combined = goal._combined_row(rows)

    assert combined.status == "PARTIAL"
    assert combined.confidence_score == 6.8
    assert "functional_confidence_score=8.0" in combined.evidence
    assert combined.blocker == "codexdaily_v3_historical_coverage"


def test_manifest_coverage_flags_missing_and_stale_current_code_manifests(tmp_path) -> None:
    root = tmp_path
    manifest_dir = root / "out" / "codexdaily_v3_2026-05-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "codexdaily_v3_manifest_2026-05-01.json").write_text(
        json.dumps({"pipeline_name": "Codex Daily V3"}),
        encoding="utf-8",
    )

    row = goal._manifest_coverage_row(root, "v3", ["2026-05-01", "2026-05-02"])

    assert row.status == "PARTIAL"
    assert "manifests=1" in row.evidence
    assert "missing=1" in row.evidence
    assert "missing_visible_policy=1" in row.evidence
    assert row.blocker == "regenerate current-code manifests for missing/stale dates"


def test_manifest_coverage_prefers_external_codexdaily_proof_dir(tmp_path) -> None:
    root = tmp_path / "root"
    proof = tmp_path / "proof"
    manifest_dir = proof / "v3" / "codexdaily_v3_2026-05-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "codexdaily_v3_manifest_2026-05-01.json").write_text(
        json.dumps(
            {
                "pipeline_name": "Codex Daily V3",
                "visible_signal_policy": {
                    "active_execute_cap": None,
                    "active_board_cap": None,
                    "active_target_ticket_cap": None,
                    "risk_caps_size_and_label_only": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (proof / "codexdaily_historical_proof_checkpoint.json").write_text(
        json.dumps(
            {
                "proof_scope_status": "FULL",
                "proof_scope_notes": "uncapped historical discovery/candidate proof",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    row = goal._manifest_coverage_row(root, "v3", ["2026-05-01"], proof)

    assert row.status == "PASS"
    assert "manifest_sources=proof_dir:1" in row.evidence


def test_manifest_coverage_marks_capped_external_proof_partial(tmp_path) -> None:
    root = tmp_path / "root"
    proof_dir = tmp_path / "proof"
    manifest_dir = proof_dir / "v4" / "codexdaily_v4_2026-05-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "codexdaily_v4_manifest_2026-05-01.json").write_text(
        json.dumps(
            {
                "pipeline_name": "Codex Daily V4",
                "visible_signal_policy": {
                    "active_execute_cap": None,
                    "active_board_cap": None,
                    "active_target_ticket_cap": None,
                    "no_miss_reporting": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (proof_dir / "codexdaily_historical_proof_checkpoint.json").write_text(
        json.dumps(
            {
                "proof_scope_status": "CAPPED",
                "proof_scope_notes": "bot_max_rows=100,max_tickers=8,max_candidates=8",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    row = goal._manifest_coverage_row(root, "v4", ["2026-05-01"], proof_dir)

    assert row.status == "PARTIAL"
    assert "proof_scope=CAPPED" in row.evidence
    assert row.blocker == "rerun current-code CodexDaily historical proof uncapped"


def test_legacy_checkpoint_command_caps_are_detected(tmp_path) -> None:
    proof_dir = tmp_path / "proof"
    proof_dir.mkdir()
    (proof_dir / "codexdaily_historical_proof_checkpoint.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "command": (
                            "python3 -m codexuw.daily_v3 run --bot-max-rows 100 "
                            "--max-tickers 8 --max-candidates 8 --max-final-trades 0"
                        )
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status, note = goal._codexdaily_proof_scope(proof_dir)

    assert status == "CAPPED"
    assert "bot_max_rows=100" in note
    assert "max_tickers=8" in note
    assert "max_candidates=8" in note


def test_order_entry_uses_current_builder_to_label_auto_risk(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        '{"risk_config":{"min_baselines_beaten":1}}',
        encoding="utf-8",
    )
    (run_dir / "actionable_trades.csv").write_text(
        "\n".join(
            [
                "ticker,direction,strategy,buy_or_sell,call_or_put,strike_rates,expiration_date,"
                "suggested_entry_debit_credit_range,trade_legs,max_risk_per_contract,probability_score,"
                "success_probability_pct,expected_R,expected_R_per_day,validation_profit_factor,"
                "validation_scored_count,beats_baselines_count,baselines_beaten_names,baselines_beaten_details,"
                "calibrated_probability",
                "AAPL,bullish,Long Call Debit,BUY,CALL,210,2026-06-19,debit 7.50-7.70,"
                "Buy 1 AAPL 2026-06-19 210C @ debit 7.50-7.70 limit,"
                "770.65,62,58,0.24,0.012,1.42,30,2,BASELINE_A;BASELINE_B,details,0.58",
            ]
        ),
        encoding="utf-8",
    )
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    (matrix_dir / "goal_acceptance_matrix.csv").write_text(
        f"date,run_dir\n2026-05-29,{run_dir}\n",
        encoding="utf-8",
    )

    row = goal._option_pattern_order_entry_row(
        matrix_dir,
        {"gate_pass_trade_count": "1", "gate_fail_trade_count": "0"},
    )

    assert row.status == "PASS"
    assert "source=current_builder_from_goal_matrix" in row.evidence
    assert "major_tickers_seen=AAPL" in row.evidence


def test_functional_gates_pass_for_v3_v4_and_trade_desk() -> None:
    assert goal._v3_functional_row().status == "PASS"
    assert goal._v4_functional_row().status == "PASS"
    assert goal._trade_desk_row().status == "PASS"
