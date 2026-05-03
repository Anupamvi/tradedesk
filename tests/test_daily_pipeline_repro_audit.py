from uwos.daily_pipeline_repro_audit import artifact_names, compare_artifacts, select_folders


def test_compare_artifacts_flags_byte_drift(tmp_path):
    date_text = "2026-04-23"
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    for name in artifact_names(date_text):
        (run_a / name).write_text("same", encoding="utf-8")
        (run_b / name).write_text("same", encoding="utf-8")
    (run_b / f"trade_decision_book_all_{date_text}.csv").write_text("different", encoding="utf-8")

    rows = compare_artifacts(run_a, run_b, date_text)
    by_name = {row["artifact"]: row for row in rows}

    assert by_name[f"planned_trade_journal_{date_text}.csv"]["same"] is True
    assert by_name[f"trade_decision_book_all_{date_text}.csv"]["same"] is False


def test_select_folders_reports_missing_specific_date(tmp_path):
    folders, incomplete = select_folders(tmp_path, ["2026-04-23"], None, None)

    assert folders == []
    assert incomplete == [{"date": "2026-04-23", "missing": ["folder_or_required_inputs"]}]
