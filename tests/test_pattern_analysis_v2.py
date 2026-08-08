from pathlib import Path

import pytest

from uwos.pattern_analysis_v2 import PIPELINE_VERSION
from uwos.pattern_analysis_v2.core import parse_args
from uwos.pattern_analysis_v2 import core as pattern_v2
from uwos.options_pattern_pipeline_v1 import core as shared_engine


PRIMARY_FILES = (
    "stock-screener",
    "hot-chains",
    "chain-oi-changes",
    "dp-eod-report",
    "bot-eod-report",
)


def _write_primary_sources(day: Path, omitted: str = "") -> None:
    day.mkdir(parents=True)
    for prefix in PRIMARY_FILES:
        if prefix != omitted:
            (day / f"{prefix}-{day.name}.csv").write_text("ticker\n", encoding="utf-8")


def test_pattern_analysis_v2_defaults_to_v2_output_namespace(tmp_path):
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
        ]
    )

    assert PIPELINE_VERSION == "pattern_analysis_v2.12-family-sources-symmetric-momentum-20260803"
    assert Path(args.out_dir) == tmp_path / "out" / "pattern_analysis_v2" / "2026-05-18"
    assert args.validation_top_candidates_per_day == 500


def test_pattern_analysis_v2_respects_explicit_out_dir(tmp_path):
    explicit = tmp_path / "custom"
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
            "--out-dir",
            str(explicit),
        ]
    )

    assert Path(args.out_dir) == explicit


def test_pattern_analysis_v2_restores_shared_engine_version(monkeypatch):
    original = shared_engine.PIPELINE_VERSION
    observed = []

    def fake_run_pipeline(args):
        observed.append(shared_engine.PIPELINE_VERSION)
        return "done"

    monkeypatch.setattr(shared_engine, "run_pipeline", fake_run_pipeline)

    assert pattern_v2.run_pipeline(object()) == "done"
    assert observed == [pattern_v2.PIPELINE_VERSION]
    assert shared_engine.PIPELINE_VERSION == original


def test_pattern_analysis_v2_latest_fails_cleanly_without_source_complete_dates(tmp_path):
    with pytest.raises(ValueError, match="no source-complete UW dates"):
        parse_args(["--base-dir", str(tmp_path), "--as-of", "latest"])


@pytest.mark.parametrize("omitted", ["stock-screener", "hot-chains", "chain-oi-changes"])
def test_pattern_analysis_v2_requires_core_execution_sources(tmp_path, omitted):
    incomplete = tmp_path / "2026-05-18"
    complete = tmp_path / "2026-05-19"
    _write_primary_sources(incomplete, omitted=omitted)
    _write_primary_sources(complete)

    args = parse_args(["--base-dir", str(tmp_path), "--as-of", "latest"])

    assert args.as_of == "2026-05-19"
    completeness = pattern_v2.source_completeness_for_date(tmp_path, "2026-05-18")
    assert completeness["source_complete"] is False
    assert any(omitted in reason for reason in completeness["missing_sources"])


@pytest.mark.parametrize("omitted", ["bot-eod-report", "dp-eod-report"])
def test_pattern_analysis_v2_keeps_date_missing_optional_family_source(tmp_path, omitted):
    day = tmp_path / "2026-05-19"
    _write_primary_sources(day, omitted=omitted)

    args = parse_args(["--base-dir", str(tmp_path), "--as-of", "latest"])
    completeness = pattern_v2.source_completeness_for_date(tmp_path, "2026-05-19")

    assert args.as_of == "2026-05-19"
    assert completeness["source_complete"] is True
    assert completeness["all_five_present"] is False


def test_pattern_analysis_v2_restores_source_date_selector(monkeypatch):
    original = shared_engine.source_complete_dates
    original_completeness = shared_engine.source_completeness_for_date
    original_loader = shared_engine.load_risk_config
    monkeypatch.setattr(shared_engine, "run_pipeline", lambda args: "done")

    assert pattern_v2.run_pipeline(object()) == "done"
    assert shared_engine.source_complete_dates is original
    assert shared_engine.source_completeness_for_date is original_completeness
    assert shared_engine.load_risk_config is original_loader


def test_pattern_analysis_v2_completeness_wrapper_does_not_recurse(tmp_path):
    day = tmp_path / "2026-05-19"
    _write_primary_sources(day)
    original = shared_engine.source_completeness_for_date
    shared_engine.source_completeness_for_date = pattern_v2.source_completeness_for_date
    try:
        completeness = pattern_v2.source_completeness_for_date(tmp_path, "2026-05-19")
    finally:
        shared_engine.source_completeness_for_date = original

    assert completeness["source_complete"] is True


def test_pattern_analysis_v2_applies_managed_lifecycle(monkeypatch):
    observed = {}

    def fake_load_risk_config(config_arg, base_dir):
        return None, dict(shared_engine.DEFAULT_RISK_CONFIG), "old"

    def fake_run_pipeline(args):
        _, config, _ = shared_engine.load_risk_config(None, Path("."))
        observed.update(config)
        return "done"

    monkeypatch.setattr(shared_engine, "load_risk_config", fake_load_risk_config)
    monkeypatch.setattr(shared_engine, "run_pipeline", fake_run_pipeline)

    assert pattern_v2.run_pipeline(object()) == "done"
    assert observed["validation_horizon_sessions"] == 40
    assert observed["long_option_profit_target_pct"] == 0.50
    assert observed["long_option_stop_loss_pct"] is None
    assert observed["bot_eod_quote_policy"] == "refresh_existing"
