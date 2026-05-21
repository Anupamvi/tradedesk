from codexuw import engine
from codexuw import opportunity
from codexuw import daily_v4
from codexuw.pipeline_versions import PIPELINE_VERSION_LOCKS, pipeline_version_record


def test_codex_daily_pipeline_versions_are_locked() -> None:
    expected = {
        "v2": ("Codex Daily V2", "v2.1"),
        "v3": ("Codex Daily V3", "v3.0"),
        "v4": ("Codex Daily V4", "v4.0"),
    }

    for key, (name, version) in expected.items():
        record = pipeline_version_record(key)
        assert record["pipeline_name"] == name
        assert record["pipeline_version"] == version
        assert record["lock_status"] == "locked"
        assert record["locked_on"] == "2026-05-21"


def test_locked_versions_mark_schwab_live_portfolio_and_gex_context() -> None:
    for record in PIPELINE_VERSION_LOCKS.values():
        assert record["live_schwab_required_for_execute"] is True
        assert record["portfolio_state_required_for_execute"] is True
        assert "gex" in record["gex_context"].lower() or "gamma" in record["gex_context"].lower()


def test_pipeline_constants_match_locked_registry() -> None:
    assert engine.PIPELINE_NAME == "Codex Daily V2"
    assert engine.PIPELINE_VERSION == "v2.1"
    assert opportunity.PIPELINE_NAME_V3 == "Codex Daily V3"
    assert opportunity.PIPELINE_VERSION_V3 == "v3.0"
    assert daily_v4.PIPELINE_NAME_V4 == "Codex Daily V4"
    assert daily_v4.PIPELINE_VERSION_V4 == "v4.0"
