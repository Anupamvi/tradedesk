from claude_pipeline import PIPELINE_VERSION, __version__


def test_claude_pipeline_bootstrap_version_is_explicit() -> None:
    assert PIPELINE_VERSION == "claude-pipeline-v0.1-schwab-bootstrap-20260808"
    assert __version__ == "0.1.0"