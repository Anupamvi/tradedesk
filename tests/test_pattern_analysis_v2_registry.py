from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from uwos.pattern_analysis_v2.core import register_supplemental_artifacts
from uwos.pattern_analysis_v2.external_context import build_external_context
from uwos.pattern_analysis_v2.research_registry import build_pattern_registry
from uwos.pattern_analysis_v2.source_audit import build_primary_source_audit


PRIMARY_PREFIXES = (
    "stock-screener",
    "hot-chains",
    "chain-oi-changes",
    "dp-eod-report",
    "bot-eod-report",
)


def test_primary_source_audit_lists_every_date_and_all_five_gate(tmp_path: Path) -> None:
    complete = tmp_path / "2026-07-28"
    partial = tmp_path / "2026-07-29"
    missing_core = tmp_path / "2026-07-30"
    complete.mkdir()
    partial.mkdir()
    missing_core.mkdir()
    for prefix in PRIMARY_PREFIXES:
        (complete / f"{prefix}-2026-07-28.csv").write_text("ticker\n", encoding="utf-8")
        if prefix != "bot-eod-report":
            (partial / f"{prefix}-2026-07-29.csv").write_text("ticker\n", encoding="utf-8")
        if prefix != "chain-oi-changes":
            (missing_core / f"{prefix}-2026-07-30.csv").write_text("ticker\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    paths = build_primary_source_audit(tmp_path, out_dir, "2026-07-30")

    audit = pd.read_csv(paths["primary_source_coverage"])
    assert len(audit) == 3
    assert bool(audit.loc[audit.date.eq("2026-07-28"), "included_by_v2"].iloc[0])
    assert bool(audit.loc[audit.date.eq("2026-07-29"), "included_by_v2"].iloc[0])
    assert not bool(audit.loc[audit.date.eq("2026-07-29"), "all_five_present"].iloc[0])
    assert not bool(audit.loc[audit.date.eq("2026-07-30"), "included_by_v2"].iloc[0])
    summary = json.loads(Path(paths["primary_source_coverage_summary"]).read_text(encoding="utf-8"))
    assert summary["all_five_source_dates"] == 1
    assert summary["core_signal_dates"] == 2
    assert summary["core_dates_missing_optional_source"] == 1
    assert summary["excluded_missing_core_dates"] == 1


def test_registry_promotes_only_core_proven_family(tmp_path: Path) -> None:
    out_dir = tmp_path / "out" / "pattern_analysis_v2" / "2026-07-29"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pattern_family": "GOOD__BULLISH__LONG_OPTION__TECHNOLOGY",
                "base_pattern_family": "GOOD",
                "validation_scored_count": 100,
                "validation_average_net_r": 0.2,
                "validation_profit_factor": 2.0,
                "validation_success_probability": 0.60,
                "positive_validation_splits": 2,
                "validation_split_count": 2,
                "validation_day_clustered_profit_factor_p05": 1.40,
                "matched_null_p_value": 0.01,
                "matched_null_coverage": 0.95,
                "matched_null_median_profit_factor": 0.80,
                "confidence_tier": "PROVEN",
                "deployment_gate_failures": "",
            },
            {
                "pattern_family": "BAD__BEARISH__LONG_OPTION__ENERGY",
                "base_pattern_family": "BAD",
                "validation_scored_count": 100,
                "validation_average_net_r": -0.1,
                "validation_profit_factor": 0.5,
                "validation_success_probability": 0.35,
                "positive_validation_splits": 0,
                "validation_split_count": 2,
            },
        ]
    ).to_csv(out_dir / "discovered_pattern_families.csv", index=False)

    paths = build_pattern_registry(tmp_path, out_dir, "2026-07-29")

    registry = pd.read_csv(paths["pattern_registry"])
    assert registry.iloc[0]["pattern_id"].startswith("GOOD")
    assert registry.iloc[0]["status"] == "DEPLOYMENT_READY"
    assert bool(registry.iloc[0]["deployment_ready"])
    assert registry.iloc[0]["matched_null_p_value"] == 0.01
    assert registry.iloc[0]["clustered_pf_p05"] == 1.40
    assert registry.iloc[0]["trade_pattern_rank"] == 1
    assert registry.iloc[0]["rank_scope"] == "TRADE_PATTERN"
    assert registry.iloc[-1]["pattern_id"].startswith("BAD")
    summary = json.loads(Path(paths["pattern_ranking_summary"]).read_text(encoding="utf-8"))
    assert summary["ranking_is_probability"] is False
    assert summary["deployment_ready_count"] == 1
    assert summary["best_trade_pattern_adequate_sample"]["pattern_id"].startswith("GOOD")


def test_external_context_filters_future_x_and_requires_sec_timestamp_for_veto(tmp_path: Path) -> None:
    as_of = "2026-07-29"
    day = tmp_path / as_of
    x_dir = day / "x_scrapes" / "source"
    x_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"published_at": "2026-07-29T12:00:00Z", "text": "$AAA options setup"},
            {"published_at": "2026-07-30T12:00:00Z", "text": "$AAA future post"},
        ]
    ).to_csv(x_dir / "posts.csv", index=False)
    pd.DataFrame(
        [{"Ticker": "AAA", "Filing Type": "8-K", "Filing Date": "2026-07-29"}]
    ).to_csv(day / "sec-filings-scrape-date-only.csv", index=False)
    pd.DataFrame(
        [{"Ticker": "BBB", "Filing Type": "8-K", "accepted_at": "2026-07-29T15:00:00Z"}]
    ).to_csv(day / "sec-filings-scrape-timestamped.csv", index=False)

    out_dir = tmp_path / "out" / "pattern_analysis_v2" / as_of
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ticker": "AAA", "status": "AUTO_APPROVED"},
            {"ticker": "BBB", "status": "AUTO_APPROVED"},
        ]
    ).to_csv(out_dir / "decision_board.csv", index=False)

    paths = build_external_context(tmp_path, out_dir, as_of)

    audit = pd.read_csv(paths["external_context_audit"])
    x_row = audit[audit["source_type"].eq("X_TWITTER")].iloc[0]
    assert x_row["usable_context_count"] == 1
    assert x_row["future_item_count"] == 1
    ticker = pd.read_csv(paths["external_ticker_context"])
    aaa = ticker[ticker["ticker"].eq("AAA")].iloc[0]
    bbb = ticker[ticker["ticker"].eq("BBB")].iloc[0]
    assert not bool(aaa["external_event_veto"])
    assert bool(bbb["external_event_veto"])
    board = pd.read_csv(paths["decision_board_context"])
    assert board.loc[board.ticker.eq("AAA"), "external_adjusted_status"].iloc[0] == "AUTO_APPROVED"
    assert board.loc[board.ticker.eq("BBB"), "external_adjusted_status"].iloc[0] == "EXTERNAL_EVENT_REVIEW_REQUIRED"
    assert not board["external_can_promote"].astype(bool).any()


def test_registry_never_promotes_hard_blocked_long_vol(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        [
            {
                "pattern_family": "VOL_PREMIUM_EXPANSION__NEUTRAL__LONG_STRANGLE__NO_SECTOR",
                "base_pattern_family": "VOL_PREMIUM_EXPANSION",
                "validation_scored_count": 100,
                "validation_average_net_r": 0.50,
                "validation_profit_factor": 3.0,
                "validation_success_probability": 0.70,
                "positive_validation_splits": 2,
                "validation_split_count": 2,
                "validation_day_clustered_profit_factor_p05": 1.80,
                "matched_null_p_value": 0.01,
                "matched_null_coverage": 1.0,
                "matched_null_median_profit_factor": 0.70,
                "confidence_tier": "PROVEN",
                "deployment_gate_failures": "",
            }
        ]
    ).to_csv(out_dir / "discovered_pattern_families.csv", index=False)

    paths = build_pattern_registry(tmp_path, out_dir, "2026-07-29")
    row = pd.read_csv(paths["pattern_registry"]).iloc[0]

    assert row["status"] == "RESEARCH_ONLY_LONG_VOL_HARD_BLOCK"
    assert not bool(row["deployment_ready"])


def test_supplemental_artifacts_are_hashed_into_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "pattern_registry.csv"
    artifact.write_text("rank,pattern\n1,A\n", encoding="utf-8")
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(json.dumps({"artifact_paths": {"artifact_manifest": str(manifest)}}), encoding="utf-8")

    register_supplemental_artifacts(
        {"artifact_manifest": str(manifest)},
        {"pattern_registry": str(artifact)},
    )

    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["artifact_paths"]["pattern_registry"] == str(artifact)
    record = saved["v2_supplemental_artifacts"]["pattern_registry"]
    assert record["bytes"] == artifact.stat().st_size
    assert len(record["sha256"]) == 64
