import datetime as dt
import importlib.abc
import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd

from uwos.options_agent import audit, core, market_open_runner
from uwos.options_agent.core import RecommendationStatus, apply_portfolio_risk_annotations, output_paths, run_pipeline


def _write_minimal_uw_fixture(root: Path) -> None:
    date_dir = root / "2026-05-22"
    date_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "ticker": "WMT",
                "close": 100.0,
                "call_volume": 1000,
                "put_volume": 1200,
                "call_premium": 500_000,
                "put_premium": 1_500_000,
                "bullish_premium": 5_000_000,
                "bearish_premium": 100_000,
                "marketcap": 650_000_000_000,
                "issue_type": "Common Stock",
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "iv_rank": 60,
                "iv30d": 0.30,
            }
        ]
    ).to_csv(date_dir / "stock-screener-2026-05-22.csv", index=False)
    pd.DataFrame(
        [
            {
                "option_symbol": "WMT260619P00095000",
                "date": "2026-05-22",
                "volume": 5000,
                "open_interest": 20000,
                "premium": 1_000_000,
                "ask_side_volume": 3000,
                "bid_side_volume": 1500,
                "bid": 2.00,
                "ask": 2.20,
            },
            {
                "option_symbol": "WMT260619P00090000",
                "date": "2026-05-22",
                "volume": 2500,
                "open_interest": 15000,
                "premium": 300_000,
                "ask_side_volume": 500,
                "bid_side_volume": 1200,
                "bid": 0.80,
                "ask": 1.00,
            },
        ]
    ).to_csv(date_dir / "hot-chains-2026-05-22.csv", index=False)
    pd.DataFrame(
        [
            {
                "option_symbol": "WMT260619P00095000",
                "underlying_symbol": "WMT",
                "oi_diff_plain": 2500,
                "curr_oi": 20000,
                "volume": 5000,
            }
        ]
    ).to_csv(date_dir / "chain-oi-changes-2026-05-22.csv", index=False)


def _write_wmt_chain_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 100.0,
        "putExpDateMap": {
            "2026-06-19:28": {
                "95.0": [
                    {
                        "symbol": "WMT  260619P00095000",
                        "strikePrice": 95.0,
                        "bid": 1.40,
                        "ask": 1.40,
                        "mark": 1.40,
                        "delta": -0.22,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "90.0": [
                    {
                        "symbol": "WMT  260619P00090000",
                        "strikePrice": 90.0,
                        "bid": 0.40,
                        "ask": 0.40,
                        "mark": 0.40,
                        "delta": -0.10,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
            }
        },
        "callExpDateMap": {},
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_chain_snapshot_in_legacy_chains_layout(snapshot_dir: Path) -> None:
    _write_wmt_chain_snapshot(snapshot_dir)
    payload = json.loads((snapshot_dir / "WMT.json").read_text(encoding="utf-8"))
    (snapshot_dir / "WMT.json").unlink()
    chains_dir = snapshot_dir / "chains"
    chains_dir.mkdir()
    (chains_dir / "chain_WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _mark_strategy_expectancy_pass(frame: pd.DataFrame, tickers=None, sample: int = 5) -> pd.DataFrame:
    out = frame.copy()
    if "ticker" not in out.columns:
        return out
    mask = pd.Series(True, index=out.index)
    if tickers is not None:
        mask = out["ticker"].astype(str).isin(tickers)
    out.loc[mask, "actual_forward_expectancy_status"] = "PASS"
    out.loc[mask, "actual_forward_expectancy_sample_size"] = sample
    out.loc[mask, "actual_forward_strategy_expectancy_status"] = "PASS"
    out.loc[mask, "actual_forward_strategy_expectancy_sample_size"] = sample
    out.loc[mask, "actual_forward_strategy_expectancy_family"] = "vertical_spread"
    out.loc[mask, "actual_forward_strategy_expectancy_scope"] = "ticker_strategy"
    return out


def _write_wmt_wide_market_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 100.0,
        "putExpDateMap": {
            "2026-06-19:28": {
                "95.0": [
                    {
                        "symbol": "WMT  260619P00095000",
                        "strikePrice": 95.0,
                        "bid": 2.00,
                        "ask": 4.00,
                        "mark": 3.00,
                        "delta": -0.22,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "90.0": [
                    {
                        "symbol": "WMT  260619P00090000",
                        "strikePrice": 90.0,
                        "bid": 0.40,
                        "ask": 0.40,
                        "mark": 0.40,
                        "delta": -0.10,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
            }
        },
        "callExpDateMap": {},
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_call_debit_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 100.0,
        "putExpDateMap": {},
        "callExpDateMap": {
            "2026-06-19:28": {
                "100.0": [
                    {
                        "symbol": "WMT  260619C00100000",
                        "strikePrice": 100.0,
                        "bid": 2.00,
                        "ask": 2.00,
                        "mark": 2.00,
                        "delta": 0.55,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "105.0": [
                    {
                        "symbol": "WMT  260619C00105000",
                        "strikePrice": 105.0,
                        "bid": 0.60,
                        "ask": 0.60,
                        "mark": 0.60,
                        "delta": 0.30,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
            }
        },
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_call_debit_with_better_breakout_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 104.0,
        "putExpDateMap": {},
        "callExpDateMap": {
            "2026-06-19:28": {
                "100.0": [
                    {
                        "symbol": "WMT  260619C00100000",
                        "strikePrice": 100.0,
                        "bid": 3.50,
                        "ask": 3.50,
                        "mark": 3.50,
                        "delta": 0.55,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "105.0": [
                    {
                        "symbol": "WMT  260619C00105000",
                        "strikePrice": 105.0,
                        "bid": 0.00,
                        "ask": 0.00,
                        "mark": 0.00,
                        "delta": 0.20,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
                "110.0": [
                    {
                        "symbol": "WMT  260619C00110000",
                        "strikePrice": 110.0,
                        "bid": 1.40,
                        "ask": 1.40,
                        "mark": 1.40,
                        "delta": 0.42,
                        "volatility": 0.33,
                        "openInterest": 6000,
                        "totalVolume": 1400,
                    }
                ],
                "115.0": [
                    {
                        "symbol": "WMT  260619C00115000",
                        "strikePrice": 115.0,
                        "bid": 0.00,
                        "ask": 0.00,
                        "mark": 0.00,
                        "delta": 0.18,
                        "volatility": 0.35,
                        "openInterest": 5500,
                        "totalVolume": 1200,
                    }
                ],
            }
        },
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_red_flag_news(root: Path) -> None:
    news_dir = root / "2026-05-22" / "browser_text"
    news_dir.mkdir(parents=True, exist_ok=True)
    (news_dir / "browser-text-capture-news-WMT-2026-05-22.txt").write_text(
        "WMT faces SEC probe after downgrade warning\nAnalysts cite investigation risk and guidance cut concerns.",
        encoding="utf-8",
    )


def test_options_agent_core_has_no_daily_v4_dependency() -> None:
    source = inspect.getsource(core)

    forbidden = ("codexuw.daily_v4", "daily_v4", "codexdaily_v4")
    for token in forbidden:
        assert token not in source


def test_options_agent_package_has_no_daily_v4_dependency() -> None:
    package_dir = Path(core.__file__).parent
    forbidden = ("codexuw.daily_v4", "daily_v4", "codexdaily_v4", "out/codexdaily")

    for path in package_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} found in {path}"


def test_run_pipeline_does_not_import_daily_v4(tmp_path: Path) -> None:
    class DailyV4Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "codexuw.daily_v4":
                raise AssertionError("Options Agent imported Codex Daily V4")
            return None

    root = tmp_path
    _write_minimal_uw_fixture(root)
    blocker = DailyV4Blocker()
    prior = sys.modules.pop("codexuw.daily_v4", None)
    sys.meta_path.insert(0, blocker)
    try:
        run_pipeline("2026-05-22", root=root, top_trades=3)
    finally:
        sys.meta_path.remove(blocker)
        if prior is not None:
            sys.modules["codexuw.daily_v4"] = prior


def test_default_output_paths_use_options_agent_namespace(tmp_path: Path) -> None:
    paths = output_paths("2026-05-22", root=tmp_path)

    assert paths["out_dir"] == tmp_path / "out" / "options_agent" / "2026-05-22"
    assert paths["manifest"].name == "options_agent_manifest_2026-05-22.json"
    assert paths["report"].name == "options_agent_report_2026-05-22.md"
    assert paths["strategy_outcome_atlas"].name == "strategy_outcome_atlas.csv"
    assert paths["profitability_calibration"].name == "profitability_calibration.csv"
    assert paths["profitability_gap_plan"].name == "profitability_gap_plan.csv"
    assert paths["execution_fill_quality"].name == "execution_fill_quality.csv"


def test_portfolio_risk_annotations_do_not_suppress_qualified_trade() -> None:
    candidate = {
        "ticker": "WMT",
        "structure": "bull put spread",
        "quality_status": "qualified",
        "recommendation_status": "ENTER",
        "hard_rejects": "portfolio_concentration",
    }
    portfolio = {
        "status": "ok",
        "total_value": 100_000,
        "option_underlyings": ["WMT"],
        "large_equity_exposure": {"WMT": 7_500},
    }

    rows = apply_portfolio_risk_annotations([candidate], portfolio)

    assert len(rows) == 1
    row = rows[0]
    assert row["visible_in_final_board"] is True
    assert row["portfolio_risk_flag"] is True
    assert row["recommendation_status"] == RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
    assert row["hard_rejects"] == ""
    assert "existing option exposure in WMT" in row["portfolio_risk_note"]
    assert "large equity exposure in WMT" in row["portfolio_risk_note"]
    assert "portfolio-only blocker converted to annotation" in row["portfolio_risk_note"]
    assert "execution gate unaffected" in row["portfolio_risk_note"]


def test_non_portfolio_hard_blocker_remains_hard_blocker() -> None:
    candidate = {
        "ticker": "HOOD",
        "quality_status": "qualified",
        "recommendation_status": "ENTER",
        "hard_rejects": "bad_liquidity; portfolio_concentration",
    }
    rows = apply_portfolio_risk_annotations([candidate], {"option_underlyings": ["HOOD"]})

    row = rows[0]
    assert row["visible_in_final_board"] is True
    assert row["hard_rejects"] == "bad_liquidity"
    assert row["recommendation_status"] == RecommendationStatus.AVOID.value
    assert "objective hard blocker: bad_liquidity" in row["status_reason"]
    assert "existing option exposure in HOOD" in row["portfolio_risk_note"]


def test_objective_concentration_reject_is_not_misread_as_portfolio_risk() -> None:
    candidate = {
        "ticker": "HOOD",
        "quality_status": "qualified",
        "recommendation_status": "ENTER",
        "hard_rejects": "strike_concentration_bad_fill",
    }
    rows = apply_portfolio_risk_annotations([candidate], {})

    row = rows[0]
    assert row["recommendation_status"] == RecommendationStatus.AVOID.value
    assert row["hard_rejects"] == "strike_concentration_bad_fill"
    assert row["portfolio_risk_flag"] is False


def test_portfolio_risk_does_not_upgrade_waiting_trade_to_enter() -> None:
    candidate = {
        "ticker": "WMT",
        "quality_status": "qualified",
        "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
        "status_reason": "fresh quote required",
    }

    row = apply_portfolio_risk_annotations([candidate], {"option_underlyings": ["WMT"]})[0]

    assert row["recommendation_status"] == RecommendationStatus.WAIT_FOR_PRICE.value
    assert row["portfolio_risk_flag"] is True
    assert row["status_reason"] == "fresh quote required"
    assert "portfolio annotation only" not in row["status_reason"]


def test_candidate_generation_can_include_all_directional_rows() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": f"T{i}",
                "bias": "bullish",
                "score": 70 - i,
                "signal_premium": 1_000_000 + i,
                "quality_status": "qualified",
                "flow_reason": f"reason {i}",
            }
            for i in range(150)
        ]
        + [
            {
                "ticker": "NEUT",
                "bias": "neutral",
                "score": 99,
                "signal_premium": 9_999_999,
                "quality_status": "watch",
                "flow_reason": "neutral",
            }
        ]
    )

    candidates = core.generate_candidates(raw, limit=None, focus_tickers=())

    assert len(candidates) == 150
    assert "NEUT" not in candidates["ticker"].tolist()
    assert candidates["candidate_rank"].iloc[-1] == 150


def test_candidate_generation_rescues_core_neutral_rows_when_price_tape_is_directional() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "bias": "neutral",
                "score": 55,
                "signal_premium": 1_000_000,
                "quality_status": "watch",
                "underlying_quality_tier": "core",
                "marketcap": 500_000_000_000,
                "close": 98.0,
                "prev_close": 100.0,
                "flow_reason": "neutral UW flow",
                "flow_bias_label": "neutral",
            }
            for ticker in ("SPY", "QQQ", "IWM", "DIA", "AAPL")
        ]
    )
    regime = core.build_market_price_regime(raw, "2026-06-09")
    annotated = core.annotate_macro_tape_candidates(raw, regime)

    candidates = core.generate_candidates(annotated, limit=None, focus_tickers=core.CORE_AUDIT_TICKERS)
    aapl = candidates[candidates["ticker"].eq("AAPL")].iloc[0]
    coverage = core.build_coverage_audit(annotated, candidates, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert regime["tape_direction"] == "bearish"
    assert aapl["candidate_source"] == "macro_tape_candidate"
    assert aapl["bias"] == "bearish"
    assert aapl["flow_bias_label"] == "neutral"
    assert coverage.loc[coverage["ticker"].eq("AAPL"), "coverage_status"].tolist() == ["MACRO_TAPE_CANDIDATE"]


def test_candidate_generation_prioritizes_liquid_underlyings_before_speculative_flow() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": "JUNK",
                "bias": "bullish",
                "score": 99,
                "signal_premium": 50_000_000,
                "quality_status": "qualified",
                "underlying_quality_tier": "speculative",
                "flow_reason": "huge flow but weak underlying",
            },
            {
                "ticker": "AAPL",
                "bias": "bullish",
                "score": 70,
                "signal_premium": 2_000_000,
                "quality_status": "qualified",
                "underlying_quality_tier": "core",
                "flow_reason": "liquid large cap",
            },
        ]
    )

    candidates = core.generate_candidates(raw, limit=None, focus_tickers=())

    assert candidates["ticker"].tolist() == ["AAPL", "JUNK"]


def test_agent_dispatch_prompt_lists_all_candidate_tickers(tmp_path: Path) -> None:
    tasks = {
        "tasks": [
            {
                "ticker": f"T{i}",
                "candidate_id": f"T{i}:bullish:70",
                "bias": "bullish",
                "score": 70 - i,
            }
            for i in range(40)
        ]
    }
    paths = output_paths("2026-05-22", root=tmp_path)

    dispatch = core.build_agent_dispatch_plan(tasks, "2026-05-22", paths)
    prompt = dispatch["subagent_tasks"][0]["prompt"]

    assert "T0" in prompt
    assert "T30" in prompt
    assert "T39" in prompt
    assert dispatch["subagent_tasks"][0]["input_task_count"] == 40


def test_trade_quality_gates_reject_junk_setups() -> None:
    rejects = core._trade_quality_rejects(
        entry_credit=0.05,
        credit_width_ratio=0.05,
        max_loss=2_000,
        signal_premium=500_000,
        combined_flow_bias=0.01,
    )

    assert "entry_credit_below_0.25" in rejects
    assert "credit_width_ratio_below_18pct" in rejects
    assert "signal_premium_below_1000000" in rejects
    assert "directional_bias_below_0.10" in rejects
    assert "one_lot_max_loss_above_750" in rejects


def test_trade_tickets_require_executable_live_validated_entry() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "REVIEW",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "target_exit": 0.35,
                "invalidation": "breaks support",
                "live_validation_status": "PASS",
            },
            {
                "ticker": "WAIT",
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "AVOID",
                "recommendation_status": RecommendationStatus.AVOID.value,
                "quality_status": "qualified",
                "hard_rejects": "bad_liquidity",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "BLANK",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "ZERO",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 0.00 CREDIT",
                "entry_limit": 0.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "NOLIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
            },
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "target_exit": 0.35,
                "invalidation": "breaks support",
                "live_validation_status": "PASS",
            },
            {
                "ticker": "RISK",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "live_validation_status": "PASS",
                "portfolio_risk_flag": True,
            },
            {
                "ticker": "NOSIZE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "suggested_contracts": 0,
                "live_validation_status": "PASS",
            },
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final, {"BLANK", "ZERO", "NOLIVE", "LIVE", "RISK", "NOSIZE"})

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    tickets = core.build_trade_tickets(decision)

    assert decision.loc[decision["ticker"].eq("NOLIVE"), "execution_status"].tolist() == ["needs_live_validation"]
    assert decision.loc[decision["ticker"].eq("NOSIZE"), "execution_status"].tolist() == ["needs_sizing"]
    assert decision.loc[decision["ticker"].eq("NOSIZE"), "ready_to_enter"].tolist() == [False]
    assert tickets["ticker"].tolist() == ["LIVE", "RISK"]
    assert "REVIEW" not in tickets["ticker"].tolist()
    assert "WAIT" not in tickets["ticker"].tolist()
    assert "AVOID" not in tickets["ticker"].tolist()
    assert "NOLIVE" not in tickets["ticker"].tolist()
    assert "NOSIZE" not in tickets["ticker"].tolist()
    assert tickets.loc[tickets["ticker"].isin(["LIVE", "RISK"]), "live_validation_status"].tolist() == ["PASS", "PASS"]
    assert tickets.loc[tickets["ticker"].eq("LIVE"), "target_exit"].tolist() == [0.35]
    assert tickets.loc[tickets["ticker"].eq("LIVE"), "invalidation"].tolist() == ["breaks support"]


def test_execution_ready_ticket_requires_run_level_gates() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 3,
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
            }
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final)
    blocked_context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=Path("/tmp/snapshots"),
        portfolio_context={"status": "unavailable", "total_value": 0},
        research_task_count=1,
        external_review_count=0,
        agent_reviews_json=None,
    )
    blocked_decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=blocked_context,
    )

    assert blocked_decision["ready_to_enter"].tolist() == [False]
    assert blocked_decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    blocked_tickets = core.build_trade_tickets(blocked_decision)
    assert blocked_tickets["ready_to_enter"].tolist() == [False]
    assert blocked_tickets["target_order_status"].tolist() == ["target_order_candidate"]

    ready_context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    ready_decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=ready_context,
    )
    tickets = core.build_trade_tickets(ready_decision)
    readiness = core.build_execution_readiness(ready_decision, ready_context)

    assert ready_decision["ready_to_enter"].tolist() == [True]
    assert ready_decision["execution_gate_status"].tolist() == ["pass"]
    assert ready_decision["execution_confidence_rating"].tolist() == ["HIGH"]
    assert core.summarize_execution_readiness(readiness)["status"] == "execution_ready"
    assert tickets["ticker"].tolist() == ["LIVE"]
    assert tickets["position_max_profit"].tolist() == [750.0]
    assert tickets["position_max_loss"].tolist() == [1750.0]


def test_negative_strategy_expectancy_removes_review_row_from_yellow_targets() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BADVERT",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "status_reason": "dated UW target with fresh Schwab chain math",
                "full_ticket": (
                    "SELL 1 BADVERT 2026-06-18 370 Call / "
                    "BUY 1 BADVERT 2026-06-18 372.5 Call @ 0.85 CREDIT"
                ),
                "trade_plan": (
                    "SELL 1 BADVERT 2026-06-18 370 Call / "
                    "BUY 1 BADVERT 2026-06-18 372.5 Call @ 0.85 CREDIT"
                ),
                "entry_limit": 0.85,
                "suggested_contracts": 1,
                "max_profit": 85.0,
                "max_loss": 165.0,
                "credit_width_ratio": 0.34,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE,
                "actual_forward_strategy_expectancy_avg_pnl": -25.0,
                "actual_forward_strategy_expectancy_win_rate": 0.25,
                "actual_forward_strategy_expectancy_profit_factor": 0.5,
                "actual_forward_strategy_expectancy_family": "vertical_spread",
                "profitability_calibration_status": "WARN",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    _, target_tickets = core.split_trade_ticket_surfaces(tickets)

    assert decision["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert core.NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets.empty
    assert target_tickets.empty


def test_execution_fill_quality_audit_blocks_entries_worse_than_target() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "GOOD",
                "trade_plan": "SELL 1 GOOD 2026-07-17 100 Put @ 1.25 CREDIT",
                "entry_limit": 1.25,
                "target_entry": 1.10,
                "live_validation_status": "PASS",
                "live_quote_width_pct": 0.20,
                "live_short_oi": 500,
                "live_short_volume": 50,
            },
            {
                "recommendation_rank": 2,
                "ticker": "BAD",
                "trade_plan": "BUY 1 BAD 2026-07-17 100 Call / SELL 1 BAD 2026-07-17 105 Call @ 2.40 DEBIT",
                "entry_limit": 2.40,
                "target_entry": 1.80,
                "live_validation_status": "PASS",
                "live_quote_width_pct": 0.20,
                "live_short_oi": 500,
                "live_short_volume": 50,
                "live_long_oi": 500,
                "live_long_volume": 50,
            },
        ]
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "trade_plan": "SELL 1 GOOD 2026-07-17 100 Put @ 1.25 CREDIT",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
            },
            {
                "ticker": "BAD",
                "trade_plan": "BUY 1 BAD 2026-07-17 100 Call / SELL 1 BAD 2026-07-17 105 Call @ 2.40 DEBIT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
            },
        ]
    )

    audit = core.build_execution_fill_quality_audit(final, tickets)
    summary = core.summarize_execution_fill_quality(audit)

    assert list(audit.columns) == core.EXECUTION_FILL_QUALITY_COLUMNS
    good = audit[audit["ticker"].eq("GOOD")].iloc[0]
    bad = audit[audit["ticker"].eq("BAD")].iloc[0]
    assert good["action_surface"] == "yellow_target"
    assert good["fill_quality_status"] == "PASS"
    assert good["price_improvement_vs_target"] == 0.15
    assert bad["action_surface"] == "green_send_now"
    assert bad["fill_quality_status"] == "BLOCK"
    assert bad["slippage_vs_target"] == 0.6
    assert "debit_above_target" in bad["reason"]
    assert summary["status"] == "blocked_green_fill_quality"
    assert summary["green_block_rows"] == 1


def test_green_ticket_requires_strategy_expectancy_annotation() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NOEXP",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NOEXP 2026-06-18 100 Put / BUY 1 NOEXP 2026-06-18 95 Put @ 1.60 CREDIT",
                "trade_plan": "SELL 1 NOEXP 2026-06-18 100 Put / BUY 1 NOEXP 2026-06-18 95 Put @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "suggested_contracts": 5,
                "max_profit": 160.0,
                "max_loss": 340.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets["ticker"].tolist() == ["NOEXP"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["order_readiness"].tolist() == ["target_order_after_expectancy_evidence"]


def test_green_ticket_requires_material_position_profit() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "TOY",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 TOY 2026-06-18 100 Put / BUY 1 TOY 2026-06-18 95 Put @ 1.00 CREDIT",
                "trade_plan": "SELL 1 TOY 2026-06-18 100 Put / BUY 1 TOY 2026-06-18 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "suggested_contracts": 5,
                "max_profit": 100.0,
                "max_loss": 400.0,
                "credit_width_ratio": 0.2,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "REAL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 REAL 2026-06-18 100 Put / BUY 1 REAL 2026-06-18 95 Put @ 1.60 CREDIT",
                "trade_plan": "SELL 1 REAL 2026-06-18 100 Put / BUY 1 REAL 2026-06-18 95 Put @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "suggested_contracts": 5,
                "max_profit": 160.0,
                "max_loss": 340.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    ready, target = core.split_trade_ticket_surfaces(tickets)

    toy = decision[decision["ticker"].eq("TOY")].iloc[0]
    real = decision[decision["ticker"].eq("REAL")].iloc[0]
    assert bool(toy["ready_to_enter"]) is False
    assert core.POSITION_PROFIT_MATERIALITY_BLOCKER in toy["execution_blockers"]
    assert toy["status_label"] == "YELLOW target"
    assert bool(real["ready_to_enter"]) is True
    assert ready["ticker"].tolist() == ["REAL"]
    assert target["ticker"].tolist() == ["TOY"]
    assert target["order_readiness"].tolist() == ["target_order_profit_floor"]
    assert target["action"].tolist() == ["work_target_only_if_profit_floor_clears"]


def test_send_now_requires_strong_credit_and_trade_quality() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "THINCREDIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 0.65 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "suggested_contracts": 4,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "credit_width_ratio": 0.24,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "LOWQUALITY",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 X / SELL 1 Y @ 3.30 DEBIT",
                "trade_plan": "BUY 1 X / SELL 1 Y @ 3.30 DEBIT",
                "entry_limit": 3.3,
                "suggested_contracts": 2,
                "max_profit": 670.0,
                "max_loss": 330.0,
                "credit_width_ratio": 0.0,
                "trade_quality_status": "reviewable",
                "quality_gate_reason": "manual_quality_warning",
                "live_validation_status": "PASS",
                "agent_caution_count": 10,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "NARROWWIDTHGOOD",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False, False, True]
    assert decision["execution_gate_status"].tolist() == ["blocked", "blocked", "pass"]
    assert "send_now_credit_below_0.50" not in decision["execution_blockers"].iloc[0]
    assert "send_now_credit_width_below_30pct" in decision["execution_blockers"].iloc[0]
    assert decision.loc[decision["ticker"].eq("LOWQUALITY"), "target_order_status"].tolist() == [
        "review_only_low_trade_quality"
    ]
    assert decision.loc[decision["ticker"].eq("NARROWWIDTHGOOD"), "target_order_status"].tolist() == [
        "target_order_candidate"
    ]
    assert "NARROWWIDTHGOOD" in tickets[tickets["ready_to_enter"].map(bool)]["ticker"].tolist()
    assert "LOWQUALITY" not in tickets[tickets["ready_to_enter"].map(bool)]["ticker"].tolist()
    assert core._coverage_next_step("REVIEW_TICKET", decision.iloc[0]) == (
        "reprice in Schwab and resolve catalyst/quality review"
    )
    assert core._coverage_next_step("REVIEW_TICKET", decision.iloc[1]) == (
        "reprice in Schwab and resolve trade-quality review"
    )


def test_coverage_next_step_keeps_portfolio_notes_out_of_visible_action_text() -> None:
    plain_review = {"execution_blockers": "", "portfolio_risk_note": "", "requires_portfolio_ack": False}
    portfolio_review = {
        "execution_blockers": "",
        "portfolio_risk_note": "existing option exposure in AAPL",
        "requires_portfolio_ack": False,
    }
    portfolio_context_review = {
        "execution_blockers": "portfolio_context_required",
        "portfolio_risk_note": "",
        "requires_portfolio_ack": False,
    }

    assert "portfolio" not in core._coverage_next_step("REVIEW_TICKET", plain_review)
    assert core._coverage_next_step("REVIEW_TICKET", portfolio_review) == (
        "reprice in Schwab and resolve catalyst/quality review"
    )
    assert core._coverage_next_step("REVIEW_TICKET", portfolio_context_review) == (
        "refresh portfolio context before manual entry"
    )


def test_market_session_gate_respects_us_equity_holidays() -> None:
    holiday_noon = dt.datetime(2026, 5, 25, 10, 0, tzinfo=core.MARKET_TIME_ZONE)
    next_session = core.next_regular_market_session_start(
        dt.datetime(2026, 5, 24, 12, 0, tzinfo=core.MARKET_TIME_ZONE)
    )

    assert core.is_regular_market_day(dt.date(2026, 5, 25)) is False
    assert core.is_regular_market_session_open(holiday_noon) is False
    assert next_session == dt.datetime(2026, 5, 26, 6, 30, tzinfo=core.MARKET_TIME_ZONE)
    assert core.is_regular_market_session_open(dt.datetime(2026, 5, 26, 7, 0, tzinfo=core.MARKET_TIME_ZONE)) is True


def test_portfolio_risk_annotation_does_not_reduce_position_sizing() -> None:
    sized = core.apply_position_sizing(
        [
            {
                "ticker": "RISK",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "live_validation_status": "PASS",
                "max_loss": 200.0,
                "portfolio_risk_flag": True,
                "portfolio_risk_note": "concentration note only",
            }
        ],
        {"status": "ok", "total_value": 100_000.0, "cash": 100_000.0},
        {"sizing_stance": "normal"},
    )

    row = sized[0]
    assert row["suggested_contracts"] == 2
    assert row["risk_budget"] == 500.0
    assert row["max_position_loss"] == 400.0
    assert row["sizing_risk_flag"] is False
    assert "sizing uses the explicit risk budget" in row["sizing_note"]
    assert "portfolio annotation only" not in row["sizing_note"]
    assert row["portfolio_risk_note"] == "concentration note only"


def test_monthly_feasibility_uses_sized_position_max_profit() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GREEN",
                "ready_to_enter": True,
                "target_order_status": "",
                "max_profit": 100.0,
                "max_loss": 250.0,
                "suggested_contracts": 3,
            },
            {
                "ticker": "YELLOW",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "max_profit": 50.0,
                "max_loss": 150.0,
                "suggested_contracts": 4,
            },
        ]
    )

    monthly = core.build_monthly_feasibility(
        decision_board=pd.DataFrame(),
        trade_tickets=tickets,
        execution_context={
            "monthly_profit_target": 10_000.0,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        expectancy_evidence=pd.DataFrame(),
    )

    assert monthly.loc[monthly["metric"].eq("one_cycle_max_profit"), "value"].tolist() == [300.0]
    assert monthly.loc[monthly["metric"].eq("one_cycle_max_loss"), "value"].tolist() == [750.0]
    assert monthly.loc[monthly["metric"].eq("target_order_candidate_max_profit"), "value"].tolist() == [200.0]
    assert monthly.loc[monthly["metric"].eq("target_order_candidate_max_loss"), "value"].tolist() == [600.0]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "status"].tolist() == ["BLOCK"]


def test_monthly_feasibility_requires_expectancy_for_green_ticket_tickers() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GREEN",
                "ready_to_enter": True,
                "target_order_status": "",
                "max_profit": 3000.0,
                "max_loss": 1000.0,
                "suggested_contracts": 3,
            },
            {
                "ticker": "YELLOW",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "max_profit": 500.0,
                "max_loss": 500.0,
                "suggested_contracts": 5,
            },
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "YELLOW",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "YELLOW",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
                "note": "Broad current-ticket support is positive for a non-green ticker.",
            },
        ]
    )

    monthly = core.build_monthly_feasibility(
        decision_board=pd.DataFrame(),
        trade_tickets=tickets,
        execution_context={
            "monthly_profit_target": 10_000.0,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        expectancy_evidence=expectancy,
    )
    summary = core.summarize_monthly_feasibility(monthly)

    assert monthly.loc[monthly["metric"].eq("expectancy_evidence"), "status"].tolist() == ["PASS"]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "status"].tolist() == ["BLOCK"]
    assert "GREEN" in monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "note"].iloc[0]
    assert summary["status"] == "not_proven"
    assert "ready_ticket_expectancy_evidence" in summary["blocking_metrics"]


def test_monthly_feasibility_passes_green_ticket_expectancy_when_ready_tickers_supported() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GREEN",
                "ready_to_enter": True,
                "target_order_status": "",
                "max_profit": 3000.0,
                "max_loss": 1000.0,
                "suggested_contracts": 5,
            },
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "GREEN",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "GREEN",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
                "note": "Actual closed/forward outcomes are positive for current green ticker.",
            },
        ]
    )

    monthly = core.build_monthly_feasibility(
        decision_board=pd.DataFrame(),
        trade_tickets=tickets,
        execution_context={
            "monthly_profit_target": 10_000.0,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        expectancy_evidence=expectancy,
    )

    assert monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "status"].tolist() == ["PASS"]


def test_send_now_green_requires_positive_structure_aligned_actual_forward_support() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AMAT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "BLOCK",
                "actual_forward_expectancy_sample_size": 0,
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
            },
            {
                "ticker": "GOOGL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 400 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 400 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 14,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 14,
            },
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision.loc[decision["ticker"].eq("AMAT"), "ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision.loc[
        decision["ticker"].eq("AMAT"), "execution_blockers"
    ].iloc[0]
    assert decision.loc[decision["ticker"].eq("GOOGL"), "ready_to_enter"].tolist() == [True]
    assert tickets.loc[tickets["ticker"].eq("AMAT"), "order_readiness"].tolist() == [
        "target_order_after_expectancy_evidence"
    ]
    assert tickets.loc[tickets["ticker"].eq("AMAT"), "ready_to_enter"].tolist() == [False]
    assert tickets.loc[tickets["ticker"].eq("GOOGL"), "ready_to_enter"].tolist() == [True]


def test_closed_market_is_informational_and_does_not_block_target_ticket() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.00 CREDIT",
                "trade_plan": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.2,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "agent_support_count": 4,
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    closed_context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=4,
        external_review_count=4,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=False,
    )

    decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=closed_context,
    )
    tickets = core.build_trade_tickets(decision)
    readiness = core.build_execution_readiness(decision, closed_context)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["waiting_for_price"]
    assert "market_session_open_required" not in decision["execution_blockers"].iloc[0]
    assert "regular_session_quote_refresh_required" not in decision["execution_blockers"].iloc[0]
    assert "send_now_credit_width_below_30pct" in decision["execution_blockers"].iloc[0]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["order_readiness"].tolist() == ["target_order_price_validation"]
    assert tickets["action"].tolist() == ["work_target_limit"]
    assert "use the shown target limit" in core._ticket_next_step(tickets.iloc[0])
    coverage = core.build_coverage_audit(
        raw_universe=pd.DataFrame(),
        candidates=pd.DataFrame(),
        priced=pd.DataFrame(),
        decision_board=decision,
        no_trade=pd.DataFrame(),
        watchlist=["LIVE"],
    )
    assert coverage["next_step"].tolist() == [
        "use the shown target limit as the starting point; adjust if the live quote moves"
    ]
    quote_freshness = readiness.loc[readiness["gate"].eq("quote_freshness")]
    assert quote_freshness["status"].tolist() == ["INFO"]
    assert "execution_blocker=false" in quote_freshness["detail"].iloc[0]


def test_closed_market_clean_live_row_can_still_be_green_when_other_gates_pass() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.60 CREDIT",
                "trade_plan": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "suggested_contracts": 5,
                "max_profit": 160.0,
                "max_loss": 340.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=False,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [True]
    assert decision["execution_status"].tolist() == ["ready"]
    assert "market_session_open_required" not in decision["execution_blockers"].iloc[0]
    assert decision["execution_blockers"].tolist() == [""]
    assert tickets["ready_to_enter"].tolist() == [True]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["order_readiness"].tolist() == ["ready_to_enter"]


def test_market_open_recheck_queue_includes_only_market_session_only_targets() -> None:
    tickets = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "SESSION",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "execution_blockers": "market_session_open_required",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "SELL 1 SESSION 2026-06-05 100 Call / BUY 1 SESSION 2026-06-05 105 Call @ 0.65 CREDIT",
            },
            {
                "recommendation_rank": 2,
                "ticker": "FRESH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_live_recheck",
                "execution_blockers": "fresh_live_schwab_required",
                "entry_type": "DEBIT",
                "entry_limit": 0.65,
                "max_profit": 185.0,
                "max_loss": 65.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "BUY 1 FRESH 2026-06-05 100 Call / SELL 1 FRESH 2026-06-05 105 Call @ 0.65 DEBIT",
            },
            {
                "recommendation_rank": 3,
                "ticker": "PORT",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "execution_blockers": "market_session_open_required; portfolio_context_required",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "SELL 1 PORT 2026-06-05 100 Call / BUY 1 PORT 2026-06-05 105 Call @ 0.65 CREDIT",
            },
            {
                "recommendation_rank": 4,
                "ticker": "READY",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "order_readiness": "ready_to_enter",
                "execution_blockers": "",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "SELL 1 READY 2026-06-05 100 Call / BUY 1 READY 2026-06-05 105 Call @ 0.65 CREDIT",
            },
        ]
    )

    queue = core.build_market_open_recheck_queue(tickets)

    assert queue["ticker"].tolist() == ["FRESH", "SESSION"]
    assert queue["required_recheck"].str.contains("fresh Schwab quote").tolist() == [True, True]
    assert queue["recheck_action"].str.contains("ready_to_enter=true").tolist() == [True, True]


def test_market_session_only_targets_are_yellow_until_ready() -> None:
    row = {
        "ready_to_enter": False,
        "target_order_status": "target_order_candidate",
        "execution_blockers": "market_session_open_required",
        "trade_plan": "SELL 1 SESSION 2026-06-05 100 Call / BUY 1 SESSION 2026-06-05 105 Call @ 0.65 CREDIT",
    }

    assert core._decision_badge(row) == "🟡 YELLOW target"
    assert core._decision_icon(row) == "🟡"
    assert core._decision_status_label(row) == "YELLOW target"
    assert core._ticket_order_readiness(row) == "target_order_price_validation"
    assert core._ticket_action(row) == "work_target_limit"
    assert "shown target limit" in core._ticket_next_step(row)


def test_actionability_proof_fails_green_labeled_non_ready_targets() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "SESSION",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "status_icon": "🟢",
                "status_label": "GREEN target",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "trade_plan": "SELL 1 SESSION 2026-06-05 100 Call / BUY 1 SESSION 2026-06-05 105 Call @ 0.65 CREDIT",
                "sell_leg": "SELL 1 SESSION 2026-06-05 100 Call",
                "buy_leg": "BUY 1 SESSION 2026-06-05 105 Call",
            }
        ]
    )
    green_proof = pd.DataFrame(
        [{"green_ticket_rows": 0, "valid_green_ticket_rows": 0, "invalid_green_ticket_rows": 0}]
    )

    packet = audit.build_actionability_surface_proof_packet(
        tickets=tickets,
        green_ticket_execution_proof=green_proof,
        market_open_recheck_queue=pd.DataFrame(),
    )

    assert packet["status"].tolist() == ["FAIL_ACTIONABILITY_SURFACE_INTEGRITY"]
    assert packet["target_green_label_rows"].tolist() == [1]
    assert packet["target_green_icon_rows"].tolist() == [1]


def test_trade_ticket_surfaces_sort_by_confidence() -> None:
    decision = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "LOWER",
                "trade_plan": "SELL 1 LOWER 2026-06-05 100 Call / BUY 1 LOWER 2026-06-05 105 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_gate_status": "blocked",
                "execution_blockers": "market_session_open_required",
                "target_order_status": "target_order_candidate",
                "execution_confidence_score": 75,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 95,
                "suggested_contracts": 5,
                "max_profit": 65.0,
                "max_loss": 185.0,
            },
            {
                "recommendation_rank": 2,
                "ticker": "TOP",
                "trade_plan": "SELL 1 TOP 2026-06-05 100 Call / BUY 1 TOP 2026-06-05 105 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_gate_status": "blocked",
                "execution_blockers": "market_session_open_required",
                "target_order_status": "target_order_candidate",
                "execution_confidence_score": 88,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 50,
                "suggested_contracts": 1,
                "max_profit": 65.0,
                "max_loss": 185.0,
            },
            {
                "recommendation_rank": 3,
                "ticker": "TIEHIGH",
                "trade_plan": "SELL 1 TIEHIGH 2026-06-05 100 Call / BUY 1 TIEHIGH 2026-06-05 105 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_gate_status": "blocked",
                "execution_blockers": "market_session_open_required",
                "target_order_status": "target_order_candidate",
                "execution_confidence_score": 75,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 10,
                "suggested_contracts": 1,
                "max_profit": 65.0,
                "max_loss": 185.0,
            },
        ]
    )

    tickets = core.build_trade_tickets(decision)
    _, target = core.split_trade_ticket_surfaces(tickets)
    queue = core.build_market_open_recheck_queue(tickets)

    assert tickets["ticker"].tolist() == ["TOP", "TIEHIGH", "LOWER"]
    assert target["ticker"].tolist() == ["TOP", "TIEHIGH", "LOWER"]
    assert queue["ticker"].tolist() == ["TOP", "TIEHIGH", "LOWER"]

    mixed_readiness = pd.DataFrame(
        [
            {
                "ticker": "LOW_PRICE_REFRESH",
                "ready_to_enter": False,
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 35,
                "trade_quality_confidence_rating": "LOW",
                "execution_confidence_rating": "LOW",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 90,
                "recommendation_rank": 1,
            },
            {
                "ticker": "HIGH_PROFIT_FLOOR",
                "ready_to_enter": False,
                "order_readiness": "target_order_profit_floor",
                "execution_blockers": core.POSITION_PROFIT_MATERIALITY_BLOCKER,
                "execution_confidence_score": 76,
                "trade_quality_confidence_rating": "LOW",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 10,
                "recommendation_rank": 2,
            },
        ]
    )
    sorted_mixed = core._sort_trades_by_confidence(mixed_readiness)
    assert sorted_mixed["ticker"].tolist() == ["HIGH_PROFIT_FLOOR", "LOW_PRICE_REFRESH"]


def test_ready_trade_tickets_sort_by_confidence_before_expectancy_status() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "PASSLOW",
                "ready_to_enter": True,
                "target_order_status": "ready_to_enter",
                "order_readiness": "ready_to_enter",
                "execution_confidence_score": 70,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
                "actual_forward_strategy_expectancy_status": "PASS",
            },
            {
                "ticker": "BLOCKHIGH",
                "ready_to_enter": True,
                "target_order_status": "ready_to_enter",
                "order_readiness": "ready_to_enter",
                "execution_confidence_score": 95,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 10,
                "recommendation_rank": 2,
                "actual_forward_strategy_expectancy_status": "BLOCK",
            },
        ]
    )

    ready, _ = core.split_trade_ticket_surfaces(tickets)

    assert ready["ticker"].tolist() == ["BLOCKHIGH", "PASSLOW"]


def test_trade_tickets_keep_green_before_higher_confidence_yellow() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "YELLOWHIGH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 99,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
            },
            {
                "ticker": "GREENLOW",
                "ready_to_enter": True,
                "target_order_status": "ready_to_enter",
                "order_readiness": "ready_to_enter",
                "execution_blockers": "",
                "execution_confidence_score": 70,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 50,
                "recommendation_rank": 2,
            },
        ]
    )

    sorted_tickets = core._sort_trades_by_confidence(tickets)

    assert sorted_tickets["ticker"].tolist() == ["GREENLOW", "YELLOWHIGH"]


def test_credit_direction_uses_option_legs_when_bias_is_missing() -> None:
    put_row = {
        "bias": "",
        "trade_plan": "SELL 1 SPY 2026-06-30 575 Put / BUY 1 SPY 2026-06-30 570 Put @ 1.20 CREDIT",
    }
    call_row = {
        "bias": "",
        "trade_plan": "SELL 1 SPY 2026-06-30 625 Call / BUY 1 SPY 2026-06-30 630 Call @ 1.20 CREDIT",
    }

    assert core._credit_direction(put_row) == "Bull Put"
    assert core._credit_direction(call_row) == "Bear Call"


def test_target_trade_tickets_sort_by_confidence_before_materiality_bucket() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "TINYHIGH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_profit_floor",
                "execution_blockers": core.POSITION_PROFIT_MATERIALITY_BLOCKER,
                "execution_confidence_score": 96,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
            },
            {
                "ticker": "CLEANLOW",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 78,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 50,
                "recommendation_rank": 2,
            },
        ]
    )

    _, target = core.split_trade_ticket_surfaces(tickets)

    assert target["ticker"].tolist() == ["TINYHIGH", "CLEANLOW"]


def test_market_closed_live_recheck_preserves_agentic_target_queue() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "trade_quality_status": "reviewable",
                "full_ticket": "SELL 1 AAPL 2026-06-18 200 Put / BUY 1 AAPL 2026-06-18 195 Put @ 1.50 CREDIT",
                "trade_plan": "SELL 1 AAPL 2026-06-18 200 Put / BUY 1 AAPL 2026-06-18 195 Put @ 1.50 CREDIT",
                "expiry": "2026-06-18",
                "sell_leg": "SELL 1 AAPL 2026-06-18 200 Put",
                "buy_leg": "BUY 1 AAPL 2026-06-18 195 Put",
                "entry_limit": 1.5,
                "target_exit": 0.52,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "suggested_contracts": 2,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
                "agent_support_count": 5,
                "agent_caution_count": 0,
                "agent_objective_blocker_count": 0,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock",
                "live_validation_status": "MARKET_CLOSED_RECHECK",
                "status_reason": "dated UW EOD quote; refresh Schwab chain before entry",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=4,
        external_review_agent_count=4,
        agent_dispatch_task_count=4,
        agent_reviews_json=Path("/tmp/agentic_reviews.json"),
        market_session_open=False,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    queue = core.build_market_open_recheck_queue(tickets)

    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert decision["execution_blockers"].tolist() == [""]
    assert tickets["order_readiness"].tolist() == ["target_order_price_validation"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert queue.empty


def test_market_open_recheck_proof_blocks_incomplete_rows() -> None:
    queue = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "validation_lane": "live_readiness_probe",
                "source_kind": "live_probe",
                "source_dir": "/tmp/live",
                "ticker": "MISS",
                "entry_type": "CREDIT",
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "entry_limit": 0.0,
                "target_exit": 0.1,
                "max_profit": 50,
                "max_loss": 200,
                "suggested_contracts": 0,
                "execution_confidence_score": 60,
                "trade_quality_confidence_rating": "LOW",
                "external_agent_distinct_review_count": 1,
                "trade_plan": "SELL 1 MISS 2026-06-05 100 Call",
                "required_recheck": "regular_market_session_open",
                "recheck_action": "rerun",
                "execution_blockers": "market_session_open_required; portfolio_context_required",
            }
        ],
        columns=audit.MARKET_QUEUE_AUDIT_COLUMNS,
    )

    details = audit.build_market_open_recheck_details(queue)
    packet = audit.build_market_open_recheck_proof_packet(details)

    assert packet["status"].tolist() == ["FAIL_MARKET_OPEN_RECHECK_ROWS_INCOMPLETE"]
    assert packet["row_fail_rows"].tolist() == [1]
    assert "entry_limit_not_positive" in details["fail_reasons"].iloc[0]
    assert "blockers_not_only_market_session_or_fresh_live_required" in details["fail_reasons"].iloc[0]


def test_session_only_green_shadow_proof_uses_position_scaled_fallbacks() -> None:
    details = pd.DataFrame(
        [
            {
                "source_kind": "live_probe",
                "ticker": "ALPHA",
                "entry_type": "CREDIT",
                "max_profit": 100.0,
                "max_loss": 300.0,
                "position_max_profit": pd.NA,
                "position_max_loss": pd.NA,
                "suggested_contracts": 2,
                "only_market_session_blocker": True,
                "row_pass": True,
                "fail_reasons": "",
            },
            {
                "source_kind": "live_probe",
                "ticker": "BETA",
                "entry_type": "DEBIT",
                "max_profit": 400.0,
                "max_loss": 700.0,
                "position_max_profit": 750.0,
                "position_max_loss": 500.0,
                "suggested_contracts": 1,
                "only_market_session_blocker": True,
                "row_pass": True,
                "fail_reasons": "",
            },
        ]
    )

    packet = audit.build_session_only_green_shadow_proof_packet(details)

    assert packet["status"].tolist() == ["PASS_SESSION_ONLY_GREEN_SHADOW_READY"]
    assert packet["shadow_candidate_rows"].tolist() == [2]
    assert packet["row_fail_rows"].tolist() == [0]
    assert packet["non_session_blocker_rows"].tolist() == [0]
    assert packet["credit_rows"].tolist() == [1]
    assert packet["debit_rows"].tolist() == [1]
    assert packet["position_max_profit"].tolist() == [950.0]
    assert packet["position_max_loss"].tolist() == [1100.0]
    assert packet["tickers"].tolist() == ["ALPHA, BETA"]
    assert "not execution permission" in packet["note"].iloc[0]


def test_live_rerun_preflight_blocks_mismatched_agent_reviews(tmp_path: Path) -> None:
    day = "2026-05-22"
    day_dir = tmp_path / day
    day_dir.mkdir()
    (day_dir / f"stock-screener-{day}.csv").write_text("ticker\nAAPL\n", encoding="utf-8")
    (day_dir / f"hot-chains-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")
    (day_dir / f"chain-oi-changes-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")

    reviews_json = tmp_path / "wrong_agentic_reviews.json"
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": ticker,
                        "agent": agent,
                        "verdict": "supportive",
                        "confidence": "high",
                        "note": "fixture review for wrong ticker",
                        "objective_blocker": False,
                    }
                    for ticker in ("X", "Y")
                    for agent in ("catalyst_news", "macro_regime", "structure_builder", "skeptic")
                ]
            }
        ),
        encoding="utf-8",
    )
    execution_packet = pd.DataFrame(
        [
            {
                "date": day,
                "out_dir": str(tmp_path / "fresh_live_rerun"),
                "agent_reviews_json": str(reviews_json),
                "command": (
                    "python3 -m uwos.options_agent "
                    f"--date {day} "
                    f"--base-dir {tmp_path} "
                    f"--out-dir {tmp_path / 'fresh_live_rerun'} "
                    "--live-schwab --live-portfolio "
                    f"--agent-reviews-json {reviews_json}"
                ),
            }
        ]
    )
    recheck_details = pd.DataFrame(
        [
            {"ticker": "A", "row_pass": True},
            {"ticker": "B", "row_pass": True},
        ]
    )

    preflight_details = audit.build_live_rerun_preflight_details(
        market_open_recheck_details=recheck_details,
        market_open_execution_packet=execution_packet,
    )
    preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=recheck_details,
        market_open_execution_packet=execution_packet,
        preflight_details=preflight_details,
    )

    assert preflight_packet["status"].tolist() == ["FAIL_LIVE_RERUN_PREFLIGHT"]
    assert preflight_packet["queue_tickers"].tolist() == ["A, B"]
    assert preflight_packet["covered_queue_ticker_count"].tolist() == [0]
    assert preflight_packet["missing_queue_tickers"].tolist() == ["A, B"]
    assert preflight_packet["agent_review_rows"].tolist() == [8]
    assert preflight_packet["distinct_agent_count"].tolist() == [4]
    assert preflight_packet["rerun_out_dir_clear"].tolist() == [True]
    assert preflight_packet["source_date_available"].tolist() == [True]
    assert "agent_reviews_json_missing_queue_tickers" in preflight_packet["failed_examples"].iloc[0]
    assert preflight_details["row_pass"].tolist() == [False, False]
    assert preflight_details["fail_reasons"].str.contains("ticker_missing_from_agent_reviews_json").all()


def test_target_preservation_counts_live_queue_debit_targets() -> None:
    target_audit = audit.build_target_preservation_audit(
        summary=pd.DataFrame([{"date": "2026-05-21"}, {"date": "2026-05-22"}]),
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "CREDIT",
                    "entry_type": "CREDIT",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "DEBIT",
                    "entry_type": "DEBIT",
                    "source_kind": "live_probe",
                }
            ]
        ),
    )

    assert target_audit.loc[target_audit["metric"].eq("credit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert target_audit.loc[target_audit["metric"].eq("debit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert target_audit.loc[target_audit["metric"].eq("debit_target_rows"), "value"].tolist() == [1]


def test_audit_markdown_tables_blank_nan_values(tmp_path: Path) -> None:
    path = tmp_path / "table.md"

    audit._write_markdown_table(
        path,
        "Fixture Table",
        pd.DataFrame([{"ticker": "NVDA", "rank": pd.NA, "score": float("nan")}]),
    )

    text = path.read_text(encoding="utf-8")
    assert "NVDA" in text
    assert "nan" not in text.lower()


def test_expanded_audit_writes_repeatable_goal_and_live_recheck_artifacts(tmp_path: Path) -> None:
    def write_run(run_dir: Path, day: str, entry_type: str, ticker: str, tier: str = "core", queue_rows=None) -> None:
        run_dir.mkdir(parents=True)
        review_tickers = [ticker]
        if queue_rows:
            review_tickers.extend(str(row.get("ticker", "")).upper() for row in queue_rows)
        review_tickers = sorted({symbol for symbol in review_tickers if symbol})
        review_agents = ["catalyst_news", "macro_regime", "structure_builder", "skeptic"]
        review_rows = [
            {
                "ticker": review_ticker,
                "agent": agent,
                "verdict": "supportive",
                "confidence": "high",
                "note": f"{agent} supports {review_ticker} in fixture",
                "objective_blocker": False,
            }
            for review_ticker in review_tickers
            for agent in review_agents
        ]
        manifest = {
            "as_of": day,
            "pipeline_version": "options_agent.test",
            "live_schwab_requested": run_dir.name.startswith("live_readiness_probe"),
            "chain_snapshot_dir": "" if run_dir.name.startswith("live_readiness_probe") else str(run_dir / "snapshots"),
            "row_counts": {
                "candidate_generation": 2,
                "research_tasks": 2,
                "priced_candidates": 1,
                "final_recommendations": 1,
                "decision_board": 1,
                "trade_tickets": 1,
                "target_order_candidates": 1,
                "no_trade_audit": 1,
                "ready_to_enter": 0,
                "market_open_recheck_queue": len(queue_rows or []),
                "agent_dispatch_tasks": 5,
                "external_agent_reviews": 50,
            },
            "execution_readiness_summary": {
                "status": "not_execution_ready",
                "blocking_gates": ["market_session_open", "ready_trade_tickets"],
            },
            "expectancy_evidence_summary": {
                "status": "not_proven",
                "summary_status": "BLOCK",
                "sample_size": 0,
                "note": "No sufficient positive expectancy evidence is available.",
            },
            "monthly_feasibility_summary": {
                "status": "not_proven",
                "blocking_metrics": ["ready_ticket_count", "expectancy_evidence"],
            },
            "agentic_orchestration": {
                "status": "reviews_ingested",
                "subagent_task_count": 5,
                "ingested_reviews_json": str(run_dir / "agentic_reviews.json"),
            },
            "execution_context": {
                "external_review_count": 50,
                "external_review_agent_count": 5,
                "agent_dispatch_task_count": 5,
                "agentic_review_coverage_basis": "subagent_lanes",
                "agentic_review_coverage_pct": 1.0,
                "agentic_review_lane_coverage_pct": 1.0,
                "broad_review_coverage_pct": 0.0129,
                "fresh_live_quotes_ready": run_dir.name.startswith("live_readiness_probe"),
                "portfolio_ready": run_dir.name.startswith("live_readiness_probe"),
                "market_session_open": False,
            },
        }
        (run_dir / f"options_agent_manifest_{day}.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "agentic_reviews.json").write_text(json.dumps({"reviews": review_rows}), encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "bias": "bearish",
                    "quality_status": "qualified",
                    "score": 80,
                    "flow_reason": "qualified mock row",
                },
                {
                    "ticker": f"{ticker}W",
                    "bias": "bearish",
                    "quality_status": "watch",
                    "score": 50,
                    "flow_reason": "watch mock row",
                },
            ]
        ).to_csv(run_dir / "candidate_generation.csv", index=False)
        (run_dir / "research_tasks.json").write_text(
            json.dumps({"tasks": [{"ticker": ticker}, {"ticker": f"{ticker}W"}]}),
            encoding="utf-8",
        )
        pd.DataFrame([{"ticker": ticker, "quality_status": "qualified"}]).to_csv(
            run_dir / "priced_candidates.csv",
            index=False,
        )
        pd.DataFrame([{"ticker": ticker, "recommendation_status": "ENTER"}]).to_csv(
            run_dir / "final_recommendations.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "ticker": f"{ticker}W",
                    "bias": "bearish",
                    "score": 50,
                    "reason": "watch mock row",
                    "hard_blocker": "insufficient_score_or_neutral_bias",
                }
            ]
        ).to_csv(run_dir / "no_trade_audit.csv", index=False)
        pd.DataFrame(
            [
                {
                    "recommendation_rank": 1,
                    "ticker": ticker,
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                    "order_readiness": "target_order_after_market_open_and_live_recheck",
                    "entry_type": entry_type,
                    "entry_limit": 0.5,
                    "max_profit": 50.0,
                    "max_loss": 200.0,
                    "underlying_quality_tier": tier,
                        "external_agent_review_count": 4,
                        "external_agent_distinct_review_count": 4,
                        "external_agent_review_agents": "catalyst; market_regime; skeptic; structure",
                        "trade_plan": f"SELL 1 {ticker} 2026-06-05 100 Call / BUY 1 {ticker} 2026-06-05 105 Call @ 0.50 {entry_type}",
                        "sell_leg": f"SELL 1 {ticker} 2026-06-05 100 Call",
                        "buy_leg": f"BUY 1 {ticker} 2026-06-05 105 Call",
                        "execution_blockers": "market_session_open_required",
                    }
                ]
        ).to_csv(run_dir / "trade_tickets.csv", index=False)
        coverage_rows = [
            {
                "ticker": focus_ticker,
                "coverage_status": "NO_DIRECTIONAL_EDGE",
                "status_color": "gray",
                "underlying_quality_tier": "core",
                "reason": f"{focus_ticker} has no directional edge in this mock run",
            }
            for focus_ticker in core.CORE_AUDIT_TICKERS
        ]
        overrides = {
            "AAPL": {
                "coverage_status": "REVIEW_TICKET",
                "status_color": "yellow",
                "underlying_quality_tier": "core",
                "reason": "needs live repricing",
            },
            "URA": {
                "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                "status_color": "yellow",
                "underlying_quality_tier": "excluded",
                "reason": "excluded underlying",
            },
            "OKLO": {
                "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                "status_color": "yellow",
                "underlying_quality_tier": "speculative",
                "reason": "speculative underlying",
            },
            "DVN": {
                "coverage_status": "NON_ACTIONABLE_UNDERLYING",
                "status_color": "red",
                "underlying_quality_tier": "liquid",
                "reason": "liquid non-core audit row",
            },
        }
        for row in coverage_rows:
            row.update(overrides.get(row["ticker"], {}))
        pd.DataFrame(coverage_rows).to_csv(run_dir / "ticker_coverage_audit.csv", index=False)
        pd.DataFrame(
            [
                {"metric": "one_cycle_max_profit", "value": 0, "status": "BLOCK", "note": ""},
                {"metric": "target_order_candidate_max_profit", "value": 50, "status": "INFO", "note": ""},
            ]
        ).to_csv(run_dir / "monthly_feasibility.csv", index=False)
        pd.DataFrame(
            [
                {
                    "source": "expectancy_summary",
                    "source_path": "",
                    "evidence_type": "summary",
                    "status": "BLOCK",
                    "sample_size": 0,
                    "win_rate": "",
                    "avg_pnl": "",
                    "total_pnl": "",
                    "profit_factor": "",
                    "max_drawdown": "",
                    "matched_current_tickers": ticker,
                    "matched_current_count": 1,
                    "open_or_unrealized_count": 0,
                    "note": "No sufficient positive expectancy evidence is available.",
                }
            ]
        ).to_csv(run_dir / "expectancy_evidence.csv", index=False)
        if queue_rows is None:
            (run_dir / "market_open_recheck_queue.csv").write_text("", encoding="utf-8")
        else:
            pd.DataFrame(queue_rows, columns=core.MARKET_OPEN_RECHECK_COLUMNS).to_csv(
                run_dir / "market_open_recheck_queue.csv",
                index=False,
            )
        pd.DataFrame(
            [
                {
                    "recommendation_rank": 1,
                    "ticker": ticker,
                    "live_market_quality_status": "PASS",
                    "actionability_impact": "eligible_for_yellow_or_green_surface",
                    "recommendation_status": "WAIT_FOR_PRICE",
                    "live_validation_status": "PASS",
                    "structure": "bull put spread",
                    "entry_type": entry_type,
                    "entry_limit": 0.5,
                    "target_entry": 0.4,
                    "spot_live": 100,
                    "short_strike": 100,
                    "long_strike": 105,
                    "spread_width": 5,
                    "live_quote_width_pct": 0.12,
                    "live_leg_min_liquidity": 450,
                    "live_leg_liquidity_status": "PASS",
                    "quality_gate_reason": "",
                    "trade_plan": f"SELL 1 {ticker} 2026-06-05 100 Call / BUY 1 {ticker} 2026-06-05 105 Call @ 0.50 {entry_type}",
                },
                {
                    "recommendation_rank": 2,
                    "ticker": f"{ticker}BAD",
                    "live_market_quality_status": "BLOCK",
                    "actionability_impact": "blocked_not_target_candidate",
                    "recommendation_status": "AVOID",
                    "live_validation_status": "WAIT_FOR_PRICE",
                    "structure": "bull put spread",
                    "entry_type": entry_type,
                    "entry_limit": 0.5,
                    "target_entry": 0.4,
                    "spot_live": 100,
                    "short_strike": 100,
                    "long_strike": 105,
                    "spread_width": 5,
                    "live_quote_width_pct": 0.55,
                    "live_leg_min_liquidity": 35,
                    "live_leg_liquidity_status": "BLOCK",
                    "quality_gate_reason": "live_quote_width_pct_above_40pct; live_leg_liquidity_below_100",
                    "trade_plan": f"SELL 1 {ticker} 2026-06-05 100 Call / BUY 1 {ticker} 2026-06-05 105 Call @ 0.50 {entry_type}",
                },
            ],
            columns=core.LIVE_SPREAD_QUALITY_AUDIT_COLUMNS,
        ).to_csv(run_dir / "live_spread_quality_audit.csv", index=False)

    for day in ("2026-05-20", "2026-05-21", "2026-05-22"):
        source_dir = tmp_path / day
        source_dir.mkdir()
        (source_dir / f"stock-screener-{day}.csv").write_text("ticker\nAAPL\n", encoding="utf-8")
        (source_dir / f"hot-chains-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")
        (source_dir / f"chain-oi-changes-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")

    run1 = tmp_path / "out" / "options_agent" / "multidate_quality_v017_2026-05-21"
    run2 = tmp_path / "out" / "options_agent" / "multidate_quality_v017_2026-05-22"
    live = tmp_path / "out" / "options_agent" / "live_readiness_probe_v017_2026-05-22"
    write_run(run1, "2026-05-21", "CREDIT", "AAPL", tier="core")
    write_run(run2, "2026-05-22", "DEBIT", "GOOGL", queue_rows=[])
    write_run(
        live,
        "2026-05-22",
        "CREDIT",
        "LIVEQ",
        queue_rows=[
            {
                "recommendation_rank": 1,
                "ticker": "LIVEQ",
                "entry_type": "CREDIT",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "target_order_status": "target_order_candidate",
                "entry_limit": 0.65,
                "target_exit": 0.23,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "position_max_profit": 260.0,
                "position_max_loss": 740.0,
                "suggested_contracts": 4,
                "execution_confidence_score": 86,
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "fixture core queue row",
                "trade_plan": "SELL 1 LIVEQ 2026-06-05 100 Call / BUY 1 LIVEQ 2026-06-05 105 Call @ 0.65 CREDIT",
                "required_recheck": "regular_market_session_open + fresh Schwab chain",
                "recheck_action": "rerun Options Agent during regular market hours",
                "execution_blockers": "market_session_open_required",
            }
        ],
    )

    artifacts = audit.write_expanded_audit(
        base_dir=tmp_path,
        run_dirs=[run1, run2],
        live_probe_dirs=[live],
        rerun_agent_reviews_json=live / "agentic_reviews.json",
        output_prefix=tmp_path / "out" / "options_agent" / "expanded_test",
    )

    market_queue = pd.read_csv(artifacts.paths["market_open_recheck_queue"])
    market_open_recheck_details = pd.read_csv(artifacts.paths["market_open_recheck_details"])
    market_open_recheck_packet = pd.read_csv(artifacts.paths["market_open_recheck_proof_packet"])
    live_rerun_preflight_details = pd.read_csv(artifacts.paths["live_rerun_preflight_details"])
    live_rerun_preflight_packet = pd.read_csv(artifacts.paths["live_rerun_preflight_proof_packet"])
    execution_packet = pd.read_csv(artifacts.paths["market_open_execution_packet"])
    multi_date_packet = pd.read_csv(artifacts.paths["multi_date_readiness_proof_packet"])
    verification_plan = pd.read_csv(artifacts.paths["market_session_verification_plan"])
    post_rerun_packet = pd.read_csv(artifacts.paths["post_rerun_verification_packet"])
    green_proof = pd.read_csv(artifacts.paths["green_ticket_execution_proof_packet"])
    session_shadow = pd.read_csv(artifacts.paths["session_only_green_shadow_proof_packet"])
    actionability_packet = pd.read_csv(artifacts.paths["actionability_surface_proof_packet"])
    action_surface_quality_packet = pd.read_csv(artifacts.paths["action_surface_underlying_quality_proof_packet"])
    expectancy_packet = pd.read_csv(artifacts.paths["expectancy_proof_packet"])
    ticket_expectancy_packet = pd.read_csv(artifacts.paths["ticket_expectancy_proof_packet"])
    monthly_guardrail_packet = pd.read_csv(artifacts.paths["monthly_feasibility_guardrail_proof_packet"])
    agentic_packet = pd.read_csv(artifacts.paths["agentic_coverage_proof_packet"])
    validation_packet = pd.read_csv(artifacts.paths["validation_coverage_proof_packet"])
    cutoff_packet = pd.read_csv(artifacts.paths["cutoff_visibility_proof_packet"])
    live_spread_quality = pd.read_csv(artifacts.paths["live_spread_quality_audit"])
    live_spread_quality_packet = pd.read_csv(artifacts.paths["live_spread_quality_proof_packet"])
    quality_packet = pd.read_csv(artifacts.paths["underlying_quality_proof_packet"])
    major_packet = pd.read_csv(artifacts.paths["major_name_coverage_proof_packet"])
    completion_verdict = pd.read_csv(artifacts.paths["completion_verdict"])
    readiness_dashboard = pd.read_csv(artifacts.paths["readiness_dashboard"])
    target_audit = pd.read_csv(artifacts.paths["target_preservation_audit"])
    goal_audit = pd.read_csv(artifacts.paths["goal_completion_audit"])

    assert market_queue["ticker"].tolist() == ["LIVEQ"]
    assert list(market_queue.columns) == audit.MARKET_QUEUE_AUDIT_COLUMNS
    assert list(market_open_recheck_details.columns) == audit.MARKET_OPEN_RECHECK_DETAIL_COLUMNS
    assert market_open_recheck_packet["status"].tolist() == ["PASS_LIVE_MARKET_OPEN_RECHECK_QUEUE_READY"]
    assert market_open_recheck_packet["queue_rows"].tolist() == [1]
    assert market_open_recheck_packet["row_fail_rows"].tolist() == [0]
    assert market_open_recheck_packet["only_market_session_blocker_rows"].tolist() == [1]
    assert market_open_recheck_packet["positive_entry_rows"].tolist() == [1]
    assert market_open_recheck_packet["plain_language_leg_rows"].tolist() == [1]
    assert market_open_recheck_packet["tickers"].tolist() == ["LIVEQ"]
    assert list(live_rerun_preflight_details.columns) == audit.LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS
    assert list(live_rerun_preflight_packet.columns) == audit.LIVE_RERUN_PREFLIGHT_PROOF_COLUMNS
    assert live_rerun_preflight_packet["status"].tolist() == ["PASS_LIVE_RERUN_PREFLIGHT_READY"]
    assert live_rerun_preflight_packet["queue_ticker_count"].tolist() == [1]
    assert live_rerun_preflight_packet["covered_queue_ticker_count"].tolist() == [1]
    assert live_rerun_preflight_packet["missing_queue_ticker_count"].tolist() == [0]
    assert live_rerun_preflight_packet["agent_review_rows"].tolist() == [4]
    assert live_rerun_preflight_packet["distinct_agent_count"].tolist() == [4]
    assert live_rerun_preflight_packet["rerun_out_dir_clear"].tolist() == [True]
    assert live_rerun_preflight_packet["source_date_available"].tolist() == [True]
    preflight_md = artifacts.paths["live_rerun_preflight_proof_packet_md"].read_text(encoding="utf-8")
    assert "Live Rerun Preflight Proof Packet" in preflight_md
    assert "queue-ticker agent review prerequisites" in preflight_md
    recheck_md = artifacts.paths["market_open_recheck_proof_packet_md"].read_text(encoding="utf-8")
    assert "Market-Open Recheck Proof Packet" in recheck_md
    assert "regular-session/fresh-live recheck gate" in recheck_md
    assert execution_packet["status"].tolist() == ["ready_for_regular_session_rerun"]
    assert execution_packet["yellow_recheck_rows"].tolist() == [1]
    assert "next_regular_session_start" in execution_packet.columns
    assert "Full-day U.S. equity market holidays" in execution_packet["market_calendar_note"].iloc[0]
    assert "--live-schwab --live-portfolio" in execution_packet["command"].iloc[0]
    assert str(live / "agentic_reviews.json") in execution_packet["command"].iloc[0]
    planned_rerun = live.parent / "live_readiness_probe_v018_market_open_rerun_2026-05-22"
    assert execution_packet["out_dir"].tolist() == [str(planned_rerun)]
    assert str(planned_rerun) in execution_packet["command"].iloc[0]
    packet_md = artifacts.paths["market_open_execution_packet_md"].read_text(encoding="utf-8")
    assert "Do not enter rows from the yellow queue" in packet_md
    assert "LIVEQ" in packet_md
    assert "position_max_profit" in packet_md
    assert re.search(
        r"\|\s*2026-05-22\s*\|\s*LIVEQ\s*\|\s*CREDIT\s*\|\s*4\s*\|\s*0\.65\s*\|\s*0\.23\s*\|\s*260(?:\.0)?\s*\|\s*740(?:\.0)?\s*\|",
        packet_md,
    )
    assert multi_date_packet["status"].tolist() == ["PASS_MULTI_DATE_TARGETS_WAITING_FOR_REGULAR_SESSION_LIVE_GREEN"]
    assert multi_date_packet["validation_date_count"].tolist() == [2]
    assert multi_date_packet["latest_live_probe_date"].tolist() == ["2026-05-22"]
    assert multi_date_packet["live_probe_dates"].tolist() == ["2026-05-22"]
    assert multi_date_packet["dated_yellow_target_candidates"].tolist() == [2]
    assert multi_date_packet["live_yellow_recheck_rows"].tolist() == [1]
    multi_date_md = artifacts.paths["multi_date_readiness_proof_packet_md"].read_text(encoding="utf-8")
    assert "Multi-date validation is separate from the latest live-session probe" in multi_date_md
    assert "2026-05-21, 2026-05-22" in multi_date_md
    assert verification_plan["status"].tolist() == ["WAITING_FOR_REGULAR_SESSION"]
    assert verification_plan["rerun_out_dir"].tolist() == [str(planned_rerun)]
    assert verification_plan["green_ticket_file"].tolist() == [str(planned_rerun / "green_trade_tickets.csv")]
    assert "ready_to_enter=true" in verification_plan["pass_criteria"].iloc[0]
    assert "completion verdict" in verification_plan["completion_gate"].iloc[0]
    verification_md = artifacts.paths["market_session_verification_plan_md"].read_text(encoding="utf-8")
    assert "Market-Session Verification Plan" in verification_md
    assert str(planned_rerun / "expectancy_evidence.csv") in verification_md
    assert post_rerun_packet["status"].tolist() == ["WAITING_FOR_REGULAR_SESSION_LIVE_RERUN"]
    assert post_rerun_packet["can_mark_goal_complete"].tolist() == [False]
    assert post_rerun_packet["green_ticket_rows"].tolist() == [0]
    assert str(planned_rerun / "green_trade_tickets.csv") in post_rerun_packet["evidence_files"].iloc[0]
    assert "python3 -m uwos.options_agent.audit" in post_rerun_packet["audit_regeneration_command"].iloc[0]
    assert str(artifacts.paths["summary"]) in post_rerun_packet["audit_regeneration_command"].iloc[0]
    assert str(planned_rerun) in post_rerun_packet["audit_regeneration_command"].iloc[0]
    assert "--rerun-agent-reviews-json" in post_rerun_packet["audit_regeneration_command"].iloc[0]
    post_rerun_md = artifacts.paths["post_rerun_verification_packet_md"].read_text(encoding="utf-8")
    assert "Post-Rerun Verification Packet" in post_rerun_md
    assert "Regenerate This Verification" in post_rerun_md
    assert "green rows, structure-aligned ticket expectancy, and the completion verdict must all agree" in post_rerun_md
    assert green_proof["status"].tolist() == ["BLOCK_NO_GREEN_TICKETS"]
    assert green_proof["green_ticket_rows"].tolist() == [0]
    assert green_proof["valid_green_ticket_rows"].tolist() == [0]
    green_proof_md = artifacts.paths["green_ticket_execution_proof_packet_md"].read_text(encoding="utf-8")
    assert "Green-Ticket Execution Proof Packet" in green_proof_md
    assert "Every green row must have ready_to_enter=true" in green_proof_md
    assert session_shadow["status"].tolist() == ["PASS_SESSION_ONLY_GREEN_SHADOW_READY"]
    assert session_shadow["shadow_candidate_rows"].tolist() == [1]
    assert session_shadow["row_fail_rows"].tolist() == [0]
    assert session_shadow["non_session_blocker_rows"].tolist() == [0]
    assert session_shadow["position_max_profit"].tolist() == [260.0]
    assert session_shadow["position_max_loss"].tolist() == [740.0]
    assert session_shadow["tickers"].tolist() == ["LIVEQ"]
    session_shadow_md = artifacts.paths["session_only_green_shadow_proof_packet_md"].read_text(encoding="utf-8")
    assert "Session-Only Green Shadow Proof Packet" in session_shadow_md
    assert "not execution permission" in session_shadow_md
    assert actionability_packet["status"].tolist() == ["PASS_YELLOW_ONLY_SURFACE_SEPARATED"]
    assert actionability_packet["target_order_rows"].tolist() == [2]
    assert actionability_packet["target_ready_to_enter_rows"].tolist() == [0]
    assert actionability_packet["target_missing_entry_type_rows"].tolist() == [0]
    assert actionability_packet["target_missing_plain_language_leg_rows"].tolist() == [0]
    assert actionability_packet["entry_types"].tolist() == ["CREDIT, DEBIT"]
    actionability_md = artifacts.paths["actionability_surface_proof_packet_md"].read_text(encoding="utf-8")
    assert "Structural recommendation labels such as ENTER are not execution permission" in actionability_md
    assert expectancy_packet["status"].tolist() == ["blocked_no_green_orders_and_no_positive_monthly_expectancy"]
    assert expectancy_packet["monthly_claim_allowed"].tolist() == [False]
    assert expectancy_packet["minimum_sample_size"].tolist() == [core.MIN_EXPECTANCY_SAMPLE_SIZE]
    assert "green-ticket ticker support is proven separately" in expectancy_packet["required_evidence"].iloc[0]
    assert "expectancy_summary=3" in expectancy_packet["blocking_source_counts"].iloc[0]
    expectancy_md = artifacts.paths["expectancy_proof_packet_md"].read_text(encoding="utf-8")
    assert "Monthly claim allowed: False" in expectancy_md
    assert "This packet is a claim gate" in expectancy_md
    assert ticket_expectancy_packet["status"].tolist() == ["BLOCK_NO_GREEN_TICKERS_FOR_EXPECTANCY_CLAIM"]
    assert "LIVEQ" not in ticket_expectancy_packet["ticket_tickers"].iloc[0]
    ticket_expectancy_md = artifacts.paths["ticket_expectancy_proof_packet_md"].read_text(encoding="utf-8")
    assert "Structure-Aligned Ticket Expectancy Proof Packet" in ticket_expectancy_md
    assert "Replay-only or unrelated-strategy" in ticket_expectancy_md
    assert monthly_guardrail_packet["status"].tolist() == ["FAIL_STALE_MONTHLY_FEASIBILITY_GUARDRAIL"]
    assert monthly_guardrail_packet["missing_required_metric_count"].tolist() == [3]
    assert monthly_guardrail_packet["required_metric"].tolist() == ["ready_ticket_expectancy_evidence"]
    monthly_guardrail_md = artifacts.paths["monthly_feasibility_guardrail_proof_packet_md"].read_text(encoding="utf-8")
    assert "Monthly Feasibility Guardrail Proof Packet" in monthly_guardrail_md
    assert agentic_packet["status"].tolist() == ["PASS_FULL_AGENTIC_TICKET_COVERAGE"]
    assert agentic_packet["ticket_rows_with_agentic_ready"].tolist() == [2]
    assert agentic_packet["ticket_rows_without_agentic_ready"].tolist() == [0]
    assert agentic_packet["required_min_ticket_lanes"].tolist() == [core.MIN_AGENTIC_REVIEW_LANES_PER_TICKER]
    assert agentic_packet["min_ticket_distinct_review_count"].tolist() == [4]
    assert agentic_packet["ticket_rows_below_min_ticket_lanes"].tolist() == [0]
    agentic_md = artifacts.paths["agentic_coverage_proof_packet_md"].read_text(encoding="utf-8")
    assert "Every user-facing ticket row" in agentic_md
    assert "Ticket rows below lane minimum: 0" in agentic_md
    assert validation_packet["status"].tolist() == ["PROVEN_WINDOW_COVERED"]
    assert validation_packet["tested_date_count"].tolist() == [2]
    assert validation_packet["untested_available_date_count"].tolist() == [0]
    assert validation_packet["available_dates_outside_window_count"].tolist() == [1]
    validation_md = artifacts.paths["validation_coverage_proof_packet_md"].read_text(encoding="utf-8")
    assert "2026-05-21, 2026-05-22" in validation_md
    assert "2026-05-20" in validation_md
    assert cutoff_packet["status"].tolist() == ["PASS_NO_ARTIFICIAL_CUTOFFS"]
    assert cutoff_packet["candidate_rows"].tolist() == [4]
    assert cutoff_packet["research_task_rows"].tolist() == [4]
    assert cutoff_packet["qualified_candidate_rows"].tolist() == [2]
    assert cutoff_packet["priced_candidate_rows"].tolist() == [2]
    assert cutoff_packet["expected_no_trade_rows"].tolist() == [2]
    assert cutoff_packet["no_trade_audit_rows"].tolist() == [2]
    cutoff_md = artifacts.paths["cutoff_visibility_proof_packet_md"].read_text(encoding="utf-8")
    assert "Cutoff Visibility Proof Packet" in cutoff_md
    assert "not capped by top-trades" in cutoff_md
    assert list(live_spread_quality.columns) == audit.LIVE_SPREAD_QUALITY_ROLLUP_COLUMNS
    assert live_spread_quality_packet["status"].tolist() == ["PASS_LIVE_SPREAD_QUALITY_GATED"]
    assert live_spread_quality_packet["audited_rows"].tolist() == [6]
    assert live_spread_quality_packet["block_rows"].tolist() == [3]
    assert live_spread_quality_packet["blocked_still_actionable_rows"].tolist() == [0]
    assert live_spread_quality_packet["target_candidate_block_rows"].tolist() == [0]
    live_spread_quality_md = artifacts.paths["live_spread_quality_proof_packet_md"].read_text(encoding="utf-8")
    assert "Live Spread Quality Proof Packet" in live_spread_quality_md
    assert "Bad live/snapshot spread markets were blocked" in live_spread_quality_md
    assert quality_packet["status"].tolist() == ["PASS_CORE_ONLY_TICKETS"]
    assert quality_packet["not_core_or_liquid_ticket_rows"].tolist() == [0]
    assert quality_packet["liquid_non_core_ticket_rows"].tolist() == [0]
    assert "OKLO" in quality_packet["focus_speculative_examples"].iloc[0]
    assert "URA" in quality_packet["focus_excluded_examples"].iloc[0]
    quality_md = artifacts.paths["underlying_quality_proof_packet_md"].read_text(encoding="utf-8")
    assert "Only core large-cap/index/ETF underlyings" in quality_md
    assert "DVN" in quality_md
    assert action_surface_quality_packet["status"].tolist() == ["PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"]
    assert action_surface_quality_packet["ticket_bad_underlying_rows"].tolist() == [0]
    assert action_surface_quality_packet["market_open_recheck_bad_underlying_rows"].tolist() == [0]
    assert action_surface_quality_packet["focus_bad_actionable_rows"].tolist() == [0]
    action_surface_quality_md = artifacts.paths["action_surface_underlying_quality_proof_packet_md"].read_text(encoding="utf-8")
    assert "Action-Surface Underlying Quality Proof Packet" in action_surface_quality_md
    assert "Red no-action audit tickers" in action_surface_quality_md
    assert major_packet["status"].tolist() == ["PASS_ALL_MAJOR_NAMES_EXPLAINED"]
    assert major_packet["required_ticker_count"].tolist() == [len(core.CORE_AUDIT_TICKERS)]
    assert major_packet["missing_required_ticker_count"].tolist() == [0]
    assert major_packet["required_rows_missing_reason"].tolist() == [0]
    major_md = artifacts.paths["major_name_coverage_proof_packet_md"].read_text(encoding="utf-8")
    assert "AAPL" in major_md
    assert "NVDA" in major_md
    assert "AVGO" in major_md
    assert "PLTR" in major_md
    assert target_audit.loc[target_audit["metric"].eq("credit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert target_audit.loc[target_audit["metric"].eq("debit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("use multi-agent orchestration evidence"),
        "artifact",
    ].tolist() == [str(artifacts.paths["agentic_coverage_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("validate across multiple available UW dates"),
        "artifact",
    ].tolist() == [str(artifacts.paths["validation_coverage_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove latest live probe is not the whole validation"),
        "artifact",
    ].tolist() == [str(artifacts.paths["multi_date_readiness_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove latest live probe is not the whole validation"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("avoid artificial trade-count cutoffs"),
        "artifact",
    ].tolist() == [str(artifacts.paths["cutoff_visibility_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("avoid artificial trade-count cutoffs"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prioritize liquid large-cap/index/high-volume names over junk"),
        "artifact",
    ].tolist() == [str(artifacts.paths["action_surface_underlying_quality_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("block bad live spread markets from actionable surfaces"),
        "artifact",
    ].tolist() == [str(artifacts.paths["live_spread_quality_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("block bad live spread markets from actionable surfaces"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove market-open recheck queue is complete and only session-blocked"),
        "artifact",
    ].tolist() == [str(artifacts.paths["market_open_recheck_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove market-open recheck queue is complete and only session-blocked"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove live rerun preflight has queue-ticker agent reviews"),
        "artifact",
    ].tolist() == [str(artifacts.paths["live_rerun_preflight_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove live rerun preflight has queue-ticker agent reviews"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("separate yellow target orders from green send-now orders"),
        "artifact",
    ].tolist() == [str(artifacts.paths["actionability_surface_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("explain major-name inclusion/exclusion"),
        "artifact",
    ].tolist() == [str(artifacts.paths["major_name_coverage_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("be execution-ready trade quality confidence pipeline"),
        "status",
    ].tolist() == ["NOT_ACHIEVED"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("be execution-ready trade quality confidence pipeline"),
        "artifact",
    ].tolist() == [str(artifacts.paths["green_ticket_execution_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("do not claim $10k/month readiness without evidence"),
        "artifact",
    ].tolist() == [str(artifacts.paths["ticket_expectancy_proof_packet"])]
    assert completion_verdict["can_mark_goal_complete"].tolist() == [False]
    assert completion_verdict["update_goal_action"].tolist() == ["do_not_call_update_goal_complete"]
    assert "be execution-ready trade quality confidence pipeline" in completion_verdict["blocking_requirements"].iloc[0]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("overall_completion"),
        "status",
    ].tolist() == ["ACTIVE_NOT_COMPLETE"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("overall_completion"),
        "required_next_action",
    ].iloc[0] == readiness_dashboard.loc[
        readiness_dashboard["area"].eq("execution_readiness"),
        "required_next_action",
    ].iloc[0]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("execution_readiness"),
        "status",
    ].tolist() == ["NOT_ACHIEVED"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("cutoff_visibility"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("live_spread_quality"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("market_open_recheck_quality"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("live_rerun_preflight"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("session_only_green_shadow"),
        "status",
    ].tolist() == ["PASS_SESSION_ONLY_GREEN_SHADOW_READY"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("action_surface_underlying_quality"),
        "status",
    ].tolist() == ["PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("monthly_feasibility_guardrail"),
        "status",
    ].tolist() == ["FAIL_STALE_MONTHLY_FEASIBILITY_GUARDRAIL"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("post_rerun_go_no_go"),
        "status",
    ].tolist() == ["WAITING_FOR_REGULAR_SESSION_LIVE_RERUN"]
    dashboard_md = artifacts.paths["readiness_dashboard_md"].read_text(encoding="utf-8")
    assert "Options Agent Readiness Dashboard" in dashboard_md
    assert "Use this dashboard as an index only" in dashboard_md
    completion_md = artifacts.paths["completion_verdict_md"].read_text(encoding="utf-8")
    assert "Can mark goal complete: False" in completion_md
    assert "do_not_call_update_goal_complete" in completion_md
    assert "Do not mark the goal complete yet." in artifacts.paths["goal_completion_audit_md"].read_text(encoding="utf-8")


def test_multi_date_scope_proof_passes_with_market_open_live_probe_even_without_green() -> None:
    summary = pd.DataFrame(
        [
            {"date": "2026-05-21", "trade_ticket_rows": 1, "green_ready_orders": 0, "yellow_target_candidates": 1},
            {"date": "2026-05-22", "trade_ticket_rows": 1, "green_ready_orders": 0, "yellow_target_candidates": 1},
        ]
    )
    live_probe_summary = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "market_session_open": True,
                "green_ready_orders": 0,
                "market_open_recheck_queue": 0,
            }
        ]
    )
    market_open_execution_packet = pd.DataFrame(
        [{"date": "2026-05-22", "status": "no_green_orders_present"}]
    )

    packet = audit.build_multi_date_readiness_proof_packet(
        summary=summary,
        live_probe_summary=live_probe_summary,
        market_open_execution_packet=market_open_execution_packet,
    )

    assert packet["status"].tolist() == ["PASS_MULTI_DATE_WITH_MARKET_OPEN_LIVE_PROBE_NO_GREEN_TICKETS"]
    assert packet["validation_date_count"].tolist() == [2]
    assert packet["live_market_session_open_count"].tolist() == [1]
    assert packet["live_green_ready_orders"].tolist() == [0]
    assert packet["live_yellow_recheck_rows"].tolist() == [0]


def test_market_open_probe_without_queue_supersedes_recheck_preflight(tmp_path: Path) -> None:
    day = "2026-05-22"
    day_dir = tmp_path / day
    day_dir.mkdir()
    (day_dir / f"stock-screener-{day}.csv").write_text("ticker\nAMAT\n", encoding="utf-8")
    (day_dir / f"hot-chains-{day}.csv").write_text("option_symbol\nAMAT260529C00467500\n", encoding="utf-8")
    (day_dir / f"chain-oi-changes-{day}.csv").write_text("option_symbol\nAMAT260529C00467500\n", encoding="utf-8")

    execution_packet = pd.DataFrame(
        [
            {
                "date": day,
                "status": "market_open_live_probe_no_green_orders",
                "fresh_live_quotes_ready": True,
                "portfolio_ready": True,
                "agentic_reviews_ready": True,
                "market_session_open": True,
                "green_ready_orders": 0,
                "yellow_recheck_rows": 0,
                "agent_reviews_json": str(tmp_path / "agentic_reviews.json"),
                "out_dir": str(tmp_path / "fresh_live_rerun"),
                "command": (
                    "python3 -m uwos.options_agent "
                    f"--date {day} "
                    f"--base-dir {tmp_path} "
                    f"--out-dir {tmp_path / 'fresh_live_rerun'} "
                    "--live-schwab --live-portfolio "
                    f"--agent-reviews-json {tmp_path / 'agentic_reviews.json'}"
                ),
            }
        ]
    )
    details = pd.DataFrame(columns=audit.MARKET_OPEN_RECHECK_DETAIL_COLUMNS)

    recheck_packet = audit.build_market_open_recheck_proof_packet(
        details,
        market_open_execution_packet=execution_packet,
    )
    preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=details,
        market_open_execution_packet=execution_packet,
        preflight_details=pd.DataFrame(columns=audit.LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS),
    )

    assert recheck_packet["status"].tolist() == [
        "PASS_NO_MARKET_OPEN_RECHECK_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]
    assert preflight_packet["status"].tolist() == [
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]

    green_packet = execution_packet.copy()
    green_packet.loc[0, "status"] = "green_orders_present_verify_ticket_scoped_expectancy"
    green_packet.loc[0, "green_ready_orders"] = 1
    green_recheck_packet = audit.build_market_open_recheck_proof_packet(
        details,
        market_open_execution_packet=green_packet,
    )
    green_preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=details,
        market_open_execution_packet=green_packet,
        preflight_details=pd.DataFrame(columns=audit.LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS),
    )

    assert green_recheck_packet["status"].tolist() == [
        "PASS_NO_MARKET_OPEN_RECHECK_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]
    assert "produced green orders" in green_recheck_packet["note"].iloc[0]
    assert green_preflight_packet["status"].tolist() == [
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]


def test_completed_market_open_probe_preflight_ignores_stale_dated_queue(tmp_path: Path) -> None:
    day = "2026-05-22"
    completed_out_dir = tmp_path / "live_readiness_probe_market_open_rerun_2026-05-22"
    completed_out_dir.mkdir()
    execution_packet = pd.DataFrame(
        [
            {
                "date": day,
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "fresh_live_quotes_ready": True,
                "portfolio_ready": True,
                "agentic_reviews_ready": True,
                "market_session_open": True,
                "green_ready_orders": 1,
                "yellow_recheck_rows": 0,
                "agent_reviews_json": "",
                "out_dir": str(completed_out_dir),
                "command": "",
            }
        ]
    )
    stale_dated_queue = pd.DataFrame([{"ticker": "AAPL", "source_kind": "dated_run", "row_pass": True}])

    preflight_details = audit.build_live_rerun_preflight_details(
        market_open_recheck_details=stale_dated_queue,
        market_open_execution_packet=execution_packet,
    )
    preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=stale_dated_queue,
        market_open_execution_packet=execution_packet,
        preflight_details=preflight_details,
    )

    assert preflight_details.empty
    assert preflight_packet["status"].tolist() == [
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]
    assert preflight_packet["queue_ticker_count"].tolist() == [0]
    assert preflight_packet["rerun_out_dir_clear"].tolist() == [False]
    assert "rerun_command_missing" not in str(preflight_packet["failed_examples"].iloc[0])


def test_completion_verdict_uses_best_market_open_packet_row() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "refresh_live_probe_inputs_before_rerun",
                "market_session_open": False,
                "next_regular_session_start": "2026-05-27T06:30:00-07:00",
            },
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "market_session_open": True,
                "next_regular_session_start": "2026-05-26T06:30:00-07:00",
            },
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "positive_expectancy_ready_for_monthly_claim_review",
                    "monthly_claim_allowed": True,
                }
            ]
        ),
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["market_open_packet_status"].tolist() == [
        "green_orders_present_verify_ticket_scoped_expectancy"
    ]


def test_completion_verdict_can_close_execution_goal_without_monthly_claim() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "monthly guardrail", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "market_session_open": True,
                "next_regular_session_start": "2026-05-28T06:30:00-07:00",
            }
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "blocked_no_positive_overall_strategy_expectancy",
                    "monthly_claim_allowed": False,
                }
            ]
        ),
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["monthly_claim_allowed"].tolist() == [False]
    assert verdict["update_goal_action"].tolist() == ["call_update_goal_complete"]
    assert "$10k/month" in verdict["note"].iloc[0]


def test_expectancy_proof_packet_keeps_overall_expectancy_separate_from_ticket_coverage() -> None:
    packet = audit.build_expectancy_proof_packet(
        summary=pd.DataFrame([{"date": "2026-05-22", "green_ready_orders": 0}]),
        tickets=pd.DataFrame([{"ticker": "GOOGL"}]),
        expectancy=pd.DataFrame(
            [
                {
                    "source": "expectancy_summary",
                    "status": "BLOCK",
                    "matched_current_tickers": "GOOGL",
                }
            ]
        ),
        live_probe_summary=pd.DataFrame([{"green_ready_orders": 1}]),
    )

    assert packet["status"].tolist() == ["blocked_no_positive_overall_strategy_expectancy"]
    assert packet["monthly_claim_allowed"].tolist() == [False]
    assert "ticket_scoped" not in packet["status"].iloc[0]
    assert "green-ticket ticker support is proven separately" in packet["required_evidence"].iloc[0]


def test_agentic_coverage_proof_blocks_partial_ticket_coverage() -> None:
    summary = pd.DataFrame(
        [
            {
                "date": "2026-05-21",
                "source_dir": "/tmp/non_agentic",
                "agentic_review_coverage_pct": 0.0,
                "agentic_review_lane_coverage_pct": 0.0,
            },
            {
                "date": "2026-05-22",
                "source_dir": "/tmp/agentic",
                "agentic_review_coverage_pct": 1.0,
                "agentic_review_lane_coverage_pct": 1.0,
            },
        ]
    )
    lanes = pd.DataFrame(
        [
            {
                "date": "2026-05-21",
                "ticker": "MSFT",
                "external_agent_distinct_review_count": 4,
                "run_agentic_reviews_ready": False,
            },
            {
                "date": "2026-05-22",
                "ticker": "GOOGL",
                "external_agent_distinct_review_count": 4,
                "run_agentic_reviews_ready": True,
            },
        ]
    )

    packet = audit.build_agentic_coverage_proof_packet(summary=summary, ticket_review_lanes=lanes)

    assert packet["status"].tolist() == ["PARTIAL_AGENTIC_TICKET_COVERAGE"]
    assert packet["ticket_rows"].tolist() == [2]
    assert packet["ticket_rows_with_agentic_ready"].tolist() == [1]
    assert packet["ticket_rows_without_agentic_ready"].tolist() == [1]
    assert packet["non_agentic_ticket_dates"].tolist() == ["2026-05-21"]


def test_agentic_coverage_proof_blocks_ticket_below_lane_minimum() -> None:
    summary = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "source_dir": "/tmp/agentic",
                "agentic_review_coverage_pct": 1.0,
                "agentic_review_lane_coverage_pct": 1.0,
            },
        ]
    )
    lanes = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "ticker": "GOOGL",
                "external_agent_distinct_review_count": core.MIN_AGENTIC_REVIEW_LANES_PER_TICKER - 1,
                "run_agentic_reviews_ready": True,
            },
        ]
    )

    packet = audit.build_agentic_coverage_proof_packet(summary=summary, ticket_review_lanes=lanes)

    assert packet["status"].tolist() == ["PARTIAL_AGENTIC_TICKET_COVERAGE"]
    assert packet["ticket_rows_with_agentic_ready"].tolist() == [1]
    assert packet["ticket_rows_without_agentic_ready"].tolist() == [0]
    assert packet["ticket_rows_below_min_ticket_lanes"].tolist() == [1]
    assert packet["below_min_ticket_lane_dates"].tolist() == ["2026-05-22"]


def test_cutoff_visibility_proof_blocks_stale_capped_no_trade_audit(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "options_agent" / "stale_cap_2026-05-22"
    run_dir.mkdir(parents=True)
    (run_dir / "options_agent_manifest_2026-05-22.json").write_text(
        json.dumps({"as_of": "2026-05-22", "row_counts": {"research_tasks": 3}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"ticker": "AAPL", "quality_status": "qualified"},
            {"ticker": "NVDA", "quality_status": "watch"},
            {"ticker": "MSFT", "quality_status": "watch"},
        ]
    ).to_csv(run_dir / "candidate_generation.csv", index=False)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(run_dir / "priced_candidates.csv", index=False)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(run_dir / "final_recommendations.csv", index=False)
    pd.DataFrame([{"ticker": "NVDA"}]).to_csv(run_dir / "no_trade_audit.csv", index=False)

    packet = audit.build_cutoff_visibility_proof_packet([run_dir])

    assert packet["status"].tolist() == ["FAIL_ARTIFICIAL_CUTOFF_OR_STALE_AUDIT_ROWS"]
    assert packet["expected_no_trade_rows"].tolist() == [2]
    assert packet["no_trade_audit_rows"].tolist() == [1]
    assert "stale_cap_2026-05-22" in packet["no_trade_missing_expected_runs"].iloc[0]


def test_green_ticket_execution_proof_requires_row_level_execution_gates() -> None:
    live_summary = pd.DataFrame([{"source_dir": "/tmp/live", "market_session_open": True}])
    details = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "validation_lane": "live_readiness_probe",
                "source_dir": "/tmp/live",
                "ticker": "AAPL",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "entry_type": "DEBIT",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "live_validation_status": "PASS",
                "execution_status": "ready",
                "execution_blockers": "",
                "execution_confidence_score": core.MIN_EXECUTION_CONFIDENCE_SCORE,
                "execution_confidence_rating": "MEDIUM",
                "trade_quality_confidence_rating": "MEDIUM",
                "confidence_score_pass": True,
                "execution_confidence_pass": True,
                "trade_quality_confidence_pass": True,
                "market_session_open": True,
                "trade_plan": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.25 DEBIT",
                "sell_leg": "SELL 1 AAPL 2026-06-18 205 Call",
                "buy_leg": "BUY 1 AAPL 2026-06-18 200 Call",
                "row_pass": True,
                "fail_reasons": "",
            }
        ],
        columns=audit.GREEN_TICKET_EXECUTION_DETAIL_COLUMNS,
    )

    packet = audit.build_green_ticket_execution_proof_packet(
        details=details,
        live_probe_summary=live_summary,
    )

    assert packet["status"].tolist() == ["PASS_GREEN_TICKETS_EXECUTION_READY"]
    assert packet["green_ticket_rows"].tolist() == [1]
    assert packet["valid_green_ticket_rows"].tolist() == [1]
    assert packet["confidence_score_pass_rows"].tolist() == [1]
    assert packet["execution_confidence_pass_rows"].tolist() == [1]
    assert packet["trade_quality_confidence_pass_rows"].tolist() == [1]
    assert packet["plain_language_leg_rows"].tolist() == [1]


def test_green_ticket_execution_proof_rejects_low_confidence_rows() -> None:
    live_summary = pd.DataFrame([{"source_dir": "/tmp/live", "market_session_open": True}])
    details = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "validation_lane": "live_readiness_probe",
                "source_dir": "/tmp/live",
                "ticker": "AAPL",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "entry_type": "DEBIT",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "live_validation_status": "PASS",
                "execution_status": "ready",
                "execution_blockers": "",
                "execution_confidence_score": core.MIN_EXECUTION_CONFIDENCE_SCORE - 1,
                "execution_confidence_rating": "LOW",
                "trade_quality_confidence_rating": "LOW",
                "confidence_score_pass": False,
                "execution_confidence_pass": False,
                "trade_quality_confidence_pass": False,
                "market_session_open": True,
                "trade_plan": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.25 DEBIT",
                "sell_leg": "SELL 1 AAPL 2026-06-18 205 Call",
                "buy_leg": "BUY 1 AAPL 2026-06-18 200 Call",
                "row_pass": False,
                "fail_reasons": "execution_confidence_score_below_threshold; execution_confidence_rating_not_MEDIUM_or_HIGH; trade_quality_confidence_rating_not_MEDIUM_or_HIGH",
            }
        ],
        columns=audit.GREEN_TICKET_EXECUTION_DETAIL_COLUMNS,
    )

    packet = audit.build_green_ticket_execution_proof_packet(
        details=details,
        live_probe_summary=live_summary,
    )

    assert packet["status"].tolist() == ["FAIL_INVALID_GREEN_TICKET_ROWS"]
    assert packet["confidence_score_pass_rows"].tolist() == [0]
    assert packet["execution_confidence_pass_rows"].tolist() == [0]
    assert packet["trade_quality_confidence_pass_rows"].tolist() == [0]
    assert "execution_confidence_score_below_threshold" in packet["invalid_examples"].iloc[0]


def test_green_ticket_execution_details_reject_occ_codes(tmp_path: Path) -> None:
    live_dir = tmp_path / "live_readiness_probe_v017_2026-05-22"
    live_dir.mkdir()
    (live_dir / "options_agent_manifest_2026-05-22.json").write_text(
        json.dumps({"as_of": "2026-05-22"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "entry_type": "DEBIT",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "live_validation_status": "PASS",
                "execution_status": "ready",
                "execution_blockers": "",
                "execution_confidence_score": core.MIN_EXECUTION_CONFIDENCE_SCORE,
                "execution_confidence_rating": "MEDIUM",
                "trade_quality_confidence_rating": "MEDIUM",
                "trade_plan": "BUY 1 AAPL260618C00200000 / SELL 1 AAPL260618C00205000 @ 1.25 DEBIT",
                "sell_leg": "SELL 1 AAPL260618C00205000",
                "buy_leg": "BUY 1 AAPL260618C00200000",
            }
        ]
    ).to_csv(live_dir / "green_trade_tickets.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "source_dir": str(live_dir.resolve()),
                "market_session_open": True,
            }
        ]
    )

    details = audit.build_green_ticket_execution_details(
        live_probe_dirs=[live_dir],
        live_probe_summary=summary,
    )
    packet = audit.build_green_ticket_execution_proof_packet(
        details=details,
        live_probe_summary=summary,
    )

    assert details["row_pass"].tolist() == [False]
    assert "plain_language_buy_sell_legs_missing" in details["fail_reasons"].iloc[0]
    assert packet["status"].tolist() == ["FAIL_INVALID_GREEN_TICKET_ROWS"]


def test_ticket_expectancy_proof_blocks_green_ticker_without_actual_forward_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["WARN_REPLAY_ONLY_EXPECTANCY"]
    assert packet["status"].tolist() == ["BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"]
    assert packet["green_tickers_without_positive_actual_forward"].tolist() == ["WMT"]


def test_ticket_expectancy_proof_passes_only_with_actual_forward_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["PASS_ACTUAL_FORWARD_EXPECTANCY"]
    assert packet["status"].tolist() == ["PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"]
    assert packet["tickers_with_positive_actual_forward"].tolist() == ["WMT"]


def test_ticket_expectancy_proof_rejects_ticker_only_actual_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker",
                "evidence_type": "actual_closed_trades_by_ticker",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["BLOCK_NO_POSITIVE_TICKET_EXPECTANCY"]
    assert packet["status"].tolist() == ["BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"]


def test_ticket_expectancy_proof_rejects_broad_aggregate_actual_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["BLOCK_NO_POSITIVE_TICKET_EXPECTANCY"]
    assert packet["status"].tolist() == ["BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"]


def test_expectancy_evidence_matches_goog_googl_share_class_alias(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "GOOG", "realized_pnl": 100.0, "strategy": "vertical_spread"},
        {"ticker": "GOOG", "realized_pnl": 70.0, "strategy": "vertical_spread"},
        {"ticker": "GOOG", "realized_pnl": -20.0, "strategy": "vertical_spread"},
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOGL",
                "ready_to_enter": True,
                "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 397.5 Call @ 0.63 CREDIT",
            }
        ]
    )

    expectancy = core.build_expectancy_evidence(tmp_path, pd.DataFrame(), tickets)
    by_ticker = expectancy[expectancy["evidence_type"].eq("actual_closed_trades_by_ticker")]
    by_strategy = expectancy[expectancy["evidence_type"].eq("actual_closed_trades_by_ticker_strategy")]
    annotated = core.annotate_actual_forward_expectancy(
        pd.DataFrame(
            [
                {
                    "ticker": "GOOGL",
                    "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 397.5 Call @ 0.63 CREDIT",
                }
            ]
        ),
        tmp_path,
    )

    assert by_ticker["matched_current_tickers"].tolist() == ["GOOGL"]
    assert by_ticker["status"].tolist() == ["PASS"]
    assert by_strategy["matched_current_tickers"].tolist() == ["GOOGL"]
    assert by_strategy["status"].tolist() == ["PASS"]
    assert annotated["actual_forward_expectancy_status"].tolist() == ["PASS"]
    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["PASS"]
    assert annotated["actual_forward_expectancy_source_tickers"].tolist() == ["GOOG"]


def test_strategy_expectancy_blocks_opposite_or_unrelated_ticker_history(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "GOOG", "realized_pnl": 300.0, "strategy": "long_call"},
        {"ticker": "GOOG", "realized_pnl": 200.0, "strategy": "long_call"},
        {"ticker": "GOOG", "realized_pnl": 100.0, "strategy": "long_call"},
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "GOOGL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                    "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 400 Call @ 1.50 CREDIT",
                    "entry_limit": 1.5,
                "suggested_contracts": 5,
                    "max_profit": 150.0,
                    "max_loss": 350.0,
                    "credit_width_ratio": 0.30,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=5,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    decision = core.synthesize_decision_board(annotated, market_regime={"regime": "mixed"}, execution_context=context)

    assert annotated["actual_forward_expectancy_status"].tolist() == ["PASS"]
    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert "strategy_expectancy_support_required" not in decision["execution_blockers"].iloc[0]


def test_completion_verdict_only_allows_goal_close_when_all_proofs_pass() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "next_regular_session_start": "2026-05-26T06:30:00-07:00",
            }
        ]
    )
    expectancy_packet = pd.DataFrame(
        [
            {
                "status": "positive_expectancy_ready_for_monthly_claim_review",
                "monthly_claim_allowed": True,
            }
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=expectancy_packet,
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["update_goal_action"].tolist() == ["call_update_goal_complete"]


def test_completion_verdict_blocks_unrelated_positive_expectancy() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "next_regular_session_start": "2026-05-26T06:30:00-07:00",
            }
        ]
    )
    broad_expectancy_packet = pd.DataFrame(
        [
            {
                "status": "positive_expectancy_ready_for_monthly_claim_review",
                "monthly_claim_allowed": True,
            }
        ]
    )
    ticket_expectancy_packet = pd.DataFrame(
        [
            {
                "status": "BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY",
            }
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=broad_expectancy_packet,
        ticket_expectancy_proof_packet=ticket_expectancy_packet,
    )

    assert verdict["can_mark_goal_complete"].tolist() == [False]
    assert verdict["update_goal_action"].tolist() == ["do_not_call_update_goal_complete"]
    assert verdict["ticket_expectancy_packet_status"].tolist() == [
        "BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"
    ]


def test_goal_completion_audit_can_complete_with_live_green_and_ticket_expectancy() -> None:
    summary = pd.DataFrame(
        [
            {"date": "2026-05-21", "monthly_feasibility": "proven", "expectancy_summary_status": "PASS"},
            {"date": "2026-05-22", "monthly_feasibility": "proven", "expectancy_summary_status": "PASS"},
        ]
    )
    tickets = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_type": "CREDIT", "ready_to_enter": False, "target_order_status": "target_order_candidate"},
            {"ticker": "MSFT", "entry_type": "DEBIT", "ready_to_enter": False, "target_order_status": "target_order_candidate"},
            {"ticker": "NVDA", "entry_type": "CREDIT", "ready_to_enter": True, "target_order_status": ""},
        ]
    )
    paths = {
        "validation_coverage_proof_packet": Path("/tmp/validation.csv"),
        "multi_date_readiness_proof_packet": Path("/tmp/multi_date.csv"),
        "cutoff_visibility_proof_packet": Path("/tmp/cutoff.csv"),
        "agentic_coverage_proof_packet": Path("/tmp/agentic.csv"),
        "live_spread_quality_proof_packet": Path("/tmp/live_spread_quality.csv"),
        "market_open_recheck_proof_packet": Path("/tmp/market_open_recheck.csv"),
        "live_rerun_preflight_proof_packet": Path("/tmp/live_rerun_preflight.csv"),
        "underlying_quality_proof_packet": Path("/tmp/quality.csv"),
        "action_surface_underlying_quality_proof_packet": Path("/tmp/action_surface_quality.csv"),
        "target_preservation_audit": Path("/tmp/target.csv"),
        "actionability_surface_proof_packet": Path("/tmp/actionability.csv"),
        "major_name_coverage_proof_packet": Path("/tmp/major.csv"),
        "ticket_expectancy_proof_packet": Path("/tmp/ticket_expectancy.csv"),
        "green_ticket_execution_proof_packet": Path("/tmp/green.csv"),
    }

    goal_audit = audit.build_goal_completion_audit(
        summary=summary,
        tickets=tickets,
        focus_coverage=pd.DataFrame(),
        ticket_review_lanes=pd.DataFrame(),
        agentic_coverage_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_FULL_AGENTIC_TICKET_COVERAGE",
                    "agentic_ready_dates": "2026-05-21, 2026-05-22",
                    "ticket_rows": 3,
                    "ticket_rows_with_agentic_ready": 3,
                    "ticket_rows_without_agentic_ready": 0,
                    "non_agentic_ticket_dates": "",
                }
            ]
        ),
        validation_coverage_proof=pd.DataFrame(
            [
                {
                    "status": "PROVEN_WINDOW_COVERED",
                    "window_available_source_date_count": 2,
                    "untested_available_date_count": 0,
                    "base_available_source_date_count": 2,
                    "available_dates_outside_window_count": 0,
                }
            ]
        ),
        cutoff_visibility_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_NO_ARTIFICIAL_CUTOFFS",
                    "candidate_rows": 10,
                    "research_task_rows": 10,
                    "qualified_candidate_rows": 3,
                    "priced_candidate_rows": 3,
                    "expected_no_trade_rows": 7,
                    "no_trade_audit_rows": 7,
                    "problem_runs": "",
                }
            ]
        ),
        live_spread_quality_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_LIVE_SPREAD_QUALITY_GATED",
                    "audited_rows": 5,
                    "block_rows": 1,
                    "blocked_still_actionable_rows": 0,
                    "target_candidate_block_rows": 0,
                    "blocked_tickers": "WIDE",
                }
            ]
        ),
        underlying_quality_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_CORE_ONLY_TICKETS",
                    "not_core_or_liquid_ticket_rows": 0,
                    "liquid_non_core_ticket_rows": 0,
                }
            ]
        ),
        major_name_coverage_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_ALL_MAJOR_NAMES_EXPLAINED",
                    "required_ticker_count": 17,
                    "covered_required_ticker_count": 17,
                    "missing_required_tickers": "",
                    "required_rows_missing_reason": 0,
                }
            ]
        ),
        expectancy=pd.DataFrame([{"source": "schwab_closed_trades", "status": "PASS"}]),
        market_open_recheck_queue=pd.DataFrame([{"source_kind": "live_probe"}]),
        market_open_recheck_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_LIVE_MARKET_OPEN_RECHECK_QUEUE_READY",
                    "queue_rows": 1,
                    "row_fail_rows": 0,
                    "credit_rows": 1,
                    "debit_rows": 0,
                    "tickers": "AAPL",
                }
            ]
        ),
        live_rerun_preflight_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_LIVE_RERUN_PREFLIGHT_READY",
                    "queue_ticker_count": 1,
                    "covered_queue_ticker_count": 1,
                    "missing_queue_tickers": "",
                    "rerun_out_dir_clear": True,
                    "agent_reviews_json": "/tmp/reviews.json",
                }
            ]
        ),
        live_probe_summary=pd.DataFrame(
            [
                {
                    "green_ready_orders": 1,
                    "market_session_open": True,
                    "expectancy_summary_status": "PASS",
                }
            ]
        ),
        multi_date_readiness_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_MULTI_DATE_AND_LIVE_GREEN_EVIDENCE",
                    "validation_date_count": 2,
                    "latest_live_probe_date": "2026-05-22",
                    "live_probe_dates": "2026-05-22",
                    "dated_yellow_target_candidates": 2,
                    "live_yellow_recheck_rows": 0,
                    "live_green_ready_orders": 1,
                }
            ]
        ),
        actionability_surface_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_AND_YELLOW_SURFACES_SEPARATED",
                    "target_ready_to_enter_rows": 0,
                    "target_missing_entry_type_rows": 0,
                    "target_missing_plain_language_leg_rows": 0,
                }
            ]
        ),
        action_surface_underlying_quality_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS",
                    "ticket_bad_underlying_rows": 0,
                    "market_open_recheck_bad_underlying_rows": 0,
                    "focus_bad_actionable_rows": 0,
                }
            ]
        ),
        green_ticket_execution_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKETS_EXECUTION_READY",
                    "valid_green_ticket_rows": 1,
                    "invalid_green_ticket_rows": 0,
                }
            ]
        ),
        ticket_expectancy_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
        paths=paths,
    )

    assert set(goal_audit["status"].tolist()) == {"PROVEN", "ACHIEVED"}
    assert goal_audit.loc[
        goal_audit["requirement"].eq("separate yellow target orders from green send-now orders"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("do not claim $10k/month readiness without evidence"),
        "status",
    ].tolist() == ["PROVEN"]

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=pd.DataFrame(
            [
                {
                    "status": "green_orders_present_verify_ticket_scoped_expectancy",
                    "next_regular_session_start": "2026-05-26T06:30:00-07:00",
                }
            ]
        ),
        expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "positive_expectancy_ready_for_monthly_claim_review",
                    "monthly_claim_allowed": True,
                }
            ]
        ),
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["status"].tolist() == ["COMPLETE"]
    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["update_goal_action"].tolist() == ["call_update_goal_complete"]


def test_post_rerun_verification_passes_only_when_all_completion_evidence_agrees() -> None:
    plan = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "status": "VERIFY_GREEN_ORDERS_AND_EXPECTANCY",
                "rerun_command": "python3 -m uwos.options_agent --date 2026-05-22",
                "green_ticket_file": "/tmp/live/green_trade_tickets.csv",
                "trade_ticket_file": "/tmp/live/trade_tickets.csv",
                "execution_readiness_file": "/tmp/live/execution_readiness.csv",
                "expectancy_file": "/tmp/live/expectancy_evidence.csv",
            }
        ]
    )
    live_summary = pd.DataFrame([{"market_session_open": True}])
    green_proof = pd.DataFrame(
        [
            {
                "status": "PASS_GREEN_TICKETS_EXECUTION_READY",
                "green_ticket_rows": 2,
                "valid_green_ticket_rows": 2,
                "invalid_green_ticket_rows": 0,
            }
        ]
    )
    ticket_expectancy = pd.DataFrame(
        [
            {
                "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                "green_ticker_count": 2,
            }
        ]
    )
    completion = pd.DataFrame(
        [
            {
                "status": "COMPLETE",
                "can_mark_goal_complete": True,
                "update_goal_action": "call_update_goal_complete",
                "monthly_claim_allowed": True,
            }
        ]
    )

    packet = audit.build_post_rerun_verification_packet(
        market_session_verification_plan=plan,
        live_probe_summary=live_summary,
        green_ticket_execution_proof=green_proof,
        ticket_expectancy_proof=ticket_expectancy,
        completion_verdict=completion,
        audit_regeneration_command="python3 -m uwos.options_agent.audit --summary-csv /tmp/summary.csv",
    )

    assert packet["status"].tolist() == ["PASS_READY_TO_COMPLETE_GOAL"]
    assert packet["green_ticket_rows"].tolist() == [2]
    assert packet["valid_green_ticket_rows"].tolist() == [2]
    assert packet["green_ticker_count"].tolist() == [2]
    assert packet["update_goal_action"].tolist() == ["call_update_goal_complete"]
    assert "--summary-csv /tmp/summary.csv" in packet["audit_regeneration_command"].iloc[0]
    assert "/tmp/live/expectancy_evidence.csv" in packet["evidence_files"].iloc[0]


def test_audit_summary_csv_helper_loads_run_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "options_agent" / "multidate_quality_v017_2026-05-22"
    run_dir.mkdir(parents=True)
    summary = tmp_path / "summary.csv"
    pd.DataFrame([{"source_dir": str(run_dir)}]).to_csv(summary, index=False)

    assert audit._run_dirs_from_summary_csv([str(summary)]) == [run_dir.resolve()]


def test_market_open_runner_blocks_closed_market_dry_run(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: False)

    result = market_open_runner.run_from_plan(plan_csv=plan, dry_run=True)

    assert result.status == "BLOCKED"
    assert "regular_market_session_open=false" in result.errors


def test_market_open_runner_dry_run_ready_with_live_flags_and_fresh_out_dir(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan.csv"
    post = tmp_path / "post.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    pd.DataFrame(
        [
            {
                "audit_regeneration_command": (
                    "python3 -m uwos.options_agent.audit --base-dir /tmp/base "
                    "--live-probe-dir /tmp/live --output-prefix /tmp/audit"
                )
            }
        ]
    ).to_csv(post, index=False)
    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: True)

    result = market_open_runner.run_from_plan(plan_csv=plan, post_rerun_csv=post, dry_run=True)

    assert result.status == "DRY_RUN_READY"
    assert "--live-schwab" in result.rerun_command
    assert "--live-portfolio" in result.rerun_command
    assert "--agent-reviews-json" in result.rerun_command
    assert result.audit_command[:3] == ("python3", "-m", "uwos.options_agent.audit")


def test_market_open_runner_reports_no_go_when_post_rerun_packet_still_blocks(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan.csv"
    post = tmp_path / "post.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    pd.DataFrame(
        [
            {
                "status": "FAIL_NO_GREEN_TICKETS_AFTER_RERUN",
                "can_mark_goal_complete": False,
                "update_goal_action": "do_not_call_update_goal_complete",
                "audit_regeneration_command": (
                    "python3 -m uwos.options_agent.audit --base-dir /tmp/base "
                    "--live-probe-dir /tmp/live --output-prefix /tmp/audit"
                ),
            }
        ]
    ).to_csv(post, index=False)
    calls: list[tuple[str, ...]] = []

    class Completed:
        returncode = 0

    def fake_run(command, cwd=None):
        calls.append(tuple(command))
        return Completed()

    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: True)
    monkeypatch.setattr(market_open_runner.subprocess, "run", fake_run)

    result = market_open_runner.run_from_plan(plan_csv=plan, post_rerun_csv=post)

    assert result.status == "COMPLETED_NOT_READY"
    assert result.post_rerun_status == "FAIL_NO_GREEN_TICKETS_AFTER_RERUN"
    assert "can_mark_goal_complete=false" in result.errors
    assert result.update_goal_action == "do_not_call_update_goal_complete"
    assert len(calls) == 2


def test_market_open_runner_reports_ready_only_when_post_rerun_packet_allows_completion(
    tmp_path: Path, monkeypatch
) -> None:
    plan = tmp_path / "plan.csv"
    post = tmp_path / "post.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    pd.DataFrame(
        [
            {
                "status": "PASS_READY_TO_COMPLETE_GOAL",
                "can_mark_goal_complete": True,
                "update_goal_action": "call_update_goal_complete",
                "audit_regeneration_command": (
                    "python3 -m uwos.options_agent.audit --base-dir /tmp/base "
                    "--live-probe-dir /tmp/live --output-prefix /tmp/audit"
                ),
            }
        ]
    ).to_csv(post, index=False)

    class Completed:
        returncode = 0

    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: True)
    monkeypatch.setattr(market_open_runner.subprocess, "run", lambda command, cwd=None: Completed())

    result = market_open_runner.run_from_plan(plan_csv=plan, post_rerun_csv=post)

    assert result.status == "COMPLETED_READY_TO_COMPLETE_GOAL"
    assert result.post_rerun_status == "PASS_READY_TO_COMPLETE_GOAL"
    assert result.can_mark_goal_complete is True
    assert result.update_goal_action == "call_update_goal_complete"


def test_summarize_run_counts_visible_yellow_ticket_rows_not_internal_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "options_agent" / "current_code_debit_scout_v047_agentic_2026-05-15"
    run_dir.mkdir(parents=True)
    (run_dir / "options_agent_manifest_2026-05-15.json").write_text(
        json.dumps(
            {
                "as_of": "2026-05-15",
                "row_counts": {
                    "trade_tickets": 2,
                    "ready_to_enter": 0,
                    "target_order_candidates": 4,
                    "target_order_ticket_rows": 2,
                    "market_open_recheck_queue": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"ticker": "PEP", "target_order_status": "target_order_candidate", "ready_to_enter": False},
            {"ticker": "BX", "target_order_status": "target_order_candidate", "ready_to_enter": False},
        ]
    ).to_csv(run_dir / "trade_tickets.csv", index=False)

    summary = audit.summarize_run(run_dir)

    assert summary["trade_ticket_rows"] == 2
    assert summary["yellow_target_candidates"] == 2


def test_recompute_live_capture_applies_profitability_calibration_gate(tmp_path: Path) -> None:
    source = tmp_path / "source_live_capture"
    output = tmp_path / "current_code_recompute"
    source.mkdir()
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(
            json.dumps({"ticker": "AMAT", "realized_pnl": pnl, "strategy": "vertical_spread"})
            for pnl in (120.0, 90.0, -20.0)
        )
        + "\n",
        encoding="utf-8",
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    (source / "options_agent_manifest_2026-05-22.json").write_text(
        json.dumps(
            {
                "as_of": "2026-05-22",
                "pipeline_name": "Options Agent",
                "pipeline_version": "old",
                "mode": "agentic_synthesis_pass",
                "source_root": str(tmp_path),
                "source_dir": str(tmp_path / "2026-05-22"),
                "agents": [],
                "artifacts": {},
                "row_counts": {
                    "decision_board": 1,
                    "trade_tickets": 1,
                    "green_trade_tickets": 0,
                    "target_order_ticket_rows": 1,
                    "market_open_recheck_queue": 0,
                    "ready_to_enter": 0,
                },
                "execution_context": context,
                "agentic_orchestration": {"status": "reviews_ingested", "subagent_task_count": 5},
                "market_regime": {"regime": "risk_off"},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "AMAT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "bias": "bearish",
                "structure": "bear call spread",
                "full_ticket": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "expiry": "2026-05-29",
                "sell_leg": "SELL 1 AMAT 2026-05-29 467.5 Call",
                "buy_leg": "BUY 1 AMAT 2026-05-29 472.5 Call",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "external_agent_review_agents": "catalyst_news; macro_regime; portfolio_management; skeptic; structure_builder",
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "synthesis_score": 140.0,
                "score": 75.0,
                "target_exit": 0.30,
                "invalidation": "underlying violates breakeven",
                "sizing_note": "risk budget supports 5 contract(s)",
                "visible_in_final_board": True,
            }
        ]
    ).to_csv(source / "final_recommendations.csv", index=False)
    pd.DataFrame(columns=["ticker", "reason"]).to_csv(source / "no_trade_audit.csv", index=False)
    pd.DataFrame(columns=["ticker", "coverage_status"]).to_csv(source / "ticker_coverage_audit.csv", index=False)
    pd.DataFrame([{"ticker": "AMAT", "live_market_quality_status": "PASS", "quality_gate_reason": ""}]).to_csv(
        source / "live_spread_quality_audit.csv",
        index=False,
    )
    (source / "market_regime.json").write_text(json.dumps({"regime": "risk_off"}), encoding="utf-8")

    paths = audit.recompute_live_capture(source_dir=source, output_dir=output, base_dir=tmp_path)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    calibration = pd.read_csv(paths["profitability_calibration"])
    green = pd.read_csv(paths["green_trade_tickets"])
    tickets = pd.read_csv(paths["trade_tickets"])
    readiness = pd.read_csv(paths["execution_readiness"])
    coverage = pd.read_csv(paths["coverage_audit"])
    report = paths["report"].read_text(encoding="utf-8")

    assert manifest["mode"] == "captured_market_open_live_recompute_current_code"
    assert manifest["captured_live_recompute"]["fresh_quote_pull"] is False
    assert manifest["row_counts"]["profitability_calibration"] == len(calibration)
    assert manifest["row_counts"]["ready_to_enter"] == 0
    assert manifest["row_counts"]["green_trade_tickets"] == 0
    assert green.empty
    assert tickets["ticker"].tolist() == ["AMAT"]
    assert tickets["ready_to_enter"].map(bool).tolist() == [False]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in tickets["execution_blockers"].iloc[0]
    assert tickets["order_readiness"].tolist() == ["target_order_after_profitability_calibration"]
    assert readiness.loc[readiness["gate"].eq("ready_trade_tickets"), "status"].tolist() == ["BLOCK"]
    assert "captured market-open live recompute" in manifest["warnings"][0]
    assert coverage.loc[coverage["ticker"].eq("AMAT"), "coverage_status"].tolist() != ["READY_TICKET"]
    assert "Captured-live recompute" in report
    assert "Monthly Readiness Gate" not in report
    assert "Green send-now rows are order-entry candidates only" not in report
    assert report.index("## Send Now Orders") < report.index("## Target Orders")
    assert "| AMAT |" in report


def test_monthly_claim_requirement_status_passes_when_ticket_scoped_evidence_supports_claim() -> None:
    status, gap = audit._monthly_claim_requirement_status(
        ticket_expectancy_status="PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
        monthly_statuses=["proven"],
        expectancy_statuses=["PASS"],
        live_expectancy=["PASS"],
        live_green_ready_orders=2,
        green_ticket_status="PASS_GREEN_TICKETS_EXECUTION_READY",
    )

    assert status == "PROVEN"
    assert gap == ""


def test_monthly_claim_requirement_status_needs_review_for_ambiguous_positive_claim() -> None:
    status, gap = audit._monthly_claim_requirement_status(
        ticket_expectancy_status="PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
        monthly_statuses=["proven"],
        expectancy_statuses=["PASS"],
        live_expectancy=["PASS"],
        live_green_ready_orders=2,
        green_ticket_status="FAIL_INVALID_GREEN_TICKET_ROWS",
    )

    assert status == "NEEDS_REVIEW"
    assert "neither cleanly blocked nor fully supported" in gap


def test_execution_readiness_gap_does_not_request_duplicate_live_run_after_green_ticket() -> None:
    gap = audit._execution_readiness_remaining_gap(
        green_ticket_status="PASS_GREEN_TICKETS_EXECUTION_READY",
        live_green_ready_orders=1,
        ticket_expectancy_status="BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY",
        live_expectancy=["BLOCK"],
    )

    assert "Order-entry readiness is proven" in gap
    assert "positive structure-aligned actual/forward expectancy evidence" in gap
    assert "do not request another live run" in gap


def test_execution_readiness_gap_clears_after_ticket_expectancy_passes() -> None:
    gap = audit._execution_readiness_remaining_gap(
        green_ticket_status="PASS_GREEN_TICKETS_EXECUTION_READY",
        live_green_ready_orders=1,
        ticket_expectancy_status="PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
        live_expectancy=["BLOCK"],
    )

    assert gap == ""


def test_agentic_review_coverage_threshold_blocks_execution_ticket() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 3,
            }
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final)
    thin_review_context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=1,
        external_review_agent_count=1,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )

    decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=thin_review_context,
    )

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["needs_agentic_review"]
    assert "agentic_review_coverage_below_threshold" in decision["execution_blockers"].iloc[0]
    tickets = core.build_trade_tickets(decision)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_agentic_review_lane_coverage_can_pass_when_broad_universe_coverage_is_low() -> None:
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=3864,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )
    readiness = core.build_execution_readiness(pd.DataFrame(columns=["ready_to_enter"]), context)
    agentic_gate = readiness[readiness["gate"].eq("agentic_reviews")].iloc[0]

    assert context["agentic_reviews_ready"] is True
    assert context["agentic_review_coverage_basis"] == "subagent_lanes"
    assert context["agentic_review_coverage_pct"] == 1.0
    assert context["broad_review_coverage_pct"] == 0.0129
    assert agentic_gate["status"] == "PASS"
    assert "coverage_basis=subagent_lanes" in agentic_gate["detail"]
    assert "coverage=1.0" in agentic_gate["detail"]
    assert "broad_universe_coverage=0.0129" in agentic_gate["detail"]


def test_execution_readiness_distinguishes_no_send_now_orders_from_ready_pipeline() -> None:
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=4,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )
    readiness = core.build_execution_readiness(pd.DataFrame(columns=["ready_to_enter"]), context)

    summary = core.summarize_execution_readiness(readiness)

    assert summary["status"] == "gates_pass_no_send_now_orders"
    assert summary["blocking_gates"] == ["ready_trade_tickets"]


def test_target_order_candidates_exclude_unvalidated_and_low_quality_underlyings() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NFLX",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NFLX 2026-06-18 92 Call / BUY 1 NFLX 2026-06-18 97 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 NFLX 2026-06-18 92 Call / BUY 1 NFLX 2026-06-18 97 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "agent_support_count": 3,
            },
            {
                "ticker": "OKLO",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 OKLO 2026-06-18 75 Call / BUY 1 OKLO 2026-06-18 80 Call @ 0.94 CREDIT",
                "trade_plan": "SELL 1 OKLO 2026-06-18 75 Call / BUY 1 OKLO 2026-06-18 80 Call @ 0.94 CREDIT",
                "entry_limit": 0.94,
                "suggested_contracts": 1,
                "max_profit": 94.0,
                "max_loss": 406.0,
                "credit_width_ratio": 0.188,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "speculative",
                "underlying_quality_reason": "marketcap_below_20000000000",
                "agent_support_count": 3,
            },
            {
                "ticker": "DATED",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 DATED 2026-05-29 47 Put / BUY 1 DATED 2026-05-29 46 Put @ 0.32 CREDIT",
                "trade_plan": "SELL 1 DATED 2026-05-29 47 Put / BUY 1 DATED 2026-05-29 46 Put @ 0.32 CREDIT",
                "entry_limit": 0.32,
                "suggested_contracts": 1,
                "max_profit": 32.0,
                "max_loss": 68.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "",
                "status_reason": "dated UW EOD quote; refresh Schwab chain before entry",
                "underlying_quality_tier": "core",
                "agent_support_count": 3,
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
            },
            {
                "ticker": "CAUTION",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 CAUTION 2026-05-29 47 Put / BUY 1 CAUTION 2026-05-29 46 Put @ 0.32 CREDIT",
                "trade_plan": "SELL 1 CAUTION 2026-05-29 47 Put / BUY 1 CAUTION 2026-05-29 46 Put @ 0.32 CREDIT",
                "entry_limit": 0.32,
                "suggested_contracts": 3,
                "max_profit": 32.0,
                "max_loss": 68.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "",
                "status_reason": "dated UW EOD quote; refresh Schwab chain before entry; external agent caution",
                "underlying_quality_tier": "core",
                "agent_caution_count": 5,
                "external_agent_distinct_review_count": 4,
            },
            {
                "ticker": "CHAIN",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 CHAIN 2026-05-29 47 Put / BUY 1 CHAIN 2026-05-29 46 Put @ 0.32 CREDIT",
                "trade_plan": "SELL 1 CHAIN 2026-05-29 47 Put / BUY 1 CHAIN 2026-05-29 46 Put @ 0.32 CREDIT",
                "entry_limit": 0.32,
                "suggested_contracts": 1,
                "max_profit": 32.0,
                "max_loss": 68.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "CHAIN_UNAVAILABLE",
                "underlying_quality_tier": "core",
                "agent_support_count": 3,
            },
            {
                "ticker": "GM",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 GM 2026-06-18 76 Call / BUY 1 GM 2026-06-18 81 Call @ 0.92 CREDIT",
                "trade_plan": "SELL 1 GM 2026-06-18 76 Call / BUY 1 GM 2026-06-18 81 Call @ 0.92 CREDIT",
                "entry_limit": 0.92,
                "suggested_contracts": 1,
                "max_profit": 92.0,
                "max_loss": 408.0,
                "credit_width_ratio": 0.184,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "liquid",
                "agent_caution_count": 5,
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    blocked_context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=Path("/tmp/snapshots"),
        portfolio_context={"status": "unavailable", "total_value": 0},
        research_task_count=10,
        external_review_count=1,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=blocked_context)
    tickets = core.build_trade_tickets(decision)
    monthly = core.build_monthly_feasibility(
        decision,
        tickets,
        {"monthly_profit_target": 10_000, "fresh_live_quotes_ready": False, "portfolio_ready": False},
        pd.DataFrame(),
    )

    assert tickets["ticker"].tolist() == ["NFLX", "DATED", "CAUTION"]
    assert tickets["ready_to_enter"].tolist() == [False, False, False]
    assert tickets["order_readiness"].tolist() == [
        "target_order_after_agentic_review",
        "target_order_after_agentic_review",
        "target_order_after_agentic_review",
    ]
    assert "agentic_review_coverage_below_threshold" in tickets["execution_blockers"].iloc[0]
    green, yellow = core.split_trade_ticket_surfaces(tickets)
    assert green.empty
    assert yellow["ticker"].tolist() == ["NFLX", "DATED", "CAUTION"]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_count"), "value"].tolist() == [0]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_count"), "status"].tolist() == ["BLOCK"]
    assert monthly.loc[monthly["metric"].eq("target_order_candidate_count"), "value"].tolist() == [3]
    assert decision.loc[decision["ticker"].eq("OKLO"), "target_order_status"].tolist() == [
        "not_actionable_underlying_quality"
    ]
    assert decision.loc[decision["ticker"].eq("CHAIN"), "target_order_status"].tolist() == [
        "not_actionable_unvalidated_chain"
    ]
    assert decision.loc[decision["ticker"].eq("DATED"), "target_order_status"].tolist() == [
        "target_order_candidate"
    ]
    assert decision.loc[decision["ticker"].eq("CAUTION"), "target_order_status"].tolist() == [
        "target_order_candidate"
    ]
    assert decision.loc[decision["ticker"].eq("GM"), "target_order_status"].tolist() == [
        "not_actionable_underlying_quality"
    ]


def test_target_order_candidate_preserves_debit_plan_without_credit_width_gate() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.50 DEBIT",
                "trade_plan": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.50 DEBIT",
                "entry_limit": 1.5,
                "suggested_contracts": 3,
                "max_profit": 350.0,
                "max_loss": 150.0,
                "credit_width_ratio": 0.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=Path("/tmp/snapshots"),
        portfolio_context={"status": "unavailable", "total_value": 0},
        research_task_count=1,
        external_review_count=4,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["entry_type"].tolist() == ["DEBIT"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["order_readiness"].tolist() == ["target_order_after_portfolio_sizing"]


def test_expectancy_evidence_blocks_monthly_target_on_negative_actual_history(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 25.0}
        for i in range(40)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "pnl_1x": 100.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame([{"ticker": "WMT"}])
    tickets = pd.DataFrame([{"ticker": "WMT", "max_profit": 100.0, "max_loss": 400.0}])

    evidence = core.build_expectancy_evidence(tmp_path, decision, tickets)
    summary = core.summarize_expectancy_evidence(evidence)
    feasibility = core.build_monthly_feasibility(
        decision,
        tickets,
        {
            "monthly_profit_target": 10_000,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        evidence,
    )

    assert "schwab_closed_trades" in summary["blocking_sources"]
    assert summary["status"] == "not_proven"
    assert evidence.loc[evidence["source"].eq("codexuw_replay_decision_pass"), "status"].tolist() == ["PASS"]
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["BLOCK"]
    assert feasibility.loc[feasibility["metric"].eq("expectancy_evidence"), "status"].tolist() == ["BLOCK"]


def test_expectancy_evidence_does_not_pass_on_unrelated_positive_actual_history(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {"ticker": "AAPL", "strategy": "vertical_spread", "realized_pnl": 100.0}
        for _ in range(40)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "pnl_1x": 100.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    evidence = core.build_expectancy_evidence(tmp_path, pd.DataFrame([{"ticker": "WMT"}]), pd.DataFrame())
    summary = core.summarize_expectancy_evidence(evidence)

    assert evidence.loc[evidence["source"].eq("schwab_closed_trades"), "status"].tolist() == ["BLOCK"]
    assert evidence.loc[evidence["source"].eq("schwab_closed_trades"), "matched_current_count"].tolist() == [0]
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["WARN"]
    assert summary["status"] == "mixed"


def test_expectancy_evidence_blocks_on_negative_actual_strategy_cohort(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {"ticker": "AAPL", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 20.0}
        for i in range(40)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {"ticker": "WMT", "pnl_1x": 100.0, "exact_evaluated": True, "decision_pass": True}
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame([{"ticker": "WMT"}])
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "trade_plan": "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT",
                "ready_to_enter": True,
            }
        ]
    )

    evidence = core.build_expectancy_evidence(tmp_path, decision, tickets)
    summary = core.summarize_expectancy_evidence(evidence)
    cohort = evidence[evidence["source"].eq("schwab_closed_trades_strategy_cohort")].iloc[0]

    assert cohort["evidence_type"] == "actual_closed_trades_strategy_cohort"
    assert cohort["status"] == "BLOCK"
    assert cohort["sample_size"] == 40
    assert cohort["matched_current_count"] == 0
    assert "vertical_spread" in cohort["note"]
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["BLOCK"]
    assert summary["status"] == "not_proven"


def test_strategy_outcome_atlas_surfaces_positive_and_negative_strategy_families(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = []
    rows.extend({"ticker": "AAPL", "strategy": "short_put", "realized_pnl": 100.0} for _ in range(34))
    rows.extend({"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 20.0} for i in range(40))
    rows.extend({"ticker": "GOOG", "strategy": "long_call", "realized_pnl": 90.0} for _ in range(3))
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "trade_plan": "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT",
                "ready_to_enter": False,
            }
        ]
    )

    atlas = core.build_strategy_outcome_atlas(tmp_path, pd.DataFrame(), tickets)
    summary = core.summarize_strategy_outcome_atlas(atlas)
    family = atlas[atlas["scope"].eq("strategy_family")]
    current = atlas[atlas["scope"].eq("current_ticker_strategy")].iloc[0]

    assert family.loc[family["strategy_family"].eq("short_put"), "status"].tolist() == ["PASS"]
    assert family.loc[family["strategy_family"].eq("vertical_spread"), "status"].tolist() == ["BLOCK"]
    assert current["ticker"] == "WMT"
    assert current["strategy_family"] == "vertical_spread"
    assert current["status"] == "BLOCK"
    assert current["suggested_action"] == "do_not_promote_current_strategy_family"
    assert summary["positive_strategy_families"] == ["short_put"]
    assert summary["negative_current_strategy_families"] == ["vertical_spread"]
    assert summary["blocking_current_ticker_strategy_rows"] == 1


def test_strategy_outcome_atlas_requires_ticker_strategy_support_for_current_rows(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [{"ticker": "AAPL", "strategy": "short_put", "realized_pnl": 100.0} for _ in range(34)]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "trade_plan": "SELL 1 GOOG 2026-06-19 95 Put @ 1.00 CREDIT",
                "strategy": "short_put",
                "ready_to_enter": False,
            }
        ]
    )

    atlas = core.build_strategy_outcome_atlas(tmp_path, pd.DataFrame(), tickets)
    current = atlas[atlas["scope"].eq("current_ticker_strategy")].iloc[0]
    summary = core.summarize_strategy_outcome_atlas(atlas)

    assert summary["positive_strategy_families"] == ["short_put"]
    assert current["ticker"] == "GOOG"
    assert current["strategy_family"] == "short_put"
    assert current["status"] == "BLOCK"
    assert current["sample_size"] == 0
    assert current["suggested_action"] == "keep_watch_only_until_ticker_strategy_outcomes_exist"


def test_expectancy_evidence_uses_project_schwab_closed_trades_for_overlay_root(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    overlay_root = project / "overlays" / "options_agent_fixture"
    overlay_root.mkdir(parents=True)
    closed_dir = project / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_path = closed_dir / "closed_trades_acct_3326.jsonl"
    closed_rows = [{"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": 100.0} for _ in range(40)]
    closed_path.write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "project_root", lambda: project)

    evidence = core.build_expectancy_evidence(
        overlay_root,
        pd.DataFrame([{"ticker": "WMT"}]),
        pd.DataFrame([{"ticker": "WMT", "ready_to_enter": True}]),
    )
    closed = evidence[evidence["source"].eq("schwab_closed_trades")].iloc[0]

    assert closed["source_path"] == str(closed_path)
    assert closed["matched_current_tickers"] == "WMT"
    assert closed["status"] == "PASS"


def test_expectancy_evidence_prefers_visible_ticket_tickers_over_broad_decision_board(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [{"ticker": "AAPL", "strategy": "vertical_spread", "realized_pnl": 100.0} for _ in range(40)]
    closed_rows.extend(
        {"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 25.0}
        for i in range(40)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {"ticker": "AAPL", "pnl_1x": 100.0, "exact_evaluated": True, "decision_pass": True}
            for _ in range(40)
        ]
        + [
            {"ticker": "WMT", "pnl_1x": 100.0, "exact_evaluated": True, "decision_pass": True}
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame([{"ticker": "AAPL"}, {"ticker": "WMT"}])
    tickets = pd.DataFrame([{"ticker": "WMT", "ready_to_enter": False}])

    evidence = core.build_expectancy_evidence(tmp_path, decision, tickets)
    closed = evidence[evidence["source"].eq("schwab_closed_trades")].iloc[0]
    replay = evidence[evidence["source"].eq("codexuw_replay_decision_pass")].iloc[0]

    assert closed["matched_current_tickers"] == "WMT"
    assert closed["status"] == "BLOCK"
    assert replay["matched_current_tickers"] == "WMT"
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["BLOCK"]


def test_monthly_feasibility_positive_status_still_disclaims_guarantee() -> None:
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 4, "status": "PASS", "note": ""},
            {"metric": "one_cycle_max_profit", "value": 12_000, "status": "PASS", "note": ""},
            {"metric": "cycles_needed_at_max_profit", "value": 1, "status": "PASS", "note": ""},
            {"metric": "expectancy_evidence", "value": 80, "status": "PASS", "note": ""},
            {"metric": "monthly_profit_target", "value": 10_000, "status": "INFO", "note": "User target; not a guarantee."},
        ]
    )

    summary = core.summarize_monthly_feasibility(monthly)

    assert summary["status"] == "capacity_and_expectancy_positive_not_guaranteed"
    assert summary["blocking_metrics"] == []
    assert "not guaranteed" in summary["note"]


def test_management_plan_separates_review_from_entry_ready_rows() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "REVIEW",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "recommendation_rank": 1,
                "status_reason": "news needs review",
                "entry_limit": 1.0,
                "target_exit": 0.35,
                "invalidation": "thesis breaks",
                "suggested_contracts": 0,
            },
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "recommendation_rank": 2,
                "live_validation_status": "PASS",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "target_exit": 0.35,
                "invalidation": "underlying breaks breakeven",
                    "suggested_contracts": 5,
            },
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final, {"LIVE"})
    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    plan = core.build_management_plan(final, decision)

    assert plan["management_action"].tolist() == ["REVIEW", "ENTRY_READY"]
    assert "Do not enter" in plan.loc[plan["ticker"].eq("REVIEW"), "entry_condition"].iloc[0]
    assert "live quote" in plan.loc[plan["ticker"].eq("LIVE"), "entry_condition"].iloc[0]
    assert plan.loc[plan["ticker"].eq("LIVE"), "target_exit"].tolist() == [0.35]


def test_synthesis_ranking_prefers_live_validated_entry_over_raw_flow_score() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "HIGH",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "score": 95.0,
                "signal_premium": 10_000_000,
                "full_ticket": "",
                "entry_limit": "",
            },
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "score": 70.0,
                "signal_premium": 1_000_000,
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "RISK",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "score": 70.0,
                "signal_premium": 900_000,
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
                "portfolio_risk_flag": True,
            },
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "ticker": "HIGH",
                "agent": "skeptic",
                "agent_type": "built_in",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": False,
            },
            {
                "ticker": "RISK",
                "agent": "portfolio_risk",
                "agent_type": "built_in",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": True,
            },
        ]
    )

    ranked = core.apply_synthesis_ranking(final, reviews, top_trades=3)

    assert ranked["ticker"].tolist() == ["LIVE", "RISK", "HIGH"]
    assert ranked["recommendation_rank"].tolist() == [1, 2, 3]
    assert ranked.loc[ranked["ticker"].eq("RISK"), "agent_portfolio_risk_only_count"].tolist() == [1]
    assert ranked.loc[ranked["ticker"].eq("LIVE"), "synthesis_score"].iloc[0] == ranked.loc[
        ranked["ticker"].eq("RISK"), "synthesis_score"
    ].iloc[0]
    assert ranked.loc[ranked["ticker"].eq("RISK"), "agent_caution_count"].tolist() == [0]
    assert "account-context review(s) kept audit-only +0" in ranked.loc[
        ranked["ticker"].eq("RISK"), "synthesis_reason"
    ].iloc[0]


def test_position_sizing_annotates_risk_without_suppressing_trade() -> None:
    rows = core.apply_position_sizing(
        [
            {
                "ticker": "WMT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "live_validation_status": "PASS",
                "max_loss": 200.0,
                "portfolio_risk_flag": False,
            },
            {
                "ticker": "HOOD",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "live_validation_status": "PASS",
                "max_loss": 600.0,
                "portfolio_risk_flag": True,
                "portfolio_risk_note": "existing exposure",
            },
        ],
        {"status": "ok", "total_value": 100_000},
        {"sizing_stance": "normal"},
    )

    assert rows[0]["suggested_contracts"] == 2
    assert rows[0]["max_position_loss"] == 400.0
    assert rows[0]["account_risk_pct"] == 0.004
    assert rows[0]["recommendation_status"] == RecommendationStatus.ENTER.value
    assert rows[1]["suggested_contracts"] == 1
    assert rows[1]["sizing_risk_flag"] is True
    assert "one-lot exceeds normal risk budget" in rows[1]["portfolio_risk_note"]
    assert rows[1]["recommendation_status"] == RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value


def test_no_trade_audit_is_not_capped_by_top_trades() -> None:
    candidates = pd.DataFrame(
        [
            {
                "ticker": f"MISS{idx}",
                "bias": "bearish",
                "score": 70 - idx,
                "flow_reason": f"candidate {idx}",
                "quality_status": "rejected",
            }
            for idx in range(6)
        ]
    )

    audit = core.build_no_trade_audit(candidates, pd.DataFrame(), top_trades=2)

    assert audit["ticker"].tolist() == ["MISS0", "MISS1", "MISS2", "MISS3", "MISS4", "MISS5"]
    assert audit["hard_blocker"].tolist() == ["insufficient_score_or_neutral_bias"] * 6


def test_price_candidates_default_does_not_cap_qualified_candidates(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80 - idx * 0.1,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "flow_reason": f"candidate {idx}",
            }
            for idx in range(25)
        ]
    )

    priced = core.price_candidates(tmp_path / "2026-05-22", "2026-05-22", candidates)

    assert len(priced) == 25
    assert priced["ticker"].tolist() == ["WMT"] * 25


def test_price_candidates_uses_short_put_when_family_evidence_passes(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "AAPL", "realized_pnl": 100.0, "strategy": "short_put"}
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.extend(
        {"ticker": "GOOG", "realized_pnl": -50.0, "strategy": "short_put"}
        for _ in range(4)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
            }
        ]
    )

    priced, routing = core.price_candidates_with_routing_audit(
        tmp_path / "2026-05-22",
        "2026-05-22",
        candidates,
        root=tmp_path,
    )

    assert "cash secured put" in priced["structure"].tolist()
    assert "bull put spread" in priced["structure"].tolist()
    short_put = priced[priced["structure"].eq("cash secured put")].iloc[0]
    assert short_put["buy_leg"] == ""
    assert " / " not in short_put["trade_plan"]
    assert "SELL 1 WMT" in short_put["trade_plan"]
    assert routing["strategy"].tolist() == ["short_put", "bull_call_debit", "bull_put_credit"]
    assert routing.loc[routing["strategy"].eq("short_put"), "route_action"].tolist() == [
        "construct_allowed_positive_family_route"
    ]
    assert routing.loc[routing["strategy"].eq("bull_call_debit"), "route_status"].tolist() == ["construction_failed"]


def test_price_candidates_routes_bearish_core_to_put_debit_and_audits_credit_route(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bearish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80,
                "signal_premium": 5_000_000,
                "combined_flow_bias": -0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "candidate_rank": 1,
            }
        ]
    )

    priced, routing = core.price_candidates_with_routing_audit(
        tmp_path / "2026-05-22",
        "2026-05-22",
        candidates,
        root=tmp_path,
    )

    assert priced["strategy"].tolist() == ["bear_put_debit"]
    assert priced["structure"].tolist() == ["bear put debit spread"]
    assert "BUY 1 WMT 2026-06-19 95 Put" in priced["trade_plan"].iloc[0]
    assert "SELL 1 WMT 2026-06-19 90 Put" in priced["trade_plan"].iloc[0]
    assert priced["entry_limit"].tolist() == [1.4]
    assert routing["strategy"].tolist() == ["bear_put_debit", "bear_call_credit"]
    assert routing.loc[routing["strategy"].eq("bear_put_debit"), "route_status"].tolist() == ["constructed"]
    assert routing.loc[routing["strategy"].eq("bear_call_credit"), "route_status"].tolist() == ["construction_failed"]


def test_positive_broad_vertical_route_evidence_does_not_create_green_without_ticker_strategy_proof(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"WIN{idx}", "realized_pnl": 100.0, "strategy": "vertical_spread"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "bias": "bearish",
                "strategy": "bear_put_debit",
                "structure": "bear put debit spread",
                "trade_plan": "BUY 1 NEW 2026-06-19 95 Put / SELL 1 NEW 2026-06-19 90 Put @ 1.00 DEBIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "live_validation_status": "PASS",
                "entry_limit": 1.0,
                "max_profit": 400.0,
                "max_loss": 100.0,
                "position_max_profit": 1200.0,
                "suggested_contracts": 3,
                "quality_status": "qualified",
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "external_agent_distinct_review_count": 4,
                "external_agent_review_count": 4,
                "agent_support_count": 4,
                "trade_quality_status": "reviewable",
            }
        ]
    )
    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    context = {
        "fresh_live_quotes_ready": True,
        "portfolio_ready": True,
        "agentic_reviews_ready": True,
        "min_agentic_review_lanes_per_ticker": 4,
        "run_gate_blockers": [],
        "portfolio_total_value": 100_000,
        "quote_mode": "live_schwab",
    }

    decision = core.synthesize_decision_board(annotated, market_regime={"regime": "risk_off"}, execution_context=context)

    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert annotated["actual_forward_strategy_expectancy_scope"].tolist() == ["missing"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]


def test_short_put_family_fallback_is_explicit_and_does_not_mask_negative_ticker(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "AAPL", "realized_pnl": 100.0, "strategy": "short_put"}
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.extend(
        {"ticker": "BAD", "realized_pnl": -100.0, "strategy": "long_call"}
        for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {"ticker": "NEW", "structure": "cash secured put", "trade_plan": "SELL 1 NEW 2026-06-19 95 Put @ 1.00 CREDIT"},
            {"ticker": "BAD", "structure": "cash secured put", "trade_plan": "SELL 1 BAD 2026-06-19 95 Put @ 1.00 CREDIT"},
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)

    new_row = annotated[annotated["ticker"].eq("NEW")].iloc[0]
    bad_row = annotated[annotated["ticker"].eq("BAD")].iloc[0]
    assert new_row["actual_forward_strategy_expectancy_status"] == "PASS"
    assert new_row["actual_forward_strategy_expectancy_scope"] == "strategy_family"
    assert "Family-level actual/forward realized support" in new_row["actual_forward_strategy_expectancy_note"]
    assert bad_row["actual_forward_expectancy_status"] == "BLOCK"
    assert bad_row["actual_forward_strategy_expectancy_status"] == "BLOCK"


def test_short_put_cash_risk_blocks_green_and_trade_ticket_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "PUTRISK",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "cash secured put",
                "full_ticket": "SELL 1 PUTRISK 2026-07-17 150 Put @ 2.00 CREDIT",
                "trade_plan": "SELL 1 PUTRISK 2026-07-17 150 Put @ 2.00 CREDIT",
                "entry_limit": 2.0,
                "suggested_contracts": 1,
                "max_profit": 200.0,
                "max_loss": 14_800.0,
                "credit_width_ratio": "",
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "portfolio_cash": 10_000.0,
                "account_risk_pct": 0.148,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000, "cash": 10_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["target_order_status"].tolist() == ["not_actionable_cash_secured_risk"]
    assert "short_put_cash_required_above_75pct_cash" in decision["execution_blockers"].iloc[0]
    assert "send_now_credit_width_below_30pct" not in decision["execution_blockers"].iloc[0]
    assert tickets.empty


def test_negative_strategy_family_evidence_blocks_trade_ticket_surface(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"LOSS{idx}", "realized_pnl": -100.0, "strategy": "vertical_spread"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NEW 2026-06-19 100 Call / BUY 1 NEW 2026-06-19 105 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 NEW 2026-06-19 100 Call / BUY 1 NEW 2026-06-19 105 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )
    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(annotated, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert annotated["actual_forward_strategy_expectancy_scope"].tolist() == ["strategy_family"]
    assert core.NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert decision["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert tickets.empty


def test_sparse_ticker_strategy_sample_does_not_mask_negative_family_evidence(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"LOSS{idx}", "realized_pnl": -100.0, "strategy": "vertical_spread"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.append({"ticker": "MSFT", "realized_pnl": -331.0, "strategy": "vertical_spread"})
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "trade_plan": "SELL 1 MSFT 2026-07-17 410 Call / BUY 1 MSFT 2026-07-17 415 Call @ 1.24 CREDIT",
            }
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)

    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert annotated["actual_forward_strategy_expectancy_sample_size"].tolist() == [core.MIN_EXPECTANCY_SAMPLE_SIZE + 1]
    assert annotated["actual_forward_strategy_expectancy_scope"].tolist() == ["strategy_family"]
    assert "Sparse ticker-specific MSFT vertical_spread support" in annotated["actual_forward_strategy_expectancy_note"].iloc[0]


def test_profitability_calibration_passes_only_with_actual_and_replay_bucket_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(9000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]
    assert annotated["profitability_calibration_actual_sample_size"].tolist() == [core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE]
    assert annotated["profitability_calibration_actual_avg_pnl"].tolist() == [125.0]
    assert annotated["profitability_calibration_actual_profit_factor"].tolist() == ["inf"]
    assert annotated["profitability_calibration_replay_sample_size"].tolist() == [core.MIN_EXPECTANCY_SAMPLE_SIZE]
    assert annotated["profitability_calibration_replay_avg_pnl"].tolist() == [85.0]
    assert annotated["profitability_calibration_replay_profit_factor"].tolist() == ["inf"]


def test_profitability_calibration_uses_schwab_order_legs_for_vertical_route_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(1000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "vertical_spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)
    summary = core.summarize_profitability_calibration(calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["replay_bucket_status"] == "PASS"
    assert summary["actual_support_status_counts"] == {"PASS": 1}
    assert summary["replay_bucket_status_counts"] == {"PASS": 1}
    assert summary["missing_replay_bucket_rows"] == 0
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_profitability_calibration_backfills_actual_regime_from_opened_trade_date(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    regime_dir = tmp_path / "out" / "options_agent" / "2026-06-12"
    regime_dir.mkdir(parents=True)
    (regime_dir / "market_regime.json").write_text(json.dumps({"regime": "mixed"}), encoding="utf-8")
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(7100 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "regime": "mixed",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert "|bullish|mixed|" in core._calibration_key_text(row)
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_profitability_calibration_uses_leakage_safe_wheel_csp_replay_for_short_put(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    for day in ["2026-04-01", "2026-06-09"]:
        regime_dir = tmp_path / "out" / "options_agent" / day
        regime_dir.mkdir(parents=True)
        (regime_dir / "market_regime.json").write_text(json.dumps({"regime": "mixed"}), encoding="utf-8")
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(7000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Short Put",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-09T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "LIMIT",
                "price": 2.50,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717P00095000",
                            "putCall": "PUT",
                            "description": "WMT 07/17/2026 $95 Put",
                        },
                    }
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    wheel_dir = tmp_path / "out" / "fresh_wheel_replay_2026_full_ytd"
    wheel_dir.mkdir(parents=True)
    signal_dir = tmp_path / "2026-04-01"
    signal_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "option_symbol": "WMT260717P00095000",
                "date": "2026-04-01",
                "volume": 900,
                "open_interest": 1500,
                "premium": 100000.0,
                "bid": 2.40,
                "ask": 2.60,
            }
        ]
    ).to_csv(signal_dir / "hot-chains-2026-04-01.csv", index=False)
    past_rows = [
        {
            "signal_date": "2026-04-01",
            "ticker": "WMT",
            "action": "OPEN_CSP",
            "option_symbol": "WMT260717P00095000",
            "entry_credit": 2.50,
            "dte": 38,
            "exit_date": "2026-05-01",
            "pnl_per_contract": 150.0,
            "outcome_status": "scored",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    future_rows = [
        {
            "signal_date": "2026-06-01",
            "ticker": f"FUTURE{idx}",
            "action": "OPEN_CSP",
            "option_symbol": "WMT260717P00095000",
            "entry_credit": 2.50,
            "dte": 38,
            "exit_date": "2026-06-20",
            "pnl_per_contract": -500.0,
            "outcome_status": "scored",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(past_rows + future_rows).to_csv(
        wheel_dir / "fresh-wheel-replay-outcomes-2026-01-02_2026-06-20.csv",
        index=False,
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 2.50 CREDIT",
                "entry_limit": 2.50,
                "dte": 38,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final, as_of_date="2026-06-09")
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["replay_bucket_status"] == "PASS"
    assert row["replay_bucket_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_bucket_avg_pnl"] == 150.0
    assert "fresh_wheel_replay_2026_full_ytd" in row["source_path"]
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_profitability_calibration_requires_matching_replay_liquidity_bucket(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(8000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 10,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert "liquidity_deep" in row["note"]
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_can_reuse_supplied_replay_bundle(tmp_path: Path, monkeypatch) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 85.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.00,
                "dte": 38,
                "regime": "mixed",
            }
        ]
    )

    def _unexpected_replay_build(*args, **kwargs):
        raise AssertionError("replay frame should have been reused")

    monkeypatch.setattr(core, "_profitability_replay_frame", _unexpected_replay_build)

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, "shared_replay_bundle", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["source_path"] == "shared_replay_bundle"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["actual_support_sample_gap"] == 0
    assert row["replay_bucket_status"] == "PASS"
    assert row["replay_bucket_sample_gap"] == 0
    assert row["diagnostic_replay_status"] == "PASS"
    assert row["diagnostic_replay_relaxed_dimensions"] == ""


def test_profitability_calibration_requires_matching_replay_dte_bucket(tmp_path: Path) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 85.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.00,
                "dte": 38,
                "regime": "mixed",
            }
        ]
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, "shared_replay_bundle", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert row["replay_bucket_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_status"] == "PASS"
    assert row["diagnostic_replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_relaxed_dimensions"] == "dte_bucket"


def test_profitability_calibration_requires_matching_replay_economics_bucket(tmp_path: Path) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 85.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.00,
                "dte": 38,
                "regime": "mixed",
            }
        ]
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, "shared_replay_bundle", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert row["replay_bucket_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_status"] == "PASS"
    assert row["diagnostic_replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_relaxed_dimensions"] == "economics_bucket"


def test_profitability_calibration_requires_matching_replay_regime_bucket(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(8100 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
                "regime": "mixed",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "regime": "uptrend",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1500,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["actual_support_sample_gap"] == 0
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert row["replay_bucket_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert "|mixed|" in core._calibration_key_text(row)
    assert "mixed" in row["note"]
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_requires_matching_replay_direction_bucket(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(8200 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "vertical_spread",
                "direction": "bearish",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
                "regime": "mixed",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717P00100000",
                            "putCall": "PUT",
                            "description": "WMT 07/17/2026 $100 Put",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Vertical Spread",
            "direction": "bullish",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "regime": "mixed",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1500,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "vertical_spread",
                "direction": "bearish",
                "trade_plan": "WMT vertical spread @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["direction_bucket"] == "bearish"
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert "|bearish|mixed|" in core._calibration_key_text(row)
    assert "bearish" in row["note"]
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_does_not_pass_from_broad_vertical_family_only_actual_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"WIN{idx}",
            "realized_pnl": 125.0,
            "strategy": "vertical_spread",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "MSFT",
            "strategy": "Bear Call Credit Spread",
            "strategy_kind": "Credit",
            "entry_side": "credit",
            "dte": 10,
            "iv_rank": 35,
            "entry_credit_pct_width": 0.32,
            "source_contract_oi": 800,
            "pnl_1x": 75.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "structure": "Bear Call Credit Spread",
                "trade_plan": "SELL 1 MSFT 2026-06-19 430 Call / BUY 1 MSFT 2026-06-19 435 Call @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "credit_width_ratio": 0.32,
                "dte": 10,
                "iv_rank": 35,
                "live_leg_min_liquidity": 800,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)
    summary = core.summarize_profitability_calibration(calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_strategy_family"
    assert row["replay_bucket_status"] == "PASS"
    assert "actual_bucket_precision=route_or_family_only" in row["note"]
    assert summary["actual_family_only_rows"] == 1
    assert summary["missing_replay_bucket_rows"] == 0
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_summary_names_bucket_shortfalls() -> None:
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "GOOD",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "status": "PASS",
                "actual_support_scope": "actual_ticker_bucket",
                "actual_support_status": "PASS",
                "actual_support_sample_size": 3,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": 30,
                "diagnostic_replay_relaxed_dimensions": "",
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "SHORT",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 10,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": 30,
                "diagnostic_replay_relaxed_dimensions": "dte_bucket",
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "FAM",
                "strategy_route": "bull_call_debit",
                "entry_type": "DEBIT",
                "dte_bucket": "dte_15_30",
                "economics_bucket": "debit_reward_risk_mid",
                "liquidity_bucket": "liquidity_adequate",
                "status": "WARN",
                "actual_support_scope": "actual_strategy_family",
                "actual_support_status": "BLOCK",
                "actual_support_sample_size": 42,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 4,
                "diagnostic_replay_relaxed_dimensions": "liquidity_bucket",
            },
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    summary = core.summarize_profitability_calibration(calibration)
    blocker_detail = core._profitability_calibration_blocker_detail(summary)
    examples_detail = core._calibration_bucket_examples_detail(summary)

    assert summary["bucket_precision_rows"] == 2
    assert summary["bucket_shortfall_rows"] == 2
    assert summary["bucket_shortfall_routes"] == ["bull_call_debit", "short_put"]
    assert len(summary["bucket_blocker_examples"]) == 2
    short_example = next(item for item in summary["bucket_blocker_examples"] if item["ticker"] == "SHORT")
    assert short_example["actual_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 10
    assert short_example["replay_sample_gap"] == 0
    assert "bucket_shortfall_rows=2 routes=bull_call_debit,short_put" in blocker_detail
    assert "SHORT short_put/direction_unknown/dte_31_60/credit_standard" in examples_detail
    assert "missing_dims=dte_bucket" in examples_detail


def test_profitability_gap_plan_names_exact_bucket_evidence_steps() -> None:
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "PG",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": 10,
                "actual_support_sample_gap": core.MIN_EXPECTANCY_SAMPLE_SIZE - 10,
                "actual_support_avg_pnl": 14.0,
                "actual_support_profit_factor": 1.5,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "replay_bucket_sample_gap": 0,
                "replay_bucket_avg_pnl": 20.0,
                "replay_bucket_profit_factor": 2.0,
                "diagnostic_replay_status": "PASS",
                "diagnostic_replay_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "diagnostic_replay_relaxed_dimensions": "dte_bucket",
                "current_ticket_count": 1,
                "source_path": "replay.csv",
                "note": "short put needs actual samples",
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "XLF",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": 10,
                "actual_support_sample_gap": core.MIN_EXPECTANCY_SAMPLE_SIZE - 10,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "replay_bucket_sample_gap": 0,
                "diagnostic_replay_status": "PASS",
                "diagnostic_replay_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "diagnostic_replay_relaxed_dimensions": "dte_bucket",
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "KO",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "debit_reward_risk_weak",
                "liquidity_bucket": "liquidity_unknown",
                "status": "WARN",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "actual_support_sample_gap": 0,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 1,
                "replay_bucket_sample_gap": core.MIN_EXPECTANCY_SAMPLE_SIZE - 1,
                "diagnostic_replay_status": "WARN",
                "diagnostic_replay_sample_size": 1,
                "diagnostic_replay_relaxed_dimensions": "liquidity_bucket",
                "current_ticket_count": 1,
            },
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    gap_plan = core.build_profitability_gap_plan(calibration)
    summary = core.summarize_profitability_gap_plan(gap_plan)
    detail = core._profitability_gap_plan_detail(summary)

    assert list(gap_plan.columns) == core.PROFITABILITY_GAP_PLAN_COLUMNS
    short_put = gap_plan[gap_plan["strategy_route"].eq("short_put")].iloc[0]
    assert short_put["current_tickers"] == "PG,XLF"
    assert short_put["current_ticket_count"] == 2
    assert short_put["primary_gap"] == "actual_closed_outcomes_sample_gap"
    assert "Need 20 more positive closed/forward outcomes" in short_put["next_evidence_needed"]
    assert "Nearest replay support only appears after relaxing dte_bucket" in short_put["next_evidence_needed"]
    debit = gap_plan[gap_plan["strategy_route"].eq("bull_call_debit")].iloc[0]
    assert debit["primary_gap"] == "replay_exact_bucket_sample_gap"
    assert "Need 29 more leakage-safe replay outcomes" in debit["next_evidence_needed"]
    assert summary["blocking_rows"] == 2
    assert summary["primary_gap_counts"]["actual_closed_outcomes_sample_gap"] == 1
    assert "PG,XLF short_put actual_closed_outcomes_sample_gap" in detail


def test_route_opportunity_gap_surfaces_near_ready_long_call_without_promoting(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"LC{idx}",
            "realized_pnl": 100.0,
            "strategy": "Long Call",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 1)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"LC{idx}",
            "strategy": "Long Call",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 90.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("long_call")].iloc[0]
    assert row["route_status"] == "near_ready_more_actual_sample_needed"
    assert row["actual_status"] == "WARN"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 1
    assert row["replay_status"] == "PASS"
    assert row["current_ticket_count"] == 0
    assert "actual_route_sample_below_30" in row["development_gap"]
    assert summary["near_ready_routes"] == ["long_call"]
    assert summary["candidate_expansion_routes"] == []


def test_route_opportunity_gap_requires_bucket_calibration_before_execution_gap_status(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"CSP{idx}",
            "realized_pnl": 100.0,
            "strategy": "Short Put",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    wheel_dir = tmp_path / "out" / "fresh_wheel_replay_2026_full_ytd"
    wheel_dir.mkdir(parents=True)
    wheel_rows = [
        {
            "signal_date": "2026-04-01",
            "ticker": f"CSP{idx}",
            "action": "OPEN_CSP",
            "entry_credit": 2.00,
            "dte": 38,
            "exit_date": "2026-05-01",
            "pnl_per_contract": 125.0,
            "outcome_status": "scored",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(wheel_rows).to_csv(
        wheel_dir / "fresh-wheel-replay-outcomes-2026-01-02_2026-05-01.csv",
        index=False,
    )
    decision = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Short Put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 2.00 CREDIT",
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "status": "WARN",
            }
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    gap = core.build_route_opportunity_gap(
        tmp_path,
        decision,
        pd.DataFrame(),
        calibration,
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("short_put")].iloc[0]
    assert row["actual_status"] == "PASS"
    assert row["replay_status"] == "PASS"
    assert row["current_ticket_count"] == 1
    assert row["calibration_pass_rows"] == 0
    assert row["calibration_warn_rows"] == 1
    assert row["route_status"] == "current_rows_need_bucket_calibration"
    assert row["development_gap"] == "current_rows_need_route_bucket_calibration"
    assert summary["bucket_calibration_routes"] == ["short_put"]
    assert summary["current_route_execution_gap_routes"] == []
    assert "bucket_calibration_needed=short_put" in core._route_opportunity_gap_detail(summary)


def test_route_opportunity_gap_uses_leakage_safe_pattern_validation_replay_for_long_call(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"LC{idx}",
            "realized_pnl": 100.0,
            "strategy": "Long Call",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 1)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    validation_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-06-09"
    validation_dir.mkdir(parents=True)
    validation_rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Long Call Debit",
            "net_r": 0.40,
            "signal_date": "2026-05-01",
            "target_date": "2026-05-15",
            "managed_exit_date": "",
            "lead_option_symbol": "VAL260620C00100000",
            "ticker": f"VAL{idx}",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    validation_rows.extend(
        [
            {
                "sample": "TRAIN",
                "status": "SCORED",
                "blocked": False,
                "strategy_type": "Long Call Debit",
                "net_r": -5.0,
                "signal_date": "2026-05-01",
                "target_date": "2026-05-15",
                "managed_exit_date": "",
                "lead_option_symbol": "TRAIN260620C00100000",
                "ticker": "TRAIN",
            },
            {
                "sample": "VALIDATION",
                "status": "SCORED",
                "blocked": False,
                "strategy_type": "Long Call Debit",
                "net_r": -5.0,
                "signal_date": "2026-05-01",
                "target_date": "2026-06-20",
                "managed_exit_date": "",
                "lead_option_symbol": "FUTURE260620C00100000",
                "ticker": "FUTURE",
            },
            {
                "sample": "VALIDATION",
                "status": "SCORED",
                "blocked": True,
                "strategy_type": "Long Call Debit",
                "net_r": -5.0,
                "signal_date": "2026-05-01",
                "target_date": "2026-05-15",
                "managed_exit_date": "",
                "lead_option_symbol": "BLOCK260620C00100000",
                "ticker": "BLOCKED",
            },
        ]
    )
    pd.DataFrame(validation_rows).to_csv(validation_dir / "validation_details.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )

    row = gap[gap["strategy_route"].eq("long_call")].iloc[0]
    assert row["route_status"] == "near_ready_more_actual_sample_needed"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 1
    assert row["replay_status"] == "PASS"
    assert row["replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_avg_pnl"] == 0.4
    assert "validation_details.csv" in row["source_path"]
    replay, _, _ = core._pattern_validation_replay_frame(tmp_path / "out", as_of=dt.date(2026, 6, 9))
    long_call = replay[replay["strategy_route"].eq("long_call")]
    assert set(long_call["dte_bucket"]) == {"dte_31_60"}
    assert set(long_call["economics_bucket"]) == {"debit_unknown"}


def test_pattern_validation_replay_buckets_credit_spread_width_from_legs_json(tmp_path: Path) -> None:
    validation_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-06-09"
    validation_dir.mkdir(parents=True)
    legs_json = json.dumps(
        [
            {"action": "SELL", "option_symbol": "SPY260620P00540000", "strike": 540.0},
            {"action": "BUY", "option_symbol": "SPY260620P00535000", "strike": 535.0},
        ]
    )
    rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Bull Put Credit Spread",
            "net_r": 0.25,
            "signal_date": "2026-05-21",
            "target_date": "2026-05-28",
            "lead_option_symbol": "SELL SPY260620P00540000 / BUY SPY260620P00535000",
            "entry_credit": 2.00,
            "legs_json": legs_json,
            "ticker": f"SPY{idx}",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(rows).to_csv(validation_dir / "validation_details.csv", index=False)

    replay, _, _ = core._pattern_validation_replay_frame(tmp_path / "out", as_of=dt.date(2026, 6, 9))

    bull_put = replay[replay["strategy_route"].eq("bull_put_credit")]
    assert len(bull_put) == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert set(bull_put["dte_bucket"]) == {"dte_15_30"}
    assert set(bull_put["economics_bucket"]) == {"credit_width_high"}


def test_route_opportunity_gap_blocks_negative_actual_vertical_route_despite_replay(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"BPC{idx}",
            "realized_pnl": -80.0,
            "strategy": "Bull Put Credit Spread",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"BPC{idx}",
            "strategy": "Bull Put Credit Spread",
            "strategy_kind": "Credit",
            "entry_side": "credit",
            "dte": 28,
            "iv_rank": 45,
            "entry_credit_pct_width": 0.32,
            "source_contract_oi": 1200,
            "pnl_1x": 70.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "structure": "Bull Put Credit Spread",
                "trade_plan": "SELL 1 SPY 2026-06-30 540 Put / BUY 1 SPY 2026-06-30 535 Put @ 1.60 CREDIT",
            }
        ]
    )

    gap = core.build_route_opportunity_gap(
        tmp_path,
        decision,
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("bull_put_credit")].iloc[0]
    assert row["route_status"] == "actual_outcomes_negative_or_weak"
    assert row["actual_status"] == "BLOCK"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_status"] == "PASS"
    assert row["current_ticket_count"] == 1
    assert row["suggested_action"] == "Do not promote this route; require new positive closed-trade evidence before green eligibility."
    assert summary["negative_or_weak_routes"] == ["bull_put_credit"]


def test_profitability_calibration_blocks_ready_looking_green_row() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NOCAL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NOCAL 2026-06-19 100 Call / BUY 1 NOCAL 2026-06-19 105 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 NOCAL 2026-06-19 100 Call / BUY 1 NOCAL 2026-06-19 105 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 10,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 10,
                "profitability_calibration_status": "BLOCK",
                "profitability_calibration_sample_size": 0,
                "profitability_calibration_actual_status": "BLOCK",
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_note": "bucket missing",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets["ticker"].tolist() == ["NOCAL"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["order_readiness"].tolist() == ["target_order_after_profitability_calibration"]


def test_profitability_calibration_blocker_is_visible_on_yellow_target_row() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "CALWAIT",
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 CALWAIT 2026-06-19 100 Call / BUY 1 CALWAIT 2026-06-19 105 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 CALWAIT 2026-06-19 100 Call / BUY 1 CALWAIT 2026-06-19 105 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "TARGET_QUOTE_REFRESH",
                "status_reason": "dated UW target from EOD; fresh Schwab chain target quote required",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "profitability_calibration_status": "WARN",
                "profitability_calibration_sample_size": 12,
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_note": "needs more route-precise actual evidence",
            }
        ]
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    tickets = core.build_trade_tickets(decision)

    assert core.PROFITABILITY_CALIBRATION_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets["order_readiness"].tolist() == ["target_order_after_profitability_calibration"]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in tickets["execution_blockers"].iloc[0]


def test_negative_actual_calibration_support_stays_off_yellow_target_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BADBUCKET",
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 BADBUCKET 2026-07-17 100 Put @ 1.50 CREDIT",
                "trade_plan": "SELL 1 BADBUCKET 2026-07-17 100 Put @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 9850.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "TARGET_QUOTE_REFRESH",
                "status_reason": "dated UW target from EOD; fresh Schwab chain target quote required",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 3,
                "profitability_calibration_actual_avg_pnl": -111.33,
                "profitability_calibration_actual_profit_factor": 0.032,
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_note": "actual bucket is under-sampled and losing",
            }
        ]
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    tickets = core.build_trade_tickets(decision)

    assert decision["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in decision["execution_blockers"].iloc[0]
    assert core.PROFITABILITY_CALIBRATION_ACTUAL_NEGATIVE_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets.empty


def test_uncalibrated_low_profit_row_stays_off_yellow_target_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "SMALLWARN",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 SMALLWARN 2026-07-17 100 Put @ 1.50 CREDIT",
                "trade_plan": "SELL 1 SMALLWARN 2026-07-17 100 Put @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 9850.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 34,
                "profitability_calibration_actual_avg_pnl": 92.09,
                "profitability_calibration_actual_profit_factor": 1.823,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_note": "route-level support only; exact bucket still needs proof",
            }
        ]
    )

    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in decision["execution_blockers"].iloc[0]
    assert core.POSITION_PROFIT_MATERIALITY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets.empty


def test_report_labels_no_trade_section_as_preview_of_full_csv() -> None:
    no_trade = pd.DataFrame(
        [
            {"ticker": f"MISS{idx}", "bias": "bearish", "score": 70 - idx, "reason": f"candidate {idx}"}
            for idx in range(22)
        ]
    )

    report = core.render_report("2026-05-22", pd.DataFrame(), no_trade, {"row_counts": {}, "warnings": []})

    assert "Showing first 20 of 22 rows; full audit is in `no_trade_audit.csv`." in report
    assert "2 additional no-trade rows in no_trade_audit.csv" in report


def test_report_uses_position_scaled_profit_loss_for_target_order_tables() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "GOOGL",
                "ready_to_enter": False,
                "execution_status": "waiting_for_price",
                "execution_gate_status": "pass",
                "execution_blockers": "send_now_credit_width_below_30pct",
                "target_order_status": "target_order_candidate",
                "suggested_contracts": 4,
                "trade_plan": "SELL 1 GOOGL 2026-06-05 392.5 Call / BUY 1 GOOGL 2026-06-05 395 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "target_exit": 0.23,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "max_position_loss": 740.0,
                "live_validation_status": "PASS",
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "execution_confidence_score": 88,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "underlying_quality_tier": "core",
                "status_reason": "fixture target row",
            }
        ]
    )

    report = core.render_report(
        "2026-05-22",
        final,
        pd.DataFrame(),
        {"row_counts": {}, "warnings": []},
    )

    assert "Max Profit" in report
    assert "Max Loss" in report
    assert "Target Orders - Target Credits/Debits" in report
    assert (
        "| GOOGL | 🟡 YELLOW target | Call credit spread | 2026-06-05 | "
        "SELL 1 GOOGL 2026-06-05 392.5 Call | BUY 1 GOOGL 2026-06-05 395 Call | "
        "4 | 0.65 CREDIT | 0.23 | 260.0 | 740.0 | HIGH / 88 | credit/width too weak for send-now |"
    ) in report


def test_report_sanitizes_market_session_blocker_in_execution_quality() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_blockers": "market_session_open_required",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "HIGH",
                "target_order_status": "target_order_candidate",
                "trade_plan": "SELL 1 SPY 2026-06-05 600 Call / BUY 1 SPY 2026-06-05 605 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "underlying_quality_tier": "core",
            }
        ]
    )

    report = core.render_report("2026-06-09", final, pd.DataFrame(), {"row_counts": {}, "warnings": []})

    assert "market_session_open_required" not in report
    assert "fresh quote refresh" in report
    assert "work_target_limit_after_market_open_recheck" not in report
    assert "target_order_after_market_open_and_live_recheck" not in report
    assert "Market Open Recheck Queue" not in report
    assert "Market-open recheck" not in report


def test_report_target_order_table_uses_trade_ticket_surface_filters() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "PEP",
                "ready_to_enter": False,
                "execution_status": "needs_review",
                "execution_gate_status": "pass",
                "execution_blockers": "fresh_live_schwab_required",
                "target_order_status": "target_order_candidate",
                "suggested_contracts": 5,
                "trade_plan": "BUY 1 PEP 2026-06-18 155 Call / SELL 1 PEP 2026-06-18 160 Call @ 0.88 DEBIT",
                "entry_limit": 0.88,
                "target_exit": 1.58,
                "max_profit": 412.0,
                "max_loss": 88.0,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 5,
                "execution_confidence_score": 73,
                "underlying_quality_tier": "core",
            },
            {
                "recommendation_rank": 2,
                "ticker": "UNH",
                "ready_to_enter": False,
                "execution_status": "needs_review",
                "execution_gate_status": "pass",
                "execution_blockers": "ticker_agentic_review_coverage_below_threshold",
                "target_order_status": "target_order_candidate",
                "suggested_contracts": 4,
                "trade_plan": "BUY 1 UNH 2026-06-18 380 Put / SELL 1 UNH 2026-06-18 370 Put @ 3.38 DEBIT",
                "entry_limit": 3.38,
                "target_exit": 6.08,
                "max_profit": 662.0,
                "max_loss": 338.0,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 2,
                "execution_confidence_score": 72,
                "underlying_quality_tier": "core",
            },
        ]
    )

    report = core.render_report("2026-05-15", final, pd.DataFrame(), {"row_counts": {}, "warnings": []})

    assert (
        "| PEP | 🟡 YELLOW target | Call debit spread | 2026-06-18 | "
        "SELL 1 PEP 2026-06-18 160 Call | BUY 1 PEP 2026-06-18 155 Call | "
        "5 | 0.88 DEBIT | 1.58 | 2060.0 | 440.0 | MEDIUM / 73 | fresh Schwab chain |"
    ) in report
    assert "UNH |" not in report


def test_report_coverage_audit_blanks_nan_rank_values() -> None:
    coverage = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "underlying_quality_tier": "core",
                "raw_rank": 23.0,
                "candidate_rank": pd.NA,
                "bias": "neutral",
                "score": 72.7100,
                "coverage_status": "NO_DIRECTIONAL_EDGE",
                "status_color": "gray",
                "reason": "neutral flow bias",
                "next_step": "wait for directional flow",
            },
            {
                "ticker": "URA",
                "underlying_quality_tier": "excluded",
                "raw_rank": 3847.0,
                "candidate_rank": 2744.0,
                "bias": "bearish",
                "score": 17.13,
                "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                "status_color": "yellow",
                "reason": "excluded underlying",
                "next_step": "run structure expansion or live-chain construction",
            }
        ]
    )

    report = core.render_report(
        "2026-05-22",
        pd.DataFrame(),
        pd.DataFrame(),
        {"row_counts": {}, "warnings": []},
        coverage,
    )

    assert "| NVDA | ⚪ GRAY no-edge | neutral | 72.71 | GRAY no-edge | neutral flow bias | wait for directional flow |" in report
    assert "| URA | 🔴 RED no-action |" in report
    assert "YELLOW candidate | URA" not in report
    assert "nan" not in report.lower()


def test_coverage_audit_marks_speculative_and_excluded_candidates_non_actionable() -> None:
    candidates = pd.DataFrame(
        [
            {
                "ticker": "URA",
                "bias": "bearish",
                "score": 17.13,
                "underlying_quality_tier": "excluded",
                "underlying_quality_reason": "non-core ETF; not in actionable ETF allowlist",
                "flow_reason": "excluded underlying",
            },
            {
                "ticker": "OKLO",
                "bias": "neutral",
                "score": 31.53,
                "underlying_quality_tier": "speculative",
                "underlying_quality_reason": "liquidity below actionable thresholds",
                "flow_reason": "speculative underlying",
            },
            {
                "ticker": "DVN",
                "bias": "bullish",
                "score": 71.63,
                "underlying_quality_tier": "liquid",
                "underlying_quality_reason": "liquid non-core underlying",
                "flow_reason": "liquid candidate",
            },
        ]
    )

    coverage = core.build_coverage_audit(
        pd.DataFrame(),
        candidates,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        watchlist=["URA", "OKLO", "DVN"],
    )

    by_ticker = coverage.set_index("ticker")
    assert by_ticker.loc["URA", "coverage_status"] == "NON_ACTIONABLE_UNDERLYING"
    assert by_ticker.loc["URA", "status_color"] == "red"
    assert by_ticker.loc["OKLO", "coverage_status"] == "NON_ACTIONABLE_UNDERLYING"
    assert by_ticker.loc["OKLO", "status_color"] == "red"
    assert by_ticker.loc["DVN", "coverage_status"] == "NON_ACTIONABLE_UNDERLYING"
    assert by_ticker.loc["DVN", "status_color"] == "red"

    report = core.render_report(
        "2026-05-22",
        pd.DataFrame(),
        pd.DataFrame(),
        {"row_counts": {}, "warnings": []},
        coverage,
    )

    assert "| URA | 🔴 RED no-action |" in report
    assert "| OKLO | 🔴 RED no-action |" in report
    assert "YELLOW candidate | URA" not in report
    assert "YELLOW candidate | OKLO" not in report
    assert "| DVN | 🔴 RED no-action |" in report


def test_action_surface_underlying_quality_proof_blocks_audit_only_names_on_action_surfaces() -> None:
    packet = audit.build_action_surface_underlying_quality_proof_packet(
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "URA",
                    "underlying_quality_tier": "excluded",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "OKLO",
                    "underlying_quality_tier": "speculative",
                }
            ]
        ),
        focus_coverage=pd.DataFrame(
            [
                {
                    "ticker": "LOWQ",
                    "underlying_quality_tier": "unknown",
                    "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                }
            ]
        ),
    )

    assert packet["status"].tolist() == ["FAIL_LOW_QUALITY_UNDERLYING_ACTION_SURFACE"]
    assert packet["ticket_bad_underlying_rows"].tolist() == [1]
    assert packet["market_open_recheck_bad_underlying_rows"].tolist() == [1]
    assert packet["focus_bad_actionable_rows"].tolist() == [1]
    assert packet["ticket_bad_tickers"].tolist() == ["URA"]
    assert packet["market_open_recheck_bad_tickers"].tolist() == ["OKLO"]
    assert packet["focus_bad_actionable_tickers"].tolist() == ["LOWQ"]


def test_action_surface_underlying_quality_proof_blocks_liquid_non_core_action_rows() -> None:
    packet = audit.build_action_surface_underlying_quality_proof_packet(
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "DVN",
                    "underlying_quality_tier": "liquid",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "underlying_quality_tier": "core",
                }
            ]
        ),
        focus_coverage=pd.DataFrame(
            [
                {
                    "ticker": "URA",
                    "underlying_quality_tier": "excluded",
                    "coverage_status": "NON_ACTIONABLE_UNDERLYING",
                }
            ]
        ),
    )

    assert packet["status"].tolist() == ["FAIL_LOW_QUALITY_UNDERLYING_ACTION_SURFACE"]
    assert packet["ticket_bad_underlying_rows"].tolist() == [1]
    assert packet["market_open_recheck_bad_underlying_rows"].tolist() == [0]
    assert packet["focus_bad_actionable_rows"].tolist() == [0]
    assert packet["audit_only_focus_rows"].tolist() == [1]
    assert packet["audit_only_focus_tickers"].tolist() == ["URA"]
    assert packet["liquid_non_core_action_tickers"].tolist() == ["DVN"]


def test_action_surface_underlying_quality_proof_passes_core_only_action_rows() -> None:
    packet = audit.build_action_surface_underlying_quality_proof_packet(
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "underlying_quality_tier": "core",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "MSFT",
                    "underlying_quality_tier": "core",
                }
            ]
        ),
        focus_coverage=pd.DataFrame(
            [
                {
                    "ticker": "URA",
                    "underlying_quality_tier": "excluded",
                    "coverage_status": "NON_ACTIONABLE_UNDERLYING",
                }
            ]
        ),
    )

    assert packet["status"].tolist() == ["PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"]
    assert packet["ticket_bad_underlying_rows"].tolist() == [0]
    assert packet["market_open_recheck_bad_underlying_rows"].tolist() == [0]
    assert packet["focus_bad_actionable_rows"].tolist() == [0]


def test_duplicate_catalyst_reviews_do_not_expand_or_crash_dispatch() -> None:
    candidates = pd.DataFrame(
        [
            {"ticker": "MSFT", "bias": "bullish", "score": 80, "flow_reason": "call flow"},
            {"ticker": "MSFT", "bias": "bullish", "score": 75, "flow_reason": "follow-on flow"},
        ]
    )
    catalyst_reviews = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "catalyst_status": "clear",
                "catalyst_note": "first review",
                "days_to_earnings": "",
                "news_sentiment": "neutral",
                "red_flag_terms": "",
                "support_terms": "",
                "objective_blocker": False,
            },
            {
                "ticker": "MSFT",
                "catalyst_status": "clear",
                "catalyst_note": "duplicate review",
                "days_to_earnings": "",
                "news_sentiment": "neutral",
                "red_flag_terms": "",
                "support_terms": "",
                "objective_blocker": False,
            },
        ]
    )
    priced = pd.DataFrame([{"ticker": "MSFT", "recommendation_status": RecommendationStatus.ENTER.value}])

    tasks = core.build_research_tasks(candidates, {"regime": "risk_on"}, catalyst_reviews, top_trades=1)
    merged = core.apply_catalyst_reviews(priced, catalyst_reviews)

    assert len(tasks["tasks"]) == 2
    assert {task["catalyst_note"] for task in tasks["tasks"]} == {"first review"}
    assert len(merged) == 1
    assert merged["catalyst_note"].tolist() == ["first review"]


def test_run_pipeline_writes_independent_recommendation_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    _write_minimal_uw_fixture(root)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3)
    manifest = json.loads(paths["manifest"].read_text())
    orchestration = json.loads(paths["agent_orchestration"].read_text())
    research_tasks = json.loads(paths["research_tasks"].read_text())
    dispatch_plan = json.loads(paths["agent_dispatch_plan"].read_text())
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    green_tickets = pd.read_csv(paths["green_trade_tickets"])
    target_tickets = pd.read_csv(paths["target_order_candidates"])
    market_open_queue = pd.read_csv(paths["market_open_recheck_queue"])
    catalyst = pd.read_csv(paths["catalyst_reviews"])
    review_board = pd.read_csv(paths["agent_review_board"])
    structure_attempts = pd.read_csv(paths["structure_attempts"])
    strategy_routing = pd.read_csv(paths["strategy_routing_audit"])
    live_quality = pd.read_csv(paths["live_spread_quality_audit"])
    execution_fill_quality = pd.read_csv(paths["execution_fill_quality"])
    sizing = pd.read_csv(paths["sizing_audit"])
    management = pd.read_csv(paths["management_plan"])
    expectancy = pd.read_csv(paths["expectancy_evidence"])
    strategy_atlas = pd.read_csv(paths["strategy_outcome_atlas"])
    profitability_calibration = pd.read_csv(paths["profitability_calibration"])
    profitability_gap_plan = pd.read_csv(paths["profitability_gap_plan"])
    route_gap = pd.read_csv(paths["route_opportunity_gap"])
    feasibility = pd.read_csv(paths["monthly_feasibility"])
    confidence_audit = pd.read_csv(paths["confidence_audit"])
    confidence_summary = json.loads(paths["confidence_audit_json"].read_text())
    report = paths["report"].read_text(encoding="utf-8")

    assert paths["out_dir"] == root / "out" / "options_agent" / "2026-05-22"
    assert manifest["pipeline_name"] == "Options Agent"
    assert "codexdaily_v4" not in json.dumps(manifest)
    assert manifest["row_counts"]["agent_review_board"] == len(review_board)
    assert manifest["row_counts"]["agent_dispatch_tasks"] == len(dispatch_plan["subagent_tasks"])
    assert manifest["row_counts"]["structure_attempts"] == len(structure_attempts)
    assert manifest["row_counts"]["strategy_routing_audit"] == len(strategy_routing)
    assert manifest["row_counts"]["live_spread_quality_audit"] == len(live_quality)
    assert manifest["row_counts"]["execution_fill_quality"] == len(execution_fill_quality)
    assert manifest["row_counts"]["sizing_audit"] == len(sizing)
    assert manifest["row_counts"]["management_plan"] == len(management)
    assert manifest["row_counts"]["expectancy_evidence"] == len(expectancy)
    assert manifest["row_counts"]["strategy_outcome_atlas"] == len(strategy_atlas)
    assert manifest["row_counts"]["profitability_calibration"] == len(profitability_calibration)
    assert manifest["row_counts"]["profitability_gap_plan"] == len(profitability_gap_plan)
    assert manifest["row_counts"]["route_opportunity_gap"] == len(route_gap)
    assert manifest["row_counts"]["confidence_audit"] == len(confidence_audit)
    assert manifest["row_counts"]["market_open_recheck_queue"] == len(market_open_queue)
    assert manifest["row_counts"]["catalyst_evidence"] == 1
    assert manifest["agent_review_summary"]["by_agent_type"]["built_in"] == len(review_board)
    assert manifest["agent_review_summary"]["portfolio_risk_only"] == 1
    assert manifest["agentic_orchestration"]["status"] == "awaiting_subagents"
    assert "Profitability gap plan" in report
    assert manifest["artifacts"]["profitability_gap_plan"].endswith("profitability_gap_plan.csv")
    assert manifest["artifacts"]["execution_fill_quality"].endswith("execution_fill_quality.csv")
    assert research_tasks["schema_version"] == "options_agent.dispatch_tasks.v1"
    assert research_tasks["dispatch_model"] == "codex_subagents"
    assert research_tasks["tasks"][0]["candidate_id"].startswith("WMT:")
    assert dispatch_plan["dispatch_tool"] == "multi_agent_v1.spawn_agent"
    assert {task["agent"] for task in dispatch_plan["subagent_tasks"]} == {
        "catalyst_news",
        "macro_regime",
        "structure_builder",
        "skeptic",
        "portfolio_management",
    }
    assert {"candidate_id", "agent_type", "review_stage", "portfolio_risk_only", "source_artifact"}.issubset(review_board.columns)
    assert {"market_regime", "catalyst", "structure", "skeptic", "portfolio_risk"}.issubset(set(review_board["agent"]))
    assert orchestration["execution_model"].startswith("two-pass Codex multi-agent dispatch")
    assert {"from": "research_dispatch", "to": "external_subagents", "artifact": "research_tasks.json"} in orchestration[
        "handoffs"
    ]
    assert {"from": "research_dispatch", "to": "codex_subagents", "artifact": "agent_dispatch_plan.json"} in orchestration[
        "handoffs"
    ]
    assert {"from": "codex_subagents", "to": "research_dispatch", "artifact": "agentic_reviews.json"} in orchestration[
        "handoffs"
    ]
    assert {"from": "structure", "to": "synthesis", "artifact": "strategy_routing_audit.csv"} in orchestration["handoffs"]
    assert dispatch_plan["common_context"]["input_artifacts"]["strategy_routing_audit"].endswith("strategy_routing_audit.csv")
    assert strategy_routing["strategy"].tolist() == ["bull_call_debit", "bull_put_credit"]
    assert {"from": "external_subagents", "to": "research_dispatch", "artifact": "external_agent_reviews.csv"} in orchestration[
        "handoffs"
    ]
    assert {"from": "research_dispatch", "to": "synthesis", "artifact": "agent_review_board.csv"} in orchestration["handoffs"]
    assert {"from": "structure", "to": "synthesis", "artifact": "structure_attempts.csv"} in orchestration["handoffs"]
    assert {"from": "portfolio_risk", "to": "synthesis", "artifact": "final_recommendations.csv"} in orchestration["handoffs"]
    assert catalyst["ticker"].tolist() == ["WMT"]
    assert structure_attempts["attempt_stage"].tolist() == ["dated_hot_chain"]
    assert structure_attempts["attempt_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert "codexdaily_v4" not in structure_attempts.to_json()
    assert final["ticker"].tolist() == ["WMT"]
    assert final["visible_in_final_board"].tolist() == [True]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert final["max_profit"].tolist() == [100.0]
    assert decision["status_label"].tolist() == ["YELLOW review"]
    assert decision["status_icon"].tolist() == ["🟡"]
    assert decision["final_action"].tolist() == [RecommendationStatus.REVIEW.value]
    assert decision["execution_status"].tolist() == ["needs_review"]
    assert tickets.empty
    assert green_tickets.empty
    assert target_tickets.empty
    assert {"trade_plan", "sell_leg", "buy_leg", "expiry"}.issubset(tickets.columns)
    assert {"status_icon", "status_label"}.issubset(tickets.columns)
    assert "full_ticket" not in tickets.columns
    assert "live Schwab validation was not requested" in "; ".join(manifest["warnings"])
    assert decision["trade_plan"].tolist() == [
        "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT"
    ]
    assert management["management_action"].tolist() == [RecommendationStatus.REVIEW.value]
    assert manifest["expectancy_evidence_summary"]["status"] == "not_proven"
    assert "strategy_outcome_atlas_summary" in manifest
    assert manifest["confidence_audit_summary"]["status"] == "block"
    assert manifest["confidence_audit_summary"]["order_entry_confidence_rating"] == 0.0
    assert confidence_summary["status"] == "block"
    assert confidence_summary["order_entry_confidence_rating"] == 0.0
    assert confidence_audit["metric"].tolist() == [
        "profitability_confidence_rating",
        "order_entry_confidence_rating",
        "goal_confidence_gate",
    ]
    assert "expectancy_evidence" in feasibility["metric"].tolist()
    assert "Expectancy evidence" not in report
    assert "Monthly Readiness Gate" not in report
    assert "Monthly target is not proven" not in report
    assert "Structure attempt rows: 1" in report
    assert "Live spread quality audit" in report
    assert "Execution fill quality" in report
    assert "Send Now Orders" in report
    assert "Target Orders - Target Credits/Debits" in report
    assert report.index("## Send Now Orders") < report.index("## Target Orders")
    assert "Structural status counts, not order readiness" in report
    assert "Target rows show desired credits/debits" in report
    assert "Profitability confidence:" in report
    assert "Order-entry confidence: 0.0/10" in report
    assert "Profitability calibration:" in report
    assert "Route opportunity gaps:" in report
    assert "Strategy outcome atlas:" in report
    assert "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT" not in report
    assert "No green send-now orders" in report
    assert "WMT260619P00095000" not in report


def test_dispatch_only_writes_subagent_dispatch_plan_without_synthesis(tmp_path: Path) -> None:
    root = tmp_path
    _write_minimal_uw_fixture(root)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3, dispatch_only=True)
    manifest = json.loads(paths["manifest"].read_text())
    research_tasks = json.loads(paths["research_tasks"].read_text())
    dispatch_plan = json.loads(paths["agent_dispatch_plan"].read_text())
    orchestration = json.loads(paths["agent_orchestration"].read_text())
    final = pd.read_csv(paths["final_recommendations"])
    review_board = pd.read_csv(paths["agent_review_board"])
    agentic_reviews = json.loads(paths["agentic_reviews"].read_text())
    confidence_audit = pd.read_csv(paths["confidence_audit"])
    confidence_summary = json.loads(paths["confidence_audit_json"].read_text())
    strategy_atlas = pd.read_csv(paths["strategy_outcome_atlas"])
    profitability_gap_plan = pd.read_csv(paths["profitability_gap_plan"])
    route_gap = pd.read_csv(paths["route_opportunity_gap"])
    execution_fill_quality = pd.read_csv(paths["execution_fill_quality"])

    assert manifest["mode"] == "agentic_dispatch_pass"
    assert manifest["agentic_orchestration"]["status"] == "dispatch_ready"
    assert manifest["row_counts"]["research_tasks"] == 1
    assert manifest["row_counts"]["agent_dispatch_tasks"] == 5
    assert manifest["row_counts"]["final_recommendations"] == 0
    assert manifest["row_counts"]["confidence_audit"] == 3
    assert manifest["row_counts"]["strategy_outcome_atlas"] == 0
    assert manifest["row_counts"]["profitability_gap_plan"] == 0
    assert manifest["row_counts"]["route_opportunity_gap"] == 0
    assert manifest["row_counts"]["execution_fill_quality"] == 0
    assert strategy_atlas.empty
    assert profitability_gap_plan.empty
    assert route_gap.empty
    assert execution_fill_quality.empty
    assert confidence_audit["metric"].tolist() == [
        "profitability_confidence_rating",
        "order_entry_confidence_rating",
        "goal_confidence_gate",
    ]
    assert confidence_summary["order_entry_confidence_rating"] == 0.0
    assert research_tasks["dispatch_model"] == "codex_subagents"
    assert dispatch_plan["dispatch_status"] == "ready_for_codex_subagents"
    assert len(dispatch_plan["subagent_tasks"]) == 5
    assert agentic_reviews == {"reviews": []}
    assert final.empty
    assert review_board.empty
    assert {"from": "research_dispatch", "to": "codex_subagents", "artifact": "agent_dispatch_plan.json"} in orchestration[
        "handoffs"
    ]


def test_local_news_red_flag_keeps_live_validated_trade_in_review(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    _write_wmt_red_flag_news(root)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3, chain_snapshot_dir=snapshot_dir)
    evidence = pd.read_csv(paths["catalyst_evidence"])
    catalyst = pd.read_csv(paths["catalyst_reviews"])
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    review_board = pd.read_csv(paths["agent_review_board"])

    news_rows = evidence[evidence["evidence_type"].eq("local_news")]
    assert news_rows["evidence_status"].tolist() == ["news_red_flag"]
    assert "sec probe" in news_rows["red_flag_terms"].iloc[0]
    assert catalyst["catalyst_status"].tolist() == ["news_red_flag"]
    assert catalyst["news_sentiment"].tolist() == ["negative"]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert decision["execution_status"].tolist() == ["needs_review"]
    assert tickets.empty
    catalyst_reviews = review_board[review_board["agent"].eq("catalyst")]
    assert catalyst_reviews["verdict"].tolist() == ["caution"]
    assert catalyst_reviews["confidence"].tolist() == ["high"]


def test_legacy_chains_snapshot_layout_can_promote_ready_trade(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot_in_legacy_chains_layout(snapshot_dir)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3, chain_snapshot_dir=snapshot_dir)
    manifest = json.loads(paths["manifest"].read_text())
    live = pd.read_csv(paths["live_chain_validation"])
    live_quality = pd.read_csv(paths["live_spread_quality_audit"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])

    assert live["live_validation_status"].tolist() == ["PASS"]
    assert manifest["row_counts"]["live_spread_quality_audit"] == 1
    assert manifest["live_spread_quality_summary"]["status"] == "pass"
    assert live_quality["live_market_quality_status"].tolist() == ["PASS"]
    assert live["chain_source"].str.contains("chains/chain_WMT.json", regex=False).tolist() == [True]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert "fresh_live_schwab_required" in decision["execution_blockers"].iloc[0]
    assert "Schwab snapshot chain" in decision["status_reason"].iloc[0]
    assert "live Schwab chain" not in decision["status_reason"].iloc[0]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_snapshot_validation_can_fallback_to_debit_target_candidate(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_call_debit_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 100.0,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.35,
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "trade_quality_status": "reviewable",
                "status_reason": "dated credit structure missing",
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )
    context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=snapshot_dir,
        portfolio_context={"status": "missing", "total_value": 0},
        research_task_count=1,
        external_review_count=1,
        agent_reviews_json=tmp_path / "agentic_reviews.json",
    )
    updated["external_agent_distinct_review_count"] = 4
    updated["external_agent_review_count"] = 4
    updated["external_agent_review_agents"] = "catalyst_news; macro_regime; structure_builder; skeptic"
    updated["agent_support_count"] = 4
    updated = _mark_strategy_expectancy_pass(updated)
    decision = core.synthesize_decision_board(updated, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert updated["live_validation_status"].tolist() == ["PASS"]
    assert updated["recommendation_status"].tolist() == [RecommendationStatus.ENTER.value]
    assert updated["structure"].tolist() == ["bull call debit spread"]
    assert "DEBIT" in updated["trade_plan"].iloc[0]
    assert live["trade_plan"].str.contains("DEBIT", regex=False).tolist() == [True]
    assert tickets["entry_type"].tolist() == ["DEBIT"]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["order_readiness"].tolist() == ["target_order_after_portfolio_sizing"]


def test_live_validation_prefers_clean_debit_alternative_over_flow_anchored_reject(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_call_debit_with_better_breakout_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 100.0,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.35,
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "trade_quality_status": "reviewable",
                "status_reason": "dated credit structure missing",
                "iv30d": 0.30,
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )

    assert updated["live_validation_status"].tolist() == ["PASS"]
    assert updated["recommendation_status"].tolist() == [RecommendationStatus.ENTER.value]
    assert updated["structure"].tolist() == ["bull call debit spread"]
    assert updated["quality_gate_reason"].fillna("").tolist() == [""]
    assert updated["construction_source"].tolist() == ["lower_debit_better_reward_risk"]
    assert "110 Call" in updated["trade_plan"].iloc[0]
    assert "115 Call" in updated["trade_plan"].iloc[0]
    assert "debit_width_ratio_above_65pct" not in updated["status_reason"].iloc[0]
    assert live["trade_plan"].str.contains("110 Call", regex=False).tolist() == [True]


def test_live_validation_rejects_wide_live_markets_as_non_actionable(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_wide_market_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 95.0,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.35,
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "trade_quality_status": "reviewable",
                "status_reason": "dated credit structure requires live validation",
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )
    updated["suggested_contracts"] = 1
    updated["external_agent_distinct_review_count"] = 4
    updated["external_agent_review_count"] = 4
    updated["external_agent_review_agents"] = "catalyst_news; macro_regime; structure_builder; skeptic"
    updated["agent_support_count"] = 4
    context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=snapshot_dir,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=4,
        external_review_agent_count=4,
        agent_reviews_json=tmp_path / "agentic_reviews.json",
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(updated, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    live_quality = core.build_live_spread_quality_audit(updated)

    assert live["live_validation_status"].tolist() == ["PASS"]
    assert updated["recommendation_status"].tolist() == [RecommendationStatus.AVOID.value]
    assert "live_quote_width_pct_above_40pct" in updated["quality_gate_reason"].iloc[0]
    assert live_quality["live_market_quality_status"].tolist() == ["BLOCK"]
    assert live_quality["actionability_impact"].tolist() == ["blocked_not_target_candidate"]
    assert live_quality["live_leg_min_liquidity"].tolist() == [4900.0]
    assert "setup quality gate reject" in updated["status_reason"].iloc[0]
    assert decision["execution_status"].tolist() == ["blocked"]
    assert tickets.empty


def test_live_spread_quality_audit_keeps_no_realistic_spread_rows_visible() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "NFLX",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "live_validation_status": "no_realistic_spread",
                "trade_plan": "SELL 1 NFLX 2026-06-18 92 Call / BUY 1 NFLX 2026-06-18 93 Call @ 0.28 CREDIT",
                "entry_limit": 0.28,
                "quality_gate_reason": "",
            }
        ]
    )

    live_quality = core.build_live_spread_quality_audit(final)

    assert live_quality["ticker"].tolist() == ["NFLX"]
    assert live_quality["live_validation_status"].tolist() == ["NO_REALISTIC_SPREAD"]
    assert live_quality["live_market_quality_status"].tolist() == ["BLOCK"]
    assert live_quality["live_leg_liquidity_status"].tolist() == ["MISSING"]


def test_live_spread_quality_audit_defers_market_closed_recheck_rows() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "live_validation_status": "MARKET_CLOSED_RECHECK",
                "trade_plan": "SELL 1 AAPL 2026-06-18 200 Put / BUY 1 AAPL 2026-06-18 195 Put @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "quality_gate_reason": "",
            }
        ]
    )

    live_quality = core.build_live_spread_quality_audit(final)

    assert live_quality["ticker"].tolist() == ["AAPL"]
    assert live_quality["live_market_quality_status"].tolist() == ["DEFERRED_QUOTE_REFRESH"]
    assert live_quality["actionability_impact"].tolist() == ["target_order_price_validation"]
    assert core.summarize_live_spread_quality(live_quality)["status"] == "pass"


def test_live_spread_quality_proof_blocks_bad_markets_that_stay_actionable() -> None:
    live_quality = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "live_market_quality_status": "BLOCK",
                "actionability_impact": "eligible_for_yellow_or_green_surface",
                "live_quote_width_pct": 0.62,
                "live_leg_min_liquidity": 25,
                "live_leg_liquidity_status": "BLOCK",
                "quality_gate_reason": "live_quote_width_pct_above_40pct; live_leg_liquidity_below_100",
            }
        ]
    )

    packet = audit.build_live_spread_quality_proof_packet(live_quality)

    assert packet["status"].tolist() == ["FAIL_BLOCKED_LIVE_MARKETS_STILL_ACTIONABLE"]
    assert packet["blocked_still_actionable_rows"].tolist() == [1]
    assert packet["target_candidate_block_rows"].tolist() == [1]
    assert packet["quote_width_block_rows"].tolist() == [1]
    assert packet["liquidity_block_rows"].tolist() == [1]


def test_live_spread_quality_proof_allows_blocked_audit_visible_rows_without_nan_examples() -> None:
    live_quality = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "live_market_quality_status": "BLOCK",
                "actionability_impact": "visible_for_review",
                "live_quote_width_pct": float("nan"),
                "live_leg_min_liquidity": float("nan"),
                "live_leg_liquidity_status": "MISSING",
                "quality_gate_reason": float("nan"),
            }
        ]
    )

    packet = audit.build_live_spread_quality_proof_packet(live_quality)

    assert packet["status"].tolist() == ["PASS_LIVE_SPREAD_QUALITY_GATED"]
    assert packet["blocked_still_actionable_rows"].tolist() == [0]
    assert packet["target_candidate_block_rows"].tolist() == [0]
    assert "nan" not in packet["blocked_examples"].iloc[0].lower()
    assert "AAPL" in packet["blocked_examples"].iloc[0]


def test_snapshot_only_validation_does_not_fall_back_to_live_for_missing_chain(tmp_path: Path, monkeypatch) -> None:
    from codexuw.schwab_live import SchwabChainValidator

    def fail_if_live_service_is_requested(self):
        raise AssertionError("snapshot-only validation attempted live Schwab fallback")

    monkeypatch.setattr(SchwabChainValidator, "_service", fail_if_live_service_is_requested)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "bull put spread",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
            }
        ]
    )

    updated, live, notes = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )

    assert notes == []
    assert updated["live_validation_status"].tolist() == ["CHAIN_UNAVAILABLE"]
    assert "snapshot missing for WMT" in updated["live_validation_note"].iloc[0]
    assert live["live_validation_status"].tolist() == ["CHAIN_UNAVAILABLE"]


def test_live_expiry_selection_stays_inside_daily_trade_window() -> None:
    asof = dt.date(2026, 5, 22)
    contracts = pd.DataFrame(
        [
            {"right": "P", "expiry": dt.date(2026, 6, 19)},
            {"right": "P", "expiry": dt.date(2028, 1, 21)},
        ]
    )

    selected = core._select_live_expiry(
        contracts,
        asof,
        preferred_expiry=dt.date(2028, 1, 21),
        direction="Bull Put",
    )

    assert selected == dt.date(2026, 6, 19)
    assert (
        core._select_live_expiry(
            pd.DataFrame([{"right": "P", "expiry": dt.date(2028, 1, 21)}]),
            asof,
            preferred_expiry=dt.date(2028, 1, 21),
            direction="Bull Put",
        )
        is None
    )


def test_live_direction_helpers_respect_explicit_bias_before_structure_text() -> None:
    bearish_put_debit = {"bias": "bearish", "structure": "bear put debit spread"}
    bullish_call_debit = {"bias": "bullish", "structure": "bull call debit spread"}

    assert core._credit_direction(bearish_put_debit) == "Bear Call"
    assert core._debit_direction(bearish_put_debit) == "Bear Put"
    assert core._credit_direction(bullish_call_debit) == "Bull Put"
    assert core._debit_direction(bullish_call_debit) == "Bull Call"


def test_live_debit_replacement_relabels_stale_credit_route() -> None:
    row = {
        "ticker": "WMT",
        "bias": "bullish",
        "strategy": "bull_put_credit",
        "strategy_family": "vertical_spread",
        "strategy_route": "bull_put_credit",
        "entry_type": "CREDIT",
        "signal_premium": 2_000_000,
        "combined_flow_bias": 0.30,
    }
    live = {
        "debit": 1.20,
        "spread_width": 5.0,
        "short_strike": 105.0,
        "long_strike": 100.0,
        "short_leg": "WMT260717C00105000",
        "long_leg": "WMT260717C00100000",
        "target_entry": 2.25,
        "live_status": "PASS",
    }

    out = core._apply_live_debit_spread(
        row,
        live,
        direction="Bull Call",
        expiry=dt.date(2026, 7, 17),
        spot=102.0,
        asof_date=dt.date(2026, 6, 9),
    )

    assert out["strategy"] == "bull_call_debit"
    assert out["strategy_route"] == "bull_call_debit"
    assert out["strategy_family"] == "vertical_spread"
    assert out["entry_type"] == "DEBIT"
    assert out["direction"] == "Bull Call"
    assert out["structure"] == "bull call debit spread"
    assert "DEBIT" in out["trade_plan"]


def test_dated_hot_chain_construction_rejects_far_dated_only_expiry() -> None:
    candidate = {
        "ticker": "MESO",
        "bias": "bearish",
        "close": 10.0,
        "score": 61.0,
        "signal_premium": 2_000_000,
        "combined_flow_bias": -0.5,
        "quality_status": "qualified",
    }
    hot = pd.DataFrame(
        [
            {
                "ticker": "MESO",
                "right": "C",
                "expiry_dt": dt.date(2028, 1, 21),
                "dte": 609,
                "strike": 20.0,
                "bid": 0.4,
                "ask": 0.5,
                "premium": 1_000_000,
                "volume": 1_000,
                "option_symbol": "MESO280121C00020000",
            },
            {
                "ticker": "MESO",
                "right": "C",
                "expiry_dt": dt.date(2028, 1, 21),
                "dte": 609,
                "strike": 22.0,
                "bid": 0.1,
                "ask": 0.1,
                "premium": 500_000,
                "volume": 800,
                "option_symbol": "MESO280121C00022000",
            },
        ]
    )

    row = core.construct_credit_spread(candidate, hot)

    assert row["trade_plan"] == ""
    assert row["expiry"] == ""
    assert row["recommendation_status"] == RecommendationStatus.REVIEW.value
    assert "no dated UW hot-chain expiry in 7-60 DTE window" in row["status_reason"]


def test_live_snapshot_validation_promotes_visible_trade_and_then_portfolio_annotates(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        portfolio_context={
            "status": "ok",
            "total_value": 100_000,
            "option_underlyings": ["WMT"],
            "large_equity_exposure": {"WMT": 7_500},
        },
    )
    manifest = json.loads(paths["manifest"].read_text())
    live = pd.read_csv(paths["live_chain_validation"])
    structure_attempts = pd.read_csv(paths["structure_attempts"])
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    risk = pd.read_csv(paths["risk_audit"])
    sizing = pd.read_csv(paths["sizing_audit"])
    management = pd.read_csv(paths["management_plan"])

    assert manifest["row_counts"]["live_chain_validation"] == 1
    assert manifest["row_counts"]["structure_attempts"] == 2
    assert live["live_validation_status"].tolist() == ["PASS"]
    assert set(structure_attempts["attempt_stage"]) == {"dated_hot_chain", "live_schwab_chain"}
    assert structure_attempts.loc[structure_attempts["attempt_stage"].eq("live_schwab_chain"), "attempt_status"].tolist() == [
        "PASS"
    ]
    assert final["ticker"].tolist() == ["WMT"]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert final["entry_limit"].tolist() == [1.0]
    assert final["max_profit"].tolist() == [100.0]
    assert final["max_loss"].tolist() == [400.0]
    assert final["portfolio_risk_flag"].tolist() == [True]
    assert "existing option exposure in WMT" in final["portfolio_risk_note"].iloc[0]
    assert risk["visibility_action"].tolist() == ["annotated_not_hidden"]
    assert decision["final_action"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert decision["setup_quality_status"].tolist() == ["qualified"]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["execution_gate_status"].tolist() == ["blocked"]
    assert decision["portfolio_fit_status"].tolist() == ["risk_flagged"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["suggested_contracts"].tolist() == [1]
    assert "fresh_live_schwab_required" in decision["execution_blockers"].iloc[0]
    assert "agentic_reviews_required" in decision["execution_blockers"].iloc[0]
    assert "sizing uses the explicit risk budget" in decision["sizing_note"].iloc[0]
    assert "portfolio annotation only" not in decision["sizing_note"].iloc[0]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert sizing["visibility_action"].tolist() == ["annotated_not_hidden"]
    assert management["management_action"].tolist() == ["REPRICE"]


def test_external_agent_caution_keeps_target_ticket_visible(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "skeptic",
                        "verdict": "caution",
                        "confidence": "high",
                        "note": "news check requires human confirmation",
                        "objective_blocker": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
        portfolio_context={
            "status": "ok",
            "total_value": 100_000,
            "option_underlyings": ["WMT"],
        },
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    reviews = pd.read_csv(paths["external_agent_reviews"])
    research_tasks = json.loads(paths["research_tasks"].read_text())
    review_board = pd.read_csv(paths["agent_review_board"])
    manifest = json.loads(paths["manifest"].read_text())

    assert research_tasks["tasks"][0]["ticker"] == "WMT"
    assert reviews["verdict"].tolist() == ["caution"]
    assert manifest["agent_review_summary"]["external_reviews_present"] is True
    external_rows = review_board[review_board["agent_type"].eq("external")]
    assert external_rows["note"].tolist() == ["news check requires human confirmation"]
    assert final["ticker"].tolist() == ["WMT"]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert final["portfolio_risk_flag"].tolist() == [True]
    assert "external agent caution: news check requires human confirmation" in final["status_reason"].iloc[0]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["portfolio_fit_status"].tolist() == ["risk_flagged"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_built_in_caution_annotates_without_downgrading_entry_status() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "status_reason": "Schwab chain validated",
            }
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": "market_regime",
                "agent_type": "built_in",
                "verdict": "caution",
                "confidence": "medium",
                "objective_blocker": False,
                "portfolio_risk_only": False,
                "note": "risk_off tape; use smaller size",
            }
        ]
    )

    final = core.apply_agent_reviews(priced, reviews)

    assert final["recommendation_status"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert "built-in agent caution: risk_off tape; use smaller size" in final["status_reason"].iloc[0]


def test_strategy_supported_debit_spread_stays_actionable_despite_cautions() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 AAPL 2026-05-29 315 Call / SELL 1 AAPL 2026-05-29 317.5 Call @ 0.62 DEBIT",
                "trade_plan": "BUY 1 AAPL 2026-05-29 315 Call / SELL 1 AAPL 2026-05-29 317.5 Call @ 0.62 DEBIT",
                "entry_limit": 0.62,
                "suggested_contracts": 5,
                "max_profit": 188.0,
                "max_loss": 62.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 4,
                "agent_caution_count": 4,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 10,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 6,
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "risk_off"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["trade_quality_confidence_rating"].tolist() == ["HIGH"]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert decision["ready_to_enter"].tolist() == [True]
    assert tickets["ticker"].tolist() == ["AAPL"]


def test_short_dated_far_otm_debit_spread_is_not_send_now() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AMD",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 AMD 2026-06-05 462.5 Put / SELL 1 AMD 2026-06-05 460 Put @ 0.69 DEBIT",
                "trade_plan": "BUY 1 AMD 2026-06-05 462.5 Put / SELL 1 AMD 2026-06-05 460 Put @ 0.69 DEBIT",
                "entry_limit": 0.69,
                "suggested_contracts": 5,
                "max_profit": 181.0,
                "max_loss": 69.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "agent_caution_count": 0,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "spot_live": 499.0,
                "breakeven": 461.81,
                "dte": 8,
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "risk_off"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["waiting_for_price"]
    assert "send_now_debit_breakeven_move_above_4pct" in decision["execution_blockers"].iloc[0]
    assert tickets["status_label"].tolist() == ["YELLOW target"]
    assert "breakeven move too large for send-now" in tickets.apply(core._ticket_recheck_summary, axis=1).iloc[0]


def test_weak_flow_debit_spread_without_outcome_support_is_not_send_now() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "IWM",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 IWM 2026-06-30 277 Put / SELL 1 IWM 2026-06-30 257 Put @ 2.40 DEBIT",
                "trade_plan": "BUY 1 IWM 2026-06-30 277 Put / SELL 1 IWM 2026-06-30 257 Put @ 2.40 DEBIT",
                "entry_limit": 2.40,
                "suggested_contracts": 3,
                "max_profit": 1760.0,
                "max_loss": 240.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 7,
                "agent_caution_count": 1,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "spot_live": 288.55,
                "breakeven": 274.60,
                "dte": 28,
                "combined_flow_bias": -0.104,
                "actual_forward_expectancy_status": "BLOCK",
                "actual_forward_expectancy_sample_size": 0,
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "risk_off"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["waiting_for_price"]
    assert decision["status_label"].tolist() == ["YELLOW review"]
    assert "send_now_debit_directional_edge_below_threshold" in decision["execution_blockers"].iloc[0]
    assert tickets.empty
    assert "directional edge too weak for send-now" in core._ticket_recheck_summary(decision.iloc[0])


def test_portfolio_management_process_note_does_not_count_as_quality_caution() -> None:
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": "portfolio_management",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": False,
                "note": "Portfolio risk annotation: no sized spread, target exit, or invalidation is present; fresh portfolio context is required.",
            },
            {
                "ticker": "AAPL",
                "agent": "skeptic",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": False,
                "note": "flow is modest",
            },
        ]
    )

    summary = core._review_summary_by_ticker(reviews)

    assert summary["AAPL"]["caution"] == 1


def test_subagent_review_metadata_is_preserved_into_agent_review_board(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "agentic_reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "candidate_id": "WMT:bullish:88",
                        "ticker": "WMT",
                        "agent": "portfolio_management",
                        "agent_type": "subagent",
                        "review_stage": "portfolio_management",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "portfolio concentration only; setup quality remains valid",
                        "objective_blocker": False,
                        "portfolio_risk_only": True,
                        "blocker_type": "portfolio",
                        "evidence": "existing WMT exposure noted by subagent",
                        "source_artifact": "agentic_reviews.json",
                        "as_of": "2026-05-22",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
    )
    manifest = json.loads(paths["manifest"].read_text())
    agentic_reviews = json.loads(paths["agentic_reviews"].read_text())
    review_board = pd.read_csv(paths["agent_review_board"])
    subagent_rows = review_board[review_board["agent_type"].eq("subagent")]

    assert manifest["agentic_orchestration"]["status"] == "reviews_ingested"
    assert agentic_reviews["reviews"][0]["candidate_id"] == "WMT:bullish:88"
    assert subagent_rows["candidate_id"].tolist() == ["WMT:bullish:88"]
    assert subagent_rows["review_stage"].tolist() == ["portfolio_management"]
    assert subagent_rows["portfolio_risk_only"].astype(bool).tolist() == [True]
    assert subagent_rows["blocker_type"].tolist() == ["portfolio"]
    assert subagent_rows["evidence"].tolist() == ["existing WMT exposure noted by subagent"]
    assert subagent_rows["source_artifact"].tolist() == ["agentic_reviews.json"]


def test_portfolio_caution_review_does_not_stamp_every_ticket_as_portfolio_risk(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "agentic_reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "portfolio_management",
                        "agent_type": "subagent",
                        "verdict": "caution",
                        "confidence": "high",
                        "note": "correlated watch only; do not stamp as actual portfolio risk",
                        "objective_blocker": False,
                        "portfolio_risk_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
        portfolio_context={"status": "ok", "total_value": 100_000},
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    report = paths["report"].read_text(encoding="utf-8")

    assert final["portfolio_risk_flag"].tolist() == [False]
    assert str(final["portfolio_risk_note"].iloc[0]) in {"", "nan"}
    assert "Portfolio risk annotation" not in str(final["status_reason"].iloc[0])
    assert "Portfolio risk annotation" not in str(final["external_agent_review_note"].iloc[0])
    assert decision["portfolio_fit_status"].tolist() == ["clear"]
    assert "portfolio risk noted" not in report
    assert "portfolio risk noted" not in core._ticket_recheck_summary(tickets.iloc[0])
    assert "portfolio annotation only" not in report
    assert "portfolio annotation only" not in core._ticket_recheck_summary(tickets.iloc[0])


def test_portfolio_management_avoid_without_account_exposure_is_not_portfolio_risk(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "agentic_reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "portfolio_management",
                        "agent_type": "subagent",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": (
                            "Portfolio risk annotation: no sized spread, target exit, or invalidation "
                            "is present; reduced risk-off sizing and fresh portfolio context are required."
                        ),
                        "objective_blocker": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
        portfolio_context={"status": "ok", "total_value": 100_000},
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    report = paths["report"].read_text(encoding="utf-8")

    assert final["portfolio_risk_flag"].tolist() == [False]
    assert str(final["portfolio_risk_note"].iloc[0]) in {"", "nan"}
    assert decision["portfolio_fit_status"].tolist() == ["clear"]
    assert "portfolio risk noted" not in report
    assert "portfolio risk noted" not in core._ticket_recheck_summary(tickets.iloc[0])


def test_external_portfolio_avoid_annotates_without_blocking_ready_trade(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "portfolio_risk",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "portfolio crowding only; setup quality remains good",
                        "objective_blocker": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    review_board = pd.read_csv(paths["agent_review_board"])
    report = paths["report"].read_text(encoding="utf-8")
    external_rows = review_board[review_board["agent_type"].eq("external")]

    assert final["recommendation_status"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert final["portfolio_risk_flag"].tolist() == [True]
    assert "external_agent_objective_blocker" not in str(final["hard_rejects"].iloc[0])
    assert "external portfolio risk review" in final["portfolio_risk_note"].iloc[0]
    assert external_rows["portfolio_risk_only"].astype(bool).tolist() == [True]
    assert external_rows["blocker_type"].tolist() == ["portfolio"]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["requires_portfolio_ack"].tolist() == [False]
    assert "fresh_live_schwab_required" in decision["execution_blockers"].iloc[0]
    assert "portfolio_context_required" in decision["execution_blockers"].iloc[0]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert "portfolio annotation only" not in report
    assert "portfolio note is annotation only" not in report
    assert "portfolio annotation only" not in core._ticket_recheck_summary(tickets.iloc[0])


def test_external_agent_objective_blocker_blocks_without_hiding_row(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "skeptic",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "objective thesis break",
                        "objective_blocker": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])

    assert final["recommendation_status"].tolist() == [RecommendationStatus.AVOID.value]
    assert final["visible_in_final_board"].tolist() == [True]
    assert "external_agent_objective_blocker" in final["hard_rejects"].iloc[0]
    assert decision["setup_quality_status"].tolist() == ["blocked"]
    assert decision["execution_status"].tolist() == ["blocked"]


def test_confidence_audit_blocks_goal_when_current_strategy_cohort_is_negative_and_no_green_orders() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "WARN",
                "sample_size": 15,
                "win_rate": 0.60,
                "avg_pnl": 85.53,
                "total_pnl": 1283.0,
                "profit_factor": 1.448,
                "matched_current_tickers": "MSFT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "WARN",
                "sample_size": 4,
                "win_rate": 0.75,
                "avg_pnl": 86.85,
                "total_pnl": 347.4,
                "profit_factor": 4.229,
                "matched_current_tickers": "MSFT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "BLOCK",
                "sample_size": 76,
                "win_rate": 0.3947,
                "avg_pnl": -28.42,
                "total_pnl": -2160.0,
                "profit_factor": 0.732,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 95,
                "matched_current_tickers": "MSFT",
                "matched_current_count": 1,
                "note": "Actual closed-trade evidence is not positive enough.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 95, "status": "BLOCK", "note": "not proven"},
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
    )
    summary = core.summarize_confidence_audit(audit)
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert profitability["rating"] == 3.0
    assert profitability["status"] == "BLOCK"
    assert "current_strategy_cohort_negative" in profitability["blockers"]
    assert order_entry["rating"] == 0.0
    assert "no_green_ready_orders" in order_entry["blockers"]
    assert summary["status"] == "block"
    assert summary["profitability_confidence_rating"] == 3.0
    assert summary["order_entry_confidence_rating"] == 0.0


def test_confidence_audit_passes_only_with_positive_expectancy_and_green_order_entry_proof() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 5,
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 34,
                "win_rate": 0.6471,
                "avg_pnl": 92.09,
                "total_pnl": 3131.0,
                "profit_factor": 1.823,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 5,
                "win_rate": 0.60,
                "avg_pnl": 75.0,
                "total_pnl": 375.0,
                "profit_factor": 1.4,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 109,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 109, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "action_surface": "green_send_now",
                "fill_quality_status": "PASS",
                "trade_plan": "",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    summary = core.summarize_confidence_audit(audit)

    assert audit.loc[audit["metric"].eq("profitability_confidence_rating"), "status"].tolist() == ["PASS"]
    assert audit.loc[audit["metric"].eq("order_entry_confidence_rating"), "status"].tolist() == ["PASS"]
    assert audit.loc[audit["metric"].eq("goal_confidence_gate"), "status"].tolist() == ["PASS"]
    assert summary["profitability_confidence_rating"] >= 7.0
    assert summary["order_entry_confidence_rating"] >= 7.0


def test_confidence_audit_caps_order_entry_when_green_fill_quality_fails() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 40,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            }
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "action_surface": "green_send_now",
                "fill_quality_status": "BLOCK",
                "trade_plan": "SELL 1 WMT 2026-06-19 95 Put @ 1.25 CREDIT",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert order_entry["rating"] == 6.0
    assert order_entry["status"] == "BLOCK"
    assert "green_execution_fill_quality_not_all_pass" in order_entry["blockers"]


def test_confidence_audit_caps_profitability_when_strategy_cohort_is_weak_not_losing() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 5,
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "BLOCK",
                "sample_size": 110,
                "win_rate": 0.4727,
                "avg_pnl": 8.83,
                "total_pnl": 971.0,
                "profit_factor": 1.082,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 185,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 185, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "status": "PASS",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route",
                "replay_bucket_status": "PASS",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    summary = core.summarize_confidence_audit(audit)

    assert profitability["status"] == "BLOCK"
    assert profitability["rating"] == 6.0
    assert "current_strategy_cohort_weak_under_threshold" in profitability["blockers"]
    assert "current_strategy_cohort_negative" not in profitability["blockers"]
    assert summary["profitability_confidence_rating"] < 7.0
    assert "current_strategy_cohort_weak_under_threshold" in summary["blockers"]


def test_confidence_audit_blocks_goal_when_green_row_lacks_profitability_calibration() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 5,
                "profitability_calibration_status": "BLOCK",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 34,
                "win_rate": 0.6471,
                "avg_pnl": 92.09,
                "total_pnl": 3131.0,
                "profit_factor": 1.823,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 5,
                "win_rate": 0.60,
                "avg_pnl": 75.0,
                "total_pnl": 375.0,
                "profit_factor": 1.4,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 109,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 109, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "bull_call_debit",
                "status": "BLOCK",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
    )
    summary = core.summarize_confidence_audit(audit)

    assert audit.loc[audit["metric"].eq("goal_confidence_gate"), "status"].tolist() == ["BLOCK"]
    assert "profitability_calibration_not_proven" in summary["blockers"]
    assert "green_profitability_calibration_not_all_pass" in summary["blockers"]
    assert summary["profitability_confidence_rating"] < 7.0
    assert summary["order_entry_confidence_rating"] < 7.0


def test_calibrated_order_entry_blocker_summary_names_remaining_blockers() -> None:
    decision = pd.DataFrame(
        [
            {
                "ticker": "BX",
                "strategy_route": "short_put",
                "trade_plan": "SELL 1 BX 2026-07-17 115 Put @ 3.60 CREDIT",
                "ready_to_enter": False,
                "profitability_calibration_status": "PASS",
                "execution_status": "waiting_for_price",
                "target_order_status": "target_order_candidate",
                "execution_blockers": "position_profit_below_materiality_floor",
                "entry_limit": 3.60,
                "suggested_contracts": 1,
            },
            {
                "ticker": "VRT",
                "strategy_route": "short_put",
                "trade_plan": "SELL 1 VRT 2026-07-17 280 Put @ 15.15 CREDIT",
                "ready_to_enter": False,
                "profitability_calibration_status": "PASS",
                "execution_status": "needs_confidence",
                "target_order_status": "not_actionable_cash_secured_risk",
                "execution_blockers": "short_put_account_risk_above_2.00%; short_put_cash_required_above_75pct_cash",
                "entry_limit": 15.15,
                "suggested_contracts": 1,
            },
            {
                "ticker": "OK",
                "strategy_route": "short_put",
                "ready_to_enter": True,
                "profitability_calibration_status": "PASS",
                "execution_blockers": "",
            },
            {
                "ticker": "WARN",
                "ready_to_enter": False,
                "profitability_calibration_status": "WARN",
                "execution_blockers": "profitability_calibration_required_for_green",
            },
        ]
    )

    summary = core.summarize_calibrated_order_entry_blockers(decision)

    assert summary["calibrated_rows"] == 3
    assert summary["ready_rows"] == 1
    assert summary["blocked_rows"] == 2
    assert summary["blocker_counts"] == {
        "position_profit_below_materiality_floor": 1,
        "short_put_account_risk_above_2.00%": 1,
        "short_put_cash_required_above_75pct_cash": 1,
    }
    assert summary["examples"][0]["ticker"] == "BX"
    assert "position_profit_below_materiality_floor" in core._calibrated_order_entry_blocker_detail(summary)
