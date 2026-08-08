from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.catalysts import load_catalyst_context
from codexuw.engine import (
    build_entry_watchlist,
    apply_high_conviction_decision_marks,
    generate_candidates,
    is_etf_row,
    select_final_trades,
    select_ticker_pool,
    _write_execute_outcome_ledger,
    _write_recommendation_outcome_ledger,
)
from codexuw.portfolio import summarize_positions
from codexuw.provenance import build_input_provenance, file_fingerprint
from codexuw.qa import audit_run
from codexuw.schwab_live import SchwabChainValidator


def test_summarize_positions_blocks_existing_option_underlyings() -> None:
    payload = {
        "balances": {"total_value": 100_000, "cash": 10_000},
        "positions": [
            {"symbol": "MSFT", "asset_type": "EQUITY", "qty": 100, "market_value": 6_000},
            {"symbol": "GOOG  260515P00375000", "asset_type": "OPTION", "underlying": "GOOG", "short_qty": 1, "market_value": -200},
        ],
    }
    summary = summarize_positions(payload)
    assert summary["option_underlyings"] == ["GOOG"]
    assert summary["short_option_underlyings"] == ["GOOG"]
    assert summary["large_equity_exposure"] == {"MSFT": 6000.0}
    assert summary["equity_shares"] == {"MSFT": 100.0}
    assert any(row["action"] == "SELL COVERED INCOME" for row in summary["portfolio_income_actions"])
    assert any(row["action"] in {"HOLD", "ROLL", "TAKE PROFIT"} for row in summary["risk_actions"])


def test_trading_sleeve_only_protects_core_holdings_from_covered_calls() -> None:
    payload = {
        "balances": {"total_value": 100_000, "cash": 20_000},
        "positions": [
            {"symbol": "MSFT", "asset_type": "EQUITY", "qty": 100, "market_value": 42_000},
        ],
    }

    summary = summarize_positions(payload, portfolio_income_mode="trading-sleeve-only")

    assert not any(row["action"] == "SELL COVERED INCOME" for row in summary["portfolio_income_actions"])
    assert any(row["action"] == "INCOME SLEEVE ONLY" for row in summary["portfolio_income_actions"])
    assert "core equity holdings are protected" in summary["portfolio_income_actions"][0]["reason"]


def test_trading_sleeve_only_allows_explicit_income_lot_ticker() -> None:
    payload = {
        "balances": {"total_value": 100_000, "cash": 20_000},
        "positions": [
            {"symbol": "MSFT", "asset_type": "EQUITY", "qty": 100, "market_value": 42_000},
        ],
    }

    summary = summarize_positions(
        payload,
        portfolio_income_mode="trading-sleeve-only",
        covered_income_allowed_tickers=["MSFT"],
    )

    assert any(row["action"] == "SELL COVERED INCOME" for row in summary["portfolio_income_actions"])


def test_load_catalyst_context_reads_browser_text(tmp_path) -> None:
    browser_dir = tmp_path / "browser_text"
    browser_dir.mkdir()
    (browser_dir / "browser-text-capture-news-msft.txt").write_text(
        "MSFT beats earnings expectations and shows strong cloud growth.",
        encoding="utf-8",
    )
    df = load_catalyst_context(tmp_path, ["MSFT", "SPY"])
    msft = df[df["ticker"].eq("MSFT")].iloc[0]
    spy = df[df["ticker"].eq("SPY")].iloc[0]
    assert msft["catalyst_status"] == "supportive"
    assert spy["catalyst_status"] == "unknown"


def test_structured_catalyst_date_overrides_word_count_noise(tmp_path) -> None:
    browser_dir = tmp_path / "browser_text"
    browser_dir.mkdir()
    (browser_dir / "browser-text-capture-news-nvda.txt").write_text(
        "NVDA risk note: NVIDIA financial results conference call is on 2026-05-20 after market close.",
        encoding="utf-8",
    )

    df = load_catalyst_context(tmp_path, ["NVDA"], asof=dt.date(2026, 5, 6))
    nvda = df.iloc[0]

    assert nvda["catalyst_status"] == "mixed"
    assert str(nvda["catalyst_earnings_date"]) == "2026-05-20"
    assert nvda["catalyst_earnings_days"] == 14.0


def test_browser_earnings_date_must_be_on_ticker_line(tmp_path) -> None:
    browser_dir = tmp_path / "browser_text"
    browser_dir.mkdir()
    (browser_dir / "browser-text-capture-news-mixed.txt").write_text(
        "AAA is being watched for flow quality.\n"
        "BBB financial results conference call is on 2026-05-10 after close.\n",
        encoding="utf-8",
    )

    df = load_catalyst_context(tmp_path, ["AAA"], asof=dt.date(2026, 5, 6))
    aaa = df.iloc[0]

    assert aaa["catalyst_status"] == "mixed"
    assert pd.isna(aaa["catalyst_earnings_date"])
    assert pd.isna(aaa["catalyst_earnings_days"])


def test_ticker_scoped_browser_file_can_supply_company_name_event_line(tmp_path) -> None:
    browser_dir = tmp_path / "browser_text"
    browser_dir.mkdir()
    (browser_dir / "browser-text-capture-news-NVDA-LIVE.txt").write_text(
        "NVIDIA announced that its financial results conference call is on 2026-05-20 after close.\n",
        encoding="utf-8",
    )

    df = load_catalyst_context(tmp_path, ["NVDA"], asof=dt.date(2026, 5, 6))
    nvda = df.iloc[0]

    assert nvda["catalyst_status"] == "mixed"
    assert str(nvda["catalyst_earnings_date"]) == "2026-05-20"
    assert nvda["catalyst_earnings_days"] == 14.0


def test_ticker_scoped_browser_file_extracts_monthly_sales_event(tmp_path) -> None:
    browser_dir = tmp_path / "browser_text"
    browser_dir.mkdir()
    (browser_dir / "browser-text-capture-news-TSM-LIVE.txt").write_text(
        "TSMC Financial Calendar: 2026-05-08 13:30 Asia/Taipei - TSMC Monthly Sales - April 2026.\n",
        encoding="utf-8",
    )

    df = load_catalyst_context(tmp_path, ["TSM"], asof=dt.date(2026, 5, 6))
    tsm = df.iloc[0]

    assert tsm["catalyst_status"] == "caution"
    assert str(tsm["catalyst_earnings_date"]) == "2026-05-08"
    assert tsm["catalyst_earnings_days"] == 2.0


def test_trade_description_date_does_not_become_catalyst_date(tmp_path) -> None:
    browser_dir = tmp_path / "browser_text"
    browser_dir.mkdir()
    (browser_dir / "browser-text-capture-news-TSLA-LIVE.txt").write_text(
        "Tesla investor relations shows Q1 2026 earnings were released on 2026-04-22.\n"
        "The TSLA 2026-06-18 bull call debit spread candidate is not blocked by a near-term earnings event.\n",
        encoding="utf-8",
    )

    df = load_catalyst_context(tmp_path, ["TSLA"], asof=dt.date(2026, 5, 7))
    tsla = df.iloc[0]

    assert tsla["catalyst_status"] == "mixed"
    assert str(tsla["catalyst_earnings_date"]) == "2026-04-22"
    assert tsla["catalyst_earnings_days"] == -15.0


def test_input_provenance_records_export_hashes(tmp_path) -> None:
    base = tmp_path / "2026-05-06"
    base.mkdir()
    path = base / "stock-screener-2026-05-06.csv"
    path.write_text("ticker,close\nNVDA,210\n", encoding="utf-8")

    provenance = build_input_provenance(base)

    stock = provenance["exports"]["stock_screener"]
    assert stock["path"] == str(path)
    assert len(stock["sha256"]) == 64
    assert stock["size_bytes"] > 0


def test_file_fingerprint_uses_cache_when_signature_matches(tmp_path) -> None:
    path = tmp_path / "large-ish.csv"
    path.write_text("ticker,close\nNVDA,210\n", encoding="utf-8")

    first = file_fingerprint(path)
    second = file_fingerprint(path)

    assert first["hash_cache"] == "miss"
    assert second["hash_cache"] == "hit"
    assert first["sha256"] == second["sha256"]


def test_schwab_validator_can_replay_saved_snapshot_without_service(tmp_path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "NVDA.json").write_text('{"symbol":"NVDA","underlyingPrice":210.0}', encoding="utf-8")

    validator = SchwabChainValidator(tmp_path / "out", snapshot_dir=snapshot_dir)
    chain = validator.get_chain("NVDA", dt.date(2026, 5, 6), dt.date(2026, 5, 15))

    assert chain["underlyingPrice"] == 210.0
    assert "snapshot:" in validator.sources["NVDA"]


def test_execute_outcome_ledger_records_open_trade(tmp_path) -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "strategy": "Bull Call Debit Spread",
                "direction": "Bull Call",
                "expiry": "2026-05-15",
                "sell_leg": "NVDA260515C00215000",
                "buy_leg": "NVDA260515C00210000",
                "entry_action": "BUY TO OPEN debit spread",
                "entry_limit_debit": 1.82,
                "contracts": 1,
                "max_profit": 318.0,
                "max_loss": 182.0,
                "breakeven": 211.82,
                "score": 7.22,
                "confidence": "High",
                "trade_tier": "Execute Tactical",
                "edge_verdict": "positive",
            }
        ]
    )

    path = _write_execute_outcome_ledger(tmp_path / "codexuw_daily_test", dt.date(2026, 5, 6), final)
    ledger = pd.read_csv(path)

    assert ledger["outcome_status"].iloc[0] == "OPEN_REVIEW_REQUIRED"
    assert ledger["entry_price"].iloc[0] == 1.82
    assert ledger["setup_family"].iloc[0] == "debit spreads"
    assert ledger["recommended_limit"].iloc[0] == 1.82
    assert (tmp_path / "codexuw_execute_outcome_ledger.csv").exists()


def test_recommendation_outcome_ledger_records_conditional_watch(tmp_path) -> None:
    watch = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "expiry": "2026-05-15",
                "sell_leg": "NVDA260515P00200000",
                "buy_leg": "NVDA260515P00195000",
                "required_entry": 1.25,
                "mid_credit": 1.10,
                "natural_credit": 0.95,
            }
        ]
    )

    path = _write_recommendation_outcome_ledger(tmp_path / "codexuw_daily_test", dt.date(2026, 5, 6), pd.DataFrame(), watch)
    ledger = pd.read_csv(path)

    assert ledger["lane"].iloc[0] == "Enter Only At Price"
    assert ledger["outcome_status"].iloc[0] == "CONDITIONAL_NOT_FILLED"
    assert ledger["recommended_limit"].iloc[0] == 1.25


def test_qa_audit_catches_final_hard_block_token(tmp_path) -> None:
    run = tmp_path / "codexuw_daily_test"
    run.mkdir()
    asof = "2026-05-06"
    manifest = {
        "execute_rows": 1,
        "watch_rows": 0,
        "research_rows": 0,
        "avoid_rows": 0,
        "run_provenance": {
            "input_files": {
                "exports": {
                    "stock_screener": {"sha256": "a" * 64},
                    "hot_chains": {"sha256": "b" * 64},
                }
            },
            "schwab_snapshot": {"status": "ok"},
        },
    }
    (run / f"codexuw_manifest_{asof}.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "strategy": "Bull Call Debit Spread",
                "hard_rejects": "",
                "penalties": "news_catalyst_caution",
                "confirmations_failed": "",
                "trade_status_reason": "bad",
                "risk_notes": "",
                "max_loss": 100,
            }
        ]
    ).to_csv(run / f"codexuw_final_trades_{asof}.csv", index=False)
    pd.DataFrame().to_csv(run / f"codexuw_watch_trades_{asof}.csv", index=False)
    pd.DataFrame().to_csv(run / f"codexuw_research_candidates_{asof}.csv", index=False)
    pd.DataFrame().to_csv(run / f"codexuw_avoid_trades_{asof}.csv", index=False)
    pd.DataFrame(
        [{"ticker": "BAD", "strategy": "Bull Call Debit Spread", "outcome_status": "OPEN_REVIEW_REQUIRED"}]
    ).to_csv(run / f"codexuw_execute_outcome_ledger_{asof}.csv", index=False)
    pd.DataFrame([{"ticker": "BAD", "catalyst_status": "caution", "catalyst_earnings_date": ""}]).to_csv(
        run / f"codexuw_catalysts_{asof}.csv", index=False
    )
    (run / f"codexuw_trade_report_{asof}.md").write_text("## Action Board\n\n| Status | Ticker |\n|---|---|\n", encoding="utf-8")

    issues = audit_run(run, asof=asof)

    assert any("hard-block token" in issue for issue in issues)


def test_qa_action_board_duplicate_check_is_scoped_to_action_board(tmp_path) -> None:
    run = tmp_path / "codexuw_daily_test"
    run.mkdir()
    asof = "2026-05-06"
    manifest = {
        "execute_rows": 0,
        "watch_rows": 0,
        "research_rows": 0,
        "avoid_rows": 0,
        "run_provenance": {
            "input_files": {
                "exports": {
                    "stock_screener": {"sha256": "a" * 64},
                    "hot_chains": {"sha256": "b" * 64},
                }
            },
            "schwab_snapshot": {"status": "not_required"},
        },
    }
    (run / f"codexuw_manifest_{asof}.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    for name in ["final_trades", "watch_trades", "research_candidates", "avoid_trades", "execute_outcome_ledger"]:
        pd.DataFrame().to_csv(run / f"codexuw_{name}_{asof}.csv", index=False)
    (run / f"codexuw_trade_report_{asof}.md").write_text(
        "## Action Board\n\n"
        "| Status | Ticker |\n"
        "|---|---|\n"
        "| 🔵 Research | AAA |\n\n"
        "## Top Research Near-Misses\n\n"
        "| Status | Ticker |\n"
        "|---|---|\n"
        "| 🔵 Research | AAA |\n",
        encoding="utf-8",
    )

    issues = audit_run(run, asof=asof)

    assert not any("action board duplicate" in issue for issue in issues)


def test_select_ticker_pool_excludes_etfs_but_keeps_stocks() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "close": 700,
                "flow_total_premium": 2_000_000_000,
                "total_open_interest": 10_000_000,
                "avg30_volume": 50_000_000,
                "issue_type": "ETF",
                "full_name": "SPDR S&P 500 ETF",
            },
            {
                "ticker": "MSFT",
                "close": 400,
                "flow_total_premium": 100_000_000,
                "total_open_interest": 1_000_000,
                "avg30_volume": 20_000_000,
                "issue_type": "Common Stock",
                "full_name": "MICROSOFT CORP",
            },
            {
                "ticker": "NFLX",
                "close": 88,
                "flow_total_premium": 90_000_000,
                "total_open_interest": 900_000,
                "avg30_volume": 15_000_000,
                "issue_type": "Common Stock",
                "full_name": "NETFLIX INC",
            },
        ]
    )

    pool = select_ticker_pool(df, max_tickers=10)

    assert set(pool["ticker"]) == {"MSFT", "NFLX"}


def test_select_ticker_pool_zero_means_uncapped() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "close": 100,
                "flow_total_premium": 150_000_000,
                "total_open_interest": 100_000,
                "avg30_volume": 1_000_000,
                "issue_type": "Common Stock",
            },
            {
                "ticker": "BBB",
                "close": 80,
                "flow_total_premium": 120_000_000,
                "total_open_interest": 90_000,
                "avg30_volume": 900_000,
                "issue_type": "Common Stock",
            },
        ]
    )

    pool = select_ticker_pool(df, max_tickers=0)

    assert set(pool["ticker"]) == {"AAA", "BBB"}


def test_select_ticker_pool_reserves_dynamic_sector_coverage() -> None:
    df = pd.DataFrame(
        [
            {"ticker": "TECH1", "sector": "Technology", "close": 100, "flow_total_premium": 300_000_000, "total_open_interest": 100_000, "avg30_volume": 1_000_000, "issue_type": "Common Stock"},
            {"ticker": "TECH2", "sector": "Technology", "close": 100, "flow_total_premium": 250_000_000, "total_open_interest": 100_000, "avg30_volume": 1_000_000, "issue_type": "Common Stock"},
            {"ticker": "HEALTH1", "sector": "Healthcare", "close": 100, "flow_total_premium": 20_000_000, "total_open_interest": 50_000, "avg30_volume": 500_000, "issue_type": "Common Stock"},
        ]
    )

    pool = select_ticker_pool(df, max_tickers=2)

    assert set(pool["ticker"]) == {"TECH1", "HEALTH1"}


def test_is_etf_row_does_not_match_etf_inside_netflix_name() -> None:
    netflix = pd.Series({"ticker": "NFLX", "issue_type": "Common Stock", "full_name": "NETFLIX INC"})
    yieldmax = pd.Series(
        {
            "ticker": "NFLY",
            "issue_type": "ETF",
            "full_name": "YIELDMAX NFLX OPTION INCOME STRATEGY ETF",
        }
    )

    assert not is_etf_row(netflix)
    assert is_etf_row(yieldmax)


def test_generate_candidates_keeps_one_setup_per_selected_ticker_before_scoring() -> None:
    asof = dt.date(2026, 5, 20)
    expiry = dt.date(2026, 6, 18)
    sc_pool = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "close": 100.0,
                "flow_bias": 0.12,
                "flow_total_premium": 250_000_000,
                "iv_rank": 35,
                "iv30d": 0.30,
                "implied_move_perc": 0.04,
                "sector": "Technology",
            },
            {
                "ticker": "NFLX",
                "close": 88.0,
                "flow_bias": -0.05,
                "flow_total_premium": 32_000_000,
                "iv_rank": 27,
                "iv30d": 0.31,
                "implied_move_perc": 0.03,
                "sector": "Communication Services",
            },
        ]
    )
    hot = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "right": "P",
                "strike": 94.0,
                "expiry_dt": expiry,
                "dte": 29,
                "premium": 2_000_000.0,
                "volume": 50_000,
                "open_interest": 50_000,
                "option_symbol": "AAA260618P00094000",
                "bid": 1.2,
                "ask": 1.4,
            },
            {
                "ticker": "NFLX",
                "right": "C",
                "strike": 95.0,
                "expiry_dt": expiry,
                "dte": 29,
                "premium": 250_000.0,
                "volume": 1_000,
                "open_interest": 5_000,
                "option_symbol": "NFLX260618C00095000",
                "bid": 1.0,
                "ask": 1.2,
            },
        ]
    )

    candidates = generate_candidates(sc_pool, hot, pd.DataFrame(), asof=asof, max_candidates=1)

    assert {"AAA", "NFLX"}.issubset(set(candidates["ticker"]))
    nflx = candidates[candidates["ticker"].eq("NFLX")].iloc[0]
    assert nflx["candidate_coverage_source"] == "per_ticker_coverage"

    uncapped = generate_candidates(sc_pool, hot, pd.DataFrame(), asof=asof, max_candidates=0)

    assert {"AAA", "NFLX"}.issubset(set(uncapped["ticker"]))
    assert not uncapped["candidate_coverage_source"].eq("per_ticker_coverage").any()


def test_live_selection_uses_high_conviction_decision_layer() -> None:
    import pandas as pd

    scored = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "sector": "Technology",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-15",
                "dte": 30,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.25,
                "credit": 1.25,
                "spread_width": 5.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "breakeven": 451.0,
                "distance_pct": 0.08,
                "iv30d": 0.25,
                    "realized_volatility_30d": 0.20,
                    "iv_hv_ratio": 1.25,
                "combined_flow_bias": -0.12,
                "score": 6.0,
                "confidence": "Medium",
                "short_leg": "AAA260515C00450000",
                "long_leg": "AAA260515C00455000",
            },
            {
                "ticker": "BBB",
                "sector": "Technology",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-15",
                "dte": 30,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.25,
                "credit": 1.25,
                "spread_width": 5.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "breakeven": 451.0,
                "distance_pct": 0.08,
                "iv30d": 0.25,
                    "realized_volatility_30d": 0.20,
                    "iv_hv_ratio": 1.25,
                "combined_flow_bias": -0.04,
                "score": 9.0,
                "confidence": "High",
                "short_leg": "BBB260515C00450000",
                "long_leg": "BBB260515C00455000",
            },
            {
                "ticker": "CCC",
                "sector": "Technology",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-15",
                "dte": 30,
                "hard_rejects": "",
                "penalties": "wide_bid_ask",
                "credit_pct_width": 0.25,
                "credit": 1.25,
                "spread_width": 5.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "breakeven": 451.0,
                "distance_pct": 0.08,
                "iv30d": 0.25,
                    "realized_volatility_30d": 0.20,
                    "iv_hv_ratio": 1.25,
                "combined_flow_bias": -0.20,
                "score": 8.0,
                "confidence": "High",
                "short_leg": "CCC260515C00450000",
                "long_leg": "CCC260515C00455000",
            },
        ]
    )
    marked = apply_high_conviction_decision_marks(scored)
    final = select_final_trades(
        marked,
        regime={"sizing_stance": "normal"},
        risk_budget=2000,
        recent_performance={"status": "unavailable"},
        max_final_trades=1,
    )

    assert final["ticker"].tolist() == ["AAA"]
    assert marked.loc[marked["ticker"].eq("BBB"), "decision_reason"].iloc[0] == "decision_weak_flow_alignment"
    assert marked.loc[marked["ticker"].eq("CCC"), "decision_reason"].iloc[0] == "decision_marginal_live_liquidity"


def test_live_selection_rejects_secondary_income_below_standing_credit_floor() -> None:
    import pandas as pd

    scored = pd.DataFrame(
        [
            {
                "ticker": "SEC",
                "sector": "Technology",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-15",
                "dte": 23,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.24,
                "credit": 1.2,
                "spread_width": 5.0,
                "max_loss": 380.0,
                "max_profit": 120.0,
                "breakeven": 105.2,
                "distance_pct": 0.04,
                "iv30d": 0.45,
                "combined_flow_bias": -0.20,
                "score": 7.0,
                "confidence": "High",
                "short_leg": "SEC260515C00105000",
                "long_leg": "SEC260515C00110000",
            }
        ]
    )

    marked = apply_high_conviction_decision_marks(scored)
    final = select_final_trades(
        marked,
        regime={"sizing_stance": "normal"},
        risk_budget=2000,
        recent_performance={"status": "unavailable"},
        max_final_trades=1,
    )

    assert marked["decision_reason"].iloc[0] == "decision_credit_below_25pct_width"
    assert final.empty


def _secondary_income_candidate(**overrides) -> dict:
    row = {
        "ticker": "SEC",
        "sector": "Technology",
        "direction": "Bear Call",
        "strategy": "Bear Call Credit Spread",
        "expiry": "2026-08-21",
        "dte": 28,
        "hard_rejects": "",
        "penalties": "",
        "credit_pct_width": 0.27,
        "credit": 1.35,
        "spread_width": 5.0,
        "max_loss": 365.0,
        "max_profit": 135.0,
        "distance_pct": 0.10,
        "iv30d": 0.75,
        "iv_rank": 72.0,
        "combined_flow_bias": -0.20,
        "score": 7.0,
        "confidence": "High",
    }
    row.update(overrides)
    return row


def test_secondary_income_sleeve_requires_volatility_richness() -> None:
    """The secondary income sleeve is still short premium, so it must clear the same
    IV/HV bound as the primary credit lane. It previously bypassed that gate entirely,
    which is how a live run emitted eight trades at IV/HV 0.906-0.933 -- selling
    premium that was cheaper than the realised vol of the underlying."""
    import pandas as pd

    cheap = apply_high_conviction_decision_marks(
        pd.DataFrame([_secondary_income_candidate(realized_volatility_30d=1.00, iv_hv_ratio=0.75)])
    )
    assert bool(cheap["decision_eligible"].iloc[0]) is False
    assert cheap["decision_tier"].iloc[0] != "secondary_income"

    rich = apply_high_conviction_decision_marks(
        pd.DataFrame([_secondary_income_candidate(realized_volatility_30d=0.50, iv_hv_ratio=1.50)])
    )
    assert bool(rich["decision_eligible"].iloc[0]) is True
    assert rich["decision_tier"].iloc[0] == "secondary_income"


def test_secondary_income_sleeve_rejects_near_cash_denominator_artifact() -> None:
    """A tiny realized-vol denominator produces a huge IV/HV ratio on names whose
    absolute premium is negligible (cash/short-duration bond ETFs)."""
    import pandas as pd

    marked = apply_high_conviction_decision_marks(
        pd.DataFrame([_secondary_income_candidate(realized_volatility_30d=0.012, iv_hv_ratio=11.9)])
    )
    assert bool(marked["decision_eligible"].iloc[0]) is False


def test_live_selection_can_return_multiple_high_conviction_trades() -> None:
    import pandas as pd

    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        rows.append(
            {
                "ticker": ticker,
                "sector": "Industrials",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-22",
                "dte": 30,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.25,
                "credit": 1.25,
                "spread_width": 5.0,
                "max_loss": 390.0,
                "max_profit": 110.0,
                "breakeven": 101.1,
                "distance_pct": 0.08,
                "iv30d": 0.25,
                    "realized_volatility_30d": 0.20,
                    "iv_hv_ratio": 1.25,
                "combined_flow_bias": -0.20,
                "score": 7.2,
                "confidence": "High",
                "edge_sample_size": 10,
                "edge_win_rate": 0.65,
                "edge_avg_pnl": 45.0,
                "live_status": "PASS",
                "short_oi": 1000,
                "short_volume": 500,
                "long_oi": 1000,
                "long_volume": 500,
                "short_leg": f"{ticker}260522C00100000",
                "long_leg": f"{ticker}260522C00105000",
            }
        )
    marked = apply_high_conviction_decision_marks(pd.DataFrame(rows))

    final = select_final_trades(
        marked,
        regime={"sizing_stance": "normal"},
        risk_budget=5000,
        recent_performance={"status": "unavailable"},
        max_final_trades=8,
    )

    assert final["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert final["contracts"].tolist() == [2, 2, 2]
    assert final["sizing_label"].str.contains("SIZE-UP").all()
    assert final["sizing_rationale"].str.contains("High confidence").all()


def test_medium_confidence_selection_stays_one_lot_even_when_budget_allows_more() -> None:
    import pandas as pd

    scored = pd.DataFrame(
        [
            {
                "ticker": "MED",
                "sector": "Industrials",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-22",
                "dte": 30,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.25,
                "credit": 1.25,
                "spread_width": 5.0,
                "max_loss": 100.0,
                "max_profit": 110.0,
                "breakeven": 101.1,
                "distance_pct": 0.08,
                "iv30d": 0.25,
                    "realized_volatility_30d": 0.20,
                    "iv_hv_ratio": 1.25,
                "combined_flow_bias": -0.20,
                "score": 6.2,
                "confidence": "Medium",
                "short_leg": "MED260522C00100000",
                "long_leg": "MED260522C00105000",
            }
        ]
    )
    marked = apply_high_conviction_decision_marks(scored)

    final = select_final_trades(
        marked,
        regime={"sizing_stance": "normal"},
        risk_budget=5000,
        recent_performance={"status": "unavailable"},
        max_final_trades=8,
    )

    assert final["contracts"].tolist() == [1]
    assert final["sizing_label"].tolist() == ["1-lot base"]


def test_entry_watchlist_surfaces_low_credit_without_promoting_trade() -> None:
    import pandas as pd

    scored = pd.DataFrame(
        [
            {
                "ticker": "WAIT",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 14,
                "hard_rejects": "",
                "penalties": "credit_below_min_16pct_width;replay_guard_credit_below_validated_band",
                "decision_eligible": False,
                "decision_reason": "decision_final_quality_guard",
                "credit": 0.45,
                "spread_width": 5.0,
                "credit_pct_width": 0.09,
                "pop_delta_proxy": 0.88,
                "score": 4.7,
                "confidence": "Reject",
                "short_leg": "WAIT260515P00100000",
                "long_leg": "WAIT260515P00095000",
            },
            {
                "ticker": "EARN",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 14,
                "hard_rejects": "earnings_within_7d:4",
                "penalties": "credit_below_min_16pct_width",
                "decision_eligible": False,
                "decision_reason": "decision_hard_reject",
                "credit": 0.45,
                "spread_width": 5.0,
                "credit_pct_width": 0.09,
                "score": 4.7,
                "short_leg": "EARN260515P00100000",
                "long_leg": "EARN260515P00095000",
            },
        ]
    )

    watch = build_entry_watchlist(scored)

    assert watch["ticker"].tolist() == ["WAIT"]
    assert watch["watch_kind"].iloc[0] == "price_improvement_credit"
    assert watch["required_credit"].iloc[0] == 1.25
    assert "at least $1.25" in watch["trigger"].iloc[0]


def test_final_selection_keeps_one_position_per_underlying() -> None:
    import pandas as pd

    rows = []
    for ticker, dte, credit_pct in [
        ("TQQQ", 30, 0.29),
        ("TQQQ", 7, 0.28),
        ("TQQQ", 34, 0.27),
        ("TSM", 30, 0.26),
    ]:
        rows.append(
            {
                "ticker": ticker,
                "sector": "Technology",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-22",
                "dte": dte,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": credit_pct,
                "credit": credit_pct * 5.0,
                "spread_width": 5.0,
                "max_loss": 500.0 - credit_pct * 500.0,
                "max_profit": credit_pct * 500.0,
                "breakeven": 101.1,
                "distance_pct": 0.08,
                "iv30d": 0.42,
                "realized_volatility_30d": 0.30,
                "iv_hv_ratio": 1.40,
                "combined_flow_bias": -0.20,
                "score": 7.2,
                "confidence": "High",
                "short_oi": 1000,
                "short_volume": 500,
                "long_oi": 1000,
                "long_volume": 500,
                "short_leg": f"{ticker}260522C00100000",
                "long_leg": f"{ticker}260522C00105000",
            }
        )
    marked = apply_high_conviction_decision_marks(pd.DataFrame(rows))
    assert int(marked["decision_eligible"].sum()) == 4

    final = select_final_trades(
        marked,
        regime={"sizing_stance": "normal"},
        risk_budget=15000,
        recent_performance={"status": "unavailable"},
        max_final_trades=8,
    )

    assert final["ticker"].tolist() == ["TQQQ", "TSM"]
    assert final["ticker"].duplicated().sum() == 0
