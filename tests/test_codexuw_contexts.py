from __future__ import annotations

from codexuw.catalysts import load_catalyst_context
from codexuw.engine import build_entry_watchlist, apply_high_conviction_decision_marks, select_final_trades, select_ticker_pool
from codexuw.portfolio import summarize_positions


def test_summarize_positions_blocks_existing_option_underlyings() -> None:
    payload = {
        "balances": {"total_value": 100_000, "cash": 10_000},
        "positions": [
            {"symbol": "MSFT", "asset_type": "EQUITY", "market_value": 6_000},
            {"symbol": "GOOG  260515P00375000", "asset_type": "OPTION", "underlying": "GOOG", "short_qty": 1, "market_value": -200},
        ],
    }
    summary = summarize_positions(payload)
    assert summary["option_underlyings"] == ["GOOG"]
    assert summary["short_option_underlyings"] == ["GOOG"]
    assert summary["large_equity_exposure"] == {"MSFT": 6000.0}


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


def test_select_ticker_pool_excludes_etfs_but_keeps_stocks() -> None:
    import pandas as pd

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
        ]
    )

    pool = select_ticker_pool(df, max_tickers=10)

    assert pool["ticker"].tolist() == ["MSFT"]


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
                "dte": 16,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.20,
                "credit": 1.0,
                "spread_width": 5.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "breakeven": 451.0,
                "distance_pct": 0.08,
                "iv30d": 0.25,
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
                "dte": 16,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.20,
                "credit": 1.0,
                "spread_width": 5.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "breakeven": 451.0,
                "distance_pct": 0.08,
                "iv30d": 0.25,
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
                "dte": 16,
                "hard_rejects": "",
                "penalties": "wide_bid_ask",
                "credit_pct_width": 0.20,
                "credit": 1.0,
                "spread_width": 5.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "breakeven": 451.0,
                "distance_pct": 0.08,
                "iv30d": 0.25,
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


def test_live_selection_uses_secondary_income_sleeve_when_no_primary() -> None:
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

    assert marked["decision_reason"].iloc[0] == "decision_secondary_income_eligible"
    assert marked["decision_tier"].iloc[0] == "secondary_income"
    assert final["ticker"].tolist() == ["SEC"]


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
                "dte": 21,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.22,
                "credit": 1.1,
                "spread_width": 5.0,
                "max_loss": 390.0,
                "max_profit": 110.0,
                "breakeven": 101.1,
                "distance_pct": 0.08,
                "iv30d": 0.25,
                "combined_flow_bias": -0.20,
                "score": 7.2,
                "confidence": "High",
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
                "dte": 21,
                "hard_rejects": "",
                "penalties": "",
                "credit_pct_width": 0.22,
                "credit": 1.1,
                "spread_width": 5.0,
                "max_loss": 100.0,
                "max_profit": 110.0,
                "breakeven": 101.1,
                "distance_pct": 0.08,
                "iv30d": 0.25,
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
    assert watch["required_credit"].iloc[0] == 0.9
    assert "at least $0.90" in watch["trigger"].iloc[0]
