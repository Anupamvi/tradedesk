import unittest
from datetime import date, datetime, timezone

from cultra.reports import CandidateRow, DailyBoardData, render_daily_board


def ticket_payload(identifier, rank=5.25 / 220.0):
    return {
        "candidate_id": identifier,
        "symbol": "AAPL",
        "strategy_id": "CALL_DEBIT_VERTICAL",
        "hypothesis_id": "CALL_DEBIT_VERTICAL__DIRECTIONAL_COMPOSITE_V1__H40",
        "evidence_state": "MANUAL_TICKET_ENABLED",
        "legs": [
            {"occ_symbol": "AAPL260918C00100000", "action": "BUY", "ratio": 1, "expiration": "2026-09-18", "strike": 100.0, "option_type": "CALL"},
            {"occ_symbol": "AAPL260918C00105000", "action": "SELL", "ratio": 1, "expiration": "2026-09-18", "strike": 105.0, "option_type": "CALL"},
        ],
        "leg_quotes": [
            {"occ_symbol": "AAPL260918C00100000", "bid": 4.0, "ask": 4.2},
            {"occ_symbol": "AAPL260918C00105000", "bid": 1.8, "ask": 2.0},
        ],
        "probabilities": {
            "pop_net": {"point": 0.61, "lower": 0.53, "upper": 0.68, "sample_size": 150, "model_version": "pop-v3", "calibration_start": "2025-01-01", "calibration_end": "2026-06-30"},
            "p_target": {"point": 0.42},
            "p_stop": {"point": 0.19},
            "p_max_loss": {"point": 0.08},
        },
        "edge": {
            "net_expected_value": 18.5,
            "conservative_net_expected_value": 5.25,
            "maximum_loss": 220.0,
            "maximum_profit": 280.0,
            "executable_limit_price": 2.20,
            "price_convention": "DEBIT",
            "breakevens": [102.2],
            "target_pnl": 110.0,
            "stop_pnl": -77.0,
            "expected_shortfall": 200.0,
            "adverse_gap_stress_loss": 220.0,
            "conservative_return_on_max_loss": rank,
        },
        "quantity": "USER DETERMINED",
        "ranking_score": rank,
        "orats_snapshot_id": "snapshot-001",
        "provider_trade_date": "2026-08-28",
        "analytical_fields": ["confidence", "forecastVol", "iv"],
        "model_calculation": {
            "features": [["confidence", 0.72], ["iv_forecast_gap", 0.08]],
            "selection_point_return_on_max_loss": 0.10,
            "selection_conservative_return_on_max_loss": 0.04,
            "calculation_id": "c" * 64,
            "calculation_version": "CULTRA_CURRENT_MODEL_CALCULATION_V2",
        },
        "underlying_quote": {"bid": 199.9, "ask": 200.1, "timestamp": "2026-08-30T20:00:00+00:00"},
        "thesis": "uptrend with forecast-vol discount",
        "signal": "directional-composite-v1",
        "policy": {
            "entry_condition": "enter only at the stated debit",
            "profit_target": "close at +50 percent",
            "stop_condition": "close at -35 percent",
            "time_exit": "close after 40 sessions",
            "invalidation": "trend signal reverses",
            "assignment_handling": "close before exercise",
            "next_review": "2026-08-31"
        },
        "event_evidence": {"status": "CLEAR", "earnings_date": "2026-10-20", "dividend_dates": [], "source": "company IR"},
        "evidence": {
            "training": {"resolved_trades": 300, "expectancy": 12.0, "lower_confidence_bound": 4.0},
            "validation": {"resolved_trades": 180, "expectancy": 9.0, "lower_confidence_bound": 2.0},
            "holdout": {"resolved_trades": 150, "expectancy": 8.0, "lower_confidence_bound": 1.0}
        },
    }


class ReportTests(unittest.TestCase):
    def test_board_leads_unproven_and_separates_every_category(self):
        board = DailyBoardData(
            as_of=date(2026, 8, 30),
            run_id="report-run",
            overall_status="UNPROVEN",
            strategy_states={"LONG_CALL": "UNPROVEN", "IRON_CONDOR": "HOLDOUT_PASS"},
            watchlist=(CandidateRow("w1", "MSFT", "LONG_CALL", "waiting", "WATCHLIST"),),
            rejected=(CandidateRow("r1", "TSLA", "LONG_PUT", "failed edge", "REJECTED"),),
            data_unavailable=(
                CandidateRow("d1", "NVDA", "LONG_CALL", "stale quote", "DATA_UNAVAILABLE"),
            ),
            budget_unresolved=(
                CandidateRow(
                    "b1",
                    "META",
                    "IRON_CONDOR",
                    "request envelope exhausted",
                    "NOT_FULLY_EVALUATED_BUDGET",
                ),
            ),
        )
        text = render_daily_board(board)
        self.assertTrue(text.startswith("# Cultra Daily Board"))
        self.assertLess(text.index("Overall status: `UNPROVEN`"), text.index("Strategy evidence"))
        for heading in (
            "Eligible manual-review tickets",
            "Watchlist",
            "Rejected",
            "Data unavailable",
            "Not fully evaluated — request budget",
        ):
            self.assertIn(heading, text)
        for identifier in ("w1", "r1", "d1", "b1"):
            self.assertIn(identifier, text)

    def test_ticket_displays_pop_edge_exact_payload_and_quantity(self):
        text = render_daily_board(
            DailyBoardData(
                as_of=date(2026, 8, 30),
                run_id="ticket-run",
                overall_status="EVIDENCE_GATED_ACTIVE",
                tickets=(ticket_payload("eligible-1"),),
            )
        )
        self.assertIn("61.0%", text)
        self.assertIn("$18.50", text)
        self.assertIn("$5.25", text)
        self.assertIn("BUY 1x 2026-09-18 100.0 CALL", text)
        self.assertNotIn("AAPL260918C00100000", text)
        self.assertIn("snapshot-001", text)
        self.assertIn("Quantity: **USER DETERMINED**", text)
        self.assertIn("n=150", text)
        self.assertIn("company IR", text)
        self.assertIn("confidence=0.72", text)
        self.assertIn("close at -35 percent", text)
        self.assertIn("close before exercise", text)
        self.assertNotIn("MISSING", text)

    def test_renderer_never_applies_a_top_n_cap(self):
        values = tuple(
            CandidateRow(
                "watch-%03d" % index,
                "SYM%d" % index,
                "LONG_CALL",
                "visible",
                "WATCHLIST",
                rank_score=index / 1000.0,
            )
            for index in range(75)
        )
        text = render_daily_board(
            DailyBoardData(
                as_of=date(2026, 8, 30),
                run_id="uncapped-run",
                watchlist=values,
            )
        )
        self.assertEqual(text.count("| watch-"), 75)
        self.assertIn("watch-074", text)

    def test_all_tickets_are_ranked_descending_with_deterministic_ties(self):
        values = [
            ticket_payload("low", 0.01),
            ticket_payload("tie-b", 0.03),
            ticket_payload("high", 0.05),
            ticket_payload("tie-a", 0.03),
        ]
        values.extend(
            ticket_payload("extra-%02d" % index, 0.02) for index in range(30)
        )
        text = render_daily_board(
            DailyBoardData(
                as_of=date(2026, 8, 30),
                run_id="ranked-uncapped",
                overall_status="EVIDENCE_GATED_ACTIVE",
                tickets=tuple(values),
            )
        )
        self.assertEqual(text.count("### "), len(values))
        self.assertLess(
            text.index("(`high`)"),
            text.index("(`tie-a`)"),
        )
        self.assertLess(
            text.index("(`tie-a`)"),
            text.index("(`tie-b`)"),
        )
        self.assertLess(
            text.index("(`tie-b`)"),
            text.index("(`low`)"),
        )

    def test_strategy_rejection_reason_is_visible(self):
        text = render_daily_board(
            DailyBoardData(
                as_of=date(2026, 8, 30),
                run_id="rejected-family",
                strategy_states={"LONG_CALL": "REJECTED"},
                strategy_rejection_reasons={
                    "LONG_CALL": "holdout lower bound failed"
                },
            )
        )
        self.assertIn(
            "LONG_CALL (`REJECTED`): holdout lower bound failed",
            text,
        )


if __name__ == "__main__":
    unittest.main()
