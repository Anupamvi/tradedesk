import unittest
from copy import deepcopy
from dataclasses import replace

from corat.options import choose_debit_spread, choose_option_structure, evaluate_option_evidence, model_option_economics
from corat.scoring import build_stock_plan, choose_vehicle, model_stock_economics, score_candidate
from tests.helpers import empty_option, history, setup, snapshot


def chain_rows(liquid=True):
    values = []
    for strike, delta, cbid, cask, pbid, pask in [
        (110, 0.70, 12.0, 12.3, 1.0, 1.2),
        (120, 0.55, 7.0, 7.3, 3.0, 3.2),
        (130, 0.28, 2.5, 2.7, 7.5, 7.8),
        (140, 0.15, 0.8, 1.0, 13.0, 13.4),
    ]:
        values.append({
            "ticker": "AAA", "tradeDate": "2026-08-27", "expirDate": "2026-10-02", "dte": 37,
            "strike": strike, "stockPrice": 120, "callBidPrice": cbid, "callAskPrice": cask,
            "putBidPrice": pbid, "putAskPrice": pask, "callValue": (cbid+cask)/2,
            "putValue": (pbid+pask)/2, "callOpenInterest": 1000 if liquid else 1,
            "putOpenInterest": 1000 if liquid else 1, "callVolume": 200 if liquid else 0,
            "putVolume": 200 if liquid else 0, "delta": delta, "gamma": 0.01,
            "theta": -0.03, "vega": 0.08, "smvVol": 0.25, "updatedAt": "2026-08-27T20:00:00Z",
        })
    return values


class OptionScoringTest(unittest.TestCase):
    def test_bull_debit_spread_uses_conservative_fill(self):
        result = choose_debit_spread(chain_rows(), "BULLISH", 130, 10, 100, 10, 0.15)
        self.assertTrue(result.valid)
        self.assertGreater(result.expected_entry, 0)
        self.assertLess(result.expected_entry, result.natural_entry)
        self.assertAlmostEqual(result.maximum_loss, result.expected_entry * 100)

    def test_low_oi_is_disclosed_not_arbitrarily_rejected(self):
        result = choose_debit_spread(chain_rows(False), "BULLISH", 130, 10, 100, 10, 0.15)
        self.assertTrue(result.valid)
        self.assertEqual(result.legs[0].open_interest, 1)

    def test_missing_two_sided_quote_is_rejected(self):
        rows = chain_rows(False)
        for row in rows:
            row["callBidPrice"] = 0
            row["putBidPrice"] = 0
        result = choose_debit_spread(rows, "BULLISH", 130, 10, 100, 10, 0.15)
        self.assertFalse(result.valid)

    def test_explicitly_empty_bid_ask_sizes_are_not_executable(self):
        rows = chain_rows()
        for row in rows:
            row.update({"callBidSize": 0, "callAskSize": 0, "putBidSize": 0, "putAskSize": 0})
        result = choose_debit_spread(rows, "BULLISH", 130, 10, 0, 0, 1.0)
        self.assertFalse(result.valid)

    def test_attractive_option_without_exact_earnings_date_falls_back_to_stock(self):
        plan = replace(build_stock_plan(snapshot(), setup(), 100_000, 0.01), reward_risk_2=1.0)
        option = choose_debit_spread(chain_rows(), "BULLISH", 130, 10, 100, 10, 0.15)
        option = replace(option, theoretical_edge=0.10, reward_risk=2.0, valid=True, reasons=[])
        vehicle, reason = choose_vehicle(
            plan,
            option,
            {"iv_forecast_realized_ratio":1.0,"next_earnings_date":""},
            as_of="2026-08-27",
        )
        self.assertEqual(vehicle,"STOCK")
        self.assertIn("Earnings timing",reason)

    def test_orats_weeks_safely_beyond_hold_do_not_create_exact_date_veto(self):
        plan = replace(build_stock_plan(snapshot(), setup(), 100_000, 0.01), reward_risk_2=1.0)
        option = choose_debit_spread(chain_rows(), "BULLISH", 130, 10, 100, 10, 0.15)
        vehicle, _ = choose_vehicle(
            plan,
            option,
            {"weeks_to_next_earnings":11},
            as_of="2026-08-27",
            stock_economics={"expected_profit_per_share":1.0,"expected_return_on_capital":0.1},
            option_economics={"expected_profit_dollars":100.0,"expected_return_on_max_loss":0.2},
        )
        self.assertEqual(vehicle, "OPTIONS")

    def test_fund_option_does_not_require_company_earnings_date(self):
        plan = replace(build_stock_plan(snapshot(), setup(), 100_000, 0.01), reward_risk_2=1.0)
        option = choose_debit_spread(chain_rows(), "BULLISH", 130, 10, 100, 10, 0.15)
        vehicle, _ = choose_vehicle(
            plan,
            option,
            {"weeks_to_next_earnings": 0},
            as_of="2026-08-27",
            stock_economics={"expected_profit_per_share": 1.0, "expected_return_on_capital": 0.01},
            option_economics={"expected_profit_dollars": 100.0, "expected_return_on_max_loss": 0.20},
            earnings_applicable=False,
        )
        self.assertEqual(vehicle, "OPTIONS")

    def test_checked_forward_calendar_can_clear_ordinary_option_hold(self):
        plan = build_stock_plan(snapshot(), setup(), 100_000, 0.01)
        option = choose_debit_spread(chain_rows(), "BULLISH", 130, 10, 0, 0, 1.0)
        vehicle, _ = choose_vehicle(
            plan,
            option,
            {"earnings_calendar_clear_through": "2026-09-20"},
            as_of="2026-08-27",
            stock_economics={"expected_profit_per_share": 1.0, "expected_return_on_capital": 0.01},
            option_economics={"expected_profit_dollars": 100.0, "expected_return_on_max_loss": 0.20},
        )
        self.assertEqual(vehicle, "OPTIONS")

    def test_option_search_compares_all_expirations(self):
        later = chain_rows()
        for row in later:
            row["putBidPrice"] = 0
            row["putAskPrice"] = 0
        nearer = deepcopy(later)
        for row in nearer:
            row["expirDate"] = "2026-09-18"
            row["dte"] = 23
            row["callBidPrice"] *= 0.65
            row["callAskPrice"] *= 0.65
            row["callValue"] = (row["callBidPrice"] + row["callAskPrice"]) / 2.0
        returns = [0.03, 0.05, 0.04, 0.02] * 5
        option = choose_option_structure(
            later + nearer, "BULLISH", 130, 10, 0, 0, 1.0,
            scenario_returns=returns, stop_return=0.05, target_return=0.0833,
        )
        self.assertEqual(option.expiration, "2026-09-18")

    def test_credit_spread_economics_are_modeled_as_credit(self):
        rows = chain_rows()
        for row in rows:
            row["callBidPrice"] = 0
            row["callAskPrice"] = 0
        returns = [0.02, 0.03, 0.01, 0.04] * 5
        option = choose_option_structure(
            rows, "BULLISH", 130, 10, 0, 0, 1.0,
            scenario_returns=returns, stop_return=0.05, target_return=0.0833,
        )
        economics = model_option_economics(option, 120, "BULLISH", 10, returns, stop_return=0.05, target_return=0.0833)
        self.assertEqual(option.strategy, "BULL PUT CREDIT SPREAD")
        self.assertEqual(option.debit_credit, "CREDIT")
        self.assertGreater(economics["expected_profit_dollars"], 0)

    def test_structure_is_selected_on_train_and_reported_on_untouched_holdout(self):
        returns = ([0.05] * 13) + ([-0.05] * 7)
        paths = [[value] for value in returns]
        option = choose_option_structure(
            chain_rows(), "BULLISH", 130, 10, 0, 0, 1.0,
            scenario_returns=returns, scenario_paths=paths,
            stop_return=0.05, target_return=0.0833,
        )
        economics = evaluate_option_evidence(
            option, 120, "BULLISH", 10, returns,
            scenario_paths=paths, stop_return=0.05, target_return=0.0833,
        )
        self.assertEqual(option.selection_train_size, 13)
        self.assertEqual(option.selection_test_size, 7)
        self.assertGreater(economics["train_expected_profit_dollars"], 0)
        self.assertLess(economics["expected_profit_dollars"], 0)
        self.assertEqual(economics["evidence_role"], "HELD_OUT_RECENT_PATHS")

    def test_stock_plan_sizes_from_risk_not_confidence(self):
        plan = build_stock_plan(snapshot(), setup(), 100_000, 0.01)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.units, int(1000 / plan.risk_per_share))

    def test_displayed_entry_zone_never_understates_stop_risk(self):
        bullish = build_stock_plan(snapshot(), setup(), 100_000, 0.01)
        self.assertEqual(bullish.entry_high, bullish.risk_basis_price)
        self.assertAlmostEqual(bullish.risk_per_share, bullish.entry_high - bullish.stop)
        bearish = build_stock_plan(snapshot(), setup(direction="BEARISH"), 100_000, 0.01)
        self.assertEqual(bearish.entry_low, bearish.risk_basis_price)
        self.assertAlmostEqual(bearish.risk_per_share, bearish.stop - bearish.entry_low)

    def test_intraday_stop_is_charged_before_same_bar_target(self):
        snap = snapshot()
        plan = build_stock_plan(snap, setup(), 100_000, 0.01)
        sample = history(False)
        sample = replace(
            sample,
            sample_size=1,
            primary_returns=[0.05],
            primary_paths=[[0.05]],
            primary_adverse_paths=[[-0.05]],
            primary_favorable_paths=[[0.10]],
        )
        economics = model_stock_economics(snap, plan, sample)
        self.assertAlmostEqual(economics["expected_profit_per_share"], -plan.risk_per_share)

    def test_stop_uses_setup_invalidation_not_tighter_unrelated_support(self):
        snap = snapshot()
        signal = setup()
        plan = build_stock_plan(snap, signal, 100_000, 0.01)
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(plan.stop, snap.prior_high_20d - 0.25 * snap.atr14)

    def test_missing_context_and_reprice_do_not_block_positive_economics(self):
        snap = snapshot()
        signal = setup()
        plan = build_stock_plan(snap, signal, 100_000, 0.01)
        scored = score_candidate(
            snap, signal, plan, empty_option(), "STOCK", {"status":"AVAILABLE","next_earnings_date":"2026-12-01"},
            {"catalysts":[],"catalyst_strength":0,"x_strength":0,"flow_strength":0}, history(),
            {"state":"ACCELERATING LEADER"}, "STRONG RISK-ON TREND", 5, 25_000_000, 1.7, 75,
            True, True, True, False,
            {"modeled_pop":0.6,"expected_profit_per_share":2.0,"model_sample_size":30},
        )
        self.assertEqual(scored["status"], "TARGET TRADE")
        self.assertFalse(scored["blockers"])
        self.assertTrue(any("catalyst" in note for note in scored["notes"]))
        self.assertTrue(any("Schwab" in note for note in scored["notes"]))

    def test_small_history_sample_is_disclosed_not_randomly_blocked(self):
        snap = snapshot()
        signal = setup()
        plan = build_stock_plan(snap, signal, 100_000, 0.01)
        scored = score_candidate(
            snap, signal, plan, empty_option(), "STOCK", {"status":"AVAILABLE","next_earnings_date":"2026-12-01"},
            {"catalysts":[{"title":"x"}],"catalyst_strength":1,"x_strength":0,"flow_strength":0}, history(False),
            {"state":"ACCELERATING LEADER"}, "STRONG RISK-ON TREND", 5, 25_000_000, 1.7, 75,
            True, True, True, True,
            {"modeled_pop":0.6,"expected_profit_per_share":2.0,"model_sample_size":5},
        )
        self.assertEqual(scored["status"], "TARGET TRADE")
        self.assertTrue(any("sample" in note for note in scored["notes"]))

    def test_nonpositive_expected_profit_is_not_a_target(self):
        snap = snapshot()
        signal = setup()
        plan = build_stock_plan(snap, signal, 100_000, 0.01)
        scored = score_candidate(
            snap, signal, plan, empty_option(), "STOCK", {"status":"AVAILABLE","next_earnings_date":"2026-12-01"},
            {"catalysts":[],"catalyst_strength":0,"x_strength":0,"flow_strength":0}, history(),
            {"state":"ACCELERATING LEADER"}, "STRONG RISK-ON TREND", 5, 25_000_000, 1.7, 75,
            True, True, True, False,
            {"modeled_pop":0.7,"expected_profit_per_share":-0.25,"model_sample_size":30},
        )
        self.assertEqual(scored["status"], "NO TRADE — EDGE NOT POSITIVE")
        self.assertIn("Modeled expected profit is not positive", scored["blockers"])
