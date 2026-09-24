import json
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))

import schwab_market as sm  # noqa: E402


def leg(strike, delta, bid, ask):
    return {
        "strike": float(strike),
        "delta": float(delta),
        "bid": float(bid),
        "ask": float(ask),
        "quote_time_ms": 1,
        "bid_size": 10,
        "ask_size": 10,
        "oi": 50,
    }


class TestRegimeGates(unittest.TestCase):
    def test_vix_buckets(self):
        self.assertEqual(sm.vix_regime(15.84), "calm")
        self.assertEqual(sm.vix_regime(16.46), "normal")
        self.assertEqual(sm.vix_regime(17.06), "normal")
        self.assertEqual(sm.vix_regime(22.1), "elevated")
        self.assertEqual(sm.vix_regime(31), "crisis")

    def test_normal_matches_calm_geometry(self):
        calm = sm.resolve_gates("calm")
        normal = sm.resolve_gates("normal")
        self.assertEqual(calm["min_frac"], 0.12)
        self.assertEqual(normal["min_frac"], 0.12)
        self.assertEqual(calm["max_delta"], normal["max_delta"])
        self.assertEqual(calm["min_sigma"], normal["min_sigma"])
        self.assertEqual(sm.resolve_gates("elevated")["min_frac"], 0.25)

    def test_short_clears_and_not_or(self):
        g = sm.resolve_gates("normal")
        # Δ 0.207 / 0.84σ — passes cheap-vol AND, fails old Normal OR (0.20Δ or 0.90σ)
        self.assertTrue(sm.short_clears(delta=-0.207, otm=11.8, sigma=14.0, gates=g))
        old = sm.resolve_gates("normal", min_frac=0.20, or_delta=0.20, min_sigma=0.90, max_delta=0.25)
        self.assertFalse(sm.short_clears(delta=-0.207, otm=11.8, sigma=14.0, gates=old))
        # 0.246Δ / 0.76σ hard-clears 0.25 but fails 0.22 AND 0.80σ
        self.assertFalse(sm.short_clears(delta=-0.246, otm=45.0, sigma=59.15, gates=g))
        self.assertFalse(sm.short_clears(delta=-0.26, otm=20.0, sigma=10.0, gates=g))

    def test_friday_expiries_skip_under_14_dte(self):
        days = sm.friday_expiries(date(2026, 9, 15), min_dte=14, max_dte=60)
        self.assertNotIn("2026-09-18", days)
        self.assertIn("2026-10-02", days)
        self.assertIn("2026-10-16", days)
        self.assertIn("2026-10-30", days)
        self.assertNotIn("2026-11-20", days)


class TestCreditPicker(unittest.TestCase):
    def test_put_credit_picks_frac_then_dollars(self):
        # spot 100, sigma 10 → 0.80σ = 8 pts. 90-delta 0.20 put at 90.
        puts = {
            90.0: leg(90, -0.20, 1.40, 1.50),
            85.0: leg(85, -0.12, 0.30, 0.40),
            80.0: leg(80, -0.08, 0.10, 0.20),
        }
        g = sm.resolve_gates("normal")
        hit = sm._credit_put(puts, spot=100.0, sigma=10.0, gates=g)
        self.assertTrue(hit.get("ok"))
        self.assertEqual(hit["short"], 90.0)
        self.assertEqual(hit["long"], 85.0)
        # conservative credit = 1.40 - 0.40 = 1.00 → 0.20 of width 5
        self.assertGreaterEqual(hit["pricing"]["net"], 1.0)

    def test_old_normal_020_empty_on_intc_like_wing(self):
        puts = {
            87.5: leg(87.5, -0.207, 1.80, 1.90),
            82.5: leg(82.5, -0.14, 0.50, 0.60),
            77.5: leg(77.5, -0.09, 0.20, 0.31),
        }
        spot, sigma = 99.2, 14.0
        cheap = sm._credit_put(puts, spot, sigma, sm.resolve_gates("normal"))
        self.assertTrue(cheap.get("ok"), cheap)
        old = sm.resolve_gates("normal", min_frac=0.20, or_delta=0.20, min_sigma=0.90)
        miss = sm._credit_put(puts, spot, sigma, old)
        self.assertFalse(miss.get("ok"), miss)


class TestSessionFlags(unittest.TestCase):
    def test_normalize_session(self):
        self.assertEqual(sm.normalize_session("ah"), "ah")
        self.assertEqual(sm.normalize_session("after market"), "ah")
        self.assertEqual(sm.normalize_session("post market"), "ah")
        self.assertEqual(sm.normalize_session("EOD"), "ah")
        self.assertEqual(sm.normalize_session("live"), "live")
        self.assertEqual(sm.normalize_session("RTH"), "live")
        self.assertEqual(sm.normalize_session("market-open"), "live")
        self.assertEqual(sm.normalize_session("market live"), "live")
        self.assertEqual(sm.normalize_session(None), "live")
        self.assertEqual(sm.normalize_session(""), "live")

    def test_dated_out_names(self):
        ah = sm.dated_out("2026-09-18", "ah", "scan")
        live = sm.dated_out("2026-09-18", "live", "book")
        self.assertTrue(str(ah).endswith("2026-09-18/scan_ah.json"))
        self.assertTrue(str(live).endswith("2026-09-18/book_live.json"))


class TestScanBoard(unittest.TestCase):
    def test_outage_does_not_skip_universe(self):
        miss = {"ok": False, "reason": "missing spot or ATM straddle ask"}
        dead, streak = set(), {}
        self.assertEqual(
            sm._note_expiry_misses({"2026-11-06": miss}, streak, dead, n_ok=0),
            [],
        )
        self.assertEqual(dead, set())

    def test_listed_gap_skips_after_two_healthy_names(self):
        ok = {"ok": True}
        miss = {"ok": False, "reason": "missing spot or ATM straddle ask"}
        dead, streak = set(), {}
        sm._note_expiry_misses(
            {"2026-10-02": ok, "2026-11-06": miss}, streak, dead, n_ok=1
        )
        newly = sm._note_expiry_misses(
            {"2026-10-02": ok, "2026-11-06": miss}, streak, dead, n_ok=1
        )
        self.assertEqual(newly, ["2026-11-06"])
        self.assertIn("2026-11-06", dead)
        self.assertNotIn("2026-10-02", dead)

    def test_timeout_does_not_increment_or_reset(self):
        ok = {"ok": True}
        miss = {"ok": False, "reason": "missing spot or ATM straddle ask"}
        timeout = {"ok": False, "error": "timed out"}
        dead, streak = set(), {}
        sm._note_expiry_misses(
            {"2026-10-02": ok, "2026-11-06": miss}, streak, dead, n_ok=1
        )
        sm._note_expiry_misses(
            {"2026-10-02": ok, "2026-11-06": timeout}, streak, dead, n_ok=1
        )
        self.assertEqual(dead, set())
        self.assertEqual(streak.get("2026-11-06"), 1)

    def test_ic_frac_in_ic_and_tape(self):
        put = {
            "ok": True,
            "short": 330,
            "long": 320,
            "short_delta": -0.191,
            "sigma_mult": 0.97,
            "pricing": {
                "quoted": True,
                "net": 1.36,
                "credit_width": 0.136,
                "max_profit_1lot": 136,
                "max_loss_1lot": 864,
                "worse_fill": True,
                "width": 10,
            },
        }
        call = {
            "ok": True,
            "short": 390,
            "long": 400,
            "short_delta": 0.206,
            "sigma_mult": 1.14,
            "pricing": {
                "quoted": True,
                "net": 1.22,
                "credit_width": 0.122,
                "max_profit_1lot": 122,
                "max_loss_1lot": 878,
                "worse_fill": True,
                "width": 10,
            },
        }
        ic = {
            "ok": True,
            "put": put,
            "call": call,
            "pricing": {
                "quoted": True,
                "net": 2.58,
                "width": 10.0,
                "credit_width": 0.258,
                "max_profit_1lot": 258,
                "max_loss_1lot": 742,
            },
        }
        structures = {
            "AVGO": {
                "2026-10-16": {
                    "ok": True,
                    "underlying_price": 357.61,
                    "atm_straddle": {"straddle_ask": 28.5},
                    "structures": {
                        "sell_put_credit": put,
                        "sell_call_credit": call,
                        "sell_iron_condor": ic,
                        "buy_call_debit": {"ok": False},
                        "buy_put_debit": {"ok": False},
                    },
                }
            }
        }
        board = sm.scan_board(structures)
        self.assertEqual(board["credits"][0]["k"], "sell_iron_condor")
        self.assertEqual(board["credits"][0]["frac"], 0.258)
        self.assertEqual(board["credits"][0]["pop"], 79)
        wings = [c for c in board["credits"] if c["k"] != "sell_iron_condor"]
        self.assertTrue(all(c.get("in_ic") for c in wings))
        self.assertEqual(board["tape"]["AVGO"]["spot"], 357.61)
        self.assertEqual(board["tape"]["AVGO"]["sigma"]["2026-10-16"], 28.5)

    def test_dead_call_wing_cannot_become_the_condor(self):
        # NVDA 2026-09-23 AH. 245/247.5 long is 0/0, delta -999.
        # short bid − 0 is not a 2.5-wide, and it must not build the IC.
        gates = sm.resolve_gates("calm")
        puts = {
            212.5: leg(212.5, -0.217, 2.33, 2.35),
            202.5: leg(202.5, -0.101, 0.97, 0.99),
        }
        calls = {
            245.0: leg(245, 0.148, 1.27, 1.28),
            247.5: leg(247.5, -999, 0.0, 0.0),
            250.0: leg(250, 0.096, 0.74, 0.76),
            255.0: leg(255, 0.061, 0.44, 0.46),
        }
        put = sm._credit_put(puts, 225.51, 13.95, gates)
        call = sm._credit_call(calls, 225.51, 13.95, gates)
        ic = sm._iron_condor(put, call)
        self.assertTrue(put.get("ok"), put)
        self.assertEqual((put["short"], put["long"]), (212.5, 202.5))
        self.assertAlmostEqual(put["pricing"]["net"], 1.34)
        self.assertEqual(put["pricing"]["width"], 10.0)
        self.assertTrue(put["pricing"].get("quoted"))
        self.assertFalse(call.get("ok"), call)
        self.assertFalse(ic.get("ok"), ic)
        board = sm.scan_board(
            {
                "NVDA": {
                    "2026-10-16": {
                        "ok": True,
                        "underlying_price": 225.51,
                        "atm_straddle": {"straddle_ask": 13.95},
                        "structures": {
                            "sell_put_credit": put,
                            "sell_call_credit": call,
                            "sell_iron_condor": ic,
                            "buy_call_debit": {"ok": False},
                            "buy_put_debit": {"ok": False},
                        },
                    }
                }
            }
        )
        self.assertEqual([c["k"] for c in board["credits"]], ["sell_put_credit"])
        self.assertEqual(board["credits"][0]["sb"], 2.33)
        self.assertEqual(board["credits"][0]["la"], 0.99)

    def test_board_drops_credit_that_only_looks_clean(self):
        fake = {
            "ok": True,
            "short": 245,
            "long": 247.5,
            "short_delta": 0.148,
            "pricing": {
                "ok": True,
                "net": 1.27,
                "credit_width": 0.508,
                "max_profit_1lot": 127,
                "max_loss_1lot": 123,
                "worse_fill": False,
                "width": 2.5,
            },
        }
        board = sm.scan_board(
            {
                "NVDA": {
                    "2026-10-16": {
                        "ok": True,
                        "underlying_price": 225.51,
                        "atm_straddle": {"straddle_ask": 13.95},
                        "structures": {
                            "sell_put_credit": {"ok": False},
                            "sell_call_credit": fake,
                            "sell_iron_condor": {"ok": False},
                            "buy_call_debit": {"ok": False},
                            "buy_put_debit": {"ok": False},
                        },
                    }
                }
            }
        )
        self.assertEqual(board["credits"], [])

    def test_zero_ask_long_is_not_a_credit(self):
        # NVDA 245/247.5: long bid/ask 0 and delta −999. short bid − 0 is not a 2.5-wide.
        dead = sm.vertical_math(
            kind="credit",
            short={"strike": 245, "bid": 1.27, "ask": 1.28},
            long={"strike": 247.5, "bid": 0.0, "ask": 0.0},
        )
        self.assertFalse(dead["ok"])
        locked = sm.vertical_math(
            kind="credit",
            short={"strike": 335, "bid": 3.0, "ask": 3.1},
            long={"strike": 332.5, "bid": 0.1, "ask": 0.2},
        )
        self.assertFalse(locked["ok"])

    def test_compact_vertical_ok_follows_pricing(self):
        bad = sm.compact_vertical(
            {
                "symbol": "AMD",
                "expiry": "2026-10-16",
                "right": "P",
                "kind": "credit",
                "pricing": {"ok": False, "net": -0.1},
                "short": {"strike": 500, "bid": 1, "ask": 2, "delta": -0.2},
                "long": {"strike": 490, "bid": 1, "ask": 2, "delta": -0.1},
            }
        )
        self.assertFalse(bad["ok"])
        self.assertIn("pricing", bad)

    def test_compact_vertical_nearby_strikes_only(self):
        out = sm.compact_vertical(
            {
                "ok": False,
                "reason": "leg missing on Schwab chain",
                "symbol": "XOM",
                "expiry": "2026-10-16",
                "right": "P",
                "kind": "credit",
                "short": None,
                "long": None,
                "nearby": [
                    {"strike": 110, "bid": 1, "ask": 2, "oi": 9},
                    {"strike": 105, "bid": 1, "ask": 2, "oi": 8},
                ],
            }
        )
        self.assertEqual(out["nearby"], [110, 105])
        self.assertNotIn("oi", json.dumps(out))

    def test_board_uses_per_expiry_spot(self):
        def put(spot_tag):
            return {
                "ok": True,
                "short": 100,
                "long": 95,
                "short_delta": -0.2,
                "pricing": {
                    "quoted": True,
                    "net": 1.2,
                    "credit_width": 0.24,
                    "max_profit_1lot": 120,
                    "max_loss_1lot": 380,
                    "worse_fill": False,
                    "width": 5,
                },
            }

        structures = {
            "AAPL": {
                "2026-10-02": {
                    "ok": True,
                    "underlying_price": 330.0,
                    "atm_straddle": {"straddle_ask": 10.0},
                    "structures": {
                        "sell_put_credit": put("a"),
                        "sell_call_credit": {"ok": False},
                        "sell_iron_condor": {"ok": False},
                        "buy_call_debit": {"ok": False},
                        "buy_put_debit": {"ok": False},
                    },
                },
                "2026-10-16": {
                    "ok": True,
                    "underlying_price": 340.0,
                    "atm_straddle": {"straddle_ask": 12.0},
                    "structures": {
                        "sell_put_credit": put("b"),
                        "sell_call_credit": {"ok": False},
                        "sell_iron_condor": {"ok": False},
                        "buy_call_debit": {"ok": False},
                        "buy_put_debit": {"ok": False},
                    },
                },
            }
        }
        board = sm.scan_board(structures)
        by_exp = {c["exp"]: c["spot"] for c in board["credits"]}
        self.assertEqual(by_exp["2026-10-02"], 330.0)
        self.assertEqual(by_exp["2026-10-16"], 340.0)

    def test_iron_condor_forms_when_one_wing_under_100(self):
        put = {
            "ok": True,
            "short": 215,
            "long": 205,
            "short_delta": -0.216,
            "pricing": {
                "quoted": True,
                "net": 1.34,
                "width": 10.0,
                "credit_width": 0.134,
                "max_profit_1lot": 134,
                "max_loss_1lot": 866,
            },
        }
        call = {
            "ok": True,
            "short": 245,
            "long": 250,
            "short_delta": 0.199,
            "pricing": {
                "quoted": True,
                "net": 0.70,
                "width": 5.0,
                "credit_width": 0.14,
                "max_profit_1lot": 70,
                "max_loss_1lot": 430,
            },
        }
        ic = sm._iron_condor(put, call)
        self.assertTrue(ic.get("ok"), ic)
        self.assertAlmostEqual(ic["pricing"]["net"], 2.04)
        self.assertEqual(ic["pricing"]["width"], 10.0)
        self.assertAlmostEqual(ic["pricing"]["credit_width"], 0.204)

    def test_scan_one_name_one_http_for_window(self):
        calls = []

        class Cfg:
            token_path = "t"

        class Svc:
            def get_option_chain(self, symbol, **kw):
                calls.append(kw)
                return {"underlyingPrice": 228.0, "callExpDateMap": {}, "putExpDateMap": {}}

            def summarize_option_chain(self, symbol, payload):
                return {"underlying_price": 228.0}

        orig = sm.service
        sm.service = lambda: (Svc(), Cfg())
        try:
            rows = sm._scan_one_name(
                "NVDA",
                ["2026-10-09", "2026-10-16", "2026-11-20"],
                24,
                "calm",
            )
        finally:
            sm.service = orig
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["from_date"], "2026-10-09")
        self.assertEqual(calls[0]["to_date"], "2026-11-20")
        self.assertEqual(set(rows), {"2026-10-09", "2026-10-16", "2026-11-20"})
        self.assertTrue(all(r.get("reason") == "missing spot or ATM straddle ask" for r in rows.values()))

    def test_fire_clean_is_a_copy(self):
        st = {
            "ok": True,
            "long": 72.5,
            "short": 77.5,
            "long_delta": 0.47,
            "pricing": {
                "quoted": True,
                "net": 1.53,
                "debit_width": 0.31,
                "max_profit_1lot": 347,
                "max_loss_1lot": 153,
                "worse_fill": False,
            },
        }
        structures = {
            "NFLX": {
                "2026-10-16": {
                    "ok": True,
                    "underlying_price": 72.0,
                    "atm_straddle": {"straddle_ask": 5.0},
                    "structures": {
                        "sell_put_credit": {"ok": False},
                        "sell_call_credit": {"ok": False},
                        "sell_iron_condor": {"ok": False},
                        "buy_call_debit": st,
                        "buy_put_debit": {"ok": False},
                    },
                }
            }
        }
        board = sm.scan_board(structures)
        board["fire_clean"][0]["net"] = 0
        self.assertEqual(board["fire"][0]["net"], 1.53)


def _occ(right, strike, bid, ask, delta):
    return {
        "putCall": right,
        "strikePrice": strike,
        "bid": bid,
        "ask": ask,
        "delta": delta,
    }


def _exp_map(rows):
    strikes = {}
    for row in rows:
        strikes.setdefault(str(row["strikePrice"]), []).append(row)
    return {"2026-10-16:20": strikes}


class TestChainToBoard(unittest.TestCase):
    """Schwab-shaped chain → score → board. Not handmade pricing dicts."""

    def _nvda(self, *, call_255_ask, dead_ask=0):
        return {
            "underlyingPrice": 225.51,
            "callExpDateMap": _exp_map(
                [
                    _occ("CALL", 225, 0, 0, -999),
                    _occ("CALL", 227.5, 6.8, 7.0, 0.51),
                    _occ("CALL", 245, 1.27, 1.28, 0.148),
                    _occ("CALL", 247.5, 0, dead_ask, -999),
                    _occ("CALL", 250, 0.74, 0.76, 0.096),
                    _occ("CALL", 255, 0.44, call_255_ask, 0.061),
                ]
            ),
            "putExpDateMap": _exp_map(
                [
                    _occ("PUT", 225, 6.7, 6.95, -0.49),
                    _occ("PUT", 222.5, 0, 0, -999),
                    _occ("PUT", 212.5, 2.33, 2.35, -0.217),
                    _occ("PUT", 202.5, 0.97, 0.99, -0.101),
                ]
            ),
        }

    def test_zero_ask_atm_does_not_shrink_sigma(self):
        chain = self._nvda(call_255_ask=0.46)
        atm = sm.atm_straddle(chain, 225.51, expiry="2026-10-16")
        self.assertAlmostEqual(atm["straddle_ask"], 13.95)
        self.assertEqual(atm["call"]["strike"], 227.5)
        self.assertEqual(atm["put"]["strike"], 225)

    def test_dead_wing_chain_prints_the_put_not_the_condor(self):
        class Cfg:
            token_path = "t"

        scored = sm._structures_from_payload(
            "NVDA",
            "2026-10-16",
            self._nvda(call_255_ask=0.46),
            regime="calm",
            cfg=Cfg(),
        )
        self.assertTrue(scored.get("ok"), scored)
        ss = scored["structures"]
        self.assertTrue(ss["sell_put_credit"].get("ok"), ss["sell_put_credit"])
        self.assertEqual(ss["sell_put_credit"]["short"], 212.5)
        self.assertEqual(ss["sell_put_credit"]["long"], 202.5)
        self.assertAlmostEqual(ss["sell_put_credit"]["pricing"]["net"], 1.34)
        self.assertFalse(ss["sell_call_credit"].get("ok"))
        self.assertFalse(ss["sell_iron_condor"].get("ok"))
        board = sm.scan_board({"NVDA": {"2026-10-16": scored}})
        self.assertEqual([c["k"] for c in board["credits"]], ["sell_put_credit"])
        row = board["credits"][0]
        self.assertFalse(row.get("in_ic"))
        self.assertEqual(row["sb"], 2.33)
        self.assertEqual(row["la"], 0.99)
        self.assertAlmostEqual(board["tape"]["NVDA"]["sigma"]["2026-10-16"], 13.95)

    def test_quoted_call_wing_still_forms_the_condor(self):
        class Cfg:
            token_path = "t"

        # 245 bid 1.27 − 255 ask 0.05 = 1.22 on a 10-wide (frac 0.122).
        scored = sm._structures_from_payload(
            "NVDA",
            "2026-10-16",
            self._nvda(call_255_ask=0.05, dead_ask=0),
            regime="calm",
            cfg=Cfg(),
        )
        ss = scored["structures"]
        self.assertTrue(ss["sell_call_credit"].get("ok"), ss["sell_call_credit"])
        self.assertEqual(ss["sell_call_credit"]["long"], 255.0)
        self.assertTrue(ss["sell_iron_condor"].get("ok"), ss["sell_iron_condor"])
        board = sm.scan_board({"NVDA": {"2026-10-16": scored}})
        kinds = [c["k"] for c in board["credits"]]
        self.assertIn("sell_iron_condor", kinds)
        wings = [c for c in board["credits"] if c["k"] != "sell_iron_condor"]
        self.assertTrue(wings)
        self.assertTrue(all(c.get("in_ic") for c in wings))


if __name__ == "__main__":
    unittest.main()
