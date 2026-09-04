import json
import tempfile
import unittest
from pathlib import Path

from compoundcore.book import (
    book_view,
    empty_state,
    load_state,
    parse_money,
    save_state,
)
from compoundcore.sleeve import weights


class TestBook(unittest.TestCase):
    def test_parse_money(self):
        self.assertEqual(parse_money("$1,000"), 1000.0)
        self.assertEqual(parse_money(""), 0.0)
        with self.assertRaises(ValueError):
            parse_money(-1, "VOO")

    def test_empty_book_is_not_present(self):
        view = book_view(positions=empty_state()["book"]["positions"], submitted_at=None)
        self.assertFalse(view["present"])
        self.assertIsNone(view["projections"])
        self.assertFalse(view["pnl_ready"])

    def test_roundtrip_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.json"
            state = empty_state()
            state["planner"]["amount"] = 250000
            state["planner"]["weekly"] = 500
            state["book"]["positions"]["VOO"]["cost"] = 120000
            state["book"]["positions"]["VOO"]["current"] = 132000
            state["book"]["positions"]["VXUS"]["cost"] = 50000
            state["book"]["positions"]["VXUS"]["current"] = 47000
            state["book"]["monthly_add"] = 1000
            state["book"]["submitted_at"] = "2026-09-04T12:00:00Z"
            save_state(state, path)
            loaded = load_state(path)
            self.assertEqual(loaded["planner"]["amount"], 250000.0)
            self.assertEqual(loaded["book"]["positions"]["VOO"]["cost"], 120000.0)
            self.assertEqual(loaded["book"]["positions"]["VOO"]["current"], 132000.0)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["book"]["compare_to"], "default")
            self.assertEqual(raw["book"]["holdings"]["VOO"], 120000.0)

    def test_migrates_legacy_holdings_to_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "planner": {"amount": 0, "weekly": 0, "monthly": 0},
                        "book": {
                            "holdings": {"VOO": 48000, "VGT": 0, "SMH": 0, "VB": 0, "VXUS": 0, "GLDM": 0, "VGSH": 0},
                            "monthly_add": 0,
                            "compare_to": "default",
                            "submitted_at": "2026-09-04T12:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_state(path)
            self.assertEqual(loaded["book"]["positions"]["VOO"]["cost"], 48000.0)
            self.assertEqual(loaded["book"]["positions"]["VOO"]["current"], 0.0)

    def test_real_gain_and_loss(self):
        positions = {t: {"cost": 0.0, "current": 0.0, "shares": 0.0} for t in weights("default")}
        positions["VOO"] = {"cost": 48000, "current": 52800, "shares": 80}
        positions["SMH"] = {"cost": 7000, "current": 6300, "shares": 10}
        view = book_view(
            positions=positions,
            monthly_add=0,
            compare_to="default",
            submitted_at="now",
        )
        self.assertTrue(view["pnl_ready"])
        self.assertEqual(view["invested"], 55000.0)
        self.assertEqual(view["market"], 59100.0)
        self.assertEqual(view["pnl"], 4100.0)
        self.assertAlmostEqual(view["pnl_pct"], 4100.0 / 55000.0, places=10)
        voo = [r for r in view["rows"] if r["ticker"] == "VOO"][0]
        smh = [r for r in view["rows"] if r["ticker"] == "SMH"][0]
        self.assertEqual(voo["pnl"], 4800.0)
        self.assertEqual(smh["pnl"], -700.0)

    def test_unmarked_book_is_not_fake_pnl(self):
        holdings = {t: 0.0 for t in weights("default")}
        holdings["VOO"] = 48000
        view = book_view(holdings, submitted_at="now")
        self.assertTrue(view["present"])
        self.assertFalse(view["pnl_ready"])
        self.assertIsNone(view["pnl"])

    def test_mark_with_prices_keeps_cost(self):
        from compoundcore.book import apply_refresh, mark_with_prices

        positions = {t: {"cost": 0.0, "current": 0.0, "shares": 0.0} for t in weights("default")}
        positions["VOO"] = {"cost": 48000, "current": 48000, "shares": 80}
        marked = mark_with_prices(positions, {"VOO": 650.0})
        self.assertEqual(marked["VOO"]["cost"], 48000.0)
        self.assertEqual(marked["VOO"]["current"], 52000.0)

        state = empty_state()
        state["book"]["positions"] = positions
        state["book"]["submitted_at"] = "now"
        updated, report = apply_refresh(
            state,
            prices={"VOO": 650.0},
            broker={"VGT": {"shares": 10, "market": 5500, "cost": 5000}},
        )
        self.assertTrue(report["live"])
        self.assertEqual(updated["book"]["positions"]["VOO"]["cost"], 48000.0)
        self.assertEqual(updated["book"]["positions"]["VOO"]["current"], 52000.0)
        self.assertEqual(updated["book"]["positions"]["VGT"]["cost"], 5000.0)
        self.assertEqual(updated["book"]["positions"]["VGT"]["shares"], 10.0)

    def test_actual_mix_drives_rates(self):
        holdings = {t: 0.0 for t in weights("default")}
        holdings["SMH"] = 20000
        holdings["VGSH"] = 80000
        view = book_view(holdings, monthly_add=0, compare_to="default", submitted_at="now")
        self.assertTrue(view["present"])
        self.assertEqual(view["total"], 100000.0)
        self.assertAlmostEqual(view["weights"]["SMH"], 0.20, places=10)
        self.assertAlmostEqual(view["rates"]["10y"]["stress"], 0.014, places=6)
        default_base = 0.058
        self.assertLess(view["rates"]["10y"]["base"], default_base)
        self.assertEqual(view["rows"][2]["ticker"], "SMH")
        self.assertEqual(view["rows"][2]["status"], "high")
        self.assertIsNotNone(view["projections"])
        self.assertGreater(view["projections"]["10y"]["base"]["nominal"], 0)
