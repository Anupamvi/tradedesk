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
        view = book_view(empty_state()["book"]["holdings"], submitted_at=None)
        self.assertFalse(view["present"])
        self.assertIsNone(view["projections"])

    def test_roundtrip_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.json"
            state = empty_state()
            state["planner"]["amount"] = 250000
            state["planner"]["weekly"] = 500
            state["book"]["holdings"]["VOO"] = 120000
            state["book"]["holdings"]["VXUS"] = 50000
            state["book"]["monthly_add"] = 1000
            state["book"]["submitted_at"] = "2026-09-04T12:00:00Z"
            save_state(state, path)
            loaded = load_state(path)
            self.assertEqual(loaded["planner"]["amount"], 250000.0)
            self.assertEqual(loaded["book"]["holdings"]["VOO"], 120000.0)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["book"]["compare_to"], "default")

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
