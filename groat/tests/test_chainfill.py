import unittest

from groat.book import book_index, parse_occ, same_ticket, underlying_symbol
from groat.chainfill import fill_px, flatten_chain, overlay_ticker
from groat.structure import debit_spread, quote_ok


def _orats(strike=190.0, bid=4.0, ask=8.0, oi=400, expiry="2026-10-16", dte=46, delta=0.40):
    return {
        "strike": strike,
        "dte": dte,
        "expirDate": expiry,
        "stockPrice": 185.0,
        "delta": delta,
        "callBidPrice": bid,
        "callAskPrice": ask,
        "callOpenInterest": oi,
        "putBidPrice": 2.0,
        "putAskPrice": 2.2,
        "putOpenInterest": oi,
    }


class TestFillPx(unittest.TestCase):
    def test_quote_beats_last(self):
        bid, ask, src = fill_px({"bid": 1.1, "ask": 1.2, "last": 9.0, "mark": 1.15})
        self.assertEqual((bid, ask, src), (1.1, 1.2, "schwab_quote"))

    def test_last_when_bid_ask_dead(self):
        bid, ask, src = fill_px({"bid": 0, "ask": 0, "last": 2.73, "mark": None})
        self.assertEqual(src, "schwab_last")
        self.assertLess(bid, 2.73)
        self.assertGreater(ask, 2.73)

    def test_close_session_does_not_pad_mark(self):
        bid, ask, src = fill_px({"bid": 0, "ask": 0, "last": 2.73, "mark": 2.70}, allow_stale_pad=False)
        self.assertEqual(src, "none")
        self.assertFalse(bid and bid > 0 and ask and ask > 0)

    def test_schwab_closed_market_width_allows_session_quotes(self):
        self.assertTrue(quote_ok(3.9, 4.45, 356, quote_source="schwab_quote"))
        self.assertFalse(quote_ok(3.9, 4.45, 356))
        self.assertFalse(quote_ok(3.9, 4.45, 356, quote_source="schwab_mark"))
        self.assertFalse(quote_ok(3.9, 4.45, 356, quote_source="schwab_last"))


class TestOverlay(unittest.TestCase):
    def test_schwab_tight_quotes_replace_wide_orats(self):
        payload = {
            "underlying": {"last": 185.0},
            "callExpDateMap": {
                "2026-10-16:46": {
                    "190.0": [
                        {
                            "bid": 4.80,
                            "ask": 5.00,
                            "mark": 4.90,
                            "last": 4.95,
                            "openInterest": 600,
                            "delta": 0.40,
                            "strikePrice": 190.0,
                            "expirationDate": "2026-10-16",
                        }
                    ],
                    "195.0": [
                        {
                            "bid": 3.30,
                            "ask": 3.50,
                            "mark": 3.40,
                            "last": 3.40,
                            "openInterest": 400,
                            "delta": 0.30,
                            "strikePrice": 195.0,
                            "expirationDate": "2026-10-16",
                        }
                    ],
                }
            },
            "putExpDateMap": {},
        }
        flat = flatten_chain(payload)
        wide = [
            _orats(190.0, bid=4.0, ask=8.0, delta=0.40),
            _orats(195.0, bid=2.0, ask=6.0, delta=0.30, oi=400),
        ]
        self.assertFalse(quote_ok(4.0, 8.0, 400))
        over = overlay_ticker("2026-08-31", wide, flat)
        by = {row["strike"]: row for row in over}
        self.assertEqual(by[190.0]["callBidPrice"], 4.80)
        self.assertEqual(by[190.0]["callAskPrice"], 5.00)
        self.assertEqual(by[190.0]["quoteSource"], "schwab_quote")
        earn = {"usable": True, "source": "exempt", "overlaps_hold": False, "date": None}
        debit = debit_spread(over, "bullish", earn)
        self.assertIsNotNone(debit)
        self.assertTrue(debit["ok"])
        self.assertGreater(debit["target_debit"], 0)


class TestBook(unittest.TestCase):
    def test_underlying_from_occ(self):
        self.assertEqual(underlying_symbol("CVX   260925C00210000"), "CVX")
        self.assertEqual(underlying_symbol("PLTR"), "PLTR")
        occ = parse_occ("PLTR  271217P00110000")
        self.assertEqual(occ["underlying"], "PLTR")
        self.assertEqual(occ["right"], "put")
        self.assertEqual(occ["expiry"], "2027-12-17")
        self.assertEqual(occ["strike"], 110.0)

    def test_same_ticket_detects_roll(self):
        book = {"expiry": "2026-09-25", "long_strike": 210.0, "short_strike": 220.0, "structure": "210/220c"}
        picked = {"expiry": "2026-10-16", "long_strike": 210.0, "short_strike": 220.0, "legs": "BUY 210 call"}
        self.assertFalse(same_ticket(book, picked))
        picked["expiry"] = "2026-09-25"
        self.assertTrue(same_ticket(book, picked))

    def test_book_index_keeps_held_visible(self):
        import json
        import tempfile
        from pathlib import Path

        from groat.book import book_index as idx_fn

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.json"
            path.write_text(
                json.dumps({"positions": [{"ticker": "CVX", "structure": "210/220c", "entry": 2.73}]}),
                encoding="utf-8",
            )
            idx = idx_fn(path)
        self.assertIn("CVX", idx)
        self.assertTrue(idx["CVX"]["in_book"])
        self.assertIn("210/220c", idx["CVX"]["structure"])


if __name__ == "__main__":
    unittest.main()
