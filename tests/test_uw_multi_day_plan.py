import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "uw_multi_day_plan.py"
SPEC = importlib.util.spec_from_file_location("uw_multi_day_plan", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {SCRIPT_PATH}")
uw = importlib.util.module_from_spec(SPEC)
sys.modules["uw_multi_day_plan"] = uw
SPEC.loader.exec_module(uw)


def _df_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_day_pack_bytes(pack_day: str, ticker: str = "AAA", missing: str | None = None) -> bytes:
    daily = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "spot": np.nan,
                "net": 1.0,
                "abs_total": 2.0,
                "daily_sign": 1,
                "eod_ratio": 0.5,
                "persistence_5d_score": 30.0,
                "persistence_5d_label": "Noise",
                "persistence_5d_dominant_sign": 1,
                "persistence_5d_active_days": 3,
                "persistence_5d_sessions": 5,
                "persistence_10d_score": 40.0,
                "persistence_10d_label": "Noise",
                "persistence_10d_dominant_sign": 1,
                "persistence_10d_active_days": 6,
                "persistence_10d_sessions": 10,
                "oi_confirmation_score": 70.0,
                "oi_data_coverage": 1.0,
                "dp_support_1": 95.0,
                "dp_support_2": 96.0,
                "dp_resistance_1": 105.0,
                "dp_resistance_2": 106.0,
                "oi_magnet_tag": "none",
                "max_oi_strike": 100.0,
                "max_oi_concentration": 0.2,
                "max_oi_dist_pct": 0.01,
                "bias_hint": "bullish",
            }
        ]
    )
    oi = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "contract_signature": f"{ticker}-2026-03-20-100-call",
                "prev_abs": 1000.0,
                "prev_contracts": 100.0,
                "oi_prev": 1000.0,
                "oi_cur": 1100.0,
                "oi_delta": 100.0,
                "carryover_ratio": 1.0,
            }
        ]
    )
    dp = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "spot": 100.0,
                "dp_support_1": 95.0,
                "dp_support_2": 96.0,
                "dp_resistance_1": 105.0,
                "dp_resistance_2": 106.0,
            }
        ]
    )
    screener = pd.DataFrame(
        [
            {
                "trade_date": pack_day,
                "ticker": ticker,
                "call_volume": 1000,
                "put_volume": 800,
                "call_premium": 200000.0,
                "put_premium": 100000.0,
                "total_open_interest": 500000.0,
                "sector": "Tech",
                "source_file": "x.csv",
            }
        ]
    )

    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        entries = {
            "daily_features.csv": _df_csv(daily),
            "oi_carryover_signatures.csv": _df_csv(oi),
            "dp_anchors.csv": _df_csv(dp),
            "stock_screener.csv": _df_csv(screener),
        }
        for name, payload in entries.items():
            if missing and name == missing:
                continue
            zf.writestr(name, payload)
    return blob.getvalue()


class TestUwMultiDayPlan(unittest.TestCase):
    def test_discover_day_packs_nested(self) -> None:
        p1 = make_day_pack_bytes("2026-02-04", ticker="AAA")
        p2 = make_day_pack_bytes("2026-02-05", ticker="BBB")
        with tempfile.TemporaryDirectory() as td:
            outer_path = Path(td) / "weekly.zip"
            with zipfile.ZipFile(outer_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("weekly/chatgpt_pack_2026-02-04.zip", p1)
                zf.writestr("weekly/chatgpt_pack_2026-02-05.zip", p2)
                zf.writestr("dup/chatgpt_pack_2026-02-05.zip", p2)
            out = uw.discover_day_packs([outer_path])
            self.assertEqual(len(out), 2)
            self.assertIn(date(2026, 2, 4), out)
            self.assertIn(date(2026, 2, 5), out)

    def test_load_day_pack_missing_required_file_fails(self) -> None:
        broken = make_day_pack_bytes("2026-02-05", missing="stock_screener.csv")
        with self.assertRaises(SystemExit):
            uw.load_day_pack(date(2026, 2, 5), "broken", broken)

    def test_normalize_uses_dp_spot_and_flow_ratio(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "spot": np.nan,
                    "persistence_5d_score": 20.0,
                    "persistence_5d_dominant_sign": 1,
                    "persistence_10d_score": 30.0,
                    "persistence_10d_dominant_sign": 1,
                    "oi_confirmation_score": 60.0,
                    "oi_data_coverage": 1.0,
                    "dp_support_1": 90.0,
                    "dp_support_2": 91.0,
                    "dp_resistance_1": 110.0,
                    "dp_resistance_2": 111.0,
                    "oi_magnet_tag": "none",
                    "max_oi_strike": 100.0,
                    "max_oi_concentration": 0.2,
                    "max_oi_dist_pct": 0.01,
                }
            ]
        )
        dp = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "spot": 123.0,
                    "dp_support_1": 90.0,
                    "dp_support_2": 91.0,
                    "dp_resistance_1": 110.0,
                    "dp_resistance_2": 111.0,
                }
            ]
        )
        oi = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "contract_signature": "AAA-2026-03-20-120-call",
                    "prev_abs": 10.0,
                    "prev_contracts": 10.0,
                    "oi_prev": 10.0,
                    "oi_cur": 15.0,
                    "oi_delta": 5.0,
                    "carryover_ratio": 0.5,
                }
            ]
        )
        screener = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "call_volume": 1000,
                    "put_volume": 1000,
                    "call_premium": 300.0,
                    "put_premium": 100.0,
                    "total_open_interest": 1000.0,
                }
            ]
        )
        daily_n, oi_n, _ = uw.normalize_data(daily, dp, oi, screener)
        self.assertEqual(float(daily_n.iloc[0]["spot"]), 123.0)
        self.assertAlmostEqual(float(daily_n.iloc[0]["flow_ratio"]), 0.5, places=6)
        self.assertEqual(str(oi_n.iloc[0]["parsed_ticker"]), "AAA")

    def test_parse_latest_oi_signatures_filters_dte(self) -> None:
        oi = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "contract_signature": "AAA-2026-02-20-100-call",
                    "prev_abs": 100.0,
                    "prev_contracts": 10.0,
                    "oi_prev": 10.0,
                    "oi_cur": 11.0,
                    "oi_delta": 1.0,
                    "carryover_ratio": 0.1,
                },
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "contract_signature": "AAA-2026-05-20-100-call",
                    "prev_abs": 100.0,
                    "prev_contracts": 10.0,
                    "oi_prev": 10.0,
                    "oi_cur": 11.0,
                    "oi_delta": 1.0,
                    "carryover_ratio": 0.1,
                },
            ]
        )
        daily = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "ticker": "AAA",
                    "persistence_5d_score": 1.0,
                    "persistence_5d_dominant_sign": 1,
                    "persistence_10d_score": 1.0,
                    "persistence_10d_dominant_sign": 1,
                    "oi_confirmation_score": 50.0,
                    "oi_data_coverage": 1.0,
                    "dp_support_1": 90.0,
                    "dp_support_2": 91.0,
                    "dp_resistance_1": 110.0,
                    "dp_resistance_2": 111.0,
                    "oi_magnet_tag": "none",
                    "max_oi_strike": 100.0,
                    "max_oi_concentration": 0.2,
                    "max_oi_dist_pct": 0.01,
                }
            ]
        )
        dp = pd.DataFrame([{"trade_date": pd.Timestamp("2026-02-05"), "ticker": "AAA", "spot": 100.0}])
        scr = pd.DataFrame([{"trade_date": pd.Timestamp("2026-02-05"), "ticker": "AAA", "call_premium": 0.0, "put_premium": 0.0, "call_volume": 0.0, "put_volume": 0.0, "total_open_interest": 0.0}])
        _, oi_n, _ = uw.normalize_data(daily, dp, oi, scr)
        parsed = uw.parse_latest_oi_signatures(oi_n, pd.Timestamp("2026-02-05"))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(str(parsed.iloc[0]["right"]), "call")

    def test_persistence_trend_side_aware(self) -> None:
        self.assertEqual(uw.persistence_trend(30.0, 2.0), "improving")
        self.assertEqual(uw.persistence_trend(30.0, -2.0), "decaying")
        self.assertEqual(uw.persistence_trend(-30.0, -2.0), "improving")
        self.assertEqual(uw.persistence_trend(-30.0, 2.0), "decaying")

    def test_width_tiers(self) -> None:
        self.assertEqual(uw.width_by_spot(10.0), 5.0)
        self.assertEqual(uw.width_by_spot(30.0), 10.0)
        self.assertEqual(uw.width_by_spot(100.0), 15.0)
        self.assertEqual(uw.width_by_spot(200.0), 20.0)

    def test_choose_short_strike_bull_respects_dp_constraint(self) -> None:
        row = pd.Series(
            {
                "ticker": "AAA",
                "spot": 100.0,
                "dp_support_1": 90.0,
                "dp_support_2": 91.0,
                "dp_resistance_1": 110.0,
                "dp_resistance_2": 111.0,
                "oi_magnet_tag": "none",
                "max_oi_strike": 100.0,
            }
        )
        oi_latest = pd.DataFrame(
            [
                {"ticker": "AAA", "expiry": pd.Timestamp("2026-03-06"), "strike": 95.0, "right": "put", "dte": 29, "prev_abs": 100.0},
                {"ticker": "AAA", "expiry": pd.Timestamp("2026-03-06"), "strike": 85.0, "right": "put", "dte": 29, "prev_abs": 90.0},
            ]
        )
        short, long, width, _, blocked = uw.choose_short_strike(row, oi_latest, pd.Timestamp("2026-03-06"), "bull")
        self.assertLess(short, 100.0)
        self.assertLessEqual(short, 90.0 * 0.995 + 1e-9)
        self.assertEqual(width, 15.0)
        self.assertEqual(long, short - width)
        self.assertFalse(blocked)

    def test_bearish_trade_capped_watch_or_avoid(self) -> None:
        latest = pd.Timestamp("2026-02-05")
        shortlist = pd.DataFrame(
            [
                {
                    "ticker": "BBB",
                    "bias": "bear",
                    "shortlist_score": 100.0,
                    "spot": 100.0,
                    "dp_support_1": 90.0,
                    "dp_support_2": 91.0,
                    "dp_resistance_1": 110.0,
                    "dp_resistance_2": 112.0,
                    "oi_magnet_tag": "none",
                    "max_oi_strike": np.nan,
                    "signed_p10": -40.0,
                    "signed_p5": -35.0,
                    "flow_ratio": -0.5,
                    "oi_conf": 80.0,
                    "persistence_slope": -2.0,
                    "oi_trend": "improving",
                    "liq_score": 10.0,
                }
            ]
        )
        daily_latest_sorted = shortlist.copy()
        oi_latest = pd.DataFrame(
            [
                {
                    "ticker": "BBB",
                    "expiry": pd.Timestamp("2026-03-06"),
                    "strike": 120.0,
                    "right": "call",
                    "dte": 29,
                    "prev_abs": 1000.0,
                    "carryover_ratio": 0.5,
                }
            ]
        )
        trades = uw.build_trade_rows(shortlist, daily_latest_sorted, oi_latest, latest)
        self.assertGreaterEqual(len(trades), 1)
        self.assertIn(trades[0].optimal, {"Watch", "Avoid"})

    def test_render_report_trade_header_contract(self) -> None:
        trade = uw.TradeRow(
            index=1,
            ticker="AAA",
            action="Sell Put Spread",
            strategy_type="Bull Put Credit",
            strike_setup="Sell 90P / Buy 80P",
            expiry="2026-03-06",
            dte=29,
            net_credit="$2.50",
            max_profit="$250",
            max_loss="$750",
            breakeven="87.50",
            conviction_pct=70,
            confidence="Medium ⚖️",
            optimal="Watch",
            thesis="x",
            key_risks="y",
            dp_anchor_invalidation="z",
        )
        report = uw.render_report(
            latest_date=pd.Timestamp("2026-02-05"),
            prior_date=pd.Timestamp("2026-02-04"),
            pack_sources=["a", "b"],
            bull_ranked=pd.DataFrame(),
            bear_ranked=pd.DataFrame(),
            shortlist=pd.DataFrame(),
            trades=[trade],
            material_changes=["none"],
        )
        self.assertIn("| # | Ticker | Action | Strategy Type | Strike Setup | Expiry | DTE | Net Credit | Max Profit | Max Loss | Breakeven | Conviction % | Confidence | Optimal? | Thesis | Key Risks | DP Anchor / Invalidation |", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
