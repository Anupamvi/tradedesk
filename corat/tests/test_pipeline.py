import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from corat.config import UniverseItem, discover_universe
from corat.models import SourceTrace
from corat.orats import FetchBundle
from corat.pipeline import _economics_rank, _enrichment_plan, run_scan
from corat.report import render_report
from tests.helpers import trend_bars


def option_rows():
    rows = []
    for strike, delta, bid, ask in [(110, 0.70, 12, 12.2), (120, 0.55, 7, 7.2), (130, 0.28, 2.5, 2.7)]:
        rows.append({
            "ticker":"AAA","tradeDate":"2026-08-27","expirDate":"2026-10-02","dte":37,"strike":strike,
            "stockPrice":120,"callBidPrice":bid,"callAskPrice":ask,"callValue":(bid+ask)/2,
            "putBidPrice":bid,"putAskPrice":ask,"putValue":(bid+ask)/2,"callOpenInterest":1000,
            "putOpenInterest":1000,"callVolume":200,"putVolume":200,"delta":delta,"gamma":0.01,
            "theta":-0.03,"vega":0.08,"smvVol":0.25,"updatedAt":"2026-08-27T20:00:00Z",
        })
    return rows


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.run_requests = 0
    def _trace(self, endpoint, rows):
        return SourceTrace("ORATS", endpoint, "FIXTURE", "2026-08-27T21:00:00Z", "2026-08-27", len(rows), "", "", {})
    def fetch_dailies(self, tickers, start_date, end_date, batch_size=10):
        rows=[]
        for ticker in tickers:
            for bar in trend_bars(ticker, count=300, breakout=(ticker=="AAA")):
                rows.append({"ticker":ticker,"tradeDate":bar.date,"open":bar.open,"hiPx":bar.high,"loPx":bar.low,"clsPx":bar.close,"stockVolume":bar.volume,"updatedAt":bar.updated_at})
        return FetchBundle(rows, [self._trace("/hist/dailies", rows)], [])
    def fetch_asof(self, family, tickers, as_of, batch_size=10, allow_current=True):
        if family == "cores":
            rows=[{"ticker":ticker,"tradeDate":as_of,"pxCls":180 if ticker=="AAA" else 150,"pxAtmIv":180 if ticker=="AAA" else 150,"priorCls":149,"stkVolu":2_000_000,"orIvXern20d":25,"orFcst20d":23,"orHv20d":22,"clsHv60d":24,"nextErn":"2026-12-01","lastErn":"2026-07-30","cVolu":1000,"pVolu":900,"cOi":10000,"pOi":9000,"updatedAt":"2026-08-27T20:00:00Z"} for ticker in tickers]
        elif family == "ivrank":
            rows=[{"ticker":ticker,"tradeDate":as_of,"iv":25,"ivRank1y":35,"ivPct1y":40,"updatedAt":"2026-08-27T20:00:00Z"} for ticker in tickers]
        else:
            rows=[{"ticker":ticker,"tradeDate":as_of,"stockPrice":180 if ticker=="AAA" else 150,"confidence":0.9,"iv20d":0.25,"updatedAt":"2026-08-27T20:00:00Z"} for ticker in tickers]
        return FetchBundle(rows, [self._trace("/"+family, rows)], [])
    def fetch_chain(self, ticker, as_of, min_dte, max_dte):
        rows=option_rows()
        return FetchBundle(rows, [self._trace("/strikes", rows)], [])
    def usage(self):
        return {"month":"2026-08","used":0,"left":20000,"run_requests":0,"run_left":100}


class PipelineTest(unittest.TestCase):
    def test_enrichment_is_deterministic_and_prioritizes_sourced_context(self):
        enriched, options = _enrichment_plan(
            ["AAA", "BBB", "CCC", "DDD"],
            {"DDD", "CCC"},
            enrichment_limit=2,
            option_limit=3,
        )
        self.assertEqual(enriched, {"AAA", "BBB", "CCC", "DDD"})
        self.assertEqual(options, ["CCC", "DDD", "AAA"])

    def test_every_triggered_name_gets_option_coverage_before_any_quota(self):
        enriched, options = _enrichment_plan(
            ["AAA", "BBB", "CCC", "DDD"],
            {"AAA"},
            enrichment_limit=1,
            option_limit=1,
            triggered_tickers={"BBB", "DDD"},
        )
        self.assertEqual(options, ["BBB", "DDD"])
        self.assertTrue({"AAA", "BBB", "DDD"}.issubset(enriched))

    def test_dynamic_discovery_combines_market_cap_and_trading_liquidity(self):
        configured = [
            UniverseItem("SPY", "SPY", "Market", "Market", "benchmark", "SPY"),
            UniverseItem("AAA", "AAA", "Technology", "Fixture", "equity", "XLK"),
        ]
        config = {
            "discovery": {
                "maximum_equities": 3,
                "minimum_market_cap_thousands": 2_000_000,
                "market_cap_slots": 1,
                "option_volume_slots": 1,
                "stock_volume_slots": 1,
            },
            "liquidity": {"minimum_stock_price": 5, "minimum_average_dollar_volume": 25_000_000},
        }
        rows = [
            {"ticker": "AAA", "assetType": 3, "mktCap": 3_000_000, "pxCls": 100, "stkVolu": 1_000_000, "avgOptVolu20d": 10, "sectorName": "Technology", "bestEtf": "XLK"},
            {"ticker": "BIG", "assetType": 3, "mktCap": 50_000_000, "pxCls": 100, "stkVolu": 1_000_000, "avgOptVolu20d": 20, "sectorName": "Industrials", "bestEtf": "XLI"},
            {"ticker": "OPT", "assetType": 0, "mktCap": 4_000_000, "pxCls": 50, "stkVolu": 1_000_000, "avgOptVolu20d": 50_000, "sectorName": "Financials", "bestEtf": "XLF"},
            {"ticker": "VOL", "assetType": 0, "mktCap": 4_000_000, "pxCls": 20, "stkVolu": 10_000_000, "avgOptVolu20d": 5, "sectorName": "Energy", "bestEtf": "XLE"},
            {"ticker": "_SYNTH", "assetType": 3, "mktCap": 99_000_000, "pxCls": 100, "stkVolu": 10_000_000, "avgOptVolu20d": 99_000},
        ]
        selected, audit = discover_universe(config, rows, configured)
        names = {item.ticker for item in selected}
        self.assertTrue({"SPY", "AAA", "BIG", "OPT"}.issubset(names))
        self.assertNotIn("_SYNTH", names)
        self.assertEqual(audit["selected_equities"], 3)

    def test_dynamic_discovery_fills_requested_breadth_after_sleeve_overlap(self):
        configured = [UniverseItem("SPY", "SPY", "Market", "Market", "benchmark", "SPY")]
        config = {
            "discovery": {
                "maximum_equities": 4,
                "minimum_market_cap_thousands": 2_000_000,
                "market_cap_slots": 1,
                "option_volume_slots": 1,
                "stock_volume_slots": 1,
            },
            "liquidity": {"minimum_stock_price": 5},
        }
        rows = [
            {
                "ticker": ticker,
                "assetType": 3,
                "mktCap": market_cap,
                "pxCls": 100,
                "stkVolu": volume,
                "avgOptVolu20d": option_volume,
            }
            for ticker, market_cap, volume, option_volume in (
                ("AAA", 50_000_000, 5_000_000, 50_000),
                ("BBB", 40_000_000, 4_000_000, 40_000),
                ("CCC", 30_000_000, 3_000_000, 30_000),
                ("DDD", 20_000_000, 2_000_000, 20_000),
                ("EEE", 10_000_000, 1_000_000, 10_000),
            )
        ]
        selected, audit = discover_universe(config, rows, configured)
        equities = [item.ticker for item in selected if item.kind == "equity"]
        self.assertEqual(len(equities), 4)
        self.assertEqual(audit["selected_equities"], 4)

    def test_report_lead_uses_total_targets_not_only_displayed_subset(self):
        displayed = {
            "ticker": "AAA",
            "kind": "equity",
            "status": "TARGET TRADE",
            "vehicle": "STOCK",
            "setup": {"direction": "BULLISH", "name": "TEST"},
            "stock_plan": {},
            "economics": {},
            "option": {},
        }
        report = render_report(
            {
                "as_of": "2026-08-27",
                "candidates": [displayed],
                "diagnostics": {
                    "target_trades": 13,
                    "option_target_trades": 6,
                    "stock_target_trades": 7,
                },
            }
        )
        self.assertIn("13 TARGET TRADES IDENTIFIED", report)
        self.assertIn("top 1 of 13 qualifying targets", report)

    def test_economics_rank_precedes_context_score(self):
        strong_economics = {
            "vehicle": "STOCK", "score": 60,
            "stock_plan": {"risk_basis_price": 100},
            "economics": {
                "expected_return_on_capital": 0.05, "expected_profit_per_share": 5,
                "expected_profit_lower_95_per_share": 1, "modeled_pop": 0.55, "model_sample_size": 30,
            },
        }
        high_score_weak_economics = {
            "vehicle": "STOCK", "score": 95,
            "stock_plan": {"risk_basis_price": 100},
            "economics": {
                "expected_return_on_capital": 0.01, "expected_profit_per_share": 1,
                "expected_profit_lower_95_per_share": -1, "modeled_pop": 0.70, "model_sample_size": 30,
            },
        }
        self.assertGreater(_economics_rank(strong_economics), _economics_rank(high_score_weak_economics))

    def test_offline_style_pipeline_writes_fail_closed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            universe=root/"universe.csv"
            universe.write_text(
                "ticker,name,sector,theme,kind,sector_etf\n"
                "SPY,SPY,Market,Market,benchmark,SPY\n"
                "QQQ,QQQ,Market,Market,benchmark,QQQ\n"
                "XLK,XLK,Technology,Technology,sector_etf,XLK\n"
                "AAA,AAA Corp,Technology,Fixture,equity,XLK\n",
                encoding="utf-8",
            )
            context=root/"context.json"
            context.write_text(json.dumps({
                "schema_version":"corat.context.v1","as_of":"2026-08-27","market_events":[],
                "tickers":{"AAA":{"catalysts":[{"classification":"FACT","credibility":"PRIMARY","source":"AAA IR","source_url":"https://example.com/fact","published_at":"2026-08-27","title":"Fixture catalyst","direction":"BULLISH"}],"x_intelligence":[],"events":[],"options_flow":[]}},
            }),encoding="utf-8")
            cfg={
                "_config_path":"","universe_file":str(universe),"output_root":str(root/"out"),"cache_root":str(root/"cache"),"state_root":str(root/"state"),
                "lookback_calendar_days":1900,"max_final_ideas":5,"max_enriched_candidates":5,"max_option_candidates":2,
                "orats":{"base_url":"https://api.orats.io/datav2","request_timeout_seconds":1,"max_requests_per_run":10,"monthly_request_cap":20000,"batch_size":10,"requests_per_minute":1000,"min_dte":21,"max_dte":75},
                "liquidity":{"minimum_stock_price":5,"minimum_average_dollar_volume":1_000_000,"minimum_option_open_interest":100,"minimum_option_volume":10,"maximum_option_spread_pct":0.2},
                "risk":{"portfolio_nav":100000,"normal_risk_pct":0.01,"maximum_high_conviction_risk_pct":0.0125,"minimum_reward_risk":1.7,"preferred_reward_risk":2,"maximum_correlated_ideas":2},
                "history":{"minimum_analog_sample":2,"maximum_analog_sample":200,"signal_spacing_sessions":5,"primary_horizon_sessions":10,"forward_horizons":[1,3,5,10,20]},
                "actionability":{"minimum_score":60,"require_current_price":True,"require_catalyst_evidence":True,"require_historical_evidence":True,"require_earnings_date_for_options":True},
                "regime":{"benchmarks":["SPY","QQQ"],"macro_proxies":[],"sector_etfs":["XLK"]},
            }
            with mock.patch("corat.pipeline.OratsClient", FakeClient):
                result=run_scan(cfg,"SECRET_DO_NOT_WRITE","2026-08-27",tickers=["AAA"],context_path=context,posture="VALIDATION_SMOKE_RESEARCH_ONLY")
            self.assertGreaterEqual(result["diagnostics"]["target_trades"],0)
            self.assertEqual(
                result["diagnostics"]["target_trades"],
                result["diagnostics"]["option_target_trades"] + result["diagnostics"]["stock_target_trades"],
            )
            self.assertNotIn("Current entry trigger requires session-time repricing",result["candidates"][0]["blockers"])
            self.assertTrue(any("Schwab" in note for note in result["candidates"][0]["review_notes"]))
            manifest=json.loads(Path(result["artifacts"]["manifest"]).read_text())
            self.assertFalse(manifest["order_submission_surface"])
            for path in Path(result["artifacts"]["run_dir"]).iterdir():
                if path.is_file():
                    self.assertNotIn("SECRET_DO_NOT_WRITE",path.read_text(encoding="utf-8"))
