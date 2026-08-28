from codexswing.research.current_ideas import _core_forecasts
from codexswing.research.universe import discover_optionable_universe


def _core(ticker: str, asset_type: int = 3, volume: int = 50_000):
    return {
        "ticker": ticker,
        "assetType": asset_type,
        "pxCls": 100,
        "avgOptVolu20d": volume,
        "oi": 500_000,
        "confidence": 95,
        "stkPxChng1wk": 3,
        "stkPxChng1m": 5,
        "orFcst20d": 20,
        "iv30d": 25,
        "orIvFcst20d": 30,
    }


def test_full_universe_filter_and_rank() -> None:
    candidates, funnel = discover_optionable_universe(
        [_core("AAA"), _core("ETF", asset_type=7, volume=100_000), _core("THIN", volume=100)]
    )
    assert [item.ticker for item in candidates] == ["ETF", "AAA"]
    assert funnel["thin_option_volume"] == 1
    assert candidates[0].source == "ORATS_CORES_FULL_UNIVERSE"


def test_realized_and_implied_forecasts_are_not_conflated() -> None:
    values = _core_forecasts(_core("AAA"))
    assert values["realized_vol_forecast_20d_pct"] == 20
    assert values["implied_vol_forecast_20d_pct"] == 30
    assert values["semantic_guard"]["orFcst20d"].startswith("future underlying")

