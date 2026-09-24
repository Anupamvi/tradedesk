from pathlib import Path
from typing import Dict, List, Optional, Tuple

CODE_DIR = Path(__file__).resolve().parent.parent
TRADEDESK_ROOT = Path("/Users/anuppamvi/tradedesk")
TRADEDESK_ENV = TRADEDESK_ROOT / ".env"
OUT_DIR = CODE_DIR / "out" / "trendbook"
UNIVERSE_PATH = CODE_DIR / "var" / "universe.json"
PIT_TZ = "America/New_York"

ORATS_BASE = "https://api.orats.io/datav2"
ORATS_MONTHLY_CAP = 20000
ORATS_MAX_PER_MIN = 100
ORATS_TICKER_BATCH = 10
ORATS_HTTP_DEFAULT = 40

# Copied from xhigh, plus sector / RS / multi-month change fields for this desk.
CORE_FIELDS = (
    "ticker,tradeDate,pxAtmIv,mktCap,avgOptVolu20d,borrow30,iv30d,"
    "ivPctile1y,ivRank1y,ivHvXernRatio,orHv20d,orHvXern20d,orFcst20d,"
    "orIvFcst20d,nextErn,daysToNextErn,wksNextErn,lastErn,"
    "ernDate1,ernDate2,ernDate3,ernDate4,ernDate5,ernDate6,"
    "ernDate7,ernDate8,ernDate9,ernDate10,ernDate11,ernDate12,"
    "absAvgErnMv,impErnMv,impliedEarningsMove,slope,dlt25Iv30d,"
    "dlt75Iv30d,dlt95Iv30d,assetType,confidence,divDate,divAmt,tkOver,"
    "sector,sectorName,bestEtf,correlSpy1m,correlSpy1y,"
    "stkPxChng1m,stkPxChng6m,stkPxChng1y"
)

# Five years of daily tape so a bear year is in sample and campaigns are not
# left-censored at the 30-week MA birth.
LOOKBACK_DAYS = 1825
CHASE_ATR = 2.5
ATR_N = 14
STAGE_WEEKS = 30
RS_PCTILE_MIN = 70.0
MKT_CAP_MIN = 2000.0
OPT_VOL_MIN = 200.0
IV_PCTILE_CHEAP = 40.0
IV_HV_CHEAP = 0.90
OPTION_DTE_MIN = 45
OPTION_DTE_MAX = 90
STRIKE_DTE = "45,90"
INDEX_TICKERS = ("SPY", "QQQ", "IWM", "DIA")
MACRO_TICKERS = ("TLT", "GLD", "SLV", "UUP")

SECTOR_ETFS: Dict[str, str] = {
    "XLK": "technology",
    "XLF": "financials",
    "XLE": "energy",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "consumer_discretionary",
    "XLP": "consumer_staples",
    "XLU": "utilities",
    "XLB": "materials",
    "XLRE": "real_estate",
    "XLC": "communication",
    "SMH": "semiconductors",
    "SOXX": "semiconductors",
    "XBI": "biotech",
    "CIBR": "cybersecurity",
    "IGV": "software",
    "BOTZ": "robotics",
    "ITA": "defense",
    "GRID": "electrification",
    "URNM": "nuclear",
    "XAR": "aerospace",
}


def _meta(*tickers: str, group: str, etf: str) -> Dict[str, Tuple[str, str]]:
    return {t: (group, etf) for t in tickers}


TICKER_META: Dict[str, Tuple[str, str]] = {}
TICKER_META.update(_meta(*INDEX_TICKERS, group="index", etf="SPY"))
TICKER_META.update(_meta(*MACRO_TICKERS, group="macro", etf="SPY"))
TICKER_META.update({k: (v, k) for k, v in SECTOR_ETFS.items()})
TICKER_META.update(_meta("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", group="megacap", etf="QQQ"))
TICKER_META.update(
    _meta(
        "NVDA",
        "AVGO",
        "AMD",
        "TSM",
        "AMAT",
        "LRCX",
        "KLAC",
        "ASML",
        "MU",
        "SNDK",
        "INTC",
        "ARM",
        "SNPS",
        "CDNS",
        "MRVL",
        "QCOM",
        "TXN",
        group="semiconductors",
        etf="SMH",
    )
)
TICKER_META.update(
    _meta(
        "PLTR",
        "APP",
        "CRM",
        "NOW",
        "ORCL",
        "ADBE",
        "SNOW",
        "DDOG",
        "NET",
        "SHOP",
        "IBM",
        "INTU",
        group="software",
        etf="IGV",
    )
)
TICKER_META.update(_meta("CRWD", "PANW", group="cybersecurity", etf="CIBR"))
TICKER_META.update(_meta("ANET", "CSCO", group="networking", etf="XLK"))
TICKER_META.update(_meta("DELL", "SMCI", group="ai_infrastructure", etf="SMH"))
TICKER_META.update(_meta("VRT", "ETN", "PWR", "VST", "CEG", "GEV", group="power", etf="GRID"))
TICKER_META.update(_meta("LMT", "RTX", "GE", "NOC", "RKLB", group="defense", etf="ITA"))
TICKER_META.update(
    _meta("JPM", "GS", "V", "MA", "COIN", "HOOD", "PYPL", "BAC", "WFC", "AXP", group="financials", etf="XLF")
)
TICKER_META.update(
    _meta(
        "LLY",
        "UNH",
        "JNJ",
        "ABBV",
        "VRTX",
        "ISRG",
        "MRNA",
        "AMGN",
        "GILD",
        "TMO",
        "ABT",
        group="healthcare",
        etf="XLV",
    )
)
TICKER_META.update(_meta("XOM", "CVX", group="energy", etf="XLE"))
TICKER_META.update(
    _meta("COST", "HD", "TSLA", "UBER", "BKNG", "WMT", "NKE", "DIS", "MCD", "SBUX", group="consumer_discretionary", etf="XLY")
)
TICKER_META.update(_meta("KO", "PEP", "PG", group="consumer_staples", etf="XLP"))
TICKER_META.update(_meta("CAT", "DE", "BA", "HON", "UNP", group="industrials", etf="XLI"))


def ticker_group(ticker: str) -> str:
    meta = TICKER_META.get(str(ticker).upper())
    return meta[0] if meta else "other"


def ticker_etf(ticker: str) -> str:
    meta = TICKER_META.get(str(ticker).upper())
    return meta[1] if meta else "SPY"


def load_universe(path: Optional[Path] = None) -> List[str]:
    """Dynamic universe from this desk's memory + tape. Not a static ticker list."""
    from trendbook.dates import today_et
    from trendbook.discover import dynamic_universe

    return dynamic_universe(today_et(), live=False)
