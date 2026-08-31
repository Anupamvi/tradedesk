from pathlib import Path
from typing import Dict, List, Optional, Tuple

CODE_DIR = Path(__file__).resolve().parent.parent
TRADEDESK_ENV = Path("/Users/anuppamvi/tradedesk/.env")
OUT_DIR = CODE_DIR / "out" / "groat"
UNIVERSE_PATH = CODE_DIR / "configs" / "universe.txt"
BOOK_PATH = CODE_DIR / "configs" / "book.json"
MACRO_PATH = CODE_DIR / "configs" / "macro_calendar.json"
PIT_TZ = "America/New_York"

ORATS_BASE = "https://api.orats.io/datav2"
ORATS_MONTHLY_CAP = 20000
ORATS_MAX_PER_MIN = 100
ORATS_TICKER_BATCH = 10
PROBE_TICKER = "SPY"

ACCOUNT_DOLLARS = 50000.0
RISK_PCT = 0.01
RISK_PCT_MIN = 0.005
ATR_N = 14
STOP_ATR_MULT = 2.0
HOLD_SESSIONS = 15
DTE_MIN = 21
DTE_MAX = 75
DTE_LONG_PREF = (35, 60)
DTE_CREDIT_PREF = (30, 45)
QUOTE_WIDTH_ABS = 0.20
QUOTE_WIDTH_PCT_OF_MID = 0.08
MIN_OI = 50
MIN_OI_SHORT = 100
CREDIT_PCT_MIN = 0.20
CONTRACT_MULTIPLIER = 100
MAX_FINAL = 10
TRADE_SCORE_MIN = 52
WATCH_SCORE_MIN = 38
RR_PREFER = 2.0
RR_MIN = 1.2
CHASE_ATR = 2.5
EARNINGS_HOLD_DAYS = 21
# ORATS /strikes `dte` is a min,max RANGE (not a list of target slices).
STRIKE_DTE = "21,75"
SLEEVE = "groat_swing"
EVIDENCE_MAX_ANALOGS = 12
EVIDENCE_MAX_STRIKE_HTTP = 16
EVIDENCE_MAX_EARNINGS_HTTP = 6

INDEX_TICKERS = ("SPY", "QQQ", "IWM", "DIA")
MACRO_TICKERS = ("TLT", "GLD", "SLV", "UUP")
VIX_SYMBOL = "$VIX.X"

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

EARNINGS_EXEMPT = set(INDEX_TICKERS) | set(MACRO_TICKERS) | set(SECTOR_ETFS) | {
    "IBIT",
    "XBI",
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
        "INTC",
        "ARM",
        "SNPS",
        "CDNS",
        group="semiconductors",
        etf="SMH",
    )
)
TICKER_META.update(
    _meta("PLTR", "APP", "CRM", "NOW", "ORCL", "ADBE", "SNOW", "DDOG", "NET", "SHOP", "OKTA", group="software", etf="IGV")
)
TICKER_META.update(_meta("CRWD", "PANW", group="cybersecurity", etf="CIBR"))
TICKER_META.update(_meta("ANET", "CSCO", group="networking", etf="XLK"))
TICKER_META.update(_meta("DELL", "SMCI", "HPE", group="ai_infrastructure", etf="SMH"))
TICKER_META.update(_meta("VRT", "ETN", "PWR", "VST", "CEG", "GEV", group="power", etf="GRID"))
TICKER_META.update(_meta("LMT", "RTX", "GE", "NOC", "GD", "LHX", group="defense", etf="ITA"))
TICKER_META.update(_meta("RKLB", group="space", etf="XAR"))
TICKER_META.update(_meta("JPM", "GS", "V", "MA", "BAC", "WFC", "COIN", "HOOD", group="financials", etf="XLF"))
TICKER_META.update(_meta("LLY", "UNH", "JNJ", "ABBV", "VRTX", "REGN", "AMGN", "GILD", "ISRG", group="healthcare", etf="XLV"))
TICKER_META.update(_meta("XOM", "CVX", "COP", group="energy", etf="XLE"))
TICKER_META.update(_meta("COST", "HD", "TSLA", "UBER", "BKNG", group="consumer_discretionary", etf="XLY"))
TICKER_META.update(_meta("CAT", "DE", "BA", "HON", group="industrials", etf="XLI"))


def ticker_group(ticker: str) -> str:
    meta = TICKER_META.get(str(ticker).upper())
    return meta[0] if meta else "other"


def ticker_etf(ticker: str) -> str:
    meta = TICKER_META.get(str(ticker).upper())
    return meta[1] if meta else "SPY"


def quote_width_cap(mid: Optional[float]) -> Optional[float]:
    if mid is None or mid <= 0:
        return None
    return max(QUOTE_WIDTH_ABS, QUOTE_WIDTH_PCT_OF_MID * mid)


def load_universe(path: Optional[Path] = None) -> List[str]:
    target = path or UNIVERSE_PATH
    names = []
    if target.is_file():
        for raw in target.read_text(encoding="utf-8").splitlines():
            line = raw.strip().upper()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    if not names:
        names = list(INDEX_TICKERS) + list(SECTOR_ETFS) + ["NVDA", "AAPL", "MSFT"]
    seen = set()
    out = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out
