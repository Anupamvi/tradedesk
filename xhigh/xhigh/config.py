from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
TRADEDESK_ROOT = Path("/Users/anuppamvi/tradedesk")
TRADEDESK_ENV = TRADEDESK_ROOT / ".env"
OUT_DIR = CODE_DIR / "out" / "xhigh"
GATES_PATH = CODE_DIR / "configs" / "gates.json"
PIT_TZ = "America/New_York"

ORATS_BASE = "https://api.orats.io/datav2"
ORATS_MONTHLY_CAP = 20000
ORATS_MAX_PER_MIN = 100
ORATS_TICKER_BATCH = 10
ORATS_HTTP_DEFAULT = 15
CONTRACT_MULTIPLIER = 100

CORE_FIELDS = (
    "ticker,tradeDate,pxAtmIv,mktCap,avgOptVolu20d,borrow30,iv30d,"
    "ivPctile1y,ivRank1y,ivHvXernRatio,orHv20d,orHvXern20d,orFcst20d,"
    "orIvFcst20d,nextErn,daysToNextErn,wksNextErn,lastErn,"
    "ernDate1,ernDate2,ernDate3,ernDate4,ernDate5,ernDate6,"
    "ernDate7,ernDate8,ernDate9,ernDate10,ernDate11,ernDate12,"
    "absAvgErnMv,impErnMv,impliedEarningsMove,slope,dlt25Iv30d,"
    "dlt75Iv30d,dlt95Iv30d,assetType,confidence,divDate,divAmt,tkOver"
)

MOVER_INDEXES = ("$DJI", "$COMPX", "$SPX.X")
