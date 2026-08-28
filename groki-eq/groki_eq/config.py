from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
TRADEDESK_ENV = Path("/Users/anuppamvi/tradedesk/.env")
OUT_DIR = CODE_DIR / "out" / "groki-eq"
UNIVERSE_PATH = CODE_DIR / "configs" / "universe.txt"
UNIVERSE = ("SPY", "QQQ", "IWM")
TIE_ORDER = {"SPY": 0, "QQQ": 1, "IWM": 2}
PIT_TZ = "America/New_York"
ORATS_BASE = "https://api.orats.io/datav2"
ORATS_MONTHLY_CAP = 20000
ORATS_MAX_PER_MIN = 100
PROBE_TICKER = "SPY"
PROBE_PATH = "/hist/dailies"

HIGH_LOOKBACK = 20
ATR_N = 14
STOP_ATR_MULT = 2.0
TIME_STOP_SESSIONS = 15
ACCOUNT_DOLLARS = 50000.0
RISK_PCT = 0.01
MAX_NEW_PER_WEEK = 1
MAX_OPEN = 2
EXECUTE_CAP = 1
SLEEVE = "breakout_eq"
TRAIN_TEST_SPLIT = "2023-01-03"
REPLAY_DEFAULT_START = "2018-01-02"
SLEEVE_PROMOTE_PF = 1.2
MONTHLY_PROFIT_TARGET = 10000.0
