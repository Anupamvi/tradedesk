import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_DIR = Path(__file__).resolve().parent.parent
TRADEDESK_ENV = Path("/Users/anuppamvi/tradedesk/.env")
OUT_DIR = CODE_DIR / "out" / "wheelo"
UNIVERSE_PATH = CODE_DIR / "configs" / "universe.txt"
OWN_LIST_PATH = CODE_DIR / "configs" / "own_list.txt"
DEFAULT_CFG_PATH = CODE_DIR / "configs" / "default.json"
BOOK_PATH = CODE_DIR / "var" / "book.json"
PIT_TZ = "America/New_York"

ORATS_BASE = "https://api.orats.io/datav2"
ORATS_MONTHLY_CAP = 20000
ORATS_MAX_PER_MIN = 100
ORATS_TICKER_BATCH = 10
ORATS_STRIKE_DTE = "25,40"

CORE_FIELDS = (
    "ticker,tradeDate,pxAtmIv,pxCls,mktCap,avgOptVolu20d,borrow30,iv30d,"
    "ivPctile1y,ivHvXernRatio,nextErn,daysToNextErn,wksNextErn,divYield,"
    "beta1y,correlSpy1y,cVolu,pVolu,assetType,confidence,stkPxChng1wk,"
    "stkPxChng1m,stkPxChng1y,orHv20d,orFcst20d,orIvFcst20d,sectorName"
)

CONTRACT_MULTIPLIER = 100


def load_json_config(path: Path = DEFAULT_CFG_PATH) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_own_list(path: Path = OWN_LIST_PATH) -> List[str]:
    return load_universe(path)


def load_scan_universe(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    data = cfg if cfg is not None else load_json_config()
    source = str((data.get("universe") or {}).get("source") or "own_list").lower()
    if source in ("universe", "screener"):
        return load_universe()
    names = load_own_list()
    return names if names else load_universe()


def load_universe(path: Path = UNIVERSE_PATH) -> List[str]:
    if not path.is_file():
        return []
    names = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ticker = line.split()[0].upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        names.append(ticker)
    return names
