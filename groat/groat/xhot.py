"""X-HOT lane: conversation heat first, then dip/spike on the tape. X cannot print a trade alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from groat.config import CODE_DIR
from groat.num import to_float


def hot_path(asof: str) -> Path:
    return CODE_DIR / "var" / "xhot" / asof / "hot.json"


def load_hot(asof: str) -> Dict[str, dict]:
    path = hot_path(asof)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = payload.get("names") if isinstance(payload, dict) else payload
    out = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        out[ticker] = {
            "ticker": ticker,
            "heat": str(row.get("heat") or "hot"),
            "bias": str(row.get("bias") or "unknown"),
            "posts_24h": row.get("posts_24h"),
            "narrative": str(row.get("narrative") or ""),
            "tag": str(row.get("tag") or "DATA UNAVAILABLE"),
            "source": str(row.get("source") or "xhot"),
        }
    return out


def write_hot(asof: str, names: List[dict], source: str = "x_keyword_search") -> Path:
    path = hot_path(asof)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"asof": asof, "source": source, "names": names}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def classify_xhot(hot: dict, row: dict) -> dict:
    """Tape confirms X heat. `move` is dipped / will_rise / will_dip / noise — not a forecast."""
    ret1 = to_float(row.get("ret_1"))
    rvol = to_float(row.get("rvol"))
    ext = to_float(row.get("extension_atr"))
    bias = str(hot.get("bias") or "unknown")
    kind = "heat_only"
    move = "noise"
    play = "narrative only — wait for a dip into 20 EMA/AVWAP or a real volume spike"
    playable = False
    chase = False
    dumped = ret1 is not None and rvol is not None and ret1 <= -0.025 and rvol >= 1.2
    spiked = ret1 is not None and rvol is not None and ret1 >= 0.03 and rvol >= 1.5
    red = ret1 is not None and ret1 < 0
    if dumped:
        kind = "dip"
        playable = True
        if bias == "bearish":
            move = "will_dip"
            play = "X-hot dump — put debit / short only if setup G also prints; else wait for a failed bounce"
        else:
            move = "dipped"
            play = "X-hot pullback — stock or call debit if 20 EMA/AVWAP still holds"
    elif spiked:
        kind = "spike"
        chase = bool(ext is not None and ext > 2.5) or (ret1 is not None and ret1 >= 0.12)
        if chase:
            move = "will_dip"
            play = "X-hot spike already extended — do not chase; the swing is the pullback, not the rip"
            playable = False
        elif bias == "bearish":
            move = "noise"
            play = "X-hot but tape ripping against a bearish narrative — do not short the strength"
        else:
            move = "will_rise"
            playable = True
            play = "X-hot continuation — defined-risk call debit or stock, 1 lot if Crowded"
    elif red and bias == "bearish":
        kind = "soft_dip"
        move = "will_dip"
        play = "X-hot, red tape + bearish narrative — wait for a volume dump or failed high before puts"
    elif red:
        kind = "soft_dip"
        move = "dipped"
        play = "X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying"
    elif bias == "bearish":
        move = "will_dip"
        play = "X says down — wait for a volume dump or failed high; heat alone is not a short"
    elif bias == "bullish":
        move = "will_rise"
        play = "X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter"
    return {
        "kind": kind,
        "move": move,
        "play": play,
        "playable": playable,
        "chase": chase,
        "bias": bias,
        "narrative": hot.get("narrative") or "",
        "tag": hot.get("tag") or "DATA UNAVAILABLE",
        "posts_24h": hot.get("posts_24h"),
        "heat": hot.get("heat") or "hot",
        "source": hot.get("source"),
    }
