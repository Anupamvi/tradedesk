"""Replay-backed TRADE gates. Shared by daily scan and groat replay."""

from __future__ import annotations

from typing import Optional

from groat.config import CHASE_ATR
from groat.num import to_float

# Stock replay 2025-07-03→2026-08-27: B −0.15R, C −0.12R (4% win), G −0.18R.
# Keep A ≈0, D/E +0.35R, H small-n FIRE. Post-rip E was the biased daily desk-pick slice.
TRADE_SETUPS_BLOCKED = ("B", "C", "G")
E_RIP_RET1 = 0.12


def trade_park_reason(primary: Optional[str], snap: Optional[dict] = None, setup: Optional[dict] = None) -> Optional[str]:
    code = str(primary or "")
    if code in TRADE_SETUPS_BLOCKED:
        return "setup_%s_replay_park" % code
    if code != "E":
        return None
    snap = snap or {}
    setup = setup or {}
    fire = setup.get("fire") if isinstance(setup.get("fire"), dict) else {}
    ret1 = to_float(snap.get("ret_1"))
    ext = to_float(snap.get("extension_atr"))
    if fire.get("chase"):
        return "setup_E_post_rip"
    if ext is not None and ext > CHASE_ATR:
        return "setup_E_post_rip"
    if ret1 is not None and ret1 >= E_RIP_RET1:
        return "setup_E_post_rip"
    return None
