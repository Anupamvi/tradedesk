"""Replay-backed TRADE gates. Shared by daily scan and groat replay."""

from __future__ import annotations

from typing import Optional, Sequence

from groat.config import CHASE_ATR
from groat.num import fmt, to_float

# Stock replay 2025-07-03→2026-08-27: B −0.15R, C −0.12R (4% win), G −0.18R.
# Keep A ≈0, D/E +0.35R, H small-n FIRE. Post-rip E was the biased daily desk-pick slice.
# Replay: B −0.15R, C −0.12R, G −0.18R, H −0.22R. A ≈ 0R is scored below TRADE, not blocked.
TRADE_SETUPS_BLOCKED = ("B", "C", "G", "H")
E_RIP_RET1 = 0.12
D_RIP_RET1 = 0.03
CROWDED_DIP_RET1 = -0.025
CROWDED_DIP_RVOL = 1.2
# Same-ticker analog: park TRADE→WATCH only when the sample is large enough
# and there is no win. n=1 0-win and mixed 1W/3L stay TRADE (caution only).
ANALOG_VETO_MIN_N = 4
# n≥3 with ≥2 fast −1R stop-outs (hold ≤4 sessions) that outnumber wins and avg R≤0.
# Catches SHOP-style 1W/2L 3-day stops. n=1 and mixed longer-hold books stay TRADE.
ANALOG_FAST_MIN_N = 3
ANALOG_FAST_HOLD = 4
ANALOG_FAST_MIN_LOSSES = 2


def trade_park_reason(primary: Optional[str], snap: Optional[dict] = None, setup: Optional[dict] = None) -> Optional[str]:
    code = str(primary or "")
    if code in TRADE_SETUPS_BLOCKED:
        return "setup_%s_replay_park" % code
    if code not in ("D", "E"):
        return None
    snap = snap or {}
    setup = setup or {}
    fire = setup.get("fire") if isinstance(setup.get("fire"), dict) else {}
    ret1 = to_float(snap.get("ret_1"))
    ext = to_float(snap.get("extension_atr"))
    tag = "setup_%s_post_rip" % code
    if fire.get("chase"):
        return tag
    if ext is not None and ext > CHASE_ATR:
        return tag
    rip = E_RIP_RET1 if code == "E" else D_RIP_RET1
    if ret1 is not None and ret1 >= rip:
        return tag
    return None


def analog_fast_stop_reason(evidence: Optional[dict] = None) -> Optional[str]:
    """Park when most same-setup analogs are fast −1R stop-outs.

    n=1 0-win and 1W/4L with long holds stay TRADE. Missing analog is not a veto.
    """
    if not isinstance(evidence, dict):
        return None
    stock = evidence.get("stock") if isinstance(evidence.get("stock"), dict) else evidence
    n = int(stock.get("n") or 0)
    wins = int(stock.get("wins") or 0)
    avg_r = to_float(stock.get("avg_r"))
    fast = stock.get("fast_loss_n")
    if fast is None:
        fast = evidence.get("fast_loss_n")
    hits = evidence.get("hits")
    if fast is None and isinstance(hits, list):
        fast = sum(
            1
            for h in hits
            if isinstance(h, dict) and h.get("result") == "loss" and int(h.get("hold") or 99) <= ANALOG_FAST_HOLD
        )
    fast_n = int(fast or 0)
    if n < ANALOG_FAST_MIN_N or fast_n < ANALOG_FAST_MIN_LOSSES:
        return None
    if fast_n <= wins:
        return None
    if avg_r is not None and avg_r > 0:
        return None
    return "analog_fast_stop_veto"


def analog_park_reason(evidence: Optional[dict] = None) -> Optional[str]:
    if isinstance(evidence, dict):
        tagged = evidence.get("analog_veto")
        if tagged in ("analog_0win_veto", "analog_fast_stop_veto"):
            return tagged
    return analog_0win_reason(evidence) or analog_fast_stop_reason(evidence)


def analog_0win_reason(evidence: Optional[dict] = None) -> Optional[str]:
    """Park when this ticker+setup has n≥4 analogs, 0 target hits, and avg R<0.

    Time exits that print positive R are not wins and do not veto when avg R≥0.
    Missing analog is not a veto.
    """
    if not isinstance(evidence, dict):
        return None
    stock = evidence.get("stock") if isinstance(evidence.get("stock"), dict) else evidence
    n = int(stock.get("n") or 0)
    wins = int(stock.get("wins") or 0)
    avg_r = to_float(stock.get("avg_r"))
    if n >= ANALOG_VETO_MIN_N and wins <= 0 and avg_r is not None and avg_r < 0:
        return "analog_0win_veto"
    return None


def ticket_right(picked: Optional[dict] = None) -> Optional[str]:
    if not isinstance(picked, dict):
        return None
    inst = str(picked.get("instrument") or picked.get("legs") or "").lower()
    if "call" in inst:
        return "call"
    if "put" in inst:
        return "put"
    return None


def already_held_same_right_reason(
    picked: Optional[dict] = None,
    legs: Optional[Sequence[dict]] = None,
) -> Optional[str]:
    """Park a new OPTIONS ticket when Schwab already has the same call/put right.

    Opposite-right legs and share holdings do not park. Qty 0 is ignored.
    """
    right = ticket_right(picked)
    if right not in ("call", "put"):
        return None
    for leg in legs or []:
        if not isinstance(leg, dict):
            continue
        if str(leg.get("right") or "").lower() != right:
            continue
        qty = to_float(leg.get("quantity"))
        if qty is not None and qty == 0:
            continue
        return "already_held_%ss" % right
    return None


def park_trade(row: dict, reason: str) -> bool:
    if not reason or not isinstance(row, dict) or row.get("action") != "TRADE":
        return False
    row["action"] = "WATCH"
    reasons = list(row.get("reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
    row["reasons"] = reasons
    return True


def apply_same_group_book_park(row: dict, open_groups=None, open_tickers=None) -> Optional[str]:
    """Caveat on TRADE when another name in the same industry group is already open.

    Does not demote TRADE → WATCH. Desk pick may still take it; overlap is a caveat, not a veto.
    """
    if not isinstance(row, dict):
        return None
    ticker = str(row.get("ticker") or "").upper()
    if ticker and ticker in (open_tickers or set()):
        return None
    group = str(row.get("group") or "")
    if not group or group in ("other", "index", "macro"):
        return None
    if group not in (open_groups or set()):
        return None
    row["book_group_held"] = True
    row["book_group_note"] = (
        "CAVEAT: open book already has %s — size down or skip; not a second lot unless you want that group." % group
    )
    return "same_group_in_book"


def already_in_book_reason(row: Optional[dict] = None) -> Optional[str]:
    """Park a new OPTIONS ticket when book.json already has this underlying (not the same ticket)."""
    if not isinstance(row, dict):
        return None
    if row.get("action") != "TRADE" or row.get("choice") != "OPTIONS":
        return None
    if not row.get("in_book"):
        return None
    if row.get("same_ticket"):
        return None
    return "already_in_book"


def apply_already_in_book_park(row: dict) -> Optional[str]:
    reason = already_in_book_reason(row)
    if not reason:
        return None
    park_trade(row, reason)
    return reason


def crowded_park_reason(row: Optional[dict] = None) -> Optional[str]:
    """Crowded X is WATCH unless the tape already dumped."""
    if not isinstance(row, dict):
        return None
    if str(row.get("x") or "") != "Crowded":
        return None
    ret1 = to_float(row.get("ret_1"))
    rvol = to_float(row.get("rvol"))
    if ret1 is not None and ret1 <= CROWDED_DIP_RET1 and (rvol is None or rvol >= CROWDED_DIP_RVOL):
        return None
    return "crowded_no_dip"


def apply_crowded_park(row: dict) -> Optional[str]:
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    reason = crowded_park_reason(row)
    if not reason:
        return None
    park_trade(row, reason)
    return reason


def pullback_reset(row: Optional[dict] = None) -> bool:
    """Red day into 20 EMA / AVWAP — freshness may reprint."""
    if not isinstance(row, dict):
        return False
    ret1 = to_float(row.get("ret_1"))
    if ret1 is None or ret1 >= 0:
        return False
    ext = to_float(row.get("extension_atr"))
    if ext is not None and ext < 0.5:
        return True
    close = to_float(row.get("close"))
    ema = to_float(row.get("ema20"))
    atr = to_float(row.get("atr14"))
    av = to_float(row.get("avwap_swing_low"))
    if close is not None and atr is not None and atr > 0:
        if ema is not None and abs(close - ema) <= 0.6 * atr:
            return True
        if av is not None and abs(close - av) <= 0.6 * atr:
            return True
    return False


def freshness_park_reason(row: Optional[dict] = None, prior_trades: Optional[Sequence[dict]] = None) -> Optional[str]:
    """Same ticker+setup that was TRADE last session stays WATCH until a pullback or group change."""
    if not isinstance(row, dict) or not prior_trades:
        return None
    ticker = str(row.get("ticker") or "").upper()
    primary = str(row.get("primary") or "")
    if not ticker or not primary:
        return None
    prior = None
    for item in prior_trades:
        if not isinstance(item, dict):
            continue
        if str(item.get("ticker") or "").upper() == ticker and str(item.get("primary") or "") == primary:
            prior = item
            break
    if prior is None:
        return None
    real_status = {"accelerating", "emerging", "mature", "deteriorating"}
    prior_status = str(prior.get("group_status") or "")
    status = str(row.get("group_status") or "")
    if prior_status in real_status and status in real_status and prior_status != status:
        return None
    if pullback_reset(row):
        return None
    # Skipped tickets reprint. Only hide a leftover if it is still a chase.
    ret1 = to_float(row.get("ret_1"))
    ext = to_float(row.get("extension_atr"))
    if (ret1 is not None and ret1 >= 0.03) or (ext is not None and ext > 1.8):
        return "already_recommended"
    return None


def apply_freshness_park(row: dict, prior_trades=None) -> Optional[str]:
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    reason = freshness_park_reason(row, prior_trades)
    if not reason:
        return None
    park_trade(row, reason)
    return reason


def analog_sample_improved(prior: Optional[dict], current: Optional[dict]) -> bool:
    """Un-park only if the book got better without shrinking n."""
    if not isinstance(prior, dict) or not isinstance(current, dict):
        return False
    p = prior.get("stock") if isinstance(prior.get("stock"), dict) else prior
    c = current.get("stock") if isinstance(current.get("stock"), dict) else current
    pn = int(p.get("n") or 0)
    cn = int(c.get("n") or 0)
    if cn < pn:
        return False
    pw = int(p.get("wins") or 0)
    cw = int(c.get("wins") or 0)
    if cw > pw:
        return True
    pa = to_float(p.get("avg_r"))
    ca = to_float(c.get("avg_r"))
    if pa is not None and ca is not None and ca > pa and ca > 0 and cw >= pw:
        return True
    return False


def analog_persist_reason(row: Optional[dict] = None, prior_analog: Optional[dict] = None) -> Optional[str]:
    """Keep yesterday's analog veto if n merely shrank. Do not un-park because the sample got smaller."""
    if not isinstance(row, dict) or not prior_analog:
        return None
    ticker = str(row.get("ticker") or "").upper()
    primary = str(row.get("primary") or "")
    prior = prior_analog.get((ticker, primary))
    if not isinstance(prior, dict):
        return None
    current = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    live = analog_park_reason(current)
    if live:
        return live
    if analog_sample_improved(prior, current):
        return None
    p = prior.get("stock") if isinstance(prior.get("stock"), dict) else prior
    c = current.get("stock") if isinstance(current.get("stock"), dict) else current
    pn = int(p.get("n") or 0)
    cn = int(c.get("n") or 0)
    if cn < pn:
        return str(prior.get("veto") or "analog_persist")
    return None


def apply_analog_persist_park(row: dict, prior_analog=None) -> Optional[str]:
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    reason = analog_persist_reason(row, prior_analog)
    if not reason:
        return None
    park_trade(row, reason)
    ev = row.get("evidence")
    if isinstance(ev, dict):
        ev["weak"] = True
        ev["analog_veto"] = reason
        ev["analog_persist"] = True
    return reason


def regime_trade_block(regime: Optional[str] = None, session_incomplete: bool = False) -> Optional[str]:
    if session_incomplete:
        return "session_incomplete"
    if str(regime or "") == "unknown":
        return "regime_unknown"
    return None


def apply_regime_trade_block(row: dict, regime: Optional[str] = None, session_incomplete: bool = False) -> Optional[str]:
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    reason = regime_trade_block(regime, session_incomplete)
    if not reason:
        return None
    park_trade(row, reason)
    return reason


def apply_already_held_park(row: dict) -> Optional[str]:
    """TRADE OPTIONS → WATCH when Schwab holds the same right. Exact open ticket stays TRADE."""
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    if row.get("choice") != "OPTIONS":
        return None
    if row.get("same_ticket"):
        return None
    reason = already_held_same_right_reason(row.get("picked") if isinstance(row.get("picked"), dict) else None, row.get("schwab_legs"))
    if not reason:
        return None
    park_trade(row, reason)
    return reason


def apply_analog_0win_park(row: dict) -> Optional[str]:
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    reason = analog_park_reason(row.get("evidence") if isinstance(row.get("evidence"), dict) else None)
    if not reason:
        return None
    park_trade(row, reason)
    ev = row.get("evidence")
    if isinstance(ev, dict):
        ev["weak"] = True
        ev["analog_veto"] = reason
    return reason


def below_ema_reason(row: Optional[dict] = None) -> Optional[str]:
    """Bullish OPTIONS whose last is already through 20 EMA are not a new TRADE."""
    if not isinstance(row, dict):
        return None
    if row.get("choice") != "OPTIONS":
        return None
    if str(row.get("direction") or "") != "bullish":
        return None
    close = to_float(row.get("close"))
    ema = to_float(row.get("ema20"))
    if close is None or ema is None:
        return None
    if close < ema:
        return "below_20ema"
    return None


def apply_below_ema_park(row: dict) -> Optional[str]:
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return None
    reason = below_ema_reason(row)
    if not reason:
        return None
    park_trade(row, reason)
    return reason


def stamp_fill_guard(row: dict) -> dict:
    """Numeric do-not-click band on last and debit/credit. Mutates picked + row."""
    picked = row.get("picked") if isinstance(row.get("picked"), dict) else {}
    direction = str(row.get("direction") or "")
    choice = str(row.get("choice") or "")
    ema = to_float(row.get("ema20"))
    av = to_float(row.get("avwap_swing_low"))
    debit = to_float(picked.get("target_debit") if picked else None)
    if debit is None:
        debit = to_float(picked.get("debit") if picked else None)
    credit = to_float(picked.get("target_credit") if picked else None)
    if credit is None:
        credit = to_float(picked.get("credit") if picked else None)
    stop = to_float(picked.get("stop") if picked else None)
    bits = []
    guard = {
        "stock_min": None,
        "stock_max": None,
        "debit_max": debit if choice == "OPTIONS" else None,
        "credit_min": credit if choice == "OPTIONS" else None,
        "note": "",
    }
    if choice == "OPTIONS" and direction == "bullish":
        floors = [x for x in (ema, av) if x is not None]
        guard["stock_min"] = max(floors) if floors else None
        if ema is not None:
            bits.append("Do not click if last < **%s** (20 EMA)" % fmt(ema))
        if av is not None:
            bits.append("Do not click if last < **%s** (swing-low AVWAP)" % fmt(av))
        if debit is not None:
            bits.append("Do not pay more than debit **%s**" % fmt(debit))
        elif credit is not None:
            bits.append("Do not collect less than credit **%s**" % fmt(credit))
        if ema is not None:
            bits.append("Skip the open if last gaps through **%s**" % fmt(ema))
        ret1 = to_float(row.get("ret_1"))
        if ret1 is not None and ret1 >= 0.03:
            bits.append("Already ran today — do not chase; wait for 20 EMA / AVWAP")
        if picked and not picked.get("invalidation"):
            picked["invalidation"] = "close back below 20 EMA %s / swing-low AVWAP %s" % (
                fmt(ema),
                fmt(av) if av is not None else "n/a",
            )
    elif choice == "OPTIONS" and direction == "bearish":
        guard["stock_max"] = ema
        if ema is not None:
            bits.append("Do not click if last > **%s** (20 EMA)" % fmt(ema))
        if debit is not None:
            bits.append("Do not pay more than debit **%s**" % fmt(debit))
        elif credit is not None:
            bits.append("Do not collect less than credit **%s**" % fmt(credit))
        if ema is not None:
            bits.append("Skip the open if last gaps through **%s**" % fmt(ema))
        if picked and not picked.get("invalidation"):
            picked["invalidation"] = "close back above 20 EMA %s / failed-breakdown reclaim" % fmt(ema)
    elif choice == "STOCK" and stop is not None:
        if direction == "bullish":
            bits.append("Do not buy if last is already through stop **%s**" % fmt(stop))
        else:
            bits.append("Do not short if last is already through stop **%s**" % fmt(stop))
    note = ". ".join(bits)
    if note and not note.endswith("."):
        note += "."
    guard["note"] = note
    row["fill_note"] = note
    row["fill_guard"] = guard
    if picked:
        picked["fill_note"] = note
        picked["fill_guard"] = guard
        row["picked"] = picked
    return guard


def open_trade_verdict(pos: dict, snap: Optional[dict] = None) -> dict:
    """Re-evaluate an open book row. Options use 20 EMA even if book.stop is missing."""
    snap = snap or {}
    last = to_float(snap.get("close"))
    stop = to_float(pos.get("stop"))
    target = to_float(pos.get("target"))
    ema = to_float(snap.get("ema20"))
    side = str(pos.get("side") or pos.get("direction") or "long").lower()
    if "short" in side or "bear" in side:
        side = "short"
    else:
        side = "long"
    inst = str(pos.get("instrument") or pos.get("structure") or "").lower()
    is_opt = any(k in inst for k in ("call", "put", "debit", "credit", "spread"))
    is_call = "call" in inst
    is_put = "put" in inst
    verdict = "HOLD"
    why = "original thesis not invalidated"
    if last is None:
        why = "last price DATA UNAVAILABLE"
    elif stop is not None and side == "long" and last <= stop:
        verdict = "EXIT"
        why = "stop / invalidation hit"
    elif stop is not None and side == "short" and last >= stop:
        verdict = "EXIT"
        why = "stop / invalidation hit"
    elif is_opt and ema is not None and side == "long" and is_call and not is_put and last < ema:
        verdict = "EXIT"
        why = "last below 20 EMA %s — bullish option thesis invalid" % fmt(ema)
    elif is_opt and ema is not None and side == "short" and is_put and last > ema:
        verdict = "EXIT"
        why = "last above 20 EMA %s — bearish option thesis invalid" % fmt(ema)
    elif is_opt and ema is not None and side == "long" and is_put and not is_call and last > ema:
        verdict = "EXIT"
        why = "last above 20 EMA %s — put debit thesis invalid" % fmt(ema)
    elif target is not None and side == "long" and last >= target:
        verdict = "TAKE PROFIT"
        why = "structure target reached"
    elif target is not None and side == "short" and last <= target:
        verdict = "TAKE PROFIT"
        why = "structure target reached"
    return {
        "verdict": verdict,
        "why": why,
        "last": last,
        "stop": stop,
        "side": side,
    }
