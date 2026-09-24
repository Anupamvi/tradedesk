"""Market regime from SPY/QQQ/IWM/DIA, VIX, TLT, UUP, universe breadth."""

from __future__ import annotations

from typing import Dict, List, Optional

from groat.num import fmt, fmt_pct, to_float
from groat.technicals import snapshot


REGIME_STRATEGIES = {
    "strong_risk_on": "Continuation, pullbacks in leaders (A/B/D/E). Avoid fading strength.",
    "weak_risk_on": "Selective pullbacks in relative-strength names. Smaller size. Skip extended breakouts.",
    "range_chop": "Mean-reversion only with structure. Fade breakouts. Prefer defined-risk. Many days are NO TRADE.",
    "rotation": "Buy emerging groups, avoid lagging megacap if small-caps/sectors lead. Setup E first.",
    "risk_off": "Capital preservation. Failed-breakout shorts (G) only with sector confirmation. No chase longs.",
    "high_vol_liquidation": "Stand aside unless a predefined event/hedge. Do not buy the first washout.",
    "post_correction_recovery": "RS leaders reclaiming AVWAP/20 EMA. Prefer confirmation over first bounce.",
    "short_covering": "Do not chase. Wait for pullback into reclaimed AVWAP. Treat as tactical, not a new trend yet.",
    "unknown": "Insufficient tape. DATA UNAVAILABLE for a confident regime. Default to NO TRADE unless a name is exceptional.",
}


def _snap(bars_map: Dict[str, list], ticker: str, asof: str, bench: Optional[list] = None) -> dict:
    return snapshot(bars_map.get(ticker) or [], asof, bench_bars=bench)


def classify(
    asof: str,
    bars_map: Dict[str, list],
    vix_bars: Optional[list] = None,
    universe_snaps: Optional[List[dict]] = None,
) -> Dict[str, object]:
    spy_bars = bars_map.get("SPY") or []
    spy = _snap(bars_map, "SPY", asof)
    qqq = _snap(bars_map, "QQQ", asof, spy_bars)
    iwm = _snap(bars_map, "IWM", asof, spy_bars)
    dia = _snap(bars_map, "DIA", asof, spy_bars)
    tlt = _snap(bars_map, "TLT", asof, spy_bars)
    uup = _snap(bars_map, "UUP", asof, spy_bars)
    vix = snapshot(vix_bars or bars_map.get("$VIX.X") or [], asof) if (vix_bars or bars_map.get("$VIX.X")) else {}

    breadth = _breadth(universe_snaps or [])
    label = "unknown"
    notes = []
    if not spy.get("ok"):
        notes.append("SPY tape DATA UNAVAILABLE")
        return {
            "asof": asof,
            "regime": label,
            "why": notes,
            "playbook": REGIME_STRATEGIES[label],
            "spy": spy,
            "qqq": qqq,
            "iwm": iwm,
            "dia": dia,
            "tlt": tlt,
            "uup": uup,
            "vix": vix,
            "breadth": breadth,
        }

    vix_px = to_float((vix or {}).get("close"))
    vix_ret5 = to_float((vix or {}).get("ret_5"))
    spy_ret5 = to_float(spy.get("ret_5"))
    spy_ret20 = to_float(spy.get("ret_20"))
    trend = str(spy.get("trend") or "")
    iwm_rs20 = to_float(iwm.get("rs_20"))
    qqq_rs20 = to_float(qqq.get("rs_20"))

    if vix_px is not None and vix_px >= 28 and spy_ret5 is not None and spy_ret5 <= -0.03:
        label = "high_vol_liquidation"
        notes.append("VIX elevated with a sharp SPY decline")
    elif trend == "strong_up" and (spy_ret20 or 0) > 0.01 and (iwm_rs20 is None or iwm_rs20 > -0.04):
        label = "strong_risk_on"
        notes.append("SPY stacked above 20/50/200 with a rising 20 EMA")
    elif trend in ("up", "strong_up") and spy_ret20 is not None and spy_ret20 < 0:
        label = "weak_risk_on"
        notes.append("Uptrend structure but 20-day return is negative")
    elif trend == "range" or (
        spy.get("ema20")
        and spy.get("sma50")
        and abs(float(spy["ema20"]) - float(spy["sma50"])) / float(spy["close"]) < 0.01
        and abs(spy_ret20 or 0) < 0.02
    ):
        label = "range_chop"
        notes.append("20/50 compressed and SPY 20-day return is small")
    elif trend in ("down", "strong_down"):
        if spy.get("above_ema20") and spy_ret5 and spy_ret5 > 0.03:
            label = "short_covering"
            notes.append("Downtrend but a sharp short-term bounce")
        elif spy.get("above_ema20") and spy.get("above_sma50") is False:
            label = "post_correction_recovery"
            notes.append("Reclaiming the 20 EMA after being below the 50")
        else:
            label = "risk_off"
            notes.append("SPY below key moving averages")
    elif iwm_rs20 is not None and qqq_rs20 is not None and abs(iwm_rs20 - qqq_rs20) > 0.04:
        label = "rotation"
        notes.append("IWM and QQQ 20-day relative strength diverge")
    elif trend == "up":
        label = "weak_risk_on"
        notes.append("Uptrend without full 20>50>200 stack")
    else:
        label = "unknown"
        notes.append("Tape does not match a clean regime bucket")

    if vix_px is None:
        notes.append("VIX DATA UNAVAILABLE")
    if not tlt.get("ok"):
        notes.append("TLT DATA UNAVAILABLE")
    if not uup.get("ok"):
        notes.append("UUP / DXY proxy DATA UNAVAILABLE")

    return {
        "asof": asof,
        "regime": label,
        "why": notes,
        "playbook": REGIME_STRATEGIES.get(label) or REGIME_STRATEGIES["unknown"],
        "spy": spy,
        "qqq": qqq,
        "iwm": iwm,
        "dia": dia,
        "tlt": tlt,
        "uup": uup,
        "vix": vix,
        "vix_px": vix_px,
        "vix_ret_5": vix_ret5,
        "breadth": breadth,
        "qqq_rs_20": qqq_rs20,
        "iwm_rs_20": iwm_rs20,
    }


def _breadth(snaps: List[dict]) -> Dict[str, object]:
    usable = [s for s in snaps if s.get("ok")]
    n = len(usable)
    if n == 0:
        return {"n": 0, "pct_20": None, "pct_50": None, "pct_200": None, "note": "DATA UNAVAILABLE"}

    def pct(key):
        hits = sum(1 for s in usable if s.get(key) is True)
        return hits / float(n)

    return {
        "n": n,
        "pct_20": pct("above_ema20"),
        "pct_50": pct("above_sma50"),
        "pct_200": pct("above_sma200"),
        "note": "universe sample, not official NYSE breadth",
    }


def render_regime(reg: dict) -> List[str]:
    spy = reg.get("spy") or {}
    qqq = reg.get("qqq") or {}
    iwm = reg.get("iwm") or {}
    vix = reg.get("vix") or {}
    b = reg.get("breadth") or {}
    lines = [
        "# Market regime",
        "",
        "**Regime:** %s" % (reg.get("regime") or "unknown"),
        "",
        (reg.get("playbook") or ""),
        "",
        "## Why",
        "",
    ]
    for note in reg.get("why") or []:
        lines.append("- %s" % note)
    lines.extend(
        [
            "",
            "## Tape",
            "",
            "| name | close | 5d | 20d | 60d | vs 20 EMA | vs 50 | vs 200 | trend |",
            "|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for label, snap in (("SPY", spy), ("QQQ", qqq), ("IWM", iwm), ("DIA", reg.get("dia") or {}), ("TLT", reg.get("tlt") or {})):
        if not snap:
            continue
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                label,
                fmt(snap.get("close")),
                fmt_pct(snap.get("ret_5")),
                fmt_pct(snap.get("ret_20")),
                fmt_pct(snap.get("ret_60")),
                "above" if snap.get("above_ema20") else ("below" if snap.get("above_ema20") is False else "DATA UNAVAILABLE"),
                "above" if snap.get("above_sma50") else ("below" if snap.get("above_sma50") is False else "DATA UNAVAILABLE"),
                "above" if snap.get("above_sma200") else ("below" if snap.get("above_sma200") is False else "DATA UNAVAILABLE"),
                snap.get("trend") or "",
            )
        )
    lines.extend(
        [
            "",
            "VIX: %s (5d %s)" % (fmt(reg.get("vix_px") or vix.get("close")), fmt_pct(reg.get("vix_ret_5") or vix.get("ret_5"))),
            "",
            "Universe breadth (n=%s): >20 EMA %s · >50 DMA %s · >200 DMA %s"
            % (
                b.get("n") or 0,
                fmt_pct(b.get("pct_20")),
                fmt_pct(b.get("pct_50")),
                fmt_pct(b.get("pct_200")),
            ),
            "",
        ]
    )
    return lines
