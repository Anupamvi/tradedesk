"""Desk pick: one best options ticket and one best stock ticket from TRADE rows."""

from __future__ import annotations

from typing import Dict, List, Optional

from groat.num import fmt, fmt_pct, to_float
from groat.setups import SETUP_NAMES


def _picked(row: dict) -> dict:
    p = row.get("picked")
    return p if isinstance(p, dict) else {}


def _otm_pct(row: dict) -> Optional[float]:
    close = to_float(row.get("close"))
    picked = _picked(row)
    long_k = to_float(picked.get("long_strike") or picked.get("strike"))
    inst = str(picked.get("instrument") or "")
    if close is None or close <= 0 or long_k is None:
        return None
    if "put" in inst:
        return 1.0 - (long_k / close)
    return (long_k / close) - 1.0


def _net_delta(row: dict) -> Optional[float]:
    return to_float(_picked(row).get("delta"))


def score_option_ticket(row: dict) -> float:
    s = 0.0
    pop = to_float(row.get("naive_pop"))
    if pop is not None:
        s += pop * 80.0
    conf = to_float(row.get("opt_conf"))
    if conf is not None:
        s += conf * 0.35
    s += (to_float(row.get("score")) or 0) * 0.25
    otm = _otm_pct(row)
    if otm is not None:
        if otm <= 0.015:
            s += 16
        elif otm <= 0.03:
            s += 8
        elif otm > 0.05:
            s -= 12
    delta = abs(_net_delta(row) or 0)
    if delta >= 0.18:
        s += 14
    elif delta >= 0.12:
        s += 7
    elif 0 < delta < 0.10:
        s -= 16
    tag = str(row.get("x") or "")
    if tag == "Crowded":
        s -= 14
    elif tag == "Quiet":
        s += 6
    elif tag == "Informed":
        s += 5
    ret1 = to_float(row.get("ret_1"))
    if ret1 is not None and ret1 >= 0.12:
        s -= 10
    elif ret1 is not None and 0.03 <= ret1 <= 0.11:
        s += 3
    return s


def score_stock_ticket(row: dict) -> float:
    s = to_float(row.get("score")) or 0
    rs = to_float(row.get("rs_20")) or 0
    s += min(12.0, max(-8.0, rs * 40.0))
    tag = str(row.get("x") or "")
    if tag == "Crowded":
        s -= 8
    elif tag == "Informed":
        s += 4
    ret1 = to_float(row.get("ret_1"))
    if ret1 is not None and ret1 >= 0.15:
        s -= 10
    return s


def _opt_why(row: dict) -> List[str]:
    notes = []
    otm = _otm_pct(row)
    if otm is not None:
        if otm <= 0.02:
            notes.append("long strike is near the money (%.1f%% from last)" % (otm * 100))
        else:
            notes.append("long strike is %.1f%% OTM" % (otm * 100))
    d = abs(_net_delta(row) or 0)
    if d:
        notes.append("net delta %.2f" % d)
    pop = to_float(row.get("naive_pop"))
    if pop is not None:
        notes.append("naive POP %s" % fmt_pct(pop, 0))
    if row.get("opt_conf") is not None:
        notes.append("conf %s" % row.get("opt_conf"))
    tag = row.get("x") or "DATA UNAVAILABLE"
    notes.append("X %s" % tag)
    return notes


def desk_picks(trades: List[dict]) -> Dict[str, object]:
    opts = [r for r in trades if r.get("choice") == "OPTIONS"]
    stocks = [r for r in trades if r.get("choice") == "STOCK"]
    best_opt = max(opts, key=score_option_ticket) if opts else None
    best_stk = max(stocks, key=score_stock_ticket) if stocks else None
    caution = []
    for row in opts:
        d = abs(_net_delta(row) or 0)
        if row.get("x") == "Crowded" or d < 0.10:
            caution.append(row)
    ranked_opts = sorted(opts, key=score_option_ticket, reverse=True)
    return {
        "best_options": best_opt,
        "best_stock": best_stk,
        "caution": caution,
        "ranked_options": ranked_opts,
    }


def render_desk_picks(picks: dict) -> List[str]:
    lines = ["## Desk pick", ""]
    best_opt = picks.get("best_options")
    best_stk = picks.get("best_stock")
    ranked = picks.get("ranked_options") or []
    if not best_opt and not best_stk:
        lines.append("No TRADE rows to pick. Valid.")
        lines.append("")
        return lines
    if best_opt:
        p = _picked(best_opt)
        lines.append(
            "**Take options: %s** — %s. Pay **%s**. Naive POP **%s**, conf **%s**."
            % (
                best_opt.get("ticker"),
                p.get("legs") or best_opt.get("choice"),
                ("debit %s" % fmt(p.get("target_debit"))) if p.get("target_debit") is not None else (
                    "credit %s" % fmt(p.get("target_credit")) if p.get("target_credit") is not None else "n/a"
                ),
                fmt_pct(best_opt.get("naive_pop"), 0) if best_opt.get("naive_pop") is not None else "n/a",
                best_opt.get("opt_conf") if best_opt.get("opt_conf") is not None else "n/a",
            )
        )
        lines.append("")
        lines.append(
            "Why this one: "
            + "; ".join(_opt_why(best_opt))
            + ". Setup %s (%s). Sub-50%% naive POP is normal for an OTM/near-OTM debit — conf is structure quality, not P(win)."
            % (best_opt.get("primary") or "", SETUP_NAMES.get(best_opt.get("primary") or "", ""))
        )
        lines.append("")
        lines.append(
            "Act: work the fill at or inside the stated debit/credit. 1 lot first if X is Crowded. Invalidation: %s."
            % (p.get("invalidation") or "thesis/setup break")
        )
        if best_opt.get("held") or best_opt.get("in_book"):
            lines.append("")
            lines.append(best_opt.get("held_note") or "Already held. Shown for visibility — do not add.")
        lines.append("")
    else:
        lines.append("**Options:** none cleared. Valid.")
        lines.append("")
    if ranked:
        lines.append("Why this one, not the others:")
        lines.append("")
        for row in ranked[:6]:
            tag = " **← take this**" if best_opt is not None and row.get("ticker") == best_opt.get("ticker") else ""
            held = " IN BOOK" if row.get("in_book") or row.get("held") else ""
            lines.append("- **%s**%s%s — %s." % (row.get("ticker"), tag, held, "; ".join(_opt_why(row))))
        lines.append("")
    if best_stk:
        p = _picked(best_stk)
        lines.append(
            "**Stock if you want one: %s** — buy ~%s, stop **%s**, target **%s**, %s shares. Setup %s."
            % (
                best_stk.get("ticker"),
                fmt(p.get("entry") or best_stk.get("close")),
                fmt(p.get("stop")),
                fmt(p.get("target")),
                p.get("shares") or "",
                SETUP_NAMES.get(best_stk.get("primary") or "", best_stk.get("primary") or ""),
            )
        )
        ret1 = to_float(best_stk.get("ret_1"))
        if ret1 is not None and ret1 >= 0.12:
            lines.append("")
            lines.append("After a ≥12%% day, waiting for a pullback into 20 EMA / AVWAP is valid. Do not invent calls if the chain failed.")
        lines.append("")
    caution = picks.get("caution") or []
    if caution:
        bits = []
        for row in caution:
            d = abs(_net_delta(row) or 0)
            why = []
            if row.get("x") == "Crowded":
                why.append("Crowded X")
            if d and d < 0.10:
                why.append("net delta %.2f" % d)
            bits.append("%s (%s)" % (row.get("ticker"), ", ".join(why) or "caution"))
        lines.append("Caution / size down: " + "; ".join(bits) + ".")
        lines.append("")
    extra = str(picks.get("evidence_line") or "").strip()
    if extra:
        lines.append(extra)
        lines.append("")
    lines.append("Take **one** options ticket unless you explicitly want two uncorrelated names. Prefer 1–3 positions total.")
    lines.append("")
    return lines
