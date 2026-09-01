"""CLICK / SKIP / WATCH. Plain-language why. Not a profit promise."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from xhigh.num import fmt, to_float


GREEN = "#15803d"
AMBER = "#a16207"
RED = "#b91c1c"
MUTED = "#57534e"


def _span(color: str, text: str) -> str:
    return '<span style="color:%s;font-weight:700">%s</span>' % (color, text)


def classify(row: dict, gates: Optional[dict] = None) -> str:
    floor = 40
    if gates:
        floor = int((gates.get("score") or {}).get("conf_floor") or 40)
    ev = to_float(row.get("ev_proxy"))
    pop = to_float(row.get("pop_delta"))
    conf = int(row.get("conf") or 0)
    if pop is None or ev is None:
        return "WATCH"
    if ev > 0 and conf >= floor:
        return "CLICK"
    return "SKIP"


def need_line(row: dict) -> str:
    s = row.get("structure")
    last = to_float(row.get("last") or row.get("spot"))
    expiry = row.get("expiry_s") or row.get("expiry") or ""
    if s == "call_debit":
        long_k = to_float(row.get("long_strike"))
        short_k = to_float(row.get("short_strike"))
        debit = to_float(row.get("debit"))
        if long_k is None or debit is None:
            return "DATA UNAVAILABLE"
        be = long_k + debit
        cap = short_k if short_k is not None else long_k
        return "Needs stock > %s by %s. Max at %s+" % (fmt(be), expiry, fmt(cap, 0))
    if s == "put_debit":
        long_k = to_float(row.get("long_strike"))
        debit = to_float(row.get("debit"))
        short_k = to_float(row.get("short_strike"))
        if long_k is None or debit is None:
            return "DATA UNAVAILABLE"
        be = long_k - debit
        return "Needs stock < %s by %s. Max at %s" % (fmt(be), expiry, fmt(short_k or 0, 0))
    if s == "call_credit":
        short_k = to_float(row.get("short_strike") or row.get("strike"))
        return "Keep credit if stock stays < %s through %s" % (fmt(short_k, 0), expiry)
    if s in ("put_credit", "csp"):
        short_k = to_float(row.get("short_strike") or row.get("strike"))
        return "Keep credit if stock stays > %s through %s" % (fmt(short_k, 0), expiry)
    if s == "iron_condor":
        lo = to_float(row.get("put_short"))
        hi = to_float(row.get("call_short"))
        return "Keep credit if stock stays %s–%s through %s" % (fmt(lo, 0), fmt(hi, 0), expiry)
    if last is not None:
        return "Vs last %s" % fmt(last)
    return "DATA UNAVAILABLE"


def risk_line(row: dict) -> str:
    s = row.get("structure")
    if s in ("call_debit", "put_debit"):
        debit = to_float(row.get("debit"))
        if debit is None:
            return "DATA UNAVAILABLE"
        return "Lose at most $%s / lot" % fmt(debit * 100, 0)
    if s in ("put_credit", "call_credit", "iron_condor"):
        width = to_float(row.get("width"))
        credit = to_float(row.get("credit"))
        if width is None or credit is None:
            return "DATA UNAVAILABLE"
        return "Lose at most $%s / lot" % fmt(max(0, width - credit) * 100, 0)
    if s == "csp":
        strike = to_float(row.get("strike"))
        if strike is None:
            return "DATA UNAVAILABLE"
        return "Cash $%s; assigned at %s" % (fmt(strike * 100, 0), fmt(strike, 0))
    return "DATA UNAVAILABLE"


def why_line(row: dict) -> str:
    s = row.get("structure")
    ev = to_float(row.get("ev_proxy"))
    pop = to_float(row.get("pop_delta"))
    if row.get("action") == "CLICK" or (ev is not None and ev > 0):
        return "Typical win is bigger than typical loss. Hit rate may still be modest. Size small. Not a promise."
    if s == "csp":
        return "Looks like a high hit-rate credit, but it ties up full strike cash. Not a small defined-risk bet."
    if s in ("put_credit", "call_credit", "iron_condor"):
        return "You keep a small credit most days; one break costs more than many wins. Math says skip."
    if s in ("call_debit", "put_debit"):
        return "You need a directional move. Delta-POP is low, so typical loss > typical win."
    if ev is not None and ev <= 0:
        return "Expected win does not cover expected loss."
    return "Not enough to click."


def decorate(row: dict, gates: Optional[dict] = None) -> dict:
    out = dict(row)
    out["action"] = classify(out, gates)
    out["need_s"] = need_line(out)
    out["risk_s"] = risk_line(out)
    out["why_s"] = why_line(out)
    pill = {"CLICK": "🟢 CLICK", "SKIP": "🔴 SKIP", "WATCH": "🟡 WATCH"}
    out["do_s"] = pill.get(out["action"], out["action"])
    return out


def render_recommendation(date: str, click: List[dict], skip: List[dict], watch: List[dict], macro: Optional[dict] = None) -> List[str]:
    lines = [
        "# xhigh %s" % date,
        "",
        "## Recommendation",
        "",
    ]
    n_c, n_s, n_w = len(click), len(skip), len(watch)
    if not click:
        lines.append("%s · skip %s · watch %s" % (_span(AMBER, "CLICK 0"), n_s, n_w))
        lines.append("")
        lines.append("**Do nothing.** No row has typical win > typical loss. Empty is valid.")
        lines.append("")
    else:
        lines.append("%s · skip %s · watch %s" % (_span(GREEN, "CLICK %s" % n_c), n_s, n_w))
        lines.append("")
        for i, row in enumerate(click, 1):
            lines.append(
                "### %s %s %s" % (_span(GREEN, "🟢 CLICK"), row.get("ticker"), fmt(row.get("last")))
            )
            lines.append("")
            lines.append(
                "**%s** · %s · %s"
                % (row.get("strategy"), row.get("expiry_s") or row.get("expiry"), row.get("target_s"))
            )
            lines.append("")
            lines.append("- **Need:** %s" % row.get("need_s"))
            lines.append("- **Risk:** %s" % row.get("risk_s"))
            lines.append("- **POP (delta):** %s · **EV rank:** %s · **conf:** %s" % (row.get("pop_s"), row.get("ev_proxy"), row.get("conf")))
            lines.append("- **Why this one:** %s" % row.get("why_s"))
            lines.append("- **Not a promise.** Re-quote live before click. Do not fill last night’s number at 9:30.")
            lines.append("")
            if i == 1 and n_c == 1:
                lines.append("%s the other %s legal rows. Strikes are fine; payout vs loss is not." % (_span(RED, "🔴 SKIP"), n_s))
                lines.append("")
    lines.extend(
        [
            "### How to read this",
            "",
            "1. **Geometry first** — strike must sit on live last. A 270-call on a $186 stock is a bug, never a trade.",
            "2. **Then money** — CLICK only if typical win > typical loss (EV > 0). High POP with tiny credit still **SKIP**.",
            "3. **CSP is not a $100 debit.** Full cash at the strike. Ranked last on purpose.",
            "4. **POP is delta, not a crystal ball.** I cannot tell you it will be profitable. I can tell you which row is the only one worth considering.",
            "",
        ]
    )
    macro = macro or {}
    if macro.get("spy_last") is not None:
        lines.append(
            '<span style="color:%s">SPY %s · 5d %s%% · VIX %s</span>'
            % (
                MUTED,
                fmt(macro.get("spy_last")),
                macro.get("spy_5d") if macro.get("spy_5d") is not None else "n/a",
                fmt(macro.get("vix_last")) if macro.get("vix_last") is not None else "n/a",
            )
        )
        lines.append("")
    return lines
