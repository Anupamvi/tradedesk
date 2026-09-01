"""CLICK / SKIP / WATCH. Wheel and swing scored separately. Not a profit promise."""

from __future__ import annotations

from typing import List, Optional

from xhigh.num import fmt, to_float
from xhigh.score import DEFINED_CREDIT, DEBIT, csp_annualized, credit_over_width, short_abs_delta


def wheel_stress(row: dict):
    last = to_float(row.get("last") or row.get("spot"))
    strike = to_float(row.get("strike"))
    credit = to_float(row.get("credit")) or 0.0
    if last is None or last <= 0 or strike is None:
        return None, None
    px = last * 0.5
    basis = strike - credit
    pnl = (px - basis) * 100.0
    return px, pnl


def stress_line(row: dict) -> str:
    px, pnl = wheel_stress(row)
    if px is None or pnl is None:
        return "DATA UNAVAILABLE"
    return "If stock is %s in 6 months (half of last), this lot ≈ $%s vs %s basis. Not a forecast." % (
        fmt(px, 0),
        fmt(pnl, 0),
        fmt(to_float(row.get("strike")) - (to_float(row.get("credit")) or 0), 0),
    )


def six_month_through_short(row: dict, gates: Optional[dict] = None) -> bool:
    short = to_float(row.get("short_strike") or row.get("strike"))
    low = to_float(row.get("low_126"))
    through = 0.85
    if gates:
        through = float((gates.get("csp") or {}).get("through_strike_frac") or 0.85)
    if low is None or short is None or short <= 0:
        return False
    return low < short * through


def low_line(row: dict) -> str:
    low = to_float(row.get("low_126"))
    if low is None:
        return ""
    return "6-month low %s" % fmt(low)


def _floor(gates: Optional[dict]) -> int:
    if not gates:
        return 40
    return int((gates.get("score") or {}).get("conf_floor") or 40)


def classify(row: dict, gates: Optional[dict] = None) -> str:
    floor = _floor(gates)
    pop = to_float(row.get("pop_delta"))
    conf = int(row.get("conf") or 0)
    structure = row.get("structure")
    if pop is None:
        return "WATCH"
    if conf < floor:
        return "SKIP"
    g_csp = (gates or {}).get("csp") or {}
    g_score = (gates or {}).get("score") or {}
    if structure == "csp":
        ann = csp_annualized(row)
        credit = to_float(row.get("credit"))
        strike = to_float(row.get("strike"))
        dlt = short_abs_delta(row)
        if ann is None or credit is None or strike is None or strike <= 0 or dlt is None:
            return "WATCH"
        min_ann = float(g_csp.get("click_annualized_min") or 0.08)
        max_d = float(g_csp.get("click_max_abs_delta") or 0.25)
        if ann < min_ann or dlt > max_d:
            return "SKIP"
        if six_month_through_short(row, gates):
            return "SKIP"
        return "CLICK"
    if structure == "put_credit":
        ev = to_float(row.get("ev_proxy"))
        frac = credit_over_width(row)
        width_min = float(g_score.get("credit_width_min") or 0.18)
        pop_min = float(g_score.get("credit_pop_min") or 0.70)
        g_pc = (gates or {}).get("put_credit") or {}
        considerate_min = float(g_pc.get("considerate_width_min") or 0.06)
        if ev is not None and ev > 0:
            return "CLICK"
        if frac is not None and frac >= width_min and pop >= pop_min:
            return "CLICK"
        if six_month_through_short(row, gates) and frac is not None and frac >= considerate_min and pop >= pop_min:
            return "CLICK"
        return "SKIP"
    if structure in DEFINED_CREDIT:
        ev = to_float(row.get("ev_proxy"))
        frac = credit_over_width(row)
        width_min = float(g_score.get("credit_width_min") or 0.18)
        pop_min = float(g_score.get("credit_pop_min") or 0.70)
        if ev is not None and ev > 0:
            return "CLICK"
        if frac is not None and frac >= width_min and pop >= pop_min:
            return "CLICK"
        return "SKIP"
    ev = to_float(row.get("ev_proxy"))
    if ev is None:
        return "WATCH"
    if ev > 0:
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
    if s == "put_credit":
        short_k = to_float(row.get("short_strike") or row.get("strike"))
        long_k = to_float(row.get("long_strike"))
        low = low_line(row)
        base = "Keep credit if stock stays > %s through %s. Defined-risk floor %s" % (
            fmt(short_k, 0),
            expiry,
            fmt(long_k, 0),
        )
        if low:
            return "%s. %s" % (base, low)
        return base
    if s == "csp":
        strike = to_float(row.get("strike"))
        if last is None or strike is None or last <= 0:
            return "Assignment at %s through %s" % (fmt(strike, 0), expiry)
        otm = (last - strike) / last
        return "Paid to wait. Assignment at %s is buying ~%.0f%% below last. Expiry %s" % (
            fmt(strike, 0),
            otm * 100,
            expiry,
        )
    if s == "iron_condor":
        lo = to_float(row.get("put_short"))
        hi = to_float(row.get("call_short"))
        return "Keep credit if stock stays %s–%s through %s" % (fmt(lo, 0), fmt(hi, 0), expiry)
    if last is not None:
        return "Vs last %s" % fmt(last)
    return "DATA UNAVAILABLE"


def risk_line(row: dict) -> str:
    s = row.get("structure")
    if s in DEBIT:
        debit = to_float(row.get("debit"))
        if debit is None:
            return "DATA UNAVAILABLE"
        return "Lose at most $%s / lot" % fmt(debit * 100, 0)
    if s in DEFINED_CREDIT:
        width = to_float(row.get("width"))
        credit = to_float(row.get("credit"))
        if width is None or credit is None:
            return "DATA UNAVAILABLE"
        return "Lose at most $%s / lot" % fmt(max(0, width - credit) * 100, 0)
    if s == "csp":
        strike = to_float(row.get("strike"))
        if strike is None:
            return "DATA UNAVAILABLE"
        base = "Cash $%s; assigned at %s (you wanted the stock)" % (fmt(strike * 100, 0), fmt(strike, 0))
        extra = stress_line(row)
        if extra and extra != "DATA UNAVAILABLE":
            return base + ". " + extra
        return base
    return "DATA UNAVAILABLE"


def why_line(row: dict) -> str:
    s = row.get("structure")
    action = row.get("action")
    ann = csp_annualized(row)
    if s == "csp":
        low_126 = to_float(row.get("low_126"))
        strike = to_float(row.get("strike"))
        if action != "CLICK" and low_126 is not None and strike is not None and low_126 < strike * 0.85:
            return "Last 6 months already traded at %s, through your %s strike. This is not a quiet dip." % (
                fmt(low_126, 0),
                fmt(strike, 0),
            )
        if action == "CLICK" and ann is not None:
            return "Paid %.1f%% annualized on cash to wait for a pullback you would own. If the stock halves in 6 months you still own it — see stress. Not a growth forecast." % (
                ann * 100
            )
        if ann is not None:
            return "Not paid enough to tie cash at the strike (%.1f%% annualized vs 8%% hurdle)." % (ann * 100)
        return "Wheel credit too thin vs cash tied up."
    if action == "CLICK":
        if s == "put_credit" and six_month_through_short(row):
            return "Naked CSP skipped: 6-month low already traded through the short strike. This put spread caps the loss if that low returns."
        if s in DEBIT:
            return "Typical win is bigger than typical loss on this defined-risk debit. Hit rate may still be modest. Size small."
        if s in DEFINED_CREDIT:
            return "Credit is a large enough slice of the width, or defined-risk EV is positive."
        return "Typical win is bigger than typical loss. Not a promise."
    if s in DEFINED_CREDIT:
        return "Credit is too small vs the width. You keep pennies; one break costs more than many wins."
    if s in DEBIT:
        return "You need a directional move. Delta-POP is low, so typical loss > typical win."
    return "Not enough to click."


def decorate(row: dict, gates: Optional[dict] = None) -> dict:
    out = dict(row)
    out["action"] = classify(out, gates)
    out["need_s"] = need_line(out)
    out["risk_s"] = risk_line(out)
    out["why_s"] = why_line(out)
    if out.get("structure") == "csp":
        out["stress_s"] = stress_line(out)
    ann = csp_annualized(out)
    if ann is not None and out.get("structure") == "csp":
        out["yield_s"] = "%.1f%% ann." % (ann * 100)
    pill = {"CLICK": "🟢 CLICK", "SKIP": "🔴 SKIP", "WATCH": "🟡 WATCH"}
    out["do_s"] = pill.get(out["action"], out["action"])
    if out.get("structure") == "csp":
        out["sleeve"] = "wheel"
    elif out.get("structure") in DEBIT:
        out["sleeve"] = "swing"
    else:
        out["sleeve"] = "credit"
    return out


def _click_block(row: dict) -> List[str]:
    sleeve = row.get("sleeve") or ""
    label = {"wheel": "WHEEL", "swing": "SWING", "credit": "CREDIT"}.get(sleeve, "")
    extra = ""
    if label:
        extra = " · %s" % label
    lines = [
        "### 🟢 CLICK — %s %s%s" % (row.get("ticker"), fmt(row.get("last")), extra),
        "",
        "**%s** · %s · %s"
        % (row.get("strategy"), row.get("expiry_s") or row.get("expiry"), row.get("target_s")),
        "",
        "- **Need:** %s" % row.get("need_s"),
        "- **Risk:** %s" % row.get("risk_s"),
        "- **POP (delta):** %s · **rank:** %s · **conf:** %s"
        % (row.get("pop_s"), row.get("yield_s") or row.get("ev_proxy"), row.get("conf")),
        "- **Why this one:** %s" % row.get("why_s"),
    ]
    if row.get("structure") == "csp":
        lines.append("- **6-month stress:** %s" % (row.get("stress_s") or stress_line(row)))
    if low_line(row):
        lines.append("- **Tape:** %s" % low_line(row))
    lines.extend(
        [
            "- **Not a promise.** Re-quote live before click. Do not fill last night’s number at 9:30.",
            "",
        ]
    )
    return lines


def render_recommendation(date: str, click: List[dict], skip: List[dict], watch: List[dict], macro: Optional[dict] = None) -> List[str]:
    lines = [
        "# xhigh %s" % date,
        "",
        "## Recommendation",
        "",
    ]
    n_c, n_s, n_w = len(click), len(skip), len(watch)
    wheel = [r for r in click if r.get("structure") == "csp"]
    swing = [r for r in click if r.get("structure") in DEBIT]
    credit = [r for r in click if r.get("structure") in DEFINED_CREDIT]
    if not click:
        lines.append("**🟡 CLICK 0** · skip %s · watch %s" % (n_s, n_w))
        lines.append("")
        lines.append("**Do nothing.** No swing with defined-risk EV > 0, and no wheel paid enough to wait. Empty is valid.")
        lines.append("")
    else:
        lines.append(
            "**🟢 CLICK %s** · wheel %s · swing %s · credit %s · skip %s"
            % (n_c, len(wheel), len(swing), len(credit), n_s)
        )
        lines.append("")
        for row in swing + wheel + credit:
            lines.extend(_click_block(row))
        if n_s:
            lines.append("**🔴 SKIP** %s legal rows. Geometry is fine; payout vs cash or width is not." % n_s)
            lines.append("")
    lines.extend(
        [
            "### How to read this",
            "",
            "1. **Geometry first** — strike must sit on live last. A 270-call on a $186 stock is a bug, never a trade.",
            "2. **Swing CLICK** — defined-risk debit where typical win > typical loss.",
            "3. **Wheel** — Naked CSP only if paid ≥ 8% annualized and the 6-month low did **not** already trade through the strike. If it did, recommend a **put credit** instead (defined-risk). A 50% drop is shown in dollars on naked puts. Not a growth forecast.",
            "4. **Credit CLICK** — defined-risk spread where credit is ≥ 18% of width (or EV > 0). Skinny credits **SKIP** even if POP is high.",
            "5. **POP is delta, not a forecast.** I cannot promise profit.",
            "",
        ]
    )
    macro = macro or {}
    if macro.get("spy_last") is not None:
        lines.append(
            "_SPY %s · 5d %s%% · VIX %s_"
            % (
                fmt(macro.get("spy_last")),
                macro.get("spy_5d") if macro.get("spy_5d") is not None else "n/a",
                fmt(macro.get("vix_last")) if macro.get("vix_last") is not None else "n/a",
            )
        )
        lines.append("")
    return lines
