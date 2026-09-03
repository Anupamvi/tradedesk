"""CLICK / SKIP / WATCH. Wheel and swing scored separately. Not a profit promise."""

from __future__ import annotations

from typing import List, Optional

from xhigh.dates import parse_any_date
from xhigh.num import fmt, to_float
from xhigh.score import DEFINED_CREDIT, DEBIT, csp_annualized, credit_over_width, rr_line, short_abs_delta


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


def dividend_inside(row: dict) -> Optional[bool]:
    asof = parse_any_date(row.get("asof"))
    expiry = parse_any_date(row.get("expiry"))
    div = parse_any_date(row.get("div_date"))
    if not asof or not expiry:
        return None
    if not div:
        return None
    if div <= asof:
        return False
    return div <= expiry


def debit_blockers(row: dict, gates: Optional[dict] = None) -> tuple:
    g_score = (gates or {}).get("score") or {}
    dmin = float(g_score.get("debit_long_delta_min") or 0.50)
    rmin = float(g_score.get("debit_rr_min") or 1.5)
    dte_min = int(g_score.get("debit_click_dte_min") or 35)
    ld = to_float(row.get("long_delta"))
    rr = to_float(row.get("rr"))
    dte = to_float(row.get("dte"))
    last = to_float(row.get("last") or row.get("spot"))
    long_k = to_float(row.get("long_strike"))
    short_k = to_float(row.get("short_strike"))
    s = row.get("structure")
    if ld is None or rr is None or dte is None or last is None or long_k is None:
        return "WATCH", []
    reasons = []
    if abs(ld) < dmin:
        reasons.append("|delta| %.2f (need ≥ %.2f)" % (abs(ld), dmin))
    if rr < rmin:
        reasons.append("R/R %.1f (need ≥ %.1f)" % (rr, rmin))
    if dte < dte_min:
        reasons.append("DTE %.0f (need ≥ %s). A 25–30 DTE debit dies on one down day." % (dte, dte_min))
    if s == "call_debit" and long_k > last + 1e-9:
        reasons.append("long %s is OTM vs last %s — no chase" % (fmt(long_k), fmt(last)))
    if s == "put_debit" and long_k < last - 1e-9:
        reasons.append("long %s is OTM vs last %s — no chase" % (fmt(long_k), fmt(last)))
    if s == "call_debit":
        inside = dividend_inside(row)
        if inside is True:
            reasons.append("ex-div %s is before expiry — call debit through the dividend" % row.get("div_date"))
        elif inside is None and not reasons:
            return "WATCH", []
    if s == "put_debit":
        low = to_float(row.get("low_126"))
        if low is not None and short_k is not None and short_k < low:
            reasons.append("max at %s is below the 6-month low %s" % (fmt(short_k, 0), fmt(low)))
    if reasons:
        return "SKIP", reasons
    return "CLICK", []


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
    if structure in DEFINED_CREDIT:
        frac = credit_over_width(row)
        width_min = float(g_score.get("credit_width_min") or 0.10)
        pop_min = float(g_score.get("credit_pop_min") or 0.70)
        if structure in ("put_credit", "iron_condor") and dividend_inside(row) is True:
            return "SKIP"
        if frac is not None and frac >= width_min and pop >= pop_min:
            return "CLICK"
        return "SKIP"
    if structure in DEBIT:
        action, _ = debit_blockers(row, gates)
        return action
    return "WATCH"


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


def why_line(row: dict, gates: Optional[dict] = None) -> str:
    s = row.get("structure")
    action = row.get("action")
    ann = csp_annualized(row)
    g_score = (gates or {}).get("score") or {}
    width_min = float(g_score.get("credit_width_min") or 0.10)
    pop_min = float(g_score.get("credit_pop_min") or 0.70)
    if s == "csp":
        low_126 = to_float(row.get("low_126"))
        strike = to_float(row.get("strike"))
        if action != "CLICK" and six_month_through_short(row, gates):
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
        if s == "put_credit" and six_month_through_short(row, gates):
            return "Naked CSP skipped (6-month low through the strike). Spread R/R is acceptable and loss is capped."
        if s in DEBIT:
            return "Long is at/ITM, DTE ≥ 35, no ex-div in the life. Typical win still bigger than typical loss. Hit rate is modest. Size small."
        if s in DEFINED_CREDIT:
            frac = credit_over_width(row)
            pct = "n/a" if frac is None else "%.1f%% of width" % (frac * 100)
            return "Paid %s. P:R %s. Loss is capped. Not a naked stock substitute." % (pct, rr_line(row))
        return "Typical win is bigger than typical loss. Not a promise."
    if s in DEFINED_CREDIT:
        if s in ("put_credit", "iron_condor") and dividend_inside(row) is True:
            return "ex-div %s is before expiry. A short put through the dividend is the KO pattern inverted — stock drops on the ex-date into the short put." % (
                row.get("div_date"),
            )
        frac = credit_over_width(row)
        pop = to_float(row.get("pop_delta"))
        pct = "n/a" if frac is None else "%.1f%% of width" % (frac * 100)
        if frac is not None and frac < width_min:
            return "Credit is %s (need ≥%.0f%% of width). P:R %s. 8–15%% OTM credits are not 1:4 debits." % (
                pct,
                width_min * 100,
                rr_line(row),
            )
        if pop is not None and pop < pop_min:
            return "POP %.0f%% (need ≥%.0f%%). Paid %s. P:R %s. Width is fine; the delta hit-rate proxy is not." % (
                pop * 100,
                pop_min * 100,
                pct,
                rr_line(row),
            )
        return "Credit is %s. P:R %s. Not a click." % (pct, rr_line(row))
    if s in DEBIT:
        _action, reasons = debit_blockers(row, gates)
        if reasons:
            return "Not a click: %s." % "; ".join(reasons)
        return "Long is OTM, DTE is short, or a dividend sits inside the expiry."
    return "Not enough to click."


def decorate(row: dict, gates: Optional[dict] = None) -> dict:
    out = dict(row)
    out["action"] = classify(out, gates)
    out["need_s"] = need_line(out)
    out["risk_s"] = risk_line(out)
    out["why_s"] = why_line(out, gates)
    out["rr_s"] = rr_line(out)
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
        "- **Profit:risk:** %s · **POP (delta):** %s · **conf:** %s"
        % (row.get("rr_s") or rr_line(row), row.get("pop_s"), row.get("conf")),
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


def risk_dollars(row: dict) -> Optional[float]:
    s = row.get("structure")
    if s in DEBIT:
        debit = to_float(row.get("debit"))
        if debit is None:
            return None
        return max(0.0, debit * 100.0)
    if s in DEFINED_CREDIT:
        width = to_float(row.get("width"))
        credit = to_float(row.get("credit"))
        if width is None or credit is None:
            return None
        return max(0.0, (width - credit) * 100.0)
    if s == "csp":
        strike = to_float(row.get("strike"))
        if strike is None:
            return None
        return max(0.0, strike * 100.0)
    return None


def sort_clicks(click: List[dict]) -> List[dict]:
    def key(row: dict):
        risk = risk_dollars(row)
        return (risk is None, risk if risk is not None else 0.0)

    return sorted(click, key=key)


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
        lines.append(
            "**Do nothing.** No debit with |delta| ≥ 0.50, DTE ≥ 35, long at/ITM, and no ex-div in the life — and no credit paid ≥ 10% of width with POP ≥ 70%. Empty is valid."
        )
        lines.append("")
    else:
        lines.append(
            "**🟢 CLICK %s** · wheel %s · swing %s · credit %s · skip %s"
            % (n_c, len(wheel), len(swing), len(credit), n_s)
        )
        lines.append("")
        for row in click:
            lines.extend(_click_block(row))
        if n_s:
            lines.append("**🔴 SKIP** %s legal rows. Geometry is fine; the click rule is not." % n_s)
            lines.append("")
    lines.extend(
        [
            "### How to read this",
            "",
            "1. **Geometry first** — strike must sit on live last. A 270-call on a $186 stock is a bug, never a trade.",
            "2. **Sleeves are independent.** A name can have a swing debit and a defined-risk credit. Rank small dollars-at-risk first. Do not hide a passing credit because a debit also passed.",
            "3. **Swing CLICK** — long at/ITM (|delta| ≥ 0.50), DTE ≥ 35, R/R ≥ 1.5, and no ex-div before expiry. A 25-DTE 0.35-delta debit is how KO lost 34% in a day. EV is not the click rule.",
            "4. **Wheel** — Naked CSP only if paid ≥ 8% annualized and the 6-month low did **not** already trade through the strike. If it did, recommend a **put credit** instead (defined-risk). A 50% drop is shown in dollars on naked puts. Not a growth forecast.",
            "5. **Credit CLICK** — paid at least **10% of the spread width** and POP ≥ 70%. An 8–15% OTM put is naturally ~1:7; requiring 1:4 emptied the board. 1:14 still SKIP.",
            "6. **POP is delta, not a forecast.** I cannot promise profit.",
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
