"""Quality 70 / premium 30 / overlay. Never invent missing ORATS or Schwab numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from wheelo.dates import days_until, parse_any_date, usable_date
from wheelo.num import iv_decimal, to_float


def tier_score(
    value: float,
    excellent: float,
    good: float,
    fair: float,
    lower_is_better: bool = False,
) -> int:
    if lower_is_better:
        if value <= excellent:
            return 100
        if value <= good:
            return 75
        if value <= fair:
            return 50
        return 25
    if value >= excellent:
        return 100
    if value >= good:
        return 75
    if value >= fair:
        return 50
    return 25


@dataclass
class QualityScore:
    size_score: int = 0
    borrow_score: int = 0
    path_score: int = 0
    beta_score: int = 0
    earnings_score: int = 0
    div_score: int = 0
    pe_score: int = 0
    confidence_score: int = 0
    yfinance_note: str = ""
    composite: float = 0.0
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class PremiumScore:
    csp_strike: float = 0.0
    csp_premium: float = 0.0
    csp_bid: float = 0.0
    csp_ask: float = 0.0
    csp_yield_ann: float = 0.0
    csp_yield_score: int = 0
    cc_strike: float = 0.0
    cc_premium: float = 0.0
    cc_bid: float = 0.0
    cc_ask: float = 0.0
    cc_yield_ann: float = 0.0
    cc_yield_score: int = 0
    iv_rank: float = 0.0
    iv_rank_score: int = 0
    spread_pct: float = 0.0
    spread_score: int = 0
    dte: int = 0
    expiry: str = ""
    composite: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class SentimentAdjustment:
    total: float = 0.0
    notes: List[str] = field(default_factory=list)
    x_status: str = "DATA UNAVAILABLE"


def compute_mean_reversion(bars: List[dict], drawdown_pct: float = 10, recovery_days: int = 30) -> Optional[float]:
    closes = []
    for row in bars or []:
        px = to_float(row.get("close"))
        if px is not None:
            closes.append(px)
    if len(closes) < 30:
        return None
    n = len(closes)
    peak = closes[0]
    in_dd = False
    start = 0
    recovered = 0
    events = 0
    for i, px in enumerate(closes):
        if px > peak:
            peak = px
        dd = (peak - px) / peak * 100.0 if peak else 0.0
        if not in_dd and dd >= drawdown_pct:
            in_dd = True
            start = i
            events += 1
        elif in_dd and px >= peak:
            in_dd = False
            if i - start <= recovery_days:
                recovered += 1
    if events == 0:
        return 100.0
    return 100.0 * recovered / events


def compute_sigma_strike(spot: float, iv: float, dte: int, side: str = "put", sigma: float = 1.0) -> float:
    move = spot * iv * math.sqrt(max(dte, 1) / 365.0) * sigma
    if side == "call":
        return round(spot + move, 2)
    return round(spot - move, 2)


def earnings_days(core: dict, asof: str) -> Optional[int]:
    nxt = usable_date(core.get("next_ern")) or parse_any_date(core.get("next_ern"))
    if nxt:
        d = days_until(asof, nxt)
        if d is not None:
            return d
    wks = to_float(core.get("wks_next_ern"))
    if wks is not None and 0 < wks <= 26:
        return max(1, int(round(wks * 7)))
    days = core.get("days_to_ern")
    if days is None:
        return None
    try:
        n = int(days)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


def stage0_reason(core: dict, quote: Optional[dict], asof: str, cfg: dict) -> Optional[str]:
    u = cfg.get("universe") or {}
    spot = None
    if quote:
        spot = to_float(quote.get("last"))
    if spot is None:
        spot = to_float(core.get("px"))
    if spot is None:
        return "missing_spot"
    lo = float(u.get("min_price") or 15)
    hi_raw = u.get("max_price")
    hi = float(hi_raw) if hi_raw not in (None, "", 0, 0.0) else None
    if spot < lo or (hi is not None and spot > hi):
        return "price_range"
    vol = to_float(core.get("avg_opt_vol_20d"))
    if vol is None:
        return "DATA UNAVAILABLE option_volume"
    if vol < float(u.get("min_option_volume") or 500):
        return "option_volume"
    mcap = to_float(core.get("mkt_cap"))
    if mcap is None:
        return "DATA UNAVAILABLE mkt_cap"
    if mcap < float(u.get("min_market_cap_mm") or 5000):
        return "mkt_cap"
    borrow = to_float(core.get("borrow30"))
    if borrow is not None and borrow > float(u.get("max_borrow") or 25):
        return "borrow"
    earn = earnings_days(core, asof)
    close_days = int((cfg.get("management") or {}).get("earnings_close_days") or 7)
    if earn is not None and 0 <= earn <= close_days:
        return "earnings_%sd" % earn
    return None


def score_quality(
    core: dict,
    quote: Optional[dict],
    bars: Optional[List[dict]],
    yf: Optional[dict],
    cfg: dict,
    asof: str,
) -> QualityScore:
    q = (cfg.get("scoring") or {}).get("quality") or {}
    qs = QualityScore()
    mcap = to_float(core.get("mkt_cap")) or 0.0
    qs.size_score = tier_score(mcap, q.get("mcap_excellent", 50000), q.get("mcap_good", 20000), q.get("mcap_fair", 5000))
    borrow = to_float(core.get("borrow30"))
    if borrow is None:
        qs.borrow_score = 50
    else:
        qs.borrow_score = tier_score(
            borrow,
            q.get("borrow_excellent", 2),
            q.get("borrow_good", 8),
            q.get("borrow_fair", 15),
            lower_is_better=True,
        )
    mr = compute_mean_reversion(bars or [])
    if mr is not None:
        qs.path_score = tier_score(mr, 75, 50, 25)
    else:
        chg = to_float(core.get("chg_1m"))
        if chg is None:
            qs.path_score = 50
        else:
            qs.path_score = tier_score(
                chg,
                q.get("path_excellent", 0),
                q.get("path_good", -10),
                q.get("path_fair", -25),
            )
    beta = to_float(core.get("beta1y"))
    if beta is None:
        qs.beta_score = 50
    else:
        qs.beta_score = tier_score(
            abs(beta),
            q.get("beta_excellent", 1.3),
            q.get("beta_good", 1.8),
            q.get("beta_fair", 2.5),
            lower_is_better=True,
        )
    earn = earnings_days(core, asof)
    if earn is None:
        qs.earnings_score = 50
    else:
        qs.earnings_score = tier_score(
            earn,
            q.get("earn_excellent", 45),
            q.get("earn_good", 21),
            q.get("earn_fair", 14),
        )
    source = str((cfg.get("universe") or {}).get("source") or "own_list").lower()
    growth_book = source in ("own_list", "own")
    if growth_book:
        qs.div_score = 50
        qs.pe_score = 50
    else:
        div = to_float(core.get("div_yield"))
        if quote and to_float(quote.get("div_yield")) is not None:
            div = to_float(quote.get("div_yield"))
        if div is None:
            qs.div_score = 25
        else:
            qs.div_score = tier_score(div, q.get("div_excellent", 3), q.get("div_good", 1.5), q.get("div_fair", 0.5))
        pe = None
        if quote:
            pe = to_float(quote.get("pe_ratio"))
        if pe is None or pe <= 0:
            qs.pe_score = 50
        else:
            qs.pe_score = tier_score(
                pe,
                q.get("pe_excellent", 15),
                q.get("pe_good", 25),
                q.get("pe_fair", 40),
                lower_is_better=True,
            )
    conf = to_float(core.get("confidence"))
    if conf is None:
        qs.confidence_score = 50
    else:
        qs.confidence_score = tier_score(
            conf,
            q.get("conf_excellent", 80),
            q.get("conf_good", 60),
            q.get("conf_fair", 40),
        )
    qs.composite = (
        q.get("size_weight", 0.22) * qs.size_score
        + q.get("borrow_weight", 0.10) * qs.borrow_score
        + q.get("path_weight", 0.22) * qs.path_score
        + q.get("beta_weight", 0.08) * qs.beta_score
        + q.get("earnings_weight", 0.18) * qs.earnings_score
        + q.get("div_weight", 0.05) * qs.div_score
        + q.get("pe_weight", 0.05) * qs.pe_score
        + q.get("confidence_weight", 0.10) * qs.confidence_score
    )
    if yf and isinstance(yf, dict) and yf.get("ok"):
        roe = to_float(yf.get("roe"))
        de = to_float(yf.get("debt_equity"))
        fcf = to_float(yf.get("fcf_yield"))
        bonus = 0.0
        n = 0
        if roe is not None:
            bonus += tier_score(roe, 15, 10, 5)
            n += 1
        if de is not None:
            bonus += tier_score(de, 0.5, 1.0, 2.0, lower_is_better=True)
            n += 1
            if de > 3.0:
                qs.disqualified = True
                qs.disqualify_reason = "yf D/E %.2f > 3" % de
        if fcf is not None:
            bonus += tier_score(fcf, 5, 3, 1)
            n += 1
        if n:
            yf_comp = bonus / n
            qs.composite = 0.8 * qs.composite + 0.2 * yf_comp
        qs.yfinance_note = "yfinance_ok"
    elif yf and isinstance(yf, dict) and yf.get("error"):
        qs.yfinance_note = "yfinance_unavailable"
    else:
        qs.yfinance_note = "yfinance_skipped"
    return qs


def _best_expiry_rows(rows: List[dict], target_dte: int) -> Tuple[List[dict], int, str]:
    by_exp = {}
    for row in rows or []:
        dte = to_float(row.get("dte"))
        exp = str(row.get("expirDate") or "")[:10]
        if dte is None:
            continue
        by_exp.setdefault(exp, {"dte": int(dte), "rows": []})
        by_exp[exp]["rows"].append(row)
    if not by_exp:
        return [], 0, ""
    best = min(by_exp.items(), key=lambda kv: abs(kv[1]["dte"] - target_dte))
    return best[1]["rows"], best[1]["dte"], best[0]


def _closest_strike(
    rows: List[dict],
    target: float,
    side: str,
    spot: Optional[float] = None,
    max_spread: float = 0.20,
    otm_min_pct: float = 0.02,
    otm_max_pct: float = 0.15,
) -> Optional[dict]:
    bid_key = "putBidPrice" if side == "put" else "callBidPrice"
    ask_key = "putAskPrice" if side == "put" else "callAskPrice"
    eligible = []
    for row in rows:
        strike = to_float(row.get("strike"))
        bid = to_float(row.get(bid_key))
        ask = to_float(row.get(ask_key))
        if strike is None or bid is None or bid <= 0:
            continue
        if side == "put" and spot is not None:
            if strike > spot * (1.0 - otm_min_pct) or strike < spot * (1.0 - otm_max_pct):
                continue
        if side == "call" and spot is not None:
            if strike < spot * (1.0 + otm_min_pct) or strike > spot * (1.0 + otm_max_pct):
                continue
        mid = (bid + (ask if ask is not None else bid)) / 2.0
        spread = ((ask - bid) / mid) if (ask is not None and mid > 0) else 0.0
        eligible.append((spread, abs(strike - target), row))
    under = [e for e in eligible if e[0] <= max_spread]
    pool = under or []
    if not pool:
        return None
    pool.sort(key=lambda e: (e[1], e[0]))
    return pool[0][2]


def score_premium(rows: List[dict], core: dict, cfg: dict) -> PremiumScore:
    p = (cfg.get("scoring") or {}).get("premium") or {}
    u = cfg.get("universe") or {}
    mgmt = cfg.get("management") or {}
    ps = PremiumScore()
    spot = to_float(core.get("px"))
    iv = core.get("iv30_dec") or iv_decimal(core.get("iv30"))
    if spot is None or iv is None or not rows:
        ps.rejected = True
        ps.reject_reason = "DATA UNAVAILABLE strikes" if not rows else "DATA UNAVAILABLE spot/iv"
        return ps
    target_dte = int(mgmt.get("dte_target") or 30)
    sigma = float(mgmt.get("sigma_otm") or 1.0)
    exp_rows, dte, expiry = _best_expiry_rows(rows, target_dte)
    if not exp_rows or dte <= 0:
        ps.rejected = True
        ps.reject_reason = "no_expiry"
        return ps
    ps.dte = dte
    ps.expiry = expiry
    max_spread = float(u.get("max_spread_pct") or 0.20)
    otm_min = float(mgmt.get("csp_otm_min_pct") or 0.02)
    otm_max = float(mgmt.get("csp_otm_max_pct") or 0.15)
    put_target = compute_sigma_strike(spot, iv, dte, side="put", sigma=sigma)
    call_target = compute_sigma_strike(spot, iv, dte, side="call", sigma=sigma)
    put_row = _closest_strike(
        exp_rows, put_target, "put", spot=spot, max_spread=max_spread, otm_min_pct=otm_min, otm_max_pct=otm_max,
    )
    call_row = _closest_strike(
        exp_rows, call_target, "call", spot=spot, max_spread=max_spread, otm_min_pct=otm_min, otm_max_pct=otm_max,
    )
    if put_row is None:
        ps.rejected = True
        ps.reject_reason = "no_put_bid"
        return ps
    ps.csp_strike = to_float(put_row.get("strike")) or 0.0
    ps.csp_bid = to_float(put_row.get("putBidPrice")) or 0.0
    ps.csp_ask = to_float(put_row.get("putAskPrice")) or 0.0
    ps.csp_premium = ps.csp_bid
    if call_row is not None:
        ps.cc_strike = to_float(call_row.get("strike")) or 0.0
        ps.cc_bid = to_float(call_row.get("callBidPrice")) or 0.0
        ps.cc_ask = to_float(call_row.get("callAskPrice")) or 0.0
        ps.cc_premium = ps.cc_bid
    if ps.csp_strike > 0 and ps.csp_premium > 0:
        ps.csp_yield_ann = (ps.csp_premium / ps.csp_strike) * (365.0 / dte) * 100.0
    if spot > 0 and ps.cc_premium > 0:
        ps.cc_yield_ann = (ps.cc_premium / spot) * (365.0 / dte) * 100.0
    mid = None
    if ps.csp_bid and ps.csp_ask:
        mid = (ps.csp_bid + ps.csp_ask) / 2.0
    if mid and mid > 0:
        ps.spread_pct = (ps.csp_ask - ps.csp_bid) / mid
    if ps.spread_pct and ps.spread_pct > max_spread:
        ps.rejected = True
        ps.reject_reason = "wide_spread"
        return ps
    if ps.csp_premium <= 0:
        ps.rejected = True
        ps.reject_reason = "no_put_bid"
        return ps
    min_credit = float(mgmt.get("min_csp_credit_pct") or 0.015)
    if ps.csp_strike > 0 and (ps.csp_premium / ps.csp_strike) < min_credit:
        ps.rejected = True
        ps.reject_reason = "credit_too_small"
        return ps
    min_iv_hv = to_float(mgmt.get("min_iv_hv"))
    iv_hv = to_float(core.get("iv_hv"))
    ivr = to_float(core.get("iv_pctile_1y"))
    if min_iv_hv is not None and iv_hv is not None and iv_hv < min_iv_hv:
        if ivr is None or ivr < 50:
            ps.rejected = True
            ps.reject_reason = "cheap_vol"
            return ps
    ivr = to_float(core.get("iv_pctile_1y")) or 0.0
    ps.iv_rank = ivr
    ps.csp_yield_score = _yield_score(ps.csp_yield_ann, p, "csp")
    ps.cc_yield_score = _yield_score(ps.cc_yield_ann, p, "cc")
    ps.iv_rank_score = tier_score(ivr, p.get("ivr_excellent", 60), p.get("ivr_good", 40), p.get("ivr_fair", 20))
    spread_pct_pts = (ps.spread_pct or 0.0) * 100.0
    ps.spread_score = tier_score(
        spread_pct_pts,
        p.get("spread_excellent", 2),
        p.get("spread_good", 4),
        p.get("spread_fair", 6),
        lower_is_better=True,
    )
    ps.composite = (
        p.get("csp_yield_weight", 0.35) * ps.csp_yield_score
        + p.get("cc_yield_weight", 0.25) * ps.cc_yield_score
        + p.get("ivr_weight", 0.20) * ps.iv_rank_score
        + p.get("spread_weight", 0.20) * ps.spread_score
    )
    return ps


def _yield_score(ann: float, p: dict, kind: str) -> int:
    if kind == "csp":
        if ann >= p.get("csp_excellent", 40):
            return 100
        if ann >= p.get("csp_good", 30):
            return 85
        if ann >= p.get("csp_fair", 20):
            return 70
        if ann >= p.get("csp_low", 10):
            return 50
        return 25
    if ann >= p.get("cc_excellent", 30):
        return 100
    if ann >= p.get("cc_good", 20):
        return 85
    if ann >= p.get("cc_fair", 10):
        return 70
    if ann >= p.get("cc_low", 5):
        return 50
    return 25


def apply_sentiment(core: dict, hot: Optional[dict], asof: str, cfg: dict) -> SentimentAdjustment:
    s = cfg.get("sentiment") or {}
    sa = SentimentAdjustment()
    adj = 0.0
    iv_hv = to_float(core.get("iv_hv"))
    if iv_hv is not None and iv_hv > 1.1:
        adj += float(s.get("iv_hv_rich") or 3)
        sa.notes.append("IV/HV rich")
    elif iv_hv is not None and iv_hv < 0.9:
        adj += float(s.get("iv_hv_cheap") or -3)
        sa.notes.append("IV/HV cheap")
    ivr = to_float(core.get("iv_pctile_1y"))
    if ivr is not None and ivr >= 60:
        adj += float(s.get("ivr_boost") or 2)
        sa.notes.append("IV rank high")
    cvol = to_float(core.get("c_vol")) or 0.0
    pvol = to_float(core.get("p_vol")) or 0.0
    tot = cvol + pvol
    if tot > 0 and pvol / tot >= 0.6:
        adj += float(s.get("put_heavy") or -2)
        sa.notes.append("put-heavy volume")
    earn = earnings_days(core, asof)
    if earn is not None and 8 <= earn <= 14:
        adj += float(s.get("earnings_14d") or -5)
        sa.notes.append("earnings %sd" % earn)
    if hot:
        sa.x_status = str(hot.get("tag") or hot.get("bias") or "present")
        bias = str(hot.get("bias") or "unknown").lower()
        tag = str(hot.get("tag") or "").lower()
        if "crowd" in tag:
            adj += float(s.get("x_bearish") or -5)
            sa.notes.append("X crowded")
        elif bias == "bullish":
            adj += float(s.get("x_bullish") or 3)
            sa.notes.append("X bullish")
        elif bias == "bearish":
            adj += float(s.get("x_bearish") or -5)
            sa.notes.append("X bearish")
    else:
        sa.x_status = "DATA UNAVAILABLE"
    cap = float(s.get("max_adjustment") or 10)
    sa.total = max(-cap, min(cap, adj))
    return sa


def compute_composite(quality: float, premium: float, sentiment: float, qw: float, pw: float) -> float:
    return quality * qw + premium * pw + sentiment


def assign_tier(composite: float, cfg: dict) -> str:
    a = cfg.get("allocation") or {}
    if composite >= float(a.get("min_composite_core") or 60):
        return "core"
    if composite >= float(a.get("min_composite_aggressive") or 45):
        return "aggressive"
    if composite >= float(a.get("min_composite_watchlist") or 35):
        return "watchlist"
    return "excluded"


def allocate_capital(candidates: List[dict], capital: float, cfg: dict) -> List[dict]:
    """Rank TRADE CSPs. User sizes cash. Do not drop a name because 1 lot exceeds a sleeve."""
    a = cfg.get("allocation") or {}
    max_pos = int(a.get("max_positions") or 5)
    ranked = sorted(
        candidates,
        key=lambda c: (
            c.get("conf") if c.get("conf") is not None else (c.get("composite") or 0),
            c.get("credit_pct") or 0,
        ),
        reverse=True,
    )
    taken = 0
    sector_count: Dict[str, int] = {}
    max_per_sector = int(a.get("max_per_sector") or 1)
    out = []
    for cand in ranked:
        if cand.get("tier") == "excluded":
            cand["allocated"] = False
            out.append(cand)
            continue
        if cand.get("conf_label") != "TRADE":
            cand["allocated"] = False
            out.append(cand)
            continue
        strike = to_float((cand.get("premium") or {}).get("csp_strike")) or to_float(cand.get("csp_strike"))
        has_ticket = (strike is not None and strike > 0) or float(cand.get("capital_required") or 0) > 0
        sector = str(cand.get("sector") or "unknown").split()[0].lower() or "unknown"
        if not has_ticket or taken >= max_pos:
            cand["allocated"] = False
            out.append(cand)
            continue
        if sector_count.get(sector, 0) >= max_per_sector:
            cand["allocated"] = False
            out.append(cand)
            continue
        cand["allocated"] = True
        taken += 1
        sector_count[sector] = sector_count.get(sector, 0) + 1
        out.append(cand)
    return out
