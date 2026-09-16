#!/usr/bin/env python3
"""Ticker quotes and option chains via tradedesk Schwab token + .env. No invented prints."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.append(str(_SCRIPT_DIR))


def tradedesk_root() -> Path:
    env = os.environ.get("UW_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for cand in (
        Path("/Users/anuppamvi/tradedesk"),
        Path.cwd(),
        *here.parents,
    ):
        if (cand / "uwos" / "schwab_auth.py").exists() and (cand / ".env").exists():
            return cand.resolve()
    raise SystemExit("tradedesk root not found (need uwos/schwab_auth.py and .env). Set UW_ROOT.")


def boot() -> Path:
    root = tradedesk_root()
    os.environ["UW_ROOT"] = str(root)
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Mega-cap chains time out at the 12s uwos default during RTH.
    os.environ.setdefault("UWOS_SCHWAB_OPTION_CHAIN_TIMEOUT_SECONDS", "45")
    return root


def fnum(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def contract_row(expiry: str, strike: Optional[float], c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": c.get("symbol"),
        "expiry": expiry,
        "strike": fnum(c.get("strikePrice")) or strike,
        "put_call": c.get("putCall") or c.get("right"),
        "bid": fnum(c.get("bid")),
        "ask": fnum(c.get("ask")),
        "last": fnum(c.get("last")),
        "mark": fnum(c.get("mark")),
        "delta": fnum(c.get("delta")),
        "theta": fnum(c.get("theta")),
        "iv": fnum(c.get("volatility")),
        "oi": fnum(c.get("openInterest")),
        "volume": fnum(c.get("totalVolume")),
        "bid_size": fnum(c.get("bidSize")),
        "ask_size": fnum(c.get("askSize")),
        "quote_time_ms": fnum(c.get("quoteTimeInLong") or c.get("quoteTime")),
    }


def iter_contracts(chain: Dict[str, Any]) -> List[Tuple[str, Optional[float], Dict[str, Any]]]:
    from uwos.schwab_auth import _iter_contracts

    out: List[Tuple[str, Optional[float], Dict[str, Any]]] = []
    for exp, strike, c in _iter_contracts(chain.get("callExpDateMap") or {}):
        out.append((exp, strike, c))
    for exp, strike, c in _iter_contracts(chain.get("putExpDateMap") or {}):
        out.append((exp, strike, c))
    return out


def find_leg(
    chain: Dict[str, Any], *, expiry: str, strike: float, right: str
) -> Optional[Dict[str, Any]]:
    from uwos.schwab_auth import _iter_contracts

    want = right[:1].upper()
    target = float(strike)
    exp = expiry[:10]
    mmap = chain.get("callExpDateMap") if want == "C" else chain.get("putExpDateMap")
    for e, s, c in _iter_contracts(mmap or {}):
        if str(e)[:10] != exp:
            continue
        if s is None or abs(float(s) - target) > 0.051:
            continue
        return contract_row(str(e)[:10], s, c)
    return None


def atm_straddle(
    chain: Dict[str, Any], spot: Optional[float], expiry: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if not spot:
        return None
    from uwos.schwab_auth import _iter_contracts

    want = (expiry or "")[:10]

    def nearest(mmap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows = []
        for e, s, c in _iter_contracts(mmap or {}):
            if want and str(e)[:10] != want:
                continue
            rows.append((e, s, c))
        if not rows:
            return None
        e, s, c = min(rows, key=lambda x: abs((x[1] if x[1] is not None else spot) - spot))
        return contract_row(str(e)[:10], s, c)

    call = nearest(chain.get("callExpDateMap") or {})
    put = nearest(chain.get("putExpDateMap") or {})
    if not call or not put:
        return None
    call_ask = call.get("ask")
    put_ask = put.get("ask")
    width = None
    if call_ask is not None and put_ask is not None:
        width = call_ask + put_ask
    return {"call": call, "put": put, "straddle_ask": width, "expiry": call.get("expiry")}


def quote_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    from uwos.schwab_auth import extract_quote_fields

    out = {}
    for sym, raw in payload.items():
        last, bid, ask = extract_quote_fields(raw if isinstance(raw, dict) else {})
        body = raw.get("quote", raw) if isinstance(raw, dict) else {}
        out[sym] = {
            "last": last,
            "bid": bid,
            "ask": ask,
            "mark": fnum(body.get("mark")),
            "change_pct": fnum(body.get("netPercentChangeInDouble") or body.get("netPercentChange")),
            "quote_time": body.get("quoteTime") or body.get("quoteTimeInLong"),
        }
    return out


def vertical_math(*, kind: str, short: Dict[str, Any], long: Dict[str, Any]) -> Dict[str, Any]:
    sb, sa = short.get("bid"), short.get("ask")
    lb, la = long.get("bid"), long.get("ask")
    ss, ls = short.get("strike"), long.get("strike")
    if None in (sb, sa, lb, la, ss, ls):
        return {"ok": False, "reason": "missing bid/ask or strike on a leg"}
    width = abs(float(ss) - float(ls))
    if width <= 0:
        return {"ok": False, "reason": "zero width"}
    if kind == "credit":
        conservative = float(sb) - float(la)
        mid = ((float(sb) + float(sa)) / 2) - ((float(lb) + float(la)) / 2)
        frac = conservative / width if width else None
        max_loss = (width - conservative) * 100
        return {
            "ok": conservative > 0,
            "kind": "credit",
            "net": round(conservative, 4),
            "mid_net": round(mid, 4),
            "width": width,
            "credit_width": round(frac, 4) if frac is not None else None,
            "max_loss_1lot": round(max_loss, 2),
            "max_profit_1lot": round(conservative * 100, 2),
            "worse_fill": abs(mid - conservative) > 0.05 * abs(mid) if mid else False,
        }
    conservative = float(la) - float(sb)
    mid = ((float(lb) + float(la)) / 2) - ((float(sb) + float(sa)) / 2)
    frac = conservative / width if width else None
    return {
        "ok": conservative > 0,
        "kind": "debit",
        "net": round(conservative, 4),
        "mid_net": round(mid, 4),
        "width": width,
        "debit_width": round(frac, 4) if frac is not None else None,
        "max_loss_1lot": round(conservative * 100, 2),
        "max_profit_1lot": round((width - conservative) * 100, 2) if width > conservative else 0.0,
        "worse_fill": abs(mid - conservative) > 0.05 * abs(mid) if mid else False,
    }


def service():
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

    cfg = SchwabAuthConfig.from_env(load_dotenv_file=True)
    return SchwabLiveDataService(cfg, interactive_login=False), cfg


def cmd_quote(args: argparse.Namespace) -> Dict[str, Any]:
    from uwos.schwab_auth import normalize_symbols

    svc, cfg = service()
    symbols = normalize_symbols(args.symbols)
    raw = svc.get_quotes(symbols)
    return {
        "source": "schwab",
        "token_path": cfg.token_path,
        "auth_mode": svc.auth_mode,
        "quotes": quote_view(raw),
    }


def cmd_chain(args: argparse.Namespace) -> Dict[str, Any]:
    svc, cfg = service()
    payload = svc.get_option_chain(
        args.symbol.upper(),
        strike_count=args.strike_count,
        include_underlying_quote=True,
        from_date=args.from_date or None,
        to_date=args.to_date or None,
    )
    summary = svc.summarize_option_chain(args.symbol.upper(), payload)
    spot = summary.get("underlying_price")
    rows = [contract_row(e, s, c) for e, s, c in iter_contracts(payload)]
    if args.right:
        want = args.right[:1].upper()
        rows = [r for r in rows if str(r.get("put_call") or "").upper()[:1] == want]
    if args.expiry:
        rows = [r for r in rows if str(r.get("expiry"))[:10] == args.expiry[:10]]
    return {
        "source": "schwab",
        "token_path": cfg.token_path,
        "auth_mode": svc.auth_mode,
        "symbol": args.symbol.upper(),
        "summary": summary,
        "atm_straddle": atm_straddle(payload, spot, expiry=args.expiry or None),
        "contracts": rows,
    }


def cmd_vertical(args: argparse.Namespace) -> Dict[str, Any]:
    svc, cfg = service()
    right = args.right[:1].upper()
    payload = svc.get_option_chain(
        args.symbol.upper(),
        strike_count=args.strike_count,
        include_underlying_quote=True,
        from_date=args.expiry,
        to_date=args.expiry,
    )
    short = find_leg(payload, expiry=args.expiry, strike=args.short, right=right)
    long = find_leg(payload, expiry=args.expiry, strike=args.long, right=right)
    if not short or not long:
        nearby = []
        from uwos.schwab_auth import _iter_contracts

        mmap = payload.get("callExpDateMap") if right == "C" else payload.get("putExpDateMap")
        for e, s, c in _iter_contracts(mmap or {}):
            if str(e)[:10] == args.expiry[:10]:
                nearby.append(contract_row(str(e)[:10], s, c))
        return {
            "ok": False,
            "reason": "leg missing on Schwab chain",
            "short": short,
            "long": long,
            "nearby": nearby[:40],
            "source": "schwab",
            "token_path": cfg.token_path,
        }
    math = vertical_math(kind=args.kind, short=short, long=long)
    summary = svc.summarize_option_chain(args.symbol.upper(), payload)
    priced = {"ok": bool(math.get("ok")), "pricing": math, "kind": args.kind, "short": short, "long": long}
    _stamp_pd(priced, short=short, long=long)
    return {
        "source": "schwab",
        "token_path": cfg.token_path,
        "auth_mode": svc.auth_mode,
        "symbol": args.symbol.upper(),
        "expiry": args.expiry,
        "right": right,
        "kind": args.kind,
        "short": short,
        "long": long,
        "pricing": math,
        "pd": priced.get("pd"),
        "pd_n": priced.get("pd_n"),
        "r_cons": priced.get("r_cons"),
        "pd_l": priced.get("pd_l"),
        "pd_reason": priced.get("pd_reason"),
        "underlying_price": summary.get("underlying_price"),
        "atm_straddle": atm_straddle(payload, summary.get("underlying_price"), expiry=args.expiry),
    }


def _stamp_pd(row: Optional[Dict[str, Any]], *, short=None, long=None) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict) or not row.get("ok"):
        return row
    from pd_rank import attach_structure_pd

    payload = dict(row)
    if short is not None:
        payload["short_leg"] = short
        if payload.get("quote_time_ms") is None:
            payload["quote_time_ms"] = short.get("quote_time_ms")
    if long is not None:
        payload["long_leg"] = long
        if payload.get("quote_time_ms") is None:
            payload["quote_time_ms"] = long.get("quote_time_ms")
    attach_structure_pd(payload)
    for key in ("pd", "pd_n", "r_cons", "pd_l", "pd_reason", "pd_s", "n_s", "r_cons_s", "l_s"):
        if key in payload:
            row[key] = payload[key]
    return row


def _index_legs(rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for r in rows:
        k = r.get("strike")
        if k is None:
            continue
        out[float(k)] = r
    return out


ALLOWED_WIDTHS = (5.0, 10.0, 2.5, 15.0)
# 1-lot credit floor so we do not print $60 wings when a $10-wide still clears.
CREDIT_DOLLAR_FLOOR = 1.00  # $100
# If credit/width is this much worse, do not take extra width for extra dollars.
FRAC_EPS = 0.015
HARD_DELTA = 0.25

# Live 2026-09-09..15: Normal 0.20 + (0.20Δ OR 0.90σ) = empty set across
# 12 names × 5 expiries. Same class as old Calm 25%+1σ. Cheap-vol Shield
# stays 0.12 / 0.22Δ / 0.80σ until conservative 0.20 actually prints.
REGIME_GATES = {
    "calm": {"max_delta": 0.22, "min_sigma": 0.80, "min_frac": 0.12, "or_delta": None},
    "normal": {"max_delta": 0.22, "min_sigma": 0.80, "min_frac": 0.12, "or_delta": None},
    "elevated": {"max_delta": 0.25, "min_sigma": 1.00, "min_frac": 0.25, "or_delta": None},
    "crisis": {"max_delta": 0.0, "min_sigma": 99.0, "min_frac": 1.0, "or_delta": None},
}

SCAN_UNIVERSE = [
    "AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META", "MSFT", "NVDA", "TSLA",
    "DELL", "HD", "NFLX", "UNH", "CRM", "ORCL", "INTC", "GE", "WMT", "ADBE",
    "SNOW", "PANW", "COST", "JPM", "XOM", "CVX", "COP",
]


def vix_regime(vix: Optional[float]) -> str:
    if vix is None:
        return "normal"
    if vix < 16:
        return "calm"
    if vix < 22:
        return "normal"
    if vix < 30:
        return "elevated"
    return "crisis"


def resolve_gates(regime: str, *, max_delta=None, min_sigma=None, min_frac=None, or_delta=None) -> Dict[str, Any]:
    key = (regime or "calm").strip().lower()
    if key not in REGIME_GATES:
        key = "calm"
    g = dict(REGIME_GATES[key])
    g["regime"] = key
    g["hard_delta"] = HARD_DELTA
    if max_delta is not None:
        g["max_delta"] = float(max_delta)
    if min_sigma is not None:
        g["min_sigma"] = float(min_sigma)
    if min_frac is not None:
        g["min_frac"] = float(min_frac)
    if or_delta is not None:
        g["or_delta"] = float(or_delta)
    return g


def short_clears(*, delta: Optional[float], otm: Optional[float], sigma: Optional[float], gates: Dict[str, Any]) -> bool:
    """Regime short-strike rule. or_delta set → (Δ≤or_delta OR σ≥min_sigma). Else AND."""
    if delta is None or otm is None or sigma is None or float(sigma) <= 0 or float(otm) <= 0:
        return False
    absd = abs(float(delta))
    hard = float(gates.get("hard_delta") or HARD_DELTA)
    if absd > hard:
        return False
    smult = float(otm) / float(sigma)
    min_sigma = float(gates.get("min_sigma") or 0)
    or_delta = gates.get("or_delta")
    if or_delta is not None:
        return absd <= float(or_delta) or smult >= min_sigma
    return absd <= float(gates.get("max_delta") or 0) and smult >= min_sigma


def friday_expiries(asof: date, *, min_dte: int = 14, max_dte: int = 60) -> List[str]:
    out: List[str] = []
    d = asof + timedelta(days=1)
    end = asof + timedelta(days=int(max_dte))
    while d <= end:
        if d.weekday() == 4:
            dte = (d - asof).days
            if dte >= int(min_dte):
                out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _credit_better(cand: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    """Winning credits: edge (credit/width) first, then dollars. Do not 15-wide a 12% scrap."""
    if best is None:
        return True
    c_math = cand["pricing"]
    b_math = best["pricing"]
    c_frac = float(c_math.get("credit_width") or 0)
    b_frac = float(b_math.get("credit_width") or 0)
    c_net = float(c_math.get("net") or 0)
    b_net = float(b_math.get("net") or 0)
    c_floor = c_net >= CREDIT_DOLLAR_FLOOR
    b_floor = b_net >= CREDIT_DOLLAR_FLOOR
    if c_floor and not b_floor:
        return True
    if b_floor and not c_floor:
        return False
    if c_frac > b_frac + FRAC_EPS:
        return True
    if b_frac > c_frac + FRAC_EPS:
        return False
    return c_net > b_net + 0.01


def _listed_pair(ks: List[float], short: float, farther: str) -> List[float]:
    """Only adjacent listed widths. Do not pair a 245 short with a 100 long."""
    out = []
    for w in ALLOWED_WIDTHS:
        target = short - w if farther == "down" else short + w
        for k in ks:
            if abs(k - target) <= 0.051:
                if farther == "down" and k < short:
                    out.append(k)
                if farther == "up" and k > short:
                    out.append(k)
    return out


def _credit_put(puts: Dict[float, Dict[str, Any]], spot: float, sigma: float, gates: Dict[str, Any]) -> Dict[str, Any]:
    ks = sorted(puts)
    best = None
    tried = 0
    min_frac = float(gates.get("min_frac") or 0)
    for sh in ks:
        short = puts[sh]
        dlt = short.get("delta")
        otm = spot - sh
        if not short_clears(delta=dlt, otm=otm, sigma=sigma, gates=gates):
            continue
        for lo in _listed_pair(ks, sh, "down"):
            long = puts[lo]
            math = vertical_math(kind="credit", short=short, long=long)
            tried += 1
            if not math.get("ok"):
                continue
            frac = math.get("credit_width") or 0
            if frac < min_frac:
                continue
            cand = {
                "ok": True,
                "action": "Sell put credit",
                "short": sh,
                "long": lo,
                "short_delta": dlt,
                "pricing": math,
                "otm": otm,
                "sigma_mult": otm / sigma if sigma else None,
            }
            _stamp_pd(cand, short=short, long=long)
            if _credit_better(cand, best):
                best = cand
    return best or {"ok": False, "reason": "no put credit met delta/sigma/width", "tried": tried}


def _credit_call(calls: Dict[float, Dict[str, Any]], spot: float, sigma: float, gates: Dict[str, Any]) -> Dict[str, Any]:
    ks = sorted(calls)
    best = None
    tried = 0
    min_frac = float(gates.get("min_frac") or 0)
    for sh in ks:
        short = calls[sh]
        dlt = short.get("delta")
        otm = sh - spot
        if not short_clears(delta=dlt, otm=otm, sigma=sigma, gates=gates):
            continue
        for hi in _listed_pair(ks, sh, "up"):
            long = calls[hi]
            math = vertical_math(kind="credit", short=short, long=long)
            tried += 1
            if not math.get("ok"):
                continue
            frac = math.get("credit_width") or 0
            if frac < min_frac:
                continue
            cand = {
                "ok": True,
                "action": "Sell call credit",
                "short": sh,
                "long": hi,
                "short_delta": dlt,
                "pricing": math,
                "otm": otm,
                "sigma_mult": otm / sigma if sigma else None,
            }
            _stamp_pd(cand, short=short, long=long)
            if _credit_better(cand, best):
                best = cand
    return best or {"ok": False, "reason": "no call credit met delta/sigma/width", "tried": tried}


def _debit_vertical(legs: Dict[float, Dict[str, Any]], *, right: str, spot: float) -> Dict[str, Any]:
    ks = sorted(legs)
    best = None
    action = "Buy call debit" if right == "C" else "Buy put debit"
    farther = "up" if right == "C" else "down"
    for sh in ks:
        for farther_k in _listed_pair(ks, sh, farther):
            long_k, short_k = (sh, farther_k) if right == "C" else (sh, farther_k)
            # Call debit: long lower (closer ATM), short higher. Put debit: long higher, short lower.
            if right == "C":
                long_k, short_k = sh, farther_k
            else:
                long_k, short_k = sh, farther_k
            long, short = legs[long_k], legs[short_k]
            ld = long.get("delta")
            if ld is None:
                continue
            if abs(float(ld)) < 0.28 or abs(float(ld)) > 0.55:
                continue
            math = vertical_math(kind="debit", short=short, long=long)
            if not math.get("ok"):
                continue
            frac = math.get("debit_width") or 1
            # <0.25 of width is a lottery long, not a vertical. >0.55 has no convexity.
            if frac < 0.25 or frac > 0.55:
                continue
            cand = {
                "ok": True,
                "action": action,
                "long": long_k,
                "short": short_k,
                "long_delta": ld,
                "pricing": math,
            }
            _stamp_pd(cand, short=short, long=long)
            score = abs(abs(float(ld)) - 0.40)
            if best is None or score < best["_s"]:
                cand["_s"] = score
                best = cand
    if not best:
        return {"ok": False, "reason": f"no {right} debit with long |delta| 0.28–0.55"}
    best.pop("_s", None)
    return best


def cmd_structures(args: argparse.Namespace) -> Dict[str, Any]:
    svc, cfg = service()
    payload = svc.get_option_chain(
        args.symbol.upper(),
        strike_count=args.strike_count,
        include_underlying_quote=True,
        from_date=args.expiry,
        to_date=args.expiry,
    )
    summary = svc.summarize_option_chain(args.symbol.upper(), payload)
    spot = summary.get("underlying_price")
    atm = atm_straddle(payload, spot, expiry=args.expiry)
    sigma = (atm or {}).get("straddle_ask")
    puts, calls = [], []
    for e, s, c in iter_contracts(payload):
        if str(e)[:10] != args.expiry[:10]:
            continue
        row = contract_row(str(e)[:10], s, c)
        right = str(row.get("put_call") or "")[:1].upper()
        if right == "P":
            puts.append(row)
        elif right == "C":
            calls.append(row)
    if spot is None or sigma is None:
        return {
            "ok": False,
            "reason": "missing spot or ATM straddle ask",
            "symbol": args.symbol.upper(),
            "expiry": args.expiry,
            "underlying_price": spot,
            "atm_straddle": atm,
            "source": "schwab",
        }
    pmap, cmap = _index_legs(puts), _index_legs(calls)
    gates = resolve_gates(
        getattr(args, "regime", None) or "calm",
        max_delta=getattr(args, "max_delta", None),
        min_sigma=getattr(args, "min_sigma", None),
        min_frac=getattr(args, "min_frac", None),
        or_delta=getattr(args, "or_delta", None),
    )
    put_c = _credit_put(pmap, float(spot), float(sigma), gates)
    call_c = _credit_call(cmap, float(spot), float(sigma), gates)
    condor = {"ok": False, "reason": "need both credit sides"}
    if isinstance(put_c, dict) and put_c.get("ok") and isinstance(call_c, dict) and call_c.get("ok"):
        pnet = float((put_c.get("pricing") or {}).get("net") or 0)
        cnet = float((call_c.get("pricing") or {}).get("net") or 0)
        if pnet >= CREDIT_DOLLAR_FLOOR and cnet >= CREDIT_DOLLAR_FLOOR:
            pwidth = float((put_c.get("pricing") or {}).get("width") or 0)
            cwidth = float((call_c.get("pricing") or {}).get("width") or 0)
            wing = max(pwidth, cwidth)
            total = pnet + cnet
            condor = {
                "ok": True,
                "action": "Sell iron condor",
                "put": put_c,
                "call": call_c,
                "pricing": {
                    "kind": "credit",
                    "net": total,
                    "width": wing,
                    "max_loss_1lot": round(max(0.0, wing - total) * 100, 2),
                    "max_profit_1lot": round(total * 100, 2),
                },
            }
            _stamp_pd(condor)
        else:
            condor = {"ok": False, "reason": "a wing is below $100 1-lot credit; print the vertical"}
    return {
        "ok": True,
        "source": "schwab",
        "token_path": cfg.token_path,
        "symbol": args.symbol.upper(),
        "expiry": args.expiry,
        "underlying_price": spot,
        "atm_straddle": atm,
        "gates": gates,
        "structures": {
            "sell_put_credit": put_c,
            "sell_call_credit": call_c,
            "sell_iron_condor": condor,
            "buy_call_debit": _debit_vertical(cmap, right="C", spot=float(spot)),
            "buy_put_debit": _debit_vertical(pmap, right="P", spot=float(spot)),
        },
    }


def cmd_scan(args: argparse.Namespace) -> Dict[str, Any]:
    """Quote VIX, map regime, structures every symbol × Friday expiry in the DTE window."""
    asof = date.fromisoformat(args.asof) if args.asof else datetime.now().date()
    skip = {s.strip()[:10] for s in (args.skip_expiry or []) if s}
    if args.expiry:
        expiries = [e[:10] for e in args.expiry]
    else:
        expiries = [e for e in friday_expiries(asof, min_dte=args.min_dte, max_dte=args.max_dte) if e not in skip]
    symbols = [s.upper() for s in (args.symbols or SCAN_UNIVERSE)]
    q = cmd_quote(argparse.Namespace(symbols=["$VIX", "SPY"] + symbols[:1]))
    quotes = (q.get("quotes") or {})
    vix = (quotes.get("$VIX") or {}).get("last")
    regime = args.regime if args.regime and args.regime != "auto" else vix_regime(vix)
    gates = resolve_gates(regime)
    structures: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    class One:
        pass
    for i, sym in enumerate(symbols, 1):
        structures[sym] = {}
        for exp in expiries:
            print(f"scan {i}/{len(symbols)} {sym} {exp}", file=sys.stderr, flush=True)
            t0 = datetime.now(timezone.utc)
            try:
                one = One()
                one.symbol = sym
                one.expiry = exp
                one.strike_count = args.strike_count
                one.regime = regime
                one.max_delta = None
                one.min_sigma = None
                one.min_frac = None
                one.or_delta = None
                data = cmd_structures(one)
                data["elapsed"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
                structures[sym][exp] = data
            except Exception as exc:
                from uwos.schwab_auth import _redact_schwab_error_text

                msg = _redact_schwab_error_text(exc)
                errors[f"{sym}:{exp}"] = msg
                structures[sym][exp] = {"ok": False, "error": msg}
    return {
        "ok": True,
        "asof": asof.isoformat(),
        "session": "scan",
        "vix": vix,
        "spy": (quotes.get("SPY") or {}).get("last"),
        "regime": regime,
        "gates": gates,
        "expiries": expiries,
        "symbols": symbols,
        "structures": structures,
        "errors": errors,
        "source": "schwab",
    }


def main() -> int:
    boot()
    p = argparse.ArgumentParser(description="Schwab quotes/chains for grok-option")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quote")
    q.add_argument("symbols", nargs="+")

    c = sub.add_parser("chain")
    c.add_argument("symbol")
    c.add_argument("--strike-count", type=int, default=int(os.environ.get("SCHWAB_STRIKE_COUNT", "12")))
    c.add_argument("--from-date", default="")
    c.add_argument("--to-date", default="")
    c.add_argument("--expiry", default="")
    c.add_argument("--right", default="", help="P or C")

    v = sub.add_parser("vertical")
    v.add_argument("--symbol", required=True)
    v.add_argument("--right", required=True, help="P or C")
    v.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    v.add_argument("--short", type=float, required=True)
    v.add_argument("--long", type=float, required=True)
    v.add_argument("--kind", choices=("credit", "debit"), required=True)
    v.add_argument("--strike-count", type=int, default=30)

    st = sub.add_parser("structures")
    st.add_argument("symbol")
    st.add_argument("--expiry", required=True)
    st.add_argument("--strike-count", type=int, default=40)
    st.add_argument("--regime", default="calm", choices=("calm", "normal", "elevated", "crisis"))
    st.add_argument("--max-delta", type=float, default=None)
    st.add_argument("--min-sigma", type=float, default=None)
    st.add_argument("--min-frac", type=float, default=None)
    st.add_argument("--or-delta", type=float, default=None, help="If set, short clears on Δ≤or_delta OR σ≥min_sigma")

    sc = sub.add_parser("scan")
    sc.add_argument("symbols", nargs="*")
    sc.add_argument("--asof", default="")
    sc.add_argument("--expiry", action="append", default=[])
    sc.add_argument("--skip-expiry", action="append", default=[])
    sc.add_argument("--min-dte", type=int, default=14)
    sc.add_argument("--max-dte", type=int, default=60)
    sc.add_argument("--regime", default="auto")
    sc.add_argument("--strike-count", type=int, default=24)

    args = p.parse_args()
    try:
        if args.cmd == "quote":
            data = cmd_quote(args)
        elif args.cmd == "chain":
            data = cmd_chain(args)
        elif args.cmd == "structures":
            data = cmd_structures(args)
        elif args.cmd == "scan":
            data = cmd_scan(args)
        else:
            data = cmd_vertical(args)
    except Exception as exc:
        from uwos.schwab_auth import _redact_schwab_error_text

        print(json.dumps({"ok": False, "error": _redact_schwab_error_text(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(data, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
