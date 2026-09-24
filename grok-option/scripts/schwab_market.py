#!/usr/bin/env python3
"""Ticker quotes and option chains via tradedesk Schwab token + .env. No invented prints."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def normalize_session(raw: Optional[str]) -> str:
    """live (RTH) or ah (after-hours / EOD). Date-only SCAN defaults to live."""
    s = (raw or "live").strip().lower().replace("_", "-")
    s = "".join(s.split())
    if s in {"ah", "eod", "after-hours", "afterhours", "aftermarket", "after-market", "postmarket", "post-market"}:
        return "ah"
    if "afterhour" in s or "aftermarket" in s or "postmarket" in s:
        return "ah"
    return "live"


def dated_out(asof: str, session: str, kind: str) -> Path:
    """kind is scan, book, or quotes. Session live → *_live.json, ah → *_ah.json."""
    sess = "ah" if normalize_session(session) == "ah" else "live"
    return _SCRIPT_DIR.parent / "out" / "grok-option" / str(asof)[:10] / f"{kind}_{sess}.json"


def _live_delta(delta: Optional[float]) -> Optional[float]:
    """Schwab uses -999 when the greek is missing. That is not a delta."""
    if delta is None or abs(float(delta)) > 1 or float(delta) != float(delta):
        return None
    return float(delta)


def _finite(v: Any) -> bool:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return x == x and x not in (float("inf"), float("-inf"))


def _positive(v: Any) -> bool:
    """A usable bid or ask. Zero, blank, and NaN are not a market."""
    return _finite(v) and float(v) > 0


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
        "delta": _live_delta(fnum(c.get("delta"))),
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
            row = contract_row(str(e)[:10], s, c)
            # A 0 ask is not the at-the-money wing. Using it makes sigma 0
            # and every short on that Friday fails, or it shrinks sigma and
            # the distance gate gets easier.
            if not _positive(row.get("ask")):
                continue
            rows.append((s, row))
        if not rows:
            return None
        _s, row = min(rows, key=lambda x: abs((x[0] if x[0] is not None else spot) - spot))
        return row

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
    if None in (sb, sa, lb, la, ss, ls) or not all(_finite(x) for x in (sb, sa, lb, la, ss, ls)):
        return {"ok": False, "reason": "missing bid/ask or strike on a leg"}
    width = abs(float(ss) - float(ls))
    if width <= 0:
        return {"ok": False, "reason": "zero width"}
    if kind == "credit":
        # A $0 long ask is not a wing. short bid − 0 prints a fake 2.5-wide
        # (NVDA 245/247.5, long 0/0, delta −999) that wins on credit/width.
        if not _positive(sb) or not _positive(la):
            return {"ok": False, "reason": "no market on short bid or long ask"}
        conservative = float(sb) - float(la)
        if conservative <= 0:
            return {"ok": False, "reason": "no credit"}
        if conservative >= width:
            return {"ok": False, "reason": "credit meets or exceeds width"}
        mid = ((float(sb) + float(sa)) / 2) - ((float(lb) + float(la)) / 2)
        frac = conservative / width if width else None
        max_loss = (width - conservative) * 100
        return {
            "ok": True,
            "quoted": True,
            "kind": "credit",
            "net": round(conservative, 4),
            "mid_net": round(mid, 4),
            "width": width,
            "credit_width": round(frac, 4) if frac is not None else None,
            "max_loss_1lot": round(max_loss, 2),
            "max_profit_1lot": round(conservative * 100, 2),
            "worse_fill": abs(mid - conservative) > 0.05 * abs(mid) if mid else False,
        }
    if not _positive(la) or not _positive(sb):
        return {"ok": False, "reason": "no market on long ask or short bid"}
    conservative = float(la) - float(sb)
    if conservative <= 0 or conservative >= width:
        return {"ok": False, "reason": "debit missing or eats the width"}
    mid = ((float(lb) + float(la)) / 2) - ((float(sb) + float(sa)) / 2)
    frac = conservative / width if width else None
    return {
        "ok": True,
        "quoted": True,
        "kind": "debit",
        "net": round(conservative, 4),
        "mid_net": round(mid, 4),
        "width": width,
        "debit_width": round(frac, 4) if frac is not None else None,
        "max_loss_1lot": round(conservative * 100, 2),
        "max_profit_1lot": round((width - conservative) * 100, 2) if width > conservative else 0.0,
        "worse_fill": abs(mid - conservative) > 0.05 * abs(mid) if mid else False,
    }


_SERVICE_LOCK = threading.Lock()


def service():
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

    # Serialize token load/refresh. HTTP stays per-client after this returns.
    with _SERVICE_LOCK:
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


def cmd_book(_args: argparse.Namespace) -> Dict[str, Any]:
    """Live Schwab equity, cash, long names, option overlay. Compact — no raw token."""
    svc, cfg = service()
    acct = svc.get_account_positions(0)
    bals = acct.get("balances") or {}
    pos = acct.get("positions") or []
    eq = [
        p
        for p in pos
        if str(p.get("asset_type") or "").upper() in ("EQUITY", "COLLECTIVE_INVESTMENT")
    ]
    opt = [p for p in pos if str(p.get("asset_type") or "").upper() == "OPTION"]
    long_eq = sorted(
        {p.get("symbol") for p in eq if (p.get("qty") or 0) > 0 and p.get("symbol")}
    )
    options = []
    for p in opt:
        options.append(
            {
                "symbol": p.get("symbol"),
                "underlying": p.get("underlying"),
                "put_call": p.get("put_call"),
                "qty": p.get("qty"),
                "short_qty": p.get("short_qty"),
                "long_qty": p.get("long_qty"),
                "mv": p.get("market_value"),
            }
        )
    return {
        "source": "schwab",
        "token_path": cfg.token_path,
        "equity": bals.get("total_value"),
        "cash": bals.get("cash"),
        "n_pos": len(pos),
        "n_eq": len(eq),
        "n_opt": len(opt),
        "long_eq": long_eq,
        "options": options,
        "asof": datetime.now(timezone.utc).isoformat(),
        "session": normalize_session(getattr(_args, "session", None)),
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
        "ok": bool(math.get("ok")),
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


def _leg_brief(leg: Any) -> Any:
    if not isinstance(leg, dict):
        return leg
    return {
        "strike": leg.get("strike"),
        "bid": leg.get("bid"),
        "ask": leg.get("ask"),
        "delta": leg.get("delta"),
    }


def compact_vertical(data: Dict[str, Any]) -> Dict[str, Any]:
    """Stdout-sized vertical. Full chain stays off stdout unless --full."""
    atm = data.get("atm_straddle") or {}
    sigma = atm.get("straddle_ask") if isinstance(atm, dict) else None
    pr = data.get("pricing") or {}
    pricing = {
        "ok": pr.get("ok"),
        "kind": pr.get("kind"),
        "net": pr.get("net"),
        "mid_net": pr.get("mid_net"),
        "width": pr.get("width"),
        "credit_width": pr.get("credit_width"),
        "debit_width": pr.get("debit_width"),
        "max_profit_1lot": pr.get("max_profit_1lot"),
        "max_loss_1lot": pr.get("max_loss_1lot"),
        "worse_fill": pr.get("worse_fill"),
    }
    if data.get("reason"):
        top_ok = False
    elif pr:
        top_ok = bool(pr.get("ok", True))
    else:
        top_ok = bool(data.get("ok", True))
    out = {
        "ok": top_ok,
        "symbol": data.get("symbol"),
        "expiry": data.get("expiry"),
        "right": data.get("right"),
        "kind": data.get("kind"),
        "spot": data.get("underlying_price"),
        "sigma": sigma,
        "short": _leg_brief(data.get("short")),
        "long": _leg_brief(data.get("long")),
        "pricing": pricing,
        "net": pr.get("net"),
        "mid": pr.get("mid_net"),
        "frac": pr.get("credit_width") or pr.get("debit_width"),
        "wf": pr.get("worse_fill"),
        "mp": pr.get("max_profit_1lot"),
        "ml": pr.get("max_loss_1lot"),
        "pd": data.get("pd"),
        "n": data.get("pd_n"),
        "r_cons": data.get("r_cons"),
        "pd_l": data.get("pd_l"),
        "pd_reason": data.get("pd_reason"),
    }
    reason = data.get("reason") or pr.get("reason")
    if reason:
        out["reason"] = reason
    nearby = data.get("nearby") or []
    if nearby:
        out["nearby"] = [
            r.get("strike") for r in nearby[:20] if isinstance(r, dict) and r.get("strike") is not None
        ]
    return out


def _quoted_structure(st: Dict[str, Any], *, credit: bool) -> bool:
    """A row is printable only when the pricer marked a real market.

    `worse_fill == False` is not that mark. A 0 ask makes mid and the
    conservative fill agree, so the dead NVDA 245/247.5 wing looked clean.
    """
    if not isinstance(st, dict) or not st.get("ok"):
        return False
    pr = st.get("pricing") or {}
    if pr.get("quoted") is not True:
        return False
    width_key = "credit_width" if credit else "debit_width"
    try:
        ml = float(pr.get("max_loss_1lot"))
        frac = float(pr.get(width_key))
    except (TypeError, ValueError):
        return False
    return ml > 0 and 0 < frac < 1


def _credit_brief(sym: str, exp: str, key: str, st: Dict[str, Any], *, spot: Any, sigma: Any) -> Optional[Dict[str, Any]]:
    credit = key in ("sell_put_credit", "sell_call_credit", "sell_iron_condor")
    if not _quoted_structure(st, credit=credit):
        return None
    if key == "sell_iron_condor":
        put, call = st.get("put") or {}, st.get("call") or {}
        if not (_quoted_structure(put, credit=True) and _quoted_structure(call, credit=True)):
            return None
    pr = st.get("pricing") or {}
    row: Dict[str, Any] = {
        "sym": sym,
        "exp": exp,
        "k": key,
        "net": pr.get("net"),
        "frac": pr.get("credit_width") or pr.get("debit_width"),
        "wf": pr.get("worse_fill"),
        "mp": pr.get("max_profit_1lot"),
        "ml": pr.get("max_loss_1lot"),
        "spot": spot,
        "sigma": sigma,
    }
    if key == "sell_iron_condor":
        put, call = st.get("put") or {}, st.get("call") or {}
        ppr, cpr = put.get("pricing") or {}, call.get("pricing") or {}
        width = pr.get("width")
        net = pr.get("net")
        frac = pr.get("credit_width")
        if frac is None and width and net is not None and float(width) > 0:
            frac = round(float(net) / float(width), 4)
        row["frac"] = frac
        put_d, call_d = put.get("short_delta"), call.get("short_delta")
        ds = [abs(float(x)) for x in (put_d, call_d) if x is not None]
        if ds:
            row["pop"] = int(round(100 * (1 - max(ds))))
        row.update(
            {
                "put_s": put.get("short"),
                "put_l": put.get("long"),
                "put_net": ppr.get("net"),
                "put_d": put.get("short_delta"),
                "put_sig": put.get("sigma_mult"),
                "put_wf": ppr.get("worse_fill"),
                "put_sb": put.get("short_bid"),
                "put_la": put.get("long_ask"),
                "call_s": call.get("short"),
                "call_l": call.get("long"),
                "call_net": cpr.get("net"),
                "call_d": call.get("short_delta"),
                "call_sig": call.get("sigma_mult"),
                "call_wf": cpr.get("worse_fill"),
                "call_sb": call.get("short_bid"),
                "call_la": call.get("long_ask"),
            }
        )
        return row
    row.update(
        {
            "short": st.get("short"),
            "long": st.get("long"),
            "sb": st.get("short_bid"),
            "la": st.get("long_ask"),
            "d": st.get("short_delta") or st.get("long_delta"),
            "sig": st.get("sigma_mult"),
        }
    )
    dlt = row.get("d")
    if dlt is not None:
        if key in ("sell_put_credit", "sell_call_credit"):
            row["pop"] = int(round(100 * (1 - abs(float(dlt)))))
        else:
            row["pop"] = int(round(100 * abs(float(dlt))))
    return row


def _is_skip_stub(data: Dict[str, Any]) -> bool:
    return str(data.get("reason") or "").startswith("skipped:")


def _note_expiry_misses(
    rows: Dict[str, Any],
    miss_streak: Dict[str, int],
    dead: set,
    *,
    n_ok: int,
) -> List[str]:
    """Update skip streaks only when this name had ≥1 successful expiry (Schwab is up)."""
    if int(n_ok or 0) < 1:
        return []
    newly: List[str] = []
    for exp, data in (rows or {}).items():
        if _is_skip_stub(data) or exp in dead:
            continue
        if _missing_listed_chain(data):
            miss_streak[exp] = int(miss_streak.get(exp) or 0) + 1
            if miss_streak[exp] >= 2 and exp not in dead:
                dead.add(exp)
                newly.append(exp)
        elif data.get("ok"):
            miss_streak[exp] = 0
    return newly


def scan_board(structures: Dict[str, Any]) -> Dict[str, Any]:
    """Geometry-pass credits + Fire + tape. Not the chain dump."""
    credits: List[Dict[str, Any]] = []
    fire: List[Dict[str, Any]] = []
    fire_clean: List[Dict[str, Any]] = []
    fire_quoted = 0
    fail_no_chain = 0
    n_skipped = 0
    tape: Dict[str, Any] = {}
    for sym, exps in (structures or {}).items():
        sigmas: Dict[str, Any] = {}
        spot = None
        for exp, data in (exps or {}).items():
            if _is_skip_stub(data):
                n_skipped += 1
                continue
            if not data.get("ok"):
                if _missing_listed_chain(data):
                    fail_no_chain += 1
                continue
            this_spot = data.get("underlying_price")
            if spot is None:
                spot = this_spot
            atm = data.get("atm_straddle") or {}
            sigma = atm.get("straddle_ask") if isinstance(atm, dict) else None
            if sigma is not None:
                sigmas[str(exp)[:10]] = sigma
            ss = data.get("structures") or {}
            for key in ("sell_put_credit", "sell_call_credit", "sell_iron_condor"):
                brief = _credit_brief(sym, exp, key, ss.get(key) or {}, spot=this_spot, sigma=sigma)
                if brief:
                    credits.append(brief)
            for key in ("buy_call_debit", "buy_put_debit"):
                st = ss.get(key) or {}
                if not st.get("ok"):
                    continue
                brief = _credit_brief(sym, exp, key, st, spot=this_spot, sigma=None)
                if not brief:
                    continue
                fire_quoted += 1
                brief.pop("spot", None)
                brief.pop("sigma", None)
                fire.append(brief)
                if not (st.get("pricing") or {}).get("worse_fill"):
                    fire_clean.append(dict(brief))
        if spot is not None or sigmas:
            tape[sym] = {"spot": spot, "sigma": sigmas}
    credits.sort(key=lambda r: (-(float(r.get("frac") or 0)), -(float(r.get("net") or 0))))
    ic_keys = {(c["sym"], c["exp"]) for c in credits if c.get("k") == "sell_iron_condor"}
    for c in credits:
        if c.get("k") in ("sell_put_credit", "sell_call_credit") and (c.get("sym"), c.get("exp")) in ic_keys:
            c["in_ic"] = True
    return {
        "credits": credits,
        "fire": fire,
        "fire_clean": fire_clean,
        "fire_quoted": fire_quoted,
        "fail_no_chain": fail_no_chain,
        "n_skipped": n_skipped,
        "n_credits": len(credits),
        "n_fire": len(fire),
        "n_fire_clean": len(fire_clean),
        "tape": tape,
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
                "short_bid": short.get("bid"),
                "long_ask": long.get("ask"),
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
                "short_bid": short.get("bid"),
                "long_ask": long.get("ask"),
                "short_delta": dlt,
                "pricing": math,
                "otm": otm,
                "sigma_mult": otm / sigma if sigma else None,
            }
            _stamp_pd(cand, short=short, long=long)
            if _credit_better(cand, best):
                best = cand
    return best or {"ok": False, "reason": "no call credit met delta/sigma/width", "tried": tried}


def _iron_condor(put_c: Dict[str, Any], call_c: Dict[str, Any]) -> Dict[str, Any]:
    """One IC when both credit sides independently pass geometry.

    The $100 1-lot floor ranks standalone verticals. It does not veto the condor:
    a $1.34 put + $0.70 call is still one IC (NVDA 2026-09-23 class).
    """
    if not (_quoted_structure(put_c, credit=True) and _quoted_structure(call_c, credit=True)):
        return {"ok": False, "reason": "need both quoted credit sides"}
    pnet = float((put_c.get("pricing") or {}).get("net") or 0)
    cnet = float((call_c.get("pricing") or {}).get("net") or 0)
    if pnet <= 0 or cnet <= 0:
        return {"ok": False, "reason": "need both credit sides"}
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
            "ok": True,
            "quoted": True,
            "kind": "credit",
            "net": total,
            "width": wing,
            "credit_width": round(total / wing, 4) if wing else None,
            "max_loss_1lot": round(max(0.0, wing - total) * 100, 2),
            "max_profit_1lot": round(total * 100, 2),
        },
    }
    if not _quoted_structure(condor, credit=True):
        return {"ok": False, "reason": "condor credit is outside the wing"}
    _stamp_pd(condor)
    return condor


def _debit_vertical(legs: Dict[float, Dict[str, Any]], *, right: str, spot: float) -> Dict[str, Any]:
    ks = sorted(legs)
    best = None
    action = "Buy call debit" if right == "C" else "Buy put debit"
    farther = "up" if right == "C" else "down"
    for sh in ks:
        for farther_k in _listed_pair(ks, sh, farther):
            # Call: sh lower (long), farther higher (short). Put: sh higher (long), farther lower (short).
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


def _chain_spot(payload: Dict[str, Any]) -> Optional[float]:
    raw = (payload or {}).get("underlyingPrice")
    if raw is None:
        raw = (payload or {}).get("underlying_price")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _structures_from_payload(
    symbol: str,
    expiry: str,
    payload: Dict[str, Any],
    *,
    regime: str,
    cfg: Any,
    max_delta: Optional[float] = None,
    min_sigma: Optional[float] = None,
    min_frac: Optional[float] = None,
    or_delta: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one Friday out of an already-fetched chain blob."""
    spot = _chain_spot(payload)
    atm = atm_straddle(payload, spot, expiry=expiry)
    sigma = (atm or {}).get("straddle_ask")
    puts, calls = [], []
    for e, s, c in iter_contracts(payload):
        if str(e)[:10] != expiry[:10]:
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
            "symbol": symbol.upper(),
            "expiry": expiry,
            "underlying_price": spot,
            "atm_straddle": atm,
            "source": "schwab",
        }
    pmap, cmap = _index_legs(puts), _index_legs(calls)
    gates = resolve_gates(
        regime or "calm",
        max_delta=max_delta,
        min_sigma=min_sigma,
        min_frac=min_frac,
        or_delta=or_delta,
    )
    put_c = _credit_put(pmap, float(spot), float(sigma), gates)
    call_c = _credit_call(cmap, float(spot), float(sigma), gates)
    condor = _iron_condor(put_c, call_c)
    return {
        "ok": True,
        "source": "schwab",
        "token_path": getattr(cfg, "token_path", None),
        "symbol": symbol.upper(),
        "expiry": expiry,
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
    # Overlay summarizer spot so range-scored expiries match the live quote.
    if summary.get("underlying_price") is not None:
        payload = dict(payload or {})
        payload["underlyingPrice"] = summary.get("underlying_price")
    return _structures_from_payload(
        args.symbol.upper(),
        args.expiry,
        payload,
        regime=getattr(args, "regime", None) or "calm",
        cfg=cfg,
        max_delta=getattr(args, "max_delta", None),
        min_sigma=getattr(args, "min_sigma", None),
        min_frac=getattr(args, "min_frac", None),
        or_delta=getattr(args, "or_delta", None),
    )


def _scan_one_expiry(sym: str, exp: str, strike_count: int, regime: str) -> Dict[str, Any]:
    class One:
        pass

    one = One()
    one.symbol = sym
    one.expiry = exp
    one.strike_count = strike_count
    one.regime = regime
    one.max_delta = None
    one.min_sigma = None
    one.min_frac = None
    one.or_delta = None
    t0 = datetime.now(timezone.utc)
    try:
        data = cmd_structures(one)
    except Exception as exc:
        from uwos.schwab_auth import _redact_schwab_error_text

        msg = _redact_schwab_error_text(exc)
        data = {"ok": False, "error": msg}
        data["elapsed"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
        return data
    data["elapsed"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
    return data


def _scan_one_name(sym: str, expiries: List[str], strike_count: int, regime: str) -> Dict[str, Any]:
    """One Schwab chain for the Friday window, then score each expiry in-process."""
    if not expiries:
        return {}
    t0 = datetime.now(timezone.utc)
    try:
        svc, cfg = service()
        payload = svc.get_option_chain(
            sym,
            strike_count=strike_count,
            include_underlying_quote=True,
            from_date=expiries[0],
            to_date=expiries[-1],
        )
        summary = svc.summarize_option_chain(sym, payload)
        if isinstance(payload, dict) and summary.get("underlying_price") is not None:
            payload = dict(payload)
            payload["underlyingPrice"] = summary.get("underlying_price")
        elapsed = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
        out: Dict[str, Any] = {}
        for exp in expiries:
            data = _structures_from_payload(sym, exp, payload, regime=regime, cfg=cfg)
            data["elapsed"] = elapsed
            out[exp] = data
        return out
    except Exception:
        return {exp: _scan_one_expiry(sym, exp, strike_count, regime) for exp in expiries}


def _missing_listed_chain(data: Dict[str, Any]) -> bool:
    if data.get("ok"):
        return False
    blob = f"{data.get('reason') or ''} {data.get('error') or ''}".lower()
    return "missing spot or atm straddle" in blob


def cmd_scan(args: argparse.Namespace) -> Dict[str, Any]:
    """Quote VIX, map regime, structures every symbol × Friday expiry in the DTE window."""
    asof = date.fromisoformat(args.asof) if args.asof else datetime.now().date()
    session = normalize_session(getattr(args, "session", None))
    skip = {s.strip()[:10] for s in (args.skip_expiry or []) if s}
    if args.expiry:
        expiries = [e[:10] for e in args.expiry]
    else:
        expiries = [e for e in friday_expiries(asof, min_dte=args.min_dte, max_dte=args.max_dte) if e not in skip]
    symbols = [s.upper() for s in (args.symbols or SCAN_UNIVERSE)]
    # Worker threads do not get SIGALRM. In-process session timeout still applies.
    os.environ["UWOS_SCHWAB_OPTION_CHAIN_SUBPROCESS"] = "0"
    q = cmd_quote(argparse.Namespace(symbols=["$VIX", "SPY"] + symbols[:1]))
    quotes = (q.get("quotes") or {})
    vix = (quotes.get("$VIX") or {}).get("last")
    regime = args.regime if args.regime and args.regime != "auto" else vix_regime(vix)
    gates = resolve_gates(regime)
    structures: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    workers = max(1, int(getattr(args, "workers", 4) or 1))
    print(
        f"scan {len(symbols)} names × {len(expiries)} expiries, workers={workers} (one chain/name)",
        file=sys.stderr,
        flush=True,
    )
    pool = min(workers, max(1, len(symbols)))
    n_done = 0
    with ThreadPoolExecutor(max_workers=pool) as ex:
        futs = {
            ex.submit(_scan_one_name, sym, expiries, args.strike_count, regime): sym
            for sym in symbols
        }
        for fut in as_completed(futs):
            sym = futs[fut]
            n_done += 1
            rows = fut.result() or {}
            structures[sym] = {e: rows[e] for e in expiries if e in rows}
            for exp in expiries:
                if exp not in structures[sym]:
                    structures[sym][exp] = {"ok": False, "reason": "skipped: no listed chain"}
            n_ok = 0
            n_credit = 0
            elapsed_max = 0.0
            for exp, data in structures[sym].items():
                elapsed_max = max(elapsed_max, float(data.get("elapsed") or 0))
                if data.get("error"):
                    errors[f"{sym}:{exp}"] = data["error"]
                if data.get("ok"):
                    n_ok += 1
                    ss = data.get("structures") or {}
                    n_credit += sum(
                        1
                        for k in ("sell_put_credit", "sell_call_credit", "sell_iron_condor")
                        if (ss.get(k) or {}).get("ok")
                    )
            print(
                f"scan {n_done}/{len(symbols)} {sym} {n_ok}/{len(expiries)} {n_credit}c {elapsed_max:.0f}s",
                file=sys.stderr,
                flush=True,
            )
    skipped = [
        e
        for e in expiries
        if symbols
        and all(
            _missing_listed_chain(structures.get(s, {}).get(e) or {})
            or _is_skip_stub(structures.get(s, {}).get(e) or {})
            for s in symbols
        )
    ]
    board = scan_board(structures)
    board.update(
        {
            "ok": True,
            "asof": asof.isoformat(),
            "session": session,
            "vix": vix,
            "spy": (quotes.get("SPY") or {}).get("last"),
            "regime": regime,
            "gates": gates,
            "expiries": expiries,
            "skipped_expiries": skipped,
            "workers": workers,
            "n_names": len(symbols),
            "n_errors": len(errors),
        }
    )
    return {
        "ok": True,
        "asof": asof.isoformat(),
        "session": session,
        "vix": vix,
        "spy": (quotes.get("SPY") or {}).get("last"),
        "regime": regime,
        "gates": gates,
        "expiries": expiries,
        "skipped_expiries": skipped,
        "workers": workers,
        "symbols": symbols,
        "structures": structures,
        "errors": errors,
        "board": board,
        "source": "schwab",
    }


def main() -> int:
    boot()
    p = argparse.ArgumentParser(description="Schwab quotes/chains for grok-option")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quote")
    q.add_argument("symbols", nargs="+")

    bk = sub.add_parser("book")
    bk.add_argument("--session", default="live", help="live (RTH) or ah (after-hours).")
    bk.add_argument("--out", default="", help="Write book JSON here. Default dated book_live.json / book_ah.json.")

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
    v.add_argument("--full", action="store_true", help="Print full chain JSON. Default is compact legs+net.")

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
    sc.add_argument(
        "--session",
        default="live",
        help="live (RTH / market-open) or ah (after-hours / EOD). Date-only defaults to live.",
    )
    sc.add_argument("--expiry", action="append", default=[])
    sc.add_argument("--skip-expiry", action="append", default=[])
    sc.add_argument("--min-dte", type=int, default=14)
    sc.add_argument("--max-dte", type=int, default=60)
    sc.add_argument("--regime", default="auto")
    sc.add_argument("--strike-count", type=int, default=24)
    sc.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel names (one Schwab chain per name covering the Friday window). 1 = serial.",
    )
    sc.add_argument(
        "--out",
        default="",
        help="Write full scan JSON here. Stdout is the compact board unless --full.",
    )
    sc.add_argument("--full", action="store_true", help="Print full scan JSON to stdout (token bomb).")

    args = p.parse_args()
    try:
        if args.cmd == "quote":
            data = cmd_quote(args)
        elif args.cmd == "chain":
            data = cmd_chain(args)
        elif args.cmd == "book":
            data = cmd_book(args)
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
    if args.cmd == "scan":
        out_path = (getattr(args, "out", None) or "").strip()
        session = normalize_session(data.get("session") or getattr(args, "session", None))
        asof = str((data.get("asof") or ""))[:10]
        if not out_path and asof:
            out_path = str(dated_out(asof, session, "scan"))
        board = dict(data.get("board") or {})
        board["session"] = session
        if out_path:
            dest = Path(out_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data, default=str))
            board["out"] = str(dest.resolve())
        print(json.dumps(data if getattr(args, "full", False) else board, default=str))
        return 0
    if args.cmd == "book":
        out_path = (getattr(args, "out", None) or "").strip()
        session = normalize_session(data.get("session") or getattr(args, "session", None))
        asof = str((data.get("asof") or datetime.now(timezone.utc).date().isoformat()))[:10]
        if not out_path:
            out_path = str(dated_out(asof, session, "book"))
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, default=str))
        payload = dict(data)
        payload["out"] = str(dest.resolve())
        payload["session"] = session
        print(json.dumps(payload, default=str))
        return 0
    if args.cmd == "vertical" and not getattr(args, "full", False):
        print(json.dumps(compact_vertical(data), default=str))
        return 0
    print(json.dumps(data, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
