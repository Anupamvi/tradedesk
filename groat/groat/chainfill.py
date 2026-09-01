"""Schwab fill overlay for option structures. After-hours last/mark if bid/ask are dead."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from groat.config import CODE_DIR, DTE_MAX, DTE_MIN
from groat.num import to_float
from groat.envload import schwab_credentials
from groat.schwab import option_chain


def chains_dir(asof: str) -> Path:
    return CODE_DIR / "var" / "schwab_chains" / asof[:10]


def _dte(asof: str, expiry: str) -> Optional[float]:
    try:
        return float(
            (datetime.strptime(expiry[:10], "%Y-%m-%d") - datetime.strptime(asof[:10], "%Y-%m-%d")).days
        )
    except ValueError:
        return None


def _exp_date(key: str) -> str:
    return str(key or "").split(":")[0][:10]


def _conservative_band(px: float) -> Tuple[float, float]:
    pad = max(0.05, round(0.03 * px, 2))
    return max(0.01, round(px - pad, 2)), round(px + pad, 2)


def fill_px(leg: dict) -> Tuple[Optional[float], Optional[float], str]:
    """Conservative fill: bid/ask if live-looking, else padded mark/last. Never invent."""
    bid = to_float(leg.get("bid"))
    ask = to_float(leg.get("ask"))
    mark = to_float(leg.get("mark"))
    last = to_float(leg.get("last"))
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return bid, ask, "schwab_quote"
    if mark is not None and mark > 0:
        lo, hi = _conservative_band(mark)
        return lo, hi, "schwab_mark"
    if last is not None and last > 0:
        lo, hi = _conservative_band(last)
        return lo, hi, "schwab_last"
    return bid, ask, "none"


def _contract(row: Any) -> dict:
    if isinstance(row, list) and row:
        row = row[0]
    if not isinstance(row, dict):
        return {}
    return {
        "bid": to_float(row.get("bid")),
        "ask": to_float(row.get("ask")),
        "mark": to_float(row.get("mark")),
        "last": to_float(row.get("lastPrice") or row.get("last")),
        "oi": to_float(row.get("openInterest")),
        "vol": to_float(row.get("totalVolume")),
        "delta": to_float(row.get("delta")),
        "gamma": to_float(row.get("gamma")),
        "theta": to_float(row.get("theta")),
        "vega": to_float(row.get("vega")),
        "strike": to_float(row.get("strikePrice") or row.get("strike")),
        "expiry": str(row.get("expirationDate") or "")[:10],
    }


def quote_asof_from_payload(payload: Optional[dict]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    und = payload.get("underlying") if isinstance(payload.get("underlying"), dict) else {}
    ts = (und or {}).get("quoteTime") or (und or {}).get("tradeTime")
    if ts is None:
        return None
    try:
        ms = int(ts)
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        try:
            from zoneinfo import ZoneInfo

            dt = dt.astimezone(ZoneInfo("America/New_York"))
            return dt.strftime("%Y-%m-%d %H:%M ET")
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def flatten_chain(payload: Optional[dict]) -> Dict[Tuple[str, float], dict]:
    out: Dict[Tuple[str, float], dict] = {}
    if not isinstance(payload, dict):
        return out
    asof_print = quote_asof_from_payload(payload)

    def absorb(mmap: dict, side: str) -> None:
        for exp_key, strikes in (mmap or {}).items():
            expiry = _exp_date(str(exp_key))
            if not expiry:
                continue
            if not isinstance(strikes, dict):
                continue
            for k, rows in strikes.items():
                strike = to_float(k)
                parsed = _contract(rows)
                if strike is None:
                    strike = parsed.get("strike")
                if strike is None:
                    continue
                exp_use = expiry
                if parsed.get("expiry"):
                    exp_use = str(parsed["expiry"])[:10]
                slot = out.setdefault((exp_use, float(strike)), {})
                slot[side] = parsed
                slot["expiry"] = exp_use
                slot["strike"] = float(strike)

    absorb(payload.get("callExpDateMap") or {}, "call")
    absorb(payload.get("putExpDateMap") or {}, "put")
    und = payload.get("underlying") if isinstance(payload.get("underlying"), dict) else {}
    spot = to_float((und or {}).get("last")) or to_float((und or {}).get("mark")) or to_float(
        (und or {}).get("close")
    )
    for rec in out.values():
        rec["spot"] = spot
        rec["quote_asof"] = asof_print
    return out


def overlay_row(raw: dict, rec: dict) -> dict:
    row = dict(raw)
    call = rec.get("call") or {}
    put = rec.get("put") or {}
    cb, ca, csrc = fill_px(call)
    if cb is not None and ca is not None:
        row["callBidPrice"] = cb
        row["callAskPrice"] = ca
        row["quoteSource"] = csrc
    if call.get("oi") is not None:
        row["callOpenInterest"] = call["oi"]
    if call.get("vol") is not None:
        row["callVolume"] = call["vol"]
    if call.get("delta") is not None:
        row["delta"] = call["delta"]
    if call.get("gamma") is not None:
        row["gamma"] = call["gamma"]
    if call.get("theta") is not None:
        row["theta"] = call["theta"]
    if call.get("vega") is not None:
        row["vega"] = call["vega"]
    pb, pa, psrc = fill_px(put)
    if pb is not None and pa is not None:
        row["putBidPrice"] = pb
        row["putAskPrice"] = pa
        if not row.get("quoteSource"):
            row["quoteSource"] = psrc
    if put.get("oi") is not None:
        row["putOpenInterest"] = put["oi"]
    if rec.get("spot") is not None:
        row["stockPrice"] = rec["spot"]
        row["spotPrice"] = rec["spot"]
    if rec.get("quote_asof"):
        row["quoteAsof"] = rec["quote_asof"]
    return row


def schwab_map_to_orats(flat: Dict[Tuple[str, float], dict], asof: str) -> List[dict]:
    rows = []
    for (expiry, strike), rec in sorted(flat.items()):
        dte = _dte(asof, expiry)
        if dte is None or not (DTE_MIN <= dte <= DTE_MAX):
            continue
        call = rec.get("call") or {}
        put = rec.get("put") or {}
        cb, ca, csrc = fill_px(call)
        pb, pa, psrc = fill_px(put)
        if (cb is None or ca is None) and (pb is None or pa is None):
            continue
        rows.append(
            {
                "strike": strike,
                "dte": dte,
                "expirDate": expiry,
                "stockPrice": rec.get("spot"),
                "spotPrice": rec.get("spot"),
                "delta": call.get("delta"),
                "gamma": call.get("gamma"),
                "theta": call.get("theta"),
                "vega": call.get("vega"),
                "callBidPrice": cb,
                "callAskPrice": ca,
                "callOpenInterest": call.get("oi"),
                "callVolume": call.get("vol"),
                "putBidPrice": pb,
                "putAskPrice": pa,
                "putOpenInterest": put.get("oi"),
                "putVolume": put.get("vol"),
                "quoteSource": csrc if csrc != "none" else psrc,
                "quoteAsof": rec.get("quote_asof"),
            }
        )
    return rows


def overlay_ticker(asof: str, orats_rows: Sequence[dict], flat: Dict[Tuple[str, float], dict]) -> List[dict]:
    if not flat:
        return list(orats_rows or [])
    out = []
    seen = set()
    for raw in orats_rows or []:
        expiry = str(raw.get("expirDate") or "")[:10]
        strike = to_float(raw.get("strike"))
        rec = flat.get((expiry, float(strike))) if strike is not None else None
        if rec:
            out.append(overlay_row(raw, rec))
            seen.add((expiry, float(strike)))
        else:
            out.append(dict(raw))
    for key, rec in flat.items():
        if key in seen:
            continue
        expiry, strike = key
        dte = _dte(asof, expiry)
        if dte is None or not (DTE_MIN <= dte <= DTE_MAX):
            continue
        stub = {
            "strike": strike,
            "dte": dte,
            "expirDate": expiry,
            "stockPrice": rec.get("spot"),
            "spotPrice": rec.get("spot"),
        }
        out.append(overlay_row(stub, rec))
    return out


def fetch_and_flatten(ticker: str, asof: str, chain_fn=None) -> Dict[Tuple[str, float], dict]:
    start = (datetime.strptime(asof[:10], "%Y-%m-%d") + timedelta(days=int(DTE_MIN))).date().isoformat()
    end = (datetime.strptime(asof[:10], "%Y-%m-%d") + timedelta(days=int(DTE_MAX))).date().isoformat()
    fn = chain_fn or option_chain
    payload = fn(ticker, start, end)
    if payload is None:
        return {}
    path = chains_dir(asof) / ("%s.json" % ticker.upper())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps({"asof": asof, "ticker": ticker, "data": payload}, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return flatten_chain(payload)


def overlay_strikes(
    asof: str,
    tickers: Sequence[str],
    strikes_by_ticker: Dict[str, list],
    chain_fn=None,
    errors: Optional[list] = None,
) -> Dict[str, list]:
    if not schwab_credentials() and chain_fn is None:
        return dict(strikes_by_ticker)
    out = dict(strikes_by_ticker or {})
    for name in tickers:
        ticker = str(name or "").upper()
        if not ticker:
            continue
        try:
            flat = fetch_and_flatten(ticker, asof, chain_fn=chain_fn)
        except Exception as exc:
            if errors is not None:
                errors.append({"ticker": ticker, "error": str(exc)[:160]})
            continue
        if not flat:
            if errors is not None:
                errors.append({"ticker": ticker, "error": "Schwab chain empty"})
            continue
        existing = out.get(ticker) or []
        if existing:
            out[ticker] = overlay_ticker(asof, existing, flat)
        else:
            out[ticker] = schwab_map_to_orats(flat, asof)
    return out
