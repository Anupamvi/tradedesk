import argparse
import copy
import datetime as dt
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import urllib.parse
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uwos.eod_trade_scan_mode_a import (
    apply_chain_oi_overlay_to_screener,
    build_best_candidates,
    build_quotes,
    compute_macro_regime,
    ensure_cols,
    fnum,
    parse_occ,
)
from uwos.report import load_open_positions
from uwos.whale_source import BOT_EOD_PREFIX, load_whale_flow_source
from uwos.weekly_trade_generator import GeneratorConfig as WeeklyCreditConfig
from uwos.weekly_trade_generator import generate_for_day as generate_weekly_credit_for_day
from uwos.weekly_trade_generator import write_outputs as write_weekly_credit_outputs


def _safe_delta(val):
    """Normalize delta values: treat sentinel -999/999 and out-of-range as NaN."""
    d = fnum(val)
    if not np.isfinite(d) or abs(d) > 1.0:
        return float("nan")
    return d


def dynamic_shield_delta_cap(ivr, dte, gex_regime="", vix=20.0, strategy="credit_spread"):
    """Compute max allowed short delta for SHIELD trades.

    Based on professional credit-spread practice (Tastytrade / Option Alpha research):
    - IVR drives base delta: high IV → sell closer to ATM (more premium collected)
    - DTE adjusts: shorter DTE → further OTM (gamma risk)
    - VIX crisis (>35) → pull back despite rich premium (tail risk)
    - Negative GEX → further OTM (amplified moves)
    - Iron condors → slightly tighter per side

    Returns max abs(delta) for the short leg, or 0.0 if trade should be skipped.
    """
    # IVR base: how much delta the premium environment supports
    if ivr >= 50:
        base = 0.35
    elif ivr >= 35:
        base = 0.30
    elif ivr >= 25:
        base = 0.25
    elif ivr >= 15:
        base = 0.20
    else:
        return 0.0  # IVR too low — skip credit spreads

    # DTE adjustment: 45 DTE is the sweet spot
    if 45 <= dte <= 60:
        dte_adj = 0.0
    elif 30 <= dte < 45:
        dte_adj = -0.03
    elif 21 <= dte < 30:
        dte_adj = -0.06
    elif dte < 21:
        dte_adj = -0.10
    else:  # > 60
        dte_adj = -0.02

    # VIX crisis adjustment: rich premium but tail risk
    if vix > 40:
        vix_adj = -0.08
    elif vix > 30:
        vix_adj = -0.03
    elif vix < 15:
        vix_adj = -0.05  # low vol = thin premium, go further OTM
    else:
        vix_adj = 0.0

    # GEX overlay: negative GEX amplifies moves
    gex_adj = -0.05 if str(gex_regime).strip().lower() == "volatile" else 0.0

    # Iron condors: tighter per side (two legs exposed)
    ic_adj = -0.03 if strategy == "iron_condor" else 0.0

    cap = base + dte_adj + vix_adj + gex_adj + ic_adj
    return max(0.10, min(0.40, cap))


REQ_CSV_PREFIXES = [
    "chain-oi-changes-",
    "dp-eod-report-",
    "hot-chains-",
    "stock-screener-",
]
CSV_PREFIX_ALIASES = {}
DATE_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
UW_GEX_SOURCE = "unusual_whales_dashboard_cdp"
SCHWAB_LIVE_GEX_SOURCE = "schwab_live_chain"
SCHWAB_STALE_GEX_SOURCE = "schwab_stale_chain"
SCHWAB_LEGACY_GEX_SOURCE = "schwab_snapshot_fallback"
LIVE_GEX_SOURCES = {SCHWAB_LIVE_GEX_SOURCE}


def live_gex_entry_block_reason(row, auto_gex_required: bool) -> str:
    """Return the live-entry blocker when a trade lacks current Schwab-chain GEX."""
    if not auto_gex_required:
        return ""
    gex_source_live = str(row.get("gex_source", "") or "").strip().lower()
    if gex_source_live in LIVE_GEX_SOURCES:
        return ""
    return (
        "Schwab live chain GEX required before live entry; "
        f"current GEX source={gex_source_live or 'missing'}."
    )


def _finite_positive_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) and parsed > 0 else float("nan")


def _schwab_chain_spot(chain_data: dict) -> float:
    underlying = chain_data.get("underlying", {}) if isinstance(chain_data, dict) else {}
    if not isinstance(underlying, dict):
        underlying = {}
    for value in [
        underlying.get("mark"),
        underlying.get("last"),
        underlying.get("close"),
        chain_data.get("underlyingPrice") if isinstance(chain_data, dict) else None,
        chain_data.get("lastPrice") if isinstance(chain_data, dict) else None,
    ]:
        spot = _finite_positive_float(value)
        if np.isfinite(spot):
            return spot
    return float("nan")


def _schwab_chain_time(chain_data: dict) -> str:
    underlying = chain_data.get("underlying", {}) if isinstance(chain_data, dict) else {}
    if not isinstance(underlying, dict):
        underlying = {}
    for field in ("quoteTime", "tradeTime", "quoteTimeInLong", "tradeTimeInLong"):
        value = underlying.get(field) or chain_data.get(field)
        if value in (None, ""):
            continue
        try:
            ts = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(ts) or ts <= 0:
            continue
        if ts > 10_000_000_000:
            ts /= 1000.0
        if ts > 1_000_000_000:
            return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        return str(value)
    return ""


def compute_schwab_chain_gex(chain_data: dict, *, source: str = SCHWAB_LIVE_GEX_SOURCE):
    """Calculate ticker-level GEX from a Schwab option-chain payload."""
    if not isinstance(chain_data, dict):
        return None
    spot = _schwab_chain_spot(chain_data)
    if not np.isfinite(spot) or spot <= 0:
        return None

    total_call_gex = 0.0
    total_put_gex = 0.0
    best_put_wall = (0.0, float("nan"))
    best_call_wall = (0.0, float("nan"))
    valid_contracts = 0

    for map_name, side in [("callExpDateMap", "call"), ("putExpDateMap", "put")]:
        exp_map = chain_data.get(map_name, {}) or {}
        if not isinstance(exp_map, dict):
            continue
        for _exp_key, strike_map in exp_map.items():
            if not isinstance(strike_map, dict):
                continue
            for strike_key, contracts in strike_map.items():
                if isinstance(contracts, dict):
                    contract_iter = [contracts]
                elif isinstance(contracts, list):
                    contract_iter = contracts
                else:
                    continue
                for contract in contract_iter:
                    if not isinstance(contract, dict):
                        continue
                    gamma = contract.get("gamma")
                    oi = contract.get("openInterest")
                    if gamma is None or oi is None:
                        continue
                    try:
                        g = float(gamma)
                        o = float(oi)
                        strike_f = float(contract.get("strikePrice", strike_key))
                    except (TypeError, ValueError):
                        continue
                    if not (math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0):
                        continue
                    multiplier = _finite_positive_float(contract.get("multiplier", 100.0))
                    if not np.isfinite(multiplier):
                        multiplier = 100.0
                    gex = g * o * multiplier * spot
                    valid_contracts += 1
                    if side == "call":
                        total_call_gex += gex
                        if strike_f > spot and gex > best_call_wall[0]:
                            best_call_wall = (gex, strike_f)
                    else:
                        total_put_gex += gex
                        if strike_f < spot and gex > best_put_wall[0]:
                            best_put_wall = (gex, strike_f)

    if valid_contracts <= 0:
        return None

    net = total_call_gex - total_put_gex
    return {
        "net_gex": round(net, 2),
        "gex_regime": "pinned" if net >= 0 else "volatile",
        "gex_support": best_put_wall[1] if math.isfinite(best_put_wall[1]) else float("nan"),
        "gex_resistance": best_call_wall[1] if math.isfinite(best_call_wall[1]) else float("nan"),
        "gex_source": source,
        "gex_time": _schwab_chain_time(chain_data),
    }


def fire_long_delta_proxy_ok(strategy: str, long_strike, spot, max_otm_pct: float) -> tuple[bool, float]:
    """Fallback for chains that omit delta on an otherwise near-ATM FIRE long leg."""
    long_strike_f = _finite_positive_float(long_strike)
    spot_f = _finite_positive_float(spot)
    try:
        max_otm = float(max_otm_pct)
    except (TypeError, ValueError):
        max_otm = float("nan")
    if not (np.isfinite(long_strike_f) and np.isfinite(spot_f) and np.isfinite(max_otm) and max_otm >= 0):
        return False, float("nan")
    strat = str(strategy or "").strip()
    if strat == "Bull Call Debit":
        otm_pct = (long_strike_f / spot_f) - 1.0
    elif strat == "Bear Put Debit":
        otm_pct = 1.0 - (long_strike_f / spot_f)
    else:
        return False, float("nan")
    return bool(otm_pct <= max_otm), float(otm_pct)


def split_approval_blockers(row) -> tuple[list[str], list[str]]:
    raw = row.get("approval_blockers", "")
    items = [x for x in str(raw).split(";") if str(x).strip()]
    strategy_local = str(row.get("strategy", "")).strip()
    gex_context_local = str(row.get("gex_wall_context", "")).strip()
    iv_rank_local = fnum(row.get("iv_rank"))
    ic_income_constructive = (
        strategy_local in {"Iron Condor", "Iron Butterfly"}
        and gex_context_local == "pinned_income_constructive"
    )
    high_iv_ic_income_constructive = (
        ic_income_constructive
        and np.isfinite(iv_rank_local)
        and iv_rank_local >= 60.0
    )
    quality = []
    hard = []
    for b in items:
        token = str(b).strip()
        if (
            token == "gex_missing_evidence_lane"
            or token.startswith("bull_call_evidence")
            or token.startswith("bear_put_evidence")
            or token.startswith("bull_call_pinned_continuation")
            or token.startswith("long_delta_proxy_ok")
        ):
            quality.append(token)
            continue
        if token == "gex_source_fallback_tactical_only":
            quality.append(token)
            continue
        if (
            token.startswith("stage1_conviction_below_yes_good")
            or token == "stage1_flow_weak_or_ambiguous"
            or token.startswith("stage1_contract_flow_weak_or_ambiguous")
            or token.startswith("stage1_contract_flow_unknown")
            or token.startswith("stage1_high_iv_debit_watch_only")
        ):
            quality.append(token)
            continue
        if token.startswith("market_regime_caution"):
            quality.append(token)
            continue
        if token.startswith("bull_call_dte_near_limit"):
            quality.append(token)
            continue
        if token == "gex_missing" or token.startswith("bull_call_missing_gex_without_uptrend"):
            quality.append(token)
            continue
        if ic_income_constructive and not high_iv_ic_income_constructive and (
            token.startswith("contract_flow_directional")
            or token.startswith("flow_too_directional_for_ic")
        ):
            quality.append(token)
            continue
        # Hard blockers are reserved for structural or safety-critical failures.
        # Everything else should degrade to Tactical if the trade still clears
        # the tactical floors below.
        if (
            token.startswith("live_status:")
            or token == "live_entry_gate_fail"
            or token == "invalidation_warning"
            or token == "spot_live_missing"
            or token == "spot_drift_unknown"
            or token.startswith("spot_drift:")
            or token.startswith("bull_call_otm_too_far")
            or token.startswith("bear_put_otm_too_far")
            or token.startswith("fire_delta")
            or token.startswith("shield_gex")
            or token.startswith("ic_gex")
            or token.startswith("gex_source_fallback_uncertain")
            or token == "gex_source_stale"
            or token.startswith("shield_delta")
            or token.startswith("flow_too_directional_for_ic")
            or token.startswith("contract_flow_directional")
            or token.startswith("contract_flow_contra")
            or token.startswith("stage1_contract_flow_contra")
            or token.startswith("flow_contra_bull_put")
            or token.startswith("flow_contra_bear_call")
            or token.startswith("market_regime_block")
            or token.startswith("confidence_tier_blocked")
            or token.startswith("stage1_")
            or token.startswith("bull_call_")
        ):
            hard.append(token)
            continue
        if (
            token.startswith("likelihood_")
            or token.startswith("edge_below")
            or token.startswith("signals_below")
            or token.startswith("credit_no_touch")
            or token.startswith("shield_core")
            or token.startswith("shield_sigma")
            or token.startswith("fire_gex")
            or token.startswith("gex_context")
            or token.startswith("gex_volatile")
            or token.startswith("gex_wall")
            or token.startswith("flow_")
            or token.startswith("contract_flow_")
            or token.startswith("live_rr_weak")
            or token.startswith("market_regime_caution")
        ):
            quality.append(token)
        else:
            hard.append(token)
    return hard, quality


def normalize_probability(value) -> float:
    if isinstance(value, str) and value.strip().endswith("%"):
        raw = value.strip().rstrip("%").strip()
        pct = fnum(raw)
        return pct / 100.0 if np.isfinite(pct) else math.nan
    val = fnum(value)
    if not np.isfinite(val):
        return math.nan
    if val > 1.0 and val <= 100.0:
        return val / 100.0
    return val


def normalize_iv(value, default: float = math.nan) -> float:
    iv = fnum(value)
    if not np.isfinite(iv) or iv <= 0:
        return default
    if iv > 5.0:
        iv = iv / 100.0
    if not np.isfinite(iv) or iv <= 0 or iv > 5.0:
        return default
    return float(iv)


def _lognormal_partial_ramp_value(close: float, sigma_t: float, k_lo: float, k_hi: float) -> float:
    if sigma_t <= 0.0 or close <= 0.0 or k_hi <= k_lo:
        return math.nan
    try:
        d1_lo = (math.log(close / k_lo) + 0.5 * sigma_t * sigma_t) / sigma_t
        d2_lo = d1_lo - sigma_t
        d1_hi = (math.log(close / k_hi) + 0.5 * sigma_t * sigma_t) / sigma_t
        d2_hi = d1_hi - sigma_t
    except (ValueError, ZeroDivisionError):
        return math.nan
    return close * (_hist_norm_cdf(d1_lo) - _hist_norm_cdf(d1_hi)) - k_lo * (
        _hist_norm_cdf(d2_lo) - _hist_norm_cdf(d2_hi)
    )


def partial_ev_ml_debit(
    close,
    iv,
    dte_days,
    long_strike,
    short_strike,
    net_debit,
    direction: str,
) -> float:
    """Closed-form lognormal EV/max-loss for debit verticals, including the ramp zone."""
    close_f = fnum(close)
    iv_f = normalize_iv(iv)
    dte_f = fnum(dte_days)
    long_f = fnum(long_strike)
    short_f = fnum(short_strike)
    net_f = fnum(net_debit)
    if not (
        np.isfinite(close_f)
        and close_f > 0
        and np.isfinite(iv_f)
        and np.isfinite(dte_f)
        and np.isfinite(long_f)
        and np.isfinite(short_f)
        and np.isfinite(net_f)
        and net_f > 0
    ):
        return math.nan
    width = abs(short_f - long_f)
    if width <= 0:
        return math.nan
    sigma_t = max(iv_f, 0.05) * math.sqrt(max(int(round(dte_f)), 1) / 365.0)
    if sigma_t <= 0:
        return math.nan
    side = str(direction or "").strip().lower()
    try:
        if side == "bull":
            if not long_f < short_f:
                return math.nan
            partial_value = _lognormal_partial_ramp_value(close_f, sigma_t, long_f, short_f)
            z_hi = math.log(short_f / close_f) / sigma_t
            p_full_win = max(0.0, 1.0 - _hist_norm_cdf(z_hi))
            expected_payoff = partial_value + p_full_win * width
        elif side == "bear":
            if not short_f < long_f:
                return math.nan
            partial_call_value = _lognormal_partial_ramp_value(close_f, sigma_t, short_f, long_f)
            z_lo = math.log(short_f / close_f) / sigma_t
            z_hi = math.log(long_f / close_f) / sigma_t
            p_zone = max(0.0, _hist_norm_cdf(z_hi) - _hist_norm_cdf(z_lo))
            partial_put_value = width * p_zone - partial_call_value
            z_short = math.log(short_f / close_f) / sigma_t
            p_full_win = max(0.0, _hist_norm_cdf(z_short))
            expected_payoff = p_full_win * width + partial_put_value
        else:
            return math.nan
    except (ValueError, ZeroDivisionError):
        return math.nan
    ev_per_share = expected_payoff - net_f
    return ev_per_share / net_f


def debit_partial_ev_for_row(row, *, use_live: bool = True) -> float:
    strategy_local = str(row.get("strategy", "")).strip()
    if strategy_local not in {"Bull Call Debit", "Bear Put Debit"}:
        return math.nan
    direction = "bull" if strategy_local == "Bull Call Debit" else "bear"
    spot = fnum(row.get("spot_live_effective")) if use_live else math.nan
    if not np.isfinite(spot) or spot <= 0:
        spot = fnum(row.get("spot_asof_close"))
    net = fnum(row.get("live_net_bid_ask")) if use_live else math.nan
    if not np.isfinite(net) or net <= 0:
        net = fnum(row.get("net"))
    return partial_ev_ml_debit(
        spot,
        row.get("iv30d"),
        row.get("dte"),
        row.get("long_strike"),
        row.get("short_strike"),
        net,
        direction,
    )


def pilot_convexity_blockers_allow(tokens) -> bool:
    """Pilot may forgive weak evidence, never structural/live safety failures."""
    blocked_exact = {
        "live_entry_gate_fail",
        "invalidation_warning",
        "spot_live_missing",
        "spot_drift_unknown",
        "gex_source_stale",
    }
    blocked_prefixes = (
        "live_status:",
        "spot_drift:",
        "gex_source_fallback_uncertain",
        "flow_contra_",
        "contract_flow_contra",
        "stage1_contract_flow_contra",
        "contract_flow_directional",
        "stage1_contract_flow_directional",
        "pilot_ev_ml_",
        "confidence_tier_blocked",
        "shield_delta",
        "shield_sigma",
        "shield_core",
        "credit_no_touch",
    )
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if token in blocked_exact:
            return False
        if any(token.startswith(prefix) for prefix in blocked_prefixes):
            return False
        if token.startswith("flow_not_confirmed:") and "/confirmed" in token:
            return False
    return True


def live_mode_date_violation(asof: dt.date, today: dt.date, historical_replay: bool, allow_current_live: bool) -> str:
    if historical_replay or allow_current_live:
        return ""
    if asof < today:
        return (
            f"Refusing live-mode run for old dated folder {asof.isoformat()} while today is {today.isoformat()}. "
            "Use --eod-live-planning for after-close/next-session planning with current Schwab quotes/GEX. "
            "Use --historical-replay only for deterministic old-date audit/backtest."
        )
    return ""


def stage2_mode_name(historical_replay: bool, eod_live_planning: bool) -> str:
    if historical_replay:
        return "historical_replay"
    if eod_live_planning:
        return "eod_live_planning"
    return "schwab_live"


def entry_gate_strict_pass(net_type: str, live_net, gate_target, eps: float = 1e-9) -> bool:
    live_net_f = fnum(live_net)
    gate_target_f = fnum(gate_target)
    if not (np.isfinite(live_net_f) and np.isfinite(gate_target_f)):
        return False
    if str(net_type or "").strip().lower() == "credit":
        return bool(live_net_f >= gate_target_f - eps)
    return bool(live_net_f <= gate_target_f + eps)


def _date_token_from_name(path: Path) -> str:
    match = DATE_TOKEN_RE.search(Path(path).name)
    return match.group(0) if match else ""


def _expected_input_date(base_dir: Path) -> str:
    name = Path(base_dir).name
    return name if DATE_TOKEN_RE.fullmatch(name) else ""


def _names_have_required_prefixes(paths: list[Path], expected_date: str = "") -> bool:
    names = [p.name for p in paths]
    for pref in REQ_CSV_PREFIXES:
        prefixes = [pref] + list(CSV_PREFIX_ALIASES.get(pref, []))
        if not any(
            any(name.startswith(pfx) for pfx in prefixes)
            and (not expected_date or expected_date in name)
            for name in names
        ):
            return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_commit() -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (cp.stdout or "").strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def unzip_inputs_if_needed(base_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(p for p in base_dir.glob("*.zip") if not p.name.startswith(BOT_EOD_PREFIX))
    if not zips:
        raise FileNotFoundError(f"No input CSV/ZIP files found in {base_dir}")
    expected_date = _expected_input_date(base_dir)
    # Re-extract if any ZIP is newer than the oldest extracted CSV
    existing = sorted(out_dir.glob("*.csv"))
    if existing:
        oldest_csv = min(p.stat().st_mtime for p in existing)
        newest_zip = max(p.stat().st_mtime for p in zips)
        has_required = _names_have_required_prefixes(existing, expected_date)
        if newest_zip <= oldest_csv and has_required:
            return  # cache is fresh
        # Stale cache — clear and re-extract
        for p in existing:
            p.unlink()
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
            if names:
                date_matches = [n for n in names if not expected_date or expected_date in Path(n).name]
                name = date_matches[0] if date_matches else names[0]
                target = out_dir / Path(name).name
                with zf.open(name, "r") as src:
                    target.write_bytes(src.read())
                continue
            nested_zips = sorted([n for n in zf.namelist() if n.lower().endswith(".zip")])
            for nested_name in nested_zips:
                with zf.open(nested_name, "r") as nested_src:
                    nested_bytes = nested_src.read()
                with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested_zf:
                    nested_csvs = sorted(
                        [n for n in nested_zf.namelist() if n.lower().endswith(".csv")]
                    )
                    if not nested_csvs:
                        continue
                    date_matches = [
                        n for n in nested_csvs if not expected_date or expected_date in Path(n).name
                    ]
                    nested_csv = date_matches[0] if date_matches else nested_csvs[0]
                    target = out_dir / Path(nested_csv).name
                    with nested_zf.open(nested_csv, "r") as src:
                        target.write_bytes(src.read())


def detect_asof_from_names(paths):
    pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    vals = []
    for p in paths:
        m = pat.search(p.name)
        if m:
            vals.append(m.group(1))
    if not vals:
        raise ValueError("Could not detect as-of date from filenames.")
    return sorted(vals)[-1]


def _parse_external_scanner_leg(text: str):
    m = re.search(
        r"\b([A-Z][A-Z0-9.\-]{0,9})\s+(\d{4}-\d{2}-\d{2})\s+([0-9]+(?:\.[0-9]+)?)([CP])\b",
        str(text or "").upper(),
    )
    if not m:
        return None
    return {
        "ticker": m.group(1).replace(".", ""),
        "expiry": m.group(2),
        "strike": float(m.group(3)),
        "right": m.group(4),
    }


def _external_scanner_quote_row(quotes: pd.DataFrame, leg: dict):
    if not isinstance(leg, dict) or quotes is None or quotes.empty:
        return None
    expiry = str(leg.get("expiry", "")).strip()
    strike = fnum(leg.get("strike"))
    if not expiry or not np.isfinite(strike):
        return None
    q = quotes[
        (quotes["ticker"].astype(str).str.upper() == str(leg.get("ticker", "")).upper())
        & (quotes["right"].astype(str).str.upper() == str(leg.get("right", "")).upper())
        & (quotes["expiry"].astype(str) == expiry)
        & (pd.to_numeric(quotes["strike"], errors="coerce").sub(float(strike)).abs() < 0.0001)
    ].copy()
    if q.empty:
        return None
    q["_liq"] = (
        pd.to_numeric(q.get("volume"), errors="coerce").fillna(0)
        + pd.to_numeric(q.get("open_interest"), errors="coerce").fillna(0)
    )
    return q.sort_values("_liq", ascending=False).iloc[0].to_dict()


def _external_scanner_contract_flow(long_q: dict, short_q: dict) -> str:
    """Conservative exact-leg flow check for external scanner candidates."""
    if not long_q:
        return "unknown"
    long_ask = fnum(long_q.get("ask_side_volume"))
    long_bid = fnum(long_q.get("bid_side_volume"))
    short_ask = fnum(short_q.get("ask_side_volume")) if short_q else math.nan
    short_bid = fnum(short_q.get("bid_side_volume")) if short_q else math.nan
    if not np.isfinite(long_ask):
        long_ask = 0.0
    if not np.isfinite(long_bid):
        long_bid = 0.0
    if not np.isfinite(short_ask):
        short_ask = 0.0
    if not np.isfinite(short_bid):
        short_bid = 0.0
    long_confirmed = long_ask >= 10 and long_ask >= max(1.0, 1.20 * long_bid)
    long_contra = long_bid >= 10 and long_bid >= max(1.0, 1.50 * long_ask)
    short_adverse = short_ask >= 25 and short_ask >= max(1.0, 1.50 * short_bid)
    if long_contra:
        return "contra"
    if long_confirmed and not short_adverse:
        return "confirmed"
    return "weak_or_ambiguous"


def _external_scanner_stage1_diag(conviction, contract_flow: str) -> str:
    tokens = []
    conv = fnum(conviction)
    if not np.isfinite(conv) or conv < 65:
        cv = "nan" if not np.isfinite(conv) else f"{conv:.0f}"
        tokens.append(f"stage1_conviction_below_yes_good:{cv}<65")
    flow = str(contract_flow or "").strip().lower()
    if flow == "contra":
        tokens.append("stage1_contract_flow_contra")
    elif flow in {"", "unknown", "weak_or_ambiguous"}:
        tokens.append("stage1_flow_weak_or_ambiguous")
        tokens.append(f"stage1_contract_flow_{flow or 'unknown'}")
    return ";".join(tokens)


def load_external_scanner_candidates(
    base: Path,
    asof_str: str,
    asof: dt.date,
    quotes: pd.DataFrame,
    screener=None,
    include_built_rows: bool = True,
) -> list[dict]:
    """Import local audited scanner structures into the daily candidate universe.

    These rows are not approvals.  They simply stop externally discovered,
    positive-EV structures from disappearing before Stage-2/live validation.
    """
    source_frames = []
    rec_path = base / f"options_scan_{asof_str}_audited_recommendations.csv"
    if rec_path.exists():
        rec_df = pd.read_csv(rec_path, low_memory=False)
        if not rec_df.empty:
            rec_df = rec_df.copy()
            rec_df["_coverage_source"] = "audited_recommendations"
            source_frames.append(rec_df)

    built_path = base / f"options_scan_{asof_str}_audited_built_rows.csv"
    if include_built_rows and built_path.exists():
        built_df = pd.read_csv(built_path, low_memory=False)
        if not built_df.empty:
            built_df = built_df.copy()
            built_df["_ev_num"] = pd.to_numeric(built_df.get("EV/ML"), errors="coerce")
            built_df["_pop_num"] = pd.to_numeric(built_df.get("POP"), errors="coerce")
            built_df["_conv_num"] = pd.to_numeric(built_df.get("Conviction"), errors="coerce")
            action_s = built_df.get("Action", pd.Series("", index=built_df.index)).astype(str)
            built_df = built_df[
                action_s.str.contains("BUY", case=False, na=False)
                & built_df["_ev_num"].notna()
                & (built_df["_ev_num"] >= 0.50)
                & (built_df["_pop_num"].fillna(0) >= 0.10)
                & (built_df["_conv_num"].fillna(0) >= 40)
            ].copy()
            if not built_df.empty:
                built_df = built_df.sort_values(
                    ["_ev_num", "_pop_num", "_conv_num"],
                    ascending=[False, False, False],
                ).head(60)
                built_df["_coverage_source"] = "audited_built_rows_top_ev"
                source_frames.append(built_df)

    if not source_frames:
        return []

    screener_map = {}
    if screener is not None and not screener.empty and "ticker" in screener.columns:
        sc_tmp = screener.copy()
        sc_tmp["ticker"] = sc_tmp["ticker"].astype(str).str.upper().str.replace(".", "", regex=False)
        for _, sc_row in sc_tmp.drop_duplicates("ticker", keep="last").iterrows():
            screener_map[str(sc_row.get("ticker", "")).upper()] = sc_row.to_dict()

    raw = pd.concat(source_frames, ignore_index=True, sort=False)
    rows = []
    seen = set()
    excluded_tickers = {"SPY", "QQQ", "IWM", "DIA", "VIX", "SPX", "NDX", "RUT"}
    for _, r in raw.iterrows():
        buy_leg = str(r.get("Buy leg", "") or r.get("Buy Leg", "") or "").strip()
        sell_leg = str(r.get("Sell leg", "") or r.get("Sell Leg", "") or "").strip()
        buy = _parse_external_scanner_leg(buy_leg)
        sell = _parse_external_scanner_leg(sell_leg)
        if not buy or not sell:
            continue
        if buy["ticker"] != sell["ticker"] or buy["expiry"] != sell["expiry"] or buy["right"] != sell["right"]:
            continue
        ticker = buy["ticker"]
        if ticker in excluded_tickers:
            continue
        right = buy["right"]
        long_strike = fnum(buy["strike"])
        short_strike = fnum(sell["strike"])
        if right == "C" and short_strike > long_strike:
            strategy = "Bull Call Debit"
            trade_direction = "bullish"
            breakeven = long_strike
        elif right == "P" and short_strike < long_strike:
            strategy = "Bear Put Debit"
            trade_direction = "bearish"
            breakeven = long_strike
        else:
            continue
        expiry = buy["expiry"]
        key = (ticker, expiry, right, round(float(long_strike), 4), round(float(short_strike), 4))
        if key in seen:
            continue
        seen.add(key)
        long_q = _external_scanner_quote_row(quotes, buy)
        short_q = _external_scanner_quote_row(quotes, sell)
        if not long_q or not short_q:
            continue
        width = abs(short_strike - long_strike)
        net_text = str(r.get("Net", "") or "")
        net_match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", net_text)
        net = fnum(net_match.group(0)) if net_match else math.nan
        if not np.isfinite(net) or net <= 0 or not np.isfinite(width) or width <= 0:
            continue
        debit_frac = net / width
        if debit_frac > 0.45:
            continue
        exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (exp_date - asof).days
        if dte <= 0:
            continue
        sc_meta = screener_map.get(ticker, {})
        bullish_premium = fnum(sc_meta.get("bullish_premium"))
        bearish_premium = fnum(sc_meta.get("bearish_premium"))
        total_directional_premium = bullish_premium + bearish_premium
        flow_premium_bias = math.nan
        if (
            np.isfinite(bullish_premium)
            and np.isfinite(bearish_premium)
            and total_directional_premium > 0
        ):
            flow_premium_bias = (bullish_premium - bearish_premium) / total_directional_premium
        if np.isfinite(flow_premium_bias) and abs(flow_premium_bias) >= 0.18:
            flow_direction = "bullish" if flow_premium_bias > 0 else "bearish"
            flow_confirmation = "confirmed"
        else:
            flow_direction = "neutral_or_ambiguous"
            flow_confirmation = "weak_or_ambiguous"
        issue_type = str(sc_meta.get("issue_type", "") or "").strip().lower()
        is_index = str(sc_meta.get("is_index", "") or "").strip().lower() in {"1", "true", "t", "yes", "y"}
        if is_index or (issue_type and issue_type not in {"common stock", "adr"}):
            continue
        if strategy == "Bull Call Debit":
            breakeven = long_strike + net
        else:
            breakeven = long_strike - net
        max_profit = max(0.0, (width - net) * 100.0)
        max_loss = net * 100.0
        conviction = fnum(r.get("Conviction"))
        if not np.isfinite(conviction):
            conviction = fnum(r.get("Conv"))
        optimal = "Yes-Prime" if conviction >= 80 else "Yes-Good" if conviction >= 65 else "Watch Only"
        contract_flow = _external_scanner_contract_flow(long_q, short_q)
        diag = _external_scanner_stage1_diag(conviction, contract_flow)
        source_name = str(r.get("_coverage_source", "external_scanner"))
        imported_ev_ml = fnum(r.get("EV/ML"))
        partial_ev_ml_asof = partial_ev_ml_debit(
            fnum(sc_meta.get("close")),
            sc_meta.get("iv30d"),
            dte,
            long_strike,
            short_strike,
            net,
            "bull" if strategy == "Bull Call Debit" else "bear",
        )
        ev_source = "partial_payoff_asof" if np.isfinite(partial_ev_ml_asof) else "unavailable"
        rows.append(
            {
                "ticker": ticker,
                "action": "BUY",
                "strategy": strategy,
                "track": "FIRE",
                "expiry": exp_date,
                "dte": dte,
                "long_leg": str(long_q.get("option_symbol", "")),
                "short_leg": str(short_q.get("option_symbol", "")),
                "long_strike": float(long_strike),
                "short_strike": float(short_strike),
                "width": float(width),
                "net": float(net),
                "net_type": "debit",
                "max_profit": float(max_profit),
                "max_loss": float(max_loss),
                "breakeven": float(breakeven),
                "conviction": int(round(conviction)) if np.isfinite(conviction) else 0,
                "tier": str(r.get("Size", "") or r.get("Sizing", "") or "External"),
                "optimal": optimal,
                "notes": (
                    f"External scanner candidate from {source_name}; "
                    f"imported_EV/ML={r.get('EV/ML', '')}; "
                    f"partial_EV/ML={partial_ev_ml_asof:.3f} "
                    f"({ev_source}); POP={r.get('POP', '')}; "
                    f"execution={r.get('Execution', '')}"
                ),
                "source": f"external_scanner:{source_name}",
                "coverage_source": source_name,
                "thesis": "External audited scanner candidate; requires daily live/risk approval.",
                "invalidation": "Follow daily live invalidation and entry gate.",
                "flow_direction": flow_direction,
                "flow_confirmation": flow_confirmation,
                "flow_premium_bias": flow_premium_bias,
                "external_trade_direction": trade_direction,
                "contract_flow_confirmation": contract_flow,
                "stage1_diagnostics": diag,
                "stage1_not_actionable_reason": diag,
                "stage1_flow_diagnostic": ";".join([t for t in diag.split(";") if t.startswith("stage1_flow_")]),
                "stage1_contract_flow_diagnostic": ";".join(
                    [t for t in diag.split(";") if t.startswith("stage1_contract_flow_")]
                ),
                "spot_asof_close": fnum(sc_meta.get("close")),
                "iv_rank": fnum(sc_meta.get("iv_rank")),
                "iv30d": fnum(sc_meta.get("iv30d")),
                "implied_move": fnum(sc_meta.get("implied_move")),
                "implied_move_perc": fnum(sc_meta.get("implied_move_perc")),
                "bullish_premium": fnum(sc_meta.get("bullish_premium")),
                "bearish_premium": fnum(sc_meta.get("bearish_premium")),
                "call_premium": fnum(sc_meta.get("call_premium")),
                "put_premium": fnum(sc_meta.get("put_premium")),
                "external_ev_ml": partial_ev_ml_asof,
                "external_ev_ml_imported": imported_ev_ml,
                "external_ev_ml_source": ev_source,
                "external_pop": fnum(r.get("POP")),
            }
        )
    return rows


def pick_csvs(base_dir: Path):
    unz = base_dir / "_unzipped_mode_a"
    unzip_inputs_if_needed(base_dir, unz)
    csvs = sorted(unz.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {unz}")
    expected_date = _expected_input_date(base_dir)

    out = {}
    for pref in REQ_CSV_PREFIXES:
        prefixes = [pref] + list(CSV_PREFIX_ALIASES.get(pref, []))
        matches = [p for p in csvs if any(p.name.startswith(pfx) for pfx in prefixes)]
        if expected_date:
            matches = [p for p in matches if expected_date in p.name]
        if not matches:
            suffix = f" for {expected_date}" if expected_date else ""
            raise FileNotFoundError(f"Missing required CSV prefix: {pref}{suffix}")
        out[pref] = sorted(matches)[-1]
    selected_dates = {_date_token_from_name(p) for p in out.values()}
    selected_dates.discard("")
    if len(selected_dates) > 1:
        detail = ", ".join(f"{k}{v.name}" for k, v in out.items())
        raise ValueError(f"Mixed daily input dates selected: {sorted(selected_dates)} from {detail}")
    return out


def resolve_chain_oi_overlay(path_text: str, out_dir: Path) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(
            [
                *path.glob("chain-oi-changes-*.csv"),
                *path.glob("chain-oi-changes-*.zip"),
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No chain-oi-changes CSV/ZIP found in overlay dir: {path}")
        path = candidates[-1]
    if not path.exists():
        raise FileNotFoundError(f"Missing chain OI overlay path: {path}")
    if path.suffix.lower() == ".csv":
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported chain OI overlay path; expected CSV or ZIP: {path}")
    overlay_dir = out_dir / "_overlay_inputs"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
        if not names:
            raise FileNotFoundError(f"No CSV inside chain OI overlay ZIP: {path}")
        preferred = [n for n in names if Path(n).name.startswith("chain-oi-changes-")]
        name = preferred[0] if preferred else names[0]
        target = overlay_dir / Path(name).name
        with zf.open(name, "r") as src:
            target.write_bytes(src.read())
    return target


def overlay_tickers_from_chain_oi(oi_df: pd.DataFrame) -> list[str]:
    if oi_df is None or oi_df.empty or "option_symbol" not in oi_df.columns:
        return []
    parsed = oi_df["option_symbol"].astype(str).map(parse_occ)
    tickers = set()
    for val in parsed[parsed.notna()]:
        if not val:
            continue
        ticker = str(val[0]).upper().strip()
        if ticker:
            tickers.add(ticker)
    return sorted(tickers)


def fetch_schwab_underlying_spots(tickers: list[str]) -> dict[str, float]:
    symbols = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not symbols:
        return {}
    try:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService, extract_quote_fields
    except Exception as exc:
        print(f"  [overlay] Schwab spot fetch unavailable: {exc}", file=sys.stderr)
        return {}
    try:
        cfg_live = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(config=cfg_live, interactive_login=False)
        out = {}
        for i in range(0, len(symbols), 80):
            batch = symbols[i : i + 80]
            payload = svc.get_quotes(batch)
            for requested in batch:
                raw = payload.get(requested) or payload.get(requested.upper()) or {}
                last, bid, ask = extract_quote_fields(raw)
                spot = last
                if (spot is None or not np.isfinite(fnum(spot))) and bid is not None and ask is not None:
                    if bid > 0 and ask > 0:
                        spot = 0.5 * (bid + ask)
                spot = fnum(spot)
                if np.isfinite(spot) and spot > 0:
                    out[requested.upper()] = spot
        return out
    except Exception as exc:
        print(f"  [overlay] Schwab spot fetch failed: {exc}", file=sys.stderr)
        return {}


def round_strike(x):
    try:
        return round(float(x), 3)
    except Exception:
        return math.nan


def build_leg_map(quotes: pd.DataFrame):
    q = quotes.copy()
    q["k"] = list(
        zip(
            q["ticker"].astype(str).str.upper(),
            q["right"].astype(str).str.upper(),
            q["expiry"],
            q["strike"].map(round_strike),
        )
    )
    return q.drop_duplicates("k").set_index("k")["option_symbol"].to_dict()


def strategy_right(strategy: str):
    s = str(strategy).strip()
    if s in {"Bull Call Debit", "Bear Call Credit"}:
        return "C"
    if s in {"Bear Put Debit", "Bull Put Credit"}:
        return "P"
    if s in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        return "IC"
    return ""


def action_cell(strategy: str, track: str, optimal: str):
    if optimal == "Watch Only":
        return "\U0001F525\U0001F7E8 WATCH ONLY" if str(track).upper() == "FIRE" else "\U0001F6E1\ufe0f\U0001F7E8 WATCH ONLY"
    s = str(strategy)
    if s == "Bull Call Debit":
        return "\U0001F525\U0001F7E6 BULL CALL DEBIT"
    if s == "Bear Put Debit":
        return "\U0001F525\U0001F7E7 BEAR PUT DEBIT"
    if s == "Bull Put Credit":
        return "\U0001F6E1\ufe0f\U0001F7E9 BULL PUT CREDIT"
    if s == "Bear Call Credit":
        return "\U0001F6E1\ufe0f\U0001F7E5 BEAR CALL CREDIT"
    if s == "Iron Condor":
        return "\U0001F6E1\ufe0f\U0001F7EA IRON CONDOR"
    if s == "Iron Butterfly":
        return "\U0001F6E1\ufe0f\U0001F7EA IRON BUTTERFLY"
    if s == "Long Iron Condor":
        return "\U0001F525\U0001F7EA LONG IRON CONDOR"
    return s.upper()

def strike_setup(
    strategy,
    long_strike,
    short_strike,
    width,
    long_put_strike=None,
    short_put_strike=None,
    short_call_strike=None,
    long_call_strike=None,
):
    ls = float(long_strike) if np.isfinite(fnum(long_strike)) else math.nan
    ss = float(short_strike) if np.isfinite(fnum(short_strike)) else math.nan
    w = float(width) if np.isfinite(fnum(width)) else math.nan
    if strategy == "Bull Call Debit":
        return f"Buy {ls:.2f}C / Sell {ss:.2f}C ({w:.2f}w)"
    if strategy == "Bear Put Debit":
        return f"Buy {ls:.2f}P / Sell {ss:.2f}P ({w:.2f}w)"
    if strategy == "Bull Put Credit":
        return f"Sell {ss:.2f}P / Buy {ls:.2f}P ({w:.2f}w)"
    if strategy == "Bear Call Credit":
        return f"Sell {ss:.2f}C / Buy {ls:.2f}C ({w:.2f}w)"
    if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        lp = fnum(long_put_strike)
        sp = fnum(short_put_strike)
        sc = fnum(short_call_strike)
        lc = fnum(long_call_strike)
        if np.isfinite(lp) and np.isfinite(sp) and np.isfinite(sc) and np.isfinite(lc):
            if strategy == "Long Iron Condor":
                return f"Buy {lp:.2f}P / Sell {sp:.2f}P + Buy {lc:.2f}C / Sell {sc:.2f}C"
            return f"Sell {sp:.2f}P / Buy {lp:.2f}P + Sell {sc:.2f}C / Buy {lc:.2f}C"
    return "N/A"


def parse_gate_value(entry_gate: str):
    m = re.match(r"^\s*(>=|<=)\s*([0-9]*\.?[0-9]+)\s*(cr|db)\s*$", str(entry_gate).strip(), re.I)
    if not m:
        return None, None, None
    op, val, unit = m.groups()
    return op, float(val), unit.lower()


def calc_target_max(net_type: str, width: float, net: float):
    if net_type == "credit":
        return net * 100.0, (width - net) * 100.0
    return (width - net) * 100.0, net * 100.0


def calc_be(strategy, long_strike, short_strike, net):
    ls = float(long_strike)
    ss = float(short_strike)
    if strategy == "Bull Call Debit":
        return ls + net
    if strategy == "Bear Put Debit":
        return ls - net
    if strategy == "Bull Put Credit":
        return ss - net
    if strategy == "Bear Call Credit":
        return ss + net
    return math.nan


def calc_be_text(row, net):
    strategy = str(row.get("strategy", "")).strip()
    if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        sp = fnum(row.get("short_put_strike", row.get("short_strike")))
        sc = fnum(row.get("short_call_strike"))
        lp = fnum(row.get("long_put_strike"))
        lc = fnum(row.get("long_call_strike"))
        if strategy == "Long Iron Condor" and np.isfinite(lp) and np.isfinite(lc) and np.isfinite(net):
            return f"{(lp - net):.2f} / {(lc + net):.2f}"
        if np.isfinite(sp) and np.isfinite(sc) and np.isfinite(net):
            return f"{(sp - net):.2f} / {(sc + net):.2f}"
        return "N/A"
    return px(calc_be(strategy, row.get("long_strike"), row.get("short_strike"), net))


def calc_reward_risk(net_type: str, width: float, net: float) -> float:
    w = fnum(width)
    n = fnum(net)
    if not np.isfinite(w) or not np.isfinite(n) or w <= 0 or n <= 0 or n >= w:
        return math.nan
    if str(net_type).strip().lower() == "credit":
        return n / max(1e-9, (w - n))
    return (w - n) / max(1e-9, n)


def money(x):
    return "N/A" if not np.isfinite(fnum(x)) else f"${float(x):,.2f}"


def px(x):
    return "N/A" if not np.isfinite(fnum(x)) else f"{float(x):.2f}"


def likelihood_strength(verdict: str, edge_pct: float, signals: float):
    v = str(verdict).strip().upper()
    e = fnum(edge_pct)
    n = fnum(signals)
    if v == "LOW_SAMPLE":
        return "Low Sample"
    if not np.isfinite(e):
        return "N/A"
    if e < 0:
        return "Negative-Strong" if e <= -15 else "Negative"
    if np.isfinite(n) and n < 100:
        return "Low Sample"
    if e >= 25:
        return "Strong"
    if e >= 10:
        return "Moderate"
    return "Weak"


def strategy_is_credit_family(strategy: str) -> bool:
    s = str(strategy or "").strip()
    return s in {"Bull Put Credit", "Bear Call Credit", "Iron Condor", "Iron Butterfly"}


def normalize_track(track: str, strategy: str) -> str:
    t = str(track or "").strip().upper()
    if t in {"FIRE", "SHIELD"}:
        return t
    s = str(strategy or "").strip()
    if s in {"Bull Put Credit", "Bear Call Credit", "Iron Condor", "Iron Butterfly"}:
        return "SHIELD"
    if s in {"Bull Call Debit", "Bear Put Debit", "Long Iron Condor"}:
        return "FIRE"
    return "UNKNOWN"


def fetch_open_positions_from_schwab(cache_csv: Path):
    try:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    except Exception as exc:
        return False, f"import_error:{exc}"
    try:
        cfg_live = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(cfg_live)
        svc.connect()
        cli = svc._client
        resp = cli.get_accounts(fields=[cli.Account.Fields.POSITIONS])
        resp.raise_for_status()
        raw = resp.json()
        accounts = raw if isinstance(raw, list) else [raw]
        rows = []
        for a in accounts:
            sec = a.get("securitiesAccount", {}) if isinstance(a, dict) else {}
            acct_num = sec.get("accountNumber", "")
            for p in sec.get("positions", []) or []:
                inst = p.get("instrument") or {}
                rows.append(
                    {
                        "account_number": acct_num,
                        "symbol": inst.get("symbol", ""),
                        "description": inst.get("description", ""),
                        "asset_type": inst.get("assetType", ""),
                        "position_type": p.get("positionType", ""),
                        "long_quantity": p.get("longQuantity"),
                        "short_quantity": p.get("shortQuantity"),
                        "average_price": p.get("averagePrice"),
                        "market_value": p.get("marketValue"),
                        "maintenance_requirement": p.get("maintenanceRequirement"),
                        "current_day_profit_loss": p.get("currentDayProfitLoss"),
                        "current_day_profit_loss_pct": p.get("currentDayProfitLossPercentage"),
                    }
                )
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cache_csv, index=False)
        return True, ""
    except Exception as exc:
        return False, f"fetch_error:{exc}"


def build_portfolio_risk_book(open_positions_csv: Path):
    if not open_positions_csv.exists():
        return {"ok": False, "error": f"missing_open_positions_csv:{open_positions_csv}"}
    try:
        pos = load_open_positions(open_positions_csv)
    except Exception as exc:
        return {"ok": False, "error": f"load_open_positions_failed:{exc}"}
    if pos.empty:
        return {
            "ok": True,
            "total_risk": 0.0,
            "short_put_risk": 0.0,
            "symbol_risk": {},
            "short_put_expiry_risk": {},
            "option_underlyings": [],
        }
    is_option = (
        pos["asset_type"].astype(str).str.upper().eq("OPTION")
        | pos["strategy"].astype(str).str.contains("Option", case=False, na=False)
        | pos["symbol"].astype(str).str.contains(r"\d{6}[CP]\d{8}", na=False)
    )
    pos = pos[is_option].copy()
    option_underlyings_all = sorted(
        {
            str(x).upper().strip()
            for x in pos.get("underlying", pd.Series(dtype=str)).dropna().tolist()
            if str(x).strip() and str(x).strip().upper() not in {"NAN", "UNKNOWN"}
        }
    )
    pos["risk"] = pos["risk"].map(fnum).fillna(np.nan)
    pos = pos[pos["risk"].notna()].copy()
    pos["risk"] = pos["risk"].abs()
    pos = pos[pos["risk"] > 0].copy()
    if pos.empty:
        return {
            "ok": True,
            "total_risk": 0.0,
            "short_put_risk": 0.0,
            "symbol_risk": {},
            "short_put_expiry_risk": {},
            "option_underlyings": option_underlyings_all,
        }
    short_put_mask = pos["strategy"].astype(str).isin(["Short Put Option", "Bull Put Credit"])
    symbol_risk = (
        pos.groupby(pos["underlying"].astype(str).str.upper().str.strip())["risk"].sum().to_dict()
    )
    expiry_risk = {}
    if short_put_mask.any():
        sp = pos[short_put_mask & pos["expiry"].notna()].copy()
        if not sp.empty:
            expiry_risk = sp.groupby(sp["expiry"].dt.date.astype(str))["risk"].sum().to_dict()
    total_risk = float(pos["risk"].sum())
    short_put_risk = float(pos.loc[short_put_mask, "risk"].sum())
    return {
        "ok": True,
        "total_risk": total_risk,
        "short_put_risk": short_put_risk,
        "symbol_risk": {str(k): float(v) for k, v in symbol_risk.items()},
        "short_put_expiry_risk": {str(k): float(v) for k, v in expiry_risk.items()},
        "option_underlyings": option_underlyings_all,
    }


def candidate_uses_short_put_risk(strategy: str) -> bool:
    s = str(strategy or "").strip()
    return s in {"Bull Put Credit", "Iron Condor", "Iron Butterfly"}


def _hist_quote_map(quotes: pd.DataFrame) -> pd.DataFrame:
    q = quotes.copy()
    if "option_symbol" not in q.columns:
        return pd.DataFrame()
    q["option_symbol"] = q["option_symbol"].astype(str).str.strip()
    q = q[q["option_symbol"] != ""].copy()
    if q.empty:
        return pd.DataFrame()
    return q.drop_duplicates("option_symbol", keep="last").set_index("option_symbol", drop=False)


def _hist_leg(qmap: pd.DataFrame, symbol: object) -> dict:
    sym = str(symbol or "").strip()
    if not sym or qmap.empty or sym not in qmap.index:
        return {
            "symbol": sym,
            "missing": True,
            "bid": math.nan,
            "ask": math.nan,
            "delta": math.nan,
            "delta_source": "",
            "iv": math.nan,
            "right": "",
            "strike": math.nan,
            "expiry": "",
            "volume": math.nan,
            "open_interest": math.nan,
        }
    row = qmap.loc[sym]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return {
        "symbol": sym,
        "missing": False,
        "bid": fnum(row.get("bid")),
        "ask": fnum(row.get("ask")),
        "delta": _safe_delta(row.get("delta")),
        "delta_source": "quoted" if np.isfinite(_safe_delta(row.get("delta"))) else "",
        "iv": fnum(row.get("iv")),
        "right": str(row.get("right", "") or "").strip().upper(),
        "strike": fnum(row.get("strike")),
        "expiry": str(row.get("expiry", ""))[:10],
        "volume": fnum(row.get("volume")),
        "open_interest": fnum(row.get("open_interest")),
    }


def _hist_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _hist_estimate_delta(leg: dict, spot: float, asof: dt.date) -> dict:
    if np.isfinite(fnum(leg.get("delta"))):
        return leg
    if bool(leg.get("missing")) or not np.isfinite(spot) or spot <= 0:
        return leg
    strike = fnum(leg.get("strike"))
    iv = fnum(leg.get("iv"))
    right = str(leg.get("right", "") or "").upper().strip()
    expiry_raw = str(leg.get("expiry", "") or "")[:10]
    try:
        expiry = dt.datetime.strptime(expiry_raw, "%Y-%m-%d").date()
    except Exception:
        return leg
    if not (np.isfinite(strike) and strike > 0 and np.isfinite(iv) and iv > 0 and right in {"C", "P"}):
        return leg
    if iv > 5.0:
        iv = iv / 100.0
    if iv <= 0 or iv > 5.0:
        return leg
    dte = max(1, (expiry - asof).days)
    t = dte / 365.0
    r = 0.04
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    except (ValueError, ZeroDivisionError):
        return leg
    if not np.isfinite(d1):
        return leg
    est = _hist_norm_cdf(d1) if right == "C" else _hist_norm_cdf(d1) - 1.0
    if not np.isfinite(est) or abs(est) > 1.0:
        return leg
    out = dict(leg)
    out["delta"] = float(est)
    out["delta_source"] = "bs_iv_estimate"
    return out


def _hist_valid_bid_ask(leg: dict, need_bid: bool, need_ask: bool) -> bool:
    if bool(leg.get("missing")):
        return False
    if need_bid and not np.isfinite(fnum(leg.get("bid"))):
        return False
    if need_ask and not np.isfinite(fnum(leg.get("ask"))):
        return False
    return True


def _hist_spread_net(net_type: str, short_leg: dict, long_leg: dict) -> float:
    nt = str(net_type or "").strip().lower()
    if nt == "credit":
        if not (_hist_valid_bid_ask(short_leg, True, False) and _hist_valid_bid_ask(long_leg, False, True)):
            return math.nan
        return fnum(short_leg.get("bid")) - fnum(long_leg.get("ask"))
    if nt == "debit":
        if not (_hist_valid_bid_ask(long_leg, False, True) and _hist_valid_bid_ask(short_leg, True, False)):
            return math.nan
        return fnum(long_leg.get("ask")) - fnum(short_leg.get("bid"))
    return math.nan


def _hist_parse_invalidation(text: object):
    raw = str(text or "")
    m = re.search(r"(<=|>=|<|>)\s*(-?[0-9]+(?:\.[0-9]+)?)", raw)
    if not m:
        return "", math.nan
    try:
        return m.group(1), float(m.group(2))
    except (TypeError, ValueError):
        return "", math.nan


def _hist_invalidation_breached(op: str, level: float, price: float) -> bool:
    if not op or not np.isfinite(level) or not np.isfinite(price):
        return False
    if op == "<":
        return price < level
    if op == "<=":
        return price <= level
    if op == ">":
        return price > level
    if op == ">=":
        return price >= level
    return False


def _hist_entry_structure(strategy: str, row: pd.Series, spot: float, live_net: float) -> tuple:
    s = str(strategy or "").strip()
    if not np.isfinite(spot) or spot <= 0:
        return False, "missing_asof_underlying_close"

    long_strike = fnum(row.get("long_strike"))
    short_strike = fnum(row.get("short_strike"))
    long_put = fnum(row.get("long_put_strike"))
    short_put = fnum(row.get("short_put_strike"))
    short_call = fnum(row.get("short_call_strike"))
    long_call = fnum(row.get("long_call_strike"))

    if s == "Bull Call Debit":
        if not (np.isfinite(long_strike) and np.isfinite(short_strike) and long_strike < short_strike):
            return False, "bull_call_strike_order_invalid"
        return True, "ok"
    if s == "Bear Put Debit":
        if not (np.isfinite(long_strike) and np.isfinite(short_strike) and long_strike > short_strike):
            return False, "bear_put_strike_order_invalid"
        return True, "ok"
    if s == "Bull Put Credit":
        if not (np.isfinite(long_strike) and np.isfinite(short_strike) and long_strike < short_strike):
            return False, "bull_put_strike_order_invalid"
        if np.isfinite(live_net) and spot <= (short_strike - live_net):
            return False, "spot_below_bull_put_breakeven"
        return True, "ok"
    if s == "Bear Call Credit":
        if not (np.isfinite(short_strike) and np.isfinite(long_strike) and short_strike < long_strike):
            return False, "bear_call_strike_order_invalid"
        if np.isfinite(live_net) and spot >= (short_strike + live_net):
            return False, "spot_above_bear_call_breakeven"
        return True, "ok"
    if s == "Long Iron Condor":
        if not (
            np.isfinite(long_put)
            and np.isfinite(short_put)
            and np.isfinite(short_call)
            and np.isfinite(long_call)
            and short_put < long_put < long_call < short_call
        ):
            return False, "long_condor_strike_order_invalid"
        return True, "ok"
    if s in {"Iron Condor", "Iron Butterfly"}:
        if not (
            np.isfinite(long_put)
            and np.isfinite(short_put)
            and np.isfinite(short_call)
            and np.isfinite(long_call)
            and long_put < short_put <= short_call < long_call
        ):
            return False, "condor_strike_order_invalid"
        if not (short_put < spot < short_call):
            return False, "spot_outside_short_strikes"
        if np.isfinite(live_net):
            lower_be = short_put - live_net
            upper_be = short_call + live_net
            if not (lower_be < spot < upper_be):
                return False, "spot_outside_condor_breakevens"
        return True, "ok"
    return True, "ok"


def build_historical_replay_live_table(
    shortlist: pd.DataFrame,
    quotes: pd.DataFrame,
    spot_map: dict,
    asof_str: str,
    live_csv: Path,
    live_final_csv: Path,
) -> int:
    """Build a Stage-2-compatible table from dated local UW files.

    This is for audit/backtest replay only. It deliberately does not call Schwab
    or yfinance for current quotes, because that would mix today's market state
    into an old-date decision review.
    """
    qmap = _hist_quote_map(quotes)
    asof = dt.datetime.strptime(asof_str, "%Y-%m-%d").date()
    rows = []
    for _, row in shortlist.iterrows():
        rec = row.to_dict()
        ticker = str(row.get("ticker", "")).strip().upper()
        strategy = str(row.get("strategy", "")).strip()
        net_type = str(row.get("net_type", "")).strip().lower()
        spot = fnum(spot_map.get(ticker))
        width = fnum(row.get("width"))

        short_leg = _hist_estimate_delta(_hist_leg(qmap, row.get("short_leg")), spot, asof)
        long_leg = _hist_estimate_delta(_hist_leg(qmap, row.get("long_leg")), spot, asof)
        short_put_leg = _hist_estimate_delta(_hist_leg(qmap, row.get("short_put_leg")), spot, asof)
        long_put_leg = _hist_estimate_delta(_hist_leg(qmap, row.get("long_put_leg")), spot, asof)
        short_call_leg = _hist_estimate_delta(_hist_leg(qmap, row.get("short_call_leg")), spot, asof)
        long_call_leg = _hist_estimate_delta(_hist_leg(qmap, row.get("long_call_leg")), spot, asof)

        live_net = math.nan
        missing_quote = False
        if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            condor_type = "debit" if strategy == "Long Iron Condor" else "credit"
            put_net = _hist_spread_net(condor_type, short_put_leg, long_put_leg)
            call_net = _hist_spread_net(condor_type, short_call_leg, long_call_leg)
            if np.isfinite(put_net) and np.isfinite(call_net):
                live_net = put_net + call_net
            else:
                missing_quote = True
            put_width = abs(fnum(short_put_leg.get("strike")) - fnum(long_put_leg.get("strike")))
            call_width = abs(fnum(long_call_leg.get("strike")) - fnum(short_call_leg.get("strike")))
            width_live = max(put_width if np.isfinite(put_width) else math.nan, call_width if np.isfinite(call_width) else math.nan)
            if not np.isfinite(width_live):
                width_live = width
        else:
            live_net = _hist_spread_net(net_type, short_leg, long_leg)
            missing_quote = not np.isfinite(live_net)
            width_live = abs(fnum(short_leg.get("strike")) - fnum(long_leg.get("strike")))
            if not np.isfinite(width_live):
                width_live = width

        entry_ok, entry_reason = _hist_entry_structure(strategy, row, spot, live_net)
        _, gate_target, _ = parse_gate_value(str(row.get("entry_gate", "")))
        gate_pass = False
        if np.isfinite(live_net) and np.isfinite(gate_target):
            gate_pass = live_net <= gate_target if net_type == "debit" else live_net >= gate_target

        if not np.isfinite(spot):
            live_status = "missing_underlying_quote"
        elif not entry_ok:
            live_status = "invalid_entry_structure"
        elif missing_quote:
            live_status = "missing_live_quote"
        elif not gate_pass:
            live_status = "fails_live_entry_gate"
        else:
            live_status = "ok_live"

        width_for_max = width_live if np.isfinite(width_live) else width
        if strategy in {"Iron Condor", "Iron Butterfly"} and np.isfinite(width_for_max) and np.isfinite(live_net):
            live_max_profit = live_net * 100.0
            live_max_loss = max(0.0, width_for_max - live_net) * 100.0
        elif strategy == "Long Iron Condor" and np.isfinite(width_for_max) and np.isfinite(live_net):
            live_max_profit = max(0.0, width_for_max - live_net) * 100.0
            live_max_loss = live_net * 100.0
        else:
            live_max_profit, live_max_loss = calc_target_max(net_type, width_for_max, live_net)

        inv_op, inv_level = _hist_parse_invalidation(row.get("invalidation", ""))
        rec.update(
            {
                "live_status": live_status,
                "is_final_live_valid": bool(live_status == "ok_live"),
                "invalidation_breached_live": bool(_hist_invalidation_breached(inv_op, inv_level, spot)),
                "invalidation_rule_op": inv_op,
                "invalidation_rule_level": inv_level,
                "invalidation_eval_price_live": spot,
                "live_net_bid_ask": live_net,
                "live_max_profit": live_max_profit,
                "live_max_loss": live_max_loss,
                "gate_pass_live": bool(gate_pass),
                "short_bid_live": short_leg.get("bid"),
                "short_ask_live": short_leg.get("ask"),
                "short_delta_live": short_leg.get("delta"),
                "short_delta_source_live": short_leg.get("delta_source"),
                "long_bid_live": long_leg.get("bid"),
                "long_ask_live": long_leg.get("ask"),
                "long_delta_live": long_leg.get("delta"),
                "long_delta_source_live": long_leg.get("delta_source"),
                "short_put_bid_live": short_put_leg.get("bid"),
                "short_put_ask_live": short_put_leg.get("ask"),
                "short_put_delta_live": short_put_leg.get("delta"),
                "short_put_delta_source_live": short_put_leg.get("delta_source"),
                "long_put_bid_live": long_put_leg.get("bid"),
                "long_put_ask_live": long_put_leg.get("ask"),
                "short_call_bid_live": short_call_leg.get("bid"),
                "short_call_ask_live": short_call_leg.get("ask"),
                "short_call_delta_live": short_call_leg.get("delta"),
                "short_call_delta_source_live": short_call_leg.get("delta_source"),
                "long_call_bid_live": long_call_leg.get("bid"),
                "long_call_ask_live": long_call_leg.get("ask"),
                "spot_live_last": spot,
                "spot_live_bid": math.nan,
                "spot_live_ask": math.nan,
                "width_live": width_live,
                "entry_structure_ok_live": bool(entry_ok),
                "entry_structure_reason_live": entry_reason,
                "historical_replay": True,
                "historical_replay_asof": asof_str,
                "chain_status_live": "HISTORICAL_REPLAY",
                "chain_query_symbol_live": ticker,
            }
        )
        rows.append(rec)

    replay = pd.DataFrame(rows)
    replay.to_csv(live_csv, index=False)
    replay.to_csv(live_final_csv, index=False)
    return int(len(replay))


def run():
    ap = argparse.ArgumentParser(description="MODE A two-stage runner (discovery + live execution)")
    ap.add_argument("--base-dir", default=r"c:\uw_root\2026-02-05")
    ap.add_argument(
        "--chain-oi-overlay",
        default="",
        help="Optional next-day chain-oi-changes CSV/ZIP/dir overlay. Keeps the base EOD date, but allows OI rows from the overlay date.",
    )
    ap.add_argument("--config", default=str((Path(__file__).resolve().parent / "rulebook_config_goal_holistic.yaml")))
    ap.add_argument("--out-dir", default=r"c:\uw_root\out")
    ap.add_argument("--top-trades", type=int, default=20)
    ap.add_argument("--output", default="")
    ap.add_argument(
        "--strict-stage2",
        action="store_true",
        help="Deprecated compatibility flag; Stage-2 is strict by default unless --allow-stale-stage2 is passed.",
    )
    ap.add_argument(
        "--allow-stale-stage2",
        action="store_true",
        help="Opt in to reusing existing same-date live files if Stage-2 live pricing fails.",
    )
    ap.add_argument(
        "--historical-replay",
        action="store_true",
        help="Replay an old daily folder using dated local UW quotes and as-of stock close instead of current Schwab live quotes.",
    )
    ap.add_argument(
        "--eod-live-planning",
        action="store_true",
        help=(
            "Default for normal runs: use dated EOD files for discovery, then refresh Schwab quotes "
            "and live chain GEX now for next-session planning."
        ),
    )
    ap.add_argument(
        "--allow-current-live-on-historical-date",
        action="store_true",
        help="Legacy override: allow current Schwab live quotes/GEX on a folder dated before today.",
    )
    ap.add_argument(
        "--no-auto-collect-uw-gex",
        action="store_true",
        help="Deprecated compatibility flag; live GEX is calculated from Schwab option-chain snapshots.",
    )
    ap.add_argument(
        "--uw-remote-debugging-url",
        default="http://127.0.0.1:9222",
        help="Chrome/Atlas remote debugging URL used for authenticated UW GEX collection.",
    )
    ap.add_argument(
        "--uw-gex-wait-sec",
        type=float,
        default=1.0,
        help="Seconds to wait after each UW GEX ticker navigation.",
    )
    ap.add_argument(
        "--uw-gex-max-tickers",
        type=int,
        default=0,
        help="Maximum shortlist tickers to collect UW GEX for; 0 means all missing shortlist tickers.",
    )
    args = ap.parse_args()
    if args.historical_replay and args.eod_live_planning:
        ap.error("--historical-replay and --eod-live-planning are mutually exclusive")
    args.eod_live_planning = not bool(args.historical_replay)
    run_started_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    csvs = pick_csvs(base)
    asof_str = detect_asof_from_names(list(csvs.values()))
    chain_oi_overlay_csv = ""
    chain_oi_overlay_date = ""
    if str(args.chain_oi_overlay or "").strip():
        overlay_csv = resolve_chain_oi_overlay(str(args.chain_oi_overlay), out_dir)
        csvs["chain-oi-changes-"] = overlay_csv
        chain_oi_overlay_csv = str(overlay_csv)
        try:
            chain_oi_overlay_date = detect_asof_from_names([overlay_csv])
        except Exception:
            chain_oi_overlay_date = ""
    asof = dt.datetime.strptime(asof_str, "%Y-%m-%d").date()
    live_mode_error = live_mode_date_violation(
        asof,
        dt.date.today(),
        bool(args.historical_replay),
        bool(args.allow_current_live_on_historical_date or args.eod_live_planning),
    )
    if live_mode_error:
        raise RuntimeError(live_mode_error)

    if not args.output:
        output_path = base / f"anu-expert-trade-table-{asof_str}.md"
    else:
        output_path = Path(args.output).resolve()

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    whale_flow = load_whale_flow_source(base, asof_str, cfg)
    bot_eod_source = whale_flow.source_path
    whale_source_name = bot_eod_source.name
    whale_tables = whale_flow.as_rank_tables()
    whale_symbol_summary_csv = out_dir / f"whale-symbol-summary-{asof_str}.csv"
    whale_top_trades_csv = out_dir / f"whale-top-trades-{asof_str}.csv"
    whale_flow.symbol_summary.to_csv(whale_symbol_summary_csv, index=False)
    whale_flow.top_trades.to_csv(whale_top_trades_csv, index=False)
    print(
        f"Loaded {whale_flow.source_label} whale source: "
        f"{whale_source_name}; scanned={whale_flow.total_rows:,}; "
        f"yes_prime={whale_flow.yes_prime_rows:,}; symbols={len(whale_flow.symbol_summary):,}",
        file=sys.stderr,
    )
    approval_cfg = cfg.get("approval", {}) if isinstance(cfg, dict) else {}
    engine_cfg = cfg.get("engine", {}) if isinstance(cfg, dict) else {}
    # Width-based entry gate tolerance — read early so we can pass to the pricer subprocess.
    entry_tol_width_pct = fnum(approval_cfg.get("entry_tolerance_width_pct", 0.025))
    entry_tol_floor = fnum(approval_cfg.get("entry_tolerance_floor", 0.25))
    if not np.isfinite(entry_tol_width_pct) or entry_tol_width_pct < 0:
        entry_tol_width_pct = 0.025
    if not np.isfinite(entry_tol_floor) or entry_tol_floor < 0:
        entry_tol_floor = 0.25
    discovery_multiplier = fnum(engine_cfg.get("discovery_multiplier", 5))
    if not np.isfinite(discovery_multiplier) or discovery_multiplier < 1:
        discovery_multiplier = 5
    configured_external_scanner_raw = engine_cfg.get("external_scanner_mode", "off")
    if isinstance(configured_external_scanner_raw, bool):
        configured_external_scanner_mode = "off" if configured_external_scanner_raw is False else "on"
    else:
        configured_external_scanner_mode = str(configured_external_scanner_raw).strip().lower()
    if configured_external_scanner_mode in {"", "0", "false", "no", "none", "disabled"}:
        configured_external_scanner_mode = "off"
    if configured_external_scanner_mode and configured_external_scanner_mode != "off":
        print(
            "  [coverage] Ignoring external scanner artifacts for daily pipeline; "
            "using dated UW source files only.",
            file=sys.stderr,
        )
    external_scanner_mode = "off"
    show_external_scanner_coverage_section = False
    discovery_top = max(int(args.top_trades), int(round(int(args.top_trades) * float(discovery_multiplier))))
    # EOD discovery has millions of rows and is materially richer than the
    # final display table.  Do not let the final report size decide how many
    # candidates get a Schwab Stage-2 quote/GEX review.  This is intentionally
    # a pre-approval coverage fix, not a gate loosening: broader EOD candidates
    # still have to clear the same live-entry, likelihood, flow, GEX, and
    # High/Medium/Watch classifiers later.
    eod_stage2_cap = fnum(engine_cfg.get("eod_stage2_candidate_cap", np.nan))
    if not np.isfinite(eod_stage2_cap) or eod_stage2_cap <= 0:
        eod_stage2_cap = max(200, int(args.top_trades) * 10)
    discovery_top = max(int(discovery_top), int(eod_stage2_cap))
    final_max_per_ticker = int(engine_cfg.get("final_max_trades_per_ticker", 1))
    final_max_per_ticker = max(1, final_max_per_ticker)
    min_shield_in_output = int(engine_cfg.get("min_shield_in_output", 0))
    backtest_min_signals = fnum(approval_cfg.get("min_signals", 100))
    if not np.isfinite(backtest_min_signals) or backtest_min_signals <= 0:
        backtest_min_signals = 100
    hot_df = pd.read_csv(csvs["hot-chains-"], low_memory=False)
    oi_df = pd.read_csv(csvs["chain-oi-changes-"], low_memory=False)
    dp_df = pd.read_csv(csvs["dp-eod-report-"], low_memory=False)
    sc_df = pd.read_csv(csvs["stock-screener-"], low_memory=False)
    overlay_spot_map = {}
    if chain_oi_overlay_csv:
        overlay_tickers = overlay_tickers_from_chain_oi(oi_df)
        existing_spots = {}
        if "ticker" in sc_df.columns and "close" in sc_df.columns:
            existing_spots = (
                sc_df.assign(ticker=sc_df["ticker"].astype(str).str.upper().str.strip())
                .drop_duplicates("ticker")
                .set_index("ticker")["close"]
                .map(fnum)
                .to_dict()
            )
        missing_spot_tickers = [
            t for t in overlay_tickers
            if (not np.isfinite(fnum(existing_spots.get(t)))) or fnum(existing_spots.get(t)) <= 0
        ]
        if missing_spot_tickers and not args.historical_replay:
            overlay_spot_map = fetch_schwab_underlying_spots(missing_spot_tickers)
            print(
                f"  [overlay] fetched Schwab spots for {len(overlay_spot_map)}/{len(missing_spot_tickers)} missing overlay tickers",
                file=sys.stderr,
            )
        elif missing_spot_tickers:
            print(
                f"  [overlay] historical replay: {len(missing_spot_tickers)} overlay tickers lack dated spot and will stay diagnostic-only",
                file=sys.stderr,
            )
        sc_df = apply_chain_oi_overlay_to_screener(sc_df, oi_df, overlay_spot_map=overlay_spot_map)

    ensure_cols(hot_df, csvs["hot-chains-"].name, ["option_symbol", "date", "bid", "ask", "volume", "open_interest"])
    ensure_cols(
        oi_df,
        csvs["chain-oi-changes-"].name,
        ["option_symbol", "curr_date", "last_bid", "last_ask", "curr_oi", "volume"],
    )
    ensure_cols(
        sc_df,
        csvs["stock-screener-"].name,
        [
            "ticker",
            "close",
            "issue_type",
            "is_index",
            "next_earnings_date",
            "bullish_premium",
            "bearish_premium",
            "call_premium",
            "put_premium",
            "put_call_ratio",
        ],
    )
    spot_map = (
        sc_df.assign(ticker=sc_df["ticker"].astype(str).str.upper().str.strip())
        .drop_duplicates("ticker")
        .set_index("ticker")["close"]
        .map(fnum)
        .to_dict()
    )
    _sc_norm = sc_df.assign(ticker=sc_df["ticker"].astype(str).str.upper().str.strip()).drop_duplicates("ticker")
    sector_map = _sc_norm.set_index("ticker")["sector"].dropna().to_dict() if "sector" in sc_df.columns else {}
    if not sector_map:
        print("  [warn] sector_map is empty — sector concentration cap will treat all tickers as 'Unknown'", file=sys.stderr)
    playbook_cfg = cfg.get("playbook", {}) if isinstance(cfg, dict) else {}
    risk_limits_cfg = playbook_cfg.get("risk_limits", {}) if isinstance(playbook_cfg, dict) else {}
    position_mgmt_cfg = playbook_cfg.get("position_management", {}) if isinstance(playbook_cfg, dict) else {}
    take_profit_credit_pct = fnum(position_mgmt_cfg.get("take_profit_pct_credit_max_profit", 0.50))
    take_profit_debit_pct = fnum(position_mgmt_cfg.get("take_profit_pct_debit_cost", 0.80))
    stop_loss_credit_pct = fnum(position_mgmt_cfg.get("stop_loss_pct_credit_max_loss", 0.50))
    stop_loss_debit_pct = fnum(position_mgmt_cfg.get("stop_loss_pct_debit_max_loss", 0.45))
    if not np.isfinite(take_profit_credit_pct) or take_profit_credit_pct <= 0:
        take_profit_credit_pct = 0.50
    if not np.isfinite(take_profit_debit_pct) or take_profit_debit_pct <= 0:
        take_profit_debit_pct = 0.80
    if not np.isfinite(stop_loss_credit_pct) or stop_loss_credit_pct <= 0:
        stop_loss_credit_pct = 0.50
    if not np.isfinite(stop_loss_debit_pct) or stop_loss_debit_pct <= 0:
        stop_loss_debit_pct = 0.45
    max_sector_share = fnum(risk_limits_cfg.get("max_sector_share", 1.0))
    if not np.isfinite(max_sector_share) or max_sector_share <= 0:
        max_sector_share = 1.0
    max_strike_distance_pct = fnum(cfg.get("gates", {}).get("max_strike_distance_pct", 0.80))
    if not np.isfinite(max_strike_distance_pct) or max_strike_distance_pct <= 0:
        max_strike_distance_pct = math.nan
    _ = dp_df  # loaded intentionally; stage-1 model already relies on screener + quotes + whale tables.

    oi_quote_asof = dt.datetime.strptime(chain_oi_overlay_date, "%Y-%m-%d").date() if chain_oi_overlay_date else asof
    quotes = build_quotes(
        hot_df,
        oi_df,
        asof,
        csvs["hot-chains-"].name,
        csvs["chain-oi-changes-"].name,
        hot_asof=asof,
        oi_asof=oi_quote_asof,
    )
    discovery_cfg = copy.deepcopy(cfg)
    if chain_oi_overlay_csv:
        discovery_pricing_cfg = discovery_cfg.setdefault("pricing", {})
        source_kinds = list(discovery_pricing_cfg.get("executable_source_kinds") or [])
        if "oi" not in source_kinds:
            source_kinds.append("oi")
        discovery_pricing_cfg["executable_source_kinds"] = source_kinds
    build_best_candidates._force_historical = bool(args.historical_replay or asof < dt.date.today())
    build_best_candidates._local_root = base.parent
    best = build_best_candidates(asof, discovery_cfg, sc_df, quotes, whale_tables, top_trades=discovery_top)

    external_scanner_candidates = []

    if not best:
        raise RuntimeError("No stage-1 candidates produced.")

    leg_map = build_leg_map(quotes)
    shortlist_rows = []
    dropped_stage1 = []

    def strike_sanity_ok(
        ticker: str,
        strategy: str,
        long_strike_v: float,
        short_strike_v: float,
        long_put_v: float,
        short_put_v: float,
        short_call_v: float,
        long_call_v: float,
    ) -> bool:
        if not np.isfinite(max_strike_distance_pct):
            return True
        spot = fnum(spot_map.get(ticker))
        if not np.isfinite(spot) or spot <= 0:
            return True
        s = str(strategy).strip()
        to_check = []
        if s in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            to_check.extend([long_put_v, short_put_v, short_call_v, long_call_v])
        else:
            to_check.extend([long_strike_v, short_strike_v])
        for strike_val in to_check:
            x = fnum(strike_val)
            if not np.isfinite(x) or x <= 0:
                continue
            dist = abs((x / spot) - 1.0)
            if dist > max_strike_distance_pct:
                return False
        return True

    for r in best:
        ticker = str(r["ticker"]).upper()
        strategy = str(r["strategy"])
        expiry = r["expiry"]
        long_strike = round_strike(r.get("long_strike"))
        short_strike = round_strike(r.get("short_strike"))

        short_leg = ""
        long_leg = ""
        short_put_leg = ""
        long_put_leg = ""
        short_call_leg = ""
        long_call_leg = ""
        long_put_strike = fnum(r.get("long_put_strike", long_strike))
        short_put_strike = fnum(r.get("short_put_strike", short_strike))
        short_call_strike = fnum(r.get("short_call_strike"))
        long_call_strike = fnum(r.get("long_call_strike"))
        if not strike_sanity_ok(
            ticker,
            strategy,
            long_strike,
            short_strike,
            long_put_strike,
            short_put_strike,
            short_call_strike,
            long_call_strike,
        ):
            dropped_stage1.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                    "stage": "stage1",
                    "drop_reason": "strike_sanity_fail",
                }
            )
            continue

        if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            short_put_leg = str(r.get("short_put_symbol", "")).strip()
            long_put_leg = str(r.get("long_put_symbol", "")).strip()
            short_call_leg = str(r.get("short_call_symbol", "")).strip()
            long_call_leg = str(r.get("long_call_symbol", "")).strip()

            if not short_put_leg:
                short_put_key = (ticker, "P", expiry, round_strike(short_put_strike))
                short_put_leg = leg_map.get(short_put_key, "")
            if not long_put_leg:
                long_put_key = (ticker, "P", expiry, round_strike(long_put_strike))
                long_put_leg = leg_map.get(long_put_key, "")
            if not short_call_leg:
                short_call_key = (ticker, "C", expiry, round_strike(short_call_strike))
                short_call_leg = leg_map.get(short_call_key, "")
            if not long_call_leg:
                long_call_key = (ticker, "C", expiry, round_strike(long_call_strike))
                long_call_leg = leg_map.get(long_call_key, "")

            if not short_put_leg or not long_put_leg or not short_call_leg or not long_call_leg:
                dropped_stage1.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy,
                        "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                        "stage": "stage1",
                        "drop_reason": "missing_leg_symbol_mapping",
                    }
                )
                continue
            short_leg = short_put_leg
            long_leg = long_put_leg
        else:
            right = strategy_right(strategy)
            if strategy in {"Bull Call Debit", "Bear Put Debit"}:
                long_key = (ticker, right, expiry, long_strike)
                short_key = (ticker, right, expiry, short_strike)
            else:
                short_key = (ticker, right, expiry, short_strike)
                long_key = (ticker, right, expiry, long_strike)

            long_leg = leg_map.get(long_key)
            short_leg = leg_map.get(short_key)
            if not long_leg or not short_leg:
                dropped_stage1.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy,
                        "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                        "stage": "stage1",
                        "drop_reason": "missing_leg_symbol_mapping",
                    }
                )
                continue

        net = fnum(r.get("net"))
        net_type = str(r.get("net_type", "")).strip().lower()
        if not np.isfinite(net):
            dropped_stage1.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                    "stage": "stage1",
                    "drop_reason": "invalid_net",
                }
            )
            continue
        entry_gate = f">= {net:.2f} cr" if net_type == "credit" else f"<= {net:.2f} db"

        shortlist_row = {
            "ticker": ticker,
            "strategy": strategy,
            "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
            "short_leg": short_leg,
            "long_leg": long_leg,
            "short_put_leg": short_put_leg or short_leg,
            "long_put_leg": long_put_leg or long_leg,
            "short_call_leg": short_call_leg,
            "long_call_leg": long_call_leg,
            "net_type": net_type,
            "entry_gate": entry_gate,
            "width": float(r["width"]),
            "conviction": int(r["conviction"]),
            "track": str(r.get("track", "")),
            "confidence_tier": str(r.get("tier", "")),
            "optimal_stage1": str(r.get("optimal", "")),
            "notes_stage1": str(r.get("notes", "")),
            "thesis": str(r.get("thesis", "")),
            "invalidation": str(r.get("invalidation", "")),
            "sigma_pass_stage1": r.get("sigma_pass", np.nan),
            "core_ok_stage1": r.get("core_ok", np.nan),
            "high_beta_pass_stage1": r.get("high_beta_pass", np.nan),
            "earnings_label_stage1": str(r.get("earnings_label", "")),
            "range_neutrality_stage1": r.get("range_neutrality", np.nan),
            "long_strike": float(long_strike) if np.isfinite(fnum(long_strike)) else np.nan,
            "short_strike": float(short_strike) if np.isfinite(fnum(short_strike)) else np.nan,
            "long_put_strike": float(long_put_strike) if np.isfinite(long_put_strike) else np.nan,
            "short_put_strike": float(short_put_strike) if np.isfinite(short_put_strike) else np.nan,
            "short_call_strike": float(short_call_strike) if np.isfinite(short_call_strike) else np.nan,
            "long_call_strike": float(long_call_strike) if np.isfinite(long_call_strike) else np.nan,
            "put_width": float(r.get("put_width")) if np.isfinite(fnum(r.get("put_width"))) else np.nan,
            "call_width": float(r.get("call_width")) if np.isfinite(fnum(r.get("call_width"))) else np.nan,
            "iv_rank": float(r["iv_rank"]) if r.get("iv_rank") is not None and np.isfinite(fnum(r.get("iv_rank"))) else np.nan,
        }
        for extra_key, extra_value in r.items():
            if extra_key in shortlist_row:
                continue
            if extra_value is None:
                shortlist_row[extra_key] = np.nan
            elif isinstance(extra_value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                shortlist_row[extra_key] = extra_value
            elif hasattr(extra_value, "isoformat"):
                shortlist_row[extra_key] = extra_value.isoformat()
        shortlist_rows.append(shortlist_row)

    shortlist = pd.DataFrame(shortlist_rows)
    if shortlist.empty:
        raise RuntimeError("No shortlist rows with valid leg symbols.")
    stage1_rank = {"Yes-Prime": 0, "Yes-Good": 1, "Watch Only": 2}
    shortlist["_stage1_rank"] = shortlist["optimal_stage1"].map(stage1_rank).fillna(3).astype(int)

    def _eod_stage2_score(row: pd.Series) -> float:
        """Rank EOD candidates for Stage-2 coverage without approving them.

        The previous sort was dominated by Stage-1 label and conviction, which
        made the pre-Stage-2 cap behave like a hard veto.  This score still
        respects Stage-1, but also preserves high-conviction EOD structures
        with sane economics so Schwab pricing/GEX can make the real decision.
        """
        score = 0.0
        opt = str(row.get("optimal_stage1", "") or "").strip()
        if opt == "Yes-Prime":
            score += 120.0
        elif opt == "Yes-Good":
            score += 90.0
        elif opt == "Watch Only":
            score += 30.0

        conv = fnum(row.get("conviction"))
        if np.isfinite(conv):
            score += conv * 1.25

        width_val = fnum(row.get("width"))
        net_val = math.nan
        gate_text = str(row.get("entry_gate", "") or "")
        m = re.search(r"([0-9]*\.?[0-9]+)", gate_text)
        if m:
            net_val = fnum(m.group(1))
        net_type_val = str(row.get("net_type", "") or "").strip().lower()
        if np.isfinite(width_val) and width_val > 0 and np.isfinite(net_val):
            frac = max(0.0, min(1.5, net_val / width_val))
            if net_type_val == "debit":
                score += max(0.0, 1.0 - frac) * 35.0
                if frac <= 0.35:
                    score += 20.0
            elif net_type_val == "credit":
                score += min(frac, 1.0) * 35.0
                if frac >= 0.25:
                    score += 15.0

        strategy_val = str(row.get("strategy", "") or "").strip()
        if strategy_val in {"Bull Call Debit", "Bear Put Debit"}:
            score += 15.0
        elif strategy_val in {"Iron Condor", "Iron Butterfly"}:
            score += 8.0

        if strategy_val in {"Bull Call Debit", "Bear Put Debit"}:
            expiry_ts = pd.to_datetime(row.get("expiry"), errors="coerce")
            if pd.notna(expiry_ts):
                dte_val = (expiry_ts.date() - asof).days
                approval_max_dte = fnum(approval_cfg.get("bull_call_approval_max_dte", np.nan))
                approval_dte_grace = fnum(approval_cfg.get("bull_call_approval_dte_grace_days", 0))
                if np.isfinite(approval_max_dte) and approval_max_dte > 0:
                    if dte_val <= approval_max_dte:
                        score += 35.0
                    elif np.isfinite(approval_dte_grace) and dte_val <= approval_max_dte + max(0.0, approval_dte_grace):
                        score += 8.0
                    else:
                        score -= 25.0

        track_val = str(row.get("track", "") or "").strip().upper()
        if track_val == "FIRE":
            score += 8.0
        elif track_val == "SHIELD":
            score += 6.0

        iv_rank_val = fnum(row.get("iv_rank"))
        if np.isfinite(iv_rank_val):
            if strategy_val in {"Bull Call Debit", "Bear Put Debit"} and iv_rank_val > 75:
                score -= 15.0
            elif strategy_val in {"Iron Condor", "Iron Butterfly"} and iv_rank_val >= 40:
                score += 8.0

        note_text = str(row.get("notes_stage1", "") or "").lower()
        if "contra" in note_text or "contradict" in note_text:
            score -= 45.0
        if "confirmed" in note_text:
            score += 12.0
        if "earnings" in note_text:
            score -= 8.0
        return score

    shortlist["_eod_stage2_score"] = shortlist.apply(_eod_stage2_score, axis=1)
    shortlist = (
        shortlist.sort_values(
            ["_eod_stage2_score", "_stage1_rank", "conviction"],
            ascending=[False, True, False],
        )
        .head(max(1, int(discovery_top)))
        .drop(columns=["_stage1_rank", "_eod_stage2_score"])
        .reset_index(drop=True)
    )
    shortlist_csv = out_dir / f"shortlist_trades_{asof_str}_mode_a.csv"
    shortlist.to_csv(shortlist_csv, index=False)

    # Live GEX is calculated from Schwab option-chain snapshots created by
    # Stage-2 pricing. Historical replay cannot use current Schwab chains, so it
    # only accepts pre-existing, date-matched UW captures from the dated folder.
    auto_gex_required = bool(approval_cfg.get("require_gex_regime", False))
    if auto_gex_required and not args.no_auto_collect_uw_gex:
        if args.historical_replay:
            print("  [gex] Historical replay uses existing date-matched UW GEX only; live UW collection skipped")
        elif args.eod_live_planning:
            print("  [gex] EOD live planning uses current Schwab option-chain GEX; UW auto-collection skipped")
        else:
            print("  [gex] Schwab live option-chain GEX calculation is primary; UW auto-collection skipped")

    likelihood_csv = out_dir / f"setup_likelihood_{asof_str}.csv"
    likelihood_yf_cache_dir = repo_root / "out" / "cache" / "setup_likelihood_yf" / asof_str
    likelihood_cmd = [
        sys.executable,
        "-m",
        "uwos.setup_likelihood_backtest",
        "--setups-csv",
        str(shortlist_csv),
        "--asof-date",
        asof_str,
        "--root-dir",
        str(repo_root),
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(likelihood_yf_cache_dir.resolve()),
        "--lookback-years",
        "2",
        "--min-signals",
        str(int(backtest_min_signals)),
    ]
    subprocess.run(likelihood_cmd, check=True)

    live_csv = out_dir / f"live_trade_table_{asof_str}.csv"
    live_final_csv = out_dir / f"live_trade_table_{asof_str}_final.csv"
    stage2_mode = stage2_mode_name(bool(args.historical_replay), bool(args.eod_live_planning))
    stage2_reused_existing = False
    stage2_error = ""
    if args.historical_replay:
        replay_rows = build_historical_replay_live_table(
            shortlist=shortlist,
            quotes=quotes,
            spot_map=spot_map,
            asof_str=asof_str,
            live_csv=live_csv,
            live_final_csv=live_final_csv,
        )
        print(
            f"  [stage2] Historical replay wrote {replay_rows} dated-quote rows; Schwab live pricing skipped",
            file=sys.stderr,
        )
    else:
        cmd = [
            sys.executable,
            "-m",
            "uwos.pricer",
            "--shortlist-csv",
            str(shortlist_csv),
            "--out-dir",
            str(out_dir),
            "--top",
            str(int(discovery_top)),
            "--min-conviction",
            "0",
            "--save-chain-dir",
            str((out_dir / f"schwab_snapshot_{asof_str}" / "chains").resolve()),
            "--snapshot-out-json",
            str((out_dir / f"schwab_snapshot_{asof_str}.json").resolve()),
            "--entry-tol-width-pct",
            str(entry_tol_width_pct),
            "--entry-tol-floor",
            str(entry_tol_floor),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            stage2_error = str(exc)
            if args.allow_stale_stage2 and live_csv.exists() and live_final_csv.exists():
                stage2_reused_existing = True
                print(
                    "WARN: Stage-2 live pricing failed; reusing existing same-date live outputs: "
                    f"{live_csv.name}, {live_final_csv.name}"
                )
            else:
                raise

    if not live_csv.exists():
        raise FileNotFoundError(f"Missing live output: {live_csv}")
    if not live_final_csv.exists():
        raise FileNotFoundError(f"Missing live final output: {live_final_csv}")

    live = pd.read_csv(live_csv, low_memory=False)
    key = ["ticker", "strategy", "expiry", "short_leg", "long_leg", "short_call_leg", "long_call_leg"]
    for col in ["short_call_leg", "long_call_leg"]:
        if col not in shortlist.columns:
            shortlist[col] = ""
        if col not in live.columns:
            live[col] = ""
    for col in key:
        if col not in shortlist.columns:
            shortlist[col] = ""
        if col not in live.columns:
            live[col] = ""
        shortlist[col] = shortlist[col].fillna("").astype(str)
        live[col] = live[col].fillna("").astype(str)
    base_live_cols = [
        "live_status",
        "is_final_live_valid",
        "invalidation_breached_live",
        "invalidation_rule_op",
        "invalidation_rule_level",
        "invalidation_eval_price_live",
        "live_net_bid_ask",
        "live_max_profit",
        "live_max_loss",
        "gate_pass_live",
        "short_bid_live",
        "short_ask_live",
        "short_delta_live",
        "long_bid_live",
        "long_ask_live",
        "short_put_bid_live",
        "short_put_ask_live",
        "short_put_delta_live",
        "long_put_bid_live",
        "long_put_ask_live",
        "short_call_bid_live",
        "short_call_ask_live",
        "short_call_delta_live",
        "long_call_bid_live",
        "long_call_ask_live",
        "long_delta_live",
        "net_gex",
        "gex_regime",
        "gex_support",
        "gex_resistance",
        "spot_live_last",
        "spot_live_bid",
        "spot_live_ask",
        "entry_structure_ok_live",
        "entry_structure_reason_live",
    ]
    live_cols = [c for c in (key + base_live_cols) if c in live.columns]
    mdf = shortlist.merge(live[live_cols], on=key, how="left", suffixes=("", "_live"))

    if likelihood_csv.exists():
        like_df = pd.read_csv(likelihood_csv, low_memory=False)
        like_df["ticker"] = like_df["ticker"].astype(str).str.upper().str.strip()
        like_df["strategy"] = like_df["strategy"].astype(str).str.strip()
        like_df["expiry"] = like_df["expiry"].astype(str).str[:10]
        like_df["entry_gate"] = like_df["entry_gate"].astype(str).str.strip()
        like_keep = [
            "ticker",
            "strategy",
            "expiry",
            "entry_gate",
            "hist_success_pct",
            "edge_pct",
            "signals",
            "verdict",
            "confidence",
            "credit_no_touch_pct",
            "base_hist_success_pct",
            "base_edge_pct",
            "base_signals",
            "base_wins",
            "conditioning_level",
            "conditioning_profile",
            "unsupported_context",
        ]
        like_keep = [c for c in like_keep if c in like_df.columns]
        like_df = like_df[like_keep].drop_duplicates(subset=["ticker", "strategy", "expiry", "entry_gate"])
        mdf["entry_gate"] = mdf["entry_gate"].astype(str).str.strip()
        mdf["expiry"] = mdf["expiry"].astype(str).str[:10]
        mdf = mdf.merge(
            like_df,
            on=["ticker", "strategy", "expiry", "entry_gate"],
            how="left",
            suffixes=("", "_lk"),
        )
        mdf["verdict"] = mdf["verdict"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        mdf["confidence"] = mdf["confidence"].fillna("Unknown").astype(str)
    else:
        mdf["hist_success_pct"] = np.nan
        mdf["edge_pct"] = np.nan
        mdf["signals"] = np.nan
        mdf["verdict"] = "UNKNOWN"
        mdf["confidence"] = "Unknown"
        mdf["credit_no_touch_pct"] = np.nan

    # --- GEX enrichment ---
    # Live runs use Schwab option-chain snapshots as the operative GEX source.
    # Historical replay uses only date-matched UW captures because current Schwab
    # chains would not represent the historical as-of date.
    gex_by_ticker = {}
    gex_source_counts = {}
    uw_gex_summary_csv = base / "enrichments" / "uw" / f"uw_gex_summary_{asof_str}.csv"
    uw_gex_strikes_csv = base / "enrichments" / "uw" / f"uw_gex_strikes_{asof_str}.csv"
    uw_gex_status_csv = base / "enrichments" / "uw" / f"gex_collection_status_{asof_str}.csv"

    def _record_gex_source(source: str) -> None:
        gex_source_counts[source] = int(gex_source_counts.get(source, 0)) + 1

    if args.historical_replay and uw_gex_summary_csv.exists():
        try:
            uw_summary = pd.read_csv(uw_gex_summary_csv, low_memory=False)
            uw_strikes = pd.read_csv(uw_gex_strikes_csv, low_memory=False) if uw_gex_strikes_csv.exists() else pd.DataFrame()
            def _url_date_matches(value: object) -> bool:
                raw = str(value or "").strip()
                if not raw:
                    return False
                try:
                    parsed = urllib.parse.urlparse(raw)
                    qs = urllib.parse.parse_qs(parsed.query)
                    dates = qs.get("date") or []
                    return bool(dates and str(dates[0]) == asof_str)
                except Exception:
                    return False

            if "source_url" in uw_summary.columns:
                before_rows = len(uw_summary)
                uw_summary = uw_summary[uw_summary["source_url"].map(_url_date_matches)].copy()
                dropped_rows = before_rows - len(uw_summary)
                if dropped_rows:
                    print(
                        f"  [gex] WARN: ignored {dropped_rows} UW GEX summary rows with non-{asof_str} source dates",
                        file=sys.stderr,
                    )
            if "uw_time" in uw_summary.columns:
                before_rows = len(uw_summary)
                uw_summary = uw_summary[uw_summary["uw_time"].astype(str).str.startswith(asof_str)].copy()
                dropped_rows = before_rows - len(uw_summary)
                if dropped_rows:
                    print(
                        f"  [gex] WARN: ignored {dropped_rows} UW GEX summary rows with non-{asof_str} payload times",
                        file=sys.stderr,
                    )
            if not uw_strikes.empty and "source_url" in uw_strikes.columns:
                before_rows = len(uw_strikes)
                uw_strikes = uw_strikes[uw_strikes["source_url"].map(_url_date_matches)].copy()
                dropped_rows = before_rows - len(uw_strikes)
                if dropped_rows:
                    print(
                        f"  [gex] WARN: ignored {dropped_rows} UW GEX strike rows with non-{asof_str} source dates",
                        file=sys.stderr,
                    )
            if not uw_strikes.empty and "uw_time" in uw_strikes.columns:
                before_rows = len(uw_strikes)
                uw_strikes = uw_strikes[uw_strikes["uw_time"].astype(str).str.startswith(asof_str)].copy()
                dropped_rows = before_rows - len(uw_strikes)
                if dropped_rows:
                    print(
                        f"  [gex] WARN: ignored {dropped_rows} UW GEX strike rows with non-{asof_str} payload times",
                        file=sys.stderr,
                    )
            uw_support_resistance = {}
            if not uw_strikes.empty and {"ticker", "strike"}.issubset(uw_strikes.columns):
                uw_strikes["ticker"] = uw_strikes["ticker"].astype(str).str.upper().str.strip()
                uw_strikes["_strike_num"] = pd.to_numeric(uw_strikes["strike"], errors="coerce")
                uw_strikes["_spot_num"] = pd.to_numeric(uw_strikes.get("spot"), errors="coerce")
                uw_strikes["_put_wall_abs"] = pd.to_numeric(uw_strikes.get("put_gamma_oi"), errors="coerce").abs()
                uw_strikes["_call_wall_abs"] = pd.to_numeric(uw_strikes.get("call_gamma_oi"), errors="coerce").abs()
                for ticker, grp in uw_strikes.groupby("ticker", dropna=False):
                    spot_vals = pd.to_numeric(grp["_spot_num"], errors="coerce").dropna()
                    spot_v = float(spot_vals.iloc[-1]) if not spot_vals.empty else math.nan
                    support = math.nan
                    resistance = math.nan
                    if np.isfinite(spot_v):
                        puts = grp[(grp["_strike_num"] < spot_v) & np.isfinite(grp["_put_wall_abs"])]
                        calls = grp[(grp["_strike_num"] > spot_v) & np.isfinite(grp["_call_wall_abs"])]
                        if not puts.empty:
                            support = float(puts.sort_values("_put_wall_abs", ascending=False).iloc[0]["_strike_num"])
                        if not calls.empty:
                            resistance = float(calls.sort_values("_call_wall_abs", ascending=False).iloc[0]["_strike_num"])
                    uw_support_resistance[str(ticker).upper()] = {
                        "gex_support": support,
                        "gex_resistance": resistance,
                    }

            if not uw_summary.empty and "ticker" in uw_summary.columns:
                for _, row in uw_summary.iterrows():
                    ticker = str(row.get("ticker", "")).strip().upper()
                    if not ticker:
                        continue
                    net = fnum(row.get("gamma_oi_per_1pct"))
                    if not np.isfinite(net):
                        net = fnum(row.get("gamma_dir_per_1pct"))
                    if not np.isfinite(net):
                        continue
                    sr = uw_support_resistance.get(ticker, {})
                    gex_by_ticker[ticker] = {
                        "net_gex": round(float(net), 2),
                        "gex_regime": "pinned" if net >= 0 else "volatile",
                        "gex_support": sr.get("gex_support", float("nan")),
                        "gex_resistance": sr.get("gex_resistance", float("nan")),
                        "gex_source": UW_GEX_SOURCE,
                        "gex_time": str(row.get("uw_time", "") or ""),
                    }
                    _record_gex_source(UW_GEX_SOURCE)
            if gex_by_ticker:
                print(
                    f"  [gex] Loaded {len(gex_by_ticker)} tickers from UW dashboard capture: {uw_gex_summary_csv.name}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"  [gex] WARN: failed reading UW GEX capture {uw_gex_summary_csv}: {exc}", file=sys.stderr)

    snapshot_chain_dir = out_dir / f"schwab_snapshot_{asof_str}" / "chains"
    if (not args.historical_replay) and snapshot_chain_dir.is_dir():
        schwab_gex_source = SCHWAB_STALE_GEX_SOURCE if stage2_reused_existing else SCHWAB_LIVE_GEX_SOURCE
        for chain_path in sorted(snapshot_chain_dir.glob("chain_*.json")):
            ticker = chain_path.name[len("chain_"):-len(".json")]
            try:
                with open(chain_path) as _f:
                    chain_data = json.load(_f)
            except Exception:
                continue
            gex_info = compute_schwab_chain_gex(chain_data, source=schwab_gex_source)
            if not gex_info:
                continue
            gex_by_ticker[ticker.upper()] = gex_info
            _record_gex_source(schwab_gex_source)
        if gex_source_counts.get(schwab_gex_source):
            freshness = "stale same-date" if stage2_reused_existing else "live"
            print(
                f"  [gex] Calculated {gex_source_counts[schwab_gex_source]} tickers from {freshness} Schwab option-chain snapshots",
                file=sys.stderr,
            )
    elif args.historical_replay:
        print(
            "  [gex] Historical replay: current Schwab chain GEX disabled; "
            "only date-matched UW GEX captures are used",
            file=sys.stderr,
        )
    elif auto_gex_required:
        print(
            f"  [gex] WARN: Schwab chain snapshot directory missing; live GEX gate will block: {snapshot_chain_dir}",
            file=sys.stderr,
        )

    # Apply GEX to merged dataframe
    if gex_by_ticker:
        for col in ["net_gex", "gex_regime", "gex_support", "gex_resistance", "gex_source", "gex_time"]:
            if col not in mdf.columns:
                if col in {"gex_regime", "gex_source", "gex_time"}:
                    mdf[col] = ""
                else:
                    mdf[col] = float("nan")
        for idx, row in mdf.iterrows():
            t = str(row.get("ticker", "")).strip().upper()
            gex_info = gex_by_ticker.get(t)
            if gex_info:
                for col, val in gex_info.items():
                    if col in {"gex_regime", "gex_source", "gex_time"}:
                        mdf.at[idx, col] = str(val) if val is not None else ""
                    else:
                        try:
                            mdf.at[idx, col] = float(val) if val is not None else float("nan")
                        except (TypeError, ValueError):
                            mdf.at[idx, col] = float("nan")
        print(f"  [gex] Enriched {len(gex_by_ticker)} tickers with GEX regime data ({gex_source_counts})", file=sys.stderr)

    # Reuse Stage-1 macro data if available; otherwise fetch fresh
    _macro = getattr(build_best_candidates, "_last_macro", None)
    if _macro is None:
        try:
            _macro = compute_macro_regime(asof, force_historical=bool(args.historical_replay))
            print(f"  [macro] SPY 5d={_macro['spy_5d_ret']:+.2%}, VIX={_macro['vix_level']:.1f}, regime={_macro['regime']}", file=sys.stderr)
        except Exception:
            _macro = {"spy_5d_ret": 0.0, "vix_level": 20.0, "regime": "neutral"}
    _vix_level = _macro.get("vix_level", 20.0)
    _macro_regime = _macro.get("regime", "neutral")

    require_likelihood_pass = bool(approval_cfg.get("require_likelihood_pass", True))
    shield_live_valid_overrides_quality = bool(
        approval_cfg.get("shield_live_valid_overrides_quality", False)
    )
    shield_live_valid_min_no_touch = fnum(approval_cfg.get("shield_live_valid_min_no_touch_floor", 0.0))
    shield_live_valid_min_edge = fnum(approval_cfg.get("shield_live_valid_min_edge_floor", 0.0))
    enable_dual_books = bool(approval_cfg.get("enable_dual_books", True))
    core_size_mult = fnum(approval_cfg.get("core_size_mult", 1.00))
    tactical_size_mult = fnum(approval_cfg.get("tactical_size_mult", 0.50))
    enable_scout_book = bool(approval_cfg.get("enable_scout_book", False))
    scout_size_mult = fnum(approval_cfg.get("scout_size_mult", 0.25))
    scout_min_edge_pct = fnum(
        approval_cfg.get(
            "scout_min_edge_pct",
            approval_cfg.get("bull_call_evidence_min_edge_pct", 5.0),
        )
    )
    scout_max_edge_pct = fnum(
        approval_cfg.get(
            "scout_max_edge_pct",
            approval_cfg.get("stage1_watch_promotion_min_edge_pct", 8.0),
        )
    )
    scout_block_gex_volatile_breakout = bool(
        approval_cfg.get("scout_block_gex_volatile_breakout", False)
    )
    scout_live_entry_enabled = bool(approval_cfg.get("scout_live_entry_enabled", False))
    enable_pilot_book = bool(approval_cfg.get("enable_pilot_book", False))
    pilot_live_entry_enabled = bool(approval_cfg.get("pilot_live_entry_enabled", False))
    pilot_enter_near_miss_within_tolerance = bool(
        approval_cfg.get("pilot_enter_near_miss_within_tolerance", False)
    )
    pilot_size_mult = fnum(approval_cfg.get("pilot_size_mult", 0.10))
    pilot_max_loss = fnum(approval_cfg.get("pilot_max_loss", 150.0))
    pilot_min_ev_ml = fnum(approval_cfg.get("pilot_min_ev_ml", 1.50))
    pilot_high_pop_min = normalize_probability(approval_cfg.get("pilot_high_pop_min", 0.25))
    pilot_floor_pop_min = normalize_probability(approval_cfg.get("pilot_floor_pop_min", 0.15))
    pilot_min_conviction = fnum(approval_cfg.get("pilot_min_conviction", 40))
    pilot_pass_min_signals = fnum(approval_cfg.get("pilot_pass_min_signals", 100))
    pilot_pass_min_edge = fnum(approval_cfg.get("pilot_pass_min_edge_pct", 1.0))
    pilot_min_dte = fnum(approval_cfg.get("pilot_min_dte", 5))
    pilot_max_dte = fnum(approval_cfg.get("pilot_max_dte", 45))
    pilot_min_reward_risk = fnum(approval_cfg.get("pilot_min_reward_risk", 1.50))
    pilot_max_debit_frac = fnum(approval_cfg.get("pilot_max_debit_frac", 0.25))
    pilot_max_long_otm_pct = fnum(approval_cfg.get("pilot_max_long_otm_pct", 0.15))
    enable_native_pilot_book = bool(approval_cfg.get("enable_native_pilot_book", False))
    native_pilot_max_loss = fnum(approval_cfg.get("native_pilot_max_loss", pilot_max_loss))
    native_pilot_min_partial_ev_ml = fnum(approval_cfg.get("native_pilot_min_partial_ev_ml", 0.0))
    native_pilot_min_signals = fnum(approval_cfg.get("native_pilot_min_signals", 60))
    native_pilot_min_edge = fnum(approval_cfg.get("native_pilot_min_edge_pct", pilot_pass_min_edge))
    native_pilot_require_contract_confirmed = bool(
        approval_cfg.get("native_pilot_require_contract_confirmed", True)
    )
    native_pilot_market_confidences = {
        str(x).strip().upper()
        for x in approval_cfg.get("native_pilot_market_confidences", ["Medium", "High"])
        if str(x).strip()
    }
    allow_bear_put_scout_lane = bool(approval_cfg.get("allow_bear_put_scout_lane", False))
    bear_put_scout_likelihood_strengths = {
        str(x).strip().upper()
        for x in approval_cfg.get("bear_put_scout_likelihood_strengths", ["Negative"])
        if str(x).strip()
    }
    bear_put_scout_require_negative_edge = bool(
        approval_cfg.get("bear_put_scout_require_negative_edge", True)
    )
    bear_put_scout_min_signals = fnum(approval_cfg.get("bear_put_scout_min_signals", 60))
    bear_put_scout_min_dte = fnum(approval_cfg.get("bear_put_scout_min_dte", 14))
    bear_put_scout_max_dte = fnum(approval_cfg.get("bear_put_scout_max_dte", 35))
    bear_put_scout_max_iv_rank = fnum(approval_cfg.get("bear_put_scout_max_iv_rank", 30))
    bear_put_scout_max_vix = fnum(approval_cfg.get("bear_put_scout_max_vix", 20))
    bear_put_scout_require_spy_5d_nonnegative = bool(
        approval_cfg.get("bear_put_scout_require_spy_5d_nonnegative", True)
    )
    bear_put_scout_min_reward_risk = fnum(approval_cfg.get("bear_put_scout_min_reward_risk", 2.0))
    bear_put_scout_max_debit_frac = fnum(approval_cfg.get("bear_put_scout_max_debit_frac", 0.35))
    tactical_min_conviction = fnum(approval_cfg.get("tactical_min_conviction", 60))
    tactical_min_conviction_grace = fnum(approval_cfg.get("tactical_min_conviction_grace", 2))
    tactical_min_edge_pct = fnum(approval_cfg.get("tactical_min_edge_pct", 0.0))
    tactical_require_verdict_pass = bool(approval_cfg.get("tactical_require_verdict_pass", True))
    enable_event_momentum_scout = bool(approval_cfg.get("enable_event_momentum_scout", False))
    event_momentum_scout_min_conviction = fnum(
        approval_cfg.get("event_momentum_scout_min_conviction", 30)
    )
    event_momentum_scout_max_dte = fnum(approval_cfg.get("event_momentum_scout_max_dte", 35))
    event_momentum_scout_min_reward_risk = fnum(
        approval_cfg.get("event_momentum_scout_min_reward_risk", 1.8)
    )
    event_momentum_scout_max_debit_frac = fnum(
        approval_cfg.get("event_momentum_scout_max_debit_frac", 0.35)
    )
    event_momentum_scout_require_contract_confirmed = bool(
        approval_cfg.get("event_momentum_scout_require_contract_confirmed", True)
    )
    event_momentum_scout_require_breakeven_cross = bool(
        approval_cfg.get("event_momentum_scout_require_breakeven_cross", True)
    )
    allow_debit_momentum_scout_lane = bool(
        approval_cfg.get("allow_debit_momentum_scout_lane", False)
    )
    debit_momentum_scout_min_conviction = fnum(approval_cfg.get("debit_momentum_scout_min_conviction", 40))
    debit_momentum_scout_min_edge_pct = fnum(approval_cfg.get("debit_momentum_scout_min_edge_pct", 8.0))
    debit_momentum_scout_bear_min_edge_pct = fnum(
        approval_cfg.get(
            "debit_momentum_scout_bear_min_edge_pct",
            approval_cfg.get("min_edge_pct_bear", approval_cfg.get("min_edge_pct", 12.0)),
        )
    )
    debit_momentum_scout_min_signals = fnum(approval_cfg.get("debit_momentum_scout_min_signals", 100))
    debit_momentum_scout_min_dte = fnum(approval_cfg.get("debit_momentum_scout_min_dte", 14))
    debit_momentum_scout_max_dte = fnum(approval_cfg.get("debit_momentum_scout_max_dte", 45))
    debit_momentum_scout_min_reward_risk = fnum(approval_cfg.get("debit_momentum_scout_min_reward_risk", 2.0))
    debit_momentum_scout_max_debit_frac = fnum(approval_cfg.get("debit_momentum_scout_max_debit_frac", 0.35))
    debit_momentum_scout_max_iv_rank = fnum(approval_cfg.get("debit_momentum_scout_max_iv_rank", 80))
    debit_momentum_scout_require_contract_confirmed = bool(
        approval_cfg.get("debit_momentum_scout_require_contract_confirmed", True)
    )
    debit_momentum_scout_require_verdict_pass = bool(
        approval_cfg.get("debit_momentum_scout_require_verdict_pass", True)
    )
    debit_momentum_scout_min_regime_score = fnum(
        approval_cfg.get("debit_momentum_scout_min_regime_score", 55)
    )
    debit_momentum_scout_bear_require_flow_confirmed = bool(
        approval_cfg.get("debit_momentum_scout_bear_require_flow_confirmed", True)
    )
    debit_momentum_scout_block_gex_volatile_breakout = bool(
        approval_cfg.get("debit_momentum_scout_block_gex_volatile_breakout", True)
    )
    debit_momentum_scout_bear_likelihood_strengths = {
        str(x).strip().upper()
        for x in approval_cfg.get("debit_momentum_scout_bear_likelihood_strengths", ["Moderate", "Strong"])
        if str(x).strip()
    }
    min_edge_pct = fnum(approval_cfg.get("min_edge_pct", 0.0))
    min_edge_pct_bear = fnum(approval_cfg.get("min_edge_pct_bear", min_edge_pct))
    min_edge_pct_shield = fnum(approval_cfg.get("min_edge_pct_shield", min_edge_pct))
    if not np.isfinite(min_edge_pct_bear):
        min_edge_pct_bear = min_edge_pct
    if not np.isfinite(min_edge_pct_shield):
        min_edge_pct_shield = min_edge_pct
    min_signals = fnum(approval_cfg.get("min_signals", 100))
    tactical_min_signals = fnum(approval_cfg.get("tactical_min_signals", min_signals))
    tactical_min_signals_grace = fnum(approval_cfg.get("tactical_min_signals_grace", 5))
    enable_medium_book = bool(approval_cfg.get("enable_medium_book", True))
    medium_size_mult = fnum(approval_cfg.get("medium_size_mult", 0.25))
    medium_allowed_strategies = {
        str(x).strip()
        for x in approval_cfg.get("medium_allowed_strategies", ["Bull Call Debit"])
        if str(x).strip()
    }
    medium_min_conviction = fnum(approval_cfg.get("medium_min_conviction", 40))
    medium_min_edge_pct = fnum(approval_cfg.get("medium_min_edge_pct", 8.0))
    medium_bear_min_edge_pct = fnum(approval_cfg.get("medium_bear_min_edge_pct", 12.0))
    medium_shield_min_edge_pct = fnum(approval_cfg.get("medium_shield_min_edge_pct", 8.0))
    medium_min_signals = fnum(approval_cfg.get("medium_min_signals", min_signals))
    medium_require_verdict_pass = bool(approval_cfg.get("medium_require_verdict_pass", True))
    medium_min_reward_risk = fnum(approval_cfg.get("medium_min_reward_risk", 1.50))
    medium_max_debit_frac = fnum(approval_cfg.get("medium_max_debit_frac", 0.35))
    medium_min_dte = fnum(approval_cfg.get("medium_min_dte", 14))
    medium_max_dte = fnum(approval_cfg.get("medium_max_dte", 60))
    medium_max_iv_rank = fnum(approval_cfg.get("medium_max_iv_rank", 85))
    medium_require_contract_confirmed = bool(approval_cfg.get("medium_require_contract_confirmed", True))
    medium_allowed_market_confidences = {
        str(x).strip().upper()
        for x in approval_cfg.get("medium_allowed_market_confidences", ["Medium", "High"])
        if str(x).strip()
    }
    enable_quant_edge_book = bool(approval_cfg.get("enable_quant_edge_book", True))
    quant_edge_size_mult = fnum(approval_cfg.get("quant_edge_size_mult", medium_size_mult))
    quant_edge_allowed_strategies = {
        str(x).strip()
        for x in approval_cfg.get("quant_edge_allowed_strategies", ["Bull Call Debit", "Iron Condor"])
        if str(x).strip()
    }
    quant_edge_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("quant_edge_allowed_verdicts", ["PASS"])
        if str(x).strip()
    }
    quant_edge_min_edge_pct = fnum(approval_cfg.get("quant_edge_min_edge_pct", 5.0))
    quant_edge_min_signals = fnum(approval_cfg.get("quant_edge_min_signals", 150))
    quant_edge_min_dte = fnum(approval_cfg.get("quant_edge_min_dte", 30))
    quant_edge_max_dte = fnum(approval_cfg.get("quant_edge_max_dte", 60))
    quant_edge_min_iv_rank = fnum(approval_cfg.get("quant_edge_min_iv_rank", 30))
    quant_edge_max_iv_rank = fnum(approval_cfg.get("quant_edge_max_iv_rank", 85))
    quant_edge_max_per_day = max(0, int(approval_cfg.get("quant_edge_max_per_day", 3)))
    quant_edge_max_per_ticker_day = max(0, int(approval_cfg.get("quant_edge_max_per_ticker_day", 1)))
    quant_edge_blocker_substrings = tuple(
        str(x).strip().lower()
        for x in approval_cfg.get(
            "quant_edge_disallowed_blocker_substrings",
            [
                "contra",
                "live_entry_gate_fail",
                "market_regime_block",
                "invalid_entry_structure",
                "earnings_risk",
                "dte_too_long",
                "liquidity",
            ],
        )
        if str(x).strip()
    )
    enable_short_dte_edge_book = bool(approval_cfg.get("enable_short_dte_edge_book", True))
    short_dte_edge_size_mult = fnum(approval_cfg.get("short_dte_edge_size_mult", 0.15))
    short_dte_edge_allowed_strategies = {
        str(x).strip()
        for x in approval_cfg.get("short_dte_edge_allowed_strategies", ["Bull Call Debit"])
        if str(x).strip()
    }
    short_dte_edge_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("short_dte_edge_allowed_verdicts", ["PASS"])
        if str(x).strip()
    }
    short_dte_edge_min_edge_pct = fnum(approval_cfg.get("short_dte_edge_min_edge_pct", 5.0))
    short_dte_edge_min_signals = fnum(approval_cfg.get("short_dte_edge_min_signals", 120))
    short_dte_edge_min_confidence_score = fnum(approval_cfg.get("short_dte_edge_min_confidence_score", 3))
    short_dte_edge_min_dte = fnum(approval_cfg.get("short_dte_edge_min_dte", 21))
    short_dte_edge_max_dte = fnum(approval_cfg.get("short_dte_edge_max_dte", 35))
    short_dte_edge_max_iv_rank = fnum(approval_cfg.get("short_dte_edge_max_iv_rank", 55))
    short_dte_edge_max_per_day = max(0, int(approval_cfg.get("short_dte_edge_max_per_day", 1)))
    short_dte_edge_max_per_ticker_day = max(0, int(approval_cfg.get("short_dte_edge_max_per_ticker_day", 1)))
    short_dte_edge_require_qualified_candidate = bool(
        approval_cfg.get("short_dte_edge_require_qualified_candidate", True)
    )
    short_dte_edge_blocker_substrings = tuple(
        str(x).strip().lower()
        for x in approval_cfg.get(
            "short_dte_edge_disallowed_blocker_substrings",
            [
                "contra",
                "live_entry_gate_fail",
                "invalid_entry_structure",
                "earnings_risk",
                "liquidity",
            ],
        )
        if str(x).strip()
    )
    enable_high_signal_edge_book = bool(approval_cfg.get("enable_high_signal_edge_book", True))
    high_signal_edge_size_mult = fnum(approval_cfg.get("high_signal_edge_size_mult", 0.15))
    high_signal_edge_allowed_strategies = {
        str(x).strip()
        for x in approval_cfg.get("high_signal_edge_allowed_strategies", ["Bull Call Debit"])
        if str(x).strip()
    }
    high_signal_edge_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("high_signal_edge_allowed_verdicts", ["PASS"])
        if str(x).strip()
    }
    high_signal_edge_min_edge_pct = fnum(approval_cfg.get("high_signal_edge_min_edge_pct", 8.0))
    high_signal_edge_min_signals = fnum(approval_cfg.get("high_signal_edge_min_signals", 180))
    high_signal_edge_min_confidence_score = fnum(approval_cfg.get("high_signal_edge_min_confidence_score", 3))
    high_signal_edge_min_dte = fnum(approval_cfg.get("high_signal_edge_min_dte", 21))
    high_signal_edge_max_dte = fnum(approval_cfg.get("high_signal_edge_max_dte", 60))
    high_signal_edge_max_iv_rank = fnum(approval_cfg.get("high_signal_edge_max_iv_rank", 55))
    high_signal_edge_max_per_day = max(0, int(approval_cfg.get("high_signal_edge_max_per_day", 1)))
    high_signal_edge_max_per_ticker_day = max(0, int(approval_cfg.get("high_signal_edge_max_per_ticker_day", 1)))
    high_signal_edge_require_qualified_candidate = bool(
        approval_cfg.get("high_signal_edge_require_qualified_candidate", True)
    )
    high_signal_edge_require_contract_flow_confirmed = bool(
        approval_cfg.get("high_signal_edge_require_contract_flow_confirmed", True)
    )
    high_signal_edge_require_gex_context = bool(
        approval_cfg.get("high_signal_edge_require_gex_context", True)
    )
    high_signal_edge_excluded_approval_regimes = {
        str(x).strip().lower()
        for x in approval_cfg.get("high_signal_edge_excluded_approval_regimes", ["mid/range"])
        if str(x).strip()
    }
    high_signal_edge_allowed_gex_wall_contexts = {
        str(x).strip()
        for x in approval_cfg.get("high_signal_edge_allowed_gex_wall_contexts", [])
        if str(x).strip()
    }
    high_signal_edge_blocker_substrings = tuple(
        str(x).strip().lower()
        for x in approval_cfg.get(
            "high_signal_edge_disallowed_blocker_substrings",
            [
                "contra",
                "live_entry_gate_fail",
                "invalid_entry_structure",
                "earnings_risk",
                "liquidity",
                "missing_live",
                "missing_leg",
                "gex_missing",
                "uw_gex",
            ],
        )
        if str(x).strip()
    )
    enable_regime_weekly_book = bool(approval_cfg.get("enable_regime_weekly_book", False))
    regime_weekly_size_mult = fnum(approval_cfg.get("regime_weekly_size_mult", tactical_size_mult))
    regime_weekly_max_per_day = max(0, int(approval_cfg.get("regime_weekly_max_per_day", 1)))
    regime_weekly_max_per_ticker_day = max(0, int(approval_cfg.get("regime_weekly_max_per_ticker_day", 1)))
    regime_weekly_min_score = fnum(approval_cfg.get("regime_weekly_min_score", 125.0))
    regime_weekly_min_confidence_score = fnum(approval_cfg.get("regime_weekly_min_confidence_score", 4))
    regime_weekly_promoted_confidence_score = fnum(
        approval_cfg.get("regime_weekly_promoted_confidence_score", 7.0)
    )
    regime_weekly_min_edge_score = fnum(approval_cfg.get("regime_weekly_min_edge_score", 1))
    regime_weekly_min_edge_pct = fnum(approval_cfg.get("regime_weekly_min_edge_pct", -20.0))
    regime_weekly_min_reward_risk = fnum(approval_cfg.get("regime_weekly_min_reward_risk", 1.20))
    regime_weekly_max_debit_frac = fnum(approval_cfg.get("regime_weekly_max_debit_frac", 0.45))
    regime_weekly_min_dte = fnum(approval_cfg.get("regime_weekly_min_dte", 14))
    regime_weekly_max_dte = fnum(approval_cfg.get("regime_weekly_max_dte", 70))
    regime_weekly_bear_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("regime_weekly_bear_allowed_verdicts", ["FAIL", "UNKNOWN"])
        if str(x).strip()
    }
    regime_weekly_bear_allowed_market_regimes = {
        str(x).strip().lower()
        for x in approval_cfg.get("regime_weekly_bear_allowed_market_regimes", ["neutral", "risk_on"])
        if str(x).strip()
    }
    regime_weekly_bear_excluded_approval_prefixes = tuple(
        str(x).strip().lower()
        for x in approval_cfg.get("regime_weekly_bear_excluded_approval_prefixes", ["high/"])
        if str(x).strip()
    )
    regime_weekly_bear_min_edge_pct = fnum(approval_cfg.get("regime_weekly_bear_min_edge_pct", regime_weekly_min_edge_pct))
    regime_weekly_bear_max_edge_pct = fnum(approval_cfg.get("regime_weekly_bear_max_edge_pct", 0.0))
    regime_weekly_bull_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("regime_weekly_bull_allowed_verdicts", ["PASS"])
        if str(x).strip()
    }
    regime_weekly_bull_allowed_approval_prefixes = tuple(
        str(x).strip().lower()
        for x in approval_cfg.get("regime_weekly_bull_allowed_approval_prefixes", ["high/"])
        if str(x).strip()
    )
    regime_weekly_bull_allowed_market_regimes = {
        str(x).strip().lower()
        for x in approval_cfg.get("regime_weekly_bull_allowed_market_regimes", ["risk_off"])
        if str(x).strip()
    }
    regime_weekly_bull_allow_gex_volatile = bool(
        approval_cfg.get("regime_weekly_bull_allow_gex_volatile", True)
    )
    regime_weekly_bull_min_edge_pct = fnum(approval_cfg.get("regime_weekly_bull_min_edge_pct", 0.0))
    regime_weekly_enable_income = bool(approval_cfg.get("regime_weekly_enable_income", True))
    regime_weekly_income_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("regime_weekly_income_allowed_verdicts", ["PASS"])
        if str(x).strip()
    }
    regime_weekly_income_min_credit_no_touch_pct = fnum(
        approval_cfg.get("regime_weekly_income_min_credit_no_touch_pct", 0.0)
    )
    regime_weekly_block_contract_flow_states = {
        str(x).strip().lower()
        for x in approval_cfg.get(
            "regime_weekly_block_contract_flow_states",
            ["contra", "directional"],
        )
        if str(x).strip()
    }
    enable_income_book = bool(approval_cfg.get("enable_income_book", True))
    income_size_mult = fnum(approval_cfg.get("income_size_mult", 0.25))
    income_allowed_strategies = {
        str(x).strip()
        for x in approval_cfg.get("income_allowed_strategies", ["Iron Condor"])
        if str(x).strip()
    }
    income_min_edge_pct = fnum(approval_cfg.get("income_min_edge_pct", 5.0))
    income_min_signals = fnum(approval_cfg.get("income_min_signals", 100))
    income_min_hist_success_pct = fnum(approval_cfg.get("income_min_hist_success_pct", 55.0))
    income_min_dte = fnum(approval_cfg.get("income_min_dte", 28))
    income_max_dte = fnum(approval_cfg.get("income_max_dte", 70))
    income_max_iv_rank = fnum(approval_cfg.get("income_max_iv_rank", 85))
    income_allowed_verdicts = {
        str(x).strip().upper()
        for x in approval_cfg.get("income_allowed_verdicts", ["PASS", "LOW_SAMPLE"])
        if str(x).strip()
    }
    income_allowed_gex_regimes = {
        str(x).strip().lower()
        for x in approval_cfg.get("income_allowed_gex_regimes", ["pinned"])
        if str(x).strip()
    }
    income_min_credit_no_touch_pct = fnum(approval_cfg.get("income_min_credit_no_touch_pct", 0.0))
    enable_qualified_book = bool(approval_cfg.get("enable_qualified_book", True))
    qualified_min_confidence_score = fnum(approval_cfg.get("qualified_min_confidence_score", 5.0))
    qualified_min_edge_score = fnum(approval_cfg.get("qualified_min_edge_score", 1.0))
    confidence_high_min_score = fnum(approval_cfg.get("confidence_high_min_score", 7.0))
    confidence_medium_min_score = fnum(approval_cfg.get("confidence_medium_min_score", 5.0))
    medium_review_min_pop = normalize_probability(approval_cfg.get("medium_review_min_pop", 0.60))
    medium_review_min_confidence_score = fnum(approval_cfg.get("medium_review_min_confidence_score", 5.0))
    medium_review_min_edge_score = fnum(approval_cfg.get("medium_review_min_edge_score", 1.0))
    medium_block_earnings_risk = bool(approval_cfg.get("medium_block_earnings_risk", True))
    high_enter_min_pop = normalize_probability(approval_cfg.get("high_enter_min_pop", 0.60))
    high_enter_preferred_pop = normalize_probability(approval_cfg.get("high_enter_preferred_pop", 0.65))
    high_enter_min_confidence_score = fnum(approval_cfg.get("high_enter_min_confidence_score", confidence_high_min_score))
    high_enter_min_edge_pct = fnum(approval_cfg.get("high_enter_min_edge_pct", 0.0))
    high_enter_require_positive_edge = bool(approval_cfg.get("high_enter_require_positive_edge", True))
    high_enter_require_clean_flow_or_technical = bool(approval_cfg.get("high_enter_require_clean_flow_or_technical", True))
    approval_regime_rules_enabled = bool(approval_cfg.get("approval_regime_rules_enabled", True))
    approval_mid_down_block_bull_calls = bool(approval_cfg.get("approval_mid_down_block_bull_calls", True))
    approval_mid_down_bear_put_min_pop = normalize_probability(approval_cfg.get("approval_mid_down_bear_put_min_pop", 0.65))
    approval_low_range_debit_max_be_distance_pct = fnum(approval_cfg.get("approval_low_range_debit_max_be_distance_pct", 0.03))
    approval_high_down_min_signals = fnum(approval_cfg.get("approval_high_down_min_signals", 150))
    approval_high_range_breakout_min_edge_pct = fnum(approval_cfg.get("approval_high_range_breakout_min_edge_pct", 12.0))
    approval_high_range_breakout_min_confidence_score = fnum(approval_cfg.get("approval_high_range_breakout_min_confidence_score", 7.0))
    max_same_direction_pct = fnum(engine_cfg.get("max_same_direction_pct", 0.70))
    max_same_expiry_count = int(engine_cfg.get("max_same_expiry_count", 8))
    require_invalidation_clear = bool(approval_cfg.get("require_invalidation_clear", False))
    block_invalidation_warning = bool(approval_cfg.get("block_invalidation_warning", False))
    allow_stage1_watch_promotion = bool(approval_cfg.get("allow_stage1_watch_promotion", True))
    stage1_promote_min_conv = fnum(approval_cfg.get("stage1_watch_promotion_min_conviction", 58))
    stage1_promote_min_edge = fnum(approval_cfg.get("stage1_watch_promotion_min_edge_pct", 5.0))
    stage1_promote_min_signals = fnum(approval_cfg.get("stage1_watch_promotion_min_signals", min_signals))
    allow_fire_breakout_exception = bool(approval_cfg.get("allow_fire_breakout_exception", True))
    fire_breakout_min_conviction = fnum(approval_cfg.get("fire_breakout_min_conviction", 40))
    fire_breakout_min_edge = fnum(approval_cfg.get("fire_breakout_min_edge_pct", 12.0))
    fire_breakout_min_signals = fnum(approval_cfg.get("fire_breakout_min_signals", min_signals))
    fire_breakout_min_long_delta = fnum(approval_cfg.get("fire_breakout_min_long_delta", 0.35))
    fire_breakout_require_risk_on = bool(approval_cfg.get("fire_breakout_require_risk_on", True))
    fire_breakout_max_wall_distance_pct = fnum(approval_cfg.get("fire_breakout_max_wall_distance_pct", 0.01))
    allow_bull_call_evidence_lane = bool(approval_cfg.get("allow_bull_call_evidence_lane", True))
    bull_call_evidence_min_edge = fnum(approval_cfg.get("bull_call_evidence_min_edge_pct", 8.0))
    bull_call_evidence_min_signals = fnum(approval_cfg.get("bull_call_evidence_min_signals", 120))
    bull_call_evidence_min_conviction = fnum(approval_cfg.get("bull_call_evidence_min_conviction", 30))
    bull_call_evidence_min_long_delta = fnum(approval_cfg.get("bull_call_evidence_min_long_delta", 0.30))
    bull_call_evidence_max_dte = fnum(approval_cfg.get("bull_call_evidence_max_dte", 35))
    bull_call_evidence_min_reward_risk = fnum(approval_cfg.get("bull_call_evidence_min_reward_risk", 2.0))
    bull_call_evidence_require_contract_confirmed = bool(
        approval_cfg.get("bull_call_evidence_require_contract_confirmed", True)
    )
    bull_call_evidence_allow_gex_missing = bool(
        approval_cfg.get("bull_call_evidence_allow_gex_missing", True)
    )
    allow_bear_put_evidence_lane = bool(approval_cfg.get("allow_bear_put_evidence_lane", True))
    bear_put_evidence_min_edge = fnum(approval_cfg.get("bear_put_evidence_min_edge_pct", 12.0))
    bear_put_evidence_min_signals = fnum(approval_cfg.get("bear_put_evidence_min_signals", 120))
    bear_put_evidence_min_conviction = fnum(approval_cfg.get("bear_put_evidence_min_conviction", 30))
    bear_put_evidence_min_long_delta = fnum(approval_cfg.get("bear_put_evidence_min_long_delta", 0.25))
    bear_put_evidence_min_dte = fnum(approval_cfg.get("bear_put_evidence_min_dte", 14))
    bear_put_evidence_max_dte = fnum(approval_cfg.get("bear_put_evidence_max_dte", 60))
    bear_put_evidence_min_reward_risk = fnum(approval_cfg.get("bear_put_evidence_min_reward_risk", 1.5))
    bear_put_evidence_max_debit_frac = fnum(approval_cfg.get("bear_put_evidence_max_debit_frac", 0.45))
    bear_put_evidence_max_iv_rank = fnum(approval_cfg.get("bear_put_evidence_max_iv_rank", 60))
    bear_put_evidence_require_contract_confirmed = bool(
        approval_cfg.get("bear_put_evidence_require_contract_confirmed", True)
    )
    allow_bull_call_pinned_continuation_lane = bool(
        approval_cfg.get("allow_bull_call_pinned_continuation_lane", False)
    )
    bull_call_pinned_continuation_min_edge = fnum(
        approval_cfg.get("bull_call_pinned_continuation_min_edge_pct", 8.0)
    )
    bull_call_pinned_continuation_min_signals = fnum(
        approval_cfg.get("bull_call_pinned_continuation_min_signals", 120)
    )
    bull_call_pinned_continuation_min_conviction = fnum(
        approval_cfg.get("bull_call_pinned_continuation_min_conviction", 40)
    )
    bull_call_pinned_continuation_core_min_conviction = fnum(
        approval_cfg.get("bull_call_pinned_continuation_core_min_conviction", 55)
    )
    bull_call_pinned_continuation_min_dte = fnum(
        approval_cfg.get("bull_call_pinned_continuation_min_dte", 31)
    )
    bull_call_pinned_continuation_max_dte = fnum(
        approval_cfg.get("bull_call_pinned_continuation_max_dte", 45)
    )
    bull_call_pinned_continuation_min_reward_risk = fnum(
        approval_cfg.get("bull_call_pinned_continuation_min_reward_risk", 2.0)
    )
    bull_call_pinned_continuation_max_debit_frac = fnum(
        approval_cfg.get("bull_call_pinned_continuation_max_debit_frac", 0.35)
    )
    bull_call_pinned_continuation_min_strength = str(
        approval_cfg.get("bull_call_pinned_continuation_min_likelihood_strength", "Moderate")
    ).strip()
    bull_call_pinned_continuation_allowed_gex_contexts = {
        str(x).strip()
        for x in approval_cfg.get(
            "bull_call_pinned_continuation_allowed_gex_contexts",
            ["pinned_resistance_above_call_wall"],
        )
        if str(x).strip()
    }
    bull_call_pinned_continuation_allowed_regime_confidences = {
        str(x).strip()
        for x in approval_cfg.get(
            "bull_call_pinned_continuation_allowed_regime_confidences",
            ["Medium", "High"],
        )
        if str(x).strip()
    }
    bull_call_pinned_continuation_require_contract_confirmed = bool(
        approval_cfg.get("bull_call_pinned_continuation_require_contract_confirmed", True)
    )
    bull_call_pinned_continuation_block_short_dte_high_edge = bool(
        approval_cfg.get("bull_call_pinned_continuation_block_short_dte_high_edge", True)
    )
    allow_fire_delta_moneyness_proxy = bool(
        approval_cfg.get("allow_fire_delta_moneyness_proxy", False)
    )
    fire_delta_proxy_max_long_otm_pct = fnum(
        approval_cfg.get("fire_delta_proxy_max_long_otm_pct", 0.02)
    )
    bull_call_approval_max_dte = fnum(
        approval_cfg.get("bull_call_approval_max_dte", bull_call_evidence_max_dte)
    )
    bull_call_approval_dte_grace_days = fnum(
        approval_cfg.get("bull_call_approval_dte_grace_days", 4)
    )
    if not np.isfinite(bull_call_approval_dte_grace_days) or bull_call_approval_dte_grace_days < 0:
        bull_call_approval_dte_grace_days = 0.0
    bull_call_approval_min_reward_risk = fnum(
        approval_cfg.get("bull_call_approval_min_reward_risk", bull_call_evidence_min_reward_risk)
    )
    bull_call_short_dte_high_edge_block = bool(
        approval_cfg.get("bull_call_short_dte_high_edge_block", False)
    )
    bull_call_short_dte_high_edge_max_dte = fnum(
        approval_cfg.get("bull_call_short_dte_high_edge_max_dte", 31)
    )
    bull_call_short_dte_high_edge_min_edge = fnum(
        approval_cfg.get("bull_call_short_dte_high_edge_min_edge_pct", 13.2)
    )
    bull_call_approval_require_contract_confirmed = bool(
        approval_cfg.get("bull_call_approval_require_contract_confirmed", True)
    )
    bull_call_market_regime_enabled = bool(
        approval_cfg.get("bull_call_market_regime_enabled", True)
    )
    bull_call_low_regime_blocks = bool(
        approval_cfg.get("bull_call_low_regime_blocks", True)
    )
    bull_call_medium_regime_tactical = bool(
        approval_cfg.get("bull_call_medium_regime_tactical", True)
    )
    bull_call_regime_low_score = fnum(approval_cfg.get("bull_call_regime_low_score", 50))
    bull_call_regime_high_score = fnum(approval_cfg.get("bull_call_regime_high_score", 75))
    bull_call_block_downtrend_without_high_vix = bool(
        approval_cfg.get("bull_call_block_downtrend_without_high_vix", True)
    )
    bull_call_missing_gex_requires_uptrend = bool(
        approval_cfg.get("bull_call_missing_gex_requires_uptrend", True)
    )
    bull_call_trend_vix_floor = fnum(approval_cfg.get("bull_call_trend_vix_floor", 22.0))
    if not np.isfinite(bull_call_trend_vix_floor) or bull_call_trend_vix_floor <= 0:
        bull_call_trend_vix_floor = 22.0
    min_likelihood_strength = str(approval_cfg.get("min_likelihood_strength", "")).strip()
    min_likelihood_strength_bear = str(approval_cfg.get("min_likelihood_strength_bear", min_likelihood_strength)).strip()
    min_likelihood_strength_shield = str(approval_cfg.get("min_likelihood_strength_shield", min_likelihood_strength)).strip()
    disallow_likelihood_strengths = {
        str(x).strip().upper()
        for x in approval_cfg.get("disallow_likelihood_strengths", [])
        if str(x).strip()
    }
    disallow_confidence_tiers = {
        str(x).strip().upper()
        for x in approval_cfg.get("disallow_confidence_tiers", [])
        if str(x).strip()
    }
    require_shield_sigma_pass = bool(approval_cfg.get("require_shield_sigma_pass", False))
    shield_sigma_require_data = bool(approval_cfg.get("shield_sigma_require_data", True))
    require_shield_core = bool(approval_cfg.get("require_shield_core", False))
    require_live_shield_short_delta = bool(approval_cfg.get("require_live_shield_short_delta", False))
    max_abs_short_delta_shield = fnum(approval_cfg.get("max_abs_short_delta_shield", 0.20))
    shield_delta_dynamic = bool(approval_cfg.get("shield_delta_dynamic", False))
    # FIRE delta gate
    require_fire_long_delta = bool(approval_cfg.get("require_fire_long_delta", False))
    min_abs_long_delta_fire = fnum(approval_cfg.get("min_abs_long_delta_fire", 0.15))
    # GEX regime gate
    require_gex_regime = bool(approval_cfg.get("require_gex_regime", False))
    min_fire_pinned_gex_abs = fnum(approval_cfg.get("min_fire_pinned_gex_abs", 10_000_000))
    if not np.isfinite(min_fire_pinned_gex_abs) or min_fire_pinned_gex_abs < 0:
        min_fire_pinned_gex_abs = 10_000_000.0
    fire_volatile_breakout_tactical_only = bool(
        approval_cfg.get("fire_volatile_breakout_tactical_only", True)
    )
    fire_missing_gex_context_tactical_only = bool(
        approval_cfg.get("fire_missing_gex_context_tactical_only", True)
    )
    fire_pinned_no_wall_tactical_only = bool(
        approval_cfg.get("fire_pinned_no_wall_tactical_only", True)
    )
    gex_fallback_tactical_only = bool(
        approval_cfg.get("gex_fallback_tactical_only", True)
    )
    gex_fallback_requires_clean_non_gex = bool(
        approval_cfg.get("gex_fallback_requires_clean_non_gex", True)
    )
    # entry_tol_width_pct / entry_tol_floor read earlier (before pricer subprocess)
    require_spot_alignment = bool(approval_cfg.get("require_spot_alignment", True))
    spot_alignment_require_live = bool(approval_cfg.get("spot_alignment_require_live", True))
    max_spot_asof_drift_pct = fnum(approval_cfg.get("max_spot_asof_drift_pct", 0.35))
    max_bull_call_long_otm_pct = fnum(approval_cfg.get("max_bull_call_long_otm_pct"))
    max_bear_put_long_otm_pct = fnum(approval_cfg.get("max_bear_put_long_otm_pct"))
    exclude_debit_moneyness_fail_from_output = bool(
        approval_cfg.get("exclude_debit_moneyness_fail_from_output", False)
    )
    min_credit_no_touch_pct = fnum(approval_cfg.get("min_credit_no_touch_pct"))
    credit_no_touch_require_data = bool(approval_cfg.get("credit_no_touch_require_data", False))
    ic_exempt_from_no_touch = bool(approval_cfg.get("ic_exempt_from_no_touch", False))
    enable_restrike_optimizer = bool(approval_cfg.get("enable_restrike_optimizer", True))
    invalidation_eval_mode = str(approval_cfg.get("invalidation_eval_mode", "auto")).strip().lower()
    if invalidation_eval_mode not in {"auto", "live", "asof_close"}:
        invalidation_eval_mode = "auto"
    if invalidation_eval_mode == "asof_close":
        use_asof_close_for_invalidation = True
    elif invalidation_eval_mode == "live":
        use_asof_close_for_invalidation = False
    else:
        use_asof_close_for_invalidation = bool(asof < dt.date.today())
    gates_cfg_local = cfg.get("gates", {}) if isinstance(cfg, dict) else {}
    tactical_max_debit_pct_width = fnum(gates_cfg_local.get("tactical_max_debit_pct_width", 0.35))
    tactical_debit_pct_width_tolerance = fnum(
        gates_cfg_local.get(
            "tactical_debit_pct_width_tolerance",
            approval_cfg.get("tactical_debit_pct_width_tolerance", 0.02),
        )
    )
    if not np.isfinite(tactical_debit_pct_width_tolerance) or tactical_debit_pct_width_tolerance < 0:
        tactical_debit_pct_width_tolerance = 0.0
    min_live_reward_risk = fnum(gates_cfg_local.get("min_live_reward_risk", 1.50))
    min_debit_reward_risk = fnum(gates_cfg_local.get("min_debit_reward_risk", min_live_reward_risk))
    min_credit_reward_risk = fnum(gates_cfg_local.get("min_credit_reward_risk", min_live_reward_risk))
    min_credit_pct_width_cfg = fnum(gates_cfg_local.get("min_credit_pct_width", 0.30))
    max_credit_pct_width_cfg = fnum(gates_cfg_local.get("max_credit_pct_width", 0.55))
    if not np.isfinite(min_debit_reward_risk) or min_debit_reward_risk < 0:
        min_debit_reward_risk = min_live_reward_risk
    if not np.isfinite(min_credit_reward_risk) or min_credit_reward_risk < 0:
        min_credit_reward_risk = min_live_reward_risk
    if not np.isfinite(min_credit_pct_width_cfg) or min_credit_pct_width_cfg <= 0:
        min_credit_pct_width_cfg = 0.30
    if not np.isfinite(max_credit_pct_width_cfg) or max_credit_pct_width_cfg <= 0:
        max_credit_pct_width_cfg = 0.55
    ideal_credit_low_pct = max(0.30, min_credit_pct_width_cfg)
    ideal_credit_high_pct = min(0.40, max_credit_pct_width_cfg)
    if ideal_credit_high_pct < ideal_credit_low_pct:
        ideal_credit_high_pct = ideal_credit_low_pct
    final_validity_gate_enabled = bool(approval_cfg.get("final_validity_gate_enabled", True))
    valid_trade_min_pop = normalize_probability(approval_cfg.get("valid_trade_min_pop", medium_review_min_pop))
    valid_trade_min_confidence_score = fnum(
        approval_cfg.get("valid_trade_min_confidence_score", medium_review_min_confidence_score)
    )
    valid_trade_min_edge_score = fnum(
        approval_cfg.get("valid_trade_min_edge_score", medium_review_min_edge_score)
    )
    valid_trade_require_positive_edge_pct = bool(
        approval_cfg.get("valid_trade_require_positive_edge_pct", True)
    )
    valid_trade_require_likelihood_pass = bool(
        approval_cfg.get("valid_trade_require_likelihood_pass", True)
    )
    valid_trade_require_live_price = bool(
        approval_cfg.get("valid_trade_require_live_price", True)
    )
    valid_trade_block_earnings_risk = bool(
        approval_cfg.get("valid_trade_block_earnings_risk", True)
    )
    valid_trade_block_liquidity_flags = bool(
        approval_cfg.get("valid_trade_block_liquidity_flags", True)
    )
    valid_trade_min_credit_pct_width = fnum(
        approval_cfg.get("valid_trade_min_credit_pct_width", max(0.25, min_credit_pct_width_cfg))
    )
    valid_trade_max_credit_pct_width = fnum(
        approval_cfg.get("valid_trade_max_credit_pct_width", max_credit_pct_width_cfg)
    )
    valid_trade_max_debit_pct_width = fnum(
        approval_cfg.get("valid_trade_max_debit_pct_width", gates_cfg_local.get("max_debit_pct_width", 0.45))
    )
    valid_trade_min_debit_reward_risk = fnum(
        approval_cfg.get("valid_trade_min_debit_reward_risk", min_debit_reward_risk)
    )
    if not np.isfinite(valid_trade_min_pop):
        valid_trade_min_pop = 0.60
    if not np.isfinite(valid_trade_min_confidence_score):
        valid_trade_min_confidence_score = 5.0
    if not np.isfinite(valid_trade_min_edge_score):
        valid_trade_min_edge_score = 1.0
    if not np.isfinite(valid_trade_min_credit_pct_width) or valid_trade_min_credit_pct_width <= 0:
        valid_trade_min_credit_pct_width = 0.25
    if not np.isfinite(valid_trade_max_credit_pct_width) or valid_trade_max_credit_pct_width <= 0:
        valid_trade_max_credit_pct_width = max_credit_pct_width_cfg
    if not np.isfinite(valid_trade_max_debit_pct_width) or valid_trade_max_debit_pct_width <= 0:
        valid_trade_max_debit_pct_width = 0.45
    if not np.isfinite(valid_trade_min_debit_reward_risk) or valid_trade_min_debit_reward_risk < 0:
        valid_trade_min_debit_reward_risk = min_debit_reward_risk
    if not np.isfinite(max_spot_asof_drift_pct) or max_spot_asof_drift_pct < 0:
        max_spot_asof_drift_pct = 0.35
    if not np.isfinite(max_bull_call_long_otm_pct) or max_bull_call_long_otm_pct < 0:
        max_bull_call_long_otm_pct = math.nan
    if not np.isfinite(max_bear_put_long_otm_pct) or max_bear_put_long_otm_pct < 0:
        max_bear_put_long_otm_pct = math.nan
    if not np.isfinite(max_abs_short_delta_shield) or max_abs_short_delta_shield <= 0:
        max_abs_short_delta_shield = 0.20
    if not np.isfinite(core_size_mult) or core_size_mult <= 0:
        core_size_mult = 1.00
    if not np.isfinite(tactical_size_mult) or tactical_size_mult <= 0:
        tactical_size_mult = 0.50
    if not np.isfinite(scout_size_mult) or scout_size_mult <= 0:
        scout_size_mult = 0.25
    if not np.isfinite(pilot_size_mult) or pilot_size_mult <= 0:
        pilot_size_mult = 0.10
    if not np.isfinite(pilot_max_loss) or pilot_max_loss <= 0:
        pilot_max_loss = 150.0
    if not np.isfinite(pilot_min_ev_ml):
        pilot_min_ev_ml = 1.50
    if not np.isfinite(pilot_high_pop_min) or pilot_high_pop_min <= 0:
        pilot_high_pop_min = 0.25
    if not np.isfinite(pilot_floor_pop_min) or pilot_floor_pop_min <= 0:
        pilot_floor_pop_min = 0.15
    if pilot_floor_pop_min > pilot_high_pop_min:
        pilot_floor_pop_min = pilot_high_pop_min
    if not np.isfinite(pilot_min_conviction):
        pilot_min_conviction = 40.0
    if not np.isfinite(pilot_pass_min_signals) or pilot_pass_min_signals <= 0:
        pilot_pass_min_signals = 100.0
    if not np.isfinite(pilot_pass_min_edge):
        pilot_pass_min_edge = 1.0
    if not np.isfinite(pilot_min_dte) or pilot_min_dte < 0:
        pilot_min_dte = 5.0
    if not np.isfinite(pilot_max_dte) or pilot_max_dte <= 0:
        pilot_max_dte = 45.0
    if pilot_max_dte < pilot_min_dte:
        pilot_max_dte = pilot_min_dte
    if not np.isfinite(pilot_min_reward_risk) or pilot_min_reward_risk < 0:
        pilot_min_reward_risk = 1.50
    if not np.isfinite(pilot_max_debit_frac) or pilot_max_debit_frac <= 0:
        pilot_max_debit_frac = 0.25
    if not np.isfinite(pilot_max_long_otm_pct) or pilot_max_long_otm_pct < 0:
        pilot_max_long_otm_pct = 0.15
    if not np.isfinite(native_pilot_max_loss) or native_pilot_max_loss <= 0:
        native_pilot_max_loss = pilot_max_loss
    if not np.isfinite(native_pilot_min_partial_ev_ml):
        native_pilot_min_partial_ev_ml = 0.0
    if not np.isfinite(native_pilot_min_signals) or native_pilot_min_signals <= 0:
        native_pilot_min_signals = 60.0
    if not np.isfinite(native_pilot_min_edge):
        native_pilot_min_edge = pilot_pass_min_edge
    if not np.isfinite(scout_min_edge_pct):
        scout_min_edge_pct = (
            bull_call_evidence_min_edge
            if np.isfinite(bull_call_evidence_min_edge)
            else 5.0
        )
    if not np.isfinite(scout_max_edge_pct):
        scout_max_edge_pct = max(tactical_min_edge_pct, scout_min_edge_pct)
    if scout_max_edge_pct < scout_min_edge_pct:
        scout_max_edge_pct = scout_min_edge_pct
    if (
        not np.isfinite(bull_call_short_dte_high_edge_max_dte)
        or bull_call_short_dte_high_edge_max_dte <= 0
    ):
        bull_call_short_dte_high_edge_max_dte = 31
    if (
        not np.isfinite(bull_call_short_dte_high_edge_min_edge)
        or bull_call_short_dte_high_edge_min_edge <= 0
    ):
        bull_call_short_dte_high_edge_min_edge = 13.2
    if (
        not np.isfinite(bull_call_pinned_continuation_min_edge)
        or bull_call_pinned_continuation_min_edge < 0
    ):
        bull_call_pinned_continuation_min_edge = 8.0
    if (
        not np.isfinite(bull_call_pinned_continuation_min_signals)
        or bull_call_pinned_continuation_min_signals < 0
    ):
        bull_call_pinned_continuation_min_signals = 120
    if (
        not np.isfinite(bull_call_pinned_continuation_min_conviction)
        or bull_call_pinned_continuation_min_conviction < 0
    ):
        bull_call_pinned_continuation_min_conviction = 40
    if (
        not np.isfinite(bull_call_pinned_continuation_core_min_conviction)
        or bull_call_pinned_continuation_core_min_conviction < bull_call_pinned_continuation_min_conviction
    ):
        bull_call_pinned_continuation_core_min_conviction = max(
            55.0, bull_call_pinned_continuation_min_conviction
        )
    if (
        not np.isfinite(bull_call_pinned_continuation_min_dte)
        or bull_call_pinned_continuation_min_dte < 0
    ):
        bull_call_pinned_continuation_min_dte = 31
    if (
        not np.isfinite(bull_call_pinned_continuation_max_dte)
        or bull_call_pinned_continuation_max_dte < bull_call_pinned_continuation_min_dte
    ):
        bull_call_pinned_continuation_max_dte = 45
    if (
        not np.isfinite(bull_call_pinned_continuation_min_reward_risk)
        or bull_call_pinned_continuation_min_reward_risk < 0
    ):
        bull_call_pinned_continuation_min_reward_risk = 2.0
    if (
        not np.isfinite(bull_call_pinned_continuation_max_debit_frac)
        or bull_call_pinned_continuation_max_debit_frac <= 0
    ):
        bull_call_pinned_continuation_max_debit_frac = 0.35
    if not bull_call_pinned_continuation_allowed_gex_contexts:
        bull_call_pinned_continuation_allowed_gex_contexts = {"pinned_resistance_above_call_wall"}
    if not bull_call_pinned_continuation_allowed_regime_confidences:
        bull_call_pinned_continuation_allowed_regime_confidences = {"Medium", "High"}
    if not np.isfinite(fire_delta_proxy_max_long_otm_pct) or fire_delta_proxy_max_long_otm_pct < 0:
        fire_delta_proxy_max_long_otm_pct = 0.02
    if not bear_put_scout_likelihood_strengths:
        bear_put_scout_likelihood_strengths = {"NEGATIVE"}
    if not np.isfinite(bear_put_scout_min_signals) or bear_put_scout_min_signals < 0:
        bear_put_scout_min_signals = 60
    if not np.isfinite(bear_put_scout_min_dte) or bear_put_scout_min_dte < 0:
        bear_put_scout_min_dte = 14
    if not np.isfinite(bear_put_scout_max_dte) or bear_put_scout_max_dte <= 0:
        bear_put_scout_max_dte = 35
    if not np.isfinite(bear_put_scout_max_iv_rank) or bear_put_scout_max_iv_rank <= 0:
        bear_put_scout_max_iv_rank = 30
    if not np.isfinite(bear_put_scout_max_vix) or bear_put_scout_max_vix <= 0:
        bear_put_scout_max_vix = 20
    if not np.isfinite(bear_put_scout_min_reward_risk) or bear_put_scout_min_reward_risk < 0:
        bear_put_scout_min_reward_risk = 2.0
    if not np.isfinite(bear_put_scout_max_debit_frac) or bear_put_scout_max_debit_frac <= 0:
        bear_put_scout_max_debit_frac = 0.35
    if not np.isfinite(tactical_min_conviction) or tactical_min_conviction < 0:
        tactical_min_conviction = 60
    if not np.isfinite(tactical_min_conviction_grace) or tactical_min_conviction_grace < 0:
        tactical_min_conviction_grace = 0.0
    if not np.isfinite(tactical_min_edge_pct):
        tactical_min_edge_pct = 0.0
    if not np.isfinite(event_momentum_scout_min_conviction):
        event_momentum_scout_min_conviction = 30
    if not np.isfinite(event_momentum_scout_max_dte) or event_momentum_scout_max_dte <= 0:
        event_momentum_scout_max_dte = 35
    if not np.isfinite(event_momentum_scout_min_reward_risk):
        event_momentum_scout_min_reward_risk = 1.8
    if not np.isfinite(event_momentum_scout_max_debit_frac) or event_momentum_scout_max_debit_frac <= 0:
        event_momentum_scout_max_debit_frac = 0.35
    if not np.isfinite(debit_momentum_scout_min_conviction):
        debit_momentum_scout_min_conviction = 40
    if not np.isfinite(debit_momentum_scout_min_edge_pct):
        debit_momentum_scout_min_edge_pct = 8.0
    if not np.isfinite(debit_momentum_scout_bear_min_edge_pct):
        debit_momentum_scout_bear_min_edge_pct = max(12.0, debit_momentum_scout_min_edge_pct)
    if not np.isfinite(debit_momentum_scout_min_signals) or debit_momentum_scout_min_signals < 0:
        debit_momentum_scout_min_signals = 100
    if not np.isfinite(debit_momentum_scout_min_dte) or debit_momentum_scout_min_dte < 0:
        debit_momentum_scout_min_dte = 14
    if not np.isfinite(debit_momentum_scout_max_dte) or debit_momentum_scout_max_dte <= 0:
        debit_momentum_scout_max_dte = 45
    if not np.isfinite(debit_momentum_scout_min_reward_risk):
        debit_momentum_scout_min_reward_risk = 2.0
    if not np.isfinite(debit_momentum_scout_max_debit_frac) or debit_momentum_scout_max_debit_frac <= 0:
        debit_momentum_scout_max_debit_frac = 0.35
    if not np.isfinite(debit_momentum_scout_max_iv_rank) or debit_momentum_scout_max_iv_rank <= 0:
        debit_momentum_scout_max_iv_rank = 80
    if not np.isfinite(debit_momentum_scout_min_regime_score):
        debit_momentum_scout_min_regime_score = 55
    if not debit_momentum_scout_bear_likelihood_strengths:
        debit_momentum_scout_bear_likelihood_strengths = {"MODERATE", "STRONG"}
    if not np.isfinite(tactical_min_signals) or tactical_min_signals <= 0:
        tactical_min_signals = min_signals
    if not np.isfinite(tactical_min_signals_grace) or tactical_min_signals_grace < 0:
        tactical_min_signals_grace = 0.0
    enforce_pretrade_caps = bool(approval_cfg.get("enforce_pretrade_portfolio_caps", False))
    pretrade_caps_require_data = bool(approval_cfg.get("pretrade_caps_require_data", False))
    pretrade_open_positions_csv = str(approval_cfg.get("pretrade_open_positions_csv", "")).strip()
    block_same_underlying_option_overlap = bool(
        approval_cfg.get("block_same_underlying_option_overlap", True)
    )
    if args.historical_replay and enforce_pretrade_caps and not pretrade_open_positions_csv:
        enforce_pretrade_caps = False
        pretrade_caps_require_data = False
        block_same_underlying_option_overlap = False
    risk_cfg = cfg.get("playbook", {}).get("risk_limits", {}) if isinstance(cfg, dict) else {}
    short_put_limit = fnum(risk_cfg.get("short_put_max_share", 0.35))
    symbol_limit = fnum(risk_cfg.get("single_symbol_max_share", 0.10))
    expiry_limit = fnum(risk_cfg.get("single_expiry_max_share_short_put", 0.25))
    if not np.isfinite(short_put_limit) or short_put_limit <= 0:
        short_put_limit = 0.35
    if not np.isfinite(symbol_limit) or symbol_limit <= 0:
        symbol_limit = 0.10
    if not np.isfinite(expiry_limit) or expiry_limit <= 0:
        expiry_limit = 0.25

    def gate_context(row):
        net_type = str(row.get("net_type", "")).strip().lower()
        live_status = str(row.get("live_status", "")).strip()
        live_net = fnum(row.get("live_net_bid_ask"))
        gate_pass_raw = bool(row.get("gate_pass_live")) if pd.notna(row.get("gate_pass_live")) else False
        _, gate_target, _ = parse_gate_value(row.get("entry_gate", ""))
        # Width-based tolerance: max(floor, width × pct)
        w = fnum(row.get("width"))
        if not np.isfinite(w) or w <= 0:
            w = 0.0
        width_tol = w * entry_tol_width_pct if entry_tol_width_pct > 0 else 0.0
        tol_total = max(entry_tol_floor, width_tol)
        near_miss = False
        pass_effective = gate_pass_raw
        miss_abs = math.nan
        strict_gate_pass = False
        if np.isfinite(gate_target) and np.isfinite(live_net):
            strict_gate_pass = entry_gate_strict_pass(net_type, live_net, gate_target)
            if net_type == "debit":
                miss_abs = max(0.0, live_net - gate_target)
            else:
                miss_abs = max(0.0, gate_target - live_net)
            if (not strict_gate_pass) and miss_abs <= tol_total and (
                gate_pass_raw or live_status == "fails_live_entry_gate"
            ):
                near_miss = True
                pass_effective = True

        return {
            "gate_target": gate_target,
            "gate_live_net": live_net,
            "gate_tol_total": tol_total,
            "gate_miss_abs": miss_abs,
            "gate_pass_strict": bool(strict_gate_pass),
            "gate_pass_effective": bool(pass_effective),
            "gate_near_miss": bool(near_miss),
        }

    gate_ctx_df = pd.DataFrame([gate_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), gate_ctx_df], axis=1)
    mdf["live_reward_risk"] = mdf.apply(
        lambda row: (
            fnum(row.get("live_max_profit")) / fnum(row.get("live_max_loss"))
            if np.isfinite(fnum(row.get("live_max_profit")))
            and np.isfinite(fnum(row.get("live_max_loss")))
            and fnum(row.get("live_max_loss")) > 0
            else math.nan
        ),
        axis=1,
    )

    _spy_5d_ret = fnum(_macro.get("spy_5d_ret", 0.0))
    if not np.isfinite(_spy_5d_ret):
        _spy_5d_ret = 0.0
    if not np.isfinite(_vix_level):
        _vix_level = 20.0

    def _conditioning_token(profile: str, key: str) -> str:
        for part in str(profile or "").split(";"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            if k.strip().lower() == key.lower():
                return v.strip().lower()
        return ""

    def market_regime_context(row):
        strategy_local = str(row.get("strategy", "")).strip()
        macro_regime = str(_macro_regime or "neutral").strip().lower()
        profile = str(row.get("conditioning_profile", "") or "")
        trend = _conditioning_token(profile, "trend")
        range_neutral = _conditioning_token(profile, "range_neutral")
        flow_dir = str(row.get("flow_direction", "")).strip().lower()
        flow_conf = str(row.get("flow_confirmation", "")).strip().lower()
        contract_flow = str(row.get("contract_flow_confirmation", "")).strip().lower()
        gex_regime = str(row.get("gex_regime", "")).strip().lower()

        spot_ref = fnum(row.get("spot_live_effective"))
        if not np.isfinite(spot_ref) or spot_ref <= 0:
            spot_ref = fnum(row.get("spot_asof_close"))
        if not np.isfinite(spot_ref) or spot_ref <= 0:
            spot_ref = fnum(spot_map.get(str(row.get("ticker", "")).strip().upper()))
        breakeven = fnum(row.get("breakeven"))
        if not np.isfinite(breakeven) and strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
            long_strike = fnum(row.get("long_strike"))
            live_net = fnum(row.get("live_net_bid_ask"))
            if np.isfinite(long_strike) and np.isfinite(live_net):
                breakeven = long_strike + live_net if strategy_local == "Bull Call Debit" else long_strike - live_net
        be_distance_pct = math.nan
        if np.isfinite(spot_ref) and spot_ref > 0 and np.isfinite(breakeven):
            if strategy_local == "Bear Put Debit":
                be_distance_pct = (spot_ref - breakeven) / spot_ref
            else:
                be_distance_pct = (breakeven - spot_ref) / spot_ref

        score = 55.0
        reasons = []
        if macro_regime == "risk_on":
            score += 10.0
            reasons.append("macro risk_on")
        elif macro_regime == "risk_off":
            score -= 5.0
            reasons.append("macro risk_off")
        else:
            reasons.append(f"macro {macro_regime or 'neutral'}")

        if _spy_5d_ret >= 0.02:
            score += 10.0
            reasons.append(f"SPY5d {_spy_5d_ret:+.1%}")
        elif _spy_5d_ret >= 0.01:
            score += 5.0
            reasons.append(f"SPY5d {_spy_5d_ret:+.1%}")
        elif _spy_5d_ret <= -0.03:
            score -= 10.0
            reasons.append(f"SPY5d {_spy_5d_ret:+.1%}")
        elif _spy_5d_ret <= -0.015:
            score -= 5.0
            reasons.append(f"SPY5d {_spy_5d_ret:+.1%}")

        if _vix_level > 25:
            score -= 10.0
            reasons.append(f"VIX {_vix_level:.1f}")
        elif _vix_level > 20:
            score -= 5.0
            reasons.append(f"VIX {_vix_level:.1f}")

        if strategy_local == "Bull Call Debit":
            if trend == "up":
                score += 10.0
                reasons.append("ticker trend up")
            elif trend == "down":
                score -= 10.0
                reasons.append("ticker trend down")
            elif trend:
                reasons.append(f"ticker trend {trend}")
            if range_neutral == "true":
                score -= 3.0
                reasons.append("range-neutral")
            if gex_regime == "volatile":
                score += 5.0
                reasons.append("GEX volatile")
            elif gex_regime == "pinned":
                score -= 8.0
                reasons.append("GEX pinned")
            if np.isfinite(be_distance_pct):
                if be_distance_pct <= 0.03:
                    score += 10.0
                elif be_distance_pct <= 0.05:
                    score += 4.0
                elif be_distance_pct > 0.10:
                    score -= 12.0
                elif be_distance_pct > 0.07:
                    score -= 8.0
                elif be_distance_pct > 0.05:
                    score -= 4.0
                reasons.append(f"BE {be_distance_pct:+.1%}")
            if contract_flow == "confirmed":
                score += 10.0
                reasons.append("contract flow confirmed")
            elif contract_flow in {"contra", "directional"}:
                score -= 15.0
                reasons.append(f"contract flow {contract_flow}")
            if flow_conf == "confirmed" and flow_dir == "bullish":
                score += 8.0
                reasons.append("ticker flow bullish")
            elif flow_conf == "confirmed" and flow_dir == "bearish":
                score -= 25.0
                reasons.append("ticker flow bearish")
        elif strategy_local == "Bear Put Debit":
            # The generic macro block above is intentionally bull-biased.
            # Invert its net effect for bearish debit spreads so bear setups
            # are not scored through a bullish lens.
            if macro_regime == "risk_on":
                score -= 20.0
                reasons.append("bear setup vs risk_on")
            elif macro_regime == "risk_off":
                score += 15.0
                reasons.append("bear setup risk_off")

            if _spy_5d_ret >= 0.02:
                score -= 20.0
                reasons.append("bear setup vs strong SPY")
            elif _spy_5d_ret >= 0.01:
                score -= 10.0
                reasons.append("bear setup vs SPY up")
            elif _spy_5d_ret <= -0.03:
                score += 20.0
                reasons.append("SPY breakdown")
            elif _spy_5d_ret <= -0.015:
                score += 10.0
                reasons.append("SPY weak")

            if trend == "down":
                score += 10.0
                reasons.append("ticker trend down")
            elif trend == "up":
                score -= 10.0
                reasons.append("ticker trend up")
            elif trend:
                reasons.append(f"ticker trend {trend}")
            if range_neutral == "true":
                score -= 3.0
                reasons.append("range-neutral")
            if gex_regime == "volatile":
                score += 5.0
                reasons.append("GEX volatile")
            elif gex_regime == "pinned":
                score -= 8.0
                reasons.append("GEX pinned")
            if np.isfinite(be_distance_pct):
                if be_distance_pct <= 0.03:
                    score += 10.0
                elif be_distance_pct <= 0.05:
                    score += 4.0
                elif be_distance_pct > 0.10:
                    score -= 12.0
                elif be_distance_pct > 0.07:
                    score -= 8.0
                elif be_distance_pct > 0.05:
                    score -= 4.0
                reasons.append(f"BE {be_distance_pct:+.1%}")
            if contract_flow == "confirmed":
                score += 10.0
                reasons.append("contract flow confirmed")
            elif contract_flow in {"contra", "directional"}:
                score -= 15.0
                reasons.append(f"contract flow {contract_flow}")
            if flow_conf == "confirmed" and flow_dir == "bearish":
                score += 8.0
                reasons.append("ticker flow bearish")
            elif flow_conf == "confirmed" and flow_dir == "bullish":
                score -= 25.0
                reasons.append("ticker flow bullish")

        score = max(0.0, min(100.0, score))
        if np.isfinite(bull_call_regime_high_score) and score >= bull_call_regime_high_score:
            confidence = "High"
        elif np.isfinite(bull_call_regime_low_score) and score < bull_call_regime_low_score:
            confidence = "Low"
        else:
            confidence = "Medium"
        return {
            "market_regime": macro_regime or "neutral",
            "market_regime_score": round(score, 1),
            "market_regime_confidence": confidence,
            "market_regime_reason": "; ".join(reasons[:8]),
            "spy_5d_ret": _spy_5d_ret,
            "vix_level": _vix_level,
            "breakeven_distance_pct": be_distance_pct,
        }

    market_ctx_df = pd.DataFrame([market_regime_context(r) for _, r in mdf.iterrows()])
    mdf = mdf.reset_index(drop=True)
    # Overwrite any pre-existing live/pricer placeholders.  Concatenating here
    # can create duplicate columns, and pandas row.get("breakeven_distance_pct")
    # then returns a Series instead of a scalar, which silently drops distance
    # points from the confidence model.
    for _ctx_col in market_ctx_df.columns:
        mdf[_ctx_col] = market_ctx_df[_ctx_col].values

    def gex_wall_context(row):
        gex_regime = str(row.get("gex_regime", "")).strip().lower()
        strategy_local = str(row.get("strategy", "")).strip()
        live_net_val = fnum(row.get("live_net_bid_ask"))
        long_strike_val = fnum(row.get("long_strike"))
        support_val = fnum(row.get("gex_support"))
        resistance_val = fnum(row.get("gex_resistance"))
        if not gex_regime:
            return ""
        if gex_regime == "volatile":
            return "volatile_avoid_credit" if str(row.get("track", "")).strip().upper() == "SHIELD" else "volatile_breakout_possible"
        if strategy_local == "Bull Call Debit" and np.isfinite(live_net_val) and np.isfinite(long_strike_val):
            be = long_strike_val + live_net_val
            if np.isfinite(resistance_val):
                return "pinned_supportive_below_call_wall" if be <= resistance_val else "pinned_resistance_above_call_wall"
            return "pinned_no_call_wall"
        if strategy_local == "Bear Put Debit" and np.isfinite(live_net_val) and np.isfinite(long_strike_val):
            be = long_strike_val - live_net_val
            if np.isfinite(support_val):
                return "pinned_supportive_above_put_wall" if be >= support_val else "pinned_support_below_put_wall"
            return "pinned_no_put_wall"
        if strategy_local in {"Iron Condor", "Iron Butterfly"}:
            return "pinned_income_constructive"
        return "pinned"

    mdf["gex_wall_context"] = mdf.apply(gex_wall_context, axis=1)

    def approval_regime_label(row):
        vix_val = fnum(row.get("vix_level"))
        spy_ret = fnum(row.get("spy_5d_ret"))
        if np.isfinite(vix_val) and vix_val >= 22:
            vol_bucket = "high"
        elif np.isfinite(vix_val) and vix_val < 16:
            vol_bucket = "low"
        else:
            vol_bucket = "mid"
        if np.isfinite(spy_ret) and spy_ret <= -0.015:
            direction_bucket = "down"
        elif np.isfinite(spy_ret) and spy_ret >= 0.015:
            direction_bucket = "up"
        else:
            direction_bucket = "range"
        return f"{vol_bucket}/{direction_bucket}"

    def profit_safety_model(row):
        strategy_local = str(row.get("strategy", "") or "").strip()
        flow_conf = str(row.get("flow_confirmation", "") or "").strip().lower()
        flow_dir = str(row.get("flow_direction", "") or "").strip().lower()
        contract_flow = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
        gex_ctx = str(row.get("gex_wall_context", "") or "").strip().lower()
        gex_regime = str(row.get("gex_regime", "") or "").strip().lower()
        iv_rank = fnum(row.get("iv_rank"))
        iv_val = fnum(row.get("iv"))
        if not np.isfinite(iv_val):
            iv_val = fnum(row.get("iv30d"))
        hv_val = math.nan
        for hv_col in ["hv", "hv20", "hv_20", "realized_vol", "realized_vol_20d", "rv20", "rv_20"]:
            hv_val = fnum(row.get(hv_col))
            if np.isfinite(hv_val):
                break
        be_distance_pct = fnum(row.get("breakeven_distance_pct"))
        if np.isfinite(be_distance_pct):
            be_distance_pct = abs(be_distance_pct)
        edge_pct_local = fnum(row.get("edge_pct"))
        hist_success = normalize_probability(row.get("hist_success_pct"))
        regime = str(row.get("approval_regime", "") or approval_regime_label(row)).strip().lower()
        expected_flow_dir = ""
        if strategy_local in {"Bull Call Debit", "Bull Put Credit"}:
            expected_flow_dir = "bullish"
        elif strategy_local in {"Bear Put Debit", "Bear Call Credit"}:
            expected_flow_dir = "bearish"
        elif strategy_local in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            expected_flow_dir = "neutral"

        flow_score = 0
        flow_edges = []
        if flow_conf == "confirmed" and (not expected_flow_dir or expected_flow_dir == "neutral" or flow_dir == expected_flow_dir):
            flow_score += 2
            flow_edges.append("flow_confirmation")
        elif flow_conf == "confirmed":
            flow_edges.append("flow_mismatch")
        if contract_flow == "confirmed":
            flow_score += 1
            flow_edges.append("contract_flow_confirmation")
        elif contract_flow in {"contra", "directional"}:
            flow_edges.append(f"contract_flow_{contract_flow}")
        flow_score = int(max(0, min(3, flow_score)))

        technical_score = 0
        technical_edges = []
        if gex_ctx in {"pinned_supportive_below_call_wall", "pinned_supportive_above_put_wall", "pinned_income_constructive", "volatile_breakout_possible"}:
            technical_score += 2
            technical_edges.append("technical_gex_level")
        elif gex_regime == "pinned" and strategy_local in {"Iron Condor", "Iron Butterfly"}:
            technical_score += 2
            technical_edges.append("technical_gex_level")
        elif gex_regime:
            technical_score += 1
        market_conf = str(row.get("market_regime_confidence", "") or "").strip().lower()
        if market_conf == "high":
            technical_score += 1
            technical_edges.append("market_regime_alignment")
        technical_score = int(max(0, min(3, technical_score)))

        volatility_score = 0
        volatility_edges = []
        if np.isfinite(iv_val) and np.isfinite(hv_val) and iv_val > hv_val:
            volatility_score = 2
            volatility_edges.append("iv_gt_hv")
        elif strategy_local in {"Iron Condor", "Iron Butterfly", "Bear Call Credit", "Bull Put Credit"} and np.isfinite(iv_rank) and iv_rank >= 50:
            volatility_score = 1
            volatility_edges.append("iv_rank_income_edge")
        elif strategy_local in {"Bull Call Debit", "Bear Put Debit"} and np.isfinite(iv_rank) and iv_rank <= 55:
            volatility_score = 1
            volatility_edges.append("iv_not_expensive_for_debit")
        volatility_score = int(max(0, min(2, volatility_score)))

        distance_score = 0
        distance_edges = []
        if strategy_local in {"Iron Condor", "Iron Butterfly"}:
            spot_val = fnum(row.get("spot_live_effective"))
            if not np.isfinite(spot_val) or spot_val <= 0:
                spot_val = fnum(row.get("spot_asof_close"))
            if not np.isfinite(spot_val) or spot_val <= 0:
                spot_val = fnum(row.get("close"))
            be_low = fnum(row.get("breakeven_low"))
            be_high = fnum(row.get("breakeven_high"))
            range_distance_pct = math.nan
            if (
                np.isfinite(spot_val)
                and spot_val > 0
                and np.isfinite(be_low)
                and np.isfinite(be_high)
                and be_low < spot_val < be_high
            ):
                range_distance_pct = min((spot_val - be_low) / spot_val, (be_high - spot_val) / spot_val)
            if np.isfinite(range_distance_pct):
                if range_distance_pct >= 0.05:
                    distance_score = 2
                    distance_edges.append("wide_income_range")
                elif range_distance_pct >= 0.03:
                    distance_score = 1
                    distance_edges.append("moderate_income_range")
            elif np.isfinite(hist_success) and hist_success >= 0.60:
                distance_score = 1
                distance_edges.append("range_probability")
        elif np.isfinite(be_distance_pct):
            if be_distance_pct <= 0.03:
                distance_score = 2
                distance_edges.append("close_breakeven")
            elif be_distance_pct <= 0.05:
                distance_score = 1
                distance_edges.append("moderate_breakeven")
        distance_score = int(max(0, min(2, distance_score)))

        mean_reversion_edges = []
        if strategy_local in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"} and gex_regime == "pinned":
            mean_reversion_edges.append("mean_reversion_setup")
        elif regime.endswith("/range") and gex_regime == "pinned" and strategy_local in {"Bear Call Credit", "Bull Put Credit"}:
            mean_reversion_edges.append("mean_reversion_setup")

        explicit_edges = []
        explicit_edges.extend([x for x in flow_edges if x in {"flow_confirmation", "contract_flow_confirmation"}])
        explicit_edges.extend(technical_edges[:1])
        explicit_edges.extend(volatility_edges[:1])
        explicit_edges.extend(distance_edges[:1])
        explicit_edges.extend(mean_reversion_edges[:1])
        explicit_edges = list(dict.fromkeys(explicit_edges))
        confidence_score = flow_score + technical_score + volatility_score + distance_score
        if np.isfinite(confidence_high_min_score) and confidence_score >= confidence_high_min_score:
            confidence_model = "High"
        elif np.isfinite(confidence_medium_min_score) and confidence_score >= confidence_medium_min_score:
            confidence_model = "Medium"
        else:
            confidence_model = "Reject"
        return pd.Series({
            "approval_regime": approval_regime_label(row),
            "edge_score": int(len(explicit_edges)),
            "edge_sources": ";".join(explicit_edges),
            "confidence_score": int(confidence_score),
            "confidence_model": confidence_model,
            "confidence_flow_score": flow_score,
            "confidence_technical_score": technical_score,
            "confidence_volatility_score": volatility_score,
            "confidence_distance_score": distance_score,
            "volatility_edge_missing": not (np.isfinite(iv_val) and np.isfinite(hv_val)),
        })

    mdf = pd.concat([mdf.reset_index(drop=True), mdf.apply(profit_safety_model, axis=1)], axis=1)

    def fire_breakout_exception(row) -> bool:
        if not allow_fire_breakout_exception:
            return False
        if str(row.get("track", "")).strip().upper() != "FIRE":
            return False
        if str(row.get("strategy", "")).strip() != "Bull Call Debit":
            return False
        if fire_breakout_require_risk_on and str(_macro_regime).strip().lower() != "risk_on":
            return False
        if str(row.get("verdict", "")).strip().upper() != "PASS":
            return False
        conv = fnum(row.get("conviction"))
        edge = fnum(row.get("edge_pct"))
        sig = fnum(row.get("signals"))
        long_delta = abs(_safe_delta(row.get("long_delta_live")))
        if np.isfinite(fire_breakout_min_conviction) and (not np.isfinite(conv) or conv < fire_breakout_min_conviction):
            return False
        if np.isfinite(fire_breakout_min_edge) and (not np.isfinite(edge) or edge < fire_breakout_min_edge):
            return False
        if np.isfinite(fire_breakout_min_signals) and (not np.isfinite(sig) or sig < fire_breakout_min_signals):
            return False
        if np.isfinite(fire_breakout_min_long_delta) and (not np.isfinite(long_delta) or long_delta < fire_breakout_min_long_delta):
            return False
        flow_dir = str(row.get("flow_direction", "")).strip().lower()
        flow_conf = str(row.get("flow_confirmation", "")).strip().lower()
        if flow_conf == "confirmed" and flow_dir != "bullish":
            return False
        if flow_dir != "bullish":
            return False
        if str(row.get("contract_flow_confirmation", "")).strip().lower() != "confirmed":
            return False
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        if not (ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)):
            return False
        gex_regime = str(row.get("gex_regime", "")).strip().lower()
        if not gex_regime:
            return False
        if gex_regime == "pinned":
            spot_ref = fnum(row.get("spot_live_effective"))
            if not np.isfinite(spot_ref) or spot_ref <= 0:
                spot_ref = fnum(row.get("spot_live_last"))
            if not np.isfinite(spot_ref) or spot_ref <= 0:
                spot_ref = fnum(row.get("spot_asof_close"))
            resistance = fnum(row.get("gex_resistance"))
            if not (np.isfinite(spot_ref) and spot_ref > 0 and np.isfinite(resistance)):
                return False
            wall_dist = abs(resistance / spot_ref - 1.0)
            if np.isfinite(fire_breakout_max_wall_distance_pct) and wall_dist > fire_breakout_max_wall_distance_pct:
                return False
        return True

    def bull_call_evidence_lane(row) -> bool:
        if not allow_bull_call_evidence_lane:
            return False
        if str(row.get("track", "")).strip().upper() != "FIRE":
            return False
        if str(row.get("strategy", "")).strip() != "Bull Call Debit":
            return False
        if str(row.get("verdict", "")).strip().upper() != "PASS":
            return False
        edge = fnum(row.get("edge_pct"))
        sig = fnum(row.get("signals"))
        conv = fnum(row.get("conviction"))
        long_delta = abs(_safe_delta(row.get("long_delta_live")))
        dte = fnum(row.get("dte"))
        reward_risk = fnum(row.get("live_reward_risk"))
        if np.isfinite(bull_call_evidence_min_edge) and (not np.isfinite(edge) or edge < bull_call_evidence_min_edge):
            return False
        if np.isfinite(bull_call_evidence_min_signals) and (not np.isfinite(sig) or sig < bull_call_evidence_min_signals):
            return False
        if np.isfinite(bull_call_evidence_min_conviction) and (not np.isfinite(conv) or conv < bull_call_evidence_min_conviction):
            return False
        if np.isfinite(bull_call_evidence_max_dte) and (not np.isfinite(dte) or dte > bull_call_evidence_max_dte):
            return False
        if (
            np.isfinite(bull_call_evidence_min_reward_risk)
            and (not np.isfinite(reward_risk) or reward_risk < bull_call_evidence_min_reward_risk)
        ):
            return False
        if (
            np.isfinite(bull_call_evidence_min_long_delta)
            and (not np.isfinite(long_delta) or long_delta < bull_call_evidence_min_long_delta)
        ):
            return False
        flow_dir = str(row.get("flow_direction", "")).strip().lower()
        flow_conf = str(row.get("flow_confirmation", "")).strip().lower()
        if flow_conf == "confirmed" and flow_dir == "bearish":
            return False
        contract_flow = str(row.get("contract_flow_confirmation", "")).strip().lower()
        if bull_call_evidence_require_contract_confirmed and contract_flow != "confirmed":
            return False
        if contract_flow in {"contra", "directional"}:
            return False
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        if not (ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)):
            return False
        return True

    def bear_put_evidence_lane(row) -> bool:
        """Positive-edge bearish debit lane; not a mirror of the contrarian Scout lane."""
        if not allow_bear_put_evidence_lane:
            return False
        if str(row.get("track", "")).strip().upper() != "FIRE":
            return False
        if str(row.get("strategy", "")).strip() != "Bear Put Debit":
            return False
        if str(row.get("verdict", "")).strip().upper() != "PASS":
            return False
        edge = fnum(row.get("edge_pct"))
        sig = fnum(row.get("signals"))
        conv = fnum(row.get("conviction"))
        long_delta = abs(_safe_delta(row.get("long_delta_live")))
        dte = fnum(row.get("dte"))
        reward_risk = fnum(row.get("live_reward_risk"))
        iv_rank = fnum(row.get("iv_rank"))
        width = fnum(row.get("width_live"))
        if not np.isfinite(width):
            width = fnum(row.get("width"))
        net = fnum(row.get("live_net_bid_ask"))
        if not np.isfinite(net):
            net = fnum(row.get("live_net_mark"))
        debit_frac = (
            net / width
            if np.isfinite(net) and np.isfinite(width) and width > 0
            else math.nan
        )
        if np.isfinite(bear_put_evidence_min_edge) and (not np.isfinite(edge) or edge < bear_put_evidence_min_edge):
            return False
        if np.isfinite(bear_put_evidence_min_signals) and (not np.isfinite(sig) or sig < bear_put_evidence_min_signals):
            return False
        if (
            np.isfinite(bear_put_evidence_min_conviction)
            and (not np.isfinite(conv) or conv < bear_put_evidence_min_conviction)
        ):
            return False
        if np.isfinite(bear_put_evidence_min_dte) and (not np.isfinite(dte) or dte < bear_put_evidence_min_dte):
            return False
        if np.isfinite(bear_put_evidence_max_dte) and (not np.isfinite(dte) or dte > bear_put_evidence_max_dte):
            return False
        if (
            np.isfinite(bear_put_evidence_min_reward_risk)
            and (not np.isfinite(reward_risk) or reward_risk < bear_put_evidence_min_reward_risk)
        ):
            return False
        if (
            np.isfinite(bear_put_evidence_min_long_delta)
            and (not np.isfinite(long_delta) or long_delta < bear_put_evidence_min_long_delta)
        ):
            return False
        if (
            np.isfinite(bear_put_evidence_max_debit_frac)
            and (not np.isfinite(debit_frac) or debit_frac > bear_put_evidence_max_debit_frac)
        ):
            return False
        if (
            np.isfinite(bear_put_evidence_max_iv_rank)
            and (not np.isfinite(iv_rank) or iv_rank > bear_put_evidence_max_iv_rank)
        ):
            return False
        flow_dir = str(row.get("flow_direction", "")).strip().lower()
        flow_conf = str(row.get("flow_confirmation", "")).strip().lower()
        if flow_conf == "confirmed" and flow_dir == "bullish":
            return False
        contract_flow = str(row.get("contract_flow_confirmation", "")).strip().lower()
        if bear_put_evidence_require_contract_confirmed and contract_flow != "confirmed":
            return False
        if contract_flow in {"contra", "directional", "weak_or_ambiguous", "unknown"}:
            return False
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        if not (ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)):
            return False
        return True

    def rank_likelihood_strength(value: str) -> int:
        s = str(value or "").strip().upper()
        order = {
            "NEGATIVE-STRONG": 0,
            "NEGATIVE": 1,
            "WEAK": 2,
            "MODERATE": 3,
            "STRONG": 4,
        }
        return order.get(s, -1)

    mdf["likelihood_strength"] = mdf.apply(
        lambda row: likelihood_strength(
            str(row.get("verdict", "")),
            fnum(row.get("edge_pct")),
            fnum(row.get("signals")),
        ),
        axis=1,
    )

    def bull_call_pinned_continuation_lane(row) -> bool:
        """Backtested tactical lane for pinned-resistance bull-call continuation."""
        if not allow_bull_call_pinned_continuation_lane:
            return False
        if str(row.get("track", "")).strip().upper() != "FIRE":
            return False
        if str(row.get("strategy", "")).strip() != "Bull Call Debit":
            return False
        if str(row.get("verdict", "")).strip().upper() != "PASS":
            return False
        strength = str(row.get("likelihood_strength", "")).strip()
        if rank_likelihood_strength(strength) < rank_likelihood_strength(
            bull_call_pinned_continuation_min_strength
        ):
            return False
        edge = fnum(row.get("edge_pct"))
        sig = fnum(row.get("signals"))
        conv = fnum(row.get("conviction"))
        dte = fnum(row.get("dte"))
        reward_risk = fnum(row.get("live_reward_risk"))
        width = fnum(row.get("width_live"))
        if not np.isfinite(width):
            width = fnum(row.get("width"))
        net = fnum(row.get("live_net_bid_ask"))
        if not np.isfinite(net):
            net = fnum(row.get("live_net_mark"))
        debit_frac = (
            net / width
            if np.isfinite(net) and np.isfinite(width) and width > 0
            else math.nan
        )
        if np.isfinite(bull_call_pinned_continuation_min_edge) and (
            not np.isfinite(edge) or edge < bull_call_pinned_continuation_min_edge
        ):
            return False
        if np.isfinite(bull_call_pinned_continuation_min_signals) and (
            not np.isfinite(sig) or sig < bull_call_pinned_continuation_min_signals
        ):
            return False
        if np.isfinite(bull_call_pinned_continuation_min_conviction) and (
            not np.isfinite(conv) or conv < bull_call_pinned_continuation_min_conviction
        ):
            return False
        if (
            not np.isfinite(dte)
            or dte < bull_call_pinned_continuation_min_dte
            or dte > bull_call_pinned_continuation_max_dte
        ):
            return False
        if np.isfinite(bull_call_pinned_continuation_min_reward_risk) and (
            not np.isfinite(reward_risk)
            or reward_risk < bull_call_pinned_continuation_min_reward_risk
        ):
            return False
        if np.isfinite(bull_call_pinned_continuation_max_debit_frac) and (
            not np.isfinite(debit_frac)
            or debit_frac > bull_call_pinned_continuation_max_debit_frac
        ):
            return False
        if (
            bull_call_pinned_continuation_block_short_dte_high_edge
            and np.isfinite(dte)
            and np.isfinite(edge)
            and dte < bull_call_short_dte_high_edge_max_dte
            and edge > bull_call_short_dte_high_edge_min_edge
        ):
            return False
        contract_flow = str(row.get("contract_flow_confirmation", "")).strip().lower()
        if bull_call_pinned_continuation_require_contract_confirmed and contract_flow != "confirmed":
            return False
        if contract_flow in {"contra", "directional"}:
            return False
        flow_dir = str(row.get("flow_direction", "")).strip().lower()
        flow_conf = str(row.get("flow_confirmation", "")).strip().lower()
        if flow_conf == "confirmed" and flow_dir == "bearish":
            return False
        if str(row.get("gex_wall_context", "")).strip() not in bull_call_pinned_continuation_allowed_gex_contexts:
            return False
        if (
            str(row.get("market_regime_confidence", "")).strip()
            not in bull_call_pinned_continuation_allowed_regime_confidences
        ):
            return False
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        if not (ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)):
            return False
        return True

    def _regime_weekly_live_ok(row) -> bool:
        live_status = str(row.get("live_status", "") or "").strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        return ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)

    def _regime_weekly_earnings_clean(row) -> bool:
        raw_label = row.get("earnings_label_stage1", "")
        label = "" if pd.isna(raw_label) else str(raw_label).strip().upper()
        if label in {"", "NAN", "NA", "NONE", "<NA>"}:
            label = ""
        if label:
            return label == "PASS"
        raw_notes = row.get("notes_stage1", "")
        if pd.isna(raw_notes) or str(raw_notes).strip().lower() in {"", "nan", "na", "none", "<na>"}:
            raw_notes = row.get("notes", "")
        text = str(raw_notes or "").upper()
        if "ER-RISK" in text or "CROSSED" in text or re.search(r"\bWITHIN\d+\b", text):
            return False
        match = re.search(r"\bEARNINGS\s*=\s*([A-Z0-9_-]+)", text)
        return (match.group(1).strip().upper() == "PASS") if match else True

    def _regime_weekly_debit_frac(row) -> float:
        width_val = fnum(row.get("width_live"))
        if not np.isfinite(width_val) or width_val <= 0:
            width_val = fnum(row.get("width"))
        net_val = fnum(row.get("live_net_bid_ask"))
        if not np.isfinite(net_val):
            net_val = fnum(row.get("live_net_mark"))
        if not np.isfinite(net_val):
            net_val = fnum(row.get("net"))
        if not (np.isfinite(width_val) and width_val > 0 and np.isfinite(net_val)):
            return math.nan
        return abs(net_val) / width_val

    def regime_weekly_score(row) -> float:
        txt = " ".join(
            str(row.get(col, "") or "")
            for col in [
                "flow_confirmation",
                "contract_flow_confirmation",
                "flow_confidence",
                "gex_regime",
            ]
        ).lower()
        score = 0.0
        score += fnum(row.get("confidence_score")) * 12.0 if np.isfinite(fnum(row.get("confidence_score"))) else 0.0
        score += fnum(row.get("edge_score")) * 8.0 if np.isfinite(fnum(row.get("edge_score"))) else 0.0
        score += fnum(row.get("edge_pct")) * 0.6 if np.isfinite(fnum(row.get("edge_pct"))) else 0.0
        rr_val = fnum(row.get("live_reward_risk"))
        if np.isfinite(rr_val):
            score += min(max(rr_val, 0.0), 5.0) * 10.0
        hist_val = fnum(row.get("hist_success_pct"))
        if np.isfinite(hist_val):
            score += hist_val * 0.15
        signals_val = fnum(row.get("signals"))
        if np.isfinite(signals_val):
            score += signals_val * 0.03
        if re.search(r"confirmed|strong|volatile|pinned", txt):
            score += 8.0
        if str(row.get("strategy", "") or "").strip() in {"Iron Condor", "Iron Butterfly"}:
            no_touch_val = fnum(row.get("credit_no_touch_pct"))
            if np.isfinite(no_touch_val):
                score += no_touch_val * 0.25
        return float(score)

    def regime_weekly_blockers(row) -> list[str]:
        if not enable_regime_weekly_book:
            return ["disabled"]
        blockers: list[str] = []
        strategy_local = str(row.get("strategy", "") or "").strip()
        verdict_local = str(row.get("verdict", "") or "").strip().upper()
        approval_regime_local = str(row.get("approval_regime", "") or "").strip().lower()
        market_regime_local = str(row.get("market_regime", "") or "").strip().lower()
        gex_regime_local = str(row.get("gex_regime", "") or "").strip().lower()
        contract_flow_local = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
        flow_conf_local = str(row.get("flow_confirmation", "") or "").strip().lower()
        flow_dir_local = str(row.get("flow_direction", "") or "").strip().lower()
        confidence_val = fnum(row.get("confidence_score"))
        edge_score_val = fnum(row.get("edge_score"))
        edge_pct_val = fnum(row.get("edge_pct"))
        rr_val = fnum(row.get("live_reward_risk"))
        dte_val = fnum(row.get("dte"))
        score_val = fnum(row.get("regime_weekly_score"))
        if not np.isfinite(score_val):
            score_val = regime_weekly_score(row)
        if not _regime_weekly_live_ok(row):
            blockers.append("live_not_ok")
        if not _regime_weekly_earnings_clean(row):
            blockers.append("earnings_risk")
        if contract_flow_local in regime_weekly_block_contract_flow_states:
            blockers.append(f"contract_flow_block:{contract_flow_local or 'unknown'}")
        if np.isfinite(regime_weekly_min_score) and score_val < regime_weekly_min_score:
            blockers.append(f"score_below:{score_val:.1f}<{regime_weekly_min_score:g}")
        if (
            np.isfinite(regime_weekly_min_confidence_score)
            and (not np.isfinite(confidence_val) or confidence_val < regime_weekly_min_confidence_score)
        ):
            blockers.append(f"confidence_below:{confidence_val if np.isfinite(confidence_val) else 'nan'}<{regime_weekly_min_confidence_score:g}")
        if (
            np.isfinite(regime_weekly_min_edge_score)
            and (not np.isfinite(edge_score_val) or edge_score_val < regime_weekly_min_edge_score)
        ):
            blockers.append(f"edge_score_below:{edge_score_val if np.isfinite(edge_score_val) else 'nan'}<{regime_weekly_min_edge_score:g}")
        if (
            np.isfinite(regime_weekly_min_edge_pct)
            and (not np.isfinite(edge_pct_val) or edge_pct_val < regime_weekly_min_edge_pct)
        ):
            blockers.append(f"edge_pct_below:{edge_pct_val if np.isfinite(edge_pct_val) else 'nan'}<{regime_weekly_min_edge_pct:g}")
        if (
            np.isfinite(regime_weekly_min_reward_risk)
            and (not np.isfinite(rr_val) or rr_val < regime_weekly_min_reward_risk)
        ):
            blockers.append(f"rr_below:{rr_val if np.isfinite(rr_val) else 'nan'}<{regime_weekly_min_reward_risk:g}")
        debit_frac = _regime_weekly_debit_frac(row)
        if (
            strategy_local in {"Bull Call Debit", "Bear Put Debit"}
            and np.isfinite(regime_weekly_max_debit_frac)
            and (not np.isfinite(debit_frac) or debit_frac > regime_weekly_max_debit_frac)
        ):
            blockers.append(f"debit_width_below:{debit_frac if np.isfinite(debit_frac) else 'nan'}>{regime_weekly_max_debit_frac:g}")
        if np.isfinite(regime_weekly_min_dte) and (not np.isfinite(dte_val) or dte_val < regime_weekly_min_dte):
            blockers.append(f"dte_below:{dte_val if np.isfinite(dte_val) else 'nan'}<{regime_weekly_min_dte:g}")
        if np.isfinite(regime_weekly_max_dte) and (not np.isfinite(dte_val) or dte_val > regime_weekly_max_dte):
            blockers.append(f"dte_above:{dte_val if np.isfinite(dte_val) else 'nan'}>{regime_weekly_max_dte:g}")
        if blockers:
            return blockers

        if strategy_local == "Bear Put Debit":
            if regime_weekly_bear_allowed_verdicts and verdict_local not in regime_weekly_bear_allowed_verdicts:
                blockers.append(f"bear_verdict:{verdict_local or 'UNKNOWN'}")
            if regime_weekly_bear_allowed_market_regimes and market_regime_local not in regime_weekly_bear_allowed_market_regimes:
                blockers.append(f"bear_market_regime:{market_regime_local or 'unknown'}")
            if any(approval_regime_local.startswith(prefix) for prefix in regime_weekly_bear_excluded_approval_prefixes):
                blockers.append(f"bear_approval_regime:{approval_regime_local or 'unknown'}")
            if (
                np.isfinite(regime_weekly_bear_min_edge_pct)
                and (not np.isfinite(edge_pct_val) or edge_pct_val < regime_weekly_bear_min_edge_pct)
            ):
                blockers.append(f"bear_edge_below:{edge_pct_val if np.isfinite(edge_pct_val) else 'nan'}<{regime_weekly_bear_min_edge_pct:g}")
            if (
                np.isfinite(regime_weekly_bear_max_edge_pct)
                and (not np.isfinite(edge_pct_val) or edge_pct_val > regime_weekly_bear_max_edge_pct)
            ):
                blockers.append(f"bear_edge_above:{edge_pct_val if np.isfinite(edge_pct_val) else 'nan'}>{regime_weekly_bear_max_edge_pct:g}")
            if flow_conf_local == "confirmed" and flow_dir_local == "bullish":
                blockers.append("bear_against_confirmed_bullish_flow")
            return blockers

        if strategy_local == "Bull Call Debit":
            if regime_weekly_bull_allowed_verdicts and verdict_local not in regime_weekly_bull_allowed_verdicts:
                blockers.append(f"bull_verdict:{verdict_local or 'UNKNOWN'}")
            regime_ok = any(
                approval_regime_local.startswith(prefix)
                for prefix in regime_weekly_bull_allowed_approval_prefixes
            )
            regime_ok = regime_ok or market_regime_local in regime_weekly_bull_allowed_market_regimes
            regime_ok = regime_ok or (regime_weekly_bull_allow_gex_volatile and gex_regime_local == "volatile")
            if not regime_ok:
                blockers.append(
                    f"bull_regime:{approval_regime_local or 'unknown'}/"
                    f"{market_regime_local or 'unknown'}/{gex_regime_local or 'unknown'}"
                )
            if (
                np.isfinite(regime_weekly_bull_min_edge_pct)
                and (not np.isfinite(edge_pct_val) or edge_pct_val < regime_weekly_bull_min_edge_pct)
            ):
                blockers.append(f"bull_edge_below:{edge_pct_val if np.isfinite(edge_pct_val) else 'nan'}<{regime_weekly_bull_min_edge_pct:g}")
            if flow_conf_local == "confirmed" and flow_dir_local == "bearish":
                blockers.append("bull_against_confirmed_bearish_flow")
            return blockers

        if regime_weekly_enable_income and strategy_local in {"Iron Condor", "Iron Butterfly"}:
            if regime_weekly_income_allowed_verdicts and verdict_local not in regime_weekly_income_allowed_verdicts:
                blockers.append(f"income_verdict:{verdict_local or 'UNKNOWN'}")
            if not (market_regime_local == "neutral" or gex_regime_local == "pinned"):
                blockers.append(f"income_regime:{market_regime_local or 'unknown'}/{gex_regime_local or 'unknown'}")
            no_touch_val = fnum(row.get("credit_no_touch_pct"))
            if (
                np.isfinite(regime_weekly_income_min_credit_no_touch_pct)
                and regime_weekly_income_min_credit_no_touch_pct > 0
                and (not np.isfinite(no_touch_val) or no_touch_val < regime_weekly_income_min_credit_no_touch_pct)
            ):
                blockers.append(
                    f"income_no_touch_below:"
                    f"{no_touch_val if np.isfinite(no_touch_val) else 'nan'}"
                    f"<{regime_weekly_income_min_credit_no_touch_pct:g}"
                )
            return blockers
        return [f"unsupported_strategy:{strategy_local or 'unknown'}"]

    def regime_weekly_base_checks(row) -> bool:
        return len(regime_weekly_blockers(row)) == 0

    def apply_regime_weekly_selection(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["regime_weekly_score"] = out.apply(regime_weekly_score, axis=1)
        out["regime_weekly_lane"] = False
        out["regime_weekly_candidate"] = False
        out["regime_weekly_blockers"] = out.apply(
            lambda rr: ";".join(regime_weekly_blockers(rr)),
            axis=1,
        )
        if enable_regime_weekly_book and regime_weekly_max_per_day > 0:
            regime_weekly_candidate_mask = out["regime_weekly_blockers"].astype(str).eq("")
            out.loc[regime_weekly_candidate_mask, "regime_weekly_candidate"] = True
            regime_weekly_candidates = out[regime_weekly_candidate_mask].copy()
            if not regime_weekly_candidates.empty:
                regime_weekly_candidates["_regime_weekly_conf"] = pd.to_numeric(
                    regime_weekly_candidates.get("confidence_score"),
                    errors="coerce",
                ).fillna(-1e9)
                regime_weekly_candidates["_regime_weekly_edge_score"] = pd.to_numeric(
                    regime_weekly_candidates.get("edge_score"),
                    errors="coerce",
                ).fillna(-1e9)
                regime_weekly_candidates["_regime_weekly_rr"] = pd.to_numeric(
                    regime_weekly_candidates.get("live_reward_risk"),
                    errors="coerce",
                ).fillna(-1e9)
                regime_weekly_candidates = regime_weekly_candidates.sort_values(
                    [
                        "regime_weekly_score",
                        "_regime_weekly_conf",
                        "_regime_weekly_edge_score",
                        "_regime_weekly_rr",
                        "conviction",
                    ],
                    ascending=[False, False, False, False, False],
                )
                keep_regime_weekly = []
                per_regime_weekly_ticker = defaultdict(int)
                for ridx, rrow in regime_weekly_candidates.iterrows():
                    rticker = str(rrow.get("ticker", "") or "").strip().upper()
                    if (
                        regime_weekly_max_per_ticker_day > 0
                        and per_regime_weekly_ticker[rticker] >= regime_weekly_max_per_ticker_day
                    ):
                        continue
                    if len(keep_regime_weekly) >= regime_weekly_max_per_day:
                        break
                    keep_regime_weekly.append(ridx)
                    per_regime_weekly_ticker[rticker] += 1
                if keep_regime_weekly:
                    out.loc[keep_regime_weekly, "regime_weekly_lane"] = True
                    if np.isfinite(regime_weekly_promoted_confidence_score):
                        promoted_score = pd.to_numeric(
                            out.loc[keep_regime_weekly, "confidence_score"],
                            errors="coerce",
                        ).fillna(0)
                        out.loc[keep_regime_weekly, "confidence_score"] = np.maximum(
                            promoted_score,
                            regime_weekly_promoted_confidence_score,
                        )
                    out.loc[keep_regime_weekly, "confidence_model"] = "High"
                    out.loc[keep_regime_weekly, "edge_sources"] = (
                        out.loc[keep_regime_weekly, "edge_sources"].fillna("").astype(str)
                        + ";regime_weekly_selector"
                    ).str.strip(";")
        return out

    mdf = apply_regime_weekly_selection(mdf)

    def stage1_context(row):
        opt = str(row.get("optimal_stage1", "")).strip()
        is_yes = opt in {"Yes-Prime", "Yes-Good"}
        verdict = str(row.get("verdict", "")).strip().upper()
        edge = fnum(row.get("edge_pct"))
        sig = fnum(row.get("signals"))
        conv = fnum(row.get("conviction"))
        promoted = False
        reason = ""
        if is_yes:
            reason = "stage1_yes"
        elif allow_stage1_watch_promotion:
            cond = (
                verdict == "PASS"
                and (not np.isfinite(stage1_promote_min_conv) or (np.isfinite(conv) and conv >= stage1_promote_min_conv))
                and (not np.isfinite(stage1_promote_min_edge) or (np.isfinite(edge) and edge >= stage1_promote_min_edge))
                and (
                    not np.isfinite(stage1_promote_min_signals)
                    or stage1_promote_min_signals <= 0
                    or (np.isfinite(sig) and sig >= stage1_promote_min_signals)
                )
            )
            promoted = bool(cond)
            reason = "stage1_promoted" if promoted else "stage1_watch_blocked"
            if (not promoted) and fire_breakout_exception(row):
                promoted = True
                reason = "stage1_breakout_exception"
            if (not promoted) and bull_call_evidence_lane(row):
                promoted = True
                reason = "bull_call_evidence_lane"
            if (not promoted) and bear_put_evidence_lane(row):
                promoted = True
                reason = "bear_put_evidence_lane"
            if (not promoted) and bull_call_pinned_continuation_lane(row):
                promoted = True
                reason = "bull_call_pinned_continuation_lane"
            if (not promoted) and bool(row.get("regime_weekly_lane", False)):
                promoted = True
                reason = "regime_weekly_lane"
        else:
            reason = "stage1_watch_blocked"
        return {
            "stage1_is_yes": bool(is_yes),
            "stage1_promoted": bool(promoted),
            "fire_breakout_exception": bool(fire_breakout_exception(row)),
            "bull_call_evidence_lane": bool(bull_call_evidence_lane(row)),
            "bear_put_evidence_lane": bool(bear_put_evidence_lane(row)),
            "bull_call_pinned_continuation_lane": bool(bull_call_pinned_continuation_lane(row)),
            "stage1_effective": bool(is_yes or promoted),
            "stage1_blocked": bool((not is_yes) and (not promoted)),
            "stage1_eval_reason": reason,
        }

    stage1_ctx_df = pd.DataFrame([stage1_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), stage1_ctx_df], axis=1)

    def invalidation_context(row):
        op = str(row.get("invalidation_rule_op", "")).strip()
        lvl = fnum(row.get("invalidation_rule_level"))
        live_px = fnum(row.get("invalidation_eval_price_live"))
        eval_source = "live"
        eval_px = live_px
        if use_asof_close_for_invalidation:
            ticker = str(row.get("ticker", "")).strip().upper()
            asof_px = fnum(spot_map.get(ticker))
            if np.isfinite(asof_px):
                eval_px = asof_px
                eval_source = "asof_close"
        breached = False
        if op in {"<", "<="} and np.isfinite(lvl) and np.isfinite(eval_px):
            breached = eval_px < lvl if op == "<" else eval_px <= lvl
        elif op in {">", ">="} and np.isfinite(lvl) and np.isfinite(eval_px):
            breached = eval_px > lvl if op == ">" else eval_px >= lvl
        elif pd.notna(row.get("invalidation_breached_live")):
            breached = bool(row.get("invalidation_breached_live"))
        return {
            "invalidation_eval_source": eval_source,
            "invalidation_eval_price_effective": eval_px,
            "invalidation_breached_effective": bool(breached),
        }

    invalid_ctx_df = pd.DataFrame([invalidation_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), invalid_ctx_df], axis=1)

    def spot_context(row):
        ticker = str(row.get("ticker", "")).strip().upper()
        asof_spot = fnum(spot_map.get(ticker))
        live_last = fnum(row.get("spot_live_last"))
        live_bid = fnum(row.get("spot_live_bid"))
        live_ask = fnum(row.get("spot_live_ask"))
        if np.isfinite(live_last):
            live_spot = float(live_last)
        elif np.isfinite(live_bid) and np.isfinite(live_ask):
            live_spot = (float(live_bid) + float(live_ask)) / 2.0
        elif np.isfinite(live_bid):
            live_spot = float(live_bid)
        elif np.isfinite(live_ask):
            live_spot = float(live_ask)
        else:
            live_spot = math.nan
        drift = math.nan
        if np.isfinite(asof_spot) and asof_spot > 0 and np.isfinite(live_spot):
            drift = abs(live_spot - asof_spot) / asof_spot
        return {
            "spot_asof_close": asof_spot,
            "spot_live_effective": live_spot,
            "spot_asof_live_drift_pct": drift,
        }

    spot_ctx_df = pd.DataFrame([spot_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), spot_ctx_df], axis=1)

    def external_ev_context(row):
        asof_ev = fnum(row.get("external_ev_ml"))
        source = str(row.get("external_ev_ml_source", "") or "").strip()
        debit_live_ev = debit_partial_ev_for_row(row, use_live=True)
        debit_asof_ev = debit_partial_ev_for_row(row, use_live=False)
        debit_target_ev = math.nan
        strategy_local = str(row.get("strategy", "")).strip()
        if strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
            direction = "bull" if strategy_local == "Bull Call Debit" else "bear"
            spot = fnum(row.get("spot_live_effective"))
            if not np.isfinite(spot) or spot <= 0:
                spot = fnum(row.get("spot_asof_close"))
            target_net = fnum(row.get("gate_target"))
            if not np.isfinite(target_net) or target_net <= 0:
                target_net = fnum(row.get("net"))
            debit_target_ev = partial_ev_ml_debit(
                spot,
                row.get("iv30d"),
                row.get("dte"),
                row.get("long_strike"),
                row.get("short_strike"),
                target_net,
                direction,
            )
        if np.isfinite(debit_live_ev):
            debit_effective = debit_live_ev
            debit_effective_source = "partial_payoff_live"
        elif np.isfinite(debit_asof_ev):
            debit_effective = debit_asof_ev
            debit_effective_source = "partial_payoff_asof"
        else:
            debit_effective = math.nan
            debit_effective_source = "unavailable"
        live_ev = math.nan
        if str(row.get("source", "") or "").startswith("external_scanner:"):
            live_ev = debit_live_ev
        if np.isfinite(live_ev):
            effective = live_ev
            effective_source = "partial_payoff_live"
        elif np.isfinite(asof_ev):
            effective = asof_ev
            effective_source = source or "partial_payoff_asof"
        else:
            effective = math.nan
            effective_source = source or "unavailable"
        return {
            "debit_partial_ev_ml_live": debit_live_ev,
            "debit_partial_ev_ml_asof": debit_asof_ev,
            "debit_partial_ev_ml_target_live": debit_target_ev,
            "debit_partial_ev_ml_effective": debit_effective,
            "debit_partial_ev_ml_effective_source": debit_effective_source,
            "external_ev_ml_live": live_ev,
            "external_ev_ml_effective": effective,
            "external_ev_ml_effective_source": effective_source,
        }

    external_ev_df = pd.DataFrame([external_ev_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), external_ev_df], axis=1)

    def ev_score(row):
        hs = fnum(row.get("hist_success_pct"))
        p = hs / 100.0 if np.isfinite(hs) else math.nan
        live_max_profit = fnum(row.get("live_max_profit"))
        live_max_loss = fnum(row.get("live_max_loss"))
        if not (np.isfinite(live_max_profit) and np.isfinite(live_max_loss)):
            gate_target = fnum(row.get("gate_target"))
            width = fnum(row.get("width"))
            net_type = str(row.get("net_type", "")).strip().lower()
            if np.isfinite(gate_target) and np.isfinite(width):
                tgt_max_profit, tgt_max_loss = calc_target_max(net_type, width, gate_target)
                live_max_profit = tgt_max_profit
                live_max_loss = tgt_max_loss
        ev_cash = math.nan
        if np.isfinite(p) and np.isfinite(live_max_profit) and np.isfinite(live_max_loss):
            ev_cash = (p * live_max_profit) - ((1.0 - p) * live_max_loss)
        ev_risk = ev_cash / live_max_loss if (np.isfinite(ev_cash) and np.isfinite(live_max_loss) and live_max_loss > 0) else math.nan
        conv = fnum(row.get("conviction"))
        edge = fnum(row.get("edge_pct"))
        gate_near_miss = bool(row.get("gate_near_miss")) if pd.notna(row.get("gate_near_miss")) else False

        score = 0.0
        if np.isfinite(ev_risk):
            score += 100.0 * ev_risk
        elif np.isfinite(ev_cash):
            score += ev_cash / 10.0
        if np.isfinite(edge):
            score += 0.5 * edge
        if np.isfinite(conv):
            score += 0.1 * (conv - 50.0)
        if gate_near_miss:
            score -= 0.5
        return score

    mdf["_ev_sort"] = mdf.apply(ev_score, axis=1)

    def strike_distance(base_row, cand_row):
        cols = [
            "long_strike",
            "short_strike",
            "long_put_strike",
            "short_put_strike",
            "short_call_strike",
            "long_call_strike",
        ]
        dist = 0.0
        used = False
        for c in cols:
            a = fnum(base_row.get(c))
            b = fnum(cand_row.get(c))
            if np.isfinite(a) and np.isfinite(b):
                dist += abs(a - b)
                used = True
        return dist if used else 1e9

    if enable_restrike_optimizer and not mdf.empty:
        stage1_rank_map = {"Yes-Prime": 0, "Yes-Good": 1, "Watch Only": 2}
        restrike_from = pd.Series([pd.NA] * len(mdf), index=mdf.index, dtype="Int64")
        restrike_reason = pd.Series([""] * len(mdf), index=mdf.index, dtype="string")
        selected_idx = []
        family_cols = ["ticker", "strategy", "expiry", "track"]

        for _, fam in mdf.groupby(family_cols, dropna=False):
            fam_local = fam.copy()
            fam_local["_stage1_rank"] = (
                fam_local["optimal_stage1"].map(stage1_rank_map).fillna(3).astype(int)
            )
            fam_local = fam_local.sort_values(
                ["_stage1_rank", "conviction", "_ev_sort"],
                ascending=[True, False, False],
            )
            base_idx = fam_local.index[0]
            base_row = mdf.loc[base_idx]
            pick_idx = base_idx

            base_live_status = str(base_row.get("live_status", "")).strip()
            base_gate_effective = (
                bool(base_row.get("gate_pass_effective"))
                if pd.notna(base_row.get("gate_pass_effective"))
                else False
            )
            base_struct_ok = (
                bool(base_row.get("entry_structure_ok_live"))
                if pd.notna(base_row.get("entry_structure_ok_live"))
                else True
            )

            def _short_delta_clean_for_restrike(rr) -> bool:
                if not require_live_shield_short_delta:
                    return True
                strategy_name = str(rr.get("strategy", "") or "").strip()
                if strategy_name not in {"Iron Condor", "Iron Butterfly", "Bear Call Credit", "Bull Put Credit"}:
                    return True
                cap = max_abs_short_delta_shield
                if not np.isfinite(cap) or cap <= 0:
                    cap = 0.32
                if strategy_name in {"Iron Condor", "Iron Butterfly"}:
                    put_delta = abs(_safe_delta(rr.get("short_put_delta_live")))
                    call_delta = abs(_safe_delta(rr.get("short_call_delta_live")))
                    if not np.isfinite(put_delta) or not np.isfinite(call_delta):
                        return False
                    return put_delta <= cap and call_delta <= cap
                short_delta = abs(_safe_delta(rr.get("short_delta_live")))
                return np.isfinite(short_delta) and short_delta <= cap

            base_delta_clean = _short_delta_clean_for_restrike(base_row)
            needs_restrike = (
                base_live_status == "fails_live_entry_gate"
                or (not base_gate_effective)
                or (not base_struct_ok)
                or (not base_delta_clean)
            )
            if needs_restrike and len(fam_local) > 1:
                exec_pool = fam_local[
                    fam_local["live_status"].astype(str).eq("ok_live")
                    & fam_local["gate_pass_effective"].fillna(False).astype(bool)
                    & fam_local["entry_structure_ok_live"].fillna(True).astype(bool)
                ].copy()
                if not exec_pool.empty:
                    exec_pool = exec_pool[exec_pool.apply(_short_delta_clean_for_restrike, axis=1)].copy()
                if not exec_pool.empty:
                    exec_pool["_dist"] = exec_pool.apply(
                        lambda rr: strike_distance(base_row, rr), axis=1
                    )
                    exec_pool = exec_pool.sort_values(
                        ["_dist", "_ev_sort", "conviction"],
                        ascending=[True, False, False],
                    )
                    pick_idx = exec_pool.index[0]
                    restrike_from.loc[pick_idx] = int(base_idx)
                    restrike_reason.loc[pick_idx] = (
                        "family_restrike_from_delta_fail"
                        if not base_delta_clean
                        else "family_restrike_from_gate_fail"
                    )

            selected_idx.append(int(pick_idx))

        selected_unique = sorted(set(selected_idx))
        mdf = mdf.loc[selected_unique].copy()
        mdf["restrike_replaced_from"] = restrike_from.loc[selected_unique].values
        mdf["restrike_reason"] = restrike_reason.loc[selected_unique].fillna("").astype(str).values
        mdf = mdf.reset_index(drop=True)

    # Re-run the tactical weekly selector after restrike pruning so the final
    # approval path evaluates the executable spread variant, not a discarded
    # pre-restrike row.
    mdf = apply_regime_weekly_selection(mdf)
    regime_weekly_mask = mdf.get("regime_weekly_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
    if regime_weekly_mask.any():
        if "stage1_promoted" in mdf.columns:
            mdf.loc[regime_weekly_mask, "stage1_promoted"] = True
        if "stage1_effective" in mdf.columns:
            mdf.loc[regime_weekly_mask, "stage1_effective"] = True
        if "stage1_blocked" in mdf.columns:
            mdf.loc[regime_weekly_mask, "stage1_blocked"] = False
        if "stage1_eval_reason" in mdf.columns:
            mdf.loc[regime_weekly_mask, "stage1_eval_reason"] = "regime_weekly_lane"

    def bool_or_none(value):
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None

    def approval_blockers(row):
        blockers = []
        live_status = str(row.get("live_status", "")).strip()
        strategy_local = str(row.get("strategy", "")).strip()
        track = normalize_track(str(row.get("track", "")), strategy_local)
        live_bad_status = live_status in {
            "chain_error",
            "chain_not_success",
            "bad_occ_symbol",
            "missing_leg_in_live_chain",
            "missing_underlying_quote",
            "invalid_entry_structure",
            "missing_live_quote",
        }
        if live_bad_status:
            blockers.append(f"live_status:{live_status or 'unknown'}")
            return blockers

        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        if not ok_live:
            blockers.append("live_entry_gate_fail")
        shield_live_quality_override = bool(
            shield_live_valid_overrides_quality and track == "SHIELD" and ok_live
        )

        # [T9] Live R/R quality check — use debit/credit-specific floors.
        # A global 1.50x floor is appropriate for debit spreads, but it
        # mechanically rejects normal premium-selling spreads where collecting
        # 30-40% of width implies reward/risk of roughly 0.43-0.67.
        _live_rr = fnum(row.get("live_reward_risk"))
        _net_type_for_rr = str(row.get("net_type", "")).strip().lower()
        _rr_floor = (
            min_credit_reward_risk if _net_type_for_rr == "credit"
            else min_debit_reward_risk if _net_type_for_rr == "debit"
            else min_live_reward_risk
        )
        if np.isfinite(_live_rr) and np.isfinite(_rr_floor) and _live_rr < _rr_floor:
            blockers.append(f"live_rr_weak:{_live_rr:.2f}<{_rr_floor:.2f}")

        if str(row.get("source", "") or "").startswith("external_scanner:") and strategy_local in {
            "Bull Call Debit",
            "Bear Put Debit",
        }:
            external_ev_effective = fnum(row.get("external_ev_ml_effective"))
            if not np.isfinite(external_ev_effective):
                blockers.append("pilot_ev_ml_unavailable")
            elif np.isfinite(pilot_min_ev_ml) and external_ev_effective < pilot_min_ev_ml:
                blockers.append(f"pilot_ev_ml_below:{external_ev_effective:.3f}<{pilot_min_ev_ml:g}")

        if require_likelihood_pass:
            verdict = str(row.get("verdict", "")).strip().upper()
            edge = fnum(row.get("edge_pct"))
            sig = fnum(row.get("signals"))
            if verdict != "PASS":
                blockers.append(f"likelihood_verdict:{verdict or 'UNKNOWN'}")
            # Per-strategy edge threshold: bears and SHIELD get lower bar
            _is_bear = strategy_local in {"Bear Put Debit", "Bear Call Credit"}
            _eff_edge_min = (
                min_edge_pct_bear if _is_bear
                else min_edge_pct_shield if track == "SHIELD"
                else min_edge_pct
            )
            if np.isfinite(_eff_edge_min) and (not np.isfinite(edge) or edge < _eff_edge_min):
                blockers.append(f"edge_below:{edge if np.isfinite(edge) else 'nan'}<{_eff_edge_min}")
            if np.isfinite(min_signals) and min_signals > 0 and (not np.isfinite(sig) or sig < min_signals):
                blockers.append(f"signals_below:{sig if np.isfinite(sig) else 'nan'}<{min_signals}")

        strength = str(row.get("likelihood_strength", "")).strip()
        strength_rank = rank_likelihood_strength(strength)
        # Per-strategy strength threshold: bears and SHIELD accept Weak
        _is_bear = strategy_local in {"Bear Put Debit", "Bear Call Credit"}
        _eff_strength = (
            min_likelihood_strength_bear if _is_bear
            else min_likelihood_strength_shield if track == "SHIELD"
            else min_likelihood_strength
        )
        _eff_strength_rank = rank_likelihood_strength(_eff_strength)
        if (
            _eff_strength
            and _eff_strength_rank >= 0
            and strength_rank >= 0
            and strength_rank < _eff_strength_rank
        ):
            blockers.append(f"likelihood_strength_below:{strength}<{_eff_strength}")
        if (
            _eff_strength
            and _eff_strength_rank >= 0
            and strength_rank < 0
        ):
            blockers.append(f"likelihood_strength_unranked:{strength or 'N/A'}")
        if str(strength).strip().upper() in disallow_likelihood_strengths:
            blockers.append(f"likelihood_strength_blocked:{strength}")

        invalidated_effective = (
            bool(row.get("invalidation_breached_effective"))
            if pd.notna(row.get("invalidation_breached_effective"))
            else False
        )
        if (require_invalidation_clear or block_invalidation_warning) and invalidated_effective:
            blockers.append("invalidation_warning")

        stage1_effective = bool(row.get("stage1_effective")) if pd.notna(row.get("stage1_effective")) else False
        if not stage1_effective and not shield_live_quality_override:
            stage1_diag_raw = str(row.get("stage1_diagnostics", "") or row.get("stage1_not_actionable_reason", "")).strip()
            stage1_diag_tokens = [x.strip() for x in stage1_diag_raw.split(";") if x.strip()]
            if stage1_diag_tokens:
                blockers.extend(stage1_diag_tokens)
            else:
                blockers.append("stage1_not_actionable")
        elif bool(row.get("bull_call_evidence_lane")) and not bool(row.get("stage1_is_yes")):
            blockers.append("bull_call_evidence_lane_tactical")
        elif bool(row.get("bear_put_evidence_lane")) and not bool(row.get("stage1_is_yes")):
            blockers.append("bear_put_evidence_lane_tactical")
        elif bool(row.get("bull_call_pinned_continuation_lane")) and not bool(row.get("stage1_is_yes")):
            blockers.append("bull_call_pinned_continuation_lane_tactical")

        if require_spot_alignment:
            spot_asof = fnum(row.get("spot_asof_close"))
            spot_live = fnum(row.get("spot_live_effective"))
            if not np.isfinite(spot_live):
                if spot_alignment_require_live:
                    blockers.append("spot_live_missing")
            elif np.isfinite(spot_asof) and spot_asof > 0:
                drift = fnum(row.get("spot_asof_live_drift_pct"))
                if not np.isfinite(drift):
                    blockers.append("spot_drift_unknown")
                elif drift > max_spot_asof_drift_pct:
                    blockers.append(f"spot_drift:{drift:.2%}>{max_spot_asof_drift_pct:.2%}")

        long_strike = fnum(row.get("long_strike"))
        spot_ref = fnum(row.get("spot_live_effective"))
        if not np.isfinite(spot_ref) or spot_ref <= 0:
            spot_ref = fnum(row.get("spot_asof_close"))
        if np.isfinite(spot_ref) and spot_ref > 0 and np.isfinite(long_strike):
            if strategy_local == "Bull Call Debit" and np.isfinite(max_bull_call_long_otm_pct):
                long_otm = (long_strike / spot_ref) - 1.0
                if long_otm > max_bull_call_long_otm_pct:
                    blockers.append(
                        f"bull_call_otm_too_far:{long_otm:.1%}>{max_bull_call_long_otm_pct:.1%}"
                    )
            elif strategy_local == "Bear Put Debit" and np.isfinite(max_bear_put_long_otm_pct):
                long_otm = 1.0 - (long_strike / spot_ref)
                if long_otm > max_bear_put_long_otm_pct:
                    blockers.append(
                        f"bear_put_otm_too_far:{long_otm:.1%}>{max_bear_put_long_otm_pct:.1%}"
                    )

        if strategy_local == "Bull Call Debit":
            bull_call_dte = fnum(row.get("dte"))
            bull_call_edge = fnum(row.get("edge_pct"))
            bull_call_rr = fnum(row.get("live_reward_risk"))
            bull_call_contract_flow = str(row.get("contract_flow_confirmation", "")).strip().lower()
            bull_call_regime_conf = str(row.get("market_regime_confidence", "")).strip()
            if (
                bull_call_short_dte_high_edge_block
                and np.isfinite(bull_call_dte)
                and np.isfinite(bull_call_edge)
                and bull_call_dte < bull_call_short_dte_high_edge_max_dte
                and bull_call_edge > bull_call_short_dte_high_edge_min_edge
            ):
                blockers.append(
                    "bull_call_short_dte_high_edge:"
                    f"dte={bull_call_dte:g}<{bull_call_short_dte_high_edge_max_dte:g},"
                    f"edge={bull_call_edge:.2f}>{bull_call_short_dte_high_edge_min_edge:.2f}"
                )
            if (
                np.isfinite(bull_call_approval_max_dte)
                and (not np.isfinite(bull_call_dte) or bull_call_dte > bull_call_approval_max_dte)
            ):
                if (
                    np.isfinite(bull_call_dte)
                    and bull_call_dte <= bull_call_approval_max_dte + bull_call_approval_dte_grace_days
                ):
                    blockers.append(
                        f"bull_call_dte_near_limit:{bull_call_dte:g}>{bull_call_approval_max_dte:g}"
                    )
                else:
                    blockers.append(
                        f"bull_call_dte_too_long:{bull_call_dte if np.isfinite(bull_call_dte) else 'nan'}>{bull_call_approval_max_dte}"
                    )
            if (
                np.isfinite(bull_call_approval_min_reward_risk)
                and (not np.isfinite(bull_call_rr) or bull_call_rr < bull_call_approval_min_reward_risk)
            ):
                blockers.append(
                    f"bull_call_rr_weak:{bull_call_rr if np.isfinite(bull_call_rr) else 'nan'}<{bull_call_approval_min_reward_risk}"
                )
            if bull_call_approval_require_contract_confirmed and bull_call_contract_flow != "confirmed":
                blockers.append(f"bull_call_contract_flow_not_confirmed:{bull_call_contract_flow or 'missing'}")
            if bull_call_market_regime_enabled:
                if bull_call_regime_conf == "Low" and bull_call_low_regime_blocks:
                    blockers.append("market_regime_block:Low")
                elif bull_call_regime_conf == "Medium" and bull_call_medium_regime_tactical:
                    blockers.append("market_regime_caution:Medium")
            bull_call_regime_reason = str(row.get("market_regime_reason", "")).strip().lower()
            bull_call_vix = fnum(row.get("vix_level"))
            if not np.isfinite(bull_call_vix):
                bull_call_vix = fnum(row.get("market_vix_level"))
            if not np.isfinite(bull_call_vix):
                bull_call_vix = _vix_level
            bull_call_gex_context = str(row.get("gex_wall_context", "")).strip()
            bull_call_trend_down = "ticker trend down" in bull_call_regime_reason
            bull_call_trend_up = "ticker trend up" in bull_call_regime_reason
            if (
                bull_call_block_downtrend_without_high_vix
                and bull_call_trend_down
                and np.isfinite(bull_call_vix)
                and bull_call_vix < bull_call_trend_vix_floor
            ):
                blockers.append(
                    f"bull_call_downtrend_without_high_vix:{bull_call_vix:.1f}<{bull_call_trend_vix_floor:.1f}"
                )
            if (
                bull_call_missing_gex_requires_uptrend
                and not bull_call_gex_context
                and not bull_call_trend_up
                and np.isfinite(bull_call_vix)
                and bull_call_vix < bull_call_trend_vix_floor
            ):
                blockers.append(
                    f"bull_call_missing_gex_without_uptrend:{bull_call_vix:.1f}<{bull_call_trend_vix_floor:.1f}"
                )

        confidence_tier = str(row.get("confidence_tier", "")).strip().upper()
        if (
            confidence_tier
            and confidence_tier in disallow_confidence_tiers
            and not shield_live_quality_override
        ):
            blockers.append(f"confidence_tier_blocked:{confidence_tier}")

        flow_dir = str(row.get("flow_direction", "")).strip().lower()
        flow_conf = str(row.get("flow_confirmation", "")).strip().lower()
        flow_premium_bias = fnum(row.get("flow_premium_bias"))
        contract_flow = str(row.get("contract_flow_confirmation", "")).strip().lower()
        if flow_conf:
            def directional_flow_ok(expected_direction: str) -> bool:
                # If ticker-level flow is strongly confirmed against the trade,
                # respect that veto.  Otherwise allow selected-contract flow to
                # confirm the actual leg being traded; aggregate ticker flow is
                # often mixed around large hedges and multi-leg prints.
                if flow_conf == "confirmed":
                    return flow_dir == expected_direction
                return contract_flow == "confirmed"

            if strategy_local == "Bull Call Debit":
                if not directional_flow_ok("bullish"):
                    blockers.append(
                        f"flow_not_confirmed:{flow_dir or 'unknown'}/{flow_conf}"
                    )
            elif strategy_local == "Bear Put Debit":
                if not directional_flow_ok("bearish"):
                    blockers.append(
                        f"flow_not_confirmed:{flow_dir or 'unknown'}/{flow_conf}"
                    )
            elif strategy_local == "Bull Put Credit":
                if flow_dir == "bearish" and flow_conf == "confirmed":
                    blockers.append(
                        f"flow_contra_bull_put:{flow_premium_bias:+.2f}"
                    )
            elif strategy_local == "Bear Call Credit":
                if flow_dir == "bullish" and flow_conf == "confirmed":
                    blockers.append(
                        f"flow_contra_bear_call:{flow_premium_bias:+.2f}"
                    )
            elif strategy_local in {"Iron Condor", "Iron Butterfly"}:
                if flow_dir in {"bullish", "bearish"} and flow_conf == "confirmed":
                    blockers.append(
                        f"flow_too_directional_for_ic:{flow_dir}"
                    )

        if contract_flow:
            if strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
                if contract_flow in {"contra", "weak_or_ambiguous", "unknown"}:
                    blockers.append(f"contract_flow_{contract_flow}")
            elif strategy_local in {"Bull Put Credit", "Bear Call Credit"}:
                if contract_flow == "contra":
                    blockers.append(f"contract_flow_{contract_flow}")
            elif strategy_local in {"Iron Condor", "Iron Butterfly"}:
                if contract_flow == "directional":
                    blockers.append("contract_flow_directional")

        if track == "SHIELD":
            if require_shield_sigma_pass:
                sigma_stage1 = bool_or_none(row.get("sigma_pass_stage1"))
                if sigma_stage1 is None:
                    if shield_sigma_require_data:
                        blockers.append("shield_sigma_unknown")
                elif sigma_stage1 is False:
                    blockers.append("shield_sigma_fail")
            if require_shield_core:
                core_stage1 = bool_or_none(row.get("core_ok_stage1"))
                if core_stage1 is not True:
                    blockers.append("shield_core_fail")
            if require_live_shield_short_delta:
                # Compute per-trade delta cap: dynamic (IVR/DTE/VIX/GEX-aware) or static
                if shield_delta_dynamic:
                    _ivr = fnum(row.get("iv_rank"))
                    if not np.isfinite(_ivr):
                        _ivr = 30.0  # conservative fallback
                    _dte_val = fnum(row.get("dte"))
                    if not np.isfinite(_dte_val):
                        _dte_val = 45
                    _gex_r = str(row.get("gex_regime", "")).strip().lower()
                    _strat_type = "iron_condor" if strategy_local in {"Iron Condor", "Iron Butterfly"} else "credit_spread"
                    _delta_cap = dynamic_shield_delta_cap(
                        ivr=_ivr, dte=int(_dte_val), gex_regime=_gex_r,
                        vix=_vix_level, strategy=_strat_type,
                    )
                    if _delta_cap <= 0.0:
                        # IVR too low for credit spreads in dynamic mode
                        blockers.append(f"shield_delta_insufficient_ivr:{_ivr:.0f}")
                        _delta_cap = None  # skip further delta checks
                else:
                    _delta_cap = max_abs_short_delta_shield

                if _delta_cap is not None:
                    if strategy_local in {"Bull Put Credit", "Bear Call Credit"}:
                        short_delta = _safe_delta(row.get("short_delta_live"))
                        if not np.isfinite(short_delta):
                            blockers.append("shield_delta_missing")
                        elif abs(short_delta) > _delta_cap:
                            blockers.append(f"shield_delta_fail:{short_delta:+.2f}>{_delta_cap:.2f}")
                    elif strategy_local in {"Iron Condor", "Iron Butterfly"}:
                        put_delta = _safe_delta(row.get("short_put_delta_live"))
                        call_delta = _safe_delta(row.get("short_call_delta_live"))
                        if not np.isfinite(put_delta) or not np.isfinite(call_delta):
                            blockers.append("shield_delta_missing")
                        elif abs(put_delta) > _delta_cap or abs(call_delta) > _delta_cap:
                            blockers.append(f"shield_delta_fail:put={put_delta:+.2f},call={call_delta:+.2f}>{_delta_cap:.2f}")

            # IC/IB profitability is terminal (expiry-zone), not path-dependent,
            # so the no-touch metric is irrelevant for them.
            _is_ic = strategy_local in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}
            _skip_no_touch = _is_ic and ic_exempt_from_no_touch
            if np.isfinite(min_credit_no_touch_pct) and min_credit_no_touch_pct > 0 and not _skip_no_touch:
                no_touch = fnum(row.get("credit_no_touch_pct"))
                if np.isfinite(no_touch):
                    if no_touch < min_credit_no_touch_pct:
                        blockers.append(
                            f"credit_no_touch_below:{no_touch:.2f}<{min_credit_no_touch_pct:.2f}"
                        )
                elif credit_no_touch_require_data:
                    blockers.append("credit_no_touch_unknown")

        # FIRE long-leg delta gate: reject lottery tickets  [B1 fix: moved out of SHIELD block]
        if track == "FIRE" and require_fire_long_delta:
            if strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
                long_delta = _safe_delta(row.get("long_delta_live"))
                if not np.isfinite(long_delta):
                    proxy_ok, proxy_otm = fire_long_delta_proxy_ok(
                        strategy_local,
                        row.get("long_strike"),
                        row.get("spot_live_effective")
                        if np.isfinite(fnum(row.get("spot_live_effective")))
                        else row.get("spot_asof_close"),
                        fire_delta_proxy_max_long_otm_pct,
                    )
                    if allow_fire_delta_moneyness_proxy and proxy_ok:
                        blockers.append(f"long_delta_proxy_ok:otm={proxy_otm:.2%}")
                    else:
                        blockers.append("fire_delta_missing")
                elif abs(long_delta) < min_abs_long_delta_fire:
                    blockers.append(f"fire_delta_low:{long_delta:+.2f}")

        # GEX regime gate  [B1 fix: moved out of SHIELD block — applies to both tracks]
        if require_gex_regime:
            gex_regime = str(row.get("gex_regime", "")).strip().lower()
            gex_wall_ctx = str(row.get("gex_wall_context", "")).strip()
            if not gex_regime:
                if bool(row.get("bull_call_evidence_lane")) and bull_call_evidence_allow_gex_missing:
                    blockers.append("gex_missing_evidence_lane")
                else:
                    blockers.append("gex_missing")
            else:
                gex_source = str(row.get("gex_source", "")).strip().lower()
                if gex_source == SCHWAB_STALE_GEX_SOURCE:
                    blockers.append("gex_source_stale")
                if gex_source == SCHWAB_LEGACY_GEX_SOURCE:
                    if gex_fallback_tactical_only:
                        blockers.append("gex_source_fallback_tactical_only")
                    if gex_fallback_requires_clean_non_gex:
                        fallback_reasons = []
                        verdict_fb = str(row.get("verdict", "")).strip().upper()
                        edge_fb = fnum(row.get("edge_pct"))
                        signals_fb = fnum(row.get("signals"))
                        _is_bear_fb = strategy_local in {"Bear Put Debit", "Bear Call Credit"}
                        edge_min_fb = (
                            min_edge_pct_bear if _is_bear_fb
                            else min_edge_pct_shield if track == "SHIELD"
                            else min_edge_pct
                        )
                        flow_confirm_fb = str(row.get("flow_confirmation", "")).strip().lower()
                        contract_confirm_fb = str(row.get("contract_flow_confirmation", "")).strip().lower()
                        if not stage1_effective:
                            fallback_reasons.append("stage1")
                        if not ok_live:
                            fallback_reasons.append("live")
                        if verdict_fb != "PASS":
                            fallback_reasons.append("verdict")
                        if np.isfinite(edge_min_fb) and (not np.isfinite(edge_fb) or edge_fb < edge_min_fb):
                            fallback_reasons.append("edge")
                        if (
                            np.isfinite(min_signals)
                            and min_signals > 0
                            and (not np.isfinite(signals_fb) or signals_fb < min_signals)
                        ):
                            fallback_reasons.append("signals")
                        if flow_confirm_fb != "confirmed":
                            fallback_reasons.append("flow")
                        if contract_confirm_fb != "confirmed":
                            fallback_reasons.append("contract_flow")
                        if fallback_reasons:
                            blockers.append(
                                "gex_source_fallback_uncertain:" + ",".join(fallback_reasons)
                            )
                if track == "SHIELD" and gex_regime == "volatile":
                    blockers.append("shield_gex_volatile")
                elif track == "FIRE" and gex_regime == "pinned":
                    net_gex_val = fnum(row.get("net_gex"))
                    # Only block FIRE if GEX is strongly pinned (not marginal)
                    if np.isfinite(net_gex_val) and abs(net_gex_val) >= min_fire_pinned_gex_abs:
                        live_net_val = fnum(row.get("live_net_bid_ask"))
                        long_strike_val = fnum(row.get("long_strike"))
                        resistance_val = fnum(row.get("gex_resistance"))
                        support_val = fnum(row.get("gex_support"))
                        wall_supportive = False
                        if (
                            strategy_local == "Bull Call Debit"
                            and np.isfinite(live_net_val)
                            and np.isfinite(long_strike_val)
                            and np.isfinite(resistance_val)
                        ):
                            wall_supportive = (long_strike_val + live_net_val) <= resistance_val
                        elif (
                            strategy_local == "Bear Put Debit"
                            and np.isfinite(live_net_val)
                            and np.isfinite(long_strike_val)
                            and np.isfinite(support_val)
                        ):
                            wall_supportive = (long_strike_val - live_net_val) >= support_val
                        if not wall_supportive and not bool(row.get("fire_breakout_exception")):
                            blockers.append("fire_gex_pinned")
                # IC-specific: block in volatile regime (amplified moves break IC range)
                # [T8] was: require pinned — too strict, ICs work in neutral too
                if strategy_local in {"Iron Condor", "Iron Butterfly"} and gex_regime == "volatile":
                    blockers.append("ic_gex_volatile")

            # GEX context quality overlays. These are not always hard vetoes,
            # but they should prevent a trade from being treated as Core:
            # - volatile breakout buckets have had low hit-rate in audit
            # - missing wall context makes pinned GEX less actionable
            # - no clear GEX context should stay reduced-size at most
            if track == "FIRE" and strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
                if fire_volatile_breakout_tactical_only and gex_wall_ctx == "volatile_breakout_possible":
                    blockers.append("gex_volatile_breakout_tactical_only")
                if fire_pinned_no_wall_tactical_only and gex_wall_ctx in {"pinned_no_call_wall", "pinned_no_put_wall"}:
                    blockers.append(f"gex_wall_missing:{gex_wall_ctx}")
                if fire_missing_gex_context_tactical_only and gex_regime and not gex_wall_ctx:
                    blockers.append("gex_context_missing_tactical_only")

        return blockers

    mdf["approval_blockers"] = mdf.apply(
        lambda row: ";".join(approval_blockers(row)),
        axis=1,
    )
    blockers_split = mdf.apply(split_approval_blockers, axis=1)
    mdf["hard_blockers"] = blockers_split.apply(lambda x: ";".join(x[0]))
    mdf["quality_blockers"] = blockers_split.apply(lambda x: ";".join(x[1]))

    def _has_quality_stage1_context(tokens) -> bool:
        for token_raw in tokens:
            token = str(token_raw).strip()
            if not token:
                continue
            if (
                token.startswith("stage1_conviction_below_yes_good")
                or token == "stage1_flow_weak_or_ambiguous"
                or token.startswith("stage1_contract_flow_weak_or_ambiguous")
                or token.startswith("stage1_contract_flow_unknown")
                or token.startswith("stage1_high_iv_debit_watch_only")
            ):
                return True
        return False

    def _earnings_state(row) -> str:
        """Parse the Stage-1 earnings gate without treating Earnings=PASS as risk."""
        label = str(row.get("earnings_label_stage1", "") or "").strip().upper()
        if label in {"PASS", "ER-RISK", "CROSSED", "UNKNOWN"} or label.startswith("WITHIN"):
            return label
        text = str(row.get("notes_stage1", "") or row.get("notes", "") or "").upper()
        match = re.search(r"\bEARNINGS\s*=\s*([A-Z0-9_-]+)", text)
        if match:
            return match.group(1).strip().upper()
        if "ER-RISK" in text:
            return "ER-RISK"
        if "CROSSED" in text:
            return "CROSSED"
        match = re.search(r"\bWITHIN\d+\b", text)
        if match:
            return match.group(0)
        return label

    def _earnings_clean(row) -> bool:
        return _earnings_state(row) == "PASS"

    def _row_live_price_ok(row) -> bool:
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        return ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)

    def _quant_edge_allows_blockers(row) -> bool:
        text = ";".join(
            str(row.get(col, "") or "").lower()
            for col in ["hard_blockers", "approval_blockers"]
        )
        return not any(token and token in text for token in quant_edge_blocker_substrings)

    def _short_dte_edge_allows_blockers(row) -> bool:
        text = ";".join(
            str(row.get(col, "") or "").lower()
            for col in ["hard_blockers", "approval_blockers", "quality_blockers", "missing_data_flags"]
        )
        return not any(token and token in text for token in short_dte_edge_blocker_substrings)

    def _high_signal_edge_allows_blockers(row) -> bool:
        text = ";".join(
            str(row.get(col, "") or "").lower()
            for col in [
                "hard_blockers",
                "approval_blockers",
                "quality_blockers",
                "missing_data_flags",
                "profit_safety_approval_blockers",
            ]
        )
        return not any(token and token in text for token in high_signal_edge_blocker_substrings)

    def _quant_edge_lane(row) -> bool:
        if not enable_quant_edge_book:
            return False
        strategy_local = str(row.get("strategy", "") or "").strip()
        verdict_local = str(row.get("verdict", "") or "").strip().upper()
        edge_val = fnum(row.get("edge_pct"))
        signals_val = fnum(row.get("signals"))
        dte_val = fnum(row.get("dte"))
        iv_rank_val = fnum(row.get("iv_rank"))
        if quant_edge_allowed_strategies and strategy_local not in quant_edge_allowed_strategies:
            return False
        if quant_edge_allowed_verdicts and verdict_local not in quant_edge_allowed_verdicts:
            return False
        checks = [
            np.isfinite(edge_val) and edge_val >= quant_edge_min_edge_pct,
            np.isfinite(signals_val) and signals_val >= quant_edge_min_signals,
            np.isfinite(dte_val) and dte_val >= quant_edge_min_dte and dte_val <= quant_edge_max_dte,
            np.isfinite(iv_rank_val) and iv_rank_val >= quant_edge_min_iv_rank and iv_rank_val <= quant_edge_max_iv_rank,
            _row_live_price_ok(row),
            _earnings_clean(row),
            _quant_edge_allows_blockers(row),
        ]
        return bool(all(checks))

    def _short_dte_edge_base_checks(row) -> bool:
        if not enable_short_dte_edge_book:
            return False
        strategy_local = str(row.get("strategy", "") or "").strip()
        verdict_local = str(row.get("verdict", "") or "").strip().upper()
        edge_val = fnum(row.get("edge_pct"))
        signals_val = fnum(row.get("signals"))
        dte_val = fnum(row.get("dte"))
        iv_rank_val = fnum(row.get("iv_rank"))
        confidence_score_val = fnum(row.get("confidence_score"))
        if short_dte_edge_allowed_strategies and strategy_local not in short_dte_edge_allowed_strategies:
            return False
        if short_dte_edge_allowed_verdicts and verdict_local not in short_dte_edge_allowed_verdicts:
            return False
        checks = [
            np.isfinite(edge_val) and edge_val >= short_dte_edge_min_edge_pct,
            np.isfinite(signals_val) and signals_val >= short_dte_edge_min_signals,
            np.isfinite(confidence_score_val) and confidence_score_val >= short_dte_edge_min_confidence_score,
            np.isfinite(dte_val) and dte_val >= short_dte_edge_min_dte and dte_val <= short_dte_edge_max_dte,
            np.isfinite(iv_rank_val) and iv_rank_val <= short_dte_edge_max_iv_rank,
            _row_live_price_ok(row),
            _earnings_clean(row),
            _short_dte_edge_allows_blockers(row),
        ]
        return bool(all(checks))

    def _high_signal_edge_base_checks(row) -> bool:
        if not enable_high_signal_edge_book:
            return False
        strategy_local = str(row.get("strategy", "") or "").strip()
        verdict_local = str(row.get("verdict", "") or "").strip().upper()
        approval_regime_local = str(row.get("approval_regime", "") or "").strip().lower()
        gex_wall_context_local = str(row.get("gex_wall_context", "") or "").strip()
        contract_flow_local = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
        edge_val = fnum(row.get("edge_pct"))
        signals_val = fnum(row.get("signals"))
        dte_val = fnum(row.get("dte"))
        iv_rank_val = fnum(row.get("iv_rank"))
        confidence_score_val = fnum(row.get("confidence_score"))
        if high_signal_edge_allowed_strategies and strategy_local not in high_signal_edge_allowed_strategies:
            return False
        if high_signal_edge_allowed_verdicts and verdict_local not in high_signal_edge_allowed_verdicts:
            return False
        if approval_regime_local in high_signal_edge_excluded_approval_regimes:
            return False
        if high_signal_edge_allowed_gex_wall_contexts and gex_wall_context_local not in high_signal_edge_allowed_gex_wall_contexts:
            return False
        if high_signal_edge_require_gex_context and not gex_wall_context_local:
            return False
        if high_signal_edge_require_contract_flow_confirmed and contract_flow_local != "confirmed":
            return False
        checks = [
            np.isfinite(edge_val) and edge_val >= high_signal_edge_min_edge_pct,
            np.isfinite(signals_val) and signals_val >= high_signal_edge_min_signals,
            np.isfinite(confidence_score_val) and confidence_score_val >= high_signal_edge_min_confidence_score,
            np.isfinite(dte_val) and dte_val >= high_signal_edge_min_dte and dte_val <= high_signal_edge_max_dte,
            np.isfinite(iv_rank_val) and iv_rank_val <= high_signal_edge_max_iv_rank,
            _row_live_price_ok(row),
            _earnings_clean(row),
            _high_signal_edge_allows_blockers(row),
        ]
        return bool(all(checks))

    mdf["quant_edge_lane"] = mdf.apply(_quant_edge_lane, axis=1)
    # Short-DTE edge promotions are applied after Qualified Review is computed.
    # That keeps the lane from bypassing research-only/safety downgrades.
    mdf["short_dte_edge_lane"] = False
    # High-signal edge promotions are also post-Qualified-Review. The lane exists
    # for high-sample setups that replay says were overblocked by hard cliffs.
    mdf["high_signal_edge_lane"] = False

    def execution_book(row):
        hard_tokens = [x for x in str(row.get("hard_blockers", "")).split(";") if str(x).strip()]
        quality_tokens = [x for x in str(row.get("quality_blockers", "")).split(";") if str(x).strip()]
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        _strat_for_scout = str(row.get("strategy", "")).strip()
        _edge_for_scout = fnum(row.get("edge_pct"))
        _signals_for_scout = fnum(row.get("signals"))
        _dte_for_scout = fnum(row.get("dte"))
        _iv_rank_for_scout = fnum(row.get("iv_rank"))
        _vix_for_scout = fnum(row.get("vix_level"))
        _spy_5d_for_scout = fnum(row.get("spy_5d_ret"))
        _rr_for_scout = fnum(row.get("live_reward_risk"))
        _width_for_scout = fnum(row.get("width_live"))
        if not np.isfinite(_width_for_scout):
            _width_for_scout = fnum(row.get("width"))
        _net_for_scout = fnum(row.get("live_net_bid_ask"))
        if not np.isfinite(_net_for_scout):
            _net_for_scout = fnum(row.get("live_net_mark"))
        _debit_frac_for_scout = (
            _net_for_scout / _width_for_scout
            if np.isfinite(_net_for_scout) and np.isfinite(_width_for_scout) and _width_for_scout > 0
            else math.nan
        )
        _likelihood_strength_for_scout = str(row.get("likelihood_strength", "") or "").strip().upper()
        def _bear_put_scout_allows_hard_tokens(tokens):
            """Scout can downsize weak evidence, but must not override safety/contra vetoes."""
            allowed_exact = {
                "stage1_flow_weak_or_ambiguous",
            }
            allowed_prefixes = (
                "stage1_conviction_below_yes_good:",
            )
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                if token in allowed_exact:
                    continue
                if any(token.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                return False
            return True

        def _event_momentum_allows_hard_tokens(tokens):
            """Event Scout can forgive Stage-1/high-IV discovery blocks only."""
            allowed_exact = {
                "stage1_flow_weak_or_ambiguous",
            }
            allowed_prefixes = (
                "stage1_conviction_below_yes_good:",
                "stage1_high_iv_debit_watch_only:",
            )
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                if token in allowed_exact:
                    continue
                if any(token.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                return False
            return True

        def _debit_momentum_allows_hard_tokens(tokens):
            """Debit Scout can forgive only discovery/sample weakness.

            Core/Tactical still require the full sample/verdict gates.  Scout has
            its own lower min-signal and low-sample rules, so do not let the
            higher Core/Tactical `signals_below` or LOW_SAMPLE tokens kill Scout
            before those Scout-specific checks run.
            """
            allowed_exact = {
                "stage1_flow_weak_or_ambiguous",
                "likelihood_verdict:LOW_SAMPLE",
                "likelihood_strength_unranked:Low Sample",
                "likelihood_strength_blocked:Low Sample",
            }
            allowed_prefixes = (
                "stage1_conviction_below_yes_good:",
                "stage1_high_iv_debit_watch_only:",
                "signals_below:",
            )
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                if token in allowed_exact:
                    continue
                if any(token.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                return False
            return True

        def _medium_allows_hard_tokens(tokens):
            """Medium is a real but smaller book; do not override safety/contra failures."""
            allowed_exact = {
                "stage1_flow_weak_or_ambiguous",
            }
            allowed_prefixes = (
                "stage1_conviction_below_yes_good:",
                "stage1_high_iv_debit_watch_only:",
                "bull_call_dte_too_long:",
                "bull_call_rr_weak:",
            )
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                if token in allowed_exact:
                    continue
                if any(token.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                return False
            return True

        def _income_allows_hard_tokens(tokens):
            """Income may forgive discovery/sample weakness, never structural risk."""
            allowed_prefixes = (
                "stage1_conviction_below_yes_good:",
                "signals_below:",
            )
            forbidden_prefixes = (
                "shield_delta_fail",
                "live_status:",
                "live_entry_gate_fail",
                "credit_below",
                "stage1_contract_flow_contra",
                "contract_flow_contra",
            )
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                if any(token.startswith(prefix) for prefix in forbidden_prefixes):
                    return False
                if any(token.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                return False
            return True

        _conv_for_event = fnum(row.get("conviction"))
        _breakeven_for_event = fnum(row.get("breakeven"))
        _spot_for_event = fnum(row.get("spot_live_effective"))
        if not np.isfinite(_spot_for_event):
            _spot_for_event = fnum(row.get("spot_live"))
        _notes_for_event = str(row.get("notes_stage1", "") or row.get("notes", "") or "").upper()
        _earnings_for_event = str(row.get("earnings_label_stage1", "") or "").upper()
        _has_event_context = (
            "ER-RISK" in _notes_for_event
            or "EARN" in _notes_for_event
            or "ER" in _earnings_for_event
            or any(str(t).startswith("stage1_high_iv_debit_watch_only:") for t in hard_tokens)
        )
        _contract_flow_for_event = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
        _event_direction_ok = True
        if event_momentum_scout_require_breakeven_cross:
            if _strat_for_scout == "Bull Call Debit":
                _event_direction_ok = (
                    np.isfinite(_spot_for_event)
                    and np.isfinite(_breakeven_for_event)
                    and _spot_for_event >= _breakeven_for_event
                )
            elif _strat_for_scout == "Bear Put Debit":
                _event_direction_ok = (
                    np.isfinite(_spot_for_event)
                    and np.isfinite(_breakeven_for_event)
                    and _spot_for_event <= _breakeven_for_event
                )
        event_momentum_scout_candidate = (
            enable_scout_book
            and enable_event_momentum_scout
            and _strat_for_scout in {"Bull Call Debit", "Bear Put Debit"}
            and ok_live
            and _has_event_context
            and _event_momentum_allows_hard_tokens(hard_tokens)
            and np.isfinite(_conv_for_event)
            and _conv_for_event >= event_momentum_scout_min_conviction
            and np.isfinite(_dte_for_scout)
            and _dte_for_scout <= event_momentum_scout_max_dte
            and np.isfinite(_rr_for_scout)
            and _rr_for_scout >= event_momentum_scout_min_reward_risk
            and np.isfinite(_debit_frac_for_scout)
            and _debit_frac_for_scout <= event_momentum_scout_max_debit_frac
            and (
                (not event_momentum_scout_require_contract_confirmed)
                or _contract_flow_for_event == "confirmed"
            )
            and _event_direction_ok
        )

        _contract_flow_for_debit = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
        _flow_dir_for_debit = str(row.get("flow_direction", "") or "").strip().lower()
        _flow_conf_for_debit = str(row.get("flow_confirmation", "") or "").strip().lower()
        _verdict_for_debit = str(row.get("verdict", "") or "").strip().upper()
        _regime_score_for_debit = fnum(row.get("market_regime_score"))
        _quality_for_debit = [str(t).strip() for t in quality_tokens if str(t).strip()]
        _scout_quality_blocked = scout_block_gex_volatile_breakout and any(
            t.startswith("gex_volatile_breakout") for t in _quality_for_debit
        )
        _debit_edge_floor = debit_momentum_scout_min_edge_pct
        if _strat_for_scout == "Bear Put Debit":
            _debit_edge_floor = max(debit_momentum_scout_min_edge_pct, debit_momentum_scout_bear_min_edge_pct)
        _debit_direction_ok = True
        if _flow_conf_for_debit == "confirmed":
            if _strat_for_scout == "Bull Call Debit" and _flow_dir_for_debit == "bearish":
                _debit_direction_ok = False
            elif _strat_for_scout == "Bear Put Debit" and _flow_dir_for_debit == "bullish":
                _debit_direction_ok = False
        _debit_bear_quality_ok = True
        if _strat_for_scout == "Bear Put Debit":
            if (
                debit_momentum_scout_bear_require_flow_confirmed
                and not (_flow_conf_for_debit == "confirmed" and _flow_dir_for_debit == "bearish")
            ):
                _debit_bear_quality_ok = False
            if _likelihood_strength_for_scout not in debit_momentum_scout_bear_likelihood_strengths:
                _debit_bear_quality_ok = False
            if debit_momentum_scout_block_gex_volatile_breakout and any(
                t.startswith("gex_volatile_breakout") for t in _quality_for_debit
            ):
                _debit_bear_quality_ok = False
        debit_momentum_scout_candidate = (
            enable_scout_book
            and allow_debit_momentum_scout_lane
            and _strat_for_scout in {"Bull Call Debit", "Bear Put Debit"}
            and not _scout_quality_blocked
            and ok_live
            and _debit_momentum_allows_hard_tokens(hard_tokens)
            and _debit_direction_ok
            and _debit_bear_quality_ok
            and (
                (not debit_momentum_scout_require_verdict_pass)
                or _verdict_for_debit == "PASS"
            )
            and np.isfinite(_conv_for_event)
            and _conv_for_event >= debit_momentum_scout_min_conviction
            and np.isfinite(_edge_for_scout)
            and _edge_for_scout >= _debit_edge_floor
            and np.isfinite(_signals_for_scout)
            and _signals_for_scout >= debit_momentum_scout_min_signals
            and np.isfinite(_dte_for_scout)
            and _dte_for_scout >= debit_momentum_scout_min_dte
            and _dte_for_scout <= debit_momentum_scout_max_dte
            and np.isfinite(_rr_for_scout)
            and _rr_for_scout >= debit_momentum_scout_min_reward_risk
            and np.isfinite(_debit_frac_for_scout)
            and _debit_frac_for_scout <= debit_momentum_scout_max_debit_frac
            and np.isfinite(_iv_rank_for_scout)
            and _iv_rank_for_scout <= debit_momentum_scout_max_iv_rank
            and (
                (not debit_momentum_scout_require_contract_confirmed)
                or _contract_flow_for_debit == "confirmed"
            )
            and _contract_flow_for_debit not in {"contra", "directional"}
            and np.isfinite(_regime_score_for_debit)
            and _regime_score_for_debit >= debit_momentum_scout_min_regime_score
        )

        _medium_contract_flow_ok = _contract_flow_for_debit not in {
            "contra",
            "directional",
            "weak_or_ambiguous",
            "unknown",
        }
        _medium_edge_floor = medium_min_edge_pct
        if _strat_for_scout in {"Bear Put Debit", "Bear Call Credit"}:
            _medium_edge_floor = max(medium_min_edge_pct, medium_bear_min_edge_pct)
        elif str(row.get("track", "")).strip().upper() == "SHIELD":
            _medium_edge_floor = max(medium_min_edge_pct, medium_shield_min_edge_pct)
        _medium_debit_ok = True
        if _strat_for_scout in {"Bull Call Debit", "Bear Put Debit"}:
            _medium_debit_ok = (
                np.isfinite(_debit_frac_for_scout)
                and (
                    (not np.isfinite(medium_max_debit_frac))
                    or _debit_frac_for_scout <= medium_max_debit_frac
                )
            )
        _medium_regime_conf = str(row.get("market_regime_confidence", "") or "").strip().upper()
        medium_book_candidate = (
            enable_medium_book
            and ok_live
            and (
                not medium_allowed_strategies
                or _strat_for_scout in medium_allowed_strategies
            )
            and _strat_for_scout not in {"Long Iron Condor"}
            and _medium_allows_hard_tokens(hard_tokens)
            and _medium_contract_flow_ok
            and (
                (not medium_require_contract_confirmed)
                or _contract_flow_for_debit == "confirmed"
            )
            and (
                (not medium_require_verdict_pass)
                or _verdict_for_debit == "PASS"
            )
            and np.isfinite(_conv_for_event)
            and _conv_for_event >= medium_min_conviction
            and np.isfinite(fnum(row.get("confidence_score")))
            and fnum(row.get("confidence_score")) >= medium_review_min_confidence_score
            and np.isfinite(fnum(row.get("edge_score")))
            and fnum(row.get("edge_score")) >= medium_review_min_edge_score
            and (
                (not np.isfinite(medium_review_min_pop))
                or (
                    np.isfinite(normalize_probability(row.get("hist_success_pct")))
                    and normalize_probability(row.get("hist_success_pct")) >= medium_review_min_pop
                )
            )
            and (
                (not medium_block_earnings_risk)
                or _earnings_clean(row)
            )
            and np.isfinite(_edge_for_scout)
            and _edge_for_scout >= _medium_edge_floor
            and np.isfinite(_signals_for_scout)
            and _signals_for_scout >= medium_min_signals
            and np.isfinite(_dte_for_scout)
            and _dte_for_scout >= medium_min_dte
            and _dte_for_scout <= medium_max_dte
            and (
                not medium_allowed_market_confidences
                or _medium_regime_conf in medium_allowed_market_confidences
            )
            and (
                (not np.isfinite(medium_max_iv_rank))
                or (np.isfinite(_iv_rank_for_scout) and _iv_rank_for_scout <= medium_max_iv_rank)
            )
            and (
                _strat_for_scout not in {"Bull Call Debit", "Bear Put Debit"}
                or (
                    _medium_debit_ok
                    and np.isfinite(_rr_for_scout)
                    and _rr_for_scout >= medium_min_reward_risk
                )
            )
        )

        _hist_success_for_income = fnum(row.get("hist_success_pct"))
        _credit_no_touch_for_income = fnum(row.get("credit_no_touch_pct"))
        _gex_for_income = str(row.get("gex_regime", "") or "").strip().lower()
        _income_contract_flow_ok = _contract_flow_for_debit not in {"contra", "directional"}
        income_book_candidate = (
            enable_income_book
            and ok_live
            and _income_allows_hard_tokens(hard_tokens)
            and (
                not income_allowed_strategies
                or _strat_for_scout in income_allowed_strategies
            )
            and _strat_for_scout in {"Iron Condor", "Iron Butterfly"}
            and _income_contract_flow_ok
            and (
                not income_allowed_verdicts
                or _verdict_for_debit in income_allowed_verdicts
            )
            and np.isfinite(_edge_for_scout)
            and _edge_for_scout >= income_min_edge_pct
            and np.isfinite(_signals_for_scout)
            and _signals_for_scout >= income_min_signals
            and np.isfinite(_hist_success_for_income)
            and _hist_success_for_income >= income_min_hist_success_pct
            and np.isfinite(_dte_for_scout)
            and _dte_for_scout >= income_min_dte
            and _dte_for_scout <= income_max_dte
            and (
                not income_allowed_gex_regimes
                or _gex_for_income in income_allowed_gex_regimes
            )
            and (
                (not np.isfinite(income_max_iv_rank))
                or (np.isfinite(_iv_rank_for_scout) and _iv_rank_for_scout <= income_max_iv_rank)
            )
            and (
                (not np.isfinite(income_min_credit_no_touch_pct))
                or income_min_credit_no_touch_pct <= 0
                or (
                    np.isfinite(_credit_no_touch_for_income)
                    and _credit_no_touch_for_income >= income_min_credit_no_touch_pct
                )
            )
        )

        bear_put_scout_candidate = (
            enable_scout_book
            and allow_bear_put_scout_lane
            and _strat_for_scout == "Bear Put Debit"
            and not _scout_quality_blocked
            and ok_live
            and _bear_put_scout_allows_hard_tokens(hard_tokens)
            and _likelihood_strength_for_scout in bear_put_scout_likelihood_strengths
            and (not bear_put_scout_require_negative_edge or (np.isfinite(_edge_for_scout) and _edge_for_scout < 0))
            and np.isfinite(_signals_for_scout)
            and _signals_for_scout >= bear_put_scout_min_signals
            and np.isfinite(_dte_for_scout)
            and _dte_for_scout >= bear_put_scout_min_dte
            and _dte_for_scout <= bear_put_scout_max_dte
            and np.isfinite(_iv_rank_for_scout)
            and _iv_rank_for_scout <= bear_put_scout_max_iv_rank
            and np.isfinite(_vix_for_scout)
            and _vix_for_scout < bear_put_scout_max_vix
            and (
                (not bear_put_scout_require_spy_5d_nonnegative)
                or (np.isfinite(_spy_5d_for_scout) and _spy_5d_for_scout >= 0)
            )
            and np.isfinite(_rr_for_scout)
            and _rr_for_scout >= bear_put_scout_min_reward_risk
            and np.isfinite(_debit_frac_for_scout)
            and _debit_frac_for_scout <= bear_put_scout_max_debit_frac
        )
        _source_for_pilot = str(row.get("source", "") or "").strip()
        _external_pilot_source = _source_for_pilot.startswith("external_scanner:")
        _pilot_only_source = bool(row.get("pilot_only_candidate")) or _external_pilot_source
        _native_pilot_source = (
            enable_native_pilot_book
            and not _pilot_only_source
            and _strat_for_scout in {"Bull Call Debit", "Bear Put Debit"}
            and str(row.get("net_type", "")).strip().lower() == "debit"
        )
        _pilot_loss = fnum(row.get("live_max_loss"))
        if not np.isfinite(_pilot_loss) or _pilot_loss <= 0:
            _pilot_loss = fnum(row.get("max_loss"))
        if not np.isfinite(_pilot_loss) or _pilot_loss <= 0:
            _pilot_gate = fnum(row.get("gate_target"))
            _pilot_width = fnum(row.get("width_live"))
            if not np.isfinite(_pilot_width):
                _pilot_width = fnum(row.get("width"))
            if np.isfinite(_pilot_gate) and np.isfinite(_pilot_width):
                _, _target_loss = calc_target_max(str(row.get("net_type", "")).strip().lower(), _pilot_width, _pilot_gate)
                _pilot_loss = fnum(_target_loss)
        _pilot_pop = normalize_probability(row.get("external_pop"))
        if not np.isfinite(_pilot_pop):
            _pilot_pop = normalize_probability(row.get("hist_success_pct"))
        _pilot_ev_ml = fnum(row.get("external_ev_ml_effective"))
        if not np.isfinite(_pilot_ev_ml):
            _pilot_ev_ml = fnum(row.get("external_ev_ml"))
        if _native_pilot_source:
            _pilot_ev_ml = fnum(row.get("debit_partial_ev_ml_target_live"))
            if not np.isfinite(_pilot_ev_ml):
                _pilot_ev_ml = fnum(row.get("debit_partial_ev_ml_effective"))
        _pilot_conv = fnum(row.get("conviction"))
        _pilot_pop_ok = (
            np.isfinite(_pilot_pop)
            and (
                _pilot_pop >= pilot_high_pop_min
                or (
                    _pilot_pop >= pilot_floor_pop_min
                    and _verdict_for_debit == "PASS"
                    and np.isfinite(_signals_for_scout)
                    and _signals_for_scout >= pilot_pass_min_signals
                    and np.isfinite(_edge_for_scout)
                    and _edge_for_scout >= pilot_pass_min_edge
                )
            )
        )
        _pilot_rr_ok = (
            (not np.isfinite(_rr_for_scout))
            or (np.isfinite(pilot_min_reward_risk) and _rr_for_scout >= pilot_min_reward_risk)
        )
        _pilot_spot = fnum(row.get("spot_live_effective"))
        if not np.isfinite(_pilot_spot) or _pilot_spot <= 0:
            _pilot_spot = fnum(row.get("spot_asof_close"))
        _pilot_long = fnum(row.get("long_strike"))
        _pilot_long_otm = 0.0
        if np.isfinite(_pilot_spot) and _pilot_spot > 0 and np.isfinite(_pilot_long):
            if _strat_for_scout == "Bull Call Debit":
                _pilot_long_otm = max(0.0, (_pilot_long / _pilot_spot) - 1.0)
            elif _strat_for_scout == "Bear Put Debit":
                _pilot_long_otm = max(0.0, 1.0 - (_pilot_long / _pilot_spot))
        _pilot_moneyness_ok = (
            (not np.isfinite(pilot_max_long_otm_pct))
            or _pilot_long_otm <= pilot_max_long_otm_pct
        )
        _pilot_loss_cap = native_pilot_max_loss if _native_pilot_source else pilot_max_loss
        _pilot_ev_floor = native_pilot_min_partial_ev_ml if _native_pilot_source else pilot_min_ev_ml
        _market_conf_for_pilot = str(row.get("market_regime_confidence", "") or "").strip().upper()
        _native_pilot_quality_ok = True
        if _native_pilot_source:
            _native_pilot_quality_ok = (
                _verdict_for_debit in {"PASS", "LOW_SAMPLE"}
                and (
                    not native_pilot_market_confidences
                    or _market_conf_for_pilot in native_pilot_market_confidences
                )
                and (
                    (not native_pilot_require_contract_confirmed)
                    or _contract_flow_for_debit == "confirmed"
                )
                and np.isfinite(_signals_for_scout)
                and _signals_for_scout >= native_pilot_min_signals
                and np.isfinite(_edge_for_scout)
                and _edge_for_scout >= native_pilot_min_edge
            )
        pilot_convexity_candidate = (
            enable_pilot_book
            and (_pilot_only_source or _native_pilot_source)
            and _strat_for_scout in {"Bull Call Debit", "Bear Put Debit"}
            and str(row.get("net_type", "")).strip().lower() == "debit"
            and ok_live
            and _native_pilot_quality_ok
            and pilot_convexity_blockers_allow(hard_tokens + quality_tokens)
            and np.isfinite(_pilot_loss)
            and _pilot_loss > 0
            and _pilot_loss <= _pilot_loss_cap
            and np.isfinite(_pilot_ev_ml)
            and _pilot_ev_ml >= _pilot_ev_floor
            and _pilot_pop_ok
            and np.isfinite(_pilot_conv)
            and _pilot_conv >= pilot_min_conviction
            and np.isfinite(_dte_for_scout)
            and _dte_for_scout >= pilot_min_dte
            and _dte_for_scout <= pilot_max_dte
            and _pilot_moneyness_ok
            and _pilot_rr_ok
            and np.isfinite(_debit_frac_for_scout)
            and _debit_frac_for_scout <= pilot_max_debit_frac
            and _contract_flow_for_debit != "directional"
        )
        if pilot_convexity_candidate:
            return "Pilot"
        if _pilot_only_source:
            return "Watch"
        if bool(row.get("regime_weekly_lane", False)):
            return "Tactical"
        if bool(row.get("quant_edge_lane", False)) or bool(row.get("short_dte_edge_lane", False)):
            return "Medium"
        if hard_tokens:
            if income_book_candidate:
                return "Income"
            if medium_book_candidate:
                return "Medium"
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        if not quality_tokens:
            return "Core"
        if not enable_dual_books:
            return "Watch"
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        if not ok_live:
            return "Watch"
        if income_book_candidate:
            return "Income"
        if medium_book_candidate:
            return "Medium"
        track_local = str(row.get("track", "")).strip().upper()
        shield_override_live = bool(
            shield_live_valid_overrides_quality and track_local == "SHIELD" and ok_live
        )
        # T3 floor: even with live-valid override, enforce minimum quality floors
        if shield_override_live:
            _no_touch = fnum(row.get("credit_no_touch_pct"))
            _edge = fnum(row.get("edge_pct"))
            if np.isfinite(shield_live_valid_min_no_touch) and shield_live_valid_min_no_touch > 0:
                if not np.isfinite(_no_touch) or _no_touch < shield_live_valid_min_no_touch:
                    shield_override_live = False
            if np.isfinite(shield_live_valid_min_edge) and shield_live_valid_min_edge > 0:
                if not np.isfinite(_edge) or _edge < shield_live_valid_min_edge:
                    shield_override_live = False
        if (
            not bool(row.get("stage1_effective"))
            and not shield_override_live
            and not _has_quality_stage1_context(quality_tokens)
        ):
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        conv = fnum(row.get("conviction"))
        evidence_lane = (
            bool(row.get("bull_call_evidence_lane"))
            or bool(row.get("bear_put_evidence_lane"))
            or bool(row.get("bull_call_pinned_continuation_lane"))
        )
        _strat = str(row.get("strategy", "")).strip()
        edge = fnum(row.get("edge_pct"))
        pinned_continuation_lane = bool(row.get("bull_call_pinned_continuation_lane"))
        pinned_core_candidate = (
            pinned_continuation_lane
            and np.isfinite(conv)
            and conv >= bull_call_pinned_continuation_core_min_conviction
            and str(row.get("flow_confirmation", "")).strip().lower() == "confirmed"
            and str(row.get("flow_direction", "")).strip().lower() == "bullish"
            and not any(str(t).startswith("long_delta_proxy_ok") for t in quality_tokens)
        )
        scout_candidate = (
            enable_scout_book
            and evidence_lane
            and _strat == "Bull Call Debit"
            and not _scout_quality_blocked
            and np.isfinite(edge)
            and edge >= scout_min_edge_pct
            and edge < scout_max_edge_pct
        )
        ic_income_constructive = (
            _strat in {"Iron Condor", "Iron Butterfly"}
            and str(row.get("gex_wall_context", "")).strip() == "pinned_income_constructive"
        )
        if (
            not evidence_lane
            and not ic_income_constructive
            and np.isfinite(tactical_min_conviction)
            and (
                not np.isfinite(conv)
                or conv < max(0.0, tactical_min_conviction - tactical_min_conviction_grace)
            )
        ):
            if income_book_candidate:
                return "Income"
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        _is_bear_tac = _strat in {"Bear Put Debit", "Bear Call Credit"}
        _tac_edge = (
            max(tactical_min_edge_pct, min_edge_pct_bear) if _is_bear_tac
            else max(tactical_min_edge_pct, min_edge_pct_shield) if track_local == "SHIELD"
            else tactical_min_edge_pct
        )
        if (
            not ic_income_constructive
            and np.isfinite(_tac_edge)
            and (not np.isfinite(edge) or edge < _tac_edge)
        ):
            if income_book_candidate:
                return "Income"
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if (scout_candidate or bear_put_scout_candidate) else "Watch"
        # [T9] Tactical debit/width cap — block expensive-for-width Tactical trades
        if _strat in {"Bull Call Debit", "Bear Put Debit"}:
            _tac_width = fnum(row.get("width_live"))
            if not np.isfinite(_tac_width):
                _tac_width = fnum(row.get("width"))
            _tac_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(_tac_net):
                _tac_net = fnum(row.get("live_net_mark"))
            if np.isfinite(_tac_width) and _tac_width > 0 and np.isfinite(_tac_net):
                _tac_debit_pct = _tac_net / _tac_width
                if (
                    np.isfinite(tactical_max_debit_pct_width)
                    and _tac_debit_pct > tactical_max_debit_pct_width + tactical_debit_pct_width_tolerance
                ):
                    return "Watch"
        sig = fnum(row.get("signals"))
        if (
            (not shield_override_live)
            and (not evidence_lane)
            and (not ic_income_constructive)
            and np.isfinite(tactical_min_signals)
            and (
                not np.isfinite(sig)
                or sig < max(0.0, tactical_min_signals - tactical_min_signals_grace)
            )
        ):
            if income_book_candidate:
                return "Income"
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        verdict = str(row.get("verdict", "")).strip().upper()
        if (
            (not shield_override_live)
            and (not ic_income_constructive)
            and tactical_require_verdict_pass
            and verdict != "PASS"
        ):
            if income_book_candidate:
                return "Income"
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        if pinned_core_candidate:
            return "Core"
        if pinned_continuation_lane:
            return "Tactical"
        if debit_momentum_scout_candidate:
            return "Scout"
        if event_momentum_scout_candidate:
            return "Scout"
        if bear_put_scout_candidate:
            return "Scout"
        if scout_candidate:
            return "Scout"
        if income_book_candidate:
            return "Income"
        return "Tactical"

    mdf["execution_book"] = mdf.apply(execution_book, axis=1)

    def _gate_num(value) -> str:
        val = fnum(value)
        return "nan" if not np.isfinite(val) else f"{val:g}"

    def _append_semicolon_tokens(base, extra) -> str:
        base_items = [x.strip() for x in str(base or "").split(";") if x.strip()]
        extra_items = [x.strip() for x in str(extra or "").split(";") if x.strip()]
        seen = set(base_items)
        for item in extra_items:
            if item not in seen:
                base_items.append(item)
                seen.add(item)
        return ";".join(base_items)

    quant_edge_mask = mdf.get("quant_edge_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
    quant_edge_mask = quant_edge_mask & mdf["execution_book"].astype(str).eq("Medium")
    if quant_edge_mask.any() and (quant_edge_max_per_day > 0 or quant_edge_max_per_ticker_day > 0):
        quant_rows = mdf.loc[quant_edge_mask].copy()
        quant_rows["_quant_edge_edge"] = pd.to_numeric(quant_rows.get("edge_pct"), errors="coerce").fillna(-1e9)
        quant_rows["_quant_edge_signals"] = pd.to_numeric(quant_rows.get("signals"), errors="coerce").fillna(-1e9)
        quant_rows["_quant_edge_conviction"] = pd.to_numeric(quant_rows.get("conviction"), errors="coerce").fillna(-1e9)
        quant_rows = quant_rows.sort_values(
            ["_quant_edge_edge", "_quant_edge_signals", "_quant_edge_conviction"],
            ascending=[False, False, False],
        )
        keep_quant = []
        per_quant_ticker = defaultdict(int)
        for qidx, qrow in quant_rows.iterrows():
            qticker = str(qrow.get("ticker", "") or "").strip().upper()
            if quant_edge_max_per_ticker_day > 0 and per_quant_ticker[qticker] >= quant_edge_max_per_ticker_day:
                continue
            if quant_edge_max_per_day > 0 and len(keep_quant) >= quant_edge_max_per_day:
                break
            keep_quant.append(qidx)
            per_quant_ticker[qticker] += 1
        drop_quant = sorted(set(quant_rows.index) - set(keep_quant))
        if drop_quant:
            mdf.loc[drop_quant, "book_blockers"] = mdf.loc[drop_quant].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("book_blockers", ""),
                    f"quant_edge_daily_cap:{len(keep_quant)}/{quant_edge_max_per_day}",
                ),
                axis=1,
            )
            mdf.loc[drop_quant, "execution_book"] = "Watch"

    short_dte_edge_mask = mdf.get("short_dte_edge_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
    short_dte_edge_mask = short_dte_edge_mask & mdf["execution_book"].astype(str).eq("Medium")
    if short_dte_edge_mask.any() and (short_dte_edge_max_per_day > 0 or short_dte_edge_max_per_ticker_day > 0):
        short_rows = mdf.loc[short_dte_edge_mask].copy()
        short_rows["_short_dte_conf"] = pd.to_numeric(short_rows.get("confidence_score"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_edge"] = pd.to_numeric(short_rows.get("edge_pct"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_signals"] = pd.to_numeric(short_rows.get("signals"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_conviction"] = pd.to_numeric(short_rows.get("conviction"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_hist"] = pd.to_numeric(short_rows.get("hist_success_pct"), errors="coerce").fillna(-1e9)
        short_rows = short_rows.sort_values(
            ["_short_dte_conf", "_short_dte_edge", "_short_dte_signals", "_short_dte_conviction", "_short_dte_hist"],
            ascending=[False, False, False, False, False],
        )
        keep_short = []
        per_short_ticker = defaultdict(int)
        for sidx, srow in short_rows.iterrows():
            sticker = str(srow.get("ticker", "") or "").strip().upper()
            if short_dte_edge_max_per_ticker_day > 0 and per_short_ticker[sticker] >= short_dte_edge_max_per_ticker_day:
                continue
            if short_dte_edge_max_per_day > 0 and len(keep_short) >= short_dte_edge_max_per_day:
                break
            keep_short.append(sidx)
            per_short_ticker[sticker] += 1
        drop_short = sorted(set(short_rows.index) - set(keep_short))
        if drop_short:
            mdf.loc[drop_short, "book_blockers"] = mdf.loc[drop_short].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("book_blockers", ""),
                    f"short_dte_edge_daily_cap:{len(keep_short)}/{short_dte_edge_max_per_day}",
                ),
                axis=1,
            )
            mdf.loc[drop_short, "execution_book"] = "Watch"

    def tactical_floor_blockers(row) -> str:
        if str(row.get("execution_book", "")).strip() != "Watch":
            return ""
        quality_tokens = [x for x in str(row.get("quality_blockers", "")).split(";") if str(x).strip()]
        if not quality_tokens:
            return ""
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        if not ok_live:
            return ""
        if not enable_dual_books:
            return "tactical_book_disabled"

        track_local = str(row.get("track", "")).strip().upper()
        shield_override_live = bool(
            shield_live_valid_overrides_quality and track_local == "SHIELD" and ok_live
        )
        if shield_override_live:
            _no_touch = fnum(row.get("credit_no_touch_pct"))
            _edge = fnum(row.get("edge_pct"))
            if np.isfinite(shield_live_valid_min_no_touch) and shield_live_valid_min_no_touch > 0:
                if not np.isfinite(_no_touch) or _no_touch < shield_live_valid_min_no_touch:
                    shield_override_live = False
            if np.isfinite(shield_live_valid_min_edge) and shield_live_valid_min_edge > 0:
                if not np.isfinite(_edge) or _edge < shield_live_valid_min_edge:
                    shield_override_live = False

        floor_tokens = []
        if (
            not bool(row.get("stage1_effective"))
            and not shield_override_live
            and not _has_quality_stage1_context(quality_tokens)
        ):
            floor_tokens.append("stage1_not_effective")

        conv = fnum(row.get("conviction"))
        evidence_lane = (
            bool(row.get("bull_call_evidence_lane"))
            or bool(row.get("bear_put_evidence_lane"))
            or bool(row.get("bull_call_pinned_continuation_lane"))
        )
        strategy_local = str(row.get("strategy", "")).strip()
        ic_income_constructive = (
            strategy_local in {"Iron Condor", "Iron Butterfly"}
            and str(row.get("gex_wall_context", "")).strip() == "pinned_income_constructive"
        )
        if (
            not evidence_lane
            and not ic_income_constructive
            and np.isfinite(tactical_min_conviction)
        ):
            tactical_conviction_floor = max(
                0.0,
                tactical_min_conviction - tactical_min_conviction_grace,
            )
            if not np.isfinite(conv) or conv < tactical_conviction_floor:
                floor_tokens.append(
                    f"tactical_conviction_below:{_gate_num(conv)}<{_gate_num(tactical_min_conviction)}"
                )
            elif conv < tactical_min_conviction:
                floor_tokens.append(
                    f"tactical_conviction_near_limit:{_gate_num(conv)}<{_gate_num(tactical_min_conviction)}"
                )

        edge = fnum(row.get("edge_pct"))
        is_bear_tactical = strategy_local in {"Bear Put Debit", "Bear Call Credit"}
        tactical_edge_floor = (
            max(tactical_min_edge_pct, min_edge_pct_bear) if is_bear_tactical
            else max(tactical_min_edge_pct, min_edge_pct_shield) if track_local == "SHIELD"
            else tactical_min_edge_pct
        )
        if (
            not ic_income_constructive
            and np.isfinite(tactical_edge_floor)
            and (not np.isfinite(edge) or edge < tactical_edge_floor)
        ):
            floor_tokens.append(
                f"tactical_edge_below:{_gate_num(edge)}<{_gate_num(tactical_edge_floor)}"
            )

        if strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
            tactical_width = fnum(row.get("width_live"))
            if not np.isfinite(tactical_width):
                tactical_width = fnum(row.get("width"))
            tactical_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(tactical_net):
                tactical_net = fnum(row.get("live_net_mark"))
            if np.isfinite(tactical_width) and tactical_width > 0 and np.isfinite(tactical_net):
                tactical_debit_pct = tactical_net / tactical_width
                if np.isfinite(tactical_max_debit_pct_width):
                    if tactical_debit_pct > tactical_max_debit_pct_width + tactical_debit_pct_width_tolerance:
                        floor_tokens.append(
                            "tactical_debit_pct_width_high:"
                            f"{_gate_num(tactical_debit_pct)}>{_gate_num(tactical_max_debit_pct_width)}"
                        )
                    elif tactical_debit_pct > tactical_max_debit_pct_width:
                        floor_tokens.append(
                            "tactical_debit_pct_width_near_limit:"
                            f"{_gate_num(tactical_debit_pct)}>{_gate_num(tactical_max_debit_pct_width)}"
                        )

        signals = fnum(row.get("signals"))
        if (
            (not shield_override_live)
            and (not evidence_lane)
            and (not ic_income_constructive)
            and np.isfinite(tactical_min_signals)
        ):
            tactical_signals_floor = max(
                0.0,
                tactical_min_signals - tactical_min_signals_grace,
            )
            if not np.isfinite(signals) or signals < tactical_signals_floor:
                floor_tokens.append(
                    f"tactical_signals_below:{_gate_num(signals)}<{_gate_num(tactical_min_signals)}"
                )
            elif signals < tactical_min_signals:
                floor_tokens.append(
                    f"tactical_signals_near_limit:{_gate_num(signals)}<{_gate_num(tactical_min_signals)}"
                )

        verdict = str(row.get("verdict", "")).strip().upper()
        if (
            (not shield_override_live)
            and (not ic_income_constructive)
            and tactical_require_verdict_pass
            and verdict != "PASS"
        ):
            floor_tokens.append(f"tactical_verdict_not_pass:{verdict or 'UNKNOWN'}")

        return ";".join(floor_tokens)

    mdf["book_blockers"] = mdf.apply(tactical_floor_blockers, axis=1)
    book_blocker_mask = mdf["book_blockers"].astype(str).str.len() > 0
    if book_blocker_mask.any():
        mdf.loc[book_blocker_mask, "approval_blockers"] = mdf.loc[book_blocker_mask].apply(
            lambda row: _append_semicolon_tokens(row.get("approval_blockers", ""), row.get("book_blockers", "")),
            axis=1,
        )
        mdf.loc[book_blocker_mask, "quality_blockers"] = mdf.loc[book_blocker_mask].apply(
            lambda row: _append_semicolon_tokens(row.get("quality_blockers", ""), row.get("book_blockers", "")),
            axis=1,
        )
    mdf["research_book"] = mdf["execution_book"]
    if not pilot_live_entry_enabled:
        pilot_mask = mdf["execution_book"].astype(str).eq("Pilot")
        if pilot_mask.any():
            mdf.loc[pilot_mask, "book_blockers"] = mdf.loc[pilot_mask].apply(
                lambda row: _append_semicolon_tokens(row.get("book_blockers", ""), "pilot_research_only"),
                axis=1,
            )
            mdf.loc[pilot_mask, "approval_blockers"] = mdf.loc[pilot_mask].apply(
                lambda row: _append_semicolon_tokens(row.get("approval_blockers", ""), "pilot_research_only"),
                axis=1,
            )
            mdf.loc[pilot_mask, "hard_blockers"] = mdf.loc[pilot_mask].apply(
                lambda row: _append_semicolon_tokens(row.get("hard_blockers", ""), "pilot_research_only"),
                axis=1,
            )
            mdf.loc[pilot_mask, "execution_book"] = "Watch"
    if not scout_live_entry_enabled:
        scout_mask = mdf["execution_book"].astype(str).eq("Scout")
        if scout_mask.any():
            mdf.loc[scout_mask, "book_blockers"] = mdf.loc[scout_mask].apply(
                lambda row: _append_semicolon_tokens(row.get("book_blockers", ""), "scout_research_only"),
                axis=1,
            )
            mdf.loc[scout_mask, "approval_blockers"] = mdf.loc[scout_mask].apply(
                lambda row: _append_semicolon_tokens(row.get("approval_blockers", ""), "scout_research_only"),
                axis=1,
            )
            mdf.loc[scout_mask, "hard_blockers"] = mdf.loc[scout_mask].apply(
                lambda row: _append_semicolon_tokens(row.get("hard_blockers", ""), "scout_research_only"),
                axis=1,
            )
            mdf.loc[scout_mask, "execution_book"] = "Watch"

    mdf["pre_profit_safety_book"] = mdf["execution_book"].astype(str)

    def _is_live_price_ok(row) -> bool:
        live_status = str(row.get("live_status", "") or "").strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        return ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)

    def _approval_profit_safety_blockers(row) -> str:
        book = str(row.get("execution_book", "") or "").strip()
        if book not in {"Core", "Tactical", "Medium", "Income"}:
            return ""
        strategy_local = str(row.get("strategy", "") or "").strip()
        if bool(row.get("regime_weekly_lane", False)):
            return ""
        blockers = []
        conf_score = fnum(row.get("confidence_score"))
        edge_score_val = fnum(row.get("edge_score"))
        edge_pct_val = fnum(row.get("edge_pct"))
        pop_val = normalize_probability(row.get("hist_success_pct"))
        contract_flow = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
        flow_conf = str(row.get("flow_confirmation", "") or "").strip().lower()
        gex_regime = str(row.get("gex_regime", "") or "").strip().lower()
        gex_ctx = str(row.get("gex_wall_context", "") or "").strip().lower()
        regime = str(row.get("approval_regime", "") or "").strip().lower()
        signals_val = fnum(row.get("signals"))
        be_dist = abs(fnum(row.get("breakeven_distance_pct")))
        clean_flow_or_technical = (
            contract_flow == "confirmed"
            or flow_conf == "confirmed"
            or gex_ctx in {"pinned_supportive_below_call_wall", "pinned_supportive_above_put_wall", "pinned_income_constructive", "volatile_breakout_possible"}
        )
        if book in {"Core", "Tactical"}:
            if np.isfinite(high_enter_min_confidence_score) and (not np.isfinite(conf_score) or conf_score < high_enter_min_confidence_score):
                blockers.append(f"confidence_below_high_enter:{conf_score if np.isfinite(conf_score) else 'missing'}<{high_enter_min_confidence_score:g}")
            if np.isfinite(high_enter_min_pop) and (not np.isfinite(pop_val) or pop_val < high_enter_min_pop):
                blockers.append(f"pop_below_high_enter:{pop_val if np.isfinite(pop_val) else 'missing'}<{high_enter_min_pop:.0%}")
            edge_floor = high_enter_min_edge_pct if np.isfinite(high_enter_min_edge_pct) else 0.0
            if high_enter_require_positive_edge and (
                not np.isfinite(edge_pct_val)
                or edge_pct_val <= 0
                or (edge_floor > 0 and edge_pct_val < edge_floor)
            ):
                if edge_floor > 0:
                    blockers.append(f"edge_below_high_enter:{edge_pct_val if np.isfinite(edge_pct_val) else 'missing'}<{edge_floor:g}")
                else:
                    blockers.append("edge_not_positive_for_high_enter")
            if high_enter_require_clean_flow_or_technical and not clean_flow_or_technical:
                blockers.append("no_clean_flow_or_technical_edge")
        if book == "Medium" and (
            bool(row.get("quant_edge_lane", False)) or bool(row.get("short_dte_edge_lane", False))
        ):
            pass
        elif book == "Medium":
            if np.isfinite(medium_review_min_confidence_score) and (not np.isfinite(conf_score) or conf_score < medium_review_min_confidence_score):
                blockers.append(f"medium_confidence_below:{conf_score if np.isfinite(conf_score) else 'missing'}<{medium_review_min_confidence_score:g}")
            if np.isfinite(medium_review_min_pop) and (not np.isfinite(pop_val) or pop_val < medium_review_min_pop):
                blockers.append(f"medium_pop_below:{pop_val if np.isfinite(pop_val) else 'missing'}<{medium_review_min_pop:.0%}")
            if not _is_live_price_ok(row):
                blockers.append("medium_live_price_not_passed")
            if medium_block_earnings_risk and not _earnings_clean(row):
                blockers.append("medium_earnings_risk")
            if np.isfinite(medium_review_min_edge_score) and (not np.isfinite(edge_score_val) or edge_score_val < medium_review_min_edge_score):
                blockers.append("medium_no_explicit_edge")
        if approval_regime_rules_enabled:
            if regime == "mid/down" and strategy_local == "Bull Call Debit" and approval_mid_down_block_bull_calls:
                blockers.append("regime_mid_down_blocks_bull_call")
            if regime == "mid/down" and strategy_local == "Bear Put Debit":
                if flow_conf != "confirmed" or contract_flow != "confirmed":
                    blockers.append("regime_mid_down_bear_put_needs_confirmed_flow")
                if np.isfinite(approval_mid_down_bear_put_min_pop) and (not np.isfinite(pop_val) or pop_val < approval_mid_down_bear_put_min_pop):
                    blockers.append(f"regime_mid_down_bear_put_pop_below:{approval_mid_down_bear_put_min_pop:.0%}")
            if regime == "low/range" and strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
                if np.isfinite(approval_low_range_debit_max_be_distance_pct) and (not np.isfinite(be_dist) or be_dist > approval_low_range_debit_max_be_distance_pct):
                    blockers.append(f"regime_low_range_debit_be_too_far:{be_dist if np.isfinite(be_dist) else 'missing'}>{approval_low_range_debit_max_be_distance_pct:.1%}")
            if regime == "mid/range" and strategy_local in {"Iron Condor", "Iron Butterfly", "Bear Call Credit", "Bull Put Credit"}:
                if gex_regime != "pinned" and "supportive" not in gex_ctx and "income_constructive" not in gex_ctx:
                    blockers.append("regime_mid_range_income_needs_pinned_gex_or_wall")
            if regime == "high/down":
                if np.isfinite(approval_high_down_min_signals) and (not np.isfinite(signals_val) or signals_val < approval_high_down_min_signals):
                    blockers.append(f"regime_high_down_liquidity_below:{signals_val if np.isfinite(signals_val) else 'missing'}<{approval_high_down_min_signals:g}")
            if regime == "high/range" and strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
                high_range_full_size_block = (
                    book in {"Core", "Tactical"}
                    and (
                        contract_flow != "confirmed"
                        or not np.isfinite(conf_score)
                        or conf_score < approval_high_range_breakout_min_confidence_score
                        or not np.isfinite(edge_pct_val)
                        or edge_pct_val < approval_high_range_breakout_min_edge_pct
                    )
                )
                high_range_medium_block = (
                    book == "Medium"
                    and (
                        contract_flow != "confirmed"
                        or not np.isfinite(conf_score)
                        or conf_score < medium_review_min_confidence_score
                        or not np.isfinite(edge_pct_val)
                        or edge_pct_val < medium_min_edge_pct
                    )
                )
                if high_range_full_size_block or high_range_medium_block:
                    blockers.append("regime_high_range_blocks_directional_without_breakout")
        return ";".join(blockers)

    mdf["profit_safety_blockers"] = mdf.apply(_approval_profit_safety_blockers, axis=1)
    def _profit_safety_approval_blockers(text) -> str:
        review_only_prefixes = ("confidence_below_high_enter:",)
        tokens = []
        for token in str(text or "").split(";"):
            token = token.strip()
            if not token:
                continue
            if any(token.startswith(prefix) for prefix in review_only_prefixes):
                continue
            tokens.append(token)
        return ";".join(tokens)

    mdf["profit_safety_approval_blockers"] = mdf["profit_safety_blockers"].apply(_profit_safety_approval_blockers)
    profit_safety_approval_mask = mdf["profit_safety_approval_blockers"].fillna("").astype(str).str.len() > 0
    if profit_safety_approval_mask.any():
        mdf.loc[profit_safety_approval_mask, "approval_blockers"] = mdf.loc[profit_safety_approval_mask].apply(
            lambda row: _append_semicolon_tokens(row.get("approval_blockers", ""), row.get("profit_safety_approval_blockers", "")),
            axis=1,
        )
        mdf.loc[profit_safety_approval_mask, "hard_blockers"] = mdf.loc[profit_safety_approval_mask].apply(
            lambda row: _append_semicolon_tokens(row.get("hard_blockers", ""), row.get("profit_safety_approval_blockers", "")),
            axis=1,
        )
        mdf.loc[profit_safety_approval_mask, "execution_book"] = "Watch"

    def _missing_data_flags(row) -> str:
        flags = []
        if not str(row.get("gex_regime", "") or "").strip():
            flags.append("UW_GEX")
        if str(row.get("live_status", "") or "").strip() in {"chain_error", "chain_not_success", "missing_underlying_quote", "missing_live_quote", "missing_leg_in_live_chain"}:
            flags.append("Schwab")
        if not _earnings_state(row):
            flags.append("earnings")
        if not str(row.get("market_regime", "") or "").strip():
            flags.append("macro")
        return ";".join(flags)

    def _qualified_candidate(row) -> bool:
        if not enable_qualified_book:
            return False
        if str(row.get("execution_book", "") or "").strip() != "Watch":
            return False
        conf_score = fnum(row.get("confidence_score"))
        edge_score_val = fnum(row.get("edge_score"))
        edge_pct_val = fnum(row.get("edge_pct"))
        verdict = str(row.get("verdict", "") or "").strip().upper()
        hard = str(row.get("hard_blockers", "") or "").lower()
        severe = any(x in hard for x in ["contra", "earnings_risk", "liquidity", "debit_above", "credit_below", "pop_below_high_enter"])
        quality_signal = (
            (np.isfinite(conf_score) and conf_score >= qualified_min_confidence_score)
            or (np.isfinite(edge_score_val) and edge_score_val >= qualified_min_edge_score)
            or (np.isfinite(edge_pct_val) and edge_pct_val > 0 and verdict in {"PASS", "LOW_SAMPLE"})
        )
        return bool(quality_signal and not severe)

    def _qualified_reason(row) -> str:
        if not bool(row.get("qualified_candidate", False)):
            return ""
        parts = []
        if str(row.get("pre_profit_safety_book", "") or "") in {"Core", "Tactical", "Medium", "Income"}:
            parts.append(f"downgraded_from_{row.get('pre_profit_safety_book')}")
        if str(row.get("missing_data_flags", "") or ""):
            parts.append(f"missing_data:{row.get('missing_data_flags')}")
        hard_txt = str(row.get("hard_blockers", "") or "")
        if "shield_delta_fail" in hard_txt:
            parts.append("reject_or_restrike:short_delta_too_close")
        if str(row.get("profit_safety_blockers", "") or ""):
            parts.append(f"needs:{row.get('profit_safety_blockers')}")
        if str(row.get("edge_sources", "") or ""):
            parts.append(f"edges:{row.get('edge_sources')}")
        return "; ".join(parts[:4])

    mdf["missing_data_flags"] = mdf.apply(_missing_data_flags, axis=1)
    mdf["qualified_candidate"] = mdf.apply(_qualified_candidate, axis=1)
    mdf["qualified_reason"] = mdf.apply(_qualified_reason, axis=1)

    post_short_mask = pd.Series(False, index=mdf.index)
    if enable_short_dte_edge_book:
        post_short_mask = mdf.apply(
            lambda row: bool(
                str(row.get("execution_book", "") or "").strip() == "Watch"
                and (not short_dte_edge_require_qualified_candidate or bool(row.get("qualified_candidate", False)))
                and not str(row.get("profit_safety_approval_blockers", "") or "").strip()
                and _short_dte_edge_base_checks(row)
            ),
            axis=1,
        )
    if post_short_mask.any():
        short_rows = mdf.loc[post_short_mask].copy()
        short_rows["_short_dte_conf"] = pd.to_numeric(short_rows.get("confidence_score"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_edge"] = pd.to_numeric(short_rows.get("edge_pct"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_signals"] = pd.to_numeric(short_rows.get("signals"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_conviction"] = pd.to_numeric(short_rows.get("conviction"), errors="coerce").fillna(-1e9)
        short_rows["_short_dte_hist"] = pd.to_numeric(short_rows.get("hist_success_pct"), errors="coerce").fillna(-1e9)
        short_rows = short_rows.sort_values(
            ["_short_dte_conf", "_short_dte_edge", "_short_dte_signals", "_short_dte_conviction", "_short_dte_hist"],
            ascending=[False, False, False, False, False],
        )
        keep_short = []
        per_short_ticker = defaultdict(int)
        for sidx, srow in short_rows.iterrows():
            sticker = str(srow.get("ticker", "") or "").strip().upper()
            if short_dte_edge_max_per_ticker_day > 0 and per_short_ticker[sticker] >= short_dte_edge_max_per_ticker_day:
                continue
            if short_dte_edge_max_per_day > 0 and len(keep_short) >= short_dte_edge_max_per_day:
                break
            keep_short.append(sidx)
            per_short_ticker[sticker] += 1
        drop_short = sorted(set(short_rows.index) - set(keep_short))
        if drop_short:
            mdf.loc[drop_short, "book_blockers"] = mdf.loc[drop_short].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("book_blockers", ""),
                    f"short_dte_edge_daily_cap:{len(keep_short)}/{short_dte_edge_max_per_day}",
                ),
                axis=1,
            )
        if keep_short:
            mdf.loc[keep_short, "short_dte_edge_lane"] = True
            mdf.loc[keep_short, "execution_book"] = "Medium"
            mdf.loc[keep_short, "research_book"] = "Medium"
            mdf.loc[keep_short, "qualified_candidate"] = False
            mdf.loc[keep_short, "qualified_reason"] = mdf.loc[keep_short].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("qualified_reason", ""),
                    "promoted_short_dte_edge",
                ),
                axis=1,
            )

    post_high_signal_mask = pd.Series(False, index=mdf.index)
    if enable_high_signal_edge_book:
        post_high_signal_mask = mdf.apply(
            lambda row: bool(
                str(row.get("execution_book", "") or "").strip() == "Watch"
                and (not high_signal_edge_require_qualified_candidate or bool(row.get("qualified_candidate", False)))
                and _high_signal_edge_base_checks(row)
            ),
            axis=1,
        )
    if post_high_signal_mask.any():
        high_signal_rows = mdf.loc[post_high_signal_mask].copy()
        high_signal_rows["_high_signal_signals"] = pd.to_numeric(
            high_signal_rows.get("signals"), errors="coerce"
        ).fillna(-1e9)
        high_signal_rows["_high_signal_edge"] = pd.to_numeric(
            high_signal_rows.get("edge_pct"), errors="coerce"
        ).fillna(-1e9)
        high_signal_rows["_high_signal_conf"] = pd.to_numeric(
            high_signal_rows.get("confidence_score"), errors="coerce"
        ).fillna(-1e9)
        high_signal_rows["_high_signal_conviction"] = pd.to_numeric(
            high_signal_rows.get("conviction"), errors="coerce"
        ).fillna(-1e9)
        high_signal_rows["_high_signal_hist"] = pd.to_numeric(
            high_signal_rows.get("hist_success_pct"), errors="coerce"
        ).fillna(-1e9)
        high_signal_rows = high_signal_rows.sort_values(
            [
                "_high_signal_signals",
                "_high_signal_edge",
                "_high_signal_conf",
                "_high_signal_conviction",
                "_high_signal_hist",
            ],
            ascending=[False, False, False, False, False],
        )
        keep_high_signal = []
        per_high_signal_ticker = defaultdict(int)
        for hidx, hrow in high_signal_rows.iterrows():
            hticker = str(hrow.get("ticker", "") or "").strip().upper()
            if (
                high_signal_edge_max_per_ticker_day > 0
                and per_high_signal_ticker[hticker] >= high_signal_edge_max_per_ticker_day
            ):
                continue
            if high_signal_edge_max_per_day > 0 and len(keep_high_signal) >= high_signal_edge_max_per_day:
                break
            keep_high_signal.append(hidx)
            per_high_signal_ticker[hticker] += 1
        drop_high_signal = sorted(set(high_signal_rows.index) - set(keep_high_signal))
        if drop_high_signal:
            mdf.loc[drop_high_signal, "book_blockers"] = mdf.loc[drop_high_signal].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("book_blockers", ""),
                    f"high_signal_edge_daily_cap:{len(keep_high_signal)}/{high_signal_edge_max_per_day}",
                ),
                axis=1,
            )
        if keep_high_signal:
            mdf.loc[keep_high_signal, "high_signal_edge_lane"] = True
            mdf.loc[keep_high_signal, "execution_book"] = "Medium"
            mdf.loc[keep_high_signal, "research_book"] = "Medium"
            mdf.loc[keep_high_signal, "qualified_candidate"] = False
            mdf.loc[keep_high_signal, "qualified_reason"] = mdf.loc[keep_high_signal].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("qualified_reason", ""),
                    "promoted_high_signal_edge",
                ),
                axis=1,
            )

    if enable_medium_book:
        high_conf_downgrade_mask = (
            mdf["execution_book"].astype(str).isin(["Core", "Tactical"])
            & mdf["profit_safety_blockers"].fillna("").astype(str).str.contains(
                "confidence_below_high_enter:",
                regex=False,
            )
        )
        if high_conf_downgrade_mask.any():
            mdf.loc[high_conf_downgrade_mask, "execution_book"] = "Medium"
            mdf.loc[high_conf_downgrade_mask, "research_book"] = "Medium"
            mdf.loc[high_conf_downgrade_mask, "book_blockers"] = mdf.loc[high_conf_downgrade_mask].apply(
                lambda row: _append_semicolon_tokens(
                    row.get("book_blockers", ""),
                    "downgraded_to_medium_confidence",
                ),
                axis=1,
            )

    def _trade_fraction_of_width(row) -> float:
        width_val = fnum(row.get("width_live"))
        if not np.isfinite(width_val) or width_val <= 0:
            width_val = fnum(row.get("width"))
        net_val = fnum(row.get("live_net_bid_ask"))
        if not np.isfinite(net_val):
            net_val = fnum(row.get("live_net_mark"))
        if not np.isfinite(net_val):
            net_val = fnum(row.get("net"))
        if not (np.isfinite(width_val) and width_val > 0 and np.isfinite(net_val)):
            return math.nan
        return abs(net_val) / width_val

    def _final_validity_blockers(row) -> str:
        if not final_validity_gate_enabled:
            return ""
        book = str(row.get("execution_book", "") or "").strip()
        if book not in {"Core", "Tactical", "Medium", "Income"}:
            return ""
        tokens = []
        strategy_local = str(row.get("strategy", "") or "").strip()
        net_type = str(row.get("net_type", "") or "").strip().lower()
        verdict = str(row.get("verdict", "") or "").strip().upper()
        pop_val = normalize_probability(row.get("hist_success_pct"))
        conf_score = fnum(row.get("confidence_score"))
        edge_score_val = fnum(row.get("edge_score"))
        edge_pct_val = fnum(row.get("edge_pct"))
        frac_width = _trade_fraction_of_width(row)
        reward_risk = fnum(row.get("live_reward_risk"))
        regime_weekly = bool(row.get("regime_weekly_lane", False))

        if regime_weekly:
            if not regime_weekly_base_checks(row):
                tokens.append("final_regime_weekly_gate_failed")
        else:
            if valid_trade_require_likelihood_pass and verdict != "PASS":
                tokens.append(f"final_verdict_not_pass:{verdict or 'UNKNOWN'}")
            if np.isfinite(valid_trade_min_pop) and (not np.isfinite(pop_val) or pop_val < valid_trade_min_pop):
                tokens.append(
                    f"final_pop_below:{pop_val if np.isfinite(pop_val) else 'missing'}<{valid_trade_min_pop:.0%}"
                )
            if (
                np.isfinite(valid_trade_min_confidence_score)
                and (not np.isfinite(conf_score) or conf_score < valid_trade_min_confidence_score)
            ):
                tokens.append(
                    f"final_confidence_below:{conf_score if np.isfinite(conf_score) else 'missing'}<{valid_trade_min_confidence_score:g}"
                )
            if (
                np.isfinite(valid_trade_min_edge_score)
                and (not np.isfinite(edge_score_val) or edge_score_val < valid_trade_min_edge_score)
            ):
                tokens.append("final_no_explicit_edge")
            if valid_trade_require_positive_edge_pct and (not np.isfinite(edge_pct_val) or edge_pct_val <= 0):
                tokens.append("final_edge_pct_not_positive")
        if valid_trade_require_live_price and not _is_live_price_ok(row):
            tokens.append("final_live_price_not_passed")
        if valid_trade_block_earnings_risk and not _earnings_clean(row):
            tokens.append(f"final_earnings_risk:{_earnings_state(row) or 'UNKNOWN'}")
        if valid_trade_block_liquidity_flags:
            blocker_text = ";".join(
                str(row.get(col, "") or "").lower()
                for col in [
                    "hard_blockers",
                    "quality_blockers",
                    "approval_blockers",
                    "missing_data_flags",
                    "live_status",
                ]
            )
            liquidity_needles = (
                "liquidity",
                "missing_leg",
                "missing_live",
                "missing_underlying_quote",
                "spread_pct",
                "oi_below",
                "volume_below",
            )
            if any(needle in blocker_text for needle in liquidity_needles):
                tokens.append("final_liquidity_blocker")
        if net_type == "credit" or strategy_local in {"Iron Condor", "Iron Butterfly", "Bull Put Credit", "Bear Call Credit"}:
            if (
                np.isfinite(valid_trade_min_credit_pct_width)
                and (not np.isfinite(frac_width) or frac_width < valid_trade_min_credit_pct_width)
            ):
                tokens.append(
                    f"final_credit_width_below:{_gate_num(frac_width)}<{_gate_num(valid_trade_min_credit_pct_width)}"
                )
            if (
                np.isfinite(valid_trade_max_credit_pct_width)
                and np.isfinite(frac_width)
                and frac_width > valid_trade_max_credit_pct_width
            ):
                tokens.append(
                    f"final_credit_width_above:{_gate_num(frac_width)}>{_gate_num(valid_trade_max_credit_pct_width)}"
                )
        elif net_type == "debit" or strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
            if (
                np.isfinite(valid_trade_max_debit_pct_width)
                and (not np.isfinite(frac_width) or frac_width > valid_trade_max_debit_pct_width)
            ):
                tokens.append(
                    f"final_debit_width_above:{_gate_num(frac_width)}>{_gate_num(valid_trade_max_debit_pct_width)}"
                )
            if (
                np.isfinite(valid_trade_min_debit_reward_risk)
                and (not np.isfinite(reward_risk) or reward_risk < valid_trade_min_debit_reward_risk)
            ):
                tokens.append(
                    f"final_debit_rr_below:{_gate_num(reward_risk)}<{_gate_num(valid_trade_min_debit_reward_risk)}"
                )
        return ";".join(tokens)

    mdf["final_validity_blockers"] = mdf.apply(_final_validity_blockers, axis=1)
    final_validity_mask = mdf["final_validity_blockers"].fillna("").astype(str).str.len() > 0
    if final_validity_mask.any():
        mdf.loc[final_validity_mask, "approval_blockers"] = mdf.loc[final_validity_mask].apply(
            lambda row: _append_semicolon_tokens(row.get("approval_blockers", ""), row.get("final_validity_blockers", "")),
            axis=1,
        )
        mdf.loc[final_validity_mask, "hard_blockers"] = mdf.loc[final_validity_mask].apply(
            lambda row: _append_semicolon_tokens(row.get("hard_blockers", ""), row.get("final_validity_blockers", "")),
            axis=1,
        )
        mdf.loc[final_validity_mask, "execution_book"] = "Watch"
        mdf.loc[final_validity_mask, "research_book"] = "Watch"
        mdf.loc[final_validity_mask, "qualified_candidate"] = False

    mdf["blocker_missing_data"] = mdf["missing_data_flags"]
    mdf["blocker_final_approval"] = mdf.apply(
        lambda row: _append_semicolon_tokens(
            row.get("profit_safety_approval_blockers", ""),
            row.get("final_validity_blockers", ""),
        ),
        axis=1,
    )
    mdf["blocker_hard"] = mdf["hard_blockers"]
    mdf["blocker_risk_warning"] = mdf["quality_blockers"]
    def _reason_categories(row) -> str:
        txt = ";".join(str(row.get(c, "") or "") for c in ["approval_blockers", "hard_blockers", "quality_blockers", "profit_safety_blockers", "missing_data_flags"]).lower()
        cats = []
        checks = {
            "credit": ["credit", "no_touch"],
            "POP": ["pop", "likelihood", "sample"],
            "earnings": ["earn", "er-risk"],
            "flow": ["flow", "contra"],
            "liquidity": ["liquidity", "signals", "oi", "spread"],
            "regime": ["regime", "vix", "spy"],
            "GEX": ["gex", "wall"],
            "concentration": ["portfolio", "concentration", "overlap"],
        }
        for cat, needles in checks.items():
            if any(n in txt for n in needles):
                cats.append(cat)
        return ";".join(cats)
    mdf["reason_categories"] = mdf.apply(_reason_categories, axis=1)
    def _high_enter_ready(row) -> bool:
        book = str(row.get("execution_book", "") or "").strip()
        if book not in {"Core", "Tactical"}:
            return False
        return not bool(str(row.get("profit_safety_blockers", "") or "").strip())
    mdf["high_enter_ready"] = mdf.apply(_high_enter_ready, axis=1)
    mdf["high_enter_reason"] = mdf.apply(lambda row: "High ENTER guard passed." if bool(row.get("high_enter_ready", False)) else str(row.get("profit_safety_blockers", "") or ""), axis=1)

    approved_books = ["Core", "Tactical"]
    if enable_medium_book:
        approved_books.append("Medium")
    if enable_income_book:
        approved_books.append("Income")
    if pilot_live_entry_enabled:
        approved_books.append("Pilot")
    if scout_live_entry_enabled:
        approved_books.append("Scout")
    mdf["approved"] = mdf["execution_book"].isin(approved_books)
    mdf["size_mult"] = mdf["execution_book"].map({
        "Core": core_size_mult,
        "Tactical": tactical_size_mult,
        "Medium": medium_size_mult,
        "Income": income_size_mult,
        "Pilot": pilot_size_mult,
        "Scout": scout_size_mult,
    })
    quant_edge_approved_mask = (
        mdf.get("quant_edge_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
        & mdf["execution_book"].astype(str).eq("Medium")
    )
    if quant_edge_approved_mask.any() and np.isfinite(quant_edge_size_mult) and quant_edge_size_mult > 0:
        mdf.loc[quant_edge_approved_mask, "size_mult"] = quant_edge_size_mult
    short_dte_edge_approved_mask = (
        mdf.get("short_dte_edge_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
        & mdf["execution_book"].astype(str).eq("Medium")
    )
    if short_dte_edge_approved_mask.any() and np.isfinite(short_dte_edge_size_mult) and short_dte_edge_size_mult > 0:
        mdf.loc[short_dte_edge_approved_mask, "size_mult"] = short_dte_edge_size_mult
    high_signal_edge_approved_mask = (
        mdf.get("high_signal_edge_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
        & mdf["execution_book"].astype(str).eq("Medium")
    )
    if (
        high_signal_edge_approved_mask.any()
        and np.isfinite(high_signal_edge_size_mult)
        and high_signal_edge_size_mult > 0
    ):
        mdf.loc[high_signal_edge_approved_mask, "size_mult"] = high_signal_edge_size_mult
    regime_weekly_approved_mask = (
        mdf.get("regime_weekly_lane", pd.Series(False, index=mdf.index)).fillna(False).astype(bool)
        & mdf["execution_book"].astype(str).eq("Tactical")
    )
    if (
        regime_weekly_approved_mask.any()
        and np.isfinite(regime_weekly_size_mult)
        and regime_weekly_size_mult > 0
    ):
        mdf.loc[regime_weekly_approved_mask, "size_mult"] = regime_weekly_size_mult
    mdf["_book_rank"] = mdf["execution_book"].map({"Core": 0, "Tactical": 1, "Medium": 2, "Income": 3, "Pilot": 4, "Scout": 5, "Watch": 6}).fillna(9).astype(int)
    mdf["_edge_sort"] = pd.to_numeric(mdf.get("edge_pct"), errors="coerce").fillna(-1e9)
    mdf = (
        mdf.sort_values(
            ["approved", "_book_rank", "_ev_sort", "_edge_sort", "conviction"],
            ascending=[False, True, False, False, False],
        )
        .drop(columns=["_ev_sort", "_edge_sort", "_book_rank"])
        .reset_index(drop=True)
    )
    mdf["portfolio_cap_pass"] = pd.Series([pd.NA] * len(mdf), dtype="boolean")
    mdf["portfolio_cap_reason"] = ""
    portfolio_guard_status = "disabled_historical_replay_no_snapshot" if args.historical_replay else "disabled"
    portfolio_guard_error = ""
    portfolio_guard_snapshot_csv = ""
    portfolio_guard_base = {}
    if enforce_pretrade_caps:
        portfolio_guard_status = "enabled"
        open_pos_csv = None
        if pretrade_open_positions_csv:
            candidate = Path(pretrade_open_positions_csv).expanduser().resolve()
            if candidate.exists():
                open_pos_csv = candidate
            else:
                portfolio_guard_error = f"configured_open_positions_csv_missing:{candidate}"
        if open_pos_csv is None:
            cache_csv = out_dir / "open_positions_from_schwab.csv"
            ok_fetch, fetch_err = fetch_open_positions_from_schwab(cache_csv)
            if ok_fetch and cache_csv.exists():
                open_pos_csv = cache_csv
            else:
                if portfolio_guard_error:
                    portfolio_guard_error += f" | {fetch_err}"
                else:
                    portfolio_guard_error = fetch_err
        if open_pos_csv is not None and open_pos_csv.exists():
            portfolio_guard_snapshot_csv = str(open_pos_csv)
            book = build_portfolio_risk_book(open_pos_csv)
            if not bool(book.get("ok")):
                portfolio_guard_error = str(book.get("error", "portfolio_risk_book_failed"))
            else:
                portfolio_guard_base = dict(book)
                total_risk = float(book.get("total_risk", 0.0))
                short_put_risk = float(book.get("short_put_risk", 0.0))
                symbol_risk = {str(k): float(v) for k, v in (book.get("symbol_risk", {}) or {}).items()}
                sector_risk = defaultdict(float)
                for _sym, _risk in symbol_risk.items():
                    sector_risk[sector_map.get(str(_sym).upper().strip(), "Unknown")] += float(_risk)
                mega_cap_tech_tickers = {"AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA", "AVGO", "AMD", "NFLX"}
                option_underlyings = {
                    str(x).upper().strip()
                    for x in (book.get("option_underlyings", []) or [])
                    if str(x).strip()
                }
                short_put_expiry_risk = {
                    str(k): float(v) for k, v in (book.get("short_put_expiry_risk", {}) or {}).items()
                }
                for idx, row in mdf.iterrows():
                    if not bool(row.get("approved")):
                        continue
                    strategy = str(row.get("strategy", "")).strip()
                    ticker = str(row.get("ticker", "")).strip().upper()
                    expiry = str(row.get("expiry", ""))[:10]
                    add_risk = fnum(row.get("live_max_loss"))
                    if not np.isfinite(add_risk) or add_risk <= 0:
                        gate_val = fnum(row.get("gate_target"))
                        width_val = fnum(row.get("width"))
                        net_type_val = str(row.get("net_type", "")).strip().lower()
                        if np.isfinite(gate_val) and np.isfinite(width_val):
                            _, tgt_max_loss = calc_target_max(net_type_val, width_val, gate_val)
                            add_risk = fnum(tgt_max_loss)
                    if not np.isfinite(add_risk) or add_risk <= 0:
                        mdf.at[idx, "approved"] = False
                        mdf.at[idx, "execution_book"] = "Watch"
                        mdf.at[idx, "size_mult"] = float("nan")
                        mdf.at[idx, "portfolio_cap_pass"] = False
                        mdf.at[idx, "portfolio_cap_reason"] = "missing_trade_risk"
                        continue

                    projected_total = total_risk + add_risk
                    projected_symbol = symbol_risk.get(ticker, 0.0) + add_risk
                    projected_symbol_share = (
                        projected_symbol / projected_total if projected_total > 0 else 0.0
                    )
                    sector_name = sector_map.get(ticker, "Unknown")
                    projected_sector = float(sector_risk.get(sector_name, 0.0)) + add_risk
                    projected_sector_share = (
                        projected_sector / projected_total if projected_total > 0 else 0.0
                    )
                    reasons = []
                    if block_same_underlying_option_overlap and ticker in option_underlyings:
                        reasons.append(f"existing_option_exposure:{ticker}")
                    if projected_symbol_share > symbol_limit:
                        reasons.append(
                            f"symbol_share {projected_symbol_share:.1%} > {symbol_limit:.1%} ({ticker})"
                        )
                    if projected_sector_share > max_sector_share:
                        reasons.append(
                            f"sector_share {projected_sector_share:.1%} > {max_sector_share:.1%} ({sector_name})"
                        )
                    if ticker in mega_cap_tech_tickers and projected_sector_share > max_sector_share * 0.80:
                        reasons.append(
                            f"mega_cap_tech_cluster {projected_sector_share:.1%} near sector cap ({sector_name})"
                        )

                    if candidate_uses_short_put_risk(strategy):
                        projected_short_put = short_put_risk + add_risk
                        projected_short_put_share = (
                            projected_short_put / projected_total if projected_total > 0 else 0.0
                        )
                        if projected_short_put_share > short_put_limit:
                            reasons.append(
                                f"short_put_share {projected_short_put_share:.1%} > {short_put_limit:.1%}"
                            )
                        projected_expiry = short_put_expiry_risk.get(expiry, 0.0) + add_risk
                        projected_expiry_share = (
                            projected_expiry / projected_short_put if projected_short_put > 0 else 0.0
                        )
                        if projected_expiry_share > expiry_limit:
                            reasons.append(
                                f"short_put_expiry_share {projected_expiry_share:.1%} > {expiry_limit:.1%} ({expiry})"
                            )

                    if reasons:
                        mdf.at[idx, "approved"] = False
                        mdf.at[idx, "execution_book"] = "Watch"
                        mdf.at[idx, "size_mult"] = float("nan")
                        mdf.at[idx, "portfolio_cap_pass"] = False
                        mdf.at[idx, "portfolio_cap_reason"] = "; ".join(reasons)
                        mdf.at[idx, "approval_blockers"] = _append_semicolon_tokens(
                            row.get("approval_blockers", ""),
                            "portfolio_cap_breach",
                        )
                        mdf.at[idx, "hard_blockers"] = _append_semicolon_tokens(
                            row.get("hard_blockers", ""),
                            "portfolio_cap_breach",
                        )
                    else:
                        mdf.at[idx, "portfolio_cap_pass"] = True
                        total_risk = projected_total
                        symbol_risk[ticker] = projected_symbol
                        sector_risk[sector_name] = projected_sector
                        if candidate_uses_short_put_risk(strategy):
                            short_put_risk = short_put_risk + add_risk
                            short_put_expiry_risk[expiry] = short_put_expiry_risk.get(expiry, 0.0) + add_risk
        if not portfolio_guard_snapshot_csv and pretrade_caps_require_data:
            mdf.loc[mdf["approved"] == True, "approved"] = False
            mdf.loc[mdf["portfolio_cap_reason"].astype(str).eq(""), "portfolio_cap_reason"] = (
                "pretrade_caps_data_unavailable"
            )
            mdf.loc[mdf["portfolio_cap_pass"].isna(), "portfolio_cap_pass"] = False
    if exclude_debit_moneyness_fail_from_output and not mdf.empty:
        moneyness_fail_mask = mdf["approval_blockers"].astype(str).str.contains(
            r"bull_call_otm_too_far|bear_put_otm_too_far",
            case=False,
            regex=True,
        )
        moneyness_fail_mask = moneyness_fail_mask & ~mdf["execution_book"].astype(str).eq("Pilot")
        if moneyness_fail_mask.any():
            mdf = mdf.loc[~moneyness_fail_mask].copy()
    merged_rows_pre_filter = int(len(mdf))

    def _token_count(value: object) -> int:
        return len([x for x in str(value or "").split(";") if str(x).strip()])

    def _display_rank_score(row) -> float:
        score = 0.0
        if bool(row.get("approved")):
            score += 100000.0
        book = str(row.get("execution_book", "")).strip()
        score += {"Core": 3000.0, "Tactical": 2000.0, "Medium": 1500.0, "Income": 1400.0, "Watch": 0.0}.get(book, 0.0)
        live_status = str(row.get("live_status", "")).strip()
        if bool(row.get("is_final_live_valid")) or live_status == "ok_live":
            score += 300.0
        verdict = str(row.get("verdict", "")).strip().upper()
        if verdict == "PASS":
            score += 450.0
        elif verdict == "LOW_SAMPLE":
            score += 50.0
        elif verdict == "FAIL":
            score -= 450.0
        edge_val = fnum(row.get("edge_pct"))
        if np.isfinite(edge_val):
            score += edge_val * 10.0
        signals_val = fnum(row.get("signals"))
        if np.isfinite(signals_val):
            score += min(signals_val, 300.0)
        conv_val = fnum(row.get("conviction"))
        if np.isfinite(conv_val):
            score += conv_val
        if bool(row.get("stage1_effective")):
            score += 120.0
        else:
            score -= 50.0
        flow_confirm = str(row.get("flow_confirmation", "")).strip().lower()
        if flow_confirm == "confirmed":
            score += 160.0
        elif flow_confirm in {"weak_or_ambiguous", "conflicted"}:
            score -= 35.0
        contract_confirm = str(row.get("contract_flow_confirmation", "")).strip().lower()
        if contract_confirm == "confirmed":
            score += 200.0
        elif contract_confirm == "weak_or_ambiguous":
            score -= 80.0
        elif contract_confirm in {"contra", "directional", "unknown"}:
            score -= 260.0
        gex_ctx = str(row.get("gex_wall_context", "")).strip().lower()
        if gex_ctx == "volatile_breakout_possible":
            score += 60.0
        elif "pinned" in gex_ctx:
            score -= 120.0
        score -= 500.0 * _token_count(row.get("hard_blockers"))
        score -= 75.0 * _token_count(row.get("quality_blockers"))
        return score

    mdf["_display_rank_score"] = mdf.apply(_display_rank_score, axis=1)
    decision_audit_all = mdf.sort_values(
        ["approved", "_display_rank_score", "conviction"],
        ascending=[False, False, False],
    ).reset_index(drop=True).copy()
    mdf = decision_audit_all.copy()
    dropped_final = []
    kept_indices = []
    per_ticker_final = defaultdict(int)
    for idx, row in mdf.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if per_ticker_final[ticker] >= final_max_per_ticker:
            dropped_final.append(
                {
                    "ticker": ticker,
                    "strategy": str(row.get("strategy", "")),
                    "expiry": str(row.get("expiry", ""))[:10],
                    "stage": "final",
                    "drop_reason": "final_max_per_ticker_cap",
                    "details": f"cap={final_max_per_ticker}",
                }
            )
            continue
        per_ticker_final[ticker] += 1
        kept_indices.append(idx)
    mdf = mdf.loc[kept_indices].reset_index(drop=True)
    # --- Track-diversity-aware top-N selection ---
    # Reserve slots for SHIELD trades AND bear-direction FIRE trades so they
    # aren't buried by higher-EV bull calls.  Also enforce max_sector_share.
    _top_n = int(args.top_trades)
    _bear_strategies = {"Bear Put Debit", "Bear Call Credit"}
    _condor_strategies = {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}
    _min_bear_in_output = max(1, int(engine_cfg.get("min_bear_in_output", approval_cfg.get("min_bear_in_output", 2))))
    _min_condor_in_output = max(1, int(engine_cfg.get("min_condor_in_output", approval_cfg.get("min_condor_in_output", 2))))
    if len(mdf) > _top_n:
        _fire_rows = mdf[mdf["track"] == "FIRE"]
        _shield_rows = mdf[mdf["track"] == "SHIELD"]
        _other_rows = mdf[~mdf["track"].isin(["FIRE", "SHIELD"])]

        _bear_fire = _fire_rows[_fire_rows["strategy"].isin(_bear_strategies)]
        _bull_fire = _fire_rows[~_fire_rows["strategy"].isin(_bear_strategies)]
        _condor_rows = mdf[mdf["strategy"].isin(_condor_strategies)]

        # Enforce sector cap, expiry concentration, and direction balance across
        # reserved rows too. Reservations express preference, not cap exemption.
        _selected_rows = []
        _selected_keys = set()
        _sector_counts = defaultdict(int)
        _expiry_counts = defaultdict(int)
        _direction_counts = {"bull": 0, "bear": 0, "neutral": 0}
        _sector_limit = max(1, int(round(max_sector_share * _top_n)))
        _dir_limit = max(1, int(round(max_same_direction_pct * _top_n)))

        def _try_select_final_row(_srow) -> bool:
            if len(_selected_rows) >= _top_n:
                return False
            _row_key = (
                str(_srow.get("ticker", "")).strip().upper(),
                str(_srow.get("strategy", "")).strip(),
                str(_srow.get("expiry", "")).strip()[:10],
                str(_srow.get("long_strike", "")).strip(),
                str(_srow.get("short_strike", "")).strip(),
                str(_srow.get("long_put_strike", "")).strip(),
                str(_srow.get("short_put_strike", "")).strip(),
                str(_srow.get("short_call_strike", "")).strip(),
                str(_srow.get("long_call_strike", "")).strip(),
            )
            if _row_key in _selected_keys:
                return False
            _sticker = str(_srow.get("ticker", "")).strip().upper()
            _ssector = sector_map.get(_sticker, "Unknown")
            if _sector_counts[_ssector] >= _sector_limit:
                return False
            # [T9] Expiry concentration cap
            _sexpiry = str(_srow.get("expiry", _srow.get("expiry_date", ""))).strip()
            if _sexpiry and _expiry_counts[_sexpiry] >= max_same_expiry_count:
                return False
            # [T9] Direction balance cap
            _sstrat = str(_srow.get("strategy", "")).strip()
            _sdir = "neutral" if _sstrat in _condor_strategies else "bear" if _sstrat in {"Bear Put Debit", "Bear Call Credit"} else "bull"
            if _sdir != "neutral" and _direction_counts[_sdir] >= _dir_limit:
                return False
            _sector_counts[_ssector] += 1
            _expiry_counts[_sexpiry] += 1
            _direction_counts[_sdir] += 1
            _selected_keys.add(_row_key)
            _selected_rows.append(_srow.to_dict())
            return True

        def _take_from(_df, _limit):
            _added = 0
            for _, _srow in _df.iterrows():
                if _added >= _limit or len(_selected_rows) >= _top_n:
                    break
                if _try_select_final_row(_srow):
                    _added += 1

        # Pick best SHIELD trades up to the reservation count, subject to caps.
        _take_from(_shield_rows, min(min_shield_in_output, len(_shield_rows)))

        # Reserve best bear FIRE trades so bearish signals always surface, subject to caps.
        _take_from(_bear_fire, min(_min_bear_in_output, len(_bear_fire)))

        # Reserve best condor/income rows so neutral income setups do not vanish
        # behind directional FIRE debit spreads.
        _take_from(_condor_rows, min(_min_condor_in_output, len(_condor_rows)))

        # Fill remaining budget with bull FIRE + leftover bear + leftover SHIELD + other.
        _rest = pd.concat([_bull_fire, _bear_fire, _shield_rows, _other_rows], ignore_index=True)
        _rest = _rest.sort_values(
            ["approved", "_display_rank_score", "conviction"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        _take_from(_rest, _top_n)

        _final = pd.DataFrame(_selected_rows)
        _final = _final.sort_values(
            ["approved", "_display_rank_score", "conviction"],
            ascending=[False, False, False],
        ).reset_index(drop=True).head(_top_n)

        # Record dropped rows.
        _kept_tickers_strats = set(
            zip(_final["ticker"].astype(str).str.upper().str.strip(), _final["strategy"].astype(str).str.strip(), _final["expiry"].astype(str).str[:10])
        )
        for _, row in mdf.iterrows():
            _key = (str(row.get("ticker", "")).strip().upper(), str(row.get("strategy", "")), str(row.get("expiry", ""))[:10])
            if _key not in _kept_tickers_strats:
                dropped_final.append(
                    {
                        "ticker": _key[0],
                        "strategy": _key[1],
                        "expiry": _key[2],
                        "stage": "final",
                        "drop_reason": "final_top_limit",
                        "details": f"top={_top_n}",
                    }
                )
        mdf = _final.reset_index(drop=True)
    else:
        # Even under budget, tag sector for downstream use.
        pass
    inv_close_confirms = fnum(approval_cfg.get("invalidation_close_confirmations", 2))
    inv_close_confirms = int(inv_close_confirms) if np.isfinite(inv_close_confirms) and inv_close_confirms >= 1 else 2

    def live_entry_action(row, approved: bool) -> tuple[str, str]:
        if not approved:
            return "SKIP", "Not approved by daily pipeline."
        if args.historical_replay:
            return "WAIT", "Historical replay only; rerun without --historical-replay for live entry."
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        gate_pass_strict = bool(row.get("gate_pass_strict")) if pd.notna(row.get("gate_pass_strict")) else False
        gate_near_miss = bool(row.get("gate_near_miss")) if pd.notna(row.get("gate_near_miss")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)

        def _target_limit_text() -> str:
            gate_target = fnum(row.get("gate_target"))
            live_net = fnum(row.get("gate_live_net"))
            if not np.isfinite(live_net):
                live_net = fnum(row.get("live_net_bid_ask"))
            net_type_local = str(row.get("net_type", "") or "").strip().lower()
            if not np.isfinite(gate_target):
                return "Target limit required; entry gate unavailable."
            if net_type_local == "credit":
                text = f"Target credit >= {gate_target:.2f}"
            else:
                text = f"Target debit <= {gate_target:.2f}"
            if np.isfinite(live_net):
                text += f"; current {live_net:.2f}"
            return text + "."

        portfolio_cap_pass = row.get("portfolio_cap_pass")
        portfolio_cap_reason = str(row.get("portfolio_cap_reason", "") or "").strip()
        portfolio_review = portfolio_cap_reason and pd.notna(portfolio_cap_pass) and not bool(portfolio_cap_pass)
        gex_block_reason = live_gex_entry_block_reason(row, auto_gex_required)
        if gex_block_reason:
            return "WAIT", gex_block_reason
        execution_book = str(row.get("execution_book", "") or "").strip()
        if (
            execution_book == "Pilot"
            and pilot_enter_near_miss_within_tolerance
            and ok_live
            and gate_near_miss
            and not gate_pass_strict
        ):
            if portfolio_review:
                return "REVIEW", f"Position review required before adding risk: {portfolio_cap_reason}."
            miss_abs = fnum(row.get("gate_miss_abs"))
            tol_total = fnum(row.get("gate_tol_total"))
            tolerance_note = ""
            if np.isfinite(miss_abs) and np.isfinite(tol_total):
                tolerance_note = f" miss {miss_abs:.2f} <= tolerance {tol_total:.2f}"
            return "REVIEW", f"Pilot live quote is inside configured entry tolerance, but Pilot rows are manual-review only and cannot be ENTER ({_target_limit_text().rstrip('.')}{tolerance_note})."
        price_target_required = (
            live_status == "fails_live_entry_gate"
            or (gate_near_miss and not gate_pass_strict)
            or (
                execution_book == "Pilot"
                and ok_live
                and not gate_pass_strict
            )
        )
        if price_target_required:
            reason = _target_limit_text()
            if portfolio_review:
                reason += f" Position review: {portfolio_cap_reason}."
            return "TARGET", reason
        if portfolio_review:
            return "REVIEW", f"Position review required before adding risk: {portfolio_cap_reason}."
        if ok_live:
            confidence_model = str(row.get("confidence_model", "") or "").strip().lower()
            confidence_score = fnum(row.get("confidence_score"))
            hard_blockers = str(row.get("hard_blockers", "") or "").strip()
            profit_safety_blockers = str(row.get("profit_safety_blockers", "") or "").strip()
            quality_blockers = str(row.get("quality_blockers", "") or "").strip()
            approval_blockers = str(row.get("approval_blockers", "") or "").strip()
            blocker_text = ";".join(
                part for part in [hard_blockers, profit_safety_blockers, quality_blockers, approval_blockers] if part
            )
            hist_success_pct = fnum(row.get("hist_success_pct"))
            edge_pct = fnum(row.get("edge_pct"))
            flow_confirmation = str(row.get("flow_confirmation", "") or "").strip().lower()
            contract_flow_confirmation = str(row.get("contract_flow_confirmation", "") or "").strip().lower()
            if execution_book in {"Pilot", "Scout"}:
                return "REVIEW", f"{execution_book} book is manual-review only; use the target gate, do not treat as ENTER."
            if confidence_model == "reject" or (np.isfinite(confidence_score) and confidence_score < 5):
                return "REVIEW", f"Confidence guard blocked ENTER: model={confidence_model or 'unknown'}, score={confidence_score if np.isfinite(confidence_score) else 'unknown'}."
            enter_blockers: list[str] = []
            if hard_blockers:
                enter_blockers.append("hard blocker present")
            if profit_safety_blockers:
                enter_blockers.append("profit-safety blocker present")
            if np.isfinite(hist_success_pct) and hist_success_pct < 60.0:
                enter_blockers.append(f"POP {hist_success_pct:.1f}%<60%")
            if np.isfinite(edge_pct) and edge_pct < 5.0:
                enter_blockers.append(f"edge {edge_pct:.1f}%<5%")
            if "likelihood_verdict:FAIL" in blocker_text or "likelihood_strength_blocked" in blocker_text:
                enter_blockers.append("likelihood blocked")
            if contract_flow_confirmation in {"unknown", "contra", "weak_or_ambiguous"}:
                enter_blockers.append(f"contract flow {contract_flow_confirmation}")
            if flow_confirmation == "contra":
                enter_blockers.append("flow contra")
            if enter_blockers:
                return "REVIEW", "ENTER guard blocked: " + "; ".join(enter_blockers[:5])
            if str(row.get("execution_book", "") or "").strip() in {"Core", "Tactical"} and not bool(row.get("high_enter_ready", False)):
                reason = str(row.get("high_enter_reason", "") or row.get("profit_safety_blockers", "") or "High-confidence entry guard not satisfied.").strip()
                return "REVIEW", reason
            return "ENTER", "Live Schwab quote passed the entry gate."
        if live_status in {
            "chain_error",
            "chain_not_success",
            "missing_underlying_quote",
            "missing_live_quote",
            "missing_leg_in_live_chain",
        }:
            return "WAIT", f"Live pricing incomplete: {live_status or 'unknown'}."
        if live_status == "fails_live_entry_gate":
            return "TARGET", _target_limit_text()
        if live_status == "invalid_entry_structure":
            reason = str(row.get("entry_structure_reason_live", "") or "").strip()
            return "SKIP", f"Invalid live structure{': ' + reason if reason else ''}."
        return "SKIP", f"Live status is not executable: {live_status or 'unknown'}."

    def live_action_label(action: str) -> str:
        """Human-readable action label for reports; raw code is kept separately."""
        code = str(action or "").strip().upper()
        labels = {
            "ENTER": "🟣 ENTER NOW",
            "TARGET": "🟢 WORK LIMIT",
            "REVIEW": "🟡 REVIEW",
            "WAIT": "⚪ WAIT",
            "SKIP": "🔴 SKIP",
        }
        return labels.get(code, code or "⚪ WAIT")

    out_rows = []
    for i, r in mdf.iterrows():
        approved = bool(r["approved"])
        strategy = str(r["strategy"])
        net_type = str(r["net_type"]).lower()
        execution_book_raw = str(r.get("execution_book", "Watch")).strip() or "Watch"
        execution_book = execution_book_raw if approved else "Watch"
        size_mult_val = fnum(r.get("size_mult"))
        size_mult_txt = f"{size_mult_val:.2f}x" if approved and np.isfinite(size_mult_val) else "-"
        watch_reason_flags = []
        flow_bias_txt = ""
        flow_bias_val = fnum(r.get("flow_premium_bias"))
        if np.isfinite(flow_bias_val):
            flow_bias_txt = f", side-prem {flow_bias_val:+.1%}"
        flow_read_txt = (
            f"{str(r.get('flow_direction', '') or 'unknown')}/"
            f"{str(r.get('flow_confidence', '') or 'unknown')}"
            f" ({str(r.get('flow_primary_driver', '') or 'n/a')}; "
            f"{str(r.get('flow_confirmation', '') or 'n/a')}{flow_bias_txt})"
        )
        contract_flow_txt = str(r.get("contract_flow_confirmation", "") or "").strip()
        contract_driver_txt = str(r.get("contract_flow_driver", "") or "").strip()
        if contract_flow_txt:
            flow_read_txt += f"; leg={contract_flow_txt}"
            if contract_driver_txt:
                flow_read_txt += f" ({contract_driver_txt})"
        op, gate_val, _ = parse_gate_value(r.get("entry_gate", ""))

        live_net = fnum(r.get("live_net_bid_ask"))
        if np.isfinite(live_net):
            net_txt = f"{'Credit' if net_type == 'credit' else 'Debit'} {live_net:.2f} (Target {r['entry_gate']})"
            max_profit = money(r.get("live_max_profit"))
            max_loss = money(r.get("live_max_loss"))
            be_txt = calc_be_text(r, live_net)
        elif gate_val is not None:
            tgt_max_p, tgt_max_l = calc_target_max(net_type, float(r["width"]), gate_val)
            net_txt = f"Target {r['entry_gate']}"
            max_profit = money(tgt_max_p)
            max_loss = money(tgt_max_l)
            be_txt = calc_be_text(r, gate_val)
        else:
            net_txt = "N/A"
            max_profit = "N/A"
            max_loss = "N/A"
            be_txt = "N/A"
        live_action, live_action_reason = live_entry_action(r, approved)
        live_action_display = live_action_label(live_action)

        if approved:
            confidence_tier = str(r.get("confidence_tier", ""))
            optimal = str(r.get("optimal_stage1", ""))
            stage1_promoted = bool(r.get("stage1_promoted")) if pd.notna(r.get("stage1_promoted")) else False
            if stage1_promoted and optimal == "Watch Only":
                optimal = "Yes-Good (Promoted)"
                if confidence_tier:
                    confidence_tier = f"{confidence_tier} (Promoted)"
                else:
                    confidence_tier = "Promoted"

            gate_target_now = fnum(r.get("gate_target"))
            gate_live_now = fnum(r.get("gate_live_net"))
            gate_near_miss = bool(r.get("gate_near_miss")) if pd.notna(r.get("gate_near_miss")) else False
            gate_tol_now = fnum(r.get("gate_tol_total"))
            gate_miss_now = fnum(r.get("gate_miss_abs"))
            if np.isfinite(gate_target_now) and np.isfinite(gate_live_now):
                if net_type == "debit":
                    gate_dir = "<="
                else:
                    gate_dir = ">="
                if gate_near_miss:
                    gate_text = (
                        f"near-miss accepted (target {gate_dir} {gate_target_now:.2f}, live {gate_live_now:.2f}, "
                        f"miss {gate_miss_now:.2f}, tol {gate_tol_now:.2f})"
                    )
                else:
                    gate_text = f"PASS (target {gate_dir} {gate_target_now:.2f}, live {gate_live_now:.2f})"
            else:
                gate_text = f"PASS ({r.get('entry_gate')})"

            if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
                notes = (
                    f"Live executable; gate {gate_text}; "
                    f"put short BID/ASK {r.get('short_put_bid_live')}/{r.get('short_put_ask_live')}, "
                    f"put long BID/ASK {r.get('long_put_bid_live')}/{r.get('long_put_ask_live')}, "
                    f"call short BID/ASK {r.get('short_call_bid_live')}/{r.get('short_call_ask_live')}, "
                    f"call long BID/ASK {r.get('long_call_bid_live')}/{r.get('long_call_ask_live')}."
                )
            else:
                notes = (
                    f"Live executable; gate {gate_text}; short BID/ASK "
                    f"{r.get('short_bid_live')}/{r.get('short_ask_live')}, long BID/ASK "
                    f"{r.get('long_bid_live')}/{r.get('long_ask_live')}."
                )
            if execution_book == "Pilot" and gate_near_miss:
                notes = notes.replace(
                    "Live executable; gate near-miss accepted",
                    "Target-limit setup; use target price",
                )
            if net_type == "credit":
                width_eff = fnum(r.get("width"))
                put_w = fnum(r.get("put_width"))
                call_w = fnum(r.get("call_width"))
                if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
                    candidate_w = [w for w in (put_w, call_w, width_eff) if np.isfinite(w) and w > 0]
                    width_eff = max(candidate_w) if candidate_w else math.nan
                if np.isfinite(width_eff) and width_eff > 0:
                    ideal_low = width_eff * ideal_credit_low_pct
                    ideal_high = width_eff * ideal_credit_high_pct
                    notes += (
                        f" Ideal credit guide: {ideal_low:.2f}-{ideal_high:.2f} "
                        f"({ideal_credit_low_pct:.0%}-{ideal_credit_high_pct:.0%} of {width_eff:.2f}w)."
                    )
                notes += (
                    f" Exit plan: take profit near {take_profit_credit_pct:.0%} of max profit; "
                    f"stop/adjust near {stop_loss_credit_pct:.0%} of max defined risk or on confirmed breakeven breach."
                )
            else:
                notes += (
                    f" Exit plan: take profit near +{take_profit_debit_pct:.0%} of debit paid or when target/breakeven is hit; "
                    f"stop near -{stop_loss_debit_pct:.0%} of debit risk or on close-confirmed invalidation."
                )
            spot_asof = fnum(r.get("spot_asof_close"))
            spot_live = fnum(r.get("spot_live_effective"))
            spot_drift = fnum(r.get("spot_asof_live_drift_pct"))
            if np.isfinite(spot_asof) and np.isfinite(spot_live):
                if np.isfinite(spot_drift):
                    notes += (
                        f" Spot check asof/live: {spot_asof:.2f}/{spot_live:.2f} "
                        f"(drift {spot_drift:.1%})."
                    )
                else:
                    notes += f" Spot check asof/live: {spot_asof:.2f}/{spot_live:.2f}."
            if stage1_promoted:
                notes += " Stage-1 Watch was promoted by PASS likelihood + edge/conviction thresholds."
            restrike_reason = str(r.get("restrike_reason", "")).strip()
            if restrike_reason:
                notes += " Stage-2 restrike optimizer selected this executable strike from the same family."
            if str(optimal).strip().lower() == "watch only":
                optimal = "Yes-Good" if execution_book == "Core" else "Yes-Tactical"
            if execution_book == "Tactical":
                quality_items = [x for x in str(r.get("quality_blockers", "")).split(";") if str(x).strip()]
                if quality_items:
                    notes += (
                        " Tactical book (reduced size) due quality blockers: "
                        + ", ".join(quality_items)
                        + "."
                    )
                else:
                    notes += " Tactical book (reduced size)."
            if execution_book == "Pilot":
                notes += (
                    " Pilot book: tiny defined-risk convexity candidate only; "
                    "not Core/Tactical and not eligible for normal sizing."
                )
            portfolio_cap_reason = str(r.get("portfolio_cap_reason", "") or "").strip()
            if portfolio_cap_reason:
                notes += (
                    " Position review required before adding risk: "
                    f"{portfolio_cap_reason}. Treat as manage/add/adjust context, not a fresh blind entry."
                )
            invalidated_effective = (
                bool(r.get("invalidation_breached_effective"))
                if pd.notna(r.get("invalidation_breached_effective"))
                else False
            )
            if invalidated_effective:
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_eval = fnum(r.get("invalidation_eval_price_effective"))
                px_source = str(r.get("invalidation_eval_source", "live")).strip() or "live"
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_eval:.2f}" if np.isfinite(px_eval) else "n/a"
                notes += (
                    f" Invalidation warning only (spot check): breached ({inv_text}; level={lvl_txt}; {px_source}={px_txt}). "
                    f"Action trigger is close-confirmed: require {inv_close_confirms} daily close(s) beyond level."
                )
        else:
            confidence_tier = "Watch Only"
            optimal = "Watch Only"
            stage1_blocked = bool(r.get("stage1_blocked")) if pd.notna(r.get("stage1_blocked")) else False
            if stage1_blocked:
                watch_reason_flags.append("stage1_conviction_watch")
            blocker_items = [x for x in str(r.get("approval_blockers", "")).split(";") if str(x).strip()]
            for blk in blocker_items:
                b = str(blk).strip()
                if b.startswith("likelihood_"):
                    watch_reason_flags.append("likelihood_fail")
                elif b.startswith("edge_below"):
                    watch_reason_flags.append("edge_below_threshold")
                elif b.startswith("signals_below"):
                    watch_reason_flags.append("sample_below_threshold")
                elif b.startswith("invalidation_warning"):
                    watch_reason_flags.append("invalidation_warning")
                elif b.startswith("shield_sigma"):
                    watch_reason_flags.append("shield_sigma_fail")
                elif b.startswith("credit_no_touch"):
                    watch_reason_flags.append("credit_path_risk_fail")
                elif b.startswith("shield_core"):
                    watch_reason_flags.append("shield_core_fail")
                elif b.startswith("shield_delta"):
                    watch_reason_flags.append("shield_delta_fail")
                elif b.startswith("live_rr_weak"):
                    watch_reason_flags.append("live_rr_weak")
                elif b.startswith("fire_delta"):
                    watch_reason_flags.append("fire_delta_fail")
                elif b.startswith("stage1_conviction") or b.startswith("stage1_watch") or b.startswith("stage1_not_actionable"):
                    watch_reason_flags.append("stage1_conviction_watch")
                elif b.startswith("stage1_flow_"):
                    watch_reason_flags.append("stage1_flow_fail")
                elif b.startswith("stage1_contract_flow_"):
                    watch_reason_flags.append("stage1_contract_flow_fail")
                elif b.startswith("flow_"):
                    watch_reason_flags.append("flow_confirmation_fail")
                elif b.startswith("contract_flow_"):
                    watch_reason_flags.append("contract_flow_fail")
                elif b.startswith("fire_gex"):
                    watch_reason_flags.append("fire_gex_blocked")
                elif b.startswith("shield_gex"):
                    watch_reason_flags.append("shield_gex_blocked")
                elif b.startswith("ic_gex"):
                    watch_reason_flags.append("ic_gex_blocked")
                elif (
                    b == "gex_missing"
                    or b.startswith("gex_context")
                    or b.startswith("gex_volatile")
                    or b.startswith("gex_wall")
                ):
                    watch_reason_flags.append("gex_context_blocked")
                elif b.startswith("gex_source_fallback"):
                    watch_reason_flags.append("gex_fallback")
                elif b == "gex_source_stale":
                    watch_reason_flags.append("gex_stale")
                elif b.startswith("confidence_tier_blocked"):
                    watch_reason_flags.append("confidence_tier_blocked")
                elif b.startswith("live_entry_gate_fail") or b.startswith("live_status:"):
                    watch_reason_flags.append("live_entry_gate_miss")
                elif b.startswith("spot_drift") or b.startswith("spot_live_missing"):
                    watch_reason_flags.append("spot_data_mismatch")
                elif b.startswith("bull_call_short_dte_high_edge"):
                    watch_reason_flags.append("bull_call_precision_guard")
                elif b.startswith("bull_call_rr_weak"):
                    watch_reason_flags.append("live_rr_weak")
                elif b.startswith("bull_call_dte_too_long") or b.startswith("bull_call_dte_near_limit"):
                    watch_reason_flags.append("bull_call_dte_fail")
                elif b.startswith("bull_call_contract_flow_not_confirmed"):
                    watch_reason_flags.append("contract_flow_fail")
                elif b.startswith("market_regime_block"):
                    watch_reason_flags.append("market_regime_block")
                elif b.startswith("market_regime_caution"):
                    watch_reason_flags.append("market_regime_caution")
                elif b.startswith("bull_call_otm_too_far") or b.startswith("bear_put_otm_too_far"):
                    watch_reason_flags.append("debit_moneyness_fail")
                elif b.startswith("tactical_"):
                    watch_reason_flags.append("tactical_floor_fail")
                elif b == "scout_research_only":
                    watch_reason_flags.append("scout_research_only")
                elif b.startswith("stage1_"):
                    watch_reason_flags.append("stage1_conviction_watch")
                else:
                    watch_reason_flags.append("other_watch")
            cur_txt = f"{live_net:.2f}" if np.isfinite(live_net) else "N/A"
            live_status = str(r.get("live_status", "missing"))
            live_valid_raw = bool(r.get("is_final_live_valid")) if pd.notna(r.get("is_final_live_valid")) else False
            gate_pass_live = bool(r.get("gate_pass_live")) if pd.notna(r.get("gate_pass_live")) else False
            gate_pass_effective = bool(r.get("gate_pass_effective")) if pd.notna(r.get("gate_pass_effective")) else False
            gate_near_miss = bool(r.get("gate_near_miss")) if pd.notna(r.get("gate_near_miss")) else False
            gate_tol_now = fnum(r.get("gate_tol_total"))
            gate_miss_now = fnum(r.get("gate_miss_abs"))
            live_valid_effective = live_valid_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
            verdict_now = str(r.get("verdict", "")).strip().upper()
            edge_now = fnum(r.get("edge_pct"))
            sig_now = fnum(r.get("signals"))
            _strat_w = str(r.get("strategy", "")).strip()
            _is_bear_w = _strat_w in {"Bear Put Debit", "Bear Call Credit"}
            _track_w = str(r.get("track", "")).strip().upper()
            min_edge_req = (
                min_edge_pct_bear if _is_bear_w
                else min_edge_pct_shield if _track_w == "SHIELD"
                else fnum(approval_cfg.get("min_edge_pct", 0.0))
            )
            min_sig_req = fnum(approval_cfg.get("min_signals", 100))
            require_lk = bool(approval_cfg.get("require_likelihood_pass", True))
            invalidated_effective = (
                bool(r.get("invalidation_breached_effective"))
                if pd.notna(r.get("invalidation_breached_effective"))
                else False
            )
            if (not live_valid_effective) and live_status == "fails_live_entry_gate":
                watch_reason_flags.append("live_entry_gate_miss")
            if invalidated_effective:
                watch_reason_flags.append("invalidation_warning")
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_eval = fnum(r.get("invalidation_eval_price_effective"))
                px_source = str(r.get("invalidation_eval_source", "live")).strip() or "live"
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_eval:.2f}" if np.isfinite(px_eval) else "n/a"
                notes = (
                    f"Watch Only: live_status={live_status}; invalidation warning only "
                    f"(rule: {inv_text}; level={lvl_txt}; {px_source}={px_txt}); "
                    f"close-confirm policy requires {inv_close_confirms} daily close(s) beyond level."
                )
                extra_blockers = [b for b in blocker_items if not str(b).startswith("invalidation_warning")]
                if extra_blockers:
                    notes += " Additional blockers: " + ", ".join(extra_blockers) + "."
            else:
                reasons = []
                if blocker_items:
                    reasons.append("approval blockers: " + ", ".join(blocker_items))
                    if any(str(b).startswith("bull_call_short_dte_high_edge") for b in blocker_items):
                        reasons.append("bull-call precision guard: short DTE plus unusually high edge is blocked")
                    if any(str(b) == "fire_gex_pinned" for b in blocker_items):
                        reasons.append("GEX conflict: FIRE debit is pinned against a wall")
                    tactical_reasons = [str(b) for b in blocker_items if str(b).startswith("tactical_")]
                    if tactical_reasons:
                        reasons.append("tactical floor failed: " + ", ".join(tactical_reasons))
                    if any(str(b) == "scout_research_only" for b in blocker_items):
                        reasons.append("Scout/research lane only; not eligible for live approval")
                if not live_valid_effective:
                    if live_status == "invalid_entry_structure":
                        structure_reason = str(r.get("entry_structure_reason_live", "")).strip()
                        if structure_reason:
                            reasons.append(f"invalid structure ({structure_reason})")
                        else:
                            reasons.append("invalid structure for current spot")
                        watch_reason_flags.append("invalid_entry_structure")
                    elif live_status == "missing_underlying_quote":
                        reasons.append("missing underlying quote for live structure check")
                        watch_reason_flags.append("missing_underlying_quote")
                    else:
                        reasons.append(f"live_status={live_status}")
                portfolio_cap_reason = str(r.get("portfolio_cap_reason", "")).strip()
                if portfolio_cap_reason:
                    reasons.append(f"portfolio cap breach ({portfolio_cap_reason})")
                    watch_reason_flags.append("portfolio_cap_breach")
                if np.isfinite(live_net) and gate_val is not None and not gate_pass_effective:
                    if net_type == "debit":
                        if np.isfinite(gate_miss_now) and np.isfinite(gate_tol_now):
                            reasons.append(
                                f"entry gate miss (need debit <= {gate_val:.2f}, live {cur_txt}, miss {gate_miss_now:.2f} > tol {gate_tol_now:.2f})"
                            )
                        else:
                            reasons.append(f"entry gate miss (need debit <= {gate_val:.2f}, live {cur_txt})")
                    else:
                        if np.isfinite(gate_miss_now) and np.isfinite(gate_tol_now):
                            reasons.append(
                                f"entry gate miss (need credit >= {gate_val:.2f}, live {cur_txt}, miss {gate_miss_now:.2f} > tol {gate_tol_now:.2f})"
                            )
                        else:
                            reasons.append(f"entry gate miss (need credit >= {gate_val:.2f}, live {cur_txt})")
                    watch_reason_flags.append("live_entry_gate_miss")
                elif np.isfinite(live_net) and gate_val is not None and gate_near_miss:
                    reasons.append(
                        f"entry near-miss tolerated (target {r.get('entry_gate')}, live {cur_txt}, tol {gate_tol_now:.2f})"
                    )
                if stage1_blocked:
                    stage1_eval = (
                        str(r.get("stage1_diagnostics", "") or r.get("stage1_not_actionable_reason", "")).strip()
                        or str(r.get("stage1_eval_reason", "")).strip()
                        or "stage1_watch_blocked"
                    )
                    reasons.append(f"stage1 blocked ({stage1_eval})")
                restrike_reason = str(r.get("restrike_reason", "")).strip()
                if restrike_reason:
                    reasons.append("stage-2 restrike optimizer selected nearest executable family strike")
                if require_lk and verdict_now != "PASS":
                    if np.isfinite(edge_now):
                        reasons.append(f"likelihood {verdict_now or 'N/A'} (edge {edge_now:+.1f}%)")
                    else:
                        reasons.append(f"likelihood {verdict_now or 'N/A'}")
                    watch_reason_flags.append("likelihood_fail")
                if verdict_now == "PASS":
                    if np.isfinite(min_edge_req) and np.isfinite(edge_now) and edge_now < min_edge_req:
                        reasons.append(f"edge below threshold ({edge_now:+.1f}% < {min_edge_req:.1f}%)")
                        watch_reason_flags.append("edge_below_threshold")
                    if np.isfinite(min_sig_req) and np.isfinite(sig_now) and sig_now < min_sig_req:
                        reasons.append(f"sample below threshold ({int(sig_now)} < {int(min_sig_req)})")
                        watch_reason_flags.append("sample_below_threshold")
                if not reasons:
                    reasons.append(f"target {r.get('entry_gate', 'N/A')}; current live net={cur_txt}")
                    watch_reason_flags.append("other_watch")
                notes = "Watch Only: " + "; ".join(reasons) + "."
        if not approved and not watch_reason_flags:
            watch_reason_flags.append("other_watch")
        watch_reason_flags = sorted(set(watch_reason_flags))

        hist_success = fnum(r.get("hist_success_pct"))
        edge_pct = fnum(r.get("edge_pct"))
        signals = fnum(r.get("signals"))
        verdict = str(r.get("verdict", "")).strip().upper()
        strength = likelihood_strength(verdict, edge_pct, signals)
        if np.isfinite(hist_success):
            n_txt = f"{int(signals)}" if np.isfinite(signals) else "n/a"
            verdict_txt = verdict if verdict else "N/A"
            setup_likelihood = f"{hist_success:.1f}% {verdict_txt} ({strength}, edge {edge_pct:+.1f}%, n={n_txt})"
            conditioning_level = str(r.get("conditioning_level", "") or "").strip()
            if conditioning_level and conditioning_level not in {"unscored", "base_unconditioned"}:
                setup_likelihood += f"; ctx={conditioning_level}"
        else:
            if verdict == "UNKNOWN":
                setup_likelihood = "Unknown"
            else:
                setup_likelihood = "N/A"
        category_prefix = (
            "Pilot"
            if approved and execution_book_raw == "Pilot"
            else "Approved"
            if approved
            else "Watch Only"
        )

        out_rows.append(
            {
                "#": i + 1,
                "Category": f"{category_prefix} - {normalize_track(r.get('track', ''), strategy)}",
                "Ticker": r["ticker"],
                "Action": action_cell(strategy, str(r.get("track", "")), optimal),
                "Strategy Type": strategy,
                "Strike Setup": strike_setup(
                    strategy,
                    r["long_strike"],
                    r["short_strike"],
                    r["width"],
                    long_put_strike=r.get("long_put_strike"),
                    short_put_strike=r.get("short_put_strike"),
                    short_call_strike=r.get("short_call_strike"),
                    long_call_strike=r.get("long_call_strike"),
                ),
                "Expiry": str(r["expiry"])[:10],
                "DTE": (dt.datetime.strptime(str(r["expiry"])[:10], "%Y-%m-%d").date() - asof).days,
                "Live Action": live_action_display,
                "Live Action Code": live_action,
                "Live Check Reason": live_action_reason,
                "Entry Gate": str(r.get("entry_gate", "") or ""),
                "Net Credit/Debit": net_txt,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Breakeven": be_txt,
                "Conviction %": f"{int(r['conviction'])}%",
                "Setup Likelihood": setup_likelihood,
                "Execution Book": execution_book,
                "Size Mult": size_mult_txt,
                "UW Flow Read": flow_read_txt,
                "Stage-1 Diagnostics": str(r.get("stage1_diagnostics", "") or r.get("stage1_not_actionable_reason", "") or ""),
                "Signal Tier (Stage-1)": confidence_tier,
                "Optimal": optimal,
                "IV Rank": f"{r['iv_rank']:.0f}" if "iv_rank" in r and pd.notna(r.get("iv_rank")) else "",
                "Short Delta": f"{fnum(r.get('short_delta_live')):.2f}" if np.isfinite(fnum(r.get("short_delta_live"))) else "",
                "Long Delta": f"{fnum(r.get('long_delta_live')):.2f}" if np.isfinite(fnum(r.get("long_delta_live"))) else "",
                "Market Regime": (
                    f"{r.get('market_regime_confidence', '')} {fnum(r.get('market_regime_score')):.0f}"
                    if np.isfinite(fnum(r.get("market_regime_score")))
                    else str(r.get("market_regime_confidence", "") or "")
                ),
                "GEX Regime": str(r.get("gex_regime", "")) if r.get("gex_regime") else "",
                "GEX Source": str(r.get("gex_source", "") or ""),
                "Net GEX ($M)": f"{fnum(r.get('net_gex')) / 1e6:.1f}" if np.isfinite(fnum(r.get("net_gex"))) else "",
                "Regime Notes": str(r.get("market_regime_reason", "") or ""),
                "GEX Wall Context": str(r.get("gex_wall_context", "") or ""),
                "Watch Reason Flags": ", ".join(watch_reason_flags) if not approved else "",
                "Notes": notes,
                "Source": "Stage1(ChainOI+DP+HotChains+Screener+Whale) + Stage2(uwos.pricer)",
            }
        )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "#",
            "Category",
            "Ticker",
            "Action",
            "Strategy Type",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Live Action",
            "Live Action Code",
            "Live Check Reason",
            "Entry Gate",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Setup Likelihood",
            "Execution Book",
            "Size Mult",
            "UW Flow Read",
            "Stage-1 Diagnostics",
            "Signal Tier (Stage-1)",
            "Optimal",
            "IV Rank",
                "Short Delta",
                "Long Delta",
                "Market Regime",
                "GEX Regime",
                "GEX Source",
                "Net GEX ($M)",
                "Regime Notes",
                "GEX Wall Context",
            "Watch Reason Flags",
            "Notes",
            "Source",
        ],
    )
    cat_rank = {
        "Approved - FIRE": 0,
        "Approved - SHIELD": 1,
        "Pilot - FIRE": 2,
        "Pilot - SHIELD": 3,
        "Watch Only - FIRE": 4,
        "Watch Only - SHIELD": 5,
        "Watch Only - UNKNOWN": 6,
        "Approved - UNKNOWN": 7,
        "Pilot - UNKNOWN": 8,
    }
    out_df["_cat_rank"] = out_df["Category"].map(cat_rank).fillna(99).astype(int)
    out_df = out_df.sort_values(["_cat_rank", "#"], ascending=[True, True]).drop(columns=["_cat_rank"]).reset_index(drop=True)
    out_df["#"] = range(1, len(out_df) + 1)
    approved_count = int(mdf["approved"].sum()) if "approved" in mdf.columns else 0
    core_count = int((out_df["Execution Book"] == "Core").sum()) if "Execution Book" in out_df.columns else 0
    tactical_count = int((out_df["Execution Book"] == "Tactical").sum()) if "Execution Book" in out_df.columns else 0
    medium_count = int((out_df["Execution Book"] == "Medium").sum()) if "Execution Book" in out_df.columns else 0
    income_count = int((out_df["Execution Book"] == "Income").sum()) if "Execution Book" in out_df.columns else 0
    pilot_count = int((out_df["Execution Book"] == "Pilot").sum()) if "Execution Book" in out_df.columns else 0
    scout_count = int((out_df["Execution Book"] == "Scout").sum()) if "Execution Book" in out_df.columns else 0
    watch_book_count = int((out_df["Execution Book"] == "Watch").sum()) if "Execution Book" in out_df.columns else 0
    dropped_csv = out_dir / f"dropped_trades_{asof_str}.csv"
    dropped_rows = []
    for rec in dropped_stage1:
        dropped_rows.append(
            {
                "ticker": str(rec.get("ticker", "")),
                "strategy": str(rec.get("strategy", "")),
                "expiry": str(rec.get("expiry", ""))[:10],
                "stage": str(rec.get("stage", "stage1")),
                "drop_reason": str(rec.get("drop_reason", "unknown")),
                "details": str(rec.get("details", "")),
            }
        )
    for rec in dropped_final:
        dropped_rows.append(
            {
                "ticker": str(rec.get("ticker", "")),
                "strategy": str(rec.get("strategy", "")),
                "expiry": str(rec.get("expiry", ""))[:10],
                "stage": str(rec.get("stage", "final")),
                "drop_reason": str(rec.get("drop_reason", "unknown")),
                "details": str(rec.get("details", "")),
            }
        )
    dropped_df = pd.DataFrame(
        dropped_rows,
        columns=["ticker", "strategy", "expiry", "stage", "drop_reason", "details"],
    )
    dropped_df.to_csv(dropped_csv, index=False)

    def annotate_live_action_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        annotated = df.copy()
        actions = [
            live_entry_action(row, bool(row.get("approved", False)))
            for _, row in annotated.iterrows()
        ]
        annotated["live_action"] = [action for action, _reason in actions]
        annotated["live_check_reason"] = [reason for _action, reason in actions]
        return annotated

    decision_audit_all = annotate_live_action_columns(decision_audit_all)
    mdf = annotate_live_action_columns(mdf)
    decision_audit_csv = out_dir / f"trade_decision_book_all_{asof_str}.csv"
    decision_audit_all.to_csv(decision_audit_csv, index=False)
    decision_book_csv = out_dir / f"trade_decision_book_{asof_str}.csv"
    mdf.to_csv(decision_book_csv, index=False)

    def output_approved_mask(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(False, index=df.index)
        if "Approved" in df.columns:
            return df["Approved"].astype(str).str.upper().eq("YES")
        if "Execution Book" in df.columns:
            return df["Execution Book"].astype(str).isin(approved_books)
        if "Category" in df.columns:
            return df["Category"].astype(str).str.startswith(("Approved", "Pilot"))
        return pd.Series(False, index=df.index)

    approved_output_mask = output_approved_mask(out_df)
    approved_for_entry_df = out_df[approved_output_mask].copy()
    if not approved_for_entry_df.empty and "Live Action" in approved_for_entry_df.columns:
        approved_live_action_counts = (
            approved_for_entry_df["Live Action"].astype(str).replace("", "UNKNOWN").value_counts().to_dict()
        )
    else:
        approved_live_action_counts = {}
    approved_live_action_order = ["ENTER", "TARGET", "REVIEW", "WAIT", "SKIP", "UNKNOWN"]

    def _format_live_action_counts(counts: dict) -> str:
        ordered = [
            (action, counts[action])
            for action in approved_live_action_order
            if action in counts
        ]
        ordered.extend(
            (action, counts[action])
            for action in sorted(set(counts) - set(approved_live_action_order))
        )
        return ", ".join(f"{action}={int(count)}" for action, count in ordered) or "none"

    approved_live_action_text = _format_live_action_counts(approved_live_action_counts)

    planned_journal_csv = out_dir / f"planned_trade_journal_{asof_str}.csv"
    planned_journal_cols = [
        c
        for c in [
            "#",
            "Ticker",
            "Execution Book",
            "Live Action",
            "Live Check Reason",
            "Entry Gate",
            "Action",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Setup Likelihood",
            "UW Flow Read",
            "GEX Regime",
            "GEX Source",
            "GEX Wall Context",
            "Notes",
        ]
        if c in out_df.columns
    ]
    planned_journal_df = approved_for_entry_df.copy()
    planned_journal_df.loc[:, planned_journal_cols].to_csv(planned_journal_csv, index=False)

    manifest_path = out_dir / f"run_manifest_{asof_str}.json"
    category_order = [
        "Approved - FIRE",
        "Approved - SHIELD",
        "Pilot - FIRE",
        "Pilot - SHIELD",
        "Watch Only - FIRE",
        "Watch Only - SHIELD",
        "Watch Only - UNKNOWN",
        "Approved - UNKNOWN",
        "Pilot - UNKNOWN",
    ]
    execution_book_order = ["Core", "Tactical", "Medium", "Income", "Pilot", "Scout", "Watch"]
    table_cols = [
        c
        for c in [
            "#",
            "Ticker",
            "Action",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Setup Likelihood",
            "UW Flow Read",
            "Stage-1 Diagnostics",
            "Signal Tier (Stage-1)",
            "Optimal",
            "IV Rank",
            "Short Delta",
            "Long Delta",
            "Market Regime",
            "GEX Regime",
            "GEX Source",
            "Net GEX ($M)",
            "Regime Notes",
            "GEX Wall Context",
            "Notes",
        ]
        if c in out_df.columns
    ]

    def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
        cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
        if df.empty or not cols:
            return "_No rows_"
        compact_limits = {
            "Stage-1 Diagnostics": 95,
            "Watch Reason Flags": 95,
            "Reject Reasons": 110,
            "Reject / Action Reason": 110,
            "Notes": 120,
            "Live Check Reason": 100,
            "Regime Notes": 100,
            "GEX Wall Context": 110,
            "UW Flow Read": 100,
            "Daily Blockers": 110,
            "Daily Notes": 120,
            "Morning Reason": 120,
            "Escalation Decision": 120,
            "Source": 80,
            "Setup": 78,
            "Why": 90,
            "Edge / Sample": 70,
        }
        table_df = df[cols].fillna("").copy()
        def _compact_text(value: object, limit: int) -> str:
            text = str(value or "").strip()
            if len(text) <= limit:
                return text
            return text[: max(0, limit - 3)].rstrip() + "..."

        for col, limit in compact_limits.items():
            if col in table_df.columns:
                table_df[col] = table_df[col].map(lambda value: _compact_text(value, limit))
        return table_df.to_markdown(index=False)

    def weekly_credit_fallback_section(trigger_live_enter_rows: int) -> tuple[list[str], dict[str, object]]:
        """Run the validated weekly credit-premium engine when daily directional flow is blank."""
        meta: dict[str, object] = {
            "enabled": False,
            "status": "disabled",
            "rows_generated": 0,
            "rows_live_valid": 0,
            "rows_actionable": 0,
            "artifacts": {},
        }
        weekly_cfg = approval_cfg.get("weekly_credit_fallback", {}) if isinstance(approval_cfg, dict) else {}
        if not bool(weekly_cfg.get("enabled", True)):
            return [], meta
        if args.historical_replay:
            meta.update({"enabled": True, "status": "disabled_historical_replay"})
            return [], meta
        always_show = bool(weekly_cfg.get("always_show", False))
        if trigger_live_enter_rows > 0 and not always_show:
            meta.update({"enabled": True, "status": "not_triggered_live_enter_exists"})
            return [], meta

        fallback_dir = out_dir / f"weekly_credit_fallback_{asof_str}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        min_credit_pct = fnum(weekly_cfg.get("min_credit_pct_width", 0.18))
        min_pop = normalize_probability(weekly_cfg.get("min_pop", 0.70))
        min_conf = fnum(weekly_cfg.get("min_confidence_score", 7.0))
        max_loss_cap = fnum(weekly_cfg.get("max_loss_per_contract", 850.0))
        max_rows = max(1, int(weekly_cfg.get("max_rows", 5)))
        max_candidates_per_day = max(max_rows, int(weekly_cfg.get("max_candidates_per_day", 8)))
        top_underlyings = max(1, int(weekly_cfg.get("top_underlyings", 120)))
        max_leg_spread_pct = fnum(weekly_cfg.get("max_leg_spread_pct", 0.25))
        min_iv_rank = fnum(weekly_cfg.get("min_iv_rank", 20.0))

        try:
            gen_cfg = WeeklyCreditConfig(
                max_candidates_per_day=max_candidates_per_day,
                max_candidates_per_week=max_rows,
                top_underlyings=top_underlyings,
                min_iv_rank=min_iv_rank if np.isfinite(min_iv_rank) else 20.0,
                min_credit_pct_width=min_credit_pct if np.isfinite(min_credit_pct) else 0.18,
                max_leg_spread_pct=max_leg_spread_pct if np.isfinite(max_leg_spread_pct) else 0.25,
            )
            generated_df, diagnostics = generate_weekly_credit_for_day(base, gen_cfg)
            write_weekly_credit_outputs(generated_df, diagnostics, fallback_dir, asof_str)
        except Exception as exc:
            meta.update({"enabled": True, "status": f"generator_failed:{exc}"})
            return [
                "## Trades To Enter",
                "",
                f"- Status: generator failed: `{exc}`",
                "",
            ], meta

        generated_csv = fallback_dir / f"weekly_trade_candidates_{asof_str}.csv"
        exact_csv = fallback_dir / f"weekly_trade_setups_for_exact_backtest_{asof_str}.csv"
        price_dir = fallback_dir / "live_pricer"
        full_live_csv = price_dir / f"live_trade_table_{asof_str}.csv"
        final_live_csv = price_dir / f"live_trade_table_{asof_str}_final.csv"
        snapshot_json = price_dir / "schwab_snapshot.json"
        chain_dir = price_dir / "chains"
        meta.update(
            {
                "enabled": True,
                "status": "generated",
                "rows_generated": int(len(generated_df)),
                "artifacts": {
                    "weekly_generated_csv": str(generated_csv),
                    "weekly_exact_csv": str(exact_csv),
                    "weekly_live_csv": str(full_live_csv),
                    "weekly_final_live_csv": str(final_live_csv),
                },
            }
        )
        if generated_df.empty or not exact_csv.exists():
            meta["status"] = "no_generated_rows"
            return [
                "## Trades To Enter",
                "",
                "- Status: no generated credit-premium candidates passed the fallback discovery gates.",
                "",
            ], meta

        price_cmd = [
            sys.executable,
            "-m",
            "uwos.pricer",
            "--shortlist-csv",
            str(exact_csv),
            "--out-dir",
            str(price_dir),
            "--entry-tol-width-pct",
            str(entry_tol_width_pct),
            "--entry-tol-floor",
            str(entry_tol_floor),
            "--save-chain-dir",
            str(chain_dir),
            "--snapshot-out-json",
            str(snapshot_json),
        ]
        try:
            proc = subprocess.run(
                price_cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
            (fallback_dir / "live_pricer.log").write_text(proc.stdout or "", encoding="utf-8")
            if proc.returncode != 0:
                meta["status"] = f"live_pricer_failed:{proc.returncode}"
                return [
                    "## Trades To Enter",
                    "",
                    f"- Status: live pricer failed with return code `{proc.returncode}`.",
                    f"- Log: {fallback_dir / 'live_pricer.log'}",
                    "",
                ], meta
        except Exception as exc:
            meta["status"] = f"live_pricer_failed:{exc}"
            return [
                "## Trades To Enter",
                "",
                f"- Status: live pricer failed: `{exc}`",
                "",
            ], meta

        if not final_live_csv.exists():
            meta["status"] = "no_live_final_csv"
            return [
                "## Trades To Enter",
                "",
                "- Status: live pricer did not emit a final live-valid CSV.",
                "",
            ], meta

        live_final = pd.read_csv(final_live_csv)
        meta["rows_live_valid"] = int(len(live_final))
        if live_final.empty:
            meta["status"] = "no_live_valid_rows"
            return [
                "## Trades To Enter",
                "",
                "- Status: generated credit candidates existed, but none passed current Schwab live entry.",
                f"- Generated CSV: {generated_csv}",
                f"- Live pricer CSV: {full_live_csv}",
                "",
            ], meta

        merge_keys = [c for c in ["ticker", "strategy", "expiry", "short_leg", "long_leg"] if c in live_final.columns and c in generated_df.columns]
        enriched = live_final.merge(
            generated_df,
            on=merge_keys,
            how="left",
            suffixes=("", "_generated"),
        ) if merge_keys else live_final.copy()

        def _first_num(row, names: list[str]) -> float:
            for name in names:
                if name in row.index:
                    val = fnum(row.get(name))
                    if np.isfinite(val):
                        return val
            return math.nan

        def _confidence_label(score: float) -> str:
            if np.isfinite(score) and score >= 7:
                return "High"
            if np.isfinite(score) and score >= 5:
                return "Medium"
            return "Review"

        def _fallback_breakeven(row) -> float:
            short_strike = _first_num(row, ["short_strike", "short_strike_generated"])
            credit = _first_num(row, ["live_net_bid_ask", "entry_net", "entry_net_generated"])
            strategy_local = str(row.get("strategy", "") or "").strip()
            if not (np.isfinite(short_strike) and np.isfinite(credit)):
                return math.nan
            if strategy_local == "Bull Put Credit":
                return short_strike - credit
            if strategy_local == "Bear Call Credit":
                return short_strike + credit
            return math.nan

        def _format_strike(value: float) -> str:
            if not np.isfinite(value):
                return ""
            if abs(value - round(value)) < 0.001:
                return str(int(round(value)))
            return f"{value:.2f}".rstrip("0").rstrip(".")

        def _parse_display_leg(symbol: str, fallback_strike: float = math.nan, fallback_right: str = "") -> tuple[str, str, str]:
            raw = str(symbol or "").strip()
            parsed = parse_occ(re.sub(r"\s+", "", raw).upper())
            if parsed:
                _root, _expiry, right, strike = parsed
                return f"{_format_strike(strike)}{right}", _format_strike(strike), "Put" if right == "P" else "Call"
            right = str(fallback_right or "").strip().upper()[:1]
            strike_text = _format_strike(fallback_strike)
            if strike_text and right in {"P", "C"}:
                return f"{strike_text}{right}", strike_text, "Put" if right == "P" else "Call"
            return raw, strike_text, "Put" if right == "P" else "Call" if right == "C" else ""

        rows = []
        for _, row in enriched.iterrows():
            width_val = _first_num(row, ["width_live", "width", "width_generated"])
            credit_val = _first_num(row, ["live_net_bid_ask", "entry_net", "entry_net_generated"])
            credit_pct = (credit_val / width_val) if np.isfinite(credit_val) and np.isfinite(width_val) and width_val > 0 else math.nan
            pop_val = _first_num(row, ["pop_estimate", "pop_estimate_generated"])
            conf_score = _first_num(row, ["confidence_score", "confidence_score_generated"])
            live_loss = _first_num(row, ["live_max_loss", "max_loss", "max_loss_generated"])
            if np.isfinite(min_conf) and (not np.isfinite(conf_score) or conf_score < min_conf):
                continue
            if np.isfinite(min_pop) and (not np.isfinite(pop_val) or pop_val < min_pop):
                continue
            if np.isfinite(min_credit_pct) and (not np.isfinite(credit_pct) or credit_pct < min_credit_pct):
                continue
            if np.isfinite(max_loss_cap) and (not np.isfinite(live_loss) or live_loss > max_loss_cap):
                continue
            short_leg = str(row.get("short_leg_live_symbol") or row.get("short_leg") or "").strip()
            long_leg = str(row.get("long_leg_live_symbol") or row.get("long_leg") or "").strip()
            short_strike = _first_num(row, ["short_strike", "short_strike_generated"])
            long_strike = _first_num(row, ["long_strike", "long_strike_generated"])
            strategy_local = str(row.get("strategy", "") or "").strip()
            fallback_right = "P" if "Put" in strategy_local else "C" if "Call" in strategy_local else ""
            sell_leg_text, sell_strike_text, option_type = _parse_display_leg(short_leg, short_strike, fallback_right)
            buy_leg_text, buy_strike_text, buy_option_type = _parse_display_leg(long_leg, long_strike, fallback_right)
            rows.append(
                {
                    "Action": "ENTER",
                    "Ticker": str(row.get("ticker", "") or "").strip(),
                    "Strategy": strategy_local,
                    "Sell Leg": f"Sell {sell_leg_text}",
                    "Buy Leg": f"Buy {buy_leg_text}",
                    "Expiry": str(row.get("expiry", "") or "")[:10],
                    "Sell Strike": sell_strike_text,
                    "Buy Strike": buy_strike_text,
                    "Type": option_type or buy_option_type,
                    "Credit": f"{credit_val:.2f}" if np.isfinite(credit_val) else "",
                    "Credit % Width": f"{credit_pct:.1%}" if np.isfinite(credit_pct) else "",
                    "Max Loss": f"{live_loss:.0f}" if np.isfinite(live_loss) else "",
                    "Breakeven": f"{_fallback_breakeven(row):.2f}" if np.isfinite(_fallback_breakeven(row)) else "",
                    "POP": f"{pop_val:.0%}" if np.isfinite(pop_val) else "",
                    "Confidence": _confidence_label(conf_score),
                    "Score": f"{conf_score:.1f}" if np.isfinite(conf_score) else "",
                    "Why": str(row.get("reason") or row.get("reason_generated") or "").strip(),
                }
            )

        action_cols = [
            "Action",
            "Ticker",
            "Strategy",
            "Sell Leg",
            "Buy Leg",
            "Expiry",
            "Sell Strike",
            "Buy Strike",
            "Type",
            "Credit",
            "Credit % Width",
            "Max Loss",
            "Breakeven",
            "POP",
            "Confidence",
            "Score",
            "Why",
        ]
        actionable = pd.DataFrame(rows, columns=action_cols)
        if not actionable.empty:
            actionable["_score_sort"] = pd.to_numeric(actionable["Score"], errors="coerce").fillna(-1)
            actionable["_pop_sort"] = pd.to_numeric(
                actionable["POP"].astype(str).str.rstrip("%"),
                errors="coerce",
            ).fillna(-1)
            actionable = (
                actionable.sort_values(["Confidence", "_score_sort", "_pop_sort"], ascending=[True, False, False])
                .drop(columns=["_score_sort", "_pop_sort"])
                .head(max_rows)
            )
        action_csv = fallback_dir / f"weekly_credit_fallback_live_enter_{asof_str}.csv"
        actionable.to_csv(action_csv, index=False)
        meta["rows_actionable"] = int(len(actionable))
        meta["artifacts"]["weekly_action_csv"] = str(action_csv)
        meta["status"] = "actionable" if not actionable.empty else "no_actionable_after_safety_filters"

        section = [
            "## Trades To Enter",
            "",
            (
                "- Source: weekly credit fallback, triggered because the directional daily approval book had no live ENTER rows."
                if trigger_live_enter_rows == 0
                else "- Source: weekly credit fallback, configured to show even when directional entries exist."
            ),
            f"- Generated candidates: {int(len(generated_df))}",
            f"- Schwab live-valid rows: {int(len(live_final))}",
            f"- Actionable fallback rows: {int(len(actionable))}",
            f"- Safety gates: POP >= {min_pop:.0%}, credit >= {min_credit_pct:.0%} width, confidence score >= {min_conf:g}, max loss <= ${max_loss_cap:,.0f}",
            "",
        ]
        if actionable.empty:
            section.extend([
                "_No fallback rows survived the live-price and safety gates._",
                "",
            ])
        else:
            section.append(markdown_table(actionable, [
                "Action",
                "Ticker",
                "Strategy",
                "Sell Leg",
                "Buy Leg",
                "Expiry",
                "Sell Strike",
                "Buy Strike",
                "Credit",
                "Credit % Width",
                "Max Loss",
                "Breakeven",
                "POP",
                "Confidence",
                "Why",
            ]))
            section.append("")
        return section, meta

    def event_momentum_candidates_section(source_df: pd.DataFrame) -> list[str]:
        def _event_short_text(value: object, limit: int = 150) -> str:
            text = str(value or "").strip()
            if len(text) <= limit:
                return text
            return text[: max(0, limit - 3)].rstrip() + "..."

        def _event_compact_likelihood(value: object) -> str:
            text = str(value or "").strip()
            verdict = "LS" if "LOW_SAMPLE" in text else "FAIL" if "FAIL" in text else "PASS" if "PASS" in text else ""
            edge_match = re.search(r"edge\s+([+-]?\d+(?:\.\d+)?)%", text)
            n_match = re.search(r"n=(\d+)", text)
            parts = [verdict]
            if edge_match:
                parts.append(f"{float(edge_match.group(1)):+.1f}%")
            if n_match:
                parts.append(f"n{n_match.group(1)}")
            return " ".join([p for p in parts if p]) or _event_short_text(text, 28)

        def _event_compact_blockers(value: object) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            tokens = [x.strip() for x in re.split(r"[;,]", text) if x.strip()]
            labels: list[str] = []
            for token in tokens:
                label = ""
                if "stage1_conviction_below_yes_good" in token:
                    match = re.search(r"(\d+)<(\d+)", token)
                    label = f"S1 {match.group(1)}<{match.group(2)}" if match else "S1 low"
                elif "stage1_flow_weak_or_ambiguous" in token:
                    label = "weak flow"
                elif "stage1_contract_flow_contra" in token or "contract_flow_contra" in token:
                    label = "contract contra"
                elif "stage1_contract_flow_weak_or_ambiguous" in token or "contract_flow_weak_or_ambiguous" in token:
                    label = "contract weak"
                elif "stage1_high_iv_debit_watch_only" in token:
                    label = "high IV"
                elif "LOW_SAMPLE" in token or "Low Sample" in token:
                    label = "low sample"
                elif token.startswith("signals_below"):
                    label = "n<min"
                elif token.startswith("edge_below"):
                    label = "edge<thresh"
                elif token.startswith("likelihood_verdict:FAIL"):
                    label = "likelihood fail"
                elif "gex" in token.lower():
                    label = "GEX"
                elif token:
                    label = token.replace("_", " ")
                if label and label not in labels:
                    labels.append(label)
                if len(labels) >= 3:
                    break
            return "; ".join(labels) if labels else _event_short_text(text, 35)

        def _event_strategy(value: object) -> str:
            text = str(value or "").strip()
            return {
                "Bull Call Debit": "Bull Call",
                "Bear Put Debit": "Bear Put",
                "Iron Condor": "IC",
                "Long Iron Condor": "Long IC",
            }.get(text, text)

        section = [
            "## Event Momentum / High-IV Candidates",
            "",
            "These rows come from the full decision book, not just the final display cap. They are shown so high-interest/event names cannot silently disappear. Scout is research/watch only unless approval.scout_live_entry_enabled is explicitly turned on.",
            "",
        ]
        if source_df.empty:
            section.extend(["_No event/high-IV candidates available._", ""])
            return section
        ev = source_df.copy()
        diag = ev.get("stage1_not_actionable_reason", pd.Series("", index=ev.index)).astype(str)
        notes = ev.get("notes_stage1", pd.Series("", index=ev.index)).astype(str)
        ev = ev[
            diag.str.contains("stage1_high_iv_debit_watch_only", na=False)
            | notes.str.contains("ER-RISK|EARN", case=False, regex=True, na=False)
        ].copy()
        if ev.empty:
            section.extend(["_No event/high-IV candidates available._", ""])
            return section

        for c in ["call_premium", "put_premium", "bullish_premium", "bearish_premium"]:
            if c not in ev.columns:
                ev[c] = 0.0
            ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0.0)
        ev["_event_interest"] = (
            ev["call_premium"].abs()
            + ev["put_premium"].abs()
            + ev["bullish_premium"].abs()
            + ev["bearish_premium"].abs()
        )

        def _strike_setup(row):
            strat = str(row.get("strategy", "")).strip()
            long_strike = fnum(row.get("long_strike"))
            short_strike = fnum(row.get("short_strike"))
            if strat == "Bull Call Debit" and np.isfinite(long_strike) and np.isfinite(short_strike):
                return f"Buy {long_strike:.2f}C / Sell {short_strike:.2f}C"
            if strat == "Bear Put Debit" and np.isfinite(long_strike) and np.isfinite(short_strike):
                return f"Buy {long_strike:.2f}P / Sell {short_strike:.2f}P"
            return str(row.get("strategy", "")).strip()

        ev["Strike Setup"] = ev.apply(_strike_setup, axis=1)
        ev["Event Interest $M"] = ev["_event_interest"].map(lambda x: f"{x / 1_000_000:.1f}")
        ev["Live Debit/Credit"] = ev.apply(
            lambda r: (
                f"{fnum(r.get('live_net_bid_ask')):.2f} vs {str(r.get('entry_gate', '')).strip()}"
                if np.isfinite(fnum(r.get("live_net_bid_ask")))
                else str(r.get("entry_gate", "")).strip()
            ),
            axis=1,
        )
        ev["Spot vs BE"] = ev.apply(
            lambda r: (
                f"{fnum(r.get('spot_live_effective')):.2f} / {fnum(r.get('breakeven')):.2f}"
                if np.isfinite(fnum(r.get("spot_live_effective"))) and np.isfinite(fnum(r.get("breakeven")))
                else ""
            ),
            axis=1,
        )
        ev["RR"] = ev["live_reward_risk"].map(lambda x: f"{fnum(x):.2f}" if np.isfinite(fnum(x)) else "")

        def _fmt_event_num(value):
            v = fnum(value)
            if not np.isfinite(v):
                return ""
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.1f}"

        ev["Likelihood"] = ev.apply(
            lambda r: (
                f"{str(r.get('verdict', '')).strip()} edge {fnum(r.get('edge_pct')):.1f}% n={_fmt_event_num(r.get('signals'))}"
                if np.isfinite(fnum(r.get("edge_pct")))
                else str(r.get("verdict", "")).strip()
            ),
            axis=1,
        )
        ev["Reject / Action Reason"] = ev.apply(
            lambda r: "; ".join(
                [x.strip() for x in str(r.get("approval_blockers", "")).split(";") if x.strip()][:4]
            ),
            axis=1,
        )
        def _book_status(value: object) -> str:
            book = str(value or "").strip()
            if book == "Core":
                return "🟢 CORE"
            if book == "Tactical":
                return "🟦 TACT"
            if book == "Pilot":
                return "🟠 PILOT"
            if book == "Scout":
                return "🟡 SCOUT"
            return "🔴 WATCH"

        ev["Status"] = ev.get("execution_book", pd.Series("", index=ev.index)).map(_book_status)
        ev["St"] = ev["Status"].map(lambda x: str(x).split()[0] if str(x).strip() else "🔴")
        ev["Ticker"] = ev["ticker"].astype(str).str.upper().str.strip()
        ev["Strat"] = ev["strategy"].map(_event_strategy)
        ev["Legs"] = ev["Strike Setup"].map(lambda v: _event_short_text(v, 34))
        ev["Exp"] = ev["expiry"].astype(str).str[:10]
        ev["Conv"] = ev["conviction"].map(lambda x: f"{fnum(x):.0f}%" if np.isfinite(fnum(x)) else "")
        ev["Entry"] = ev["Live Debit/Credit"].map(lambda v: str(v).replace(" vs <=", "<=").replace(" vs >=", ">="))
        ev["Entry"] = ev["Entry"].map(lambda v: re.sub(r"\s+", " ", str(v)).strip())
        ev["Edge"] = ev["Likelihood"].map(_event_compact_likelihood)
        ev["GEX"] = ev.get("gex_regime", pd.Series("", index=ev.index)).astype(str).str.replace("_", " ")
        ev["Why"] = ev.apply(
            lambda r: (
                str(r.get("Reject / Action Reason", "")).strip()
                or str(r.get("stage1_not_actionable_reason", "")).strip()
                or str(r.get("contract_flow_confirmation", "")).strip()
            ),
            axis=1,
        )
        ev["Why"] = ev["Why"].map(_event_compact_blockers)
        ev = ev.sort_values(["_event_interest", "_display_rank_score"], ascending=[False, False]).head(25)
        section.append("**Top event/high-IV candidates**")
        section.append(
            markdown_table(
                ev,
                ["St", "Ticker", "Strat", "Legs", "Exp", "Conv", "Entry", "Edge", "GEX", "Why"],
            )
        )
        section.append("")
        section.append(
            "_Full event/high-IV diagnostics are in the decision CSV; the Markdown intentionally keeps only the compact setup table._"
        )
        section.append("")
        return section

    def bullets_from_rows(df: pd.DataFrame, value_col: str, prefix_col: str = "Ticker") -> list[str]:
        if value_col not in df.columns:
            return []
        bullets = []
        for _, row in df.iterrows():
            value = str(row.get(value_col, "") or "").strip()
            if not value:
                continue
            number = str(row.get("#", "") or "").strip()
            prefix = str(row.get(prefix_col, "") or "").strip()
            label = f"#{number} {prefix}".strip()
            bullets.append(f"- {label}: {value}")
        return bullets

    live_entry_title = "## Live Entry Summary"
    if args.historical_replay:
        live_entry_title = "## Historical Replay Entry Summary"
    elif args.eod_live_planning:
        live_entry_title = "## EOD Live Planning Entry Summary"
    live_entry_summary = [live_entry_title, ""]
    if args.historical_replay:
        live_entry_summary.append(
            "- Historical replay is validation only; use EOD live planning for actionable ENTER/TARGET/REVIEW decisions."
        )
    elif args.eod_live_planning:
        live_entry_summary.append(
            "- EOD live planning uses the dated EOD files for discovery, then current Schwab quotes and Schwab chain GEX for ENTER/TARGET/REVIEW."
        )
    if out_df.empty:
        live_entry_summary.append("- No recommendation rows were produced.")
    else:
        approved_for_entry = approved_for_entry_df.copy()
        if approved_for_entry.empty:
            live_entry_summary.append(
                "- No HIGH-confidence daily directional approvals. Check Trades To Enter first, then Watchlist."
            )
        else:
            if "Live Action" in approved_for_entry.columns:
                action_series = approved_for_entry["Live Action"].astype(str).replace("", "UNKNOWN")
            else:
                action_series = pd.Series(["UNKNOWN"] * len(approved_for_entry), index=approved_for_entry.index)
            action_counts = action_series.value_counts().to_dict()
            action_display_order = ["ENTER", "TARGET", "REVIEW", "WAIT", "SKIP", "UNKNOWN"]
            ordered_action_counts = [
                (action, action_counts[action])
                for action in action_display_order
                if action in action_counts
            ]
            ordered_action_counts.extend(
                (action, action_counts[action])
                for action in sorted(set(action_counts) - set(action_display_order))
            )
            live_entry_summary.append(
                (
                    "- Historical gate-pass action split: "
                    if args.historical_replay
                    else "- Planned live-action split: "
                    if args.eod_live_planning
                    else "- Approved live-action split: "
                )
                + ", ".join([f"{k}={int(v)}" for k, v in ordered_action_counts])
            )
            for action in action_display_order + sorted(set(action_counts) - set(action_display_order)):
                subset = approved_for_entry[action_series.eq(action)]
                if subset.empty:
                    continue
                live_entry_summary.append(f"- {action}:")
                for _, rr in subset.iterrows():
                    live_entry_summary.append(
                        "  - "
                        + f"{rr.get('Ticker', '')} {rr.get('Action', '')} "
                        + f"{rr.get('Strike Setup', '')} {rr.get('Expiry', '')}: "
                        + str(rr.get("Live Check Reason", "")).strip()
                    )
    live_entry_summary.append("")

    plan_cols = [
        "#",
        "Ticker",
        "Live Action",
        "Action",
        "Expiry",
        "DTE",
        "Net Credit/Debit",
        "Breakeven",
    ]
    strike_cols = ["#", "Ticker", "Strike Setup"]
    risk_cols = [
        "#",
        "Ticker",
        "Max Profit",
        "Max Loss",
        "Conviction %",
        "Setup Likelihood",
        "IV Rank",
        "Short Delta",
        "Long Delta",
        "Market Regime",
        "GEX Regime",
        "Net GEX ($M)",
    ]
    reason_summary_cols = [
        "#",
        "Ticker",
        "Strategy Type",
        "Expiry",
        "Conviction %",
        "Setup Likelihood",
        "Execution Book",
        "Stage-1 Diagnostics",
    ]

    mini_tables = []
    for book in execution_book_order:
        mini_tables.append(f"### {book} Book")
        book_df = out_df[out_df["Execution Book"] == book].copy()
        if book_df.empty:
            mini_tables.extend(["_No rows_", ""])
            continue
        has_rows = False
        for cat in category_order:
            sub = book_df[book_df["Category"] == cat].copy()
            if sub.empty:
                continue
            has_rows = True
            mini_tables.extend(
                [
                    f"#### {cat}",
                    "**Trade plan**",
                    markdown_table(sub, plan_cols),
                    "",
                    "**Strike setup**",
                    markdown_table(sub, strike_cols),
                    "",
                    "**Risk / edge**",
                    markdown_table(sub, risk_cols),
                    "",
                ]
            )
            notes = bullets_from_rows(sub, "Notes")
            if notes:
                mini_tables.extend(["**Notes**", *notes, ""])
        if not has_rows:
            mini_tables.extend(["_No rows_", ""])
    if not mini_tables:
        mini_tables = ["_No rows_", ""]

    action_mini_tables = []
    for book in [b for b in execution_book_order if b != "Watch"]:
        book_df = out_df[out_df["Execution Book"] == book].copy()
        if book_df.empty:
            continue
        action_mini_tables.append(f"### {book} Book")
        for cat in category_order:
            sub = book_df[book_df["Category"] == cat].copy()
            if sub.empty:
                continue
            action_mini_tables.extend(
                [
                    f"#### {cat}",
                    "**Trade plan**",
                    markdown_table(sub, plan_cols),
                    "",
                    "**Strike setup**",
                    markdown_table(sub, strike_cols),
                    "",
                    "**Risk / edge**",
                    markdown_table(sub, risk_cols),
                    "",
                ]
            )
            notes = bullets_from_rows(sub, "Notes")
            if notes:
                action_mini_tables.extend(["**Action context**", *notes, ""])
    if not action_mini_tables:
        action_mini_tables = ["_No HIGH-confidence daily directional approved rows. Use Trades To Enter first, then Watchlist._", ""]

    watch_reason_order = [
        ("stage1_conviction_watch", "Stage-1 Conviction Watch"),
        ("stage1_flow_fail", "Stage-1 Flow Weak/Contra"),
        ("stage1_contract_flow_fail", "Stage-1 Contract Flow Weak/Contra"),
        ("portfolio_cap_breach", "Portfolio Cap Breach"),
        ("invalid_entry_structure", "Invalid Entry Structure"),
        ("missing_underlying_quote", "Missing Underlying Quote"),
        ("live_entry_gate_miss", "Live Entry Gate Miss"),
        ("invalidation_warning", "Invalidation Warning (Close-Confirm)"),
        ("spot_data_mismatch", "Spot Data Mismatch"),
        ("debit_moneyness_fail", "Debit Moneyness Fail"),
        ("gex_fallback", "GEX Fallback / Unverified Wall Context"),
        ("likelihood_fail", "Likelihood Fail"),
        ("shield_sigma_fail", "Shield Sigma Gate Fail"),
        ("credit_path_risk_fail", "Credit Path-Risk Fail"),
        ("shield_core_fail", "Shield Core Gate Fail"),
        ("shield_delta_fail", "Shield Delta Gate Fail"),
        ("confidence_tier_blocked", "Confidence Tier Blocked"),
        ("edge_below_threshold", "Edge Below Threshold"),
        ("sample_below_threshold", "Sample Below Threshold"),
        ("other_watch", "Other Watch Reason"),
    ]
    watch_reason_tables = []
    watch_df = out_df[out_df["Category"].astype(str).str.startswith("Watch Only")].copy()
    reason_cols = [
        c
        for c in [
            "#",
            "Ticker",
            "Strategy Type",
            "Expiry",
            "Conviction %",
            "Setup Likelihood",
            "Execution Book",
            "Watch Reason Flags",
            "Notes",
        ]
        if c in out_df.columns
    ]
    for code, title in watch_reason_order:
        if watch_df.empty:
            break
        sub = watch_df[watch_df["Watch Reason Flags"].astype(str).str.contains(rf"\b{re.escape(code)}\b", regex=True)]
        if sub.empty:
            continue
        watch_reason_tables.extend(
            [
                f"### {title}",
                markdown_table(sub, reason_summary_cols),
                "",
            ]
        )
        flags = bullets_from_rows(sub, "Watch Reason Flags")
        if flags:
            watch_reason_tables.extend(["**Reason flags**", *flags, ""])
        notes = bullets_from_rows(sub, "Notes")
        if notes:
            watch_reason_tables.extend(["**Notes**", *notes, ""])
    if not watch_reason_tables:
        watch_reason_tables = ["_No watch-only reason rows_", ""]

    gate_diagnostics = []
    if "approval_blockers" in mdf.columns and not mdf.empty:
        diag_rows = []
        for _, row in mdf.iterrows():
            raw = str(row.get("approval_blockers", "") or "")
            tokens = [x.strip() for x in raw.split(";") if x.strip()]
            if not tokens:
                tokens = ["none"]
            for token in tokens:
                diag_rows.append(
                    {
                        "Track": normalize_track(row.get("track", ""), row.get("strategy", "")),
                        "Strategy": str(row.get("strategy", "")).strip(),
                        "Execution Book": str(row.get("execution_book", "")).strip(),
                        "Blocker": token,
                    }
                )
        if diag_rows:
            diag_df = pd.DataFrame(diag_rows)
            top_blockers = (
                diag_df.groupby(["Track", "Execution Book", "Blocker"], dropna=False)
                .size()
                .reset_index(name="Count")
                .sort_values(["Count", "Track", "Execution Book", "Blocker"], ascending=[False, True, True, True])
                .head(20)
            )
            gate_diagnostics.extend(
                [
                    "## Daily Gate Diagnostics",
                    "",
                    "**Top blockers in final output**",
                    markdown_table(top_blockers, ["Track", "Execution Book", "Blocker", "Count"]),
                    "",
                ]
            )
    if not gate_diagnostics:
        gate_diagnostics = ["## Daily Gate Diagnostics", "", "_No blocker diagnostics available._", ""]

    event_momentum_section = event_momentum_candidates_section(decision_audit_all)

    near_miss_rejected = [
        "## Near-Miss But Rejected",
        "",
        "These are the tempting Watch rows. They remain rejected because forcing trades through failed Stage-1, contract-flow, likelihood, live-entry, or GEX gates is how the model degrades.",
        "",
    ]
    if out_df.empty or "Execution Book" not in out_df.columns:
        near_miss_rejected.extend(["_No near-miss rows available._", ""])
    else:
        nm = out_df[out_df["Execution Book"].astype(str).eq("Watch")].copy()
        if nm.empty:
            near_miss_rejected.extend(["_No rejected near-misses._", ""])
        else:
            nm["_conv_num"] = pd.to_numeric(
                nm["Conviction %"].astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            ).fillna(-1)
            stage1_detail_col = nm.get("Stage-1 Diagnostics", pd.Series("", index=nm.index))
            if isinstance(stage1_detail_col, pd.DataFrame):
                stage1_detail_col = stage1_detail_col.iloc[:, 0]
            nm["_has_stage1_detail"] = stage1_detail_col.astype(str).str.len() > 0
            nm = nm.sort_values(["_conv_num", "_has_stage1_detail", "#"], ascending=[False, False, True]).head(10)
            near_cols = [
                "#",
                "Ticker",
                "Strategy Type",
                "Conviction %",
                "Setup Likelihood",
                "UW Flow Read",
                "GEX Wall Context",
                "Stage-1 Diagnostics",
                "Watch Reason Flags",
                "Notes",
            ]
            near_miss_rejected.extend([markdown_table(nm, near_cols), ""])

    def candidate_report_key(row) -> str:
        parts = [
            str(row.get("ticker", row.get("Ticker", "")) or "").strip().upper(),
            str(row.get("strategy", row.get("Strategy Type", row.get("Action", ""))) or "").strip(),
            str(row.get("expiry", row.get("Expiry", "")) or "").strip(),
        ]
        for col in [
            "long_strike",
            "short_strike",
            "long_put_strike",
            "short_put_strike",
            "short_call_strike",
            "long_call_strike",
            "Strike Setup",
        ]:
            sval = str(row.get(col, "") or "").strip()
            if sval:
                parts.append(f"{col}={sval}")
        return "|".join(parts)

    rejected_trade_reasons = [
        "## Rejected Trades and Exact Reasons",
        "",
        "These are all gated-out Watch candidates from the full daily candidate set, including rows trimmed out of the main display after ranking.",
        "",
    ]
    if mdf.empty or "execution_book" not in mdf.columns:
        rejected_trade_reasons.extend(["_No rejected trade rows available._", ""])
    else:
        shown_keys = set()
        if not out_df.empty:
            shown_keys = {candidate_report_key(row) for _, row in out_df.iterrows()}
        rejected_df = mdf[mdf["execution_book"].astype(str).eq("Watch")].copy()
        if rejected_df.empty:
            rejected_trade_reasons.extend(["_No rejected trade rows available._", ""])
        else:
            rejected_df["Report Visibility"] = rejected_df.apply(
                lambda row: "Shown in main report" if candidate_report_key(row) in shown_keys else "Trimmed from main report",
                axis=1,
            )
            rejected_df["Ticker"] = rejected_df["ticker"].fillna("").astype(str)
            rejected_df["Strategy Type"] = rejected_df["strategy"].fillna("").astype(str)
            rejected_df["Expiry"] = rejected_df["expiry"].fillna("").astype(str)
            rejected_df["Conviction %"] = rejected_df["conviction"].apply(
                lambda v: f"{float(v):.0f}%" if np.isfinite(fnum(v)) else ""
            )
            rejected_df["Setup Likelihood"] = rejected_df["likelihood_strength"].fillna("").astype(str)
            rejected_df["Stage-1 Diagnostics"] = rejected_df["stage1_diagnostics"].fillna("").astype(str)
            rejected_df["Reject Reasons"] = rejected_df["approval_blockers"].fillna("").astype(str)
            rejected_df["Notes"] = rejected_df["notes"].fillna("").astype(str)
            rejected_df["_conv_num"] = pd.to_numeric(rejected_df["conviction"], errors="coerce").fillna(-1.0)
            rejected_df = rejected_df.sort_values(
                ["Report Visibility", "_conv_num", "Ticker", "Expiry"],
                ascending=[True, False, True, True],
            )
            reject_cols = [
                "Report Visibility",
                "Ticker",
                "Strategy Type",
                "Expiry",
                "Conviction %",
                "Setup Likelihood",
                "Reject Reasons",
            ]
            rejected_trade_reasons.extend([markdown_table(rejected_df, reject_cols), ""])

    def _approved_count_from_decision_csv(path: Path):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            return None
        if df.empty:
            return 0
        cols = {str(c).strip().lower(): c for c in df.columns}
        book_col = cols.get("execution_book") or cols.get("execution book")
        if book_col:
            return int(df[book_col].astype(str).str.strip().isin(["Core", "Tactical"]).sum())
        approved_col = cols.get("approved")
        if approved_col:
            return int(df[approved_col].astype(str).str.upper().eq("YES").sum())
        category_col = cols.get("category")
        if category_col:
            return int(df[category_col].astype(str).str.startswith("Approved").sum())
        return None

    def _trailing_skip_streak_dates() -> list[str]:
        root_dir = base.parent
        candidates: list[Path] = []
        collect_root = root_dir / "out" / "daily_pipeline_collect_baseline"
        if collect_root.exists():
            candidates.extend(collect_root.glob("20??-??-??/trade_decision_book_all_*.csv"))
        legacy_out = root_dir / r"c:\uw_root\out"
        if legacy_out.exists():
            candidates.extend(legacy_out.glob("trade_decision_book_all_*.csv"))
        if out_dir.exists():
            candidates.extend(out_dir.glob("trade_decision_book_all_*.csv"))
        date_to_approved: dict[str, int] = {}
        for path in sorted(candidates):
            match = re.search(r"(20\d{2}-\d{2}-\d{2})", str(path))
            if not match:
                continue
            dtext = match.group(1)
            if dtext > asof_str:
                continue
            count = _approved_count_from_decision_csv(path)
            if count is None:
                continue
            date_to_approved[dtext] = int(count)
        date_to_approved[asof_str] = int(approved_count)
        streak: list[str] = []
        for dtext in sorted(date_to_approved.keys(), reverse=True):
            if date_to_approved[dtext] == 0:
                streak.append(dtext)
                continue
            break
        return list(reversed(streak))

    def _short_text(value: object, limit: int = 150) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _display_strike(value: object) -> str:
        parsed = fnum(value)
        if not np.isfinite(parsed):
            return ""
        if abs(parsed - round(parsed)) < 0.001:
            return str(int(round(parsed)))
        return f"{parsed:.2f}".rstrip("0").rstrip(".")

    def _display_option_leg(symbol: object) -> str:
        parsed = parse_occ(re.sub(r"\s+", "", str(symbol or "")).upper())
        if not parsed:
            return str(symbol or "").strip()
        _root, _expiry, right, strike = parsed
        return f"{_display_strike(strike)}{right}"

    def _display_strategy_legs(strategy: object, expiry: object, lead_symbol: object, pair_symbol: object) -> str:
        strategy_text = str(strategy or "").strip()
        expiry_text = str(expiry or "").strip()[:10]
        lead_leg = _display_option_leg(lead_symbol)
        pair_leg = _display_option_leg(pair_symbol)
        if "Credit" in strategy_text:
            legs = f"Sell {lead_leg} / Buy {pair_leg}"
        else:
            legs = f"Buy {lead_leg} / Sell {pair_leg}"
        return f"{strategy_text} {expiry_text}: {legs}".strip()

    def _compact_likelihood(value: object) -> str:
        text = str(value or "").strip()
        verdict = "LS" if "LOW_SAMPLE" in text else "FAIL" if "FAIL" in text else "PASS" if "PASS" in text else ""
        edge_match = re.search(r"edge\s+([+-]?\d+(?:\.\d+)?)%", text)
        n_match = re.search(r"n=(\d+)", text)
        parts = [verdict]
        if edge_match:
            parts.append(f"{float(edge_match.group(1)):+.1f}%")
        if n_match:
            parts.append(f"n{n_match.group(1)}")
        return " ".join([p for p in parts if p]) or _short_text(text, 28)

    def _compact_blockers(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        tokens = [x.strip() for x in re.split(r"[;,]", text) if x.strip()]
        labels: list[str] = []
        for token in tokens:
            label = ""
            if "stage1_conviction_below_yes_good" in token:
                match = re.search(r"(\d+)<(\d+)", token)
                label = f"S1 {match.group(1)}<{match.group(2)}" if match else "S1 low"
            elif "stage1_flow_weak_or_ambiguous" in token:
                label = "weak flow"
            elif "stage1_contract_flow_contra" in token or "contract_flow_contra" in token:
                label = "contract contra"
            elif "stage1_contract_flow_weak_or_ambiguous" in token or "contract_flow_weak_or_ambiguous" in token:
                label = "contract weak"
            elif "stage1_high_iv_debit_watch_only" in token:
                label = "high IV"
            elif "likelihood_verdict:LOW_SAMPLE" in token or "likelihood_strength_unranked:Low Sample" in token:
                label = "low sample"
            elif token.startswith("signals_below"):
                label = "n<min"
            elif token.startswith("edge_below") or token == "edge_below_threshold":
                label = "edge<thresh"
            elif token.startswith("likelihood_verdict:FAIL") or token == "likelihood_fail":
                label = "likelihood fail"
            elif "fire_gex_pinned" in token:
                label = "GEX pinned"
            elif "gex_volatile" in token:
                label = "GEX volatile"
            elif "gex_missing" in token:
                label = "GEX missing"
            elif "dte_too_long" in token or "dte_near_limit" in token:
                label = "DTE long"
            elif "rr_weak" in token:
                label = "RR weak"
            elif "market_regime_caution" in token:
                label = "market caution"
            elif token:
                label = token.replace("_", " ")
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= 3:
                break
        return "; ".join(labels) if labels else _short_text(text, 35)

    def _compact_strategy(value: object) -> str:
        text = str(value or "").strip()
        return {
            "Bull Call Debit": "Bull Call",
            "Bear Put Debit": "Bear Put",
            "Iron Condor": "IC",
            "Long Iron Condor": "Long IC",
        }.get(text, text)

    def _compact_trade_text(strategy: object, setup: object, limit: int = 48) -> str:
        strat = _compact_strategy(strategy)
        setup_text = str(setup or "").strip()
        setup_text = setup_text.replace("Buy ", "B ")
        setup_text = setup_text.replace("Sell ", "S ")
        setup_text = setup_text.replace(" + ", " + ")
        setup_text = setup_text.replace(".00", "")
        setup_text = re.sub(r"\s+", " ", setup_text).strip()
        text = f"{strat} {setup_text}".strip()
        return text

    def _compact_reason_text(value: object, limit: int = 64) -> str:
        text = str(value or "").strip()
        text = text.replace("UW flow is not strong enough yet.", "Weak UW flow.")
        text = text.replace("Historical sample is too small to trust.", "Low sample.")
        text = text.replace("Historical edge is below the required minimum.", "Edge below floor.")
        text = text.replace("Historical likelihood is not strong enough.", "Likelihood weak.")
        text = text.replace("Expiry is farther out than this rulebook prefers.", "DTE long.")
        text = text.replace("Current debit", "Debit")
        text = text.replace("Current credit", "Credit")
        text = text.replace("is above max buy price", "> max")
        text = text.replace("is below required credit", "< min")
        text = text.replace("; do not chase.", "; no chase.")
        text = text.replace("; do not sell here.", "; no sell.")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _trade_card_lines(row: pd.Series, *, include_needs: bool = False) -> list[str]:
        decision = str(row.get("Decision", "") or "").strip()
        ticker = str(row.get("Ticker", "") or "").strip().upper()
        trade = str(row.get("Trade", "") or "").strip()
        exp = str(row.get("Exp", "") or "").strip()
        now = str(row.get("Now", "") or "").strip()
        limit = str(row.get("Limit", "") or "").strip()
        flow = str(row.get("Flow", "") or "").strip()
        be = str(row.get("BE", "") or "").strip()
        edge = str(row.get("Edge", "") or "").strip()
        gex = str(row.get("GEX", "") or "").strip()
        why = str(row.get("Why", "") or "").strip()
        lines = [
            f"- **{decision} | {ticker} | {trade} | Exp {exp} | now {now}; limit {limit}**",
            f"  - Flow {flow}; BE {be}; Edge {edge}; GEX {gex}; Why: {why or 'n/a'}",
        ]
        if include_needs:
            needs = str(row.get("Needs", "") or "").strip()
            if needs:
                lines.append(f"  - Needs: {needs}")
        return lines

    def _trade_table_lines(frame: pd.DataFrame, *, include_needs: bool = False) -> list[str]:
        def cell(value: object) -> str:
            text = str(value or "").strip()
            text = text.replace("|", "/")
            text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
            return text

        def ticker_cell(value: object) -> str:
            text = str(value or "").strip().upper()
            # Pad with non-breaking spaces so Markdown previewers stop
            # squeezing this column into wrapped ticker symbols.
            return text + "\u00a0" * max(0, 6 - len(text))

        headers = ["Decision", "Ticker", "Trade", "Price", "Read", "Why / Needs"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|:--|:--|:--|:--|:--|:--|",
        ]
        for _, row in frame.iterrows():
            exp = str(row.get("Exp", "") or "").strip()
            now = str(row.get("Now", "") or "").strip()
            limit = str(row.get("Limit", "") or "").strip()
            flow = str(row.get("Flow", "") or "").strip()
            be = str(row.get("BE", "") or "").strip()
            edge = str(row.get("Edge", "") or "").strip()
            gex = str(row.get("GEX", "") or "").strip()
            why = str(row.get("Why", "") or "").strip()
            needs = str(row.get("Needs", "") or "").strip()
            price = f"Exp {exp}; Now {now}; Limit {limit}"
            read = f"Flow {flow}; BE {be}; Edge {edge}; GEX {gex}"
            why_needs = f"Why: {why or 'n/a'}"
            if include_needs and needs:
                why_needs += f"; Needs: {needs}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        cell(row.get("Decision", "")),
                        ticker_cell(row.get("Ticker", "")),
                        cell(row.get("Trade", "")),
                        cell(price),
                        cell(read),
                        cell(why_needs),
                    ]
                )
                + " |"
            )
        return lines

    def _board_quality_score(row: pd.Series) -> float:
        flow_text = str(row.get("Flow", "") or "")
        be_text = str(row.get("BE", "") or "")
        edge_text = str(row.get("Edge", "") or "")
        score = 0.0
        flow_match = re.search(r"(-?\d+(?:\.\d+)?)", flow_text)
        if flow_match:
            score += fnum(flow_match.group(1)) * 0.45
        if "Contra" in flow_text:
            score -= 35.0
        edge_match = re.search(r"([+-]?\d+(?:\.\d+)?)%", edge_text)
        if edge_match:
            score += fnum(edge_match.group(1)) * 0.75
        if "PASS" in edge_text:
            score += 12.0
        if "LOW_SAMPLE" in edge_text:
            score -= 10.0
        if "FAIL" in edge_text:
            score -= 25.0
        be_match = re.search(r"(\d+(?:\.\d+)?)%", be_text)
        if be_match:
            score -= fnum(be_match.group(1)) * 2.0
        if "Through" in be_text or "Easier" in be_text:
            score += 10.0
        elif "Moderate" in be_text:
            score += 3.0
        elif "Harder" in be_text:
            score -= 4.0
        elif "Far" in be_text:
            score -= 18.0
        return score


    def _decision_key(value: object) -> str:
        text = str(value or "").strip().upper()
        if "PILOT" in text and "ENTER" in text:
            return "PILOT ENTER"
        if "PILOT" in text and ("TARGET" in text or "WAIT" in text or "LIMIT" in text):
            return "PILOT LIMIT"
        if "PILOT" in text and "REVIEW" in text:
            return "PILOT REVIEW"
        if "PILOT" in text:
            return "PILOT REVIEW"
        if "HIGH" in text and "ENTER" in text:
            return "HIGH ENTER"
        if "HIGH" in text and ("TARGET" in text or "WAIT" in text or "LIMIT" in text):
            return "HIGH LIMIT"
        if "HIGH" in text and "REVIEW" in text:
            return "HIGH REVIEW"
        if "MEDIUM" in text and ("TARGET" in text or "WAIT" in text or "LIMIT" in text):
            return "MEDIUM LIMIT"
        if "MEDIUM" in text and "REVIEW" in text:
            return "MEDIUM REVIEW"
        if "MEDIUM" in text:
            return "MEDIUM REVIEW"
        if "INCOME" in text and "ENTER" in text:
            return "INCOME ENTER"
        if "INCOME" in text and ("TARGET" in text or "WAIT" in text or "LIMIT" in text):
            return "INCOME LIMIT"
        if "INCOME" in text and "REVIEW" in text:
            return "INCOME REVIEW"
        if "INCOME" in text:
            return "INCOME REVIEW"
        if "WATCHLIST" in text or text == "WATCH":
            return "WATCHLIST"
        if "ENTER" in text:
            return "ENTER"
        if "TARGET" in text or "WAIT" in text:
            return "WAIT FOR PRICE"
        if "REVIEW ONLY" in text:
            return "REVIEW ONLY"
        if "REVIEW" in text:
            return "REVIEW"
        if "SKIP" in text:
            return "SKIP"
        return text or "UNKNOWN"

    def _decision_label(value: object) -> str:
        key = _decision_key(value)
        if key == "HIGH ENTER":
            return "🟢 HIGH ENTER"
        if key == "HIGH LIMIT":
            return "🟢 HIGH LIMIT"
        if key == "HIGH REVIEW":
            return "🟢 HIGH REVIEW"
        if key == "MEDIUM LIMIT":
            return "🟡 MEDIUM LIMIT"
        if key == "MEDIUM REVIEW":
            return "🟠 MEDIUM REVIEW"
        if key == "INCOME ENTER":
            return "🟣 INCOME ENTER"
        if key == "INCOME LIMIT":
            return "🟣 INCOME LIMIT"
        if key == "INCOME REVIEW":
            return "🟣 INCOME REVIEW"
        if key == "PILOT ENTER":
            return "🟠 PILOT ENTER"
        if key == "PILOT LIMIT":
            return "🟠 PILOT LIMIT"
        if key == "PILOT REVIEW":
            return "🟠 PILOT REVIEW"
        if key == "WATCHLIST":
            return "⚪ WATCHLIST"
        if key == "ENTER":
            return "🟢 ENTER"
        if key == "WAIT FOR PRICE":
            return "🟡 WAIT FOR PRICE"
        if key == "REVIEW ONLY":
            return "🟠 REVIEW ONLY"
        if key == "REVIEW":
            return "🟠 REVIEW"
        if key == "SKIP":
            return "🔴 SKIP"
        return f"⚪ {key}" if key else "⚪ UNKNOWN"

    def _flow_conviction_label(row: pd.Series) -> str:
        conv = fnum(row.get("conviction"))
        if not np.isfinite(conv):
            conv = fnum(str(row.get("Conviction %", "") or "").replace("%", ""))
        contract_flow = str(row.get("contract_flow_confirmation", "") or row.get("Contract Flow", "") or "").strip().lower()
        flow_conf = str(row.get("flow_confidence", "") or "").strip().lower()
        stage1_diag = str(row.get("stage1_diagnostics", "") or row.get("Stage-1 Diagnostics", "") or "").strip().lower()
        if "contra" in contract_flow or "contra" in stage1_diag or "conflict" in stage1_diag:
            label = "🟥 Contra"
        elif "weak" in contract_flow or "ambiguous" in contract_flow or "weak" in stage1_diag or flow_conf in {"weak", "low"}:
            label = "🟥 Weak"
        elif np.isfinite(conv) and conv >= 75:
            label = "🟩 High"
        elif np.isfinite(conv) and conv >= 60:
            label = "🟨 Medium-High"
        elif np.isfinite(conv) and conv >= 45:
            label = "🟧 Medium"
        elif np.isfinite(conv):
            label = "🟥 Low"
        else:
            label = "⚪ Unknown"
        return f"{label} {conv:.0f}" if np.isfinite(conv) else label

    def _breakeven_difficulty_label(row: pd.Series) -> str:
        strategy = str(row.get("strategy", "") or row.get("Strategy", "") or row.get("Strategy Type", "") or "").strip()
        spot = fnum(row.get("spot_live_effective"))
        if not np.isfinite(spot):
            spot = fnum(row.get("spot_asof_close"))
        if not np.isfinite(spot):
            spot_vs_be = str(row.get("Spot vs BE", "") or "").strip()
            m = re.search(r"([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)", spot_vs_be)
            if m:
                spot = fnum(m.group(1))
        be = fnum(row.get("breakeven"))
        if not np.isfinite(be):
            spot_vs_be = str(row.get("Spot vs BE", "") or "").strip()
            m = re.search(r"([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)", spot_vs_be)
            if m:
                be = fnum(m.group(2))
        if not np.isfinite(be):
            long_strike = fnum(row.get("long_strike"))
            live_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(live_net):
                live_net = fnum(row.get("live_net_mark"))
            if np.isfinite(long_strike) and np.isfinite(live_net):
                if strategy == "Bull Call Debit":
                    be = long_strike + live_net
                elif strategy == "Bear Put Debit":
                    be = long_strike - live_net
        if not np.isfinite(spot) or spot <= 0 or not np.isfinite(be):
            return "⚪ N/A"
        if strategy == "Bear Put Debit":
            move_pct = (spot - be) / spot
        elif strategy == "Bull Call Debit":
            move_pct = (be - spot) / spot
        else:
            raw_dist = fnum(row.get("breakeven_distance_pct"))
            move_pct = abs(raw_dist) if np.isfinite(raw_dist) else abs(be - spot) / spot
        if move_pct <= 0:
            label = "🟩 Through BE"
        elif move_pct <= 0.025:
            label = "🟩 Easier"
        elif move_pct <= 0.05:
            label = "🟨 Moderate"
        elif move_pct <= 0.08:
            label = "🟧 Harder"
        else:
            label = "🟥 Far"
        return f"{label} {move_pct:.1%}"

    def daily_trade_board_section(approved_df: pd.DataFrame, watch_df: pd.DataFrame) -> list[str]:
        section = [
            "## Daily Directional Entries",
            "",
            "Strict daily directional entries only. Watchlist rows are shown in the next section.",
            "",
        ]
        rows: list[dict] = []
        if not approved_df.empty:
            for _, row in approved_df.iterrows():
                live_action = str(row.get("Live Action", "") or "").strip()
                book_label = str(row.get("Execution Book", "") or "").strip()
                decision_prefix = (
                    "INCOME"
                    if book_label == "Income"
                    else "MEDIUM"
                    if book_label == "Medium"
                    else "PILOT"
                    if book_label == "Pilot"
                    else "HIGH"
                )
                rows.append(
                    {
                        "Decision": f"{decision_prefix} {live_action or 'REVIEW'}",
                        "Ticker": str(row.get("Ticker", "") or "").strip().upper(),
                        "Trade": _compact_trade_text(row.get("Strategy Type", ""), row.get("Strike Setup", "")),
                        "Exp": str(row.get("Expiry", "") or "").strip(),
                        "Now": str(row.get("Net Credit/Debit", "") or "").strip(),
                        "Limit": str(row.get("Entry Gate", "") or "").strip(),
                        "Flow": _flow_conviction_label(row),
                        "BE": _breakeven_difficulty_label(row),
                        "Edge": _compact_likelihood(row.get("Setup Likelihood", "")),
                        "GEX": str(row.get("GEX Regime", "") or "").strip(),
                        "Why": _compact_reason_text(
                            str(row.get("Live Check Reason", "") or "").strip() or "Approved by daily gates."
                        ),
                    }
                )
        if not watch_df.empty:
            for _, row in watch_df.iterrows():
                rows.append(
                    {
                        "Decision": str(row.get("Decision", "") or "").strip(),
                        "Ticker": str(row.get("Ticker", "") or "").strip().upper(),
                        "Trade": _compact_trade_text(row.get("Strategy", ""), row.get("Setup", "")),
                        "Exp": str(row.get("Exp", "") or "").strip(),
                        "Now": str(row.get("Current Price", "") or "").strip(),
                        "Limit": str(row.get("Required Price", "") or "").strip(),
                        "Flow": str(row.get("Flow Conviction", "") or "").strip(),
                        "BE": str(row.get("Breakeven Difficulty", "") or "").strip(),
                        "Edge": str(row.get("Edge", "") or "").strip(),
                        "GEX": str(row.get("GEX", "") or "").strip(),
                        "Why": _compact_reason_text(row.get("Plain-English Reason", "")),
                    }
                )
        if not rows:
            section.extend(["_No daily directional ENTER rows._", ""])
            return section
        board = pd.DataFrame(rows)
        decision_rank = {
            "HIGH ENTER": 0,
            "HIGH LIMIT": 1,
            "HIGH REVIEW": 2,
            "MEDIUM LIMIT": 3,
            "MEDIUM REVIEW": 4,
            "INCOME ENTER": 5,
            "INCOME LIMIT": 6,
            "INCOME REVIEW": 7,
            "PILOT ENTER": 8,
            "PILOT LIMIT": 9,
            "PILOT REVIEW": 10,
            "ENTER": 0,
            "TARGET": 1,
            "WAIT FOR PRICE": 11,
            "REVIEW": 12,
            "REVIEW ONLY": 12,
            "WATCHLIST": 13,
            "SKIP": 13,
        }
        board["_decision_key"] = board["Decision"].map(_decision_key)
        board["_rank"] = board["_decision_key"].map(lambda x: decision_rank.get(str(x).strip(), 9))
        board["_score"] = board.apply(_board_quality_score, axis=1)
        board = board.sort_values(["_rank", "_score", "Ticker"], ascending=[True, False, True]).drop(columns=["_rank", "_decision_key", "_score"]).head(15)
        board["Decision"] = board["Decision"].map(_decision_label)
        section.extend(_trade_table_lines(board))
        section.append("")
        return section

    def review_candidate_pool_section(source_df: pd.DataFrame, limit: int = 12) -> tuple[list[str], str, int, pd.DataFrame]:
        """Expose non-approved but reviewable setups without loosening approvals.

        This is deliberately not an approval override.  It is a trader-review
        workbench for rows that have already made it through Stage-2 pricing and
        GEX enrichment, but were blocked by quality/book floors.  Safety and
        contra-flow failures stay out of this table.
        """
        review_csv = out_dir / f"review_candidate_pool_{asof_str}.csv"

        section = [
            "## Watchlist",
            "",
            "These are **not approved Medium/High trades**. They are shown only so attractive rejects do not disappear; do not treat them as the action queue.",
            "",
        ]
        if source_df.empty:
            section.extend(["_No review candidates available._", ""])
            return section, str(review_csv), 0, pd.DataFrame()

        safety_exact = {
            "invalidation_warning",
            "spot_live_missing",
            "spot_drift_unknown",
            "gex_missing",
            "gex_source_stale",
            "shield_gex_volatile",
            "ic_gex_volatile",
            "shield_delta_missing",
            "stage1_flow_conflicted",
            "market_regime_block",
        }
        safety_prefixes = (
            "live_status:",
            "contract_flow_contra",
            "stage1_contract_flow_contra",
            "contract_flow_directional",
            "stage1_contract_flow_directional",
            "contract_flow_weak_or_ambiguous",
            "stage1_contract_flow_weak_or_ambiguous",
            "flow_not_confirmed",
            "flow_too_directional_for_ic",
            "fire_delta",
            "shield_delta_fail",
            "gex_source_fallback_uncertain",
            "bull_call_otm_too_far",
            "bear_put_otm_too_far",
            "flow_contra_",
        )

        def _tokens(row: pd.Series) -> list[str]:
            raw = ";".join(
                [
                    str(row.get("approval_blockers", "") or ""),
                    str(row.get("hard_blockers", "") or ""),
                    str(row.get("quality_blockers", "") or ""),
                    str(row.get("book_blockers", "") or ""),
                ]
            )
            out: list[str] = []
            seen = set()
            for token in [x.strip() for x in raw.split(";") if x.strip()]:
                if token not in seen:
                    out.append(token)
                    seen.add(token)
            return out

        def _has_safety_blocker(tokens: list[str]) -> bool:
            for token in tokens:
                if token in safety_exact:
                    return True
                if any(token.startswith(prefix) for prefix in safety_prefixes):
                    return True
            return False

        def _entry_decision(row: pd.Series) -> str:
            live_status = str(row.get("live_status", "") or "").strip()
            live_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(live_net):
                live_net = fnum(row.get("live_net_mark"))
            gate_text = str(row.get("entry_gate", "") or "").strip()
            gate_match = re.match(r"^\s*(<=|>=)\s*([0-9]*\.?[0-9]+)", gate_text)
            gate_near_miss = bool(row.get("gate_near_miss")) if pd.notna(row.get("gate_near_miss")) else False
            if gate_match and np.isfinite(live_net):
                op = gate_match.group(1)
                gate_value = fnum(gate_match.group(2))
                if np.isfinite(gate_value):
                    if (op == "<=" and live_net <= gate_value) or (op == ">=" and live_net >= gate_value):
                        return "MEDIUM REVIEW"
                    return "MEDIUM LIMIT"
            if live_status == "fails_live_entry_gate" or gate_near_miss:
                return "MEDIUM LIMIT"
            return "MEDIUM REVIEW"

        def _setup_text(row: pd.Series) -> str:
            strategy = str(row.get("strategy", "") or "").strip()
            return strike_setup(
                strategy,
                row.get("long_strike"),
                row.get("short_strike"),
                row.get("width"),
                long_put_strike=row.get("long_put_strike"),
                short_put_strike=row.get("short_put_strike"),
                short_call_strike=row.get("short_call_strike"),
                long_call_strike=row.get("long_call_strike"),
            )

        def _entry_text(row: pd.Series) -> str:
            live_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(live_net):
                live_net = fnum(row.get("live_net_mark"))
            gate = str(row.get("entry_gate", "") or "").strip()
            if np.isfinite(live_net):
                return f"{live_net:.2f} vs {gate}"
            return gate

        def _current_price_text(row: pd.Series) -> str:
            live_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(live_net):
                live_net = fnum(row.get("live_net_mark"))
            net_type = str(row.get("net_type", "") or "").strip().lower()
            suffix = "credit" if net_type == "credit" else "debit" if net_type == "debit" else ""
            return f"{live_net:.2f} {suffix}".strip() if np.isfinite(live_net) else "N/A"

        def _required_price_text(row: pd.Series) -> str:
            gate = str(row.get("entry_gate", "") or "").strip()
            gate = gate.replace("<=", "max").replace(">=", "min").replace(" db", " debit").replace(" cr", " credit")
            return re.sub(r"\s+", " ", gate).strip()

        def _edge_text(row: pd.Series) -> str:
            verdict = str(row.get("verdict", "") or "").strip().upper()
            edge = fnum(row.get("edge_pct"))
            signals = fnum(row.get("signals"))
            parts = [verdict]
            if np.isfinite(edge):
                parts.append(f"{edge:+.1f}%")
            if np.isfinite(signals):
                parts.append(f"n={int(signals)}")
            return " ".join([p for p in parts if p])

        def _plain_reason_text(row: pd.Series, tokens: list[str], decision: str) -> str:
            reasons: list[str] = []
            live_net = fnum(row.get("live_net_bid_ask"))
            if not np.isfinite(live_net):
                live_net = fnum(row.get("live_net_mark"))
            gate_text = str(row.get("entry_gate", "") or "").strip()
            gate_match = re.match(r"^\s*(<=|>=)\s*([0-9]*\.?[0-9]+)\s*(db|cr)?", gate_text, flags=re.I)
            if decision in {"WAIT FOR PRICE", "MEDIUM LIMIT"} and gate_match and np.isfinite(live_net):
                op = gate_match.group(1)
                gate_value = fnum(gate_match.group(2))
                unit = "credit" if str(gate_match.group(3) or "").lower() == "cr" else "debit"
                if op == "<=":
                    reasons.append(f"Current debit {live_net:.2f} is above max buy price {gate_value:.2f}; do not chase.")
                else:
                    reasons.append(f"Current credit {live_net:.2f} is below required credit {gate_value:.2f}; do not sell here.")
            for token in tokens:
                if token.startswith("stage1_conviction_below_yes_good") or token == "stage1_flow_weak_or_ambiguous":
                    reasons.append("UW flow is not strong enough yet.")
                elif token.startswith("stage1_contract_flow_weak_or_ambiguous") or token.startswith("contract_flow_weak_or_ambiguous"):
                    reasons.append("The selected option legs do not have enough direct flow confirmation.")
                elif token.startswith("bull_call_dte_too_long"):
                    reasons.append("Expiry is beyond the rulebook's DTE tolerance.")
                elif token.startswith("bull_call_dte_near_limit"):
                    reasons.append("Expiry is slightly past the preferred DTE target.")
                elif token.startswith("tactical_signals_below") or token.startswith("signals_below"):
                    reasons.append("Historical sample is too small to trust.")
                elif token.startswith("likelihood_verdict:LOW_SAMPLE") or "Low Sample" in token:
                    reasons.append("Historical sample is too small to trust.")
                elif token.startswith("edge_below") or token.startswith("tactical_edge_below"):
                    reasons.append("Historical edge is below the required minimum.")
                elif token.startswith("likelihood_strength_below") or token.startswith("likelihood_strength_unranked"):
                    reasons.append("Historical likelihood is not strong enough.")
                elif token.startswith("market_regime_caution"):
                    reasons.append("Market regime is only moderate, so this is not a full-size setup.")
                elif token.startswith("gex_volatile_breakout") or token.startswith("fire_gex_pinned"):
                    reasons.append("GEX context may work against the move.")
                elif token.startswith("bull_call_rr_weak") or token.startswith("live_rr_weak"):
                    reasons.append("Reward/risk is not strong enough.")
                if len(reasons) >= 3:
                    break
            if not reasons:
                reasons.append("Review only; daily approval gates did not clear.")
            return " ".join(dict.fromkeys(reasons))

        def _needs_text(row: pd.Series, tokens: list[str]) -> str:
            needs: list[str] = []
            for token in tokens:
                if token.startswith("stage1_conviction_below_yes_good"):
                    needs.append("flow score > floor")
                elif token == "stage1_flow_weak_or_ambiguous":
                    needs.append("cleaner directional flow")
                elif token.startswith("stage1_contract_flow_weak_or_ambiguous") or token.startswith("contract_flow_weak_or_ambiguous"):
                    needs.append("direct leg-flow confirmation")
                elif token.startswith("bull_call_dte_too_long"):
                    needs.append("shorter DTE")
                elif token.startswith("bull_call_rr_weak") or token.startswith("live_rr_weak"):
                    needs.append("better reward/risk")
                elif token.startswith("tactical_conviction_below"):
                    needs.append("higher conviction")
                elif token.startswith("tactical_signals_below") or token.startswith("signals_below"):
                    needs.append("larger historical sample")
                elif token.startswith("tactical_edge_below") or token.startswith("edge_below"):
                    needs.append("higher edge")
                elif token.startswith("likelihood_verdict") or token.startswith("likelihood_strength"):
                    needs.append("better likelihood verdict")
                elif token.startswith("market_regime_caution"):
                    needs.append("stronger tape")
                elif token.startswith("fire_gex_pinned") or token.startswith("gex_volatile_breakout"):
                    needs.append("better GEX support")
                if len(needs) >= 4:
                    break
            if not needs:
                needs.append("manual review only")
            return "; ".join(dict.fromkeys(needs))

        rows: list[dict] = []
        for _, row in source_df.iterrows():
            if bool(row.get("approved")):
                continue
            tokens = _tokens(row)
            if _has_safety_blocker(tokens):
                continue
            verdict = str(row.get("verdict", "") or "").strip().upper()
            edge = fnum(row.get("edge_pct"))
            signals = fnum(row.get("signals"))
            strategy = str(row.get("strategy", "") or "").strip()
            track = str(row.get("track", "") or "").strip().upper()
            is_bear = strategy in {"Bear Put Debit", "Bear Call Credit"}
            min_edge_req = (
                min_edge_pct_bear
                if is_bear
                else min_edge_pct_shield
                if track == "SHIELD"
                else fnum(approval_cfg.get("min_edge_pct", 8.0))
            )
            if verdict != "PASS":
                continue
            if not np.isfinite(edge) or edge < min_edge_req:
                continue
            if not np.isfinite(signals) or signals < min_signals:
                continue
            if str(row.get("contract_flow_confirmation", "") or "").strip().lower() in {
                "contra",
                "directional",
                "weak_or_ambiguous",
                "unknown",
            }:
                continue
            if strategy in {"Long Iron Condor"}:
                continue
            decision = _entry_decision(row).replace("MEDIUM", "WATCHLIST")
            rows.append(
                {
                    "Decision": decision,
                    "Ticker": str(row.get("ticker", "") or "").strip().upper(),
                    "Strategy": _compact_strategy(strategy),
                    "Setup": _short_text(_setup_text(row), 58),
                    "Trade": _compact_trade_text(strategy, _setup_text(row)),
                    "Exp": str(row.get("expiry", "") or "").strip()[:10],
                    "Current Price": _current_price_text(row),
                    "Required Price": _required_price_text(row),
                    "Now": _current_price_text(row),
                    "Limit": _required_price_text(row),
                    "Conv": f"{fnum(row.get('conviction')):.0f}%" if np.isfinite(fnum(row.get("conviction"))) else "",
                    "Flow Conviction": _flow_conviction_label(row),
                    "Breakeven Difficulty": _breakeven_difficulty_label(row),
                    "Flow": _flow_conviction_label(row),
                    "BE": _breakeven_difficulty_label(row),
                    "Edge": _edge_text(row),
                    "GEX": str(row.get("gex_regime", "") or "").strip(),
                    "Plain-English Reason": _plain_reason_text(row, tokens, decision),
                    "Why": _compact_reason_text(_plain_reason_text(row, tokens, decision)),
                    "Needs": _needs_text(row, tokens),
                    "_rank": fnum(row.get("_display_rank_score")),
                    "_edge": edge,
                    "_signals": signals,
                }
            )

        if not rows:
            empty_df = pd.DataFrame(
                columns=[
                    "Decision",
                    "Ticker",
                    "Trade",
                    "Exp",
                    "Now",
                    "Limit",
                    "Flow",
                    "BE",
                    "Edge",
                    "GEX",
                    "Why",
                    "Needs",
                ]
            )
            try:
                empty_df.to_csv(review_csv, index=False)
            except Exception:
                pass
            section.extend(["_No review candidates available after safety/contra exclusions._", ""])
            return section, str(review_csv), 0, pd.DataFrame()

        review_df = pd.DataFrame(rows)
        review_df["_decision_key"] = review_df["Decision"].map(_decision_key)
        decision_rank = {
            "HIGH ENTER": 0,
            "HIGH LIMIT": 1,
            "HIGH REVIEW": 2,
            "MEDIUM LIMIT": 3,
            "MEDIUM REVIEW": 4,
            "INCOME ENTER": 5,
            "INCOME LIMIT": 6,
            "INCOME REVIEW": 7,
            "ENTER": 0,
            "TARGET": 1,
            "WAIT FOR PRICE": 8,
            "REVIEW": 9,
            "REVIEW ONLY": 9,
            "WATCHLIST": 10,
            "SKIP": 10,
        }
        review_df["_status_rank"] = review_df["_decision_key"].map(lambda x: decision_rank.get(str(x).strip(), 9))
        review_df["_score"] = review_df.apply(_board_quality_score, axis=1)
        review_df = review_df.sort_values(
            ["_status_rank", "_score", "_rank", "_edge", "_signals"],
            ascending=[True, False, False, False, False],
        ).head(limit)
        out_review_df = review_df.drop(columns=[c for c in ["_rank", "_edge", "_signals"] if c in review_df.columns])
        out_review_df = out_review_df.drop(columns=[c for c in ["_decision_key", "_status_rank", "_score"] if c in out_review_df.columns])
        out_review_df["Decision"] = out_review_df["Decision"].map(_decision_label)
        try:
            out_review_df.to_csv(review_csv, index=False)
        except Exception:
            pass
        section.extend(_trade_table_lines(out_review_df, include_needs=True))
        section.append("")
        section.append(f"Review candidate CSV: {review_csv}")
        section.append("")
        return section, str(review_csv), int(len(out_review_df)), out_review_df

    def _daily_near_miss_frame(limit: int = 5) -> pd.DataFrame:
        if out_df.empty or "Execution Book" not in out_df.columns:
            return pd.DataFrame()
        nm = out_df[out_df["Execution Book"].astype(str).eq("Watch")].copy()
        if nm.empty:
            return pd.DataFrame()
        nm["_conv_num"] = pd.to_numeric(
            nm.get("Conviction %", pd.Series("", index=nm.index)).astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        ).fillna(-1.0)
        nm = nm.sort_values(["_conv_num", "#"], ascending=[False, True]).head(limit)
        rows = []
        for _, row in nm.iterrows():
            ticker = str(row.get("Ticker", "") or "").strip().upper()
            rows.append(
                {
                    "#": str(row.get("#", "") or "").strip(),
                    "St": "🔴",
                    "Ticker": ticker,
                    "Strategy": _compact_strategy(row.get("Strategy Type", "")),
                    "Legs": _short_text(row.get("Strike Setup", ""), 44),
                    "Exp": str(row.get("Expiry", "") or "").strip(),
                    "Conv": str(row.get("Conviction %", "") or "").strip(),
                    "Edge": _compact_likelihood(row.get("Setup Likelihood", "")),
                    "Why": _compact_blockers(row.get("Stage-1 Diagnostics", "") or row.get("Watch Reason Flags", "")),
                    "Detail": " ".join(
                        [
                            str(row.get("Strategy Type", "") or "").strip(),
                            str(row.get("Strike Setup", "") or "").strip(),
                            str(row.get("Expiry", "") or "").strip(),
                        ]
                    ).strip(),
                    "Daily Setup": " ".join(
                        [
                            str(row.get("Strategy Type", "") or "").strip(),
                            str(row.get("Strike Setup", "") or "").strip(),
                            str(row.get("Expiry", "") or "").strip(),
                        ]
                    ).strip(),
                    "Daily Blockers": str(row.get("Stage-1 Diagnostics", "") or row.get("Watch Reason Flags", "") or ""),
                }
            )
        return pd.DataFrame(rows)

    def _daily_near_miss_detail_bullets(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["_No daily near-miss rows available._"]
        bullets = []
        for _, row in df.iterrows():
            ticker = str(row.get("Ticker", "") or "").strip().upper()
            detail = _short_text(row.get("Detail", ""), 130)
            why = str(row.get("Why", "") or "").strip()
            bullets.append(f"- {ticker}: {detail}. Why: {why}")
        return bullets

    def _overlay_focus_tickers(limit: int = 15) -> list[str]:
        if not chain_oi_overlay_csv or sc_df.empty or "chain_oi_overlay_contracts" not in sc_df.columns:
            return []
        focus_df = sc_df.copy()
        focus_df["ticker"] = focus_df["ticker"].astype(str).str.upper().str.strip()
        focus_df["_overlay_contracts"] = pd.to_numeric(
            focus_df.get("chain_oi_overlay_contracts"), errors="coerce"
        ).fillna(0.0)
        focus_df = focus_df[focus_df["_overlay_contracts"] > 0].copy()
        if focus_df.empty:
            return []
        issue = focus_df.get("issue_type", pd.Series("", index=focus_df.index)).astype(str).str.upper().str.strip()
        is_index = focus_df.get("is_index", pd.Series(False, index=focus_df.index)).map(
            lambda x: str(x).strip().lower() in {"1", "t", "true", "y", "yes"}
        )
        focus_df = focus_df[~(issue.isin({"ETF", "INDEX", "ETN"}) | is_index)].copy()
        focus_df = focus_df.sort_values("_overlay_contracts", ascending=False)
        return focus_df["ticker"].dropna().astype(str).head(limit).tolist()

    def _run_or_load_morning_watch(focus_tickers: list[str]) -> tuple[pd.DataFrame, str, str]:
        morning_date = chain_oi_overlay_date or asof_str
        morning_base = base.parent / morning_date
        morning_csv = morning_base / f"morning-watch-setups-{morning_date}.csv"
        morning_md = morning_base / f"morning-watch-setups-{morning_date}.md"
        generator = Path(__file__).resolve().with_name("generate_chain_only_watchlist.py")
        status = "not run"
        if generator.exists():
            cmd = [
                sys.executable,
                str(generator),
                "--date",
                morning_date,
                "--base-dir",
                str(base.parent),
                "--limit",
                "12",
                "--focus-tickers",
                ",".join([t for t in focus_tickers if t]) or "NFLX,ONDS,ASTS",
            ]
            if args.historical_replay:
                cmd.append("--historical-replay")
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=240)
                if cp.returncode == 0:
                    status = "generated"
                else:
                    status = _short_text((cp.stderr or cp.stdout or f"exit {cp.returncode}").strip(), 240)
            except Exception as exc:
                status = _short_text(f"generator failed: {exc}", 240)
        if morning_csv.exists():
            try:
                return pd.read_csv(morning_csv, low_memory=False), status, str(morning_md if morning_md.exists() else morning_csv)
            except Exception as exc:
                return pd.DataFrame(), _short_text(f"read failed: {exc}", 240), str(morning_csv)
        return pd.DataFrame(), status, str(morning_csv)

    skip_streak_dates = _trailing_skip_streak_dates()
    skip_escalation = []
    if approved_count == 0 and len(skip_streak_dates) >= 3:
        daily_nm = _daily_near_miss_frame(limit=5)
        focus = daily_nm["Ticker"].dropna().astype(str).str.upper().tolist() if not daily_nm.empty else []
        focus = list(dict.fromkeys([*focus, *_overlay_focus_tickers(limit=15)]))
        morning_df, morning_status, morning_artifact = _run_or_load_morning_watch(focus)
        skip_escalation.extend(
            [
                "## Skip-Streak Escalation",
                "",
                f"Triggered because the daily pipeline has `0` approved trades and the trailing skip streak is `{len(skip_streak_dates)}` market days: "
                + ", ".join(skip_streak_dates)
                + ".",
                "",
                "This section is not a gate-loosening override. It forces a trader-review packet: daily near-misses, deterministic chain-only morning-watch names, and the overlap/disagreement between them.",
                "",
                f"Morning-watch generator status: `{morning_status}`",
                f"Morning-watch artifact: {morning_artifact}",
                "",
                "### Daily Pipeline Top Rejected Setups",
                markdown_table(daily_nm, ["#", "St", "Ticker", "Strategy", "Legs", "Exp", "Conv", "Edge", "Why"]),
                "",
                "**Rejected setup details**",
                *_daily_near_miss_detail_bullets(daily_nm),
                "",
                "### Deterministic Morning-Watch Setups",
            ]
        )
        if morning_df.empty:
            skip_escalation.extend(["_No morning-watch rows available._", ""])
        else:
            mw = morning_df.copy().head(12)
            mw["Ticker"] = mw.get("ticker", "").astype(str).str.upper()
            mw["Morning Setup"] = mw.apply(
                lambda row: _display_strategy_legs(
                    row.get("strategy", ""),
                    row.get("expiry", ""),
                    row.get("lead_symbol", ""),
                    row.get("pair_symbol", ""),
                ),
                axis=1,
            )
            mw["Target"] = mw.get("target_value", "").astype(str)
            mw["Stretch/Floor"] = mw.get("stretch_value", "").astype(str)
            mw["Flow Conviction"] = mw.get("flow_conviction_label", "").astype(str) + " " + mw.get("flow_conviction", "").astype(str)
            mw["Breakeven Difficulty"] = mw.get("geometry_label", "").astype(str)
            mw["Morning Reason"] = mw.get("include_reason", "").astype(str)
            skip_escalation.extend(
                [
                    markdown_table(
                        mw,
                        [
                            "Ticker",
                            "Morning Setup",
                            "Target",
                            "Stretch/Floor",
                            "Flow Conviction",
                            "Breakeven Difficulty",
                            "Morning Reason",
                        ],
                    ),
                    "",
                    "### Daily vs Morning-Watch Comparison",
                ]
            )
            daily_by_ticker = {str(r.get("Ticker", "")).upper(): r for _, r in daily_nm.iterrows()} if not daily_nm.empty else {}
            morning_by_ticker = {str(r.get("Ticker", "")).upper(): r for _, r in mw.iterrows()}
            tickers = sorted(set(daily_by_ticker.keys()) | set(morning_by_ticker.keys()))
            comp_rows = []
            for ticker in tickers:
                drow = daily_by_ticker.get(ticker)
                mrow = morning_by_ticker.get(ticker)
                if drow is not None and mrow is not None:
                    decision = "Trader-review only: morning-watch confirms interest, but daily gates still veto."
                elif drow is not None:
                    decision = "Daily near-miss only: chain-only morning-watch did not independently confirm in top set."
                else:
                    decision = "Morning-watch only: not approved by full daily pipeline; needs full gate review before any trade."
                comp_rows.append(
                    {
                        "Ticker": ticker,
                        "Daily Pipeline": _short_text(drow.get("Daily Setup", "") if drow is not None else "", 130),
                        "Daily Blockers": _short_text(drow.get("Daily Blockers", "") if drow is not None else "", 150),
                        "Morning Watch": _short_text(mrow.get("Morning Setup", "") if mrow is not None else "", 130),
                        "Escalation Decision": decision,
                    }
                )
            skip_escalation.extend(
                [
                    markdown_table(
                        pd.DataFrame(comp_rows),
                        ["Ticker", "Daily Pipeline", "Daily Blockers", "Morning Watch", "Escalation Decision"],
                    ),
                    "",
                    "**Operator rule:** if this section shows a compelling overlap, do not auto-enter; run a focused near-miss audit and live entry check on that exact ticker/structure.",
                    "",
                ]
            )
    elif approved_count == 0:
        skip_escalation = [
            "## Skip-Streak Escalation",
            "",
            f"Not triggered yet. Current trailing skip streak is `{len(skip_streak_dates)}` market day(s): "
            + (", ".join(skip_streak_dates) if skip_streak_dates else "none")
            + ".",
            "",
        ]

    data_source_provenance = [
        "## Data Source Provenance",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "Field Family": "Live option bid/ask, deltas, live spot",
                        "Source": "dated local UW option-chain exports" if args.historical_replay else "Schwab API token_file",
                        "Freshness": asof_str if args.historical_replay else "run-time Stage-2",
                    },
                    {
                        "Field Family": "Stage-1 flow, OI changes, screener fields",
                        "Source": (
                            "dated local UW exports; chain OI overlay"
                            if chain_oi_overlay_csv
                            else "dated local UW exports"
                        ),
                        "Freshness": (
                            f"EOD {asof_str}; OI {chain_oi_overlay_date or 'unknown'}"
                            if chain_oi_overlay_csv
                            else asof_str
                        ),
                    },
                    {
                        "Field Family": "GEX regime, net GEX, GEX walls",
                        "Source": (
                            "Schwab live option-chain calculation"
                            if (not args.historical_replay) and gex_source_counts.get(SCHWAB_LIVE_GEX_SOURCE)
                            else "stale Schwab option-chain snapshot"
                            if (not args.historical_replay) and gex_source_counts.get(SCHWAB_STALE_GEX_SOURCE)
                            else "date-matched UW dashboard capture"
                            if args.historical_replay and gex_source_counts.get(UW_GEX_SOURCE)
                            else "date-matched UW capture required; no current fallback"
                            if args.historical_replay
                            else "not available"
                        ),
                        "Freshness": (
                            "run-time Schwab snapshot"
                            if (not args.historical_replay) and gex_source_counts.get(SCHWAB_LIVE_GEX_SOURCE)
                            else "stale same-date snapshot; live refresh failed"
                            if (not args.historical_replay) and gex_source_counts.get(SCHWAB_STALE_GEX_SOURCE)
                            else asof_str
                            if args.historical_replay and gex_source_counts.get(UW_GEX_SOURCE)
                            else "n/a"
                            if args.historical_replay
                            else "n/a"
                        ),
                    },
                    {
                        "Field Family": "Setup likelihood",
                        "Source": "conditioned local/yfinance OHLC analog model",
                        "Freshness": "historical/cache",
                    },
                ]
            ),
            ["Field Family", "Source", "Freshness"],
        ),
        "",
        f"GEX source counts: {gex_source_counts if gex_source_counts else 'none'}",
        f"UW GEX summary file: {uw_gex_summary_csv if uw_gex_summary_csv.exists() else 'not found'}",
        f"UW GEX strikes file: {uw_gex_strikes_csv if uw_gex_strikes_csv.exists() else 'not found'}",
        f"UW GEX collection status file: {uw_gex_status_csv if uw_gex_status_csv.exists() else 'not found'}",
        "",
    ]

    def _external_scanner_coverage_section() -> list[str]:
        """Show old/audited scanner recommendations that daily did not cover.

        This is intentionally not an approval override. It is a coverage guard:
        if another local dated scanner produced a recommendation, the daily
        report must either cover it in the daily book or explicitly show that it
        was absent so a human is not left to discover the miss manually.
        """
        if not show_external_scanner_coverage_section:
            return []

        source_frames = []
        read_errors = []
        rec_path = base / f"options_scan_{asof_str}_audited_recommendations.csv"
        if rec_path.exists():
            try:
                rec_df = pd.read_csv(rec_path, low_memory=False)
                if not rec_df.empty and "Ticker" in rec_df.columns:
                    rec_df = rec_df.copy()
                    rec_df["_Coverage Source"] = "audited_recommendations"
                    source_frames.append(rec_df)
            except Exception as exc:
                read_errors.append(f"{rec_path.name}: {_short_text(exc, 160)}")

        built_path = base / f"options_scan_{asof_str}_audited_built_rows.csv"
        if built_path.exists():
            try:
                built_df = pd.read_csv(built_path, low_memory=False)
                if not built_df.empty and "Ticker" in built_df.columns:
                    built_df = built_df.copy()
                    built_df["_ev_num"] = pd.to_numeric(built_df.get("EV/ML"), errors="coerce")
                    built_df["_pop_num"] = pd.to_numeric(built_df.get("POP"), errors="coerce")
                    action_s = built_df.get("Action", pd.Series("", index=built_df.index)).astype(str)
                    # Keep the external scanner's highest-EV positive debit ideas
                    # even when they did not make its final recommendations file.
                    built_df = built_df[
                        action_s.str.contains("BUY", case=False, na=False)
                        & built_df["_ev_num"].notna()
                        & (built_df["_ev_num"] >= 0.50)
                    ].copy()
                    if not built_df.empty:
                        built_df = built_df.sort_values(["_ev_num", "_pop_num"], ascending=[False, False]).head(40)
                        built_df["_Coverage Source"] = "audited_built_rows_top_ev"
                        source_frames.append(built_df)
            except Exception as exc:
                read_errors.append(f"{built_path.name}: {_short_text(exc, 160)}")

        if not source_frames:
            if read_errors:
                return [
                    "## External Scanner Coverage Reconciliation",
                    "",
                    "Could not read external scanner coverage files: " + "; ".join(read_errors),
                    "",
                ]
            return []
        rec_df = pd.concat(source_frames, ignore_index=True, sort=False)
        rec_df["_dedupe_key"] = (
            rec_df.get("Ticker", "").astype(str).str.upper().str.strip()
            + "|"
            + rec_df.get("Buy leg", "").astype(str)
            + "|"
            + rec_df.get("Sell leg", "").astype(str)
            + "|"
            + rec_df.get("Expiry", "").astype(str)
        )
        rec_df = rec_df.drop_duplicates("_dedupe_key", keep="first")

        def _parse_leg_key(text: str):
            m = re.search(
                r"\b([A-Z][A-Z0-9.\-]{0,9})\s+(\d{4}-\d{2}-\d{2})\s+([0-9]+(?:\.[0-9]+)?)([CP])\b",
                str(text or "").upper(),
            )
            if not m:
                return None
            return (m.group(1), m.group(2), round(float(m.group(3)), 4), m.group(4))

        coverage_book_df = decision_audit_all if not decision_audit_all.empty else mdf
        daily_tickers = {
            str(x).strip().upper()
            for x in coverage_book_df.get("ticker", pd.Series(dtype=str)).dropna().tolist()
            if str(x).strip()
        }
        daily_keys = set()
        for _, drow in coverage_book_df.iterrows():
            ticker = str(drow.get("ticker", "")).strip().upper()
            expiry = str(drow.get("expiry", "")).strip()
            strategy = str(drow.get("strategy", "")).strip()
            if not ticker or not expiry:
                continue
            if strategy == "Bull Call Debit":
                long_s = fnum(drow.get("long_strike"))
                short_s = fnum(drow.get("short_strike"))
                if np.isfinite(long_s) and np.isfinite(short_s):
                    daily_keys.add((ticker, expiry, round(float(long_s), 4), "C", round(float(short_s), 4), "C"))
            elif strategy == "Bear Put Debit":
                long_s = fnum(drow.get("long_strike"))
                short_s = fnum(drow.get("short_strike"))
                if np.isfinite(long_s) and np.isfinite(short_s):
                    daily_keys.add((ticker, expiry, round(float(long_s), 4), "P", round(float(short_s), 4), "P"))

        rows = []
        for _, r in rec_df.iterrows():
            ticker = str(r.get("Ticker", "")).strip().upper()
            buy_leg = str(r.get("Buy leg", "") or r.get("Buy Leg", "") or "").strip()
            sell_leg = str(r.get("Sell leg", "") or r.get("Sell Leg", "") or "").strip()
            buy_key = _parse_leg_key(buy_leg)
            sell_key = _parse_leg_key(sell_leg)
            exact_in_daily = False
            if buy_key and sell_key and buy_key[0] == sell_key[0]:
                exact_in_daily = (
                    buy_key[0],
                    buy_key[1],
                    buy_key[2],
                    buy_key[3],
                    sell_key[2],
                    sell_key[3],
                ) in daily_keys
            if exact_in_daily:
                continue
            if ticker not in daily_tickers:
                status = "Missing from daily book"
                action = "Coverage audit required"
            else:
                status = "Ticker covered; structure absent"
                action = "Compare structures"
            rows.append(
                {
                    "Status": status,
                    "Source": str(r.get("_Coverage Source", "")),
                    "Ticker": ticker,
                    "Setup": _short_text(f"{buy_leg} / {sell_leg}", 80),
                    "Exp": str(r.get("Expiry", "")),
                    "Net": str(r.get("Net", "")),
                    "EV/ML": str(r.get("EV/ML", "")),
                    "POP": str(r.get("POP", "")),
                    "Conv": str(r.get("Conviction", "")),
                    "Action": action,
                }
            )

        if not rows:
            return []
        coverage_df = pd.DataFrame(rows)
        coverage_csv = out_dir / f"external_scanner_coverage_misses_{asof_str}.csv"
        try:
            coverage_df.to_csv(coverage_csv, index=False)
        except Exception:
            pass
        return [
            "## External Scanner Coverage Reconciliation",
            "",
            "These rows came from local audited scanner recommendations/built-row files but were not exact matches in the daily-pipeline book. This section is a coverage guard, not an approval override.",
            "",
            markdown_table(
                coverage_df.head(12),
                ["Status", "Source", "Ticker", "Setup", "Exp", "Net", "EV/ML", "POP", "Conv", "Action"],
            ),
            "",
            f"Coverage CSV: {coverage_csv.name}",
            "",
        ]

    external_scanner_coverage = _external_scanner_coverage_section()
    if args.historical_replay:
        approval_summary_lines = [
            f"Historical gate-pass candidates (NOT live approvals): {approved_count} / {len(out_df)}",
            f"Historical action split: {approved_live_action_text}",
        ]
    elif args.eod_live_planning:
        approval_summary_lines = [
            f"EOD planned candidates with current Schwab refresh: {approved_count} / {len(out_df)}",
            f"Live action split: {approved_live_action_text}",
        ]
    else:
        approval_summary_lines = [
            f"Approved trades: {approved_count} / {len(out_df)}",
            f"Live action split: {approved_live_action_text}",
        ]

    def trade_setup_board_section(source_df: pd.DataFrame) -> list[str]:
        section = [
            "## Rejects",
            "",
            "Raw rejected rows from the daily approval book. Use this for audit/debug, not as the main trade board.",
            "",
        ]
        if source_df.empty:
            section.extend(["_No setup rows available._", ""])
            return section

        board = source_df.copy()
        if "_board_order" not in board.columns:
            book_rank = {
                "Core": 0,
                "Tactical": 1,
                "Medium": 2,
                "Income": 3,
                "Pilot": 4,
                "Scout": 5,
                "Watch": 6,
            }
            board["_board_order"] = board.get("Execution Book", "").map(lambda x: book_rank.get(str(x), 9))
        if "_display_rank_score" not in board.columns:
            board["_display_rank_score"] = 0

        def _status_icon(row: pd.Series) -> str:
            book = str(row.get("Execution Book", "") or "").strip()
            action = str(row.get("Live Action", "") or "").strip()
            if action == "ENTER":
                return "🟢 ENTER"
            if action == "TARGET":
                return "🟡 TARGET"
            if action == "REVIEW":
                return "🟠 REVIEW"
            if book == "Core":
                return "🟢 CORE"
            if book == "Tactical":
                return "🟦 TACT"
            if book == "Medium":
                return "🟡 MED"
            if book == "Income":
                return "🟣 INCOME"
            if book == "Pilot":
                return "🟠 PILOT"
            if book == "Scout":
                return "🟡 SCOUT"
            return "🔴 SKIP"

        def _reason(row: pd.Series) -> str:
            action = str(row.get("Live Action", "") or "").strip()
            live_reason = str(row.get("Live Check Reason", "") or "").strip()
            reject = str(row.get("Reject / Action Reason", "") or "").strip()
            watch_flags = str(row.get("Watch Reason Flags", "") or "").strip()
            notes = str(row.get("Notes", "") or "").strip()
            if action == "ENTER":
                return live_reason or "Executable now at or inside the entry gate."
            if action == "TARGET":
                return live_reason or "Use target limit only; do not chase above the gate."
            if action == "REVIEW":
                return live_reason or "Position/exposure review required before adding risk."
            return reject or watch_flags or live_reason or notes or "Rejected by daily gates."

        def _edge_sample(row: pd.Series) -> str:
            text = str(row.get("Setup Likelihood", "") or "").strip()
            text = text.replace("LOW_SAMPLE", "LOW_SAMPLE")
            edge_match = re.search(r"edge\s+([+-]?\d+(?:\.\d+)?)%", text)
            n_match = re.search(r"n=(\d+)", text)
            verdict = ""
            if "LOW_SAMPLE" in text:
                verdict = "LOW_SAMPLE"
            elif "FAIL" in text:
                verdict = "FAIL"
            elif "PASS" in text:
                verdict = "PASS"
            parts = [verdict]
            if edge_match:
                parts.append(f"edge {float(edge_match.group(1)):+.1f}%")
            if n_match:
                parts.append(f"n={n_match.group(1)}")
            return " ".join([p for p in parts if p]) or text

        board["Status"] = board.apply(_status_icon, axis=1)
        board["Reason"] = board.apply(_reason, axis=1)
        board["Edge / Sample"] = board.apply(_edge_sample, axis=1)
        board["Strategy"] = board.get("Strategy Type", board.get("Action", ""))
        board["Setup"] = board.get("Strike Setup", "")
        board["Exp"] = board.get("Expiry", "")
        board["Entry Gate"] = board.get("Net Credit/Debit", "")
        board["Conv"] = board.get("Conviction %", "")
        board["GEX"] = board.get("GEX Regime", "")
        board = board.sort_values(["_board_order", "_display_rank_score"], ascending=[True, False]).head(35)
        section.append(
            markdown_table(
                board,
                [
                    "Status",
                    "Ticker",
                    "Strategy",
                    "Setup",
                    "Exp",
                    "Live Action",
                    "Entry Gate",
                    "Conv",
                    "Edge / Sample",
                    "GEX",
                    "Reason",
                ],
            )
        )
        section.append("")
        return section

    review_candidate_pool, review_candidate_csv, review_candidate_count, review_candidate_df = review_candidate_pool_section(
        decision_audit_all,
        limit=12,
    )
    approved_display_df = out_df[out_df["Category"].astype(str).str.startswith(("Approved", "Pilot"))].copy()
    daily_trade_board = daily_trade_board_section(approved_display_df, pd.DataFrame())
    setup_board = trade_setup_board_section(out_df)

    edge_qualified_rows = int((pd.to_numeric(decision_audit_all.get("edge_score", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 1).sum()) if not decision_audit_all.empty else 0
    qualified_candidate_rows = int(decision_audit_all.get("qualified_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not decision_audit_all.empty else 0
    live_action_code_series = out_df.get("Live Action Code", out_df.get("Live Action", pd.Series(dtype=str)))
    live_enter_rows = int((live_action_code_series.fillna("").astype(str) == "ENTER").sum()) if not out_df.empty else 0
    weekly_credit_fallback, weekly_credit_meta = weekly_credit_fallback_section(live_enter_rows)

    def weekly_credit_work_limit_board(meta: dict, limit: int = 8) -> list[str]:
        """Show live-valid weekly-credit work limits without calling them approved."""
        artifacts = meta.get("artifacts", {}) if isinstance(meta, dict) else {}
        final_csv = artifacts.get("weekly_final_live_csv") or artifacts.get("weekly_live_csv") or ""
        path = Path(str(final_csv)) if final_csv else Path()
        if not path.exists():
            return []
        try:
            wdf = pd.read_csv(path)
        except Exception:
            return []
        if wdf.empty:
            return []
        valid = wdf.get("is_final_live_valid", pd.Series(False, index=wdf.index)).fillna(False).astype(bool)
        status_ok = wdf.get("live_status", pd.Series("", index=wdf.index)).fillna("").astype(str).eq("ok_live")
        board = wdf[valid & status_ok].copy()
        if board.empty:
            return []

        def _spread(row) -> str:
            ticker = str(row.get("ticker", "") or "")
            strategy = str(row.get("strategy", "") or "")
            expiry = str(row.get("expiry", "") or "")[:10]
            short_strike = fnum(row.get("short_strike"))
            long_strike = fnum(row.get("long_strike"))
            is_put = "put" in strategy.lower()
            put_call = "P" if is_put else "C"
            if np.isfinite(short_strike) and np.isfinite(long_strike):
                return f"{ticker} {strategy} Sell {short_strike:g}{put_call} / Buy {long_strike:g}{put_call} {expiry}"
            return f"{ticker} {strategy} {expiry}".strip()

        def _action(row) -> str:
            net_type = str(row.get("net_type", "") or "").strip().lower()
            live_net = fnum(row.get("live_net_bid_ask"))
            threshold = fnum(row.get("entry_gate_threshold"))
            if not np.isfinite(threshold):
                threshold = fnum(row.get("entry_net"))
            if np.isfinite(live_net) and np.isfinite(threshold):
                if net_type == "credit" and live_net >= threshold:
                    return live_action_label("ENTER")
                if net_type == "debit" and live_net <= threshold:
                    return live_action_label("ENTER")
            return live_action_label("TARGET")

        def _limit(row) -> str:
            net_type = str(row.get("net_type", "") or "").strip().lower()
            threshold = fnum(row.get("entry_gate_threshold"))
            if not np.isfinite(threshold):
                threshold = fnum(row.get("entry_net"))
            if not np.isfinite(threshold):
                return ""
            return f"{'>=' if net_type == 'credit' else '<='} {threshold:.2f} {'credit' if net_type == 'credit' else 'debit'}"

        def _live(row) -> str:
            live_net = fnum(row.get("live_net_bid_ask"))
            live_mark = fnum(row.get("live_net_mark"))
            if np.isfinite(live_net) and np.isfinite(live_mark):
                return f"bid {live_net:.2f} / mid {live_mark:.2f}"
            if np.isfinite(live_net):
                return f"live {live_net:.2f}"
            return ""

        def _pop(row) -> str:
            short_delta = abs(fnum(row.get("short_delta_live")))
            if np.isfinite(short_delta):
                return f"{max(0.0, min(1.0, 1.0 - short_delta)):.0%}"
            return ""

        board["Action"] = board.apply(_action, axis=1)
        board["Trade"] = board.apply(_spread, axis=1)
        board["Limit"] = board.apply(_limit, axis=1)
        board["Live"] = board.apply(_live, axis=1)
        board["POP"] = board.apply(_pop, axis=1)
        board["Conf"] = pd.to_numeric(board.get("confidence_score", pd.Series(np.nan, index=board.index)), errors="coerce").map(
            lambda x: f"{x:.1f}" if np.isfinite(x) else ""
        )
        board["Max Loss"] = pd.to_numeric(board.get("live_max_loss", pd.Series(np.nan, index=board.index)), errors="coerce").map(
            lambda x: money(x) if np.isfinite(x) else ""
        )
        board["_order"] = board["Action"].map({
            live_action_label("ENTER"): 0,
            live_action_label("TARGET"): 1,
            live_action_label("REVIEW"): 2,
        }).fillna(3)
        board["_score"] = pd.to_numeric(board.get("score", pd.Series(0, index=board.index)), errors="coerce").fillna(0)
        board = board.sort_values(["_order", "_score"], ascending=[True, False]).head(limit)
        actionable_count = int(meta.get("rows_actionable", 0) or 0) if isinstance(meta, dict) else 0
        status_line = (
            "- These rows passed the fallback safety gates and are repeated here as limit-order context."
            if actionable_count > 0
            else "- Review only: no weekly-credit fallback row passed all fallback safety gates. Use this as a target-price watch board, not an action queue."
        )
        return [
            "## Weekly Credit Work Limits",
            "",
            "These are live-valid rows from the weekly credit fallback book. Approved fallback entries, if any, are listed under Trades To Enter.",
            status_line,
            "`🟣 ENTER NOW` means the live bid/ask passes the target; `🟢 WORK LIMIT` means wait for the shown limit.",
            "",
            markdown_table(board, ["Action", "Trade", "Limit", "Live", "POP", "Conf", "Max Loss"]),
            "",
        ]

    actionable_limit_board = weekly_credit_work_limit_board(weekly_credit_meta)
    funnel_metrics_lines = [
        "## Daily Funnel Metrics",
        "",
        f"- Raw Stage-1 candidates: {int(len(best))}",
        f"- Shortlist rows: {int(len(shortlist))}",
        f"- Priced Stage-2 rows: {int(len(live))}",
        f"- Edge-qualified rows: {edge_qualified_rows}",
        f"- Approval rows: {int(approved_count)}",
        f"- Live ENTER rows: {live_enter_rows}",
        f"- Qualified Review rows: {qualified_candidate_rows}",
        f"- Weekly credit fallback ENTER rows: {int(weekly_credit_meta.get('rows_actionable', 0) or 0)}",
        "",
    ]

    lines = [
        *actionable_limit_board,
        *weekly_credit_fallback,
        *daily_trade_board,
        *review_candidate_pool,
        *setup_board,
        *funnel_metrics_lines,
        *live_entry_summary,
        "## Approval Summary",
        "",
        *action_mini_tables,
        "## Run Metadata",
        f"As-of date used: {asof_str}",
        "Files used: "
        + ", ".join(
            [
                csvs["chain-oi-changes-"].name,
                csvs["dp-eod-report-"].name,
                csvs["hot-chains-"].name,
                csvs["stock-screener-"].name,
                bot_eod_source.name,
                shortlist_csv.name,
                likelihood_csv.name,
                live_csv.name,
                live_final_csv.name,
                dropped_csv.name,
                manifest_path.name,
            ]
        ),
        (
            "Stage-2 note: reused existing same-date live outputs because live pricing refresh failed."
            if stage2_reused_existing
            else ""
        ),
        (
            "Stage-2 note: HISTORICAL REPLAY mode used dated UW chain quotes and dated stock close; current Schwab live pricing was not used."
            if args.historical_replay
            else ""
        ),
        (
            "Stage-2 note: EOD LIVE PLANNING mode used dated EOD discovery files, then refreshed Schwab quotes/current chain GEX at run time."
            if args.eod_live_planning
            else ""
        ),
        *approval_summary_lines,
        f"Execution book split: Core={core_count}, Tactical={tactical_count}, Medium={medium_count}, Income={income_count}, Pilot={pilot_count}, Scout={scout_count}, Watch={watch_book_count}",
        "Category split: "
        + ", ".join(
            [
                f"Approved-FIRE={int((out_df['Category'] == 'Approved - FIRE').sum())}",
                f"Approved-SHIELD={int((out_df['Category'] == 'Approved - SHIELD').sum())}",
                f"Pilot-FIRE={int((out_df['Category'] == 'Pilot - FIRE').sum())}",
                f"Pilot-SHIELD={int((out_df['Category'] == 'Pilot - SHIELD').sum())}",
                f"Watch-FIRE={int((out_df['Category'] == 'Watch Only - FIRE').sum())}",
                f"Watch-SHIELD={int((out_df['Category'] == 'Watch Only - SHIELD').sum())}",
            ]
        ),
        ("Important: NO HIGH-confidence trades passed live + likelihood gates. Medium/Income rows, if shown above, are the smaller-size action queues."
         if (core_count + tactical_count) == 0
         else ""),
        "",
        *data_source_provenance,
        *external_scanner_coverage,
        *skip_escalation,
        *event_momentum_section,
        "## Diagnostics Artifacts",
        "",
        f"- Full decision CSV: {decision_audit_csv}",
        f"- Display decision CSV: {decision_book_csv}",
        f"- Live trade CSV: {live_csv}",
        f"- Review candidate CSV: {review_candidate_csv}",
        f"- Dropped trades CSV: {dropped_csv}",
        "- Detailed blocker tables are kept in the CSV artifacts instead of flooding this Markdown report.",
        "",
        "Ticker thesis + invalidation (Yes-Prime / Yes-Good):",
    ]

    seen = set()
    for _, r in mdf.iterrows():
        if not bool(r["approved"]):
            continue
        t = str(r["ticker"])
        if t in seen:
            continue
        seen.add(t)
        lines.append(f"- {t}: {str(r.get('thesis', '')).strip()} Invalidation: {str(r.get('invalidation', '')).strip()}")
    if not seen:
        lines.append("- none")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    run_completed_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "asof_date": asof_str,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "git_commit": safe_git_commit(),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path) if cfg_path.exists() else "",
        "base_dir": str(base),
        "out_dir": str(out_dir),
        "output_md": str(output_path),
        "gex_source_counts": gex_source_counts,
        "weekly_credit_fallback": weekly_credit_meta,
        "files_used": [
            csvs["chain-oi-changes-"].name,
            csvs["dp-eod-report-"].name,
            csvs["hot-chains-"].name,
            csvs["stock-screener-"].name,
            bot_eod_source.name,
            shortlist_csv.name,
            likelihood_csv.name,
            live_csv.name,
            live_final_csv.name,
            Path(review_candidate_csv).name,
            dropped_csv.name,
        ],
        "candidate_counts": {
            "stage1_candidates_raw": int(len(best)),
            "stage1_shortlist_rows": int(len(shortlist)),
            "stage1_dropped": int(len(dropped_stage1)),
            "stage2_live_rows": int(len(live)),
            "decision_book_all_rows": int(len(decision_audit_all)),
            "display_rows": int(len(mdf)),
            "review_candidate_rows": int(review_candidate_count),
            "approved_rows": int(approved_count),
            "weekly_credit_fallback_rows": int(weekly_credit_meta.get("rows_actionable", 0) or 0),
        },
        "input_files": {
            "chain_oi_changes_csv": str(csvs["chain-oi-changes-"]),
            "chain_oi_overlay_csv": chain_oi_overlay_csv,
            "chain_oi_overlay_date": chain_oi_overlay_date,
            "dp_eod_report_csv": str(csvs["dp-eod-report-"]),
            "hot_chains_csv": str(csvs["hot-chains-"]),
            "stock_screener_csv": str(csvs["stock-screener-"]),
            "bot_eod_report": str(bot_eod_source),
            "whale_source_label": whale_flow.source_label,
            "whale_markdown_ignored": whale_flow.source_label != "legacy_whale_markdown",
        },
        "artifacts": {
            "shortlist_csv": str(shortlist_csv),
            "whale_symbol_summary_csv": str(whale_symbol_summary_csv),
            "whale_top_trades_csv": str(whale_top_trades_csv),
            "likelihood_csv": str(likelihood_csv),
            "likelihood_yf_cache_dir": str(likelihood_yf_cache_dir),
            "live_csv": str(live_csv),
            "live_final_csv": str(live_final_csv),
            "review_candidate_csv": str(review_candidate_csv),
            "dropped_csv": str(dropped_csv),
            "decision_book_csv": str(decision_book_csv),
            "planned_trade_journal_csv": str(planned_journal_csv),
            "weekly_credit_fallback": weekly_credit_meta.get("artifacts", {}),
            "manifest_json": str(manifest_path),
            "snapshot_json": str((out_dir / f"schwab_snapshot_{asof_str}.json").resolve()),
            "snapshot_chain_dir": str((out_dir / f"schwab_snapshot_{asof_str}" / "chains").resolve()),
            "uw_gex_summary_csv": str(uw_gex_summary_csv) if uw_gex_summary_csv.exists() else "",
            "uw_gex_strikes_csv": str(uw_gex_strikes_csv) if uw_gex_strikes_csv.exists() else "",
            "uw_gex_collection_status_csv": str(uw_gex_status_csv) if uw_gex_status_csv.exists() else "",
        },
        "settings": {
            "top_trades_requested": int(args.top_trades),
            "discovery_multiplier": float(discovery_multiplier),
            "discovery_top": int(discovery_top),
            "final_max_per_ticker": int(final_max_per_ticker),
            "stage2_mode": stage2_mode,
            "historical_replay": bool(args.historical_replay),
            "eod_live_planning": bool(args.eod_live_planning),
            "allow_current_live_on_historical_date": bool(
                args.allow_current_live_on_historical_date or args.eod_live_planning
            ),
            "strict_stage2": not bool(args.allow_stale_stage2),
            "allow_stale_stage2": bool(args.allow_stale_stage2),
            "stage2_reused_existing_live": bool(stage2_reused_existing),
            "stage2_error": stage2_error,
            "enforce_pretrade_portfolio_caps": bool(enforce_pretrade_caps),
            "pretrade_caps_require_data": bool(pretrade_caps_require_data),
            "block_same_underlying_option_overlap": bool(block_same_underlying_option_overlap),
            "pretrade_caps_status": portfolio_guard_status,
            "pretrade_caps_error": portfolio_guard_error,
            "pretrade_caps_snapshot_csv": portfolio_guard_snapshot_csv,
            "enable_dual_books": bool(enable_dual_books),
            "core_size_mult": float(core_size_mult),
            "tactical_size_mult": float(tactical_size_mult),
            "enable_medium_book": bool(enable_medium_book),
            "medium_size_mult": float(medium_size_mult) if np.isfinite(medium_size_mult) else None,
            "medium_allowed_strategies": sorted(medium_allowed_strategies),
            "medium_min_conviction": float(medium_min_conviction) if np.isfinite(medium_min_conviction) else None,
            "medium_min_edge_pct": float(medium_min_edge_pct) if np.isfinite(medium_min_edge_pct) else None,
            "medium_bear_min_edge_pct": (
                float(medium_bear_min_edge_pct) if np.isfinite(medium_bear_min_edge_pct) else None
            ),
            "medium_min_signals": float(medium_min_signals) if np.isfinite(medium_min_signals) else None,
            "medium_min_reward_risk": float(medium_min_reward_risk) if np.isfinite(medium_min_reward_risk) else None,
            "medium_max_debit_frac": float(medium_max_debit_frac) if np.isfinite(medium_max_debit_frac) else None,
            "enable_quant_edge_book": bool(enable_quant_edge_book),
            "quant_edge_size_mult": float(quant_edge_size_mult) if np.isfinite(quant_edge_size_mult) else None,
            "quant_edge_allowed_strategies": sorted(quant_edge_allowed_strategies),
            "quant_edge_allowed_verdicts": sorted(quant_edge_allowed_verdicts),
            "quant_edge_min_edge_pct": float(quant_edge_min_edge_pct) if np.isfinite(quant_edge_min_edge_pct) else None,
            "quant_edge_min_signals": float(quant_edge_min_signals) if np.isfinite(quant_edge_min_signals) else None,
            "quant_edge_dte_range": [
                float(quant_edge_min_dte) if np.isfinite(quant_edge_min_dte) else None,
                float(quant_edge_max_dte) if np.isfinite(quant_edge_max_dte) else None,
            ],
            "quant_edge_iv_rank_range": [
                float(quant_edge_min_iv_rank) if np.isfinite(quant_edge_min_iv_rank) else None,
                float(quant_edge_max_iv_rank) if np.isfinite(quant_edge_max_iv_rank) else None,
            ],
            "quant_edge_max_per_day": int(quant_edge_max_per_day),
            "quant_edge_max_per_ticker_day": int(quant_edge_max_per_ticker_day),
            "quant_edge_disallowed_blocker_substrings": list(quant_edge_blocker_substrings),
            "enable_short_dte_edge_book": bool(enable_short_dte_edge_book),
            "short_dte_edge_size_mult": float(short_dte_edge_size_mult) if np.isfinite(short_dte_edge_size_mult) else None,
            "short_dte_edge_allowed_strategies": sorted(short_dte_edge_allowed_strategies),
            "short_dte_edge_allowed_verdicts": sorted(short_dte_edge_allowed_verdicts),
            "short_dte_edge_min_edge_pct": float(short_dte_edge_min_edge_pct) if np.isfinite(short_dte_edge_min_edge_pct) else None,
            "short_dte_edge_min_signals": float(short_dte_edge_min_signals) if np.isfinite(short_dte_edge_min_signals) else None,
            "short_dte_edge_min_confidence_score": (
                float(short_dte_edge_min_confidence_score) if np.isfinite(short_dte_edge_min_confidence_score) else None
            ),
            "short_dte_edge_dte_range": [
                float(short_dte_edge_min_dte) if np.isfinite(short_dte_edge_min_dte) else None,
                float(short_dte_edge_max_dte) if np.isfinite(short_dte_edge_max_dte) else None,
            ],
            "short_dte_edge_max_iv_rank": float(short_dte_edge_max_iv_rank) if np.isfinite(short_dte_edge_max_iv_rank) else None,
            "short_dte_edge_max_per_day": int(short_dte_edge_max_per_day),
            "short_dte_edge_max_per_ticker_day": int(short_dte_edge_max_per_ticker_day),
            "short_dte_edge_require_qualified_candidate": bool(short_dte_edge_require_qualified_candidate),
            "short_dte_edge_disallowed_blocker_substrings": list(short_dte_edge_blocker_substrings),
            "enable_high_signal_edge_book": bool(enable_high_signal_edge_book),
            "high_signal_edge_size_mult": (
                float(high_signal_edge_size_mult) if np.isfinite(high_signal_edge_size_mult) else None
            ),
            "high_signal_edge_allowed_strategies": sorted(high_signal_edge_allowed_strategies),
            "high_signal_edge_allowed_verdicts": sorted(high_signal_edge_allowed_verdicts),
            "high_signal_edge_min_edge_pct": (
                float(high_signal_edge_min_edge_pct) if np.isfinite(high_signal_edge_min_edge_pct) else None
            ),
            "high_signal_edge_min_signals": (
                float(high_signal_edge_min_signals) if np.isfinite(high_signal_edge_min_signals) else None
            ),
            "high_signal_edge_min_confidence_score": (
                float(high_signal_edge_min_confidence_score)
                if np.isfinite(high_signal_edge_min_confidence_score)
                else None
            ),
            "high_signal_edge_dte_range": [
                float(high_signal_edge_min_dte) if np.isfinite(high_signal_edge_min_dte) else None,
                float(high_signal_edge_max_dte) if np.isfinite(high_signal_edge_max_dte) else None,
            ],
            "high_signal_edge_max_iv_rank": (
                float(high_signal_edge_max_iv_rank) if np.isfinite(high_signal_edge_max_iv_rank) else None
            ),
            "high_signal_edge_max_per_day": int(high_signal_edge_max_per_day),
            "high_signal_edge_max_per_ticker_day": int(high_signal_edge_max_per_ticker_day),
            "high_signal_edge_require_qualified_candidate": bool(high_signal_edge_require_qualified_candidate),
            "high_signal_edge_require_contract_flow_confirmed": bool(
                high_signal_edge_require_contract_flow_confirmed
            ),
            "high_signal_edge_require_gex_context": bool(high_signal_edge_require_gex_context),
            "high_signal_edge_excluded_approval_regimes": sorted(high_signal_edge_excluded_approval_regimes),
            "high_signal_edge_allowed_gex_wall_contexts": sorted(high_signal_edge_allowed_gex_wall_contexts),
            "high_signal_edge_disallowed_blocker_substrings": list(high_signal_edge_blocker_substrings),
            "enable_income_book": bool(enable_income_book),
            "income_size_mult": float(income_size_mult) if np.isfinite(income_size_mult) else None,
            "income_min_edge_pct": float(income_min_edge_pct) if np.isfinite(income_min_edge_pct) else None,
            "income_min_signals": float(income_min_signals) if np.isfinite(income_min_signals) else None,
            "income_min_hist_success_pct": float(income_min_hist_success_pct) if np.isfinite(income_min_hist_success_pct) else None,
            "enable_qualified_book": bool(enable_qualified_book),
            "qualified_min_confidence_score": float(qualified_min_confidence_score) if np.isfinite(qualified_min_confidence_score) else None,
            "qualified_min_edge_score": float(qualified_min_edge_score) if np.isfinite(qualified_min_edge_score) else None,
            "confidence_high_min_score": float(confidence_high_min_score) if np.isfinite(confidence_high_min_score) else None,
            "confidence_medium_min_score": float(confidence_medium_min_score) if np.isfinite(confidence_medium_min_score) else None,
            "final_validity_gate_enabled": bool(final_validity_gate_enabled),
            "valid_trade_min_pop": float(valid_trade_min_pop) if np.isfinite(valid_trade_min_pop) else None,
            "valid_trade_min_confidence_score": (
                float(valid_trade_min_confidence_score) if np.isfinite(valid_trade_min_confidence_score) else None
            ),
            "valid_trade_min_edge_score": float(valid_trade_min_edge_score) if np.isfinite(valid_trade_min_edge_score) else None,
            "valid_trade_require_positive_edge_pct": bool(valid_trade_require_positive_edge_pct),
            "valid_trade_min_credit_pct_width": (
                float(valid_trade_min_credit_pct_width) if np.isfinite(valid_trade_min_credit_pct_width) else None
            ),
            "valid_trade_max_debit_pct_width": (
                float(valid_trade_max_debit_pct_width) if np.isfinite(valid_trade_max_debit_pct_width) else None
            ),
            "medium_review_min_pop": float(medium_review_min_pop) if np.isfinite(medium_review_min_pop) else None,
            "high_enter_min_pop": float(high_enter_min_pop) if np.isfinite(high_enter_min_pop) else None,
            "high_enter_min_edge_pct": float(high_enter_min_edge_pct) if np.isfinite(high_enter_min_edge_pct) else None,
            "approval_regime_rules_enabled": bool(approval_regime_rules_enabled),
            "enable_pilot_book": bool(enable_pilot_book),
            "pilot_live_entry_enabled": bool(pilot_live_entry_enabled),
            "pilot_size_mult": float(pilot_size_mult) if np.isfinite(pilot_size_mult) else None,
            "pilot_max_loss": float(pilot_max_loss) if np.isfinite(pilot_max_loss) else None,
            "pilot_min_ev_ml": float(pilot_min_ev_ml) if np.isfinite(pilot_min_ev_ml) else None,
            "pilot_high_pop_min": float(pilot_high_pop_min) if np.isfinite(pilot_high_pop_min) else None,
            "pilot_floor_pop_min": float(pilot_floor_pop_min) if np.isfinite(pilot_floor_pop_min) else None,
            "pilot_min_conviction": float(pilot_min_conviction) if np.isfinite(pilot_min_conviction) else None,
            "pilot_pass_min_signals": float(pilot_pass_min_signals) if np.isfinite(pilot_pass_min_signals) else None,
            "pilot_pass_min_edge_pct": float(pilot_pass_min_edge) if np.isfinite(pilot_pass_min_edge) else None,
            "pilot_dte_range": [
                float(pilot_min_dte) if np.isfinite(pilot_min_dte) else None,
                float(pilot_max_dte) if np.isfinite(pilot_max_dte) else None,
            ],
            "pilot_min_reward_risk": float(pilot_min_reward_risk) if np.isfinite(pilot_min_reward_risk) else None,
            "pilot_max_debit_frac": float(pilot_max_debit_frac) if np.isfinite(pilot_max_debit_frac) else None,
            "pilot_max_long_otm_pct": float(pilot_max_long_otm_pct) if np.isfinite(pilot_max_long_otm_pct) else None,
            "enable_native_pilot_book": bool(enable_native_pilot_book),
            "native_pilot_max_loss": float(native_pilot_max_loss) if np.isfinite(native_pilot_max_loss) else None,
            "native_pilot_min_partial_ev_ml": (
                float(native_pilot_min_partial_ev_ml)
                if np.isfinite(native_pilot_min_partial_ev_ml)
                else None
            ),
            "native_pilot_min_signals": (
                float(native_pilot_min_signals) if np.isfinite(native_pilot_min_signals) else None
            ),
            "native_pilot_min_edge_pct": (
                float(native_pilot_min_edge) if np.isfinite(native_pilot_min_edge) else None
            ),
            "native_pilot_require_contract_confirmed": bool(native_pilot_require_contract_confirmed),
            "native_pilot_market_confidences": sorted(native_pilot_market_confidences),
            "pilot_hard_blocker_policy": "Pilot may forgive weak evidence/GEX/sample labels only; never live-entry, invalidation, stale GEX, spot drift, beyond-cap moneyness, contra contract flow, or confirmed opposite ticker-flow blockers",
            "enable_scout_book": bool(enable_scout_book),
            "scout_size_mult": float(scout_size_mult) if np.isfinite(scout_size_mult) else None,
            "scout_min_edge_pct": float(scout_min_edge_pct) if np.isfinite(scout_min_edge_pct) else None,
            "scout_max_edge_pct": float(scout_max_edge_pct) if np.isfinite(scout_max_edge_pct) else None,
            "bull_call_short_dte_high_edge_block": bool(bull_call_short_dte_high_edge_block),
            "bull_call_short_dte_high_edge_max_dte": (
                float(bull_call_short_dte_high_edge_max_dte)
                if np.isfinite(bull_call_short_dte_high_edge_max_dte)
                else None
            ),
            "bull_call_short_dte_high_edge_min_edge_pct": (
                float(bull_call_short_dte_high_edge_min_edge)
                if np.isfinite(bull_call_short_dte_high_edge_min_edge)
                else None
            ),
            "allow_bear_put_evidence_lane": bool(allow_bear_put_evidence_lane),
            "bear_put_evidence_min_edge_pct": (
                float(bear_put_evidence_min_edge) if np.isfinite(bear_put_evidence_min_edge) else None
            ),
            "bear_put_evidence_min_signals": (
                float(bear_put_evidence_min_signals) if np.isfinite(bear_put_evidence_min_signals) else None
            ),
            "bear_put_evidence_min_conviction": (
                float(bear_put_evidence_min_conviction) if np.isfinite(bear_put_evidence_min_conviction) else None
            ),
            "bear_put_evidence_min_long_delta": (
                float(bear_put_evidence_min_long_delta) if np.isfinite(bear_put_evidence_min_long_delta) else None
            ),
            "bear_put_evidence_dte_range": [
                float(bear_put_evidence_min_dte) if np.isfinite(bear_put_evidence_min_dte) else None,
                float(bear_put_evidence_max_dte) if np.isfinite(bear_put_evidence_max_dte) else None,
            ],
            "bear_put_evidence_min_reward_risk": (
                float(bear_put_evidence_min_reward_risk)
                if np.isfinite(bear_put_evidence_min_reward_risk)
                else None
            ),
            "bear_put_evidence_max_debit_frac": (
                float(bear_put_evidence_max_debit_frac)
                if np.isfinite(bear_put_evidence_max_debit_frac)
                else None
            ),
            "bear_put_evidence_max_iv_rank": (
                float(bear_put_evidence_max_iv_rank) if np.isfinite(bear_put_evidence_max_iv_rank) else None
            ),
            "bear_put_evidence_require_contract_confirmed": bool(
                bear_put_evidence_require_contract_confirmed
            ),
            "allow_bear_put_scout_lane": bool(allow_bear_put_scout_lane),
            "bear_put_scout_likelihood_strengths": sorted(bear_put_scout_likelihood_strengths),
            "bear_put_scout_require_negative_edge": bool(bear_put_scout_require_negative_edge),
            "bear_put_scout_min_signals": float(bear_put_scout_min_signals) if np.isfinite(bear_put_scout_min_signals) else None,
            "bear_put_scout_dte_range": [
                float(bear_put_scout_min_dte) if np.isfinite(bear_put_scout_min_dte) else None,
                float(bear_put_scout_max_dte) if np.isfinite(bear_put_scout_max_dte) else None,
            ],
            "bear_put_scout_max_iv_rank": float(bear_put_scout_max_iv_rank) if np.isfinite(bear_put_scout_max_iv_rank) else None,
            "bear_put_scout_max_vix": float(bear_put_scout_max_vix) if np.isfinite(bear_put_scout_max_vix) else None,
            "bear_put_scout_require_spy_5d_nonnegative": bool(bear_put_scout_require_spy_5d_nonnegative),
            "bear_put_scout_min_reward_risk": float(bear_put_scout_min_reward_risk) if np.isfinite(bear_put_scout_min_reward_risk) else None,
            "bear_put_scout_max_debit_frac": float(bear_put_scout_max_debit_frac) if np.isfinite(bear_put_scout_max_debit_frac) else None,
            "bear_put_scout_hard_blocker_policy": "may forgive stage1_conviction_below_yes_good only; never overrides live, invalidation, safety, or contra-flow blockers",
            "allow_debit_momentum_scout_lane": bool(allow_debit_momentum_scout_lane),
            "external_scanner_mode": external_scanner_mode,
            "external_scanner_candidates_added": int(len(external_scanner_candidates)),
            "show_external_scanner_coverage_section": bool(show_external_scanner_coverage_section),
            "debit_momentum_scout_min_conviction": float(debit_momentum_scout_min_conviction) if np.isfinite(debit_momentum_scout_min_conviction) else None,
            "debit_momentum_scout_min_edge_pct": float(debit_momentum_scout_min_edge_pct) if np.isfinite(debit_momentum_scout_min_edge_pct) else None,
            "debit_momentum_scout_bear_min_edge_pct": float(debit_momentum_scout_bear_min_edge_pct) if np.isfinite(debit_momentum_scout_bear_min_edge_pct) else None,
            "debit_momentum_scout_min_signals": float(debit_momentum_scout_min_signals) if np.isfinite(debit_momentum_scout_min_signals) else None,
            "debit_momentum_scout_dte_range": [
                float(debit_momentum_scout_min_dte) if np.isfinite(debit_momentum_scout_min_dte) else None,
                float(debit_momentum_scout_max_dte) if np.isfinite(debit_momentum_scout_max_dte) else None,
            ],
            "debit_momentum_scout_min_reward_risk": float(debit_momentum_scout_min_reward_risk) if np.isfinite(debit_momentum_scout_min_reward_risk) else None,
            "debit_momentum_scout_max_debit_frac": float(debit_momentum_scout_max_debit_frac) if np.isfinite(debit_momentum_scout_max_debit_frac) else None,
            "debit_momentum_scout_max_iv_rank": float(debit_momentum_scout_max_iv_rank) if np.isfinite(debit_momentum_scout_max_iv_rank) else None,
            "debit_momentum_scout_min_regime_score": float(debit_momentum_scout_min_regime_score) if np.isfinite(debit_momentum_scout_min_regime_score) else None,
            "debit_momentum_scout_bear_likelihood_strengths": sorted(debit_momentum_scout_bear_likelihood_strengths),
            "debit_momentum_scout_bear_require_flow_confirmed": bool(debit_momentum_scout_bear_require_flow_confirmed),
            "debit_momentum_scout_block_gex_volatile_breakout": bool(debit_momentum_scout_block_gex_volatile_breakout),
            "debit_momentum_scout_hard_blocker_policy": "may forgive Stage-1 low conviction / weak ambiguous flow / high-IV watch-only only; never overrides live entry, invalidation, delta, GEX uncertainty, or contra contract flow",
            "allow_bull_call_pinned_continuation_lane": bool(allow_bull_call_pinned_continuation_lane),
            "bull_call_pinned_continuation_min_edge_pct": float(bull_call_pinned_continuation_min_edge) if np.isfinite(bull_call_pinned_continuation_min_edge) else None,
            "bull_call_pinned_continuation_min_signals": float(bull_call_pinned_continuation_min_signals) if np.isfinite(bull_call_pinned_continuation_min_signals) else None,
            "bull_call_pinned_continuation_min_conviction": float(bull_call_pinned_continuation_min_conviction) if np.isfinite(bull_call_pinned_continuation_min_conviction) else None,
            "bull_call_pinned_continuation_core_min_conviction": float(bull_call_pinned_continuation_core_min_conviction) if np.isfinite(bull_call_pinned_continuation_core_min_conviction) else None,
            "bull_call_pinned_continuation_dte_range": [
                float(bull_call_pinned_continuation_min_dte) if np.isfinite(bull_call_pinned_continuation_min_dte) else None,
                float(bull_call_pinned_continuation_max_dte) if np.isfinite(bull_call_pinned_continuation_max_dte) else None,
            ],
            "bull_call_pinned_continuation_min_reward_risk": float(bull_call_pinned_continuation_min_reward_risk) if np.isfinite(bull_call_pinned_continuation_min_reward_risk) else None,
            "bull_call_pinned_continuation_max_debit_frac": float(bull_call_pinned_continuation_max_debit_frac) if np.isfinite(bull_call_pinned_continuation_max_debit_frac) else None,
            "bull_call_pinned_continuation_min_likelihood_strength": bull_call_pinned_continuation_min_strength,
            "bull_call_pinned_continuation_allowed_gex_contexts": sorted(bull_call_pinned_continuation_allowed_gex_contexts),
            "bull_call_pinned_continuation_allowed_regime_confidences": sorted(bull_call_pinned_continuation_allowed_regime_confidences),
            "allow_fire_delta_moneyness_proxy": bool(allow_fire_delta_moneyness_proxy),
            "fire_delta_proxy_max_long_otm_pct": float(fire_delta_proxy_max_long_otm_pct) if np.isfinite(fire_delta_proxy_max_long_otm_pct) else None,
            "scout_live_entry_enabled": bool(scout_live_entry_enabled),
            "scout_size_mult": float(scout_size_mult),
            "scout_min_edge_pct": float(scout_min_edge_pct),
            "scout_max_edge_pct": float(scout_max_edge_pct),
            "scout_block_gex_volatile_breakout": bool(scout_block_gex_volatile_breakout),
            "tactical_min_conviction": float(tactical_min_conviction),
            "tactical_min_edge_pct": float(tactical_min_edge_pct),
            "tactical_min_signals": float(tactical_min_signals),
            "tactical_require_verdict_pass": bool(tactical_require_verdict_pass),
            "gex_source_counts": gex_source_counts,
        },
        "counts": {
            "stage1_candidates_raw": int(len(best)),
            "stage1_shortlist_rows": int(len(shortlist)),
            "stage1_dropped": int(len(dropped_stage1)),
            "stage2_live_rows": int(len(live)),
            "merged_rows": int(merged_rows_pre_filter),
            "rows_after_final_caps": int(len(mdf)),
            "final_output_rows": int(len(out_df)),
            "approved_rows": int(approved_count),
            "edge_qualified_rows": int((pd.to_numeric(decision_audit_all.get("edge_score", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 1).sum()) if not decision_audit_all.empty else 0,
            "qualified_review_rows": int(decision_audit_all.get("qualified_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not decision_audit_all.empty else 0,
            "live_enter_rows": int(
                (
                    out_df.get(
                        "Live Action Code",
                        out_df.get("Live Action", pd.Series(dtype=str)),
                    )
                    .fillna("")
                    .astype(str)
                    == "ENTER"
                ).sum()
            ) if not out_df.empty else 0,
            "approved_core_rows": int(core_count),
            "approved_tactical_rows": int(tactical_count),
            "approved_quant_edge_rows": int(
                (
                    decision_audit_all.get("quant_edge_lane", pd.Series(False, index=decision_audit_all.index)).fillna(False).astype(bool)
                    & decision_audit_all.get("approved", pd.Series(False, index=decision_audit_all.index)).fillna(False).astype(bool)
                ).sum()
            ) if not decision_audit_all.empty else 0,
            "approved_short_dte_edge_rows": int(
                (
                    decision_audit_all.get("short_dte_edge_lane", pd.Series(False, index=decision_audit_all.index)).fillna(False).astype(bool)
                    & decision_audit_all.get("approved", pd.Series(False, index=decision_audit_all.index)).fillna(False).astype(bool)
                ).sum()
            ) if not decision_audit_all.empty else 0,
            "approved_high_signal_edge_rows": int(
                (
                    decision_audit_all.get("high_signal_edge_lane", pd.Series(False, index=decision_audit_all.index)).fillna(False).astype(bool)
                    & decision_audit_all.get("approved", pd.Series(False, index=decision_audit_all.index)).fillna(False).astype(bool)
                ).sum()
            ) if not decision_audit_all.empty else 0,
            "approved_pilot_rows": int(pilot_count),
            "approved_scout_rows": int(scout_count),
            "watch_rows": int(watch_book_count),
            "final_dropped": int(len(dropped_final)),
        },
    }
    # Atomic write: write to temp file then rename to prevent corruption from parallel runs
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    print(f"Wrote: {output_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {dropped_csv}")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print("\n".join(lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    run()
