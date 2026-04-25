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
from uwos.whale_source import BOT_EOD_PREFIX, find_bot_eod_source, load_yes_prime_whale_flow


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
    if built_path.exists():
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
            flow_direction = "bullish"
            breakeven = long_strike
        elif right == "P" and short_strike < long_strike:
            strategy = "Bear Put Debit"
            flow_direction = "bearish"
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
        flow_confirmation = "confirmed" if contract_flow == "confirmed" else "weak_or_ambiguous"
        diag = _external_scanner_stage1_diag(conviction, contract_flow)
        source_name = str(r.get("_coverage_source", "external_scanner"))
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
                    f"EV/ML={r.get('EV/ML', '')}; POP={r.get('POP', '')}; "
                    f"execution={r.get('Execution', '')}"
                ),
                "source": f"external_scanner:{source_name}",
                "coverage_source": source_name,
                "thesis": "External audited scanner candidate; requires daily live/risk approval.",
                "invalidation": "Follow daily live invalidation and entry gate.",
                "flow_direction": flow_direction,
                "flow_confirmation": flow_confirmation,
                "flow_premium_bias": 0.25 if flow_direction == "bullish" else -0.25,
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
                "external_ev_ml": fnum(r.get("EV/ML")),
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
        }
    is_option = (
        pos["asset_type"].astype(str).str.upper().eq("OPTION")
        | pos["strategy"].astype(str).str.contains("Option", case=False, na=False)
        | pos["symbol"].astype(str).str.contains(r"\d{6}[CP]\d{8}", na=False)
    )
    pos = pos[is_option].copy()
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
        "--no-auto-collect-uw-gex",
        action="store_true",
        help="Skip authenticated browser collection of UW GEX before approval.",
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
    run_started_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config).resolve()

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

    if not args.output:
        output_path = base / f"anu-expert-trade-table-{asof_str}.md"
    else:
        output_path = Path(args.output).resolve()

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    bot_eod_source = find_bot_eod_source(base, asof_str)
    whale_flow = load_yes_prime_whale_flow(bot_eod_source, cfg)
    whale_source_name = bot_eod_source.name
    whale_tables = whale_flow.as_rank_tables()
    whale_symbol_summary_csv = out_dir / f"whale-symbol-summary-{asof_str}.csv"
    whale_top_trades_csv = out_dir / f"whale-top-trades-{asof_str}.csv"
    whale_flow.symbol_summary.to_csv(whale_symbol_summary_csv, index=False)
    whale_flow.top_trades.to_csv(whale_top_trades_csv, index=False)
    print(
        "Loaded bot EOD whale source: "
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
    discovery_top = max(int(args.top_trades), int(round(int(args.top_trades) * float(discovery_multiplier))))
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
    best = build_best_candidates(asof, discovery_cfg, sc_df, quotes, whale_tables, top_trades=discovery_top)

    external_scanner_candidates = load_external_scanner_candidates(base, asof_str, asof, quotes, sc_df)
    if external_scanner_candidates:
        print(
            f"  [coverage] Added {len(external_scanner_candidates)} external scanner candidates to daily universe",
            file=sys.stderr,
        )
        discovery_top = int(discovery_top) + len(external_scanner_candidates)
        if isinstance(best, pd.DataFrame):
            best = pd.concat([best, pd.DataFrame(external_scanner_candidates)], ignore_index=True, sort=False)
        else:
            best = list(best) + external_scanner_candidates

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
    shortlist["_external_rank"] = np.where(
        shortlist.get("source", pd.Series("", index=shortlist.index)).astype(str).str.startswith("external_scanner:"),
        0,
        1,
    )
    shortlist["_external_ev_sort"] = pd.to_numeric(
        shortlist.get("external_ev_ml", pd.Series(np.nan, index=shortlist.index)),
        errors="coerce",
    ).fillna(-999.0)
    shortlist = (
        shortlist.sort_values(
            ["_stage1_rank", "_external_rank", "conviction", "_external_ev_sort"],
            ascending=[True, True, False, False],
        )
        .head(max(1, int(discovery_top)))
        .drop(columns=["_stage1_rank", "_external_rank", "_external_ev_sort"])
        .reset_index(drop=True)
    )
    shortlist_csv = out_dir / f"shortlist_trades_{asof_str}_mode_a.csv"
    shortlist.to_csv(shortlist_csv, index=False)

    # Keep GEX enrichment inside the daily pipeline. Historical replay still uses
    # the dated folder as the source of truth, but if the authenticated UW browser
    # is open we opportunistically fill any missing shortlist tickers before
    # approval. If UW is unavailable, approval still blocks on gex_missing when
    # the rulebook requires GEX rather than silently accepting an under-enriched
    # trade.
    auto_gex_required = bool(approval_cfg.get("require_gex_regime", False))
    if auto_gex_required and not args.no_auto_collect_uw_gex:
        uw_gex_dir = base / "enrichments" / "uw"
        uw_gex_summary_csv = uw_gex_dir / f"uw_gex_summary_{asof_str}.csv"
        shortlist_tickers = sorted(
            {
                str(t).strip().upper()
                for t in shortlist.get("ticker", pd.Series(dtype=str)).tolist()
                if str(t).strip()
            }
        )
        existing_gex_tickers = set()
        if uw_gex_summary_csv.exists():
            try:
                existing_gex_df = pd.read_csv(uw_gex_summary_csv)
                if "date" in existing_gex_df.columns:
                    existing_gex_df = existing_gex_df[
                        existing_gex_df["date"].astype(str).str[:10].eq(asof_str)
                    ]
                elif "uw_time" in existing_gex_df.columns:
                    existing_gex_df = existing_gex_df[
                        existing_gex_df["uw_time"].astype(str).str.startswith(asof_str)
                    ]
                if "ticker" in existing_gex_df.columns:
                    existing_gex_tickers = {
                        str(t).strip().upper()
                        for t in existing_gex_df["ticker"].tolist()
                        if str(t).strip()
                    }
            except Exception as exc:
                print(f"  [gex] WARN: could not inspect existing UW GEX summary: {exc}", file=sys.stderr)

        missing_gex_tickers = [t for t in shortlist_tickers if t not in existing_gex_tickers]
        if args.uw_gex_max_tickers and args.uw_gex_max_tickers > 0:
            missing_gex_tickers = missing_gex_tickers[: args.uw_gex_max_tickers]

        if missing_gex_tickers:
            preview = ", ".join(missing_gex_tickers[:12])
            suffix = "..." if len(missing_gex_tickers) > 12 else ""
            print(f"  [gex] collecting UW GEX for {len(missing_gex_tickers)} missing shortlist tickers ({preview}{suffix})")
            try:
                from uwos.collect_uw_enrichments_mac import (
                    collect_gex_with_cdp,
                    ensure_uw_cdp_browser,
                    normalize_gex_files,
                )

                browser_state = ensure_uw_cdp_browser(
                    remote_debugging_url=args.uw_remote_debugging_url,
                    browser_app="ChatGPT Atlas",
                    wait_sec=8.0,
                )
                if not browser_state.get("ok"):
                    raise RuntimeError(f"Could not start UW CDP browser: {browser_state}")
                if browser_state.get("browser_app") not in {"existing", ""}:
                    print(
                        f"  [gex] CDP browser ready: {browser_state.get('browser_app')} "
                        f"profile={browser_state.get('profile_dir')}"
                    )

                raw_files = collect_gex_with_cdp(
                    out_dir=uw_gex_dir,
                    date_str=asof_str,
                    tickers=missing_gex_tickers,
                    remote_debugging_url=args.uw_remote_debugging_url,
                    wait_sec=args.uw_gex_wait_sec,
                )
                normalized_gex = normalize_gex_files(uw_gex_dir, asof_str)
                summary_path_text = str(normalized_gex.get("summary_path", ""))
                strikes_path_text = str(normalized_gex.get("strikes_path", ""))
                status_path = uw_gex_dir / f"gex_collection_status_{asof_str}.csv"
                print(
                    f"  [gex] UW GEX collected raw={len(raw_files)} "
                    f"summary={Path(summary_path_text).name if summary_path_text else 'missing'} "
                    f"strikes={Path(strikes_path_text).name if strikes_path_text else 'missing'} "
                    f"status={status_path.name if status_path.exists() else 'missing'}"
                )
            except Exception as exc:
                print(f"  [gex] WARN: auto UW GEX collection failed: {exc}", file=sys.stderr)
        else:
            print(f"  [gex] UW GEX already present for all {len(shortlist_tickers)} shortlist tickers")

    likelihood_csv = out_dir / f"setup_likelihood_{asof_str}.csv"
    likelihood_cmd = [
        sys.executable,
        "-m",
        "uwos.setup_likelihood_backtest",
        "--setups-csv",
        str(shortlist_csv),
        "--asof-date",
        asof_str,
        "--root-dir",
        str(Path.cwd().resolve()),
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str((out_dir / "cache" / "yf").resolve()),
        "--lookback-years",
        "2",
        "--min-signals",
        str(int(backtest_min_signals)),
    ]
    subprocess.run(likelihood_cmd, check=True)

    live_csv = out_dir / f"live_trade_table_{asof_str}.csv"
    live_final_csv = out_dir / f"live_trade_table_{asof_str}_final.csv"
    stage2_mode = "historical_replay" if args.historical_replay else "schwab_live"
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
    # Prefer explicit Unusual Whales dashboard captures from the dated input folder.
    # Schwab-derived chain-snapshot GEX remains a fallback for tickers without UW data.
    gex_by_ticker = {}
    gex_source_counts = {}
    uw_gex_summary_csv = base / "enrichments" / "uw" / f"uw_gex_summary_{asof_str}.csv"
    uw_gex_strikes_csv = base / "enrichments" / "uw" / f"uw_gex_strikes_{asof_str}.csv"
    uw_gex_status_csv = base / "enrichments" / "uw" / f"gex_collection_status_{asof_str}.csv"

    def _record_gex_source(source: str) -> None:
        gex_source_counts[source] = int(gex_source_counts.get(source, 0)) + 1

    if uw_gex_summary_csv.exists():
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
                        "gex_source": "unusual_whales_dashboard_cdp",
                        "gex_time": str(row.get("uw_time", "") or ""),
                    }
                    _record_gex_source("unusual_whales_dashboard_cdp")
            if gex_by_ticker:
                print(
                    f"  [gex] Loaded {len(gex_by_ticker)} tickers from UW dashboard capture: {uw_gex_summary_csv.name}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"  [gex] WARN: failed reading UW GEX capture {uw_gex_summary_csv}: {exc}", file=sys.stderr)

    snapshot_chain_dir = out_dir / f"schwab_snapshot_{asof_str}" / "chains"
    if (not args.historical_replay) and snapshot_chain_dir.is_dir():
        for chain_file in [p.name for p in snapshot_chain_dir.iterdir()]:
            if not chain_file.startswith("chain_") or not chain_file.endswith(".json"):
                continue
            ticker = chain_file[len("chain_"):-len(".json")]
            if ticker.upper() in gex_by_ticker:
                continue
            try:
                with open(snapshot_chain_dir / chain_file) as _f:
                    chain_data = json.load(_f)
            except Exception:
                continue
            # Get spot price
            ul = chain_data.get("underlying", {})
            spot = None
            for fld in ("mark", "last", "close"):
                v = ul.get(fld)
                if v is not None:
                    try:
                        fv = float(v)
                        if math.isfinite(fv) and fv > 0:
                            spot = fv
                            break
                    except (TypeError, ValueError):
                        pass
            if not spot:
                continue
            total_call_gex = 0.0
            total_put_gex = 0.0
            best_put_wall = (0.0, float("nan"))
            best_call_wall = (0.0, float("nan"))
            for map_name, side in [("callExpDateMap", "call"), ("putExpDateMap", "put")]:
                exp_map = chain_data.get(map_name, {}) or {}
                for exp_key, strike_map in exp_map.items():
                    for strike_key, contracts in strike_map.items():
                        if not contracts:
                            continue
                        c = contracts[0]
                        gamma = c.get("gamma")
                        oi = c.get("openInterest")
                        if gamma is None or oi is None:
                            continue
                        try:
                            g, o = float(gamma), float(oi)
                            strike_f = float(strike_key)
                        except (TypeError, ValueError):
                            continue
                        if not (math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0):
                            continue
                        gex = g * o * 100.0 * spot
                        if side == "call":
                            total_call_gex += gex
                            if strike_f > spot and gex > best_call_wall[0]:
                                best_call_wall = (gex, strike_f)
                        else:
                            total_put_gex += gex
                            if strike_f < spot and gex > best_put_wall[0]:
                                best_put_wall = (gex, strike_f)
            net = total_call_gex - total_put_gex
            gex_by_ticker[ticker.upper()] = {
                "net_gex": round(net, 2),
                "gex_regime": "pinned" if net >= 0 else "volatile",
                "gex_support": best_put_wall[1] if math.isfinite(best_put_wall[1]) else float("nan"),
                "gex_resistance": best_call_wall[1] if math.isfinite(best_call_wall[1]) else float("nan"),
                "gex_source": "schwab_snapshot_fallback",
                "gex_time": "",
            }
            _record_gex_source("schwab_snapshot_fallback")
    elif args.historical_replay:
        print(
            "  [gex] Historical replay: current Schwab snapshot fallback disabled; "
            "only date-matched UW GEX captures are used",
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
    bull_call_approval_max_dte = fnum(
        approval_cfg.get("bull_call_approval_max_dte", bull_call_evidence_max_dte)
    )
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
    enforce_pretrade_caps = bool(approval_cfg.get("enforce_pretrade_portfolio_caps", False))
    pretrade_caps_require_data = bool(approval_cfg.get("pretrade_caps_require_data", False))
    pretrade_open_positions_csv = str(approval_cfg.get("pretrade_open_positions_csv", "")).strip()
    if args.historical_replay and enforce_pretrade_caps and not pretrade_open_positions_csv:
        enforce_pretrade_caps = False
        pretrade_caps_require_data = False
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
        if np.isfinite(gate_target) and np.isfinite(live_net):
            if net_type == "debit":
                miss_abs = max(0.0, live_net - gate_target)
            else:
                miss_abs = max(0.0, gate_target - live_net)
            if (not gate_pass_raw) and miss_abs <= tol_total and live_status == "fails_live_entry_gate":
                near_miss = True
                pass_effective = True

        return {
            "gate_target": gate_target,
            "gate_live_net": live_net,
            "gate_tol_total": tol_total,
            "gate_miss_abs": miss_abs,
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
    mdf = pd.concat([mdf.reset_index(drop=True), market_ctx_df], axis=1)

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
        else:
            reason = "stage1_watch_blocked"
        return {
            "stage1_is_yes": bool(is_yes),
            "stage1_promoted": bool(promoted),
            "fire_breakout_exception": bool(fire_breakout_exception(row)),
            "bull_call_evidence_lane": bool(bull_call_evidence_lane(row)),
            "bear_put_evidence_lane": bool(bear_put_evidence_lane(row)),
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

            if (
                (base_live_status == "fails_live_entry_gate" or (not base_gate_effective) or (not base_struct_ok))
                and len(fam_local) > 1
            ):
                exec_pool = fam_local[
                    fam_local["live_status"].astype(str).eq("ok_live")
                    & fam_local["gate_pass_effective"].fillna(False).astype(bool)
                    & fam_local["entry_structure_ok_live"].fillna(True).astype(bool)
                ].copy()
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
                    restrike_reason.loc[pick_idx] = "family_restrike_from_gate_fail"

            selected_idx.append(int(pick_idx))

        selected_unique = sorted(set(selected_idx))
        mdf = mdf.loc[selected_unique].copy()
        mdf["restrike_replaced_from"] = restrike_from.loc[selected_unique].values
        mdf["restrike_reason"] = restrike_reason.loc[selected_unique].fillna("").astype(str).values
        mdf = mdf.reset_index(drop=True)

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
                if gex_source == "schwab_snapshot_fallback":
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

    mdf["approval_blockers"] = mdf.apply(
        lambda row: ";".join(approval_blockers(row)),
        axis=1,
    )
    def split_blockers(row):
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
                or token == "gex_missing"
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

    blockers_split = mdf.apply(split_blockers, axis=1)
    mdf["hard_blockers"] = blockers_split.apply(lambda x: ";".join(x[0]))
    mdf["quality_blockers"] = blockers_split.apply(lambda x: ";".join(x[1]))

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
        if hard_tokens:
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
        if not bool(row.get("stage1_effective")) and not shield_override_live:
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        conv = fnum(row.get("conviction"))
        evidence_lane = bool(row.get("bull_call_evidence_lane")) or bool(row.get("bear_put_evidence_lane"))
        _strat = str(row.get("strategy", "")).strip()
        edge = fnum(row.get("edge_pct"))
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
            and (not np.isfinite(conv) or conv < tactical_min_conviction)
        ):
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
                if np.isfinite(tactical_max_debit_pct_width) and _tac_debit_pct > tactical_max_debit_pct_width:
                    return "Watch"
        sig = fnum(row.get("signals"))
        if (
            (not shield_override_live)
            and (not evidence_lane)
            and (not ic_income_constructive)
            and np.isfinite(tactical_min_signals)
            and (not np.isfinite(sig) or sig < tactical_min_signals)
        ):
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
            if debit_momentum_scout_candidate:
                return "Scout"
            if event_momentum_scout_candidate:
                return "Scout"
            return "Scout" if bear_put_scout_candidate else "Watch"
        if debit_momentum_scout_candidate:
            return "Scout"
        if event_momentum_scout_candidate:
            return "Scout"
        if bear_put_scout_candidate:
            return "Scout"
        if scout_candidate:
            return "Scout"
        return "Tactical"

    mdf["execution_book"] = mdf.apply(execution_book, axis=1)
    mdf["approved"] = mdf["execution_book"].isin(["Core", "Tactical", "Scout"])
    mdf["size_mult"] = mdf["execution_book"].map({
        "Core": core_size_mult,
        "Tactical": tactical_size_mult,
        "Scout": scout_size_mult,
    })
    mdf["_book_rank"] = mdf["execution_book"].map({"Core": 0, "Tactical": 1, "Scout": 2, "Watch": 3}).fillna(9).astype(int)
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
                    reasons = []
                    if projected_symbol_share > symbol_limit:
                        reasons.append(
                            f"symbol_share {projected_symbol_share:.1%} > {symbol_limit:.1%} ({ticker})"
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
                    else:
                        mdf.at[idx, "portfolio_cap_pass"] = True
                        total_risk = projected_total
                        symbol_risk[ticker] = projected_symbol
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
        score += {"Core": 3000.0, "Tactical": 2000.0, "Watch": 0.0}.get(book, 0.0)
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
    _min_bear_in_output = max(1, int(engine_cfg.get("min_bear_in_output", approval_cfg.get("min_bear_in_output", 2))))
    if len(mdf) > _top_n:
        _fire_rows = mdf[mdf["track"] == "FIRE"]
        _shield_rows = mdf[mdf["track"] == "SHIELD"]
        _other_rows = mdf[~mdf["track"].isin(["FIRE", "SHIELD"])]

        _bear_fire = _fire_rows[_fire_rows["strategy"].isin(_bear_strategies)]
        _bull_fire = _fire_rows[~_fire_rows["strategy"].isin(_bear_strategies)]

        # Enforce sector cap, expiry concentration, and direction balance across
        # reserved rows too. Reservations express preference, not cap exemption.
        _selected_rows = []
        _selected_keys = set()
        _sector_counts = defaultdict(int)
        _expiry_counts = defaultdict(int)
        _direction_counts = {"bull": 0, "bear": 0}
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
            _sdir = "bear" if _sstrat in {"Bear Put Debit", "Bear Call Credit"} else "bull"
            if _direction_counts[_sdir] >= _dir_limit:
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
        if auto_gex_required:
            gex_source_live = str(row.get("gex_source", "") or "").strip().lower()
            if gex_source_live != "unusual_whales_dashboard_cdp":
                return "WAIT", f"UW dashboard GEX required before live entry; current GEX source={gex_source_live or 'missing'}."
        live_status = str(row.get("live_status", "")).strip()
        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        if ok_live:
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
            return "SKIP", "Live entry gate failed."
        if live_status == "invalid_entry_structure":
            reason = str(row.get("entry_structure_reason_live", "") or "").strip()
            return "SKIP", f"Invalid live structure{': ' + reason if reason else ''}."
        return "SKIP", f"Live status is not executable: {live_status or 'unknown'}."

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
                elif b.startswith("gex_source_fallback"):
                    watch_reason_flags.append("gex_fallback")
                elif b.startswith("confidence_tier_blocked"):
                    watch_reason_flags.append("confidence_tier_blocked")
                elif b.startswith("live_entry_gate_fail") or b.startswith("live_status:"):
                    watch_reason_flags.append("live_entry_gate_miss")
                elif b.startswith("spot_drift") or b.startswith("spot_live_missing"):
                    watch_reason_flags.append("spot_data_mismatch")
                elif b.startswith("bull_call_otm_too_far") or b.startswith("bear_put_otm_too_far"):
                    watch_reason_flags.append("debit_moneyness_fail")
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

        out_rows.append(
            {
                "#": i + 1,
                "Category": f"{'Approved' if approved else 'Watch Only'} - {normalize_track(r.get('track', ''), strategy)}",
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
                "Live Action": live_action,
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
        "Watch Only - FIRE": 2,
        "Watch Only - SHIELD": 3,
        "Watch Only - UNKNOWN": 4,
        "Approved - UNKNOWN": 5,
    }
    out_df["_cat_rank"] = out_df["Category"].map(cat_rank).fillna(99).astype(int)
    out_df = out_df.sort_values(["_cat_rank", "#"], ascending=[True, True]).drop(columns=["_cat_rank"]).reset_index(drop=True)
    out_df["#"] = range(1, len(out_df) + 1)
    approved_count = int(mdf["approved"].sum()) if "approved" in mdf.columns else 0
    core_count = int((out_df["Execution Book"] == "Core").sum()) if "Execution Book" in out_df.columns else 0
    tactical_count = int((out_df["Execution Book"] == "Tactical").sum()) if "Execution Book" in out_df.columns else 0
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
            return df["Execution Book"].astype(str).isin(["Core", "Tactical", "Scout"])
        if "Category" in df.columns:
            return df["Category"].astype(str).str.startswith("Approved")
        return pd.Series(False, index=df.index)

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
    planned_journal_df = out_df[output_approved_mask(out_df)].copy()
    planned_journal_df.loc[:, planned_journal_cols].to_csv(planned_journal_csv, index=False)

    manifest_path = out_dir / f"run_manifest_{asof_str}.json"
    category_order = [
        "Approved - FIRE",
        "Approved - SHIELD",
        "Watch Only - FIRE",
        "Watch Only - SHIELD",
        "Watch Only - UNKNOWN",
        "Approved - UNKNOWN",
    ]
    execution_book_order = ["Core", "Tactical", "Scout", "Watch"]
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
            "These rows come from the full decision book, not just the final display cap. They are shown so high-interest/event names cannot silently disappear. Scout is allowed only when live entry, reward/risk, contract flow, and breakeven confirmation are acceptable.",
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
        def _compact_event_text(value: object, limit: int = 150) -> str:
            text = str(value or "").strip()
            if len(text) <= limit:
                return text
            return text[: max(0, limit - 3)].rstrip() + "..."

        section.append("**Event candidate details**")
        for _, row in ev.iterrows():
            ticker = str(row.get("ticker", "") or "").strip().upper()
            strategy = str(row.get("strategy", "") or "").strip()
            setup = str(row.get("Strike Setup", "") or "").strip()
            expiry = str(row.get("expiry", "") or "").strip()[:10]
            entry = str(row.get("Entry", "") or "").strip()
            edge = str(row.get("Edge", "") or "").strip()
            gex = str(row.get("GEX", "") or "").strip()
            why = _compact_event_text(row.get("Reject / Action Reason", "") or row.get("stage1_not_actionable_reason", ""), 180)
            section.append(
                f"- {ticker}: {strategy} {setup} {expiry}. Entry {entry}; {edge}; GEX {gex}; {why}"
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

    live_entry_summary = ["## Live Entry Summary", ""]
    if out_df.empty:
        live_entry_summary.append("- No recommendation rows were produced.")
    else:
        approved_for_entry = out_df[output_approved_mask(out_df)].copy()
        if approved_for_entry.empty:
            live_entry_summary.append("- SKIP: No approved trades.")
        else:
            if "Live Action" in approved_for_entry.columns:
                action_series = approved_for_entry["Live Action"].astype(str).replace("", "UNKNOWN")
            else:
                action_series = pd.Series(["UNKNOWN"] * len(approved_for_entry), index=approved_for_entry.index)
            action_counts = action_series.value_counts().to_dict()
            live_entry_summary.append(
                "- Approved live-action split: "
                + ", ".join([f"{k}={int(v)}" for k, v in sorted(action_counts.items())])
            )
            for action in ["ENTER", "WAIT", "SKIP", "UNKNOWN"]:
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

    plan_cols = ["#", "Ticker", "Action", "Expiry", "DTE", "Net Credit/Debit", "Breakeven"]
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
            elif "dte_too_long" in token:
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
            mw["Morning Setup"] = (
                mw.get("strategy", "").astype(str)
                + " "
                + mw.get("expiry", "").astype(str)
                + " "
                + mw.get("lead_symbol", "").astype(str)
                + " / "
                + mw.get("pair_symbol", "").astype(str)
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
                            "UW dashboard browser/CDP capture"
                            if gex_source_counts.get("unusual_whales_dashboard_cdp")
                            else "Schwab snapshot fallback"
                            if (not args.historical_replay) and gex_source_counts.get("schwab_snapshot_fallback")
                            else "date-matched UW capture required; no current fallback"
                            if args.historical_replay
                            else "not available"
                        ),
                        "Freshness": (
                            asof_str
                            if gex_source_counts.get("unusual_whales_dashboard_cdp")
                            else "run-time fallback"
                            if (not args.historical_replay) and gex_source_counts.get("schwab_snapshot_fallback")
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

        daily_tickers = {
            str(x).strip().upper()
            for x in mdf.get("ticker", pd.Series(dtype=str)).dropna().tolist()
            if str(x).strip()
        }
        daily_keys = set()
        for _, drow in mdf.iterrows():
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
            f"Coverage CSV: {coverage_csv}",
            "",
        ]

    external_scanner_coverage = _external_scanner_coverage_section()

    lines = [
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
        f"Approved trades: {approved_count} / {len(out_df)}",
        f"Execution book split: Core={core_count}, Tactical={tactical_count}, Scout={scout_count}, Watch={watch_book_count}",
        "Category split: "
        + ", ".join(
            [
                f"Approved-FIRE={int((out_df['Category'] == 'Approved - FIRE').sum())}",
                f"Approved-SHIELD={int((out_df['Category'] == 'Approved - SHIELD').sum())}",
                f"Watch-FIRE={int((out_df['Category'] == 'Watch Only - FIRE').sum())}",
                f"Watch-SHIELD={int((out_df['Category'] == 'Watch Only - SHIELD').sum())}",
            ]
        ),
        ("Important: NO ACTIONABLE TRADES passed live + likelihood gates."
         if approved_count == 0
         else ""),
        "",
        *live_entry_summary,
        *data_source_provenance,
        *external_scanner_coverage,
        *skip_escalation,
        *event_momentum_section,
        "## Anu Expert Trade Table",
        "Mini tables by execution book (Core/Tactical/Scout/Watch), then strategy family:",
        "",
        *mini_tables,
        *near_miss_rejected,
        *rejected_trade_reasons,
        "## Watch Only Reason Tables",
        "",
        *watch_reason_tables,
        *gate_diagnostics,
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
    output_path.write_text("\n".join(lines), encoding="utf-8-sig")
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
        "input_files": {
            "chain_oi_changes_csv": str(csvs["chain-oi-changes-"]),
            "chain_oi_overlay_csv": chain_oi_overlay_csv,
            "chain_oi_overlay_date": chain_oi_overlay_date,
            "dp_eod_report_csv": str(csvs["dp-eod-report-"]),
            "hot_chains_csv": str(csvs["hot-chains-"]),
            "stock_screener_csv": str(csvs["stock-screener-"]),
            "bot_eod_report": str(bot_eod_source),
            "whale_markdown_ignored": True,
        },
        "artifacts": {
            "shortlist_csv": str(shortlist_csv),
            "whale_symbol_summary_csv": str(whale_symbol_summary_csv),
            "whale_top_trades_csv": str(whale_top_trades_csv),
            "likelihood_csv": str(likelihood_csv),
            "live_csv": str(live_csv),
            "live_final_csv": str(live_final_csv),
            "dropped_csv": str(dropped_csv),
            "decision_book_csv": str(decision_book_csv),
            "planned_trade_journal_csv": str(planned_journal_csv),
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
            "strict_stage2": not bool(args.allow_stale_stage2),
            "allow_stale_stage2": bool(args.allow_stale_stage2),
            "stage2_reused_existing_live": bool(stage2_reused_existing),
            "stage2_error": stage2_error,
            "enforce_pretrade_portfolio_caps": bool(enforce_pretrade_caps),
            "pretrade_caps_require_data": bool(pretrade_caps_require_data),
            "pretrade_caps_status": portfolio_guard_status,
            "pretrade_caps_error": portfolio_guard_error,
            "pretrade_caps_snapshot_csv": portfolio_guard_snapshot_csv,
            "enable_dual_books": bool(enable_dual_books),
            "core_size_mult": float(core_size_mult),
            "tactical_size_mult": float(tactical_size_mult),
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
            "approved_core_rows": int(core_count),
            "approved_tactical_rows": int(tactical_count),
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
