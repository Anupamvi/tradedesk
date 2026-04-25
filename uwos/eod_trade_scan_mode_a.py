import argparse
import datetime as dt
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uwos.pricer import compute_live_net
from uwos.whale_source import BOT_EOD_PREFIX, find_bot_eod_source, load_yes_prime_whale_flow


OCC_RE = re.compile(r"^([A-Z\.]{1,10})(\d{6})([CP])(\d{8})$")


def parse_occ(sym):
    m = OCC_RE.match(str(sym))
    if not m:
        return None
    root, yymmdd, right, strike8 = m.groups()
    yy = int("20" + yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    return root, dt.date(yy, mm, dd), right, int(strike8) / 1000.0


def parse_date(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return dt.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def fnum(x):
    try:
        if pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def ensure_cols(df, name, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def md_tables(md_text):
    tables = {}
    lines = md_text.splitlines()
    heading = "Untitled"
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            if len(block) >= 2:
                hdr = [x.strip() for x in block[0].strip("|").split("|")]
                rows = []
                for r in block[1:]:
                    core = r.strip("|")
                    if re.fullmatch(r"[\s:\-\|]+", core):
                        continue
                    vals = [x.strip() for x in core.split("|")]
                    vals = vals + [""] * max(0, len(hdr) - len(vals))
                    rows.append(vals[: len(hdr)])
                key = heading
                n = 2
                while key in tables:
                    key = f"{heading} ({n})"
                    n += 1
                tables[key] = pd.DataFrame(rows, columns=hdr)
            continue
        i += 1
    return tables


def width_tier(spot, cfg):
    for t in cfg["gates"]["width_tiers"]:
        lo = float(t["min_price"])
        hi = float(t["max_price"])
        if lo <= spot < hi:
            return float(t["min_width"]), float(t["max_width"]), float(t["default_width"])
    t = cfg["gates"]["width_tiers"][-1]
    return float(t["min_width"]), float(t["max_width"]), float(t["default_width"])


def money(x):
    return "N/A" if (x is None or not np.isfinite(x)) else f"${x:,.2f}"


def px(x):
    return "N/A" if (x is None or not np.isfinite(x)) else f"{x:.2f}"


def yn_bool(x):
    return str(x).strip().lower() in {"1", "t", "true", "y", "yes"}


def find_asof(files):
    pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    vals = []
    for p in files:
        m = pat.search(p.name)
        if m:
            vals.append(m.group(1))
    if not vals:
        raise ValueError("No date found in filenames.")
    return sorted(vals)[-1]


def non_bot_input_zips(base_dir: Path):
    return sorted(p for p in base_dir.glob("*.zip") if not p.name.startswith(BOT_EOD_PREFIX))


def expected_input_date(base_dir: Path) -> str:
    name = Path(base_dir).name
    return name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name) else ""


def pick_csv_map_entry(csv_map, prefix: str, asof_str: str):
    matches = [
        (zname, cpath)
        for zname, cpath in csv_map.items()
        if zname.startswith(prefix) and asof_str in zname and asof_str in cpath.name
    ]
    if not matches:
        raise FileNotFoundError(f"Missing required zip prefix/date: {prefix}{asof_str}")
    return sorted(matches, key=lambda item: item[0])[-1]


def unzip_zips(zip_paths, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for zp in zip_paths:
        if zp.name.startswith(BOT_EOD_PREFIX):
            continue
        with zipfile.ZipFile(zp, "r") as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise FileNotFoundError(f"No CSV in {zp.name}")
            name = sorted(csvs)[0]
            dst = out_dir / Path(name).name
            with zf.open(name) as src, open(dst, "wb") as d:
                shutil.copyfileobj(src, d)
            out[zp.name] = dst
    return out


def build_quotes(hot_df, oi_df, asof, hot_csv, oi_csv, hot_asof=None, oi_asof=None):
    hot_asof = hot_asof or asof
    oi_asof = oi_asof or asof
    h = hot_df.copy()
    h["bid"] = pd.to_numeric(h.get("bid"), errors="coerce")
    h["ask"] = pd.to_numeric(h.get("ask"), errors="coerce")
    h["volume"] = pd.to_numeric(h.get("volume"), errors="coerce")
    h["open_interest"] = pd.to_numeric(h.get("open_interest"), errors="coerce")
    h["delta"] = pd.to_numeric(h["delta"], errors="coerce") if "delta" in h.columns else np.nan
    h["iv"] = pd.to_numeric(h["iv"], errors="coerce") if "iv" in h.columns else np.nan
    for _side_col in [
        "ask_side_volume",
        "bid_side_volume",
        "mid_volume",
        "sweep_volume",
        "multileg_volume",
        "stock_multi_leg_volume",
    ]:
        h[_side_col] = pd.to_numeric(h[_side_col], errors="coerce") if _side_col in h.columns else 0.0
    h["quote_date"] = pd.to_datetime(h.get("date"), errors="coerce").dt.date
    h["source_csv"] = hot_csv
    h["source_kind"] = "hot"
    h = h[
        [
            "option_symbol",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "delta",
            "iv",
            "ask_side_volume",
            "bid_side_volume",
            "mid_volume",
            "sweep_volume",
            "multileg_volume",
            "stock_multi_leg_volume",
            "quote_date",
            "source_csv",
            "source_kind",
            "next_earnings_date",
            "issue_type",
        ]
    ].copy()
    h = h[(h["quote_date"].isna()) | (h["quote_date"] == hot_asof)].copy()

    o = oi_df.copy()
    o["bid"] = pd.to_numeric(o.get("last_bid"), errors="coerce")
    o["ask"] = pd.to_numeric(o.get("last_ask"), errors="coerce")
    o["volume"] = pd.to_numeric(o.get("volume"), errors="coerce")
    o["open_interest"] = pd.to_numeric(o.get("curr_oi"), errors="coerce")
    if "delta" in o.columns:
        o["delta"] = pd.to_numeric(o["delta"], errors="coerce")
    elif "curr_delta" in o.columns:
        o["delta"] = pd.to_numeric(o["curr_delta"], errors="coerce")
    else:
        o["delta"] = np.nan
    if "iv" in o.columns:
        o["iv"] = pd.to_numeric(o["iv"], errors="coerce")
    elif "curr_iv" in o.columns:
        o["iv"] = pd.to_numeric(o["curr_iv"], errors="coerce")
    else:
        o["iv"] = np.nan
    o["quote_date"] = pd.to_datetime(o.get("curr_date"), errors="coerce").dt.date
    o["source_csv"] = oi_csv
    o["source_kind"] = "oi"
    o["issue_type"] = np.nan
    o = o[
        [
            "option_symbol",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "delta",
            "iv",
            "quote_date",
            "source_csv",
            "source_kind",
            "next_earnings_date",
            "issue_type",
        ]
    ].copy()
    o = o[(o["quote_date"].isna()) | (o["quote_date"] == oi_asof)].copy()

    q = pd.concat([h, o], ignore_index=True)
    q = q[q["option_symbol"].astype(str).str.len() > 5].copy()
    q = q[np.isfinite(q["bid"]) & np.isfinite(q["ask"]) & (q["ask"] > 0)].copy()
    parsed = q["option_symbol"].map(parse_occ)
    q = q[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    q["ticker"] = parsed.map(lambda x: x[0])
    q["expiry"] = parsed.map(lambda x: x[1])
    q["right"] = parsed.map(lambda x: x[2])
    q["strike"] = parsed.map(lambda x: x[3])
    q["prio"] = np.where(q["source_kind"] == "hot", 0, 1)
    q = q.sort_values(["option_symbol", "prio"]).drop_duplicates("option_symbol", keep="first")
    return q.drop(columns=["prio"]).reset_index(drop=True)


def apply_chain_oi_overlay_to_screener(screener, oi_df, overlay_spot_map=None):
    """Blend a next-day chain-OI overlay into screener-level flow scores.

    Stage-1 candidate discovery is ticker-driven. Without this overlay merge,
    a next-day chain OI file can supply leg quotes but still fail to introduce
    newly active tickers into the candidate universe. This keeps gates strict;
    it only makes the overlay visible to discovery and diagnostics.
    """
    if screener is None or screener.empty or oi_df is None or oi_df.empty:
        return screener
    overlay_spot_map = {
        str(k).upper().strip(): fnum(v)
        for k, v in (overlay_spot_map or {}).items()
        if str(k).strip()
    }
    oi = oi_df.copy()
    if "option_symbol" not in oi.columns:
        return screener
    parsed = oi["option_symbol"].astype(str).map(parse_occ)
    oi = oi[parsed.notna()].copy()
    if oi.empty:
        return screener
    parsed = parsed[parsed.notna()]
    oi["ticker"] = parsed.map(lambda x: x[0])
    oi["right"] = parsed.map(lambda x: x[2])
    oi["_oi_diff"] = pd.to_numeric(oi.get("oi_diff_plain"), errors="coerce").fillna(0.0)
    oi = oi[oi["_oi_diff"] > 0].copy()
    if oi.empty:
        return screener

    for col in [
        "volume",
        "prev_ask_volume",
        "prev_bid_volume",
        "prev_total_premium",
        "avg_price",
        "stock_price",
    ]:
        oi[col] = pd.to_numeric(oi.get(col), errors="coerce").fillna(0.0)
    oi["_vol_eff"] = oi["volume"].clip(lower=1.0)
    oi["_premium_eff"] = oi["prev_total_premium"]
    missing_premium = ~np.isfinite(oi["_premium_eff"]) | (oi["_premium_eff"] <= 0)
    oi.loc[missing_premium, "_premium_eff"] = (
        oi.loc[missing_premium, "avg_price"].abs() * oi.loc[missing_premium, "_vol_eff"] * 100.0
    )
    side_den = (oi["prev_ask_volume"] + oi["prev_bid_volume"]).replace(0, np.nan)
    oi["_ask_premium"] = (oi["_premium_eff"] * oi["prev_ask_volume"] / side_den).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    oi["_bid_premium"] = (oi["_premium_eff"] * oi["prev_bid_volume"] / side_den).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    is_call = oi["right"].eq("C")
    is_put = oi["right"].eq("P")
    oi["_bullish_premium_overlay"] = 0.0
    oi["_bearish_premium_overlay"] = 0.0
    oi.loc[is_call, "_bullish_premium_overlay"] = oi.loc[is_call, "_ask_premium"]
    oi.loc[is_put, "_bullish_premium_overlay"] = oi.loc[is_put, "_bid_premium"]
    oi.loc[is_put, "_bearish_premium_overlay"] = oi.loc[is_put, "_ask_premium"]
    oi.loc[is_call, "_bearish_premium_overlay"] = oi.loc[is_call, "_bid_premium"]
    oi["_call_premium_overlay"] = np.where(is_call, oi["_premium_eff"], 0.0)
    oi["_put_premium_overlay"] = np.where(is_put, oi["_premium_eff"], 0.0)
    oi["_call_ask_overlay"] = np.where(is_call, oi["prev_ask_volume"], 0.0)
    oi["_call_bid_overlay"] = np.where(is_call, oi["prev_bid_volume"], 0.0)
    oi["_put_ask_overlay"] = np.where(is_put, oi["prev_ask_volume"], 0.0)
    oi["_put_bid_overlay"] = np.where(is_put, oi["prev_bid_volume"], 0.0)

    agg = oi.groupby("ticker", as_index=False).agg(
        bullish_premium_overlay=("_bullish_premium_overlay", "sum"),
        bearish_premium_overlay=("_bearish_premium_overlay", "sum"),
        call_premium_overlay=("_call_premium_overlay", "sum"),
        put_premium_overlay=("_put_premium_overlay", "sum"),
        call_volume_ask_side_overlay=("_call_ask_overlay", "sum"),
        call_volume_bid_side_overlay=("_call_bid_overlay", "sum"),
        put_volume_ask_side_overlay=("_put_ask_overlay", "sum"),
        put_volume_bid_side_overlay=("_put_bid_overlay", "sum"),
        overlay_oi_contracts=("_oi_diff", "sum"),
        overlay_rows=("option_symbol", "size"),
        overlay_stock_price=("stock_price", "max"),
    )

    out = screener.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    missing = sorted(set(agg["ticker"]) - set(out["ticker"]))
    if missing:
        templates = []
        for ticker in missing:
            arow = agg[agg["ticker"] == ticker].iloc[0]
            spot = fnum(arow.get("overlay_stock_price"))
            if (not np.isfinite(spot) or spot <= 0) and ticker in overlay_spot_map:
                spot = fnum(overlay_spot_map.get(ticker))
            if not np.isfinite(spot) or spot <= 0:
                continue
            rec = {c: np.nan for c in out.columns}
            rec["ticker"] = ticker
            rec["close"] = spot
            if "issue_type" in rec:
                rec["issue_type"] = ""
            if "is_index" in rec:
                rec["is_index"] = False
            templates.append(rec)
        if templates:
            out = pd.concat([out, pd.DataFrame(templates)], ignore_index=True)

    out = out.merge(agg, on="ticker", how="left")
    for col in [
        "bullish_premium",
        "bearish_premium",
        "call_premium",
        "put_premium",
        "call_volume_ask_side",
        "call_volume_bid_side",
        "put_volume_ask_side",
        "put_volume_bid_side",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        overlay_col = f"{col}_overlay"
        if overlay_col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) + pd.to_numeric(
                out[overlay_col], errors="coerce"
            ).fillna(0.0)
    if overlay_spot_map and "close" in out.columns:
        live_spot = out["ticker"].astype(str).str.upper().str.strip().map(overlay_spot_map).map(fnum)
        close_val = pd.to_numeric(out["close"], errors="coerce")
        fill_mask = ((~np.isfinite(close_val)) | (close_val <= 0)) & np.isfinite(live_spot) & (live_spot > 0)
        out.loc[fill_mask, "close"] = live_spot[fill_mask]
    out["chain_oi_overlay_contracts"] = pd.to_numeric(
        out.get("overlay_oi_contracts"), errors="coerce"
    ).fillna(0.0)
    out["chain_oi_overlay_rows"] = pd.to_numeric(out.get("overlay_rows"), errors="coerce").fillna(0.0)
    drop_cols = [c for c in out.columns if c.endswith("_overlay") or c in {"overlay_oi_contracts", "overlay_rows", "overlay_stock_price"}]
    return out.drop(columns=drop_cols, errors="ignore")


def score_norm(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


FLOW_CONTEXT_BY_TICKER = {}



STAGE1_YES_GOOD_MIN = 65
STAGE1_YES_PRIME_MIN = 78
STAGE1_LONG_IC_YES_GOOD_MIN = 60


def stage1_optimal_from_conv(conv, yes_good_min=STAGE1_YES_GOOD_MIN, yes_prime_min=STAGE1_YES_PRIME_MIN):
    c = fnum(conv)
    if not np.isfinite(c):
        return "Watch Only"
    if c >= yes_prime_min:
        return "Yes-Prime"
    if c >= yes_good_min:
        return "Yes-Good"
    return "Watch Only"


def _stage1_fmt_num(value):
    v = fnum(value)
    if not np.isfinite(v):
        return "nan"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def add_stage1_diagnostics(out):
    strategy = str(out.get("strategy", "")).strip()
    optimal = str(out.get("optimal", "")).strip()
    conv = fnum(out.get("conviction"))
    yes_good_min = STAGE1_LONG_IC_YES_GOOD_MIN if strategy == "Long Iron Condor" else STAGE1_YES_GOOD_MIN
    out["stage1_yes_good_threshold"] = yes_good_min
    out["stage1_yes_prime_threshold"] = STAGE1_YES_PRIME_MIN

    tokens = []
    extra_diag = str(out.get("stage1_extra_diagnostics", "") or "").strip()
    if extra_diag:
        tokens.extend([x.strip() for x in extra_diag.split(";") if x.strip()])
    if optimal == "Watch Only" and (not np.isfinite(conv) or conv < yes_good_min):
        tokens.append(f"stage1_conviction_below_yes_good:{_stage1_fmt_num(conv)}<{_stage1_fmt_num(yes_good_min)}")

    flow_dir = str(out.get("flow_direction", "")).strip().lower()
    flow_conf = str(out.get("flow_confirmation", "")).strip().lower()
    expected = {
        "Bull Call Debit": "bullish",
        "Bull Put Credit": "bullish",
        "Bear Put Debit": "bearish",
        "Bear Call Credit": "bearish",
    }.get(strategy, "")
    if expected:
        if flow_conf in {"", "unknown", "weak_or_ambiguous"}:
            tokens.append("stage1_flow_weak_or_ambiguous")
        elif flow_conf == "conflicted":
            tokens.append("stage1_flow_conflicted")
        elif flow_conf == "confirmed" and flow_dir and flow_dir != expected:
            tokens.append(f"stage1_flow_direction_mismatch:{flow_dir}!={expected}")
    elif strategy in {"Iron Condor", "Iron Butterfly"}:
        if flow_conf == "confirmed" and flow_dir in {"bullish", "bearish"}:
            tokens.append(f"stage1_flow_too_directional_for_income:{flow_dir}")

    contract_flow = str(out.get("contract_flow_confirmation", "")).strip().lower()
    if contract_flow == "contra":
        tokens.append("stage1_contract_flow_contra")
    elif contract_flow in {"weak_or_ambiguous", "unknown"}:
        tokens.append(f"stage1_contract_flow_{contract_flow}")
    elif strategy in {"Iron Condor", "Iron Butterfly"} and contract_flow == "directional":
        tokens.append("stage1_contract_flow_directional")

    if optimal == "Watch Only" and not tokens:
        tokens.append("stage1_watch_unclassified")

    seen = set()
    deduped = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    out["stage1_diagnostics"] = ";".join(deduped)
    out["stage1_not_actionable_reason"] = out["stage1_diagnostics"]
    out["stage1_flow_diagnostic"] = ";".join([t for t in deduped if t.startswith("stage1_flow_")])
    out["stage1_contract_flow_diagnostic"] = ";".join([t for t in deduped if t.startswith("stage1_contract_flow_")])
    return out

def candidate_dict(
    ticker,
    action,
    strategy,
    track,
    expiry,
    dte,
    long_strike,
    short_strike,
    width,
    net,
    net_type,
    max_profit,
    max_loss,
    breakeven,
    conviction,
    tier,
    optimal,
    notes,
    source,
    thesis,
    invalidation,
    **extras,
):
    out = {
        "ticker": ticker,
        "action": action,
        "strategy": strategy,
        "track": track,
        "expiry": expiry,
        "dte": dte,
        "long_strike": long_strike,
        "short_strike": short_strike,
        "width": width,
        "net": net,
        "net_type": net_type,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "conviction": conviction,
        "tier": tier,
        "optimal": optimal,
        "notes": notes,
        "source": source,
        "thesis": thesis,
        "invalidation": invalidation,
    }
    flow_context = FLOW_CONTEXT_BY_TICKER.get(str(ticker).upper().strip(), {})
    if flow_context:
        out.update(flow_context)
    out.update(extras)
    add_stage1_diagnostics(out)
    return out


def _leg_flow_metrics(row):
    ask_v = fnum(row.get("ask_side_volume"))
    bid_v = fnum(row.get("bid_side_volume"))
    mid_v = fnum(row.get("mid_volume"))
    multi_v = fnum(row.get("multileg_volume")) + fnum(row.get("stock_multi_leg_volume"))
    vol_v = fnum(row.get("volume"))
    den = ask_v + bid_v
    ask_bid_bias = (ask_v - bid_v) / den if den > 0 else float("nan")
    multi_share = multi_v / vol_v if vol_v > 0 else float("nan")
    return {
        "ask_side_volume": ask_v,
        "bid_side_volume": bid_v,
        "mid_volume": mid_v,
        "multileg_volume": multi_v,
        "volume": vol_v,
        "ask_bid_bias": ask_bid_bias,
        "multileg_share": multi_share,
    }


def contract_flow_for_spread(strategy, long_row=None, short_row=None, short_put_row=None, short_call_row=None):
    strategy = str(strategy).strip()

    def _confirm_debit(row, right):
        m = _leg_flow_metrics(row)
        bias = m["ask_bid_bias"]
        if not np.isfinite(bias):
            return "unknown", "no_contract_side_data", m
        if bias >= 0.10:
            return "confirmed", f"ask_side_{'call_buying' if right == 'C' else 'put_buying'}", m
        if bias <= -0.10:
            return "contra", f"bid_side_{'call_selling' if right == 'C' else 'put_selling'}", m
        return "weak_or_ambiguous", "balanced_bid_ask", m

    def _confirm_credit_short(row, right):
        m = _leg_flow_metrics(row)
        bias = m["ask_bid_bias"]
        if not np.isfinite(bias):
            return "unknown", "no_contract_side_data", m
        if bias <= -0.10:
            return "confirmed", f"bid_side_{'call_selling' if right == 'C' else 'put_selling'}", m
        if bias >= 0.10:
            return "contra", f"ask_side_{'call_buying' if right == 'C' else 'put_buying'}", m
        return "weak_or_ambiguous", "balanced_bid_ask", m

    if strategy == "Bull Call Debit" and long_row is not None:
        status, driver, metrics = _confirm_debit(long_row, "C")
    elif strategy == "Bear Put Debit" and long_row is not None:
        status, driver, metrics = _confirm_debit(long_row, "P")
    elif strategy == "Bull Put Credit" and short_row is not None:
        status, driver, metrics = _confirm_credit_short(short_row, "P")
    elif strategy == "Bear Call Credit" and short_row is not None:
        status, driver, metrics = _confirm_credit_short(short_row, "C")
    elif strategy in {"Iron Condor", "Iron Butterfly"} and short_put_row is not None and short_call_row is not None:
        put_metrics = _leg_flow_metrics(short_put_row)
        call_metrics = _leg_flow_metrics(short_call_row)
        put_bias = put_metrics["ask_bid_bias"]
        call_bias = call_metrics["ask_bid_bias"]
        directional = (
            (np.isfinite(put_bias) and abs(put_bias) >= 0.25)
            or (np.isfinite(call_bias) and abs(call_bias) >= 0.25)
        )
        status = "directional" if directional else "confirmed_neutral"
        driver = f"short_put_bias={put_bias:+.2f};short_call_bias={call_bias:+.2f}"
        metrics = {
            "ask_bid_bias": max(
                abs(put_bias) if np.isfinite(put_bias) else 0.0,
                abs(call_bias) if np.isfinite(call_bias) else 0.0,
            ),
            "multileg_share": max(
                put_metrics["multileg_share"] if np.isfinite(put_metrics["multileg_share"]) else 0.0,
                call_metrics["multileg_share"] if np.isfinite(call_metrics["multileg_share"]) else 0.0,
            ),
        }
    else:
        status, driver, metrics = "unknown", "no_contract_side_data", {}

    return {
        "contract_flow_confirmation": status,
        "contract_flow_driver": driver,
        "contract_flow_ask_bid_bias": metrics.get("ask_bid_bias", float("nan")),
        "contract_flow_multileg_share": metrics.get("multileg_share", float("nan")),
    }


def compute_macro_regime(asof, force_historical=False):
    """Fetch SPY 5-day return and VIX level for macro regime awareness.
    Uses Schwab API (fast, reliable) with yfinance fallback.
    Returns dict with spy_5d_ret, vix_level, regime ('risk_off'|'risk_on'|'neutral').
    """
    spy_5d_ret = 0.0
    vix_level = 20.0

    # Try Schwab only for same-day non-replay runs. Historical replay must not
    # use a current SPY/$VIX quote even when replaying today's dated files.
    try:
        if (not force_historical) and asof >= dt.date.today():
            from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
            config = SchwabAuthConfig.from_env(load_dotenv_file=True)
            svc = SchwabLiveDataService(config=config, interactive_login=False)
            client = svc.connect()

            # SPY 5-day return from price history
            end_dt = dt.datetime.combine(asof + dt.timedelta(days=1), dt.time())
            start_dt = dt.datetime.combine(asof - dt.timedelta(days=10), dt.time())
            resp = client.get_price_history_every_day("SPY", start_datetime=start_dt, end_datetime=end_dt)
            resp.raise_for_status()
            candles = resp.json().get("candles", [])
            if len(candles) >= 6:
                spy_5d_ret = candles[-1]["close"] / candles[-6]["close"] - 1.0

            # VIX from quote
            quotes = svc.get_quotes(["$VIX"])
            vix_data = quotes.get("$VIX", {})
            vix_quote = vix_data.get("quote", {})
            vl = vix_quote.get("lastPrice") or vix_quote.get("closePrice")
            if vl and vl > 0:
                vix_level = float(vl)
    except Exception:
        pass

    if force_historical or asof < dt.date.today() or not np.isfinite(vix_level) or vix_level == 20.0:
        # Fallback to yfinance
        try:
            import yfinance as yf
            start = (asof - dt.timedelta(days=15)).isoformat()
            end = (asof + dt.timedelta(days=1)).isoformat()
            spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
            if spy is not None and not spy.empty:
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                close = pd.to_numeric(spy.get("Close"), errors="coerce").dropna()
                if len(close) >= 6:
                    spy_5d_ret = float(close.iloc[-1]) / float(close.iloc[-6]) - 1.0
            vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
            if vix is not None and not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_close = pd.to_numeric(vix.get("Close"), errors="coerce").dropna()
                if len(vix_close) >= 1:
                    vix_level = float(vix_close.iloc[-1])
        except Exception:
            pass

    # Use OR logic: either indicator alone can signal regime shift.
    # AND was too strict — SPY -3% + VIX 20 was classified as "neutral".
    if spy_5d_ret < -0.02 or vix_level > 25:
        regime = "risk_off"
    elif spy_5d_ret > 0.02 and vix_level < 18:
        regime = "risk_on"
    else:
        regime = "neutral"
    return {"spy_5d_ret": spy_5d_ret, "vix_level": vix_level, "regime": regime}


def build_best_candidates(asof, cfg, screener, quotes, whale_tables, top_trades=20):
    gates_cfg = cfg.get("gates", {})
    fire_cfg = cfg.get("fire", {})
    shield_cfg = cfg.get("shield", {})
    pricing_cfg = cfg.get("pricing", {})
    high_beta_cfg = cfg.get("high_beta", {})
    strategy_sel_cfg = cfg.get("strategy_selection", {})
    engine_cfg = cfg.get("engine", {}) if isinstance(cfg, dict) else {}

    min_credit = float(gates_cfg["min_credit_pct_width"])
    max_credit = fnum(gates_cfg.get("max_credit_pct_width", 1.0))
    if not np.isfinite(max_credit) or max_credit <= 0 or max_credit > 1:
        max_credit = 1.0
    max_debit = float(gates_cfg["max_debit_pct_width"])
    min_leg_open_interest = fnum(gates_cfg.get("min_leg_open_interest", 0))
    min_leg_volume = fnum(gates_cfg.get("min_leg_volume", 0))
    max_leg_spread_pct = fnum(gates_cfg.get("max_leg_spread_pct"))
    max_risk_per_trade = fnum(gates_cfg.get("max_risk_per_trade"))
    min_debit_rr = fnum(gates_cfg.get("min_debit_reward_risk"))
    min_credit_rr = fnum(gates_cfg.get("min_credit_reward_risk"))
    if not np.isfinite(min_leg_open_interest) or min_leg_open_interest < 0:
        min_leg_open_interest = 0.0
    if not np.isfinite(min_leg_volume) or min_leg_volume < 0:
        min_leg_volume = 0.0
    if not np.isfinite(max_leg_spread_pct) or max_leg_spread_pct <= 0:
        max_leg_spread_pct = math.nan
    if not np.isfinite(max_risk_per_trade) or max_risk_per_trade <= 0:
        max_risk_per_trade = math.nan
    if not np.isfinite(min_debit_rr) or min_debit_rr < 0:
        min_debit_rr = 0.0
    if not np.isfinite(min_credit_rr) or min_credit_rr < 0:
        min_credit_rr = 0.0
    exclude_etfs = bool(gates_cfg.get("exclude_etfs", True))
    executable_sources = {
        str(x).strip().lower()
        for x in pricing_cfg.get("executable_source_kinds", ["hot"])
        if str(x).strip()
    } or {"hot"}

    fire_lo, fire_hi = [int(x) for x in fire_cfg["dte_range"]]
    shield_lo, shield_hi = [int(x) for x in shield_cfg["dte_range"]]
    fire_target = int(fire_cfg["target_dte"])
    shield_target = int(shield_cfg["target_dte"])
    fire_long_itm_pct = float(fire_cfg.get("debit_long_itm_pct", 0.0))
    fire_long_otm_pct = float(fire_cfg.get("debit_long_otm_pct", 0.0))
    fire_debit_long_max_abs_moneyness_pct = fnum(
        fire_cfg.get("debit_long_max_abs_moneyness_pct", 0.12)
    )
    fire_use_dual_long_targets = bool(fire_cfg.get("debit_use_dual_long_targets", True))
    fire_call_pool_min_pct = float(fire_cfg.get("call_pool_min_pct", 0.75))
    fire_call_pool_max_pct = float(fire_cfg.get("call_pool_max_pct", 1.30))
    fire_put_pool_min_pct = float(fire_cfg.get("put_pool_min_pct", 0.75))
    fire_put_pool_max_pct = float(fire_cfg.get("put_pool_max_pct", 1.30))
    fire_max_breakeven_dist_pct = fnum(fire_cfg.get("max_breakeven_dist_pct", 0.12))
    fire_invalidation_buffer_pct = fnum(fire_cfg.get("invalidation_buffer_pct", 0.03))
    if not np.isfinite(fire_long_otm_pct) or fire_long_otm_pct < 0:
        fire_long_otm_pct = 0.0
    if (
        not np.isfinite(fire_debit_long_max_abs_moneyness_pct)
        or fire_debit_long_max_abs_moneyness_pct <= 0
    ):
        fire_debit_long_max_abs_moneyness_pct = math.nan
    if not np.isfinite(fire_call_pool_min_pct) or fire_call_pool_min_pct <= 0:
        fire_call_pool_min_pct = 0.75
    if not np.isfinite(fire_call_pool_max_pct) or fire_call_pool_max_pct <= fire_call_pool_min_pct:
        fire_call_pool_max_pct = max(1.10, fire_call_pool_min_pct + 0.10)
    if not np.isfinite(fire_put_pool_min_pct) or fire_put_pool_min_pct <= 0:
        fire_put_pool_min_pct = 0.75
    if not np.isfinite(fire_put_pool_max_pct) or fire_put_pool_max_pct <= fire_put_pool_min_pct:
        fire_put_pool_max_pct = max(1.20, fire_put_pool_min_pct + 0.10)
    if not np.isfinite(fire_max_breakeven_dist_pct) or fire_max_breakeven_dist_pct <= 0:
        fire_max_breakeven_dist_pct = math.nan
    if not np.isfinite(fire_invalidation_buffer_pct) or fire_invalidation_buffer_pct <= 0:
        fire_invalidation_buffer_pct = 0.03
    fire_allow_both_neutral = bool(fire_cfg.get("allow_both_sides_when_neutral", False))
    shield_otm = float(shield_cfg["credit_short_otm_pct"])
    shield_sigma_relax = fnum(shield_cfg.get("sigma_relaxation_factor", 1.0))
    if not np.isfinite(shield_sigma_relax) or shield_sigma_relax <= 0:
        shield_sigma_relax = 1.0
    shield_ic_sigma_relax = fnum(shield_cfg.get("iron_condor_sigma_factor", shield_sigma_relax))
    if not np.isfinite(shield_ic_sigma_relax) or shield_ic_sigma_relax <= 0:
        shield_ic_sigma_relax = shield_sigma_relax
    shield_allow_both_neutral = bool(shield_cfg.get("allow_both_sides_when_neutral", False))
    shield_min_marketcap = fnum(shield_cfg.get("min_marketcap"))
    shield_lotto_dte_max = int(shield_cfg.get("lotto_dte_max", 7))
    fire_max_iv_rank = fnum(strategy_sel_cfg.get("fire_max_iv_rank"))
    fire_enforce_max_iv_rank = bool(strategy_sel_cfg.get("fire_enforce_max_iv_rank", True))
    fire_surface_high_iv_watch = bool(strategy_sel_cfg.get("fire_surface_high_iv_watch", True))
    shield_min_iv_rank = fnum(strategy_sel_cfg.get("shield_min_iv_rank"))
    range_min_iv_rank = fnum(strategy_sel_cfg.get("range_min_iv_rank", shield_min_iv_rank))
    range_min_neutrality = fnum(strategy_sel_cfg.get("range_min_neutrality", 0.70))
    enable_iron_butterfly = bool(strategy_sel_cfg.get("enable_iron_butterfly", True))
    enable_long_iron_condor = bool(strategy_sel_cfg.get("enable_long_iron_condor", True))
    breakout_min_implied_move_pct = fnum(strategy_sel_cfg.get("breakout_min_implied_move_pct", 0.05))
    breakout_min_neutrality = fnum(strategy_sel_cfg.get("breakout_min_neutrality", 0.35))
    breakout_max_center_width_pct = fnum(strategy_sel_cfg.get("breakout_max_center_width_pct", 0.15))
    breakout_max_debit_pct_width = fnum(strategy_sel_cfg.get("breakout_max_debit_pct_width", max_debit))
    if not np.isfinite(breakout_max_debit_pct_width) or breakout_max_debit_pct_width <= 0:
        breakout_max_debit_pct_width = max_debit

    max_tickers_to_scan = int(engine_cfg.get("max_tickers_to_scan", 1000))
    max_trades_per_ticker = int(engine_cfg.get("max_trades_per_ticker", 3))
    max_trades_per_ticker_per_track = int(engine_cfg.get("max_trades_per_ticker_per_track", 1))
    max_total_trades_cfg = int(engine_cfg.get("max_total_trades", 500))
    max_total_trades = min(max_total_trades_cfg, max(1, int(top_trades)))
    max_expiries_per_ticker = int(engine_cfg.get("max_expiries_per_ticker", 4))
    strike_search_depth = max(1, int(engine_cfg.get("strike_search_depth", 6)))
    min_per_track = max(1, int(engine_cfg.get("min_per_track", 1)))
    include_watch_candidates = bool(engine_cfg.get("include_watch_candidates", True))

    high_beta_names = {
        str(x).strip().upper()
        for x in high_beta_cfg.get("tickers", [])
        if str(x).strip()
    }
    high_beta_min_dte = int(high_beta_cfg.get("min_dte", 21))
    high_beta_min_otm = float(high_beta_cfg.get("min_otm_pct", 0.08))
    high_beta_iv30d_threshold = fnum(high_beta_cfg.get("iv30d_override_threshold"))
    high_beta_require_delta = bool(high_beta_cfg.get("require_short_delta_for_core", True))
    fire_lotto_dte_max = int(fire_cfg.get("lotto_dte_max", 10))

    macro = compute_macro_regime(asof)
    macro_regime = macro["regime"]
    print(f"  [macro] SPY 5d={macro['spy_5d_ret']:+.2%}, VIX={macro['vix_level']:.1f}, regime={macro_regime}")
    # Stash macro on function for Stage-2 reuse (avoids duplicate API call)
    build_best_candidates._last_macro = macro

    sc = screener.copy()
    sc["ticker"] = sc["ticker"].astype(str).str.upper()
    sc["close"] = pd.to_numeric(sc.get("close"), errors="coerce")
    sc["bullish_premium"] = pd.to_numeric(sc.get("bullish_premium"), errors="coerce").fillna(0.0)
    sc["bearish_premium"] = pd.to_numeric(sc.get("bearish_premium"), errors="coerce").fillna(0.0)
    sc["call_premium"] = pd.to_numeric(sc.get("call_premium"), errors="coerce").fillna(0.0)
    sc["put_premium"] = pd.to_numeric(sc.get("put_premium"), errors="coerce").fillna(0.0)
    sc["put_call_ratio"] = pd.to_numeric(sc.get("put_call_ratio"), errors="coerce").fillna(1.0)
    sc["iv30d"] = pd.to_numeric(sc.get("iv30d"), errors="coerce")
    sc["implied_move"] = pd.to_numeric(sc.get("implied_move"), errors="coerce")
    sc["implied_move_perc"] = pd.to_numeric(sc.get("implied_move_perc"), errors="coerce")
    sc["iv_rank"] = pd.to_numeric(sc.get("iv_rank"), errors="coerce")

    def _num_sc_col(name, default=0.0):
        if name in sc.columns:
            return pd.to_numeric(sc.get(name), errors="coerce").fillna(default)
        return pd.Series(default, index=sc.index, dtype=float)

    for _flow_col in [
        "call_volume_ask_side",
        "call_volume_bid_side",
        "put_volume_ask_side",
        "put_volume_bid_side",
        "net_call_premium",
        "net_put_premium",
    ]:
        sc[_flow_col] = _num_sc_col(_flow_col)

    market_cap_col = ""
    for c in ["market_cap", "marketcap", "mkt_cap", "market_capitalization"]:
        if c in sc.columns:
            market_cap_col = c
            break
    if market_cap_col:
        sc["market_cap_rule"] = pd.to_numeric(sc.get(market_cap_col), errors="coerce")
    else:
        sc["market_cap_rule"] = np.nan

    # Exclude ETFs/indices before score normalization so broad index outliers do not
    # distort bull/bear directionality for single-name candidates.
    if exclude_etfs:
        issue_series = sc.get("issue_type")
        if issue_series is None:
            issue_series = pd.Series([""] * len(sc), index=sc.index)
        issue = issue_series.astype(str).str.upper().str.strip()
        is_index_series = sc.get("is_index")
        if is_index_series is None:
            is_index_mask = pd.Series([False] * len(sc), index=sc.index)
        else:
            is_index_mask = is_index_series.map(yn_bool)
        sc = sc[~(issue.isin({"ETF", "INDEX", "ETN"}) | is_index_mask)].copy()

    # Direction semantics:
    #   bullish premium = ask-side calls + bid-side puts
    #   bearish premium = ask-side puts + bid-side calls
    # Raw call-minus-put premium is only supporting color. It must not saturate a
    # marginal side-aware read into a perfect directional score.
    flow_premium_den = (sc["bullish_premium"].abs() + sc["bearish_premium"].abs()).replace(0, np.nan)
    sc["flow_premium_bias"] = (
        (sc["bullish_premium"] - sc["bearish_premium"]) / flow_premium_den
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    sc["flow_bull_volume"] = sc["call_volume_ask_side"] + sc["put_volume_bid_side"]
    sc["flow_bear_volume"] = sc["put_volume_ask_side"] + sc["call_volume_bid_side"]
    flow_volume_den = (sc["flow_bull_volume"].abs() + sc["flow_bear_volume"].abs()).replace(0, np.nan)
    sc["flow_volume_bias"] = (
        (sc["flow_bull_volume"] - sc["flow_bear_volume"]) / flow_volume_den
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    call_put_den = (sc["call_premium"].abs() + sc["put_premium"].abs()).replace(0, np.nan)
    sc["call_put_premium_bias"] = (
        (sc["call_premium"] - sc["put_premium"]) / call_put_den
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    sc["direction_bias"] = (
        0.70 * sc["flow_premium_bias"]
        + 0.20 * sc["flow_volume_bias"]
        + 0.10 * sc["call_put_premium_bias"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

    def _flow_direction_label(v):
        v = fnum(v)
        if not np.isfinite(v) or abs(v) < 0.08:
            return "neutral_or_ambiguous"
        return "bullish" if v > 0 else "bearish"

    def _flow_confidence_label(v):
        v = abs(fnum(v))
        if not np.isfinite(v):
            return "unknown"
        if v >= 0.30:
            return "strong"
        if v >= 0.15:
            return "moderate"
        if v >= 0.08:
            return "weak"
        return "ambiguous"

    def _flow_primary_driver(row):
        direction = str(row.get("flow_direction", "")).strip().lower()
        if direction == "bullish":
            call_ask = fnum(row.get("call_volume_ask_side"))
            put_bid = fnum(row.get("put_volume_bid_side"))
            return "ask_call_buying" if call_ask >= put_bid else "bid_put_selling"
        if direction == "bearish":
            put_ask = fnum(row.get("put_volume_ask_side"))
            call_bid = fnum(row.get("call_volume_bid_side"))
            return "ask_put_buying" if put_ask >= call_bid else "bid_call_selling"
        return "mixed_or_ambiguous"

    def _flow_confirmation_label(row):
        direction_bias = fnum(row.get("direction_bias"))
        premium_bias = fnum(row.get("flow_premium_bias"))
        volume_bias = fnum(row.get("flow_volume_bias"))
        if not (np.isfinite(direction_bias) and np.isfinite(premium_bias)):
            return "weak_or_ambiguous"
        if abs(direction_bias) < 0.15 or abs(premium_bias) < 0.08:
            return "weak_or_ambiguous"
        if np.sign(direction_bias) != np.sign(premium_bias):
            return "conflicted"
        if np.isfinite(volume_bias) and abs(volume_bias) >= 0.10 and np.sign(volume_bias) != np.sign(direction_bias):
            return "conflicted"
        return "confirmed"

    sc["flow_direction"] = sc["direction_bias"].apply(_flow_direction_label)
    sc["flow_confidence"] = sc["direction_bias"].apply(_flow_confidence_label)
    sc["flow_primary_driver"] = sc.apply(_flow_primary_driver, axis=1)
    sc["flow_confirmation"] = sc.apply(_flow_confirmation_label, axis=1)
    sc["bull_score"] = sc["direction_bias"].clip(lower=0.0, upper=1.0)
    sc["bear_score"] = (-sc["direction_bias"]).clip(lower=0.0, upper=1.0)
    # Directional strength (unsigned) — so bear trades score equally to bulls
    sc["dir_strength"] = sc["direction_bias"].abs().clip(upper=1.0)
    sc["interest_score"] = (
        sc["bullish_premium"].abs()
        + sc["bearish_premium"].abs()
        + sc["call_premium"].abs()
        + sc["put_premium"].abs()
    )
    global FLOW_CONTEXT_BY_TICKER
    flow_context_cols = [
        "bullish_premium",
        "bearish_premium",
        "call_premium",
        "put_premium",
        "call_volume_ask_side",
        "call_volume_bid_side",
        "put_volume_ask_side",
        "put_volume_bid_side",
        "net_call_premium",
        "net_put_premium",
        "flow_bull_volume",
        "flow_bear_volume",
        "flow_premium_bias",
        "flow_volume_bias",
        "call_put_premium_bias",
        "direction_bias",
        "flow_direction",
        "flow_confidence",
        "flow_primary_driver",
        "flow_confirmation",
    ]
    FLOW_CONTEXT_BY_TICKER = {}
    for _, _flow_row in sc.iterrows():
        _ticker = str(_flow_row.get("ticker", "")).upper().strip()
        if not _ticker:
            continue
        FLOW_CONTEXT_BY_TICKER[_ticker] = {
            col: _flow_row.get(col)
            for col in flow_context_cols
            if col in sc.columns
        }
    sc = sc.sort_values("interest_score", ascending=False).head(max(1, max_tickers_to_scan))

    whale_rank = {}
    for _, df in whale_tables.items():
        tmp = df.copy()
        tmp.columns = [c.strip().lower().replace(" ", "_") for c in tmp.columns]
        if "underlying_symbol" in tmp.columns:
            for i, sym in enumerate(tmp["underlying_symbol"].astype(str).str.upper().tolist(), 1):
                if sym and sym != "NAN":
                    whale_rank[sym] = max(whale_rank.get(sym, 0.0), max(0.0, 1.0 - (i - 1) / 60.0))

    # UW historical exports often include IV but not option delta. Stage-1
    # SHIELD generation uses delta gates; if delta is missing it can select
    # high-delta short legs that Stage-2 later rejects. Estimate missing deltas
    # from dated spot/IV so discovery and approval use aligned risk geometry.
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    sc_close_by_ticker = {}
    if "ticker" in sc.columns and "close" in sc.columns:
        sc_close_by_ticker = {
            str(t).strip().upper(): fnum(c)
            for t, c in zip(sc["ticker"], sc["close"])
            if str(t).strip()
        }

    def _estimate_missing_delta(row):
        current = fnum(row.get("delta"))
        if np.isfinite(current) and abs(current) <= 1.0:
            return current
        ticker_local = str(row.get("ticker", "")).strip().upper()
        spot_local = fnum(FLOW_CONTEXT_BY_TICKER.get(ticker_local, {}).get("close"))
        if not np.isfinite(spot_local):
            spot_local = fnum(sc_close_by_ticker.get(ticker_local))
        strike_local = fnum(row.get("strike"))
        iv_local = fnum(row.get("iv"))
        right_local = str(row.get("right", "")).strip().upper()
        expiry_local = row.get("expiry")
        if not isinstance(expiry_local, dt.date):
            expiry_local = parse_date(expiry_local)
        if not (
            np.isfinite(spot_local)
            and spot_local > 0
            and np.isfinite(strike_local)
            and strike_local > 0
            and np.isfinite(iv_local)
            and iv_local > 0
            and right_local in {"C", "P"}
            and isinstance(expiry_local, dt.date)
        ):
            return math.nan
        if iv_local > 5.0:
            iv_local = iv_local / 100.0
        if iv_local <= 0 or iv_local > 5.0:
            return math.nan
        dte_local = max(1, (expiry_local - asof).days)
        t_local = dte_local / 365.0
        r_local = 0.04
        try:
            d1 = (
                math.log(spot_local / strike_local)
                + (r_local + 0.5 * iv_local * iv_local) * t_local
            ) / (iv_local * math.sqrt(t_local))
        except (ValueError, ZeroDivisionError):
            return math.nan
        if not np.isfinite(d1):
            return math.nan
        return _norm_cdf(d1) if right_local == "C" else _norm_cdf(d1) - 1.0

    quotes = quotes.copy()
    if "delta" not in quotes.columns:
        quotes["delta"] = np.nan
    quotes["delta"] = quotes.apply(_estimate_missing_delta, axis=1)

    chains = {}
    for key, grp in quotes.groupby(["ticker", "right", "expiry"]):
        chains[key] = grp.sort_values("strike").reset_index(drop=True)

    qearn = quotes.copy()
    qearn["earn"] = qearn["next_earnings_date"].map(parse_date)
    qearn = qearn[qearn["earn"].notna()]
    qearn_map = qearn.groupby("ticker")["earn"].agg(lambda s: sorted(set(s.tolist()))[0]).to_dict() if not qearn.empty else {}

    def directional_flags(direction_bias, allow_both_neutral):
        bias = fnum(direction_bias)
        if not np.isfinite(bias):
            return (True, True) if allow_both_neutral else (False, False)
        # Keep generation aligned with the public flow label threshold.  If
        # flow_direction says neutral_or_ambiguous (<0.08), both bull and bear
        # FIRE candidates should be allowed through discovery; otherwise names
        # like NFLX can be labeled neutral while bear-put candidates are silently
        # suppressed.
        neutral_gap = 0.08
        if bias > neutral_gap:
            return True, False
        if bias < -neutral_gap:
            return False, True
        if allow_both_neutral:
            return True, True
        return False, False

    def capped_expiries(exps, lo, hi, target_dte):
        cands = [e for e in exps if lo <= (e - asof).days <= hi]
        cands = sorted(cands, key=lambda e: (abs((e - asof).days - target_dte), (e - asof).days))
        return cands[: max(1, max_expiries_per_ticker)]

    def top_by_target(df, target):
        if df is None or df.empty:
            return df.iloc[0:0].copy()
        w = df.copy()
        w["_dist"] = (pd.to_numeric(w["strike"], errors="coerce") - float(target)).abs()
        w = w.sort_values(["_dist", "strike"], ascending=[True, True]).head(strike_search_depth)
        return w.drop(columns=["_dist"])

    def stage1_shield_delta_cap(ivr, dte, strategy):
        if not np.isfinite(ivr):
            ivr = 30.0
        if ivr >= 50:
            base = 0.35
        elif ivr >= 35:
            base = 0.30
        elif ivr >= 25:
            base = 0.25
        elif ivr >= 15:
            base = 0.20
        else:
            return 0.0
        if 45 <= dte <= 60:
            dte_adj = 0.0
        elif 30 <= dte < 45:
            dte_adj = -0.03
        elif 21 <= dte < 30:
            dte_adj = -0.06
        elif dte < 21:
            dte_adj = -0.10
        else:
            dte_adj = -0.02
        vix = fnum(macro.get("vix_level"))
        if vix > 40:
            vix_adj = -0.08
        elif vix > 30:
            vix_adj = -0.03
        elif vix < 15:
            vix_adj = -0.05
        else:
            vix_adj = 0.0
        ic_adj = -0.03 if strategy == "iron_condor" else 0.0
        return max(0.10, min(0.40, base + dte_adj + vix_adj + ic_adj))

    def top_shield_shorts_by_gate(df, target, right, dte, sigma, sigma_factor, strategy):
        if df is None or df.empty:
            return df.iloc[0:0].copy()
        pool = df.copy()
        cap = stage1_shield_delta_cap(iv_rank, dte, strategy)
        if cap <= 0:
            return df.iloc[0:0].copy()
        if cap > 0:
            delta = pd.to_numeric(pool.get("delta"), errors="coerce").abs()
            pool = pool[np.isfinite(delta) & (delta <= cap)].copy()
        sigma_known = np.isfinite(sigma) and sigma > 0
        if sigma_known:
            if str(right).upper() == "P":
                pool = pool[pd.to_numeric(pool["strike"], errors="coerce") <= (spot - sigma * sigma_factor)].copy()
            else:
                pool = pool[pd.to_numeric(pool["strike"], errors="coerce") >= (spot + sigma * sigma_factor)].copy()
        return top_by_target(pool, target)

    def merge_target_candidates(df, targets):
        if df is None or df.empty:
            return df.iloc[0:0].copy()
        parts = []
        for tgt in targets:
            sub = top_by_target(df, tgt)
            if sub is not None and not sub.empty:
                parts.append(sub)
        if not parts:
            return df.iloc[0:0].copy()
        out_df = pd.concat(parts, ignore_index=True)
        if "option_symbol" in out_df.columns:
            out_df = out_df.drop_duplicates(subset=["option_symbol"]).reset_index(drop=True)
        else:
            out_df = out_df.drop_duplicates().reset_index(drop=True)
        return out_df

    def liquid_leg_ok(leg_row):
        src = str(leg_row.get("source_kind", "")).strip().lower()
        if src not in executable_sources:
            return False
        if min_leg_open_interest > 0:
            leg_oi = fnum(leg_row.get("open_interest"))
            if (not np.isfinite(leg_oi)) or leg_oi < min_leg_open_interest:
                return False
        if min_leg_volume > 0:
            leg_vol = fnum(leg_row.get("volume"))
            if (not np.isfinite(leg_vol)) or leg_vol < min_leg_volume:
                return False
        if np.isfinite(max_leg_spread_pct):
            bid = fnum(leg_row.get("bid"))
            ask = fnum(leg_row.get("ask"))
            if not (np.isfinite(bid) and np.isfinite(ask) and ask > 0):
                return False
            mid = 0.5 * (bid + ask)
            if mid <= 0:
                return False
            spread_pct = (ask - bid) / mid
            if spread_pct > max_leg_spread_pct:
                return False
        return True

    def liquid_pair_ok(short_row, long_row):
        return liquid_leg_ok(short_row) and liquid_leg_ok(long_row)

    def liq_leg_score(leg_row):
        return (
            math.log1p(max(0.0, fnum(leg_row.get("volume"))))
            + math.log1p(max(0.0, fnum(leg_row.get("open_interest"))))
        )

    def sigma_move(row, spot, dte):
        iv30 = fnum(row.get("iv30d"))
        if np.isfinite(iv30) and iv30 > 0:
            return float(spot * iv30 * math.sqrt(max(1.0, dte) / 365.0))
        implied_abs = fnum(row.get("implied_move"))
        if np.isfinite(implied_abs) and implied_abs > 0:
            return float(implied_abs)
        implied_pct = fnum(row.get("implied_move_perc"))
        if np.isfinite(implied_pct) and implied_pct > 0:
            return float(spot * implied_pct * math.sqrt(max(1.0, dte) / 30.0))
        return math.nan

    def earnings_status(earnings, expiry):
        if earnings is None:
            return {
                "verified": False,
                "crossed": False,
                "within7": False,
                "core_ok": False,
                "label": "UNKNOWN",
            }
        days_to_er = (earnings - asof).days
        crossed = asof <= earnings <= expiry
        within7 = 0 <= days_to_er <= 7
        core_ok = (not crossed) and (not within7)
        return {
            "verified": True,
            "crossed": crossed,
            "within7": within7,
            "core_ok": core_ok,
            "label": "PASS" if core_ok else ("CROSSED" if crossed else "WITHIN7"),
        }

    def is_high_beta(row):
        ticker = str(row.get("ticker", "")).upper().strip()
        if ticker in high_beta_names:
            return True
        iv30 = fnum(row.get("iv30d"))
        return bool(
            np.isfinite(high_beta_iv30d_threshold)
            and high_beta_iv30d_threshold > 0
            and np.isfinite(iv30)
            and iv30 >= high_beta_iv30d_threshold
        )

    out = []
    for _, r in sc.iterrows():
        ticker = r["ticker"]
        spot = fnum(r.get("close"))
        if not np.isfinite(spot) or spot <= 0:
            continue
        issue = str(r.get("issue_type", "")).upper().strip()
        if exclude_etfs and (issue in {"ETF", "INDEX", "ETN"} or yn_bool(r.get("is_index"))):
            continue

        earnings = parse_date(r.get("next_earnings_date"))
        if earnings is None:
            earnings = qearn_map.get(ticker)
        hb = is_high_beta(r)

        min_w, max_w, default_w = width_tier(spot, cfg)
        bull = float(r.get("bull_score", 0.0))
        bear = float(r.get("bear_score", 0.0))
        dir_str = float(r.get("dir_strength", max(bull, bear)))
        whale = whale_rank.get(ticker, 0.0)
        exps = sorted(set(quotes.loc[quotes["ticker"] == ticker, "expiry"].tolist()))
        if not exps:
            continue

        dir_bias = fnum(r.get("direction_bias"))
        allow_fire_bull, allow_fire_bear = directional_flags(
            dir_bias, allow_both_neutral=fire_allow_both_neutral
        )
        allow_shield_bull, allow_shield_bear = directional_flags(
            dir_bias, allow_both_neutral=shield_allow_both_neutral
        )
        iv_rank = fnum(r.get("iv_rank"))
        implied_move_pct_raw = fnum(r.get("implied_move_perc"))
        implied_move_pct_norm = (
            implied_move_pct_raw / 100.0
            if np.isfinite(implied_move_pct_raw) and implied_move_pct_raw > 1.0
            else implied_move_pct_raw
        )
        fire_iv_high = bool(
            np.isfinite(fire_max_iv_rank)
            and np.isfinite(iv_rank)
            and iv_rank > fire_max_iv_rank
        )
        if fire_enforce_max_iv_rank and fire_iv_high and not fire_surface_high_iv_watch:
            # Enforce debit-IV discipline for FIRE; allow SHIELD to proceed independently.
            allow_fire_bull = False
            allow_fire_bear = False
        if np.isfinite(shield_min_iv_rank) and np.isfinite(iv_rank) and iv_rank < shield_min_iv_rank:
            allow_shield_bull = False
            allow_shield_bear = False

        shield_eligible = True
        if shield_min_marketcap is not None and np.isfinite(shield_min_marketcap) and shield_min_marketcap > 0:
            mcap = fnum(r.get("market_cap_rule"))
            if not np.isfinite(mcap) or mcap < shield_min_marketcap:
                shield_eligible = False
        if not shield_eligible:
            allow_shield_bull = False
            allow_shield_bear = False

        fire_exps = capped_expiries(exps, fire_lo, fire_hi, fire_target)
        shield_exps = capped_expiries(exps, shield_lo, shield_hi, shield_target)

        # Bull Call Debit + Bear Put Debit.
        for expiry in fire_exps:
            dte = (expiry - asof).days
            dte_fit = 1.0 - min(1.0, abs(dte - fire_target) / max(1, fire_hi - fire_lo))
            er = earnings_status(earnings, expiry)

            calls = chains.get((ticker, "C", expiry))
            if allow_fire_bull and calls is not None:
                call_pool = calls[
                    (calls["strike"] >= spot * fire_call_pool_min_pct)
                    & (calls["strike"] <= spot * fire_call_pool_max_pct)
                ].copy()
                call_targets = [spot * (1.0 - fire_long_itm_pct)]
                if fire_use_dual_long_targets and fire_long_otm_pct > 0:
                    call_targets.append(spot * (1.0 + fire_long_otm_pct))
                long_candidates = merge_target_candidates(call_pool, call_targets)
                for _, lg in long_candidates.iterrows():
                    long_mny = abs(float(lg["strike"]) / spot - 1.0)
                    if np.isfinite(fire_debit_long_max_abs_moneyness_pct) and long_mny > fire_debit_long_max_abs_moneyness_pct:
                        continue
                    short_pool = call_pool[call_pool["strike"] > lg["strike"]].copy()
                    short_candidates = top_by_target(short_pool, float(lg["strike"]) + default_w)
                    for _, sh in short_candidates.iterrows():
                        if not liquid_pair_ok(sh, lg):
                            continue
                        width = float(sh["strike"] - lg["strike"])
                        if width < min_w or width > max_w:
                            continue
                        net, _ = compute_live_net(
                            net_type="debit",
                            short_bid=fnum(sh["bid"]),
                            short_ask=fnum(sh["ask"]),
                            short_mark=None,
                            long_bid=fnum(lg["bid"]),
                            long_ask=fnum(lg["ask"]),
                            long_mark=None,
                        )
                        if net is None or not np.isfinite(net) or net < 0:
                            continue
                        if net / width > max_debit:
                            continue
                        max_profit_cash = (width - net) * 100
                        max_loss_cash = net * 100
                        debit_rr = (width - net) / max(1e-9, net)
                        breakeven = float(lg["strike"] + net)
                        be_dist_pct = (breakeven - spot) / max(1e-9, spot)
                        if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                            continue
                        if debit_rr < min_debit_rr:
                            continue
                        if np.isfinite(fire_max_breakeven_dist_pct) and be_dist_pct > fire_max_breakeven_dist_pct:
                            continue
                        eff = max(0.0, min(1.0, (width - net) / width))
                        liq = min(
                            math.log1p(max(0.0, fnum(lg["volume"]))) + math.log1p(max(0.0, fnum(lg["open_interest"]))),
                            math.log1p(max(0.0, fnum(sh["volume"]))) + math.log1p(max(0.0, fnum(sh["open_interest"]))),
                        )
                        liq = max(0.0, min(1.0, liq / 20.0))
                        score = 0.34 * dir_str + 0.22 * eff + 0.18 * liq + 0.14 * dte_fit + 0.12 * whale
                        conv = int(round(100 * score))
                        # IVR bonus: low IV = cheap options = good for buying
                        if np.isfinite(iv_rank):
                            if iv_rank < 25:
                                conv = min(100, conv + 3)
                            elif iv_rank > 70:
                                conv = max(0, conv - 5)
                        # [T6] Macro regime adjustment
                        if macro_regime == "risk_off":
                            conv = max(0, conv - 10)
                        elif macro_regime == "risk_on":
                            conv = min(100, conv + 3)
                        confidence_tier = "Lotto" if dte <= fire_lotto_dte_max else "Aggressive"
                        if er["crossed"]:
                            # [T6] Penalty instead of auto-kill: reduce conviction instead of forcing LOTTO.
                            # Recompute the final Stage-1 label after the penalty so a post-ER-risk
                            # 50s conviction cannot remain Yes-Good by accident.
                            conv = max(0, conv - 15)
                        optimal = stage1_optimal_from_conv(conv)
                        stage1_extra_diag = ""
                        if fire_enforce_max_iv_rank and fire_iv_high and fire_surface_high_iv_watch:
                            optimal = "Watch Only"
                            stage1_extra_diag = (
                                "stage1_high_iv_debit_watch_only:"
                                f"{_stage1_fmt_num(iv_rank)}>{_stage1_fmt_num(fire_max_iv_rank)}"
                            )
                        gate_text = (
                            f"Gates: Liquidity PASS; WidthTier PASS; DebitWidth PASS; "
                            f"Earnings={'ER-RISK' if er['crossed'] else ('UNKNOWN' if not er['verified'] else 'PASS')}; "
                            f"IVRank={'HIGH' if fire_iv_high else 'PASS'}; "
                            f"LongMny={long_mny:.1%}."
                        )
                        # Keep invalidation tied to entry regime (spot) so it stays meaningful even
                        # when the long strike is not near ATM.
                        inv_raw = float(lg["strike"] - 0.50 * width)
                        inv_buffer = float(spot * (1.0 - fire_invalidation_buffer_pct))
                        inv_level = min(inv_raw, inv_buffer, float(spot * 0.995))
                        out.append(candidate_dict(
                            ticker, "BUY", "Bull Call Debit", "FIRE", expiry, dte, float(lg["strike"]), float(sh["strike"]),
                            width, net, "debit", max_profit_cash, max_loss_cash, breakeven, conv,
                            confidence_tier, optimal,
                            f"Debit {net:.2f} on {width:.2f} ({net/width:.2%}, R/R {debit_rr:.2f}); bull={bull:.2f}, whale={whale:.2f}. {gate_text}",
                            f"{lg['source_csv']}|{sh['source_csv']}",
                            "Bull call debit fits bullish flow with capped downside.",
                            f"Invalidate if close < {inv_level:.2f}.",
                            iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                            stage1_extra_diagnostics=stage1_extra_diag,
                            **contract_flow_for_spread("Bull Call Debit", long_row=lg, short_row=sh),
                        ))

            puts = chains.get((ticker, "P", expiry))
            if allow_fire_bear and puts is not None:
                put_pool = puts[
                    (puts["strike"] >= spot * fire_put_pool_min_pct)
                    & (puts["strike"] <= spot * fire_put_pool_max_pct)
                ].copy()
                put_targets = [spot * (1.0 + fire_long_itm_pct)]
                if fire_use_dual_long_targets and fire_long_otm_pct > 0:
                    put_targets.append(spot * (1.0 - fire_long_otm_pct))
                long_candidates = merge_target_candidates(put_pool, put_targets)
                for _, lg in long_candidates.iterrows():
                    long_mny = abs(float(lg["strike"]) / spot - 1.0)
                    if np.isfinite(fire_debit_long_max_abs_moneyness_pct) and long_mny > fire_debit_long_max_abs_moneyness_pct:
                        continue
                    short_pool = put_pool[put_pool["strike"] < lg["strike"]].copy()
                    short_candidates = top_by_target(short_pool, float(lg["strike"]) - default_w)
                    for _, sh in short_candidates.iterrows():
                        if not liquid_pair_ok(sh, lg):
                            continue
                        width = float(lg["strike"] - sh["strike"])
                        if width < min_w or width > max_w:
                            continue
                        net, _ = compute_live_net(
                            net_type="debit",
                            short_bid=fnum(sh["bid"]),
                            short_ask=fnum(sh["ask"]),
                            short_mark=None,
                            long_bid=fnum(lg["bid"]),
                            long_ask=fnum(lg["ask"]),
                            long_mark=None,
                        )
                        if net is None or not np.isfinite(net) or net < 0:
                            continue
                        if net / width > max_debit:
                            continue
                        max_profit_cash = (width - net) * 100
                        max_loss_cash = net * 100
                        debit_rr = (width - net) / max(1e-9, net)
                        breakeven = float(lg["strike"] - net)
                        be_dist_pct = (spot - breakeven) / max(1e-9, spot)
                        if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                            continue
                        if debit_rr < min_debit_rr:
                            continue
                        if np.isfinite(fire_max_breakeven_dist_pct) and be_dist_pct > fire_max_breakeven_dist_pct:
                            continue
                        eff = max(0.0, min(1.0, (width - net) / width))
                        liq = min(
                            math.log1p(max(0.0, fnum(lg["volume"]))) + math.log1p(max(0.0, fnum(lg["open_interest"]))),
                            math.log1p(max(0.0, fnum(sh["volume"]))) + math.log1p(max(0.0, fnum(sh["open_interest"]))),
                        )
                        liq = max(0.0, min(1.0, liq / 20.0))
                        score = 0.34 * dir_str + 0.22 * eff + 0.18 * liq + 0.14 * dte_fit + 0.12 * whale
                        conv = int(round(100 * score))
                        # IVR bonus: low IV = cheap options = good for buying
                        if np.isfinite(iv_rank):
                            if iv_rank < 25:
                                conv = min(100, conv + 3)
                            elif iv_rank > 70:
                                conv = max(0, conv - 5)
                        # [T6] Macro regime adjustment
                        if macro_regime == "risk_off":
                            conv = min(100, conv + 5)
                        elif macro_regime == "risk_on":
                            conv = max(0, conv - 10)
                        confidence_tier = "Lotto" if dte <= fire_lotto_dte_max else "Aggressive"
                        if er["crossed"]:
                            # [T6] Penalty instead of auto-kill: reduce conviction instead of forcing LOTTO.
                            # Recompute the final Stage-1 label after the penalty so a post-ER-risk
                            # 50s conviction cannot remain Yes-Good by accident.
                            conv = max(0, conv - 15)
                        optimal = stage1_optimal_from_conv(conv)
                        stage1_extra_diag = ""
                        if fire_enforce_max_iv_rank and fire_iv_high and fire_surface_high_iv_watch:
                            optimal = "Watch Only"
                            stage1_extra_diag = (
                                "stage1_high_iv_debit_watch_only:"
                                f"{_stage1_fmt_num(iv_rank)}>{_stage1_fmt_num(fire_max_iv_rank)}"
                            )
                        gate_text = (
                            f"Gates: Liquidity PASS; WidthTier PASS; DebitWidth PASS; "
                            f"Earnings={'ER-RISK' if er['crossed'] else ('UNKNOWN' if not er['verified'] else 'PASS')}; "
                            f"IVRank={'HIGH' if fire_iv_high else 'PASS'}; "
                            f"LongMny={long_mny:.1%}."
                        )
                        # Keep invalidation tied to entry regime (spot) so it stays meaningful even
                        # when the long strike is not near ATM.
                        inv_raw = float(lg["strike"] + 0.50 * width)
                        inv_buffer = float(spot * (1.0 + fire_invalidation_buffer_pct))
                        inv_level = max(inv_raw, inv_buffer, float(spot * 1.005))
                        out.append(candidate_dict(
                            ticker, "BUY", "Bear Put Debit", "FIRE", expiry, dte, float(lg["strike"]), float(sh["strike"]),
                            width, net, "debit", max_profit_cash, max_loss_cash, breakeven, conv,
                            confidence_tier, optimal,
                            f"Debit {net:.2f} on {width:.2f} ({net/width:.2%}, R/R {debit_rr:.2f}); bear={bear:.2f}, whale={whale:.2f}. {gate_text}",
                            f"{lg['source_csv']}|{sh['source_csv']}",
                            "Bear put debit fits bearish flow with convex downside.",
                            f"Invalidate if close > {inv_level:.2f}.",
                            iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                            stage1_extra_diagnostics=stage1_extra_diag,
                            **contract_flow_for_spread("Bear Put Debit", long_row=lg, short_row=sh),
                        ))

        # Bull Put Credit + Bear Call Credit (SHIELD with full rulebook gating).
        for expiry in shield_exps:
            dte = (expiry - asof).days
            dte_fit = 1.0 - min(1.0, abs(dte - shield_target) / max(1, shield_hi - shield_lo))
            er = earnings_status(earnings, expiry)
            sigma = sigma_move(r, spot, dte)
            sigma_known = np.isfinite(sigma) and sigma > 0
            sigma_pct = (sigma / spot) if sigma_known else math.nan

            puts = chains.get((ticker, "P", expiry))
            if allow_shield_bull and puts is not None:
                tshort = spot * (1.0 - shield_otm)
                short_pool = puts[(puts["strike"] < spot) & (puts["strike"] >= spot * 0.70)].copy()
                short_candidates = top_shield_shorts_by_gate(
                    short_pool, tshort, "P", dte, sigma, shield_sigma_relax, "credit_spread"
                )
                for _, sh in short_candidates.iterrows():
                    long_pool = puts[puts["strike"] < sh["strike"]].copy()
                    long_candidates = top_by_target(long_pool, float(sh["strike"]) - default_w)
                    for _, lg in long_candidates.iterrows():
                        if not liquid_pair_ok(sh, lg):
                            continue
                        width = float(sh["strike"] - lg["strike"])
                        if width < min_w or width > max_w:
                            continue
                        net, _ = compute_live_net(
                            net_type="credit",
                            short_bid=fnum(sh["bid"]),
                            short_ask=fnum(sh["ask"]),
                            short_mark=None,
                            long_bid=fnum(lg["bid"]),
                            long_ask=fnum(lg["ask"]),
                            long_mark=None,
                        )
                        if net is None or not np.isfinite(net) or net < 0:
                            continue
                        if net >= width:
                            continue
                        credit_ratio = net / width
                        if credit_ratio < min_credit or credit_ratio > max_credit:
                            continue
                        max_profit_cash = net * 100
                        max_loss_cash = (width - net) * 100
                        credit_rr = net / max(1e-9, (width - net))
                        if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                            continue
                        if credit_rr < min_credit_rr:
                            continue
                        eff = max(
                            0.0,
                            min(1.0, (credit_ratio - min_credit) / max(1e-9, max_credit - min_credit)),
                        )
                        liq = min(
                            math.log1p(max(0.0, fnum(lg["volume"]))) + math.log1p(max(0.0, fnum(lg["open_interest"]))),
                            math.log1p(max(0.0, fnum(sh["volume"]))) + math.log1p(max(0.0, fnum(sh["open_interest"]))),
                        )
                        liq = max(0.0, min(1.0, liq / 20.0))
                        short_fit = 1.0 - min(1.0, abs(float(sh["strike"]) - tshort) / max(1.0, 0.25 * spot))
                        score = 0.30 * bull + 0.22 * eff + 0.18 * liq + 0.14 * dte_fit + 0.10 * whale + 0.06 * short_fit
                        conv = int(round(100 * score))
                        # IVR bonus: high IV = rich premium = good for selling
                        if np.isfinite(iv_rank):
                            if iv_rank > 50:
                                conv = min(100, conv + 3)
                            elif iv_rank < 20:
                                conv = max(0, conv - 5)
                        # [T6] Macro regime: selling puts in a selloff is dangerous
                        if macro_regime == "risk_off":
                            conv = max(0, conv - 8)
                        elif macro_regime == "risk_on":
                            conv = min(100, conv + 3)  # risk_on = safe to sell puts

                        sigma_pass = bool(sigma_known and float(sh["strike"]) <= (spot - sigma * shield_sigma_relax))
                        short_otm_pct = max(0.0, (spot - float(sh["strike"])) / spot)
                        short_delta = fnum(sh.get("delta"))
                        delta_ok = (not high_beta_require_delta) or (
                            np.isfinite(short_delta) and abs(short_delta) <= 0.20
                        )
                        hb_otm_need = max(high_beta_min_otm, sigma_pct if sigma_known else high_beta_min_otm)
                        hb_pass = (not hb) or (
                            dte >= high_beta_min_dte
                            and short_otm_pct >= hb_otm_need
                            and sigma_pass
                            and delta_ok
                        )
                        core_ok = (
                            er["core_ok"]
                            and sigma_pass
                            and hb_pass
                            and dte > shield_lotto_dte_max
                        )

                        confidence_tier = (
                            "Core Income"
                            if core_ok
                            else ("Lotto" if dte <= shield_lotto_dte_max else "Aggressive")
                        )
                        if er["crossed"]:
                            confidence_tier = "REJECT"
                        elif not er["verified"]:
                            confidence_tier = "Aggressive"

                        if er["crossed"]:
                            optimal = "REJECT"
                        elif not er["verified"]:
                            optimal = "Watch Only"
                        elif conv < 65:
                            optimal = "Watch Only"
                        elif core_ok and conv >= 78:
                            optimal = "Yes-Prime"
                        else:
                            optimal = "Yes-Good"

                        if (not core_ok) and optimal == "Yes-Prime":
                            optimal = "Yes-Good"

                        gate_text = (
                            f"Gates: Earnings={er['label']}; 1sigma={'PASS' if sigma_pass else ('FAIL' if sigma_known else 'UNKNOWN')}; "
                            f"CreditWidth=PASS; WidthTier=PASS; HighBeta={'PASS' if hb_pass else 'FAIL'}; Liquidity=PASS."
                        )
                        notes = (
                            f"Credit {net:.2f} on {width:.2f} ({credit_ratio:.2%}, R/R {credit_rr:.2f}); bull={bull:.2f}, tgt={tshort:.2f}. "
                            f"{gate_text}"
                        )
                        out.append(candidate_dict(
                            ticker, "SELL", "Bull Put Credit", "SHIELD", expiry, dte, float(lg["strike"]), float(sh["strike"]),
                            width, net, "credit", max_profit_cash, max_loss_cash, float(sh["strike"] - net), conv,
                            confidence_tier, optimal, notes, f"{sh['source_csv']}|{lg['source_csv']}",
                            "Bull put credit benefits from support hold and theta decay.",
                            f"Invalidate if close < {float(sh['strike']):.2f}.",
                            iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                            sigma_pass=bool(sigma_pass),
                            core_ok=bool(core_ok),
                            high_beta_pass=bool(hb_pass),
                            earnings_label=str(er.get("label", "")),
                            **contract_flow_for_spread("Bull Put Credit", short_row=sh, long_row=lg),
                        ))

            calls = chains.get((ticker, "C", expiry))
            if allow_shield_bear and calls is not None:
                tshort = spot * (1.0 + shield_otm)
                short_pool = calls[(calls["strike"] > spot) & (calls["strike"] <= spot * 1.30)].copy()
                short_candidates = top_shield_shorts_by_gate(
                    short_pool, tshort, "C", dte, sigma, shield_sigma_relax, "credit_spread"
                )
                for _, sh in short_candidates.iterrows():
                    long_pool = calls[calls["strike"] > sh["strike"]].copy()
                    long_candidates = top_by_target(long_pool, float(sh["strike"]) + default_w)
                    for _, lg in long_candidates.iterrows():
                        if not liquid_pair_ok(sh, lg):
                            continue
                        width = float(lg["strike"] - sh["strike"])
                        if width < min_w or width > max_w:
                            continue
                        net, _ = compute_live_net(
                            net_type="credit",
                            short_bid=fnum(sh["bid"]),
                            short_ask=fnum(sh["ask"]),
                            short_mark=None,
                            long_bid=fnum(lg["bid"]),
                            long_ask=fnum(lg["ask"]),
                            long_mark=None,
                        )
                        if net is None or not np.isfinite(net) or net < 0:
                            continue
                        if net >= width:
                            continue
                        credit_ratio = net / width
                        if credit_ratio < min_credit or credit_ratio > max_credit:
                            continue
                        max_profit_cash = net * 100
                        max_loss_cash = (width - net) * 100
                        credit_rr = net / max(1e-9, (width - net))
                        if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                            continue
                        if credit_rr < min_credit_rr:
                            continue
                        eff = max(
                            0.0,
                            min(1.0, (credit_ratio - min_credit) / max(1e-9, max_credit - min_credit)),
                        )
                        liq = min(
                            math.log1p(max(0.0, fnum(lg["volume"]))) + math.log1p(max(0.0, fnum(lg["open_interest"]))),
                            math.log1p(max(0.0, fnum(sh["volume"]))) + math.log1p(max(0.0, fnum(sh["open_interest"]))),
                        )
                        liq = max(0.0, min(1.0, liq / 20.0))
                        short_fit = 1.0 - min(1.0, abs(float(sh["strike"]) - tshort) / max(1.0, 0.25 * spot))
                        score = 0.30 * bear + 0.22 * eff + 0.18 * liq + 0.14 * dte_fit + 0.10 * whale + 0.06 * short_fit
                        conv = int(round(100 * score))
                        # IVR bonus: high IV = rich premium = good for selling
                        if np.isfinite(iv_rank):
                            if iv_rank > 50:
                                conv = min(100, conv + 3)
                            elif iv_rank < 20:
                                conv = max(0, conv - 5)
                        # [T6] Macro regime: bear calls benefit from downside
                        if macro_regime == "risk_off":
                            conv = min(100, conv + 3)
                        elif macro_regime == "risk_on":
                            conv = max(0, conv - 5)  # risk_on = dangerous to sell calls

                        sigma_pass = bool(sigma_known and float(sh["strike"]) >= (spot + sigma * shield_sigma_relax))
                        short_otm_pct = max(0.0, (float(sh["strike"]) - spot) / spot)
                        short_delta = fnum(sh.get("delta"))
                        delta_ok = (not high_beta_require_delta) or (
                            np.isfinite(short_delta) and abs(short_delta) <= 0.20
                        )
                        hb_otm_need = max(high_beta_min_otm, sigma_pct if sigma_known else high_beta_min_otm)
                        hb_pass = (not hb) or (
                            dte >= high_beta_min_dte
                            and short_otm_pct >= hb_otm_need
                            and sigma_pass
                            and delta_ok
                        )
                        core_ok = (
                            er["core_ok"]
                            and sigma_pass
                            and hb_pass
                            and dte > shield_lotto_dte_max
                        )

                        confidence_tier = (
                            "Core Income"
                            if core_ok
                            else ("Lotto" if dte <= shield_lotto_dte_max else "Aggressive")
                        )
                        if er["crossed"]:
                            confidence_tier = "REJECT"
                        elif not er["verified"]:
                            confidence_tier = "Aggressive"

                        # 40% upside gate: if consensus/upside cannot be verified from data, downgrade to Watch Only.
                        consensus = str(r.get("analyst_consensus", "")).strip().lower()
                        upside = fnum(r.get("analyst_upside_pct"))
                        upside_known = bool(consensus) and np.isfinite(upside)
                        upside_fail = False
                        if upside_known:
                            upside_norm = upside / 100.0 if upside > 2.0 else upside
                            upside_fail = (consensus in {"buy", "overweight", "strong buy"}) and (upside_norm >= 0.40)

                        if er["crossed"]:
                            optimal = "REJECT"
                        elif (not er["verified"]) or (not upside_known):
                            optimal = "Watch Only"
                        elif upside_fail:
                            optimal = "REJECT"
                        elif conv < 65:
                            optimal = "Watch Only"
                        elif core_ok and conv >= 78:
                            optimal = "Yes-Prime"
                        else:
                            optimal = "Yes-Good"

                        if (not core_ok) and optimal == "Yes-Prime":
                            optimal = "Yes-Good"

                        gate_text = (
                            f"Gates: Earnings={er['label']}; 1sigma={'PASS' if sigma_pass else ('FAIL' if sigma_known else 'UNKNOWN')}; "
                            f"CreditWidth=PASS; WidthTier=PASS; HighBeta={'PASS' if hb_pass else 'FAIL'}; "
                            f"Liquidity=PASS; Upside40={'PASS' if (upside_known and not upside_fail) else ('FAIL' if upside_fail else 'UNKNOWN')}."
                        )
                        notes = (
                            f"Credit {net:.2f} on {width:.2f} ({credit_ratio:.2%}, R/R {credit_rr:.2f}); bear={bear:.2f}, tgt={tshort:.2f}. "
                            f"{gate_text}"
                        )
                        out.append(candidate_dict(
                            ticker, "SELL", "Bear Call Credit", "SHIELD", expiry, dte, float(lg["strike"]), float(sh["strike"]),
                            width, net, "credit", max_profit_cash, max_loss_cash, float(sh["strike"] + net), conv,
                            confidence_tier, optimal, notes, f"{sh['source_csv']}|{lg['source_csv']}",
                            "Bear call credit leans on overhead resistance plus theta.",
                            f"Invalidate if close > {float(sh['strike']):.2f}.",
                            iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                            sigma_pass=bool(sigma_pass),
                            core_ok=bool(core_ok),
                            high_beta_pass=bool(hb_pass),
                            earnings_label=str(er.get("label", "")),
                            **contract_flow_for_spread("Bear Call Credit", short_row=sh, long_row=lg),
                        ))

            # Iron Condor (SHIELD neutral income), built directly from live executable put/call legs.
            puts_for_condor = chains.get((ticker, "P", expiry))
            calls_for_condor = chains.get((ticker, "C", expiry))
            if puts_for_condor is not None and calls_for_condor is not None:
                tshort_put = spot * (1.0 - shield_otm)
                tshort_call = spot * (1.0 + shield_otm)

                put_short_pool = puts_for_condor[
                    (puts_for_condor["strike"] < spot) & (puts_for_condor["strike"] >= spot * 0.70)
                ].copy()
                call_short_pool = calls_for_condor[
                    (calls_for_condor["strike"] > spot) & (calls_for_condor["strike"] <= spot * 1.30)
                ].copy()

                put_short_candidates = top_shield_shorts_by_gate(
                    put_short_pool, tshort_put, "P", dte, sigma, shield_ic_sigma_relax, "iron_condor"
                )
                call_short_candidates = top_shield_shorts_by_gate(
                    call_short_pool, tshort_call, "C", dte, sigma, shield_ic_sigma_relax, "iron_condor"
                )

                put_sides = []
                for _, sh_put in put_short_candidates.iterrows():
                    long_put_pool = puts_for_condor[puts_for_condor["strike"] < sh_put["strike"]].copy()
                    long_put_candidates = top_by_target(long_put_pool, float(sh_put["strike"]) - default_w)
                    for _, lg_put in long_put_candidates.iterrows():
                        if not liquid_pair_ok(sh_put, lg_put):
                            continue
                        put_width = float(sh_put["strike"] - lg_put["strike"])
                        if put_width < min_w or put_width > max_w:
                            continue
                        put_credit, _ = compute_live_net(
                            net_type="credit",
                            short_bid=fnum(sh_put["bid"]),
                            short_ask=fnum(sh_put["ask"]),
                            short_mark=None,
                            long_bid=fnum(lg_put["bid"]),
                            long_ask=fnum(lg_put["ask"]),
                            long_mark=None,
                        )
                        if put_credit is None or not np.isfinite(put_credit) or put_credit <= 0 or put_credit >= put_width:
                            continue
                        sigma_put_pass = bool(sigma_known and float(sh_put["strike"]) <= (spot - sigma * shield_ic_sigma_relax))
                        short_put_otm_pct = max(0.0, (spot - float(sh_put["strike"])) / spot)
                        short_put_delta = fnum(sh_put.get("delta"))
                        put_delta_ok = (not high_beta_require_delta) or (
                            np.isfinite(short_put_delta) and abs(short_put_delta) <= 0.20
                        )
                        hb_otm_need = max(high_beta_min_otm, sigma_pct if sigma_known else high_beta_min_otm)
                        hb_put_pass = (not hb) or (
                            dte >= high_beta_min_dte
                            and short_put_otm_pct >= hb_otm_need
                            and sigma_put_pass
                            and put_delta_ok
                        )
                        put_sides.append(
                            {
                                "short": sh_put,
                                "long": lg_put,
                                "width": put_width,
                                "net": float(put_credit),
                                "sigma_pass": sigma_put_pass,
                                "hb_pass": hb_put_pass,
                                "short_fit": 1.0 - min(
                                    1.0,
                                    abs(float(sh_put["strike"]) - tshort_put) / max(1.0, 0.25 * spot),
                                ),
                                "liq_raw": min(liq_leg_score(sh_put), liq_leg_score(lg_put)),
                            }
                        )

                call_sides = []
                for _, sh_call in call_short_candidates.iterrows():
                    long_call_pool = calls_for_condor[calls_for_condor["strike"] > sh_call["strike"]].copy()
                    long_call_candidates = top_by_target(long_call_pool, float(sh_call["strike"]) + default_w)
                    for _, lg_call in long_call_candidates.iterrows():
                        if not liquid_pair_ok(sh_call, lg_call):
                            continue
                        call_width = float(lg_call["strike"] - sh_call["strike"])
                        if call_width < min_w or call_width > max_w:
                            continue
                        call_credit, _ = compute_live_net(
                            net_type="credit",
                            short_bid=fnum(sh_call["bid"]),
                            short_ask=fnum(sh_call["ask"]),
                            short_mark=None,
                            long_bid=fnum(lg_call["bid"]),
                            long_ask=fnum(lg_call["ask"]),
                            long_mark=None,
                        )
                        if call_credit is None or not np.isfinite(call_credit) or call_credit <= 0 or call_credit >= call_width:
                            continue
                        sigma_call_pass = bool(sigma_known and float(sh_call["strike"]) >= (spot + sigma * shield_ic_sigma_relax))
                        short_call_otm_pct = max(0.0, (float(sh_call["strike"]) - spot) / spot)
                        short_call_delta = fnum(sh_call.get("delta"))
                        call_delta_ok = (not high_beta_require_delta) or (
                            np.isfinite(short_call_delta) and abs(short_call_delta) <= 0.20
                        )
                        hb_otm_need = max(high_beta_min_otm, sigma_pct if sigma_known else high_beta_min_otm)
                        hb_call_pass = (not hb) or (
                            dte >= high_beta_min_dte
                            and short_call_otm_pct >= hb_otm_need
                            and sigma_call_pass
                            and call_delta_ok
                        )
                        call_sides.append(
                            {
                                "short": sh_call,
                                "long": lg_call,
                                "width": call_width,
                                "net": float(call_credit),
                                "sigma_pass": sigma_call_pass,
                                "hb_pass": hb_call_pass,
                                "short_fit": 1.0 - min(
                                    1.0,
                                    abs(float(sh_call["strike"]) - tshort_call) / max(1.0, 0.25 * spot),
                                ),
                                "liq_raw": min(liq_leg_score(sh_call), liq_leg_score(lg_call)),
                            }
                        )

                for p_side in put_sides:
                    for c_side in call_sides:
                        sh_put = p_side["short"]
                        lg_put = p_side["long"]
                        sh_call = c_side["short"]
                        lg_call = c_side["long"]

                        short_put = float(sh_put["strike"])
                        short_call = float(sh_call["strike"])
                        if short_put >= short_call:
                            continue

                        put_width = float(p_side["width"])
                        call_width = float(c_side["width"])
                        net = float(p_side["net"] + c_side["net"])
                        max_width = max(put_width, call_width)
                        if not np.isfinite(net) or net <= 0 or net >= max_width:
                            continue

                        credit_ratio = net / max_width
                        if credit_ratio < min_credit or credit_ratio > max_credit:
                            continue

                        risk_put = put_width - net
                        risk_call = call_width - net
                        max_loss_side = max(risk_put, risk_call)
                        if max_loss_side <= 0:
                            continue

                        max_profit_cash = net * 100
                        max_loss_cash = max_loss_side * 100
                        credit_rr = net / max(1e-9, max_loss_side)
                        if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                            continue
                        if credit_rr < min_credit_rr:
                            continue

                        eff = max(
                            0.0,
                            min(1.0, (credit_ratio - min_credit) / max(1e-9, max_credit - min_credit)),
                        )
                        liq = max(0.0, min(1.0, min(p_side["liq_raw"], c_side["liq_raw"]) / 20.0))
                        short_fit = max(0.0, min(1.0, 0.5 * (p_side["short_fit"] + c_side["short_fit"])))
                        neutrality = max(0.0, 1.0 - abs(float(bull) - float(bear)))
                        score = 0.24 * neutrality + 0.20 * eff + 0.18 * liq + 0.14 * dte_fit + 0.12 * whale + 0.12 * short_fit
                        conv = int(round(100 * score))
                        # IVR bonus: high IV = rich premium = good for IC
                        if np.isfinite(iv_rank):
                            if iv_rank > 50:
                                conv = min(100, conv + 3)
                            elif iv_rank < 20:
                                conv = max(0, conv - 5)

                        sigma_pass = bool(p_side["sigma_pass"] and c_side["sigma_pass"])
                        hb_pass = bool(p_side["hb_pass"] and c_side["hb_pass"])
                        core_ok = (
                            er["core_ok"]
                            and sigma_pass
                            and hb_pass
                            and dte > shield_lotto_dte_max
                        )
                        confidence_tier = (
                            "Core Income"
                            if core_ok
                            else ("Lotto" if dte <= shield_lotto_dte_max else "Aggressive")
                        )
                        if er["crossed"]:
                            confidence_tier = "REJECT"
                        elif not er["verified"]:
                            confidence_tier = "Aggressive"

                        if er["crossed"]:
                            optimal = "REJECT"
                        elif not er["verified"]:
                            optimal = "Watch Only"
                        elif conv < 65:
                            optimal = "Watch Only"
                        elif core_ok and conv >= 78:
                            optimal = "Yes-Prime"
                        else:
                            optimal = "Yes-Good"
                        if (not core_ok) and optimal == "Yes-Prime":
                            optimal = "Yes-Good"

                        be_low = short_put - net
                        be_high = short_call + net
                        gate_text = (
                            f"Gates: Earnings={er['label']}; 1sigma={'PASS' if sigma_pass else ('FAIL' if sigma_known else 'UNKNOWN')}; "
                            f"CreditWidth=PASS; WidthTier=PASS; HighBeta={'PASS' if hb_pass else 'FAIL'}; Liquidity=PASS."
                        )
                        notes = (
                            f"Condor credit {net:.2f} (put {put_width:.2f}w + call {call_width:.2f}w, "
                            f"{credit_ratio:.2%}, R/R {credit_rr:.2f}); neutrality={neutrality:.2f}. {gate_text}"
                        )
                        out.append(
                            candidate_dict(
                                ticker,
                                "SELL",
                                "Iron Condor",
                                "SHIELD",
                                expiry,
                                dte,
                                float(lg_put["strike"]),
                                float(sh_put["strike"]),
                                max_width,
                                net,
                                "credit",
                                max_profit_cash,
                                max_loss_cash,
                                be_low,
                                conv,
                                confidence_tier,
                                optimal,
                                notes,
                                f"{sh_put['source_csv']}|{lg_put['source_csv']}|{sh_call['source_csv']}|{lg_call['source_csv']}",
                                "Iron condor collects premium when price stays in a defined range.",
                                f"Invalidate if close < {short_put:.2f} or close > {short_call:.2f}.",
                                iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                                breakeven_low=be_low,
                                breakeven_high=be_high,
                                short_put_strike=short_put,
                                long_put_strike=float(lg_put["strike"]),
                                short_call_strike=short_call,
                                long_call_strike=float(lg_call["strike"]),
                                put_width=put_width,
                                call_width=call_width,
                                short_put_symbol=str(sh_put.get("option_symbol", "")),
                                long_put_symbol=str(lg_put.get("option_symbol", "")),
                                short_call_symbol=str(sh_call.get("option_symbol", "")),
                                long_call_symbol=str(lg_call.get("option_symbol", "")),
                                sigma_pass=bool(sigma_pass),
                                core_ok=bool(core_ok),
                                high_beta_pass=bool(hb_pass),
                                earnings_label=str(er.get("label", "")),
                                range_neutrality=float(neutrality),
                                **contract_flow_for_spread(
                                    "Iron Condor",
                                    short_put_row=sh_put,
                                    short_call_row=sh_call,
                                ),
                            )
                        )

                        # Long Iron Condor (FIRE breakout), debit-defined risk.
                        if enable_long_iron_condor:
                            long_put = float(sh_put["strike"])
                            short_put = float(lg_put["strike"])
                            long_call = float(sh_call["strike"])
                            short_call = float(lg_call["strike"])
                            if not (short_put < long_put < long_call < short_call):
                                continue

                            put_debit, _ = compute_live_net(
                                net_type="debit",
                                short_bid=fnum(lg_put["bid"]),
                                short_ask=fnum(lg_put["ask"]),
                                short_mark=None,
                                long_bid=fnum(sh_put["bid"]),
                                long_ask=fnum(sh_put["ask"]),
                                long_mark=None,
                            )
                            call_debit, _ = compute_live_net(
                                net_type="debit",
                                short_bid=fnum(lg_call["bid"]),
                                short_ask=fnum(lg_call["ask"]),
                                short_mark=None,
                                long_bid=fnum(sh_call["bid"]),
                                long_ask=fnum(sh_call["ask"]),
                                long_mark=None,
                            )
                            net_debit = (
                                float(put_debit + call_debit)
                                if (put_debit is not None and call_debit is not None)
                                else math.nan
                            )
                            max_width = max(put_width, call_width)
                            if not np.isfinite(net_debit) or net_debit <= 0 or net_debit >= max_width:
                                continue

                            debit_ratio = net_debit / max_width
                            if debit_ratio > breakout_max_debit_pct_width:
                                continue
                            max_profit_cash = (max_width - net_debit) * 100
                            max_loss_cash = net_debit * 100
                            debit_rr = (max_width - net_debit) / max(1e-9, net_debit)
                            if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                                continue
                            if debit_rr < min_debit_rr:
                                continue

                            center_width = max(0.0, long_call - long_put)
                            center_width_pct = center_width / spot
                            if (
                                np.isfinite(breakout_max_center_width_pct)
                                and center_width_pct > breakout_max_center_width_pct
                            ):
                                continue

                            neutrality = max(0.0, 1.0 - abs(float(bull) - float(bear)))
                            if (
                                np.isfinite(breakout_min_neutrality)
                                and neutrality < breakout_min_neutrality
                            ):
                                continue

                            breakout_signal = math.nan
                            if sigma_known and sigma_pct > 0:
                                breakout_signal = sigma_pct
                            elif np.isfinite(implied_move_pct_norm):
                                breakout_signal = implied_move_pct_norm
                            if (
                                np.isfinite(breakout_min_implied_move_pct)
                                and np.isfinite(breakout_signal)
                                and breakout_signal < breakout_min_implied_move_pct
                            ):
                                continue

                            liq = max(0.0, min(1.0, min(p_side["liq_raw"], c_side["liq_raw"]) / 20.0))
                            eff = max(
                                0.0,
                                min(
                                    1.0,
                                    (breakout_max_debit_pct_width - debit_ratio)
                                    / max(1e-9, breakout_max_debit_pct_width),
                                ),
                            )
                            breakout_fit = 0.0
                            if np.isfinite(breakout_signal) and breakout_signal > 0:
                                breakout_fit = max(
                                    0.0,
                                    min(1.0, (breakout_signal - center_width_pct) / max(1e-9, breakout_signal)),
                                )
                            score = (
                                0.24 * eff
                                + 0.20 * breakout_fit
                                + 0.18 * liq
                                + 0.14 * dte_fit
                                + 0.12 * whale
                                + 0.12 * neutrality
                            )
                            conv = int(round(100 * score))

                            core_ok = bool(
                                dte > fire_lotto_dte_max
                                and np.isfinite(debit_rr)
                                and debit_rr >= max(1.3, min_debit_rr)
                            )
                            confidence_tier = (
                                "CORE" if core_ok else ("Lotto" if dte <= fire_lotto_dte_max else "AGG")
                            )
                            if er["crossed"]:
                                confidence_tier = "AGG"

                            if conv < 60:
                                optimal = "Watch Only"
                            elif core_ok and conv >= 78:
                                optimal = "Yes-Prime"
                            else:
                                optimal = "Yes-Good"

                            be_low = long_put - net_debit
                            be_high = long_call + net_debit
                            breakout_sig_txt = (
                                f"{breakout_signal:.2%}" if np.isfinite(breakout_signal) else "N/A"
                            )
                            gate_text = (
                                f"Gates: DebitWidth=PASS; WidthTier=PASS; Liquidity=PASS; "
                                f"BreakoutSignal={breakout_sig_txt}; "
                                f"CenterWidth={center_width_pct:.2%}; Earnings={er['label']}."
                            )
                            notes = (
                                f"Long condor debit {net_debit:.2f} (put {put_width:.2f}w + call {call_width:.2f}w, "
                                f"debit/width {debit_ratio:.2%}, R/R {debit_rr:.2f}); neutrality={neutrality:.2f}. {gate_text}"
                            )
                            out.append(
                                candidate_dict(
                                    ticker,
                                    "BUY",
                                    "Long Iron Condor",
                                    "FIRE",
                                    expiry,
                                    dte,
                                    long_put,
                                    short_put,
                                    max_width,
                                    net_debit,
                                    "debit",
                                    max_profit_cash,
                                    max_loss_cash,
                                    be_low,
                                    conv,
                                    confidence_tier,
                                    optimal,
                                    notes,
                                    f"{sh_put['source_csv']}|{lg_put['source_csv']}|{sh_call['source_csv']}|{lg_call['source_csv']}",
                                    "Long iron condor targets an outsized move with defined risk.",
                                    f"Invalidate if close remains between {long_put:.2f} and {long_call:.2f} into late cycle.",
                                    iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                                    breakeven_low=be_low,
                                    breakeven_high=be_high,
                                    short_put_strike=short_put,
                                    long_put_strike=long_put,
                                    short_call_strike=short_call,
                                    long_call_strike=long_call,
                                    put_width=put_width,
                                    call_width=call_width,
                                    short_put_symbol=str(lg_put.get("option_symbol", "")),
                                    long_put_symbol=str(sh_put.get("option_symbol", "")),
                                    short_call_symbol=str(lg_call.get("option_symbol", "")),
                                    long_call_symbol=str(sh_call.get("option_symbol", "")),
                                    sigma_pass=math.nan,
                                    core_ok=bool(core_ok),
                                    high_beta_pass=math.nan,
                                    earnings_label=str(er.get("label", "")),
                                    range_neutrality=float(neutrality),
                                )
                            )

                # Iron Butterfly (SHIELD neutral/range income), centered short strike.
                if enable_iron_butterfly:
                    put_short_by_strike = {}
                    for _, sh_put in put_short_candidates.iterrows():
                        put_short_by_strike[round(float(sh_put["strike"]), 4)] = sh_put
                    call_short_by_strike = {}
                    for _, sh_call in call_short_candidates.iterrows():
                        call_short_by_strike[round(float(sh_call["strike"]), 4)] = sh_call
                    centers = sorted(
                        set(put_short_by_strike.keys()) & set(call_short_by_strike.keys()),
                        key=lambda k: abs(float(k) - float(spot)),
                    )[: max(1, strike_search_depth // 2)]
                    for center in centers:
                        sh_put = put_short_by_strike[center]
                        sh_call = call_short_by_strike[center]
                        if not liquid_leg_ok(sh_put) or not liquid_leg_ok(sh_call):
                            continue
                        center_strike = float(center)
                        long_put_pool = puts_for_condor[puts_for_condor["strike"] < center_strike].copy()
                        long_put_candidates = top_by_target(long_put_pool, center_strike - default_w)
                        long_call_pool = calls_for_condor[calls_for_condor["strike"] > center_strike].copy()
                        long_call_candidates = top_by_target(long_call_pool, center_strike + default_w)
                        for _, lg_put in long_put_candidates.iterrows():
                            if not liquid_pair_ok(sh_put, lg_put):
                                continue
                            put_width = center_strike - float(lg_put["strike"])
                            if put_width < min_w or put_width > max_w:
                                continue
                            for _, lg_call in long_call_candidates.iterrows():
                                if not liquid_pair_ok(sh_call, lg_call):
                                    continue
                                call_width = float(lg_call["strike"]) - center_strike
                                if call_width < min_w or call_width > max_w:
                                    continue
                                net = (
                                    fnum(sh_put.get("bid"))
                                    + fnum(sh_call.get("bid"))
                                    - fnum(lg_put.get("ask"))
                                    - fnum(lg_call.get("ask"))
                                )
                                max_width = max(put_width, call_width)
                                if not np.isfinite(net) or net <= 0 or net >= max_width:
                                    continue
                                credit_ratio = net / max_width
                                if credit_ratio < min_credit or credit_ratio > max_credit:
                                    continue
                                max_profit_cash = net * 100
                                max_loss_cash = (max_width - net) * 100
                                credit_rr = net / max(1e-9, (max_width - net))
                                if np.isfinite(max_risk_per_trade) and max_loss_cash > max_risk_per_trade:
                                    continue
                                if credit_rr < min_credit_rr:
                                    continue

                                neutrality = max(0.0, 1.0 - abs(float(bull) - float(bear)))
                                if np.isfinite(range_min_neutrality) and neutrality < range_min_neutrality:
                                    continue
                                if np.isfinite(range_min_iv_rank) and np.isfinite(iv_rank) and iv_rank < range_min_iv_rank:
                                    continue

                                eff = max(
                                    0.0,
                                    min(1.0, (credit_ratio - min_credit) / max(1e-9, max_credit - min_credit)),
                                )
                                liq_raw = min(
                                    liq_leg_score(sh_put),
                                    liq_leg_score(lg_put),
                                    liq_leg_score(sh_call),
                                    liq_leg_score(lg_call),
                                )
                                liq = max(0.0, min(1.0, liq_raw / 20.0))
                                center_fit = 1.0 - min(1.0, abs(center_strike - spot) / max(1.0, 0.20 * spot))
                                score = 0.30 * neutrality + 0.20 * eff + 0.18 * liq + 0.14 * dte_fit + 0.10 * whale + 0.08 * center_fit
                                conv = int(round(100 * score))

                                core_ok = bool(er["core_ok"] and dte > shield_lotto_dte_max and neutrality >= 0.75)
                                confidence_tier = "Core Income" if core_ok else ("Lotto" if dte <= shield_lotto_dte_max else "Aggressive")
                                if er["crossed"]:
                                    confidence_tier = "REJECT"
                                elif not er["verified"]:
                                    confidence_tier = "Aggressive"

                                if er["crossed"]:
                                    optimal = "REJECT"
                                elif not er["verified"]:
                                    optimal = "Watch Only"
                                elif conv < 65:
                                    optimal = "Watch Only"
                                elif core_ok and conv >= 78:
                                    optimal = "Yes-Prime"
                                else:
                                    optimal = "Yes-Good"

                                be_low = center_strike - net
                                be_high = center_strike + net
                                gate_text = (
                                    f"Gates: Earnings={er['label']}; RangeNeutrality={neutrality:.2f}; "
                                    f"CreditWidth=PASS; WidthTier=PASS; Liquidity=PASS."
                                )
                                notes = (
                                    f"Butterfly credit {net:.2f} @ center {center_strike:.2f} (put {put_width:.2f}w + call {call_width:.2f}w, "
                                    f"{credit_ratio:.2%}, R/R {credit_rr:.2f}). {gate_text}"
                                )
                                out.append(
                                    candidate_dict(
                                        ticker,
                                        "SELL",
                                        "Iron Butterfly",
                                        "SHIELD",
                                        expiry,
                                        dte,
                                        float(lg_put["strike"]),
                                        center_strike,
                                        max_width,
                                        net,
                                        "credit",
                                        max_profit_cash,
                                        max_loss_cash,
                                        be_low,
                                        conv,
                                        confidence_tier,
                                        optimal,
                                        notes,
                                        f"{sh_put['source_csv']}|{lg_put['source_csv']}|{sh_call['source_csv']}|{lg_call['source_csv']}",
                                        "Iron butterfly collects premium around a pinned center strike with defined wings.",
                                        f"Invalidate if close < {be_low:.2f} or close > {be_high:.2f}.",
                                        iv_rank=float(iv_rank) if np.isfinite(iv_rank) else None,
                                        breakeven_low=be_low,
                                        breakeven_high=be_high,
                                        short_put_strike=center_strike,
                                        long_put_strike=float(lg_put["strike"]),
                                        short_call_strike=center_strike,
                                        long_call_strike=float(lg_call["strike"]),
                                        put_width=put_width,
                                        call_width=call_width,
                                        short_put_symbol=str(sh_put.get("option_symbol", "")),
                                        long_put_symbol=str(lg_put.get("option_symbol", "")),
                                        short_call_symbol=str(sh_call.get("option_symbol", "")),
                                        long_call_symbol=str(lg_call.get("option_symbol", "")),
                                        sigma_pass=math.nan,
                                        core_ok=bool(core_ok),
                                        high_beta_pass=math.nan,
                                        earnings_label=str(er.get("label", "")),
                                        range_neutrality=float(neutrality),
                                    )
                                )

    if not out:
        return []

    df = pd.DataFrame(out)
    # Keep non-rejected rows so SHIELD candidates are still visible when they are
    # downgraded to Watch Only by missing/unknown gates (for example upside gate).
    if include_watch_candidates:
        df = df[df["optimal"] != "REJECT"].copy()
    else:
        df = df[df["optimal"].isin(["Yes-Prime", "Yes-Good"])].copy()
    if df.empty:
        return []
    df = df.sort_values(
        [
            "conviction",
            "max_loss",
            "max_profit",
            "track",
            "ticker",
            "strategy",
            "expiry",
            "short_strike",
            "long_strike",
        ],
        ascending=[False, True, False, True, True, True, True, True, True],
    )

    # Apply rulebook-style engine caps to avoid combinational strike spam.
    selected_rows = []
    used_keys = set()
    per_ticker_count = defaultdict(int)
    per_ticker_track_count = defaultdict(int)
    per_ticker_expiries = defaultdict(set)

    def try_add(row):
        def key_num(v):
            xv = fnum(v)
            if np.isfinite(xv):
                return round(float(xv), 4)
            return None

        key = (
            row["ticker"],
            row["strategy"],
            row["expiry"],
            key_num(row.get("long_strike")),
            key_num(row.get("short_strike")),
            key_num(row.get("long_call_strike")),
            key_num(row.get("short_call_strike")),
        )
        if key in used_keys:
            return False
        ticker = row["ticker"]
        track = str(row.get("track", "")).strip().upper() or "UNKNOWN"
        expiry = row["expiry"]
        if per_ticker_count[ticker] >= max(1, max_trades_per_ticker):
            return False
        if per_ticker_track_count[(ticker, track)] >= max(1, max_trades_per_ticker_per_track):
            return False
        if expiry not in per_ticker_expiries[ticker] and len(per_ticker_expiries[ticker]) >= max(
            1, max_expiries_per_ticker
        ):
            return False
        used_keys.add(key)
        per_ticker_count[ticker] += 1
        per_ticker_track_count[(ticker, track)] += 1
        per_ticker_expiries[ticker].add(expiry)
        selected_rows.append(row.to_dict())
        return True

    # Seed best per track AND direction, so bears surface even when bulls dominate.
    bear_strategies = {"Bear Put Debit", "Bear Call Credit"}
    for track in ["FIRE", "SHIELD"]:
        sub = df[df["track"] == track]
        if sub.empty:
            continue
        # Seed best bull candidate
        sub_bull = sub[~sub["strategy"].isin(bear_strategies)]
        if not sub_bull.empty:
            added = 0
            for _, row in sub_bull.iterrows():
                if len(selected_rows) >= max(1, max_total_trades):
                    break
                if try_add(row):
                    added += 1
                if added >= min_per_track:
                    break
        # Seed best bear candidate (ensures bears always surface if available)
        sub_bear = sub[sub["strategy"].isin(bear_strategies)]
        if not sub_bear.empty:
            added = 0
            for _, row in sub_bear.iterrows():
                if len(selected_rows) >= max(1, max_total_trades):
                    break
                if try_add(row):
                    added += 1
                if added >= min_per_track:
                    break

    for _, row in df.iterrows():
        if len(selected_rows) >= max(1, max_total_trades):
            break
        try_add(row)

    selected_rows.sort(
        key=lambda r: (
            -int(r["conviction"]),
            float(r["max_loss"]) if np.isfinite(r["max_loss"]) else 1e18,
            -float(r["max_profit"]) if np.isfinite(r["max_profit"]) else -1e18,
            str(r["track"]),
            str(r["ticker"]),
            str(r["strategy"]),
            str(r["expiry"]),
        )
    )
    return selected_rows


def row_from_candidate(i, r):
    if r["strategy"] == "Bull Call Debit":
        ss = f"Buy {r['long_strike']:.2f}C / Sell {r['short_strike']:.2f}C ({r['width']:.2f}w)"
    elif r["strategy"] == "Bear Put Debit":
        ss = f"Buy {r['long_strike']:.2f}P / Sell {r['short_strike']:.2f}P ({r['width']:.2f}w)"
    elif r["strategy"] == "Bull Put Credit":
        ss = f"Sell {r['short_strike']:.2f}P / Buy {r['long_strike']:.2f}P ({r['width']:.2f}w)"
    elif r["strategy"] in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        ss = (
            f"Sell {fnum(r.get('short_put_strike')):.2f}P / Buy {fnum(r.get('long_put_strike')):.2f}P + "
            f"Sell {fnum(r.get('short_call_strike')):.2f}C / Buy {fnum(r.get('long_call_strike')):.2f}C"
        )
        if r["strategy"] == "Long Iron Condor":
            ss = (
                f"Buy {fnum(r.get('long_put_strike')):.2f}P / Sell {fnum(r.get('short_put_strike')):.2f}P + "
                f"Buy {fnum(r.get('long_call_strike')):.2f}C / Sell {fnum(r.get('short_call_strike')):.2f}C"
            )
    else:
        ss = f"Sell {r['short_strike']:.2f}C / Buy {r['long_strike']:.2f}C ({r['width']:.2f}w)"

    net = "N/A"
    if np.isfinite(r["net"]):
        net = f"{'Credit' if r['net_type'] == 'credit' else 'Debit'} {r['net']:.2f}"

    strategy = str(r.get("strategy", "")).strip()
    strategy_upper = strategy.upper()
    track = str(r.get("track", "")).strip().upper()
    optimal = str(r.get("optimal", "")).strip()
    if optimal == "REJECT":
        action_display = "\U0001F7E5 REJECT"
    elif optimal == "Watch Only":
        action_display = (
            "\U0001F6E1\ufe0f\U0001F7E8 WATCH ONLY"
            if track == "SHIELD"
            else "\U0001F525\U0001F7E8 WATCH ONLY"
        )
    elif strategy == "Bull Put Credit":
        action_display = "\U0001F6E1\ufe0f\U0001F7E9 BULL PUT CREDIT"
    elif strategy == "Bear Call Credit":
        action_display = "\U0001F6E1\ufe0f\U0001F7E5 BEAR CALL CREDIT"
    elif strategy == "Bull Call Debit":
        action_display = "\U0001F525\U0001F7E6 BULL CALL DEBIT"
    elif strategy == "Bear Put Debit":
        action_display = "\U0001F525\U0001F7E7 BEAR PUT DEBIT"
    elif strategy == "Iron Condor":
        action_display = "\U0001F6E1\ufe0f\U0001F7EA IRON CONDOR"
    elif strategy == "Iron Butterfly":
        action_display = "\U0001F6E1\ufe0f\U0001F7EA IRON BUTTERFLY"
    elif strategy == "Long Iron Condor":
        action_display = "\U0001F525\U0001F7EA LONG IRON CONDOR"
    else:
        action_display = strategy_upper

    if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        be_low = fnum(r.get("breakeven_low"))
        be_high = fnum(r.get("breakeven_high"))
        be_txt = (
            f"{be_low:.2f} / {be_high:.2f}"
            if np.isfinite(be_low) and np.isfinite(be_high)
            else px(r.get("breakeven"))
        )
    else:
        be_txt = px(r["breakeven"])

    return {
        "#": i,
        "Ticker": r["ticker"],
        "Action": action_display,
        "Strategy Type": r["strategy"],
        "Strike Setup": ss,
        "Expiry": r["expiry"].isoformat() if hasattr(r["expiry"], "isoformat") else str(r["expiry"]),
        "DTE": int(r["dte"]),
        "Net Credit/Debit": net,
        "Max Profit": money(r["max_profit"]),
        "Max Loss": money(r["max_loss"]),
        "Breakeven": be_txt,
        "Conviction %": f"{int(r['conviction'])}%",
        "Confidence Tier": str(r.get("tier", "")),
        "Optimal": str(r.get("optimal", "")),
        "Notes": r["notes"],
        "Source": str(r["source"]).replace("|", " + "),
    }


def main():
    ap = argparse.ArgumentParser(description="MODE A - EOD TRADE SCAN")
    ap.add_argument("--base-dir", default=r"c:\uw_root\2026-02-05")
    ap.add_argument("--config", default=str((Path(__file__).resolve().parent / "rulebook_config.yaml")))
    ap.add_argument("--output", default=r"c:\uw_root\2026-02-05\anu-expert-trade-table-2026-02-05.md")
    ap.add_argument(
        "--top-trades",
        type=int,
        default=20,
        help="Maximum number of final trades to output (further bounded by rulebook engine.max_total_trades).",
    )
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    input_zips = non_bot_input_zips(base)
    if not input_zips:
        raise FileNotFoundError(f"No non-bot input .zip files found in {base}")
    asof_str = expected_input_date(base) or find_asof(input_zips)
    asof = dt.datetime.strptime(asof_str, "%Y-%m-%d").date()

    csv_map = unzip_zips(input_zips, base / "_unzipped_mode_a")
    print("Unzipped CSV files:")
    for zname, cpath in csv_map.items():
        print(f"  {zname} -> {cpath.name}")

    def pick(prefix):
        return pick_csv_map_entry(csv_map, prefix, asof_str)

    try:
        oi_zip, oi_csv = pick("chain-oi-changes-")
        dp_zip, dp_csv = pick("dp-eod-report-")
        hot_zip, hot_csv = pick("hot-chains-")
        sc_zip, sc_csv = pick("stock-screener-")
    except Exception as exc:
        print(f"[EXCEPTION] {type(exc).__name__}: {exc}")
        raise

    bot_eod_source = find_bot_eod_source(base, asof_str)
    whale_flow = load_yes_prime_whale_flow(bot_eod_source, cfg)
    whale_source_name = bot_eod_source.name
    tables = whale_flow.as_rank_tables()
    whale_symbol_summary_csv = base / "_unzipped_mode_a" / f"whale-symbol-summary-{asof_str}.csv"
    whale_top_trades_csv = base / "_unzipped_mode_a" / f"whale-top-trades-{asof_str}.csv"
    whale_flow.symbol_summary.to_csv(whale_symbol_summary_csv, index=False)
    whale_flow.top_trades.to_csv(whale_top_trades_csv, index=False)
    print(
        "Loaded bot EOD whale source: "
        f"{whale_source_name}; scanned={whale_flow.total_rows:,}; "
        f"yes_prime={whale_flow.yes_prime_rows:,}; symbols={len(whale_flow.symbol_summary):,}"
    )

    try:
        hot_df = pd.read_csv(hot_csv, low_memory=False)
        oi_df = pd.read_csv(oi_csv, low_memory=False)
        dp_df = pd.read_csv(dp_csv, low_memory=False)
        sc_df = pd.read_csv(sc_csv, low_memory=False)
    except Exception as exc:
        print(f"[EXCEPTION] {type(exc).__name__}: {exc}")
        raise

    print(f"Loaded: {hot_csv.name} rows={len(hot_df):,}")
    print(f"Loaded: {oi_csv.name} rows={len(oi_df):,}")
    print(f"Loaded: {dp_csv.name} rows={len(dp_df):,}")
    print(f"Loaded: {sc_csv.name} rows={len(sc_df):,}")

    try:
        ensure_cols(hot_df, hot_csv.name, ["option_symbol", "date", "bid", "ask", "volume", "open_interest"])
        ensure_cols(oi_df, oi_csv.name, ["option_symbol", "curr_date", "last_bid", "last_ask", "curr_oi", "volume"])
        ensure_cols(sc_df, sc_csv.name, ["ticker", "close", "issue_type", "is_index", "next_earnings_date", "bullish_premium", "bearish_premium", "call_premium", "put_premium", "put_call_ratio"])
    except Exception as exc:
        print(f"[EXCEPTION] {type(exc).__name__}: {exc}")
        raise

    quotes = build_quotes(hot_df, oi_df, asof, hot_csv.name, oi_csv.name)
    print(f"Loaded same-day quote rows (hot + oi): {len(quotes):,}")

    best = build_best_candidates(asof, cfg, sc_df, quotes, tables, top_trades=args.top_trades)
    if not best:
        raise RuntimeError("No candidates produced under hard gates.")

    out_rows = [row_from_candidate(i + 1, r) for i, r in enumerate(best)]
    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "#",
            "Ticker",
            "Action",
            "Strategy Type",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Confidence Tier",
            "Optimal",
            "Notes",
            "Source",
        ],
    )

    files_used = [oi_zip, dp_zip, hot_zip, sc_zip, bot_eod_source.name]
    lines = [
        f"As-of date used: {asof_str}",
        f"Files used: {', '.join(files_used)}",
        "",
        "## Anu Expert Trade Table",
        out_df.to_markdown(index=False),
        "",
        "Ticker thesis + invalidation (Yes-Prime / Yes-Good):",
    ]
    seen = set()
    for r in best:
        if r["optimal"] not in {"Yes-Prime", "Yes-Good"}:
            continue
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        lines.append(f"- {r['ticker']}: {r['thesis']} Invalidation: {r['invalidation']}")
    if not seen:
        lines.append("- none")

    out_path = Path(args.output).resolve()
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote: {out_path}")
    print("")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        # Some Windows consoles default to cp1252 and cannot print emoji.
        print("\n".join(lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
