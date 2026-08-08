"""Executable test of the IV-mean-reversion variance premium.

Every prior lead in this repo died at exactly this step: a clean statistical
effect that could not survive real quotes and real crossings. This runs the
IDENTICAL short-vol structure on the high-IV-change quintile (signal) and the
low-IV-change quintile (control), so any difference is the signal itself and not
the structure, the regime, or the tape.

Structure: short vertical spreads, 25-50 DTE, HELD TO EXPIRY.
  - held to expiry = 2 crossings, not 4. Closing early doubles the cost and
    earns a fraction of the theta, which is what killed the earlier vol lanes.
  - short strike ~1.0 implied sigma OTM, long strike ~1.9 sigma OTM, so the
    distance scales with each name's own vol instead of a fixed percentage.
  - both a put spread and a call spread are built, so the book is direction
    free. Measured direction skill in this dataset is AUC 0.53.
Fills are honest: sell the short leg at the BID, buy the long leg at the ASK.
"""
from __future__ import annotations

import argparse
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

E_ABS = math.sqrt(2.0 / math.pi)
VOL_ETPS = {"UVXY", "VXX", "SVIX", "SVXY", "VIXY", "UVIX", "VIXM", "VXZ",
            "SQQQ", "TQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "TNA", "TZA"}
OCC_RE = re.compile(r"^([A-Z0-9\.\-]{1,6})(\d{6})([CP])(\d{8})$")

CHAIN_COLS = ["option_symbol", "last_bid", "last_ask", "dte", "stock_price",
              "last_date", "curr_date", "last_oi", "curr_oi"]


def parse_occ(sym: pd.Series) -> pd.DataFrame:
    ex = sym.str.extract(OCC_RE)
    ex.columns = ["root", "yymmdd", "cp", "strike8"]
    out = pd.DataFrame(index=sym.index)
    out["root"] = ex["root"]
    out["expiry"] = pd.to_datetime(ex["yymmdd"], format="%y%m%d", errors="coerce")
    out["is_call"] = ex["cp"].eq("C")
    out["strike"] = pd.to_numeric(ex["strike8"], errors="coerce") / 1000.0
    return out


def load_chain(base: Path, date: pd.Timestamp, tickers: set[str]) -> pd.DataFrame | None:
    d = date.strftime("%Y-%m-%d")
    zp = base / d / f"chain-oi-changes-{d}.zip"
    if not zp.exists():
        return None
    try:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                return None
            with z.open(names[0]) as fh:
                df = pd.read_csv(fh, usecols=lambda c: c in set(CHAIN_COLS),
                                 low_memory=False)
    except Exception:
        return None
    if "option_symbol" not in df.columns:
        return None
    df["option_symbol"] = df["option_symbol"].astype(str).str.strip().str.upper()
    occ = parse_occ(df["option_symbol"])
    df = pd.concat([df, occ], axis=1)
    df = df[df["root"].isin(tickers)]
    for c in ("last_bid", "last_ask", "dte", "stock_price", "last_oi"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_vertical(chain: pd.DataFrame, spot: float, sigma_move: float,
                  is_call: bool, short_sig: float, long_sig: float,
                  min_oi: int, max_spread_pct: float):
    """Return (short_row, long_row) or None. Strikes are sigma-scaled, not fixed %."""
    side = chain[chain["is_call"] == is_call]
    if side.empty:
        return None
    sign = 1.0 if is_call else -1.0
    k_short = spot * (1.0 + sign * short_sig * sigma_move)
    k_long = spot * (1.0 + sign * long_sig * sigma_move)

    def nearest(target: float):
        d = (side["strike"] - target).abs()
        if d.empty:
            return None
        row = side.loc[d.idxmin()]
        return row if np.isfinite(d.min()) else None

    s, l = nearest(k_short), nearest(k_long)
    if s is None or l is None or s["strike"] == l["strike"]:
        return None
    if is_call and not (l["strike"] > s["strike"] > spot):
        return None
    if (not is_call) and not (l["strike"] < s["strike"] < spot):
        return None
    for row in (s, l):
        if not np.isfinite(row["last_bid"]) or not np.isfinite(row["last_ask"]):
            return None
        if row["last_bid"] <= 0.01 or row["last_ask"] <= 0:
            return None
        if (row["last_oi"] or 0) < min_oi:
            return None
        mid = 0.5 * (row["last_bid"] + row["last_ask"])
        if mid <= 0 or (row["last_ask"] - row["last_bid"]) / mid > max_spread_pct:
            return None
    return s, l


def score(s, l, width: float, expiry_spot: float, is_call: bool):
    """Honest fill: short leg sells at BID, long leg buys at ASK. Settle intrinsic."""
    credit = float(s["last_bid"]) - float(l["last_ask"])
    if credit <= 0:
        return None
    max_risk = width - credit
    if max_risk <= 0.01:
        return None
    if is_call:
        loss = max(0.0, min(expiry_spot, l["strike"]) - s["strike"])
    else:
        loss = max(0.0, s["strike"] - max(expiry_spot, l["strike"]))
    pnl = credit - loss
    return {"credit": credit, "max_risk": max_risk, "pnl": pnl,
            "r": pnl / max_risk, "width": width}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--feature", default="iv_chg_1w")
    ap.add_argument("--min-mcap", type=float, default=2e9)
    ap.add_argument("--min-dte", type=int, default=25)
    ap.add_argument("--max-dte", type=int, default=50)
    ap.add_argument("--short-sigma", type=float, default=1.0)
    ap.add_argument("--long-sigma", type=float, default=1.9)
    ap.add_argument("--min-oi", type=int, default=25)
    ap.add_argument("--max-spread-pct", type=float, default=0.25)
    ap.add_argument("--out", default="out/vrp_executable_trades.csv")
    args = ap.parse_args()

    base = Path(args.base_dir)
    keep = {"date", "ticker", "sector", "issue_type", "marketcap", "close",
            "iv30d", "iv_rank", "next_earnings_date", args.feature}
    p = pd.read_csv(args.panel, usecols=lambda c: c in keep, low_memory=False)
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["ticker"].notna()]
    p["ticker"] = p["ticker"].astype(str).str.upper()
    p = p[~p["ticker"].isin(VOL_ETPS)]
    p = p[p["issue_type"].astype(str).str.contains("Common", case=False, na=False)]
    p = p[pd.to_numeric(p["marketcap"], errors="coerce").fillna(0) >= args.min_mcap]
    p = p[pd.to_numeric(p["iv30d"], errors="coerce") > 0.01]
    p = p[pd.to_numeric(p["close"], errors="coerce") > 5.0]
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    p = p[p[args.feature].notna()]

    r = p.groupby("date")[args.feature].rank(pct=True, method="average")
    p["q"] = np.ceil(r * 5).clip(1, 5).astype(int)

    # close_by[(ticker, date)] -> settlement price, for expiry intrinsic value.
    close_by = p.set_index(["ticker", "date"])["close"].to_dict()
    sessions = np.array(sorted(p["date"].unique()))

    rows = []
    for i, sig_date in enumerate(sessions):
        # A chain-oi file dated D carries last_bid/last_ask for session D-1.
        # Entry is the session AFTER the signal, so read the file two ahead.
        if i + 2 >= len(sessions):
            break
        entry_date, quote_file_date = sessions[i + 1], sessions[i + 2]
        day = p[(p["date"] == sig_date) & (p["q"].isin([1, 5]))]
        if day.empty:
            continue
        chain = load_chain(base, pd.Timestamp(quote_file_date),
                           set(day["ticker"].unique()))
        if chain is None or chain.empty:
            continue
        chain = chain[chain["dte"].between(args.min_dte, args.max_dte)]
        if chain.empty:
            continue
        by_root = dict(tuple(chain.groupby("root")))

        for _, sig in day.iterrows():
            sub = by_root.get(sig["ticker"])
            if sub is None or sub.empty:
                continue
            spot = float(sub["stock_price"].median())
            if not np.isfinite(spot) or spot <= 0:
                continue
            # Use the expiry with the most contracts inside the DTE band.
            exp = sub["expiry"].value_counts().idxmax()
            sub = sub[sub["expiry"] == exp]
            dte = float(sub["dte"].median())
            settle = close_by.get((sig["ticker"], pd.Timestamp(exp)))
            if settle is None or not np.isfinite(settle):
                continue
            sigma_move = float(sig["iv30d"]) * math.sqrt(max(dte, 1) / 365.0)
            if not np.isfinite(sigma_move) or sigma_move <= 0:
                continue

            for is_call in (True, False):
                pick = pick_vertical(sub, spot, sigma_move, is_call,
                                     args.short_sigma, args.long_sigma,
                                     args.min_oi, args.max_spread_pct)
                if pick is None:
                    continue
                s, l = pick
                width = abs(float(l["strike"]) - float(s["strike"]))
                res = score(s, l, width, float(settle), is_call)
                if res is None:
                    continue
                rows.append({
                    "signal_date": sig_date, "entry_date": entry_date,
                    "ticker": sig["ticker"], "sector": sig["sector"],
                    "q": int(sig["q"]), "feature": float(sig[args.feature]),
                    "iv30d": float(sig["iv30d"]), "iv_rank": sig.get("iv_rank"),
                    "side": "call" if is_call else "put", "dte": dte,
                    "expiry": exp, "spot": spot, "settle": float(settle),
                    "short_strike": float(s["strike"]), "long_strike": float(l["strike"]),
                    **res,
                })
        if (i + 1) % 20 == 0:
            print(f"  {pd.Timestamp(sig_date).date()}  trades so far {len(rows):,}", flush=True)

    t = pd.DataFrame(rows)
    if t.empty:
        print("no trades built")
        return
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(outp, index=False)
    print(f"\nwrote {len(t):,} trades -> {outp}")
    summarize(t)


def pf(x: pd.Series) -> float:
    w, lo = x[x > 0].sum(), -x[x < 0].sum()
    return float(w / lo) if lo > 0 else float("inf")


def boot_p05(frame: pd.DataFrame, n: int = 600, seed: int = 3) -> float:
    """Day-clustered bootstrap of profit factor. Trades sharing a date are not independent."""
    rng = np.random.default_rng(seed)
    days = frame["signal_date"].unique()
    by = {d: frame.loc[frame["signal_date"] == d, "r"].to_numpy() for d in days}
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        v = pd.Series(np.concatenate([by[days[i]] for i in pick]))
        out.append(pf(v))
    return float(np.nanpercentile(out, 5))


def summarize(t: pd.DataFrame) -> None:
    print("\n=== SHORT VERTICALS HELD TO EXPIRY, HONEST FILLS ===")
    print(f"{'bucket':<18}{'n':>7}{'win':>8}{'avgR':>9}{'PF':>8}{'p05':>8}{'sumR':>9}")
    for q in (1, 5):
        sub = t[t["q"] == q]
        if sub.empty:
            continue
        print(f"{'q'+str(q)+(' SIGNAL' if q == 5 else ' CONTROL'):<18}{len(sub):>7}"
              f"{(sub['r'] > 0).mean():>8.1%}{sub['r'].mean():>9.4f}"
              f"{pf(sub['r']):>8.3f}{boot_p05(sub):>8.3f}{sub['r'].sum():>9.1f}")

    print("\nBY SIDE")
    for (q, side), sub in t.groupby(["q", "side"]):
        print(f"  q{q} {side:<5} n={len(sub):>5}  win {(sub['r'] > 0).mean():.1%}  "
              f"avgR {sub['r'].mean():+.4f}  PF {pf(sub['r']):.3f}")

    print("\nBY MONTH (q5 signal vs q1 control)")
    t = t.assign(mo=pd.to_datetime(t["signal_date"]).dt.to_period("M"))
    for mo, sub in t.groupby("mo"):
        a, b = sub[sub["q"] == 5], sub[sub["q"] == 1]
        if len(a) < 10 or len(b) < 10:
            continue
        print(f"  {mo}  q5 n={len(a):>4} PF {pf(a['r']):>6.3f} avgR {a['r'].mean():+.4f}"
              f"   |   q1 n={len(b):>4} PF {pf(b['r']):>6.3f} avgR {b['r'].mean():+.4f}")


if __name__ == "__main__":
    main()
