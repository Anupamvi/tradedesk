"""Short defined-risk premium on the names the detector predicts will stay QUIET.

The arithmetic that motivates this, measured in this repo:
  random long strangle   PF 0.62   -> buying premium starts 38% in the hole
  random short vertical  PF 1.24   -> 2 crossings instead of 4, positive carry
  detector selection edge          -> worth roughly +0.35-0.43 of PF
Applied to long premium the edge only reaches breakeven. Applied to the SHORT
side it starts from 1.24 instead of 0.62, and the detector is used inverted:
sell where it predicts the SMALLEST chance of a large move.

Structures are held to expiry (no exit crossing) and settle at intrinsic.
Fills are honest: the short leg sells at the BID, the long leg buys at the ASK.

An 86%-win-rate short book has a tail that six months has probably not shown, so
this is judged on CVaR and blowup rate, never on profit factor alone.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import load_chain
from .detector import _fit_predict, purge
from .features import feature_sets


def _nearest(side: pd.DataFrame, target: float):
    if side.empty:
        return None
    d = (side["strike"] - target).abs()
    if not np.isfinite(d.min()):
        return None
    return side.loc[d.idxmin()]


def _leg_ok(row, min_oi: int, max_spread_pct: float) -> bool:
    b, a = row["last_bid"], row["last_ask"]
    if not (np.isfinite(b) and np.isfinite(a)) or b <= 0.01 or a <= 0:
        return False
    if (row.get("last_oi") or 0) < min_oi:
        return False
    mid = 0.5 * (a + b)
    return mid > 0 and (a - b) / mid <= max_spread_pct


def build_vertical(chain: pd.DataFrame, spot: float, sigma: float, is_call: bool,
                   cfg: dict, rej: dict | None = None):
    def no(reason):
        if rej is not None:
            rej[reason] = rej.get(reason, 0) + 1
        return None

    sign = 1.0 if is_call else -1.0
    side = chain[chain["is_call"] == is_call]
    if side.empty:
        return no("no_side")
    s = _nearest(side, spot * (1.0 + sign * cfg["short_sigma"] * sigma))
    l = _nearest(side, spot * (1.0 + sign * cfg["long_sigma"] * sigma))
    if s is None or l is None or s["strike"] == l["strike"]:
        return no("no_strikes")
    if is_call and not (l["strike"] > s["strike"] > spot):
        return no("strike_order")
    if (not is_call) and not (l["strike"] < s["strike"] < spot):
        return no("strike_order")
    if not _leg_ok(s, cfg["min_oi"], cfg["max_spread_pct"]):
        return no("short_leg_quality")
    if not _leg_ok(l, cfg["min_oi"], cfg["max_spread_pct"]):
        return no("long_leg_quality")
    credit = float(s["last_bid"]) - float(l["last_ask"])
    width = abs(float(l["strike"]) - float(s["strike"]))
    if credit <= 0:
        return no("credit_nonpositive")
    if width - credit <= 0.01:
        return no("no_risk")
    if credit / width < cfg["min_credit_pct"]:
        return no("credit_too_thin")
    if credit / width > cfg["max_credit_pct"]:
        return no("credit_too_rich")
    return {"short_strike": float(s["strike"]), "long_strike": float(l["strike"]),
            "credit": credit, "width": width, "max_risk": width - credit,
            "expiry": s["expiry"], "is_call": is_call}


def settle(pos: dict, expiry_spot: float) -> float:
    if pos["is_call"]:
        loss = max(0.0, min(expiry_spot, pos["long_strike"]) - pos["short_strike"])
    else:
        loss = max(0.0, pos["short_strike"] - max(expiry_spot, pos["long_strike"]))
    return (pos["credit"] - loss) / pos["max_risk"]


DEFAULT_CFG = {
    "min_dte": 25, "max_dte": 50,
    "short_sigma": 1.0, "long_sigma": 1.9,
    "min_oi": 25, "max_spread_pct": 0.25,
    "min_credit_pct": 0.05, "max_credit_pct": 0.60,
    "k": 12, "both_sides": True,
}


def run(df: pd.DataFrame, base_dir: str, horizon: int = 21, pct: int = 20,
        feature_set: str = "everything", cfg: dict | None = None,
        n_controls: int = 30, min_train_months: int = 2, seed: int = 7,
        sets_fn=None, verbose: bool = True):
    """Signal book (quietest-ranked names) plus n_controls random books, one pass."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    base = Path(base_dir)
    feats = (sets_fn or feature_sets)(df)[feature_set]
    target = f"move_{horizon}_{pct}"
    months = sorted(df["month"].unique())
    rng = np.random.default_rng(seed)

    picks: dict[pd.Timestamp, list] = {}
    universe: dict[pd.Timestamp, pd.DataFrame] = {}
    folds: dict[pd.Timestamp, str] = {}
    for i in range(min_train_months, len(months)):
        te = df[df["month"] == months[i]]
        usable = df[df[target].notna()]
        tr = purge(usable[usable["month"].isin(months[:i])], te, horizon)
        if len(te) < 200 or tr[target].nunique() < 2:
            continue
        _, p = _fit_predict(tr, te, feats, target, seed)
        # INVERTED: ascending, so the lowest predicted move probability ranks first.
        t = te.assign(_s=p)
        for _, r in t.sort_values("_s").groupby("date").head(cfg["k"]).iterrows():
            picks.setdefault(r["date"], []).append((r["ticker"], float(r["iv30d"])))
        for d, g in te.groupby("date"):
            universe[d] = g[["ticker", "iv30d"]].dropna()
            folds[d] = str(months[i])
        if verbose:
            print(f"  scored fold {months[i]}", flush=True)

    sessions = np.array(sorted(df["date"].unique()))
    close_by = df.set_index(["ticker", "date"])["close"].to_dict()
    books: dict[str, list] = {"signal": []}
    for c in range(n_controls):
        books[f"control_{c}"] = []
    rej: dict[str, int] = {}

    for idx in range(1, len(sessions) - 1):
        sess = pd.Timestamp(sessions[idx])
        signal_ts = pd.Timestamp(sessions[idx - 1])
        todays = picks.get(signal_ts, [])
        if not todays:
            continue
        chain = load_chain(base, pd.Timestamp(sessions[idx + 1]))
        if chain is None or chain.empty:
            continue
        chain = chain[chain["dte"].between(cfg["min_dte"], cfg["max_dte"])]
        if chain.empty:
            continue
        by_root = dict(tuple(chain.groupby("root")))
        fold = folds.get(signal_ts, "")

        def open_book(names, out, track=False):
            r = rej if track else None
            for ticker, iv in names:
                sub = by_root.get(ticker)
                if sub is None or sub.empty:
                    if track:
                        rej["no_chain"] = rej.get("no_chain", 0) + 1
                    continue
                if not np.isfinite(iv) or iv <= 0:
                    continue
                spot = float(sub["stock_price"].median())
                if not np.isfinite(spot) or spot <= 0:
                    continue
                exp = sub["expiry"].value_counts().idxmax()
                leg_chain = sub[sub["expiry"] == exp]
                dte = float(leg_chain["dte"].median())
                sett = close_by.get((ticker, pd.Timestamp(exp)))
                if sett is None or not np.isfinite(sett):
                    if track:
                        rej["no_settlement_price"] = rej.get("no_settlement_price", 0) + 1
                    continue
                sigma = iv * math.sqrt(max(dte, 1) / 365.0)
                for is_call in ((True, False) if cfg["both_sides"] else (False,)):
                    pos = build_vertical(leg_chain, spot, sigma, is_call, cfg, r)
                    if pos is None:
                        continue
                    out.append({
                        "ticker": ticker, "entry_date": sess, "fold": fold,
                        "side": "call" if is_call else "put", "dte": dte,
                        "spot": spot, "settle": float(sett),
                        "credit": pos["credit"], "max_risk": pos["max_risk"],
                        "short_strike": pos["short_strike"],
                        "long_strike": pos["long_strike"],
                        "r": settle(pos, float(sett)),
                    })

        open_book(todays, books["signal"], track=True)
        uni = universe.get(signal_ts)
        if uni is not None and len(uni) >= len(todays):
            for c in range(n_controls):
                sel = uni.iloc[rng.choice(len(uni), size=len(todays), replace=False)]
                open_book([(r.ticker, float(r.iv30d)) for r in sel.itertuples()],
                          books[f"control_{c}"])
        if verbose and idx % 25 == 0:
            print(f"  {sess.date()}  signal trades {len(books['signal'])}", flush=True)

    signal = pd.DataFrame(books["signal"])
    controls = {k: pd.DataFrame(v) for k, v in books.items() if k != "signal"}
    return signal, controls, rej
