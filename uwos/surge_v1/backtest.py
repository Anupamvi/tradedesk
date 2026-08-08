"""Event-driven executable backtest for the move detector.

One forward pass over the sessions. Each day: mark every open position against
real chain-oi quotes, exit on whichever rule fires first, then open the day's new
picks. Entries fill at the ASK, exits at the BID -- no mid, no modelled slippage
addback. The repo has been burned by quoting a PF off an adjusted fill column.

Structure routing is not a choice, it is derived from what each lane proved:
  up lane  directional edge +0.150  -> long calls  (signed right tail)
  dn lane  directional edge +0.006  -> long strangles (magnitude only, never puts)
"""
from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .detector import _fit_predict, purge
from .features import feature_sets

OCC_RE = re.compile(r"^([A-Z0-9\.\-]{1,6})(\d{6})([CP])(\d{8})$")
CHAIN_COLS = ["option_symbol", "last_bid", "last_ask", "dte", "stock_price", "last_oi"]


@dataclass
class Position:
    ticker: str
    lane: str
    structure: str
    entry_date: pd.Timestamp
    fold: str
    legs: list                      # [(option_symbol, qty_sign)]
    entry_debit: float              # per share, already crossed
    max_hold: int
    target_mult: float
    spot_at_entry: float
    held: int = 0
    marks: list = field(default_factory=list)


def parse_occ(sym: pd.Series) -> pd.DataFrame:
    ex = sym.str.extract(OCC_RE)
    ex.columns = ["root", "yymmdd", "cp", "strike8"]
    return pd.DataFrame({
        "root": ex["root"],
        "expiry": pd.to_datetime(ex["yymmdd"], format="%y%m%d", errors="coerce"),
        "is_call": ex["cp"].eq("C"),
        "strike": pd.to_numeric(ex["strike8"], errors="coerce") / 1000.0,
    }, index=sym.index)


def load_chain(base: Path, file_date: pd.Timestamp, tickers: set[str] | None = None):
    d = pd.Timestamp(file_date).strftime("%Y-%m-%d")
    zp = base / d / f"chain-oi-changes-{d}.zip"
    if not zp.exists():
        return None
    try:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                return None
            with z.open(names[0]) as fh:
                df = pd.read_csv(fh, usecols=lambda c: c in set(CHAIN_COLS), low_memory=False)
    except Exception:
        return None
    if "option_symbol" not in df.columns:
        return None
    df["option_symbol"] = df["option_symbol"].astype(str).str.strip().str.upper()
    df = pd.concat([df, parse_occ(df["option_symbol"])], axis=1)
    if tickers is not None:
        df = df[df["root"].isin(tickers)]
    for c in ("last_bid", "last_ask", "dte", "stock_price", "last_oi"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _leg_ok(row, min_oi: int, max_spread_pct: float) -> bool:
    b, a = row["last_bid"], row["last_ask"]
    if not (np.isfinite(b) and np.isfinite(a)) or b <= 0.05 or a <= 0:
        return False
    if (row.get("last_oi") or 0) < min_oi:
        return False
    mid = 0.5 * (a + b)
    return mid > 0 and (a - b) / mid <= max_spread_pct


def build_long_call(chain, spot, sigma, cfg):
    c = chain[chain["is_call"] & chain["dte"].between(cfg["min_dte"], cfg["max_dte"])]
    if c.empty:
        return None
    target = spot * (1.0 + cfg["call_sigma"] * sigma)
    c = c.assign(_d=(c["strike"] - target).abs()).sort_values("_d")
    for _, row in c.head(6).iterrows():
        if _leg_ok(row, cfg["min_oi"], cfg["max_spread_pct"]):
            return [(row["option_symbol"], 1)], float(row["last_ask"])
    return None


def build_long_strangle(chain, spot, sigma, cfg):
    band = chain[chain["dte"].between(cfg["min_dte"], cfg["max_dte"])]
    if band.empty:
        return None
    exp = band["expiry"].value_counts().idxmax()
    band = band[band["expiry"] == exp]
    picks, debit = [], 0.0
    for is_call, mult in ((True, +1.0), (False, -1.0)):
        side = band[band["is_call"] == is_call]
        if side.empty:
            return None
        target = spot * (1.0 + mult * cfg["strangle_sigma"] * sigma)
        side = side.assign(_d=(side["strike"] - target).abs()).sort_values("_d")
        leg = None
        for _, row in side.head(6).iterrows():
            if _leg_ok(row, cfg["min_oi"], cfg["max_spread_pct"]):
                leg = row
                break
        if leg is None:
            return None
        picks.append((leg["option_symbol"], 1))
        debit += float(leg["last_ask"])
    return picks, debit


BUILDERS = {"long_call": build_long_call, "long_strangle": build_long_strangle}

DEFAULT_CFG = {
    "min_dte": 45, "max_dte": 90,       # matched to a 21-session hold, not 5
    "call_sigma": 0.5, "strangle_sigma": 0.75,
    "min_oi": 25, "max_spread_pct": 0.15,
    "max_hold": 25, "target_mult": 1.50,   # +50% take profit, NO stop (measured)
    "k": 10, "min_contract_cost": 1.00, "max_contract_cost": 40.0,
}

LANE_STRUCTURE = {"up": "long_call", "dn": "long_strangle",
                  "move": "long_strangle"}


def run(df: pd.DataFrame, base_dir: str, horizon: int = 21, pct: int = 20,
        lanes=("up", "dn"), feature_set: str = "all", cfg: dict | None = None,
        min_train_months: int = 2, seed: int = 7, verbose: bool = True) -> pd.DataFrame:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    base = Path(base_dir)
    feats = feature_sets(df)[feature_set]
    months = sorted(df["month"].unique())

    # Pre-compute each lane's daily picks per fold, strictly walk-forward.
    picks_by_date: dict[pd.Timestamp, list] = {}
    for i in range(min_train_months, len(months)):
        te_mask = df["month"] == months[i]
        for lane in lanes:
            tgt = f"{lane}_{horizon}_{pct}"
            usable = df[df[tgt].notna()]
            tr = usable[usable["month"].isin(months[:i])]
            te = df[te_mask]
            if len(te) < 200 or tr[tgt].nunique() < 2:
                continue
            _, p = _fit_predict(tr, te, feats, tgt, seed)
            t = te.assign(_s=p)
            top = t.sort_values("_s", ascending=False).groupby("date").head(cfg["k"])
            for _, r in top.iterrows():
                picks_by_date.setdefault(r["date"], []).append(
                    (lane, r["ticker"], str(months[i]), float(r["iv30d"])))
        if verbose:
            print(f"  scored fold {months[i]}", flush=True)

    sessions = np.array(sorted(df["date"].unique()))
    open_pos: list[Position] = []
    closed: list[dict] = []

    for idx, sess in enumerate(sessions):
        # A chain-oi file dated D carries quotes for session D-1.
        chain = load_chain(base, pd.Timestamp(sessions[idx + 1]) if idx + 1 < len(sessions) else None) \
            if idx + 1 < len(sessions) else None
        if chain is None or chain.empty:
            continue
        quotes = chain.set_index("option_symbol")[["last_bid", "last_ask"]]
        bid = quotes["last_bid"].to_dict()

        still: list[Position] = []
        for pos in open_pos:
            pos.held += 1
            val = 0.0
            complete = True
            for sym, qty in pos.legs:
                b = bid.get(sym)
                if b is None or not np.isfinite(b):
                    complete = False
                    break
                val += qty * float(b)
            if complete:
                pos.marks.append(val)
            reason = None
            if complete and pos.entry_debit > 0 and val >= pos.target_mult * pos.entry_debit:
                reason = "target"
            elif pos.held >= pos.max_hold:
                reason = "time"
            if reason:
                exit_val = val if complete else (pos.marks[-1] if pos.marks else 0.0)
                pnl = exit_val - pos.entry_debit
                closed.append({
                    "lane": pos.lane, "structure": pos.structure, "ticker": pos.ticker,
                    "entry_date": pos.entry_date, "exit_date": pd.Timestamp(sess),
                    "fold": pos.fold, "held": pos.held, "exit_reason": reason,
                    "entry_debit": pos.entry_debit, "exit_value": exit_val,
                    "pnl": pnl, "r": pnl / pos.entry_debit,
                    "spot_at_entry": pos.spot_at_entry,
                })
            else:
                still.append(pos)
        open_pos = still

        todays = picks_by_date.get(pd.Timestamp(sess), [])
        if not todays:
            continue
        held_names = {p.ticker for p in open_pos}
        by_root = dict(tuple(chain.groupby("root")))
        for lane, ticker, fold, iv in todays:
            if ticker in held_names:
                continue
            sub = by_root.get(ticker)
            if sub is None or sub.empty:
                continue
            spot = float(sub["stock_price"].median())
            if not np.isfinite(spot) or spot <= 0 or not np.isfinite(iv) or iv <= 0:
                continue
            structure = LANE_STRUCTURE[lane]
            built = BUILDERS[structure](
                sub, spot, iv * math.sqrt(cfg["max_hold"] / 252.0), cfg)
            if built is None:
                continue
            legs, debit = built
            if not (cfg["min_contract_cost"] <= debit <= cfg["max_contract_cost"]):
                continue
            open_pos.append(Position(
                ticker=ticker, lane=lane, structure=structure,
                entry_date=pd.Timestamp(sess), fold=fold, legs=legs,
                entry_debit=debit, max_hold=cfg["max_hold"],
                target_mult=cfg["target_mult"], spot_at_entry=spot))
            held_names.add(ticker)

    return pd.DataFrame(closed)


def _book_step(book_positions: list, book_closed: list, sess, bid, chain_by_root,
               todays, cfg):
    """Advance one book by one session: mark, exit, then open."""
    still = []
    for pos in book_positions:
        pos.held += 1
        val, complete = 0.0, True
        for sym, qty in pos.legs:
            b = bid.get(sym)
            if b is None or not np.isfinite(b):
                complete = False
                break
            val += qty * float(b)
        if complete:
            pos.marks.append(val)
        reason = None
        if complete and pos.entry_debit > 0 and val >= pos.target_mult * pos.entry_debit:
            reason = "target"
        elif pos.held >= pos.max_hold:
            reason = "time"
        if reason:
            exit_val = val if complete else (pos.marks[-1] if pos.marks else 0.0)
            pnl = exit_val - pos.entry_debit
            book_closed.append({
                "lane": pos.lane, "structure": pos.structure, "ticker": pos.ticker,
                "entry_date": pos.entry_date, "exit_date": pd.Timestamp(sess),
                "fold": pos.fold, "held": pos.held, "exit_reason": reason,
                "entry_debit": pos.entry_debit, "exit_value": exit_val,
                "pnl": pnl, "r": pnl / pos.entry_debit,
                "spot_at_entry": pos.spot_at_entry,
            })
        else:
            still.append(pos)
    book_positions[:] = still

    held_names = {p.ticker for p in book_positions}
    for lane, ticker, fold, iv in todays:
        if ticker in held_names:
            continue
        sub = chain_by_root.get(ticker)
        if sub is None or sub.empty:
            continue
        spot = float(sub["stock_price"].median())
        if not np.isfinite(spot) or spot <= 0 or not np.isfinite(iv) or iv <= 0:
            continue
        structure = LANE_STRUCTURE[lane]
        built = BUILDERS[structure](sub, spot, iv * math.sqrt(cfg["max_hold"] / 252.0), cfg)
        if built is None:
            continue
        legs, debit = built
        if not (cfg["min_contract_cost"] <= debit <= cfg["max_contract_cost"]):
            continue
        book_positions.append(Position(
            ticker=ticker, lane=lane, structure=structure,
            entry_date=pd.Timestamp(sess), fold=fold, legs=legs,
            entry_debit=debit, max_hold=cfg["max_hold"],
            target_mult=cfg["target_mult"], spot_at_entry=spot))
        held_names.add(ticker)


def run_with_controls(df: pd.DataFrame, base_dir: str, horizon: int = 21, pct: int = 20,
                      lanes=("up",), feature_set: str = "all", cfg: dict | None = None,
                      n_controls: int = 30, min_train_months: int = 2,
                      seed: int = 7, verbose: bool = True):
    """Signal book and n_controls random books, scored in ONE pass over the chains.

    The controls draw random names from the SAME daily universe the model ranked,
    take the SAME number of picks, and use identical structures and exit rules.
    Only the selection rule differs.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    base = Path(base_dir)
    feats = feature_sets(df)[feature_set]
    months = sorted(df["month"].unique())
    rng = np.random.default_rng(seed)

    picks_by_date: dict[pd.Timestamp, list] = {}
    universe_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for i in range(min_train_months, len(months)):
        te = df[df["month"] == months[i]]
        if len(te) < 200:
            continue
        for lane in lanes:
            tgt = f"{lane}_{horizon}_{pct}"
            usable = df[df[tgt].notna()]
            tr = purge(usable[usable["month"].isin(months[:i])], te, horizon)
            if tr[tgt].nunique() < 2:
                continue
            _, p = _fit_predict(tr, te, feats, tgt, seed)
            t = te.assign(_s=p)
            for _, r in t.sort_values("_s", ascending=False).groupby("date").head(cfg["k"]).iterrows():
                picks_by_date.setdefault(r["date"], []).append(
                    (lane, r["ticker"], str(months[i]), float(r["iv30d"])))
        for d, g in te.groupby("date"):
            universe_by_date[d] = g[["ticker", "iv30d"]].dropna()
        if verbose:
            print(f"  scored fold {months[i]}", flush=True)

    sessions = np.array(sorted(df["date"].unique()))
    books = {"signal": ([], [])}
    for c in range(n_controls):
        books[f"control_{c}"] = ([], [])

    lane0, fold_lookup = lanes[0], {}
    for d, picks in picks_by_date.items():
        fold_lookup[d] = picks[0][2]

    for idx, sess in enumerate(sessions):
        # A signal at the close of sessions[idx-1] is filled at THIS session's
        # close, whose quotes live in the file dated sessions[idx+1].
        if idx == 0 or idx + 1 >= len(sessions):
            continue
        chain = load_chain(base, pd.Timestamp(sessions[idx + 1]))
        if chain is None or chain.empty:
            continue
        bid = chain.set_index("option_symbol")["last_bid"].to_dict()
        by_root = dict(tuple(chain.groupby("root")))
        sess_ts = pd.Timestamp(sess)
        signal_ts = pd.Timestamp(sessions[idx - 1])
        todays = picks_by_date.get(signal_ts, [])

        _book_step(*books["signal"], sess, bid, by_root, todays, cfg)

        uni = universe_by_date.get(signal_ts)
        n_pick = len(todays)
        for c in range(n_controls):
            rand_picks = []
            if uni is not None and n_pick > 0 and len(uni) >= n_pick:
                sel = uni.iloc[rng.choice(len(uni), size=n_pick, replace=False)]
                fold = fold_lookup.get(signal_ts, "")
                rand_picks = [(lane0, r.ticker, fold, float(r.iv30d))
                              for r in sel.itertuples()]
            _book_step(*books[f"control_{c}"], sess, bid, by_root, rand_picks, cfg)

        if verbose and idx % 25 == 0:
            print(f"  {sess_ts.date()}  signal closed {len(books['signal'][1])}", flush=True)

    signal = pd.DataFrame(books["signal"][1])
    controls = {k: pd.DataFrame(v[1]) for k, v in books.items() if k != "signal"}
    return signal, controls
