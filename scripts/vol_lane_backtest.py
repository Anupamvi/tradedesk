"""Non-directional (volatility) lane backtest.

Evidence this is built on:
  - direction predictability: OOS AUC 0.53, 2/5 folds  -> noise
  - |move| predictability  : OOS AUC 0.71, 5/5 folds  -> real

So: take the SAME attention signals the engine already produces, but express them
as direction-free structures (long straddle / long strangle) instead of
directional single legs. Emits a cached outcomes CSV for fast model iteration.

Read-only w.r.t. the pipeline. No order placement.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

from uwos.options_pattern_pipeline_v1 import core

HORIZON = 5
# Long and short vol want OPPOSITE ends of the term structure:
#   long vol  - needs time, dies on the theta cliff -> 30-60 DTE, exit at HORIZON
#   short vol - IS the theta, and 4 legs of bid-ask make mid-life exits
#               unaffordable -> 7-14 DTE, hold to expiry, settle at intrinsic
# Running both at 45 DTE (v2) crippled the short side: it earned 5/43 of the
# theta while paying four spread crossings.
LANES = {
    "long": {
        "min_dte": 30, "max_dte": 60, "target_dte": 45,
        "kinds": ("long_straddle", "long_strangle"), "exit": "horizon",
    },
    "short": {
        "min_dte": 7, "max_dte": 14, "target_dte": 9,
        "kinds": ("iron_butterfly", "iron_condor"), "exit": "expiry",
    },
}
MIN_VOLUME = 25
MIN_OI = 10
MAX_LEG_SPREAD_PCT = 0.20  # loose here; the gate is applied at analysis time
WING_PCT = 0.10   # iron butterfly wing distance, as fraction of spot
OTM_PCT = 0.05    # strangle / condor short-strike distance, as fraction of spot


def quote_key(q):
    return (q["ticker"], q["expiry"], q["option_type"], round(float(q["strike"]), 4))


def usable(q, lane):
    if not q.get("ask") or not q.get("bid"):
        return False
    if q["ask"] <= 0 or q["bid"] <= 0:
        return False
    dte = q.get("dte")
    if dte is None or dte < lane["min_dte"] or dte > lane["max_dte"]:
        return False
    if (q.get("volume") or 0) < MIN_VOLUME or (q.get("open_interest") or 0) < MIN_OI:
        return False
    sp = q.get("spread_pct")
    if sp is not None and sp > MAX_LEG_SPREAD_PCT:
        return False
    return True


def _nearest(seq, target):
    best = None
    for q in seq:
        d = abs(float(q["strike"]) - target)
        if best is None or d < best[0]:
            best = (d, q)
    return best[1] if best else None


def _nearest(seq, target):
    best = None
    for q in seq:
        d = abs(float(q["strike"]) - target)
        if best is None or d < best[0]:
            best = (d, q)
    return best[1] if best else None


def _assemble(kind, legs, ticker, expiry, dte, stock, wing_width, be_move_pct):
    """legs = [(quote, side)] with side +1 = buy, -1 = sell.

    entry_net is in points: positive = net debit paid, negative = net credit taken.
    Uses ask to buy and bid to sell (no mid-price fantasy).
    """
    entry_net = 0.0
    spread = 0.0
    vols = []
    ois = []
    for q, side in legs:
        bid = float(q["bid"])
        ask = float(q["ask"])
        entry_net += ask if side > 0 else -bid
        spread += ask - bid
        vols.append(q.get("volume") or 0)
        ois.append(q.get("open_interest") or 0)

    is_credit = wing_width is not None
    if is_credit:
        credit = -entry_net
        if credit <= 0:
            return None
        max_risk_points = wing_width - credit
        # a credit worth less than 8% of the width is not worth the assignment risk
        if max_risk_points <= 0 or credit / wing_width < 0.08:
            return None
    else:
        if entry_net <= 0:
            return None
        max_risk_points = entry_net

    return {
        "kind": kind,
        "ticker": ticker,
        "expiry": expiry,
        "dte": dte,
        "stock": stock,
        "n_legs": len(legs),
        "is_credit": int(is_credit),
        "entry_net": entry_net,
        "wing_width": wing_width if wing_width is not None else "",
        "max_risk_points": max_risk_points,
        "combined_spread": spread,
        "combined_spread_pct": spread / max_risk_points,
        "spread_to_premium": spread / abs(entry_net),
        "min_volume": min(vols),
        "min_oi": min(ois),
        "strikes": "/".join(f"{float(q['strike']):g}{'B' if s > 0 else 'S'}" for q, s in legs),
        "breakeven_move_pct": be_move_pct,
        "_legs": [(quote_key(q), s) for q, s in legs],
    }


def build_vol_structures(option_quotes, tickers, lane):
    """Per ticker, the best structures for this lane near its target DTE."""
    kinds = lane["kinds"]
    by_te = defaultdict(list)
    for q in option_quotes.values():
        if q.get("ticker") not in tickers or not usable(q, lane):
            continue
        by_te[(q["ticker"], q["expiry"])].append(q)

    best_per_ticker = {}
    for (ticker, expiry), quotes in by_te.items():
        stock = core.first_positive(q.get("stock_close") for q in quotes)
        if not stock:
            continue
        calls = sorted((q for q in quotes if q["option_type"] == "call"), key=lambda q: q["strike"])
        puts = sorted((q for q in quotes if q["option_type"] == "put"), key=lambda q: q["strike"])
        if len(calls) < 2 or len(puts) < 2:
            continue
        dte = quotes[0].get("dte")
        put_by_strike = {round(float(p["strike"]), 4): p for p in puts}

        # --- ATM strike quoted on both sides, closest to spot
        atm = None
        for c in calls:
            k = round(float(c["strike"]), 4)
            p = put_by_strike.get(k)
            if p is None:
                continue
            dist = abs(k - stock) / stock
            if dist > 0.03:
                continue
            if atm is None or dist < atm[0]:
                atm = (dist, k, c, p)

        candidates = []

        # --- long straddle: BE move = debit / spot
        if atm and "long_straddle" in kinds:
            _, k, c, p = atm
            debit = float(c["ask"]) + float(p["ask"])
            candidates.append(_assemble(
                "long_straddle", [(c, 1), (p, 1)], ticker, expiry, dte, stock,
                None, debit / stock,
            ))

        # --- long strangle: BE move = (OTM distance + debit) / spot, easier side
        otm_c = _nearest([q for q in calls if q["strike"] > stock], stock * (1 + OTM_PCT))
        otm_p = _nearest([q for q in puts if q["strike"] < stock], stock * (1 - OTM_PCT))
        if otm_c and otm_p and "long_strangle" in kinds:
            debit = float(otm_c["ask"]) + float(otm_p["ask"])
            up = (float(otm_c["strike"]) + debit - stock) / stock
            dn = (stock - (float(otm_p["strike"]) - debit)) / stock
            candidates.append(_assemble(
                "long_strangle", [(otm_c, 1), (otm_p, 1)], ticker, expiry, dte, stock,
                None, min(up, dn),
            ))

        # --- iron butterfly: sell ATM straddle, buy wings ~WING_PCT away
        if atm and "iron_butterfly" in kinds:
            _, k, sc, sp = atm
            lc = _nearest([q for q in calls if q["strike"] > k], k + stock * WING_PCT)
            lp = _nearest([q for q in puts if q["strike"] < k], k - stock * WING_PCT)
            if lc and lp:
                w = max(float(lc["strike"]) - k, k - float(lp["strike"]))
                credit = float(sc["bid"]) + float(sp["bid"]) - float(lc["ask"]) - float(lp["ask"])
                candidates.append(_assemble(
                    "iron_butterfly", [(sc, -1), (sp, -1), (lc, 1), (lp, 1)],
                    ticker, expiry, dte, stock, w,
                    max(credit, 0.0) / stock,
                ))

        # --- iron condor: sell ~OTM_PCT strangle, buy wings another OTM_PCT out
        if otm_c and otm_p and "iron_condor" in kinds:
            lc = _nearest([q for q in calls if q["strike"] > otm_c["strike"]],
                          float(otm_c["strike"]) + stock * OTM_PCT)
            lp = _nearest([q for q in puts if q["strike"] < otm_p["strike"]],
                          float(otm_p["strike"]) - stock * OTM_PCT)
            if lc and lp:
                w = max(float(lc["strike"]) - float(otm_c["strike"]),
                        float(otm_p["strike"]) - float(lp["strike"]))
                credit = (float(otm_c["bid"]) + float(otm_p["bid"])
                          - float(lc["ask"]) - float(lp["ask"]))
                up = (float(otm_c["strike"]) + max(credit, 0.0) - stock) / stock
                dn = (stock - (float(otm_p["strike"]) - max(credit, 0.0))) / stock
                candidates.append(_assemble(
                    "iron_condor", [(otm_c, -1), (otm_p, -1), (lc, 1), (lp, 1)],
                    ticker, expiry, dte, stock, w, min(up, dn),
                ))

        for row in candidates:
            if row is None:
                continue
            prev = best_per_ticker.get((ticker, row["kind"]))
            tgt = lane["target_dte"]
            rank = (abs(row["dte"] - tgt), row["combined_spread_pct"])
            if prev is None or rank < (abs(prev["dte"] - tgt), prev["combined_spread_pct"]):
                best_per_ticker[(ticker, row["kind"])] = row
    return list(best_per_ticker.values())


def score_forward(row, future_snap, risk_config):
    """Close the whole structure at the forward snapshot, crossing the spread again."""
    fq = {quote_key(q): q for q in future_snap.option_quotes.values()}

    exit_net = 0.0   # points received on close (positive = credit received)
    exit_spread = 0.0
    for key, side in row["_legs"]:
        q = fq.get(key)
        if not q:
            return None, "missing_forward_quote"
        bid = float(q.get("bid") or 0.0)
        ask = float(q.get("ask") or 0.0)
        if ask <= 0:
            return None, "bad_forward_quote"
        # unwind: legs we bought are sold at bid, legs we sold are bought at ask
        exit_net += bid if side > 0 else -ask
        exit_spread += ask - bid

    slip_pct = core.configured_slippage_pct_of_spread(risk_config)
    per_leg_fee = core.configured_round_trip_spread_fees(risk_config) / 2.0
    fees = per_leg_fee * row["n_legs"]
    entry_slip = row["combined_spread"] * 100.0 * slip_pct
    exit_slip = exit_spread * 100.0 * slip_pct

    max_risk = row["max_risk_points"] * 100.0 + fees + entry_slip
    if max_risk <= 0:
        return None, "bad_max_risk"

    # works for debit and credit structures alike
    pnl = (exit_net - row["entry_net"]) * 100.0 - fees - entry_slip - exit_slip
    return {
        "exit_net": exit_net,
        "entry_cost": row["entry_net"] * 100.0,
        "round_trip_fees": fees,
        "entry_slippage": entry_slip,
        "exit_slippage": exit_slip,
        "max_risk": max_risk,
        "net_r": pnl / max_risk,
        "win": 1 if pnl > 0 else 0,
    }, "scored"


def settle_at_expiry(row, stock_final, risk_config):
    """Hold defined-risk credit structures to expiry and settle at intrinsic.

    Letting legs expire avoids a second round of bid-ask crossing entirely,
    which is the whole reason short vol is tradeable at 4 legs. Fees are still
    charged both ways to cover assignment on ITM shorts.
    """
    value = 0.0  # position value at expiry, in points
    for key, side in row["_legs"]:
        strike = key[3]
        if key[2] == "call":
            intrinsic = max(0.0, stock_final - strike)
        else:
            intrinsic = max(0.0, strike - stock_final)
        value += side * intrinsic

    slip_pct = core.configured_slippage_pct_of_spread(risk_config)
    per_leg_fee = core.configured_round_trip_spread_fees(risk_config) / 2.0
    fees = per_leg_fee * row["n_legs"]
    entry_slip = row["combined_spread"] * 100.0 * slip_pct

    max_risk = row["max_risk_points"] * 100.0 + fees + entry_slip
    if max_risk <= 0:
        return None, "bad_max_risk"

    pnl = (value - row["entry_net"]) * 100.0 - fees - entry_slip
    return {
        "exit_net": value,
        "entry_cost": row["entry_net"] * 100.0,
        "round_trip_fees": fees,
        "entry_slippage": entry_slip,
        "exit_slippage": 0.0,
        "max_risk": max_risk,
        "net_r": pnl / max_risk,
        "win": 1 if pnl > 0 else 0,
    }, "scored"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--as-of", default="2026-07-24")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/vol_lane_outcomes.csv")
    ap.add_argument("--lane", choices=sorted(LANES), default="short")
    ap.add_argument("--top-candidates-per-day", type=int, default=40)
    ap.add_argument("--min-month-dates", type=int, default=10)
    ap.add_argument("--max-dates", type=int, default=0, help="Smoke test: keep only the last N dates.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    as_of = args.as_of
    source_dates = core.source_complete_dates(base_dir)
    usable_dates = [d for d in source_dates if d <= as_of]
    if args.max_dates:
        usable_dates = usable_dates[-args.max_dates:]
    print(f"[vol] dates {len(usable_dates)} ({usable_dates[0]}..{usable_dates[-1]})", flush=True)

    cache_dir = base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    ns = core.parse_args([
        "--base-dir", str(base_dir), "--as-of", as_of,
        "--validation-top-candidates-per-day", str(args.top_candidates_per_day),
        "--missed-mover-audit-days", "0",
    ])
    config = core.base_run_config(ns, base_dir, as_of, cache_dir)
    risk_config = config["risk_config"]

    t0 = time.perf_counter()
    snapshots = {}
    for i, d in enumerate(usable_dates, 1):
        if i == 1 or i % 20 == 0 or i == len(usable_dates):
            print(f"[vol] snapshot {i}/{len(usable_dates)} {d} ({time.perf_counter()-t0:.0f}s)", flush=True)
        snapshots[d] = core.build_daily_snapshot(base_dir, d, config)
    print(f"[vol] snapshots built {time.perf_counter()-t0:.0f}s", flush=True)

    lane = LANES[args.lane]
    # stock close per date, for expiry settlement of the short lane
    price_map = {}
    for d, snap in snapshots.items():
        px = {}
        for q in snap.option_quotes.values():
            tk = q.get("ticker")
            if tk and tk not in px and q.get("stock_close"):
                px[tk] = float(q["stock_close"])
        price_map[d] = px

    def settle_date_for(expiry):
        """Last snapshot date on or before expiry, and it must be close to it."""
        prior = [x for x in usable_dates if x <= expiry]
        if not prior:
            return None
        cand = prior[-1]
        # expiry must be inside the data window, not past the end of it
        if cand == usable_dates[-1] and expiry > usable_dates[-1]:
            return None
        return cand

    # learn one pattern config per month from strictly-prior dates (point-in-time)
    fields = [
        "signal_date", "target_date", "lane", "kind", "ticker", "expiry", "dte", "stock",
        "n_legs", "is_credit", "strikes", "entry_net", "wing_width", "max_risk_points",
        "combined_spread", "combined_spread_pct", "spread_to_premium",
        "min_volume", "min_oi", "breakeven_move_pct", "market_regime", "sector",
        "pattern_family", "n_signals_for_ticker", "status", "unscorable_reason",
        "net_r", "win", "entry_cost", "max_risk", "round_trip_fees",
        "entry_slippage", "exit_slippage", "exit_net", "realized_abs_move",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        prior_cfg_month = None
        pattern_config = None
        for di, d in enumerate(usable_dates):
            target_date = core.nth_future_date(usable_dates, d, HORIZON)
            if not target_date:
                continue
            month = d[:7]
            if month != prior_cfg_month:
                train = [snapshots[x] for x in usable_dates if x[:7] < month]
                if len(train) < 10:
                    prior_cfg_month = month
                    pattern_config = core.learn_pattern_config(
                        [snapshots[x] for x in usable_dates if x < d][-20:]
                    ) if len([x for x in usable_dates if x < d]) >= 5 else None
                    continue
                pattern_config = core.learn_pattern_config(train)
                prior_cfg_month = month
            if pattern_config is None:
                continue

            signals = core.generate_signals_for_snapshot(
                snapshots[d], pattern_config, args.top_candidates_per_day,
                risk_config=risk_config,
            )
            if not signals:
                continue
            per_ticker = defaultdict(list)
            for s in signals:
                per_ticker[s["ticker"]].append(s)
            tickers = set(per_ticker)

            rows = build_vol_structures(snapshots[d].option_quotes, tickers, lane)
            fsnap = snapshots[target_date]
            for r in rows:
                sig = per_ticker[r["ticker"]][0]
                if lane["exit"] == "expiry":
                    sd = settle_date_for(r["expiry"])
                    s_final = price_map.get(sd, {}).get(r["ticker"]) if sd else None
                    if s_final is None:
                        res, note = None, "no_expiry_price"
                    else:
                        res, note = settle_at_expiry(r, s_final, risk_config)
                    exit_date = sd or ""
                    ref_price = s_final
                else:
                    res, note = score_forward(r, fsnap, risk_config)
                    exit_date = target_date
                    ref_price = price_map.get(target_date, {}).get(r["ticker"])
                rec = dict(r)
                rec.update({
                    "signal_date": d,
                    "target_date": exit_date,
                    "lane": args.lane,
                    "market_regime": str(snapshots[d].market_regime.get("regime") or ""),
                    "sector": sig.get("sector", ""),
                    "pattern_family": sig.get("pattern_family", ""),
                    "n_signals_for_ticker": len(per_ticker[r["ticker"]]),
                    "realized_abs_move": (abs(ref_price - r["stock"]) / r["stock"]) if ref_price else "",
                    "status": "SCORED" if res else "UNSCORABLE",
                    "unscorable_reason": "" if res else note,
                })
                if res:
                    rec.update(res)
                w.writerow(rec)
                written += 1
            if di % 10 == 0:
                fh.flush()
                print(f"[vol] {d} signals={len(signals)} structs={len(rows)} total={written} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"[vol] DONE rows={written} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
