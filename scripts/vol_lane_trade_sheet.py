"""Build the next-session long-vol trade sheet in the structure that was validated.

The pipeline's decision board can only express directional single-leg debits and
credit spreads. The edge that actually validated is direction-free: long
straddles and strangles, 30-60 DTE, held five sessions. This emits that trade
sheet directly from the same snapshot and the same structure builder the
backtest used, so what is printed is what was measured.

Filters applied, in the order the walk-forward applied them:
  1. volatility gate      iv_rank >= 50 and vrp_ratio > 1.0
  2. earnings window      next print 10-45 calendar days out
  3. structure            5% OTM strangle, 30-60 DTE
  4. breakeven cap        needs less than a 20% move to pay
  5. daily ticket cap     best 8 by score

Why a strangle rather than a straddle: at the same breakeven (12.8% vs 12.7%)
and the same hit rate (41.3% vs 40.5%), the strangle's average win is +0.333R
against the straddle's +0.196R. It is the same bet carrying more convexity, and
this edge lives in the tail, so it beat the straddle in every month with enough
sample.

Read-only. No order placement.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uwos.options_pattern_pipeline_v1 import core
from vol_lane_backtest import LANES, build_vol_structures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--earnings-min", type=float, default=10.0)
    ap.add_argument("--earnings-max", type=float, default=45.0)
    ap.add_argument("--max-breakeven-pct", type=float, default=20.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    as_of = args.as_of

    ns = core.parse_args([
        "--base-dir", str(base_dir), "--as-of", as_of,
        "--missed-mover-audit-days", "0",
    ])
    cache_dir = base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    config = core.base_run_config(ns, base_dir, as_of, cache_dir)
    snap = core.build_daily_snapshot(base_dir, as_of, config)

    gated = []
    for ticker, f in snap.features.items():
        if ticker.startswith("^"):
            continue
        rank = core.num(f.get("iv_rank"))
        vrp = core.num(f.get("vrp_ratio"))
        if rank is None or vrp is None or rank < 50.0 or vrp <= 1.0:
            continue
        dte_e = core.num(f.get("earnings_dte"))
        in_window = dte_e is not None and args.earnings_min <= dte_e <= args.earnings_max
        gated.append((ticker, f, dte_e, in_window))

    eligible = [g for g in gated if g[3]]
    print(f"[sheet] {as_of}: vol gate {len(gated)} names -> earnings window {len(eligible)}")
    if not eligible:
        print("[sheet] nothing tradeable. If the gate is empty the feed's realized-vol "
              "column is probably corrupt for this date (it breaks most Fridays).")
        return 1

    tickers = {t for t, _, _, _ in eligible}
    rows = build_vol_structures(snap.option_quotes, tickers, LANES["long"])
    by_ticker = {}
    for r in rows:
        if r["kind"] != "long_strangle":
            continue
        if float(r["breakeven_move_pct"]) * 100.0 >= args.max_breakeven_pct:
            continue
        cur = by_ticker.get(r["ticker"])
        # one ticket per name: the expiry whose breakeven is cheapest still
        # inside the band that was measured
        if cur is None or r["breakeven_move_pct"] < cur["breakeven_move_pct"]:
            by_ticker[r["ticker"]] = r

    feat = {t: (f, d) for t, f, d, _ in eligible}
    out = []
    for ticker, r in by_ticker.items():
        f, dte_e = feat[ticker]
        vega = core.num(f.get("tape_vega_flow"))
        multileg = core.num(f.get("hc_multileg_share"))
        score = (
            core.num(f.get("iv_rank")) / 100.0
            + (core.num(f.get("vrp_ratio")) - 1.0) * 2.0
            + (0.25 if vega is not None and vega > 0 else 0.0)
            - (0.25 if multileg is not None and multileg > 0.30 else 0.0)
        )
        out.append({
            "ticker": ticker,
            "structure": r["kind"],
            "expiry": r["expiry"],
            "dte": r["dte"],
            "strikes": r["strikes"],
            "debit_per_unit": round(float(r["entry_net"]), 2),
            "max_risk": round(float(r["max_risk_points"]) * 100.0, 2),
            "breakeven_move_pct": round(float(r["breakeven_move_pct"]) * 100.0, 2),
            "combined_spread_pct": round(float(r["combined_spread_pct"]) * 100.0, 1),
            "iv_rank": round(core.num(f.get("iv_rank")), 1),
            "vrp_ratio": round(core.num(f.get("vrp_ratio")), 3),
            "earnings_dte": int(dte_e),
            "next_earnings_date": f.get("next_earnings_date") or "",
            "dealer_vol_position": f.get("dealer_vol_position") or "unknown",
            "sector": f.get("sector") or "",
            "score": round(score, 4),
        })

    out.sort(key=lambda r: r["score"], reverse=True)
    kept = out[: args.cap]

    print(f"[sheet] structures built for {len(out)} names, taking best {len(kept)}\n")
    hdr = ["ticker", "structure", "expiry", "dte", "strikes", "debit_per_unit",
           "max_risk", "breakeven_move_pct", "iv_rank", "vrp_ratio", "earnings_dte",
           "dealer_vol_position"]
    print("  ".join(h.rjust(9)[:9] for h in hdr))
    for r in kept:
        print("  ".join(str(r[h]).rjust(9)[:9] for h in hdr))

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
            w.writeheader()
            w.writerows(kept)
        print(f"\n[sheet] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
