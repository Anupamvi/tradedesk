"""Re-score the existing validation signals at multiple hold horizons.

Read-only diagnostic: does NOT touch pipeline gates or write pipeline artifacts.
Answers "is the fixed 5d exit throwing away the edge?"
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from uwos.options_pattern_pipeline_v1 import core

HORIZONS = (1, 3, 5, 10, 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--as-of", default="2026-07-24")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/horizon_sweep.csv")
    ap.add_argument("--top-candidates-per-day", type=int, default=40)
    ap.add_argument("--min-month-dates", type=int, default=10)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    as_of = args.as_of
    source_dates = core.source_complete_dates(base_dir)
    usable_dates = [d for d in source_dates if d <= as_of]
    print(f"[hz] usable dates: {len(usable_dates)} ({usable_dates[0]}..{usable_dates[-1]})", flush=True)

    cache_dir = base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    ns = core.parse_args(
        [
            "--base-dir", str(base_dir),
            "--as-of", as_of,
            "--validation-top-candidates-per-day", str(args.top_candidates_per_day),
            "--missed-mover-audit-days", "0",
        ]
    )
    config = core.base_run_config(ns, base_dir, as_of, cache_dir)
    risk_config = config["risk_config"]

    t0 = time.perf_counter()
    snapshots = {}
    for i, d in enumerate(usable_dates, 1):
        if i == 1 or i % 20 == 0 or i == len(usable_dates):
            print(f"[hz] snapshot {i}/{len(usable_dates)}: {d}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        snapshots[d] = core.build_daily_snapshot(base_dir, d, config)
    print(f"[hz] snapshots built in {time.perf_counter()-t0:.0f}s", flush=True)

    splits = core.build_validation_splits(usable_dates, args.min_month_dates)
    print(f"[hz] splits: {len(splits)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "horizon", "signal_date", "target_date", "ticker", "direction",
        "pattern_family", "strategy_kind", "contract_profile", "dte",
        "bid_ask_spread_pct", "status", "net_r", "win", "stock_proxy_move",
        "outcome_note", "blocked", "block_reasons",
    ]
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for si, split in enumerate(splits, 1):
            vsnaps = [snapshots[d] for d in split["validation_dates"]]
            pattern_config = core.learn_pattern_config([snapshots[d] for d in split["train_dates"]])
            vsignals = []
            for snap in vsnaps:
                vsignals.extend(
                    core.generate_signals_for_snapshot(
                        snap,
                        pattern_config,
                        args.top_candidates_per_day,
                        source_rescue_max_extra=core.VALIDATION_SOURCE_RESCUE_MAX_EXTRA_SIGNALS,
                        tradeable_gap_max_extra=core.VALIDATION_TRADEABLE_GAP_MAX_EXTRA_SIGNALS,
                        risk_config=risk_config,
                    )
                )
            rows = core.score_signals(
                vsignals, snapshots, usable_dates, split["name"], "VALIDATION",
                risk_config, horizons=HORIZONS,
            )
            for r in rows:
                writer.writerow(r)
            written += len(rows)
            fh.flush()
            print(
                f"[hz] split {si}/{len(splits)} {split['name']}: "
                f"signals={len(vsignals)} rows={len(rows)} total={written} "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )

    print(f"[hz] DONE rows={written} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
