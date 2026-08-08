"""Confirmatory test of a frozen hypothesis set on data never used for selection.

`validate_strategy_universe.py` sweeps 446 hypotheses. Holm across that family demands
p < 0.000112, which nothing reached, so that sweep can only ever be *exploratory*. This
script tests a small frozen set on a held-out window, where the multiplicity burden
matches the number of claims actually being made.

Integrity rules enforced here, because the whole point is that they cannot be bent:
  * the hypothesis set is a module constant and its hash is written into the output
  * the holdout window must start strictly after the exploration sample ended
  * the Holm family size is len(PREREGISTERED), never the number that happened to pass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_strategy_universe import (  # noqa: E402
    MIN_CLUSTERED_P05,
    MAX_PERMUTATION_P,
    _assert_test_has_power,
    _clustered_p05,
    _compile_matched_pools,
    _draw_profit_factors,
    _metrics,
    _scope_mask,
    holm_adjust,
)

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")

# Frozen before the holdout is scored. Chosen from exploration on 2026-01-15..2026-06-04
# plus an economic prior: short premium is the only family whose payoff shape survived
# cost stress, and the credit vertical is the defined-risk expression of the same idea.
PREREGISTERED: tuple[dict[str, str], ...] = (
    {"scope": "sector", "scope_value": "Technology", "strategy": "cash_secured_put"},
    {"scope": "sector_state", "scope_value": "mixed", "strategy": "cash_secured_put"},
    {"scope": "all_sectors", "scope_value": "all", "strategy": "bull_put_credit_vertical"},
)

# Every signal date at or before this was visible during hypothesis selection.
EXPLORATION_THROUGH = "2026-06-04"

DIRECTION = "signal profit factor exceeds matched control and exceeds 1.0"


def preregistration_hash() -> str:
    payload = json.dumps(
        {"hypotheses": PREREGISTERED, "direction": DIRECTION, "through": EXPLORATION_THROUGH},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def confirm(
    universe: pd.DataFrame,
    *,
    holdout_start: str,
    permutations: int = 20_000,
    bootstrap_trials: int = 4_000,
    seed: int = 20260804,
) -> pd.DataFrame:
    if holdout_start <= EXPLORATION_THROUGH:
        raise SystemExit(
            f"holdout starts {holdout_start} but exploration ran through {EXPLORATION_THROUGH}. "
            "Confirming on data used to pick the hypotheses is p-hacking."
        )
    holdout = universe[universe["signal_date"].astype(str) >= holdout_start].copy()
    if holdout.empty:
        raise SystemExit(f"no rows on or after {holdout_start}; nothing to confirm against")

    rng = np.random.default_rng(seed)
    family_size = len(PREREGISTERED)
    _assert_test_has_power(family_size, permutations)

    records: list[dict[str, Any]] = []
    for definition in PREREGISTERED:
        scoped = holdout[_scope_mask(holdout, definition) & holdout["strategy"].eq(definition["strategy"])]
        # The permutation helper buckets draws by `sample`; the whole holdout is out-of-sample.
        scoped = scoped.assign(sample="TEST")
        signal = scoped[scoped["signal_selected"].astype(bool)]
        control = scoped[~scoped["signal_selected"].astype(bool)]
        row: dict[str, Any] = dict(definition)
        for prefix, frame in (("signal", signal), ("control", control)):
            for key, value in _metrics(frame).items():
                row[f"{prefix}_{key}"] = value
        beats_control = (
            row["signal_n"] > 0
            and math.isfinite(row["signal_profit_factor"])
            and row["signal_profit_factor"] > 1.0
            and row["signal_profit_factor"] > row["control_profit_factor"]
        )
        row["direction_held"] = bool(beats_control)
        if beats_control:
            row["clustered_pf_p05"] = _clustered_p05(signal, rng, bootstrap_trials)
            counts = (
                signal.groupby(["signal_date", "sector", "strategy"], observed=True)
                .size()
                .reset_index(name="signal_count")
            )
            pools = _compile_matched_pools(scoped, counts)
            null = [_draw_profit_factors(pools, rng)[1] for _ in range(permutations)]
            exceedances = int(np.sum(np.asarray(null) >= row["signal_profit_factor"]))
            row["permutation_p"] = (exceedances + 1.0) / (permutations + 1.0)
        else:
            row["clustered_pf_p05"] = float("nan")
            row["permutation_p"] = 1.0
        records.append(row)

    detail = pd.DataFrame(records)
    detail["multiplicity_family_size"] = family_size
    detail["holm_adjusted_p"] = holm_adjust(detail["permutation_p"])
    detail["confirmed"] = (
        detail["direction_held"]
        & detail["clustered_pf_p05"].ge(MIN_CLUSTERED_P05)
        & detail["holm_adjusted_p"].le(MAX_PERMUTATION_P)
    )
    detail["preregistration_hash"] = preregistration_hash()
    detail["holdout_start"] = holdout_start
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=ROOT / "out/sector_strategy_universe_v3.csv")
    parser.add_argument("--holdout-start", required=True, help="first signal date of the untouched window")
    parser.add_argument("--out", type=Path, default=ROOT / "out/preregistered_confirmation.csv")
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap-trials", type=int, default=4_000)
    args = parser.parse_args()

    universe = pd.read_csv(args.universe, low_memory=False)
    detail = confirm(
        universe,
        holdout_start=args.holdout_start,
        permutations=args.permutations,
        bootstrap_trials=args.bootstrap_trials,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out, index=False)

    print(f"pre-registration {preregistration_hash()} | holdout from {args.holdout_start}")
    print(f"required Holm-adjusted p <= {MAX_PERMUTATION_P} across {len(PREREGISTERED)} hypotheses")
    columns = [
        "scope", "scope_value", "strategy", "signal_n", "signal_profit_factor",
        "control_profit_factor", "clustered_pf_p05", "permutation_p", "holm_adjusted_p", "confirmed",
    ]
    print(detail[columns].round(4).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
