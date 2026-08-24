"""Evaluate widenings of the promoted selector policy against a completed replay.

The promoted policy ``core_bear_call_credit_dte7_30_volume100_queue9_v18`` selects
only 37 of the 781 evaluated CREDIT trades. The 744 it discards are not junk --
they run at PF 1.84 (held-out 1.88). This script re-runs the *real* selector
(``core._select_challenger_policy_rows``) under candidate policy variants so that
ranking, daily caps and ETF limits are all honoured, then scores the selections
against recorded ``pnl_1x``.

Only rows the replay actually evaluated can be scored, so every variant is
measured on the same outcome-complete population.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from uwos.options_agent import core
from uwos.options_agent import replay as rp

SPLIT_DAY = "2026-05-01"


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(core._truthy)


def base_policy() -> dict[str, Any]:
    return dict(
        next(
            item
            for item in core.SELECTOR_CHALLENGER_POLICIES
            if item["policy_id"] == core.PROMOTED_SELECTOR_POLICY_ID
        )
    )


def run_policy(detail: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    selected = core._select_challenger_policy_rows(rp._selector_frame(detail), policy)
    if selected.empty:
        return selected
    selected = selected.sort_values(
        ["signal_date", "__selector_economic_score", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    cap = int(policy.get("daily_cap") or 0)
    if cap > 0:
        selected = selected.groupby("signal_date", sort=False).head(cap)
    return selected


def score(detail: pd.DataFrame, selected: pd.DataFrame, months: float) -> dict[str, Any]:
    if selected.empty:
        return {"n": 0}
    pnl_by_id = detail.set_index("replay_row_id")["pnl_1x"]
    asof_by_id = detail.set_index("replay_row_id")["asof"]
    ids = selected["replay_row_id"]
    frame = pd.DataFrame(
        {"pnl": ids.map(pnl_by_id).astype(float), "asof": ids.map(asof_by_id)}
    ).dropna(subset=["pnl"])
    if frame.empty:
        return {"n": 0}
    pnl = frame["pnl"]
    held = frame[frame["asof"] >= SPLIT_DAY]["pnl"]

    def pf(series: pd.Series) -> float:
        losses = abs(series[series < 0].sum())
        return series[series > 0].sum() / losses if losses > 0 else float("inf")

    monthly = frame.assign(m=frame["asof"].str[:7]).groupby("m")["pnl"].sum()
    return {
        "n": len(pnl),
        "per_mo": len(pnl) / months,
        "win": 100 * (pnl > 0).mean(),
        "pf": pf(pnl),
        "total": pnl.sum(),
        "mo": pnl.sum() / months,
        "held_n": len(held),
        "held_pf": pf(held) if len(held) else float("nan"),
        "held_total": held.sum() if len(held) else 0.0,
        "pos_months": int((monthly > 0).sum()),
        "n_months": int(len(monthly)),
        "worst_month": monthly.min() if len(monthly) else 0.0,
    }


def show(label: str, res: dict[str, Any]) -> None:
    if not res.get("n"):
        print(f"{label:<44} n=0")
        return
    print(
        f"{label:<44} n={res['n']:<4d} {res['per_mo']:5.1f}/mo win {res['win']:5.1f}% "
        f"PF {res['pf']:6.3f} ${res['total']:>7,.0f} = ${res['mo']:>6,.0f}/mo | "
        f"held n={res['held_n']:<4d} PF {res['held_pf']:6.3f} | "
        f"{res['pos_months']}/{res['n_months']}mo worst ${res['worst_month']:>6,.0f}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--cap-sweep", action="store_true")
    args = parser.parse_args(argv)

    detail = pd.read_csv(
        Path(args.replay_dir) / "options_agent_replay_detail.csv", low_memory=False
    )
    months = detail["asof"].nunique() / 21.0
    evaluated = detail[_truthy(detail["exact_evaluated"]) & detail["pnl_1x"].notna()]
    print(f"replay: {detail['asof'].nunique()} sessions ({months:.1f} months), "
          f"{len(evaluated)} evaluated rows "
          f"({(evaluated['entry_side'] == 'CREDIT').sum()} credit)")
    print()

    base = base_policy()
    variants: list[tuple[str, dict[str, Any]]] = [("BASELINE (promoted)", base)]

    step = dict(base)
    step["allowed_strategy_routes"] = ("bear_call_credit", "bull_put_credit")
    variants.append(("+ bull_put_credit route", dict(step)))

    step = dict(step)
    step["required_underlying_tiers"] = ("core", "liquid")
    variants.append(("+ liquid tier", dict(step)))

    step = dict(step)
    step["max_credit_dte"] = 45
    variants.append(("+ DTE max 30->45", dict(step)))

    step = dict(step)
    step["min_contract_volume"] = 20.0
    variants.append(("+ volume 100->20  [FULL WIDEN]", dict(step)))

    for label, policy in variants:
        show(label, score(detail, run_policy(detail, policy), months))

    if args.cap_sweep:
        print()
        print("DAILY CAP SWEEP on the fully widened policy")
        widened = dict(variants[-1][1])
        for cap in (9, 15, 25, 40, 60, 100):
            policy = dict(widened)
            policy["daily_cap"] = cap
            policy["daily_sleeve_cap"] = cap
            show(f"  daily_cap={cap}", score(detail, run_policy(detail, policy), months))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
