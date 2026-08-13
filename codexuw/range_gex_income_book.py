from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_VERSION = "range-gex-income-shadow-v1-20260813"
FROZEN_V421_MAX_DRAWDOWN_1X = 603.50
PRODUCTION_GATES = {
    "minimum_total_trades": 40,
    "minimum_train_trades": 20,
    "minimum_holdout_trades": 15,
    "minimum_stress_pf": 1.50,
    "minimum_train_stress_pf": 1.25,
    "minimum_holdout_stress_pf": 1.25,
    "minimum_wilson_90": 0.60,
    "minimum_positive_month_ratio": 0.70,
    "maximum_drawdown_1x": FROZEN_V421_MAX_DRAWDOWN_1X,
}


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")


def _profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else math.inf


def _wilson_lower(wins: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return math.nan
    rate = wins / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    spread = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center - spread) / denominator


def _max_drawdown(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return math.nan
    curve = np.concatenate(([0.0], np.cumsum(values)))
    return float(np.min(curve - np.maximum.accumulate(curve)))


def derive_gex_features(summary: pd.DataFrame, strikes: pd.DataFrame) -> pd.DataFrame:
    """Build dated aggregate and wall features without using trade outcomes."""
    if summary.empty or strikes.empty:
        return pd.DataFrame()
    aggregate = summary.copy()
    strike_rows = strikes.copy()
    aggregate["asof"] = pd.to_datetime(aggregate["date"], errors="coerce").dt.normalize()
    strike_rows["asof"] = pd.to_datetime(strike_rows["date"], errors="coerce").dt.normalize()
    aggregate["ticker"] = aggregate["ticker"].astype(str).str.upper()
    strike_rows["ticker"] = strike_rows["ticker"].astype(str).str.upper()
    _numeric(
        aggregate,
        ["spot", "gamma_oi_per_1pct", "gamma_vol_per_1pct", "gamma_dir_per_1pct"],
    )
    _numeric(strike_rows, ["spot", "strike", "call_gamma_oi", "put_gamma_oi"])

    captured = pd.to_datetime(aggregate.get("captured_utc"), errors="coerce", utc=True)
    dated = pd.to_datetime(aggregate["asof"], errors="coerce", utc=True)
    aggregate["gex_capture_lag_days"] = (captured.dt.normalize() - dated.dt.normalize()).dt.days
    aggregate["gex_capture_timing"] = np.where(
        aggregate["gex_capture_lag_days"].between(0, 2, inclusive="both"),
        "point_in_time",
        "historical_api_reconstruction",
    )
    aggregate = aggregate.sort_values("captured_utc", na_position="first").drop_duplicates(
        ["asof", "ticker"], keep="last"
    )

    wall_rows: list[dict[str, Any]] = []
    for (asof, ticker), part in strike_rows.groupby(["asof", "ticker"], dropna=False):
        part = part.dropna(subset=["strike"])
        calls = part.assign(_wall_gamma=part["call_gamma_oi"].abs()).dropna(subset=["_wall_gamma"])
        puts = part.assign(_wall_gamma=part["put_gamma_oi"].abs()).dropna(subset=["_wall_gamma"])
        if calls.empty or puts.empty:
            continue
        call_wall = calls.sort_values(["_wall_gamma", "strike"], ascending=[False, True]).iloc[0]
        put_wall = puts.sort_values(["_wall_gamma", "strike"], ascending=[False, False]).iloc[0]
        gross = float(part["call_gamma_oi"].abs().fillna(0.0).sum() + part["put_gamma_oi"].abs().fillna(0.0).sum())
        spot = float(part["spot"].dropna().median()) if part["spot"].notna().any() else math.nan
        wall_rows.append(
            {
                "asof": asof,
                "ticker": ticker,
                "gex_spot": spot,
                "gex_call_wall": float(call_wall["strike"]),
                "gex_put_wall": float(put_wall["strike"]),
                "gex_wall_concentration": (
                    float(call_wall["_wall_gamma"] + put_wall["_wall_gamma"]) / gross
                    if gross > 0
                    else math.nan
                ),
                "gex_gross_strike_oi": gross,
            }
        )
    walls = pd.DataFrame(wall_rows)
    if walls.empty:
        return walls

    columns = [
        "asof",
        "ticker",
        "spot",
        "gamma_oi_per_1pct",
        "gamma_vol_per_1pct",
        "gamma_dir_per_1pct",
        "gex_capture_lag_days",
        "gex_capture_timing",
    ]
    features = aggregate[columns].rename(columns={"spot": "gex_summary_spot"}).merge(
        walls, on=["asof", "ticker"], how="inner", validate="one_to_one"
    )
    features["gex_net_to_gross"] = features["gamma_oi_per_1pct"] / features["gex_gross_strike_oi"].replace(0, np.nan)
    features["gex_directional_ratio"] = features["gamma_dir_per_1pct"].abs() / features[
        "gamma_oi_per_1pct"
    ].abs().replace(0, np.nan)
    features["gex_spot_between_walls"] = (
        (features["gex_put_wall"] < features["gex_spot"])
        & (features["gex_spot"] < features["gex_call_wall"])
    )
    return features.sort_values(["asof", "ticker"]).reset_index(drop=True)


def load_historical_gex(root: Path) -> pd.DataFrame:
    summary_frames = [pd.read_csv(path, low_memory=False) for path in sorted(root.glob("20??-??-??/enrichments/uw/uw_gex_summary_*.csv"))]
    strike_frames = [pd.read_csv(path, low_memory=False) for path in sorted(root.glob("20??-??-??/enrichments/uw/uw_gex_strikes_*.csv"))]
    if not summary_frames or not strike_frames:
        return pd.DataFrame()
    return derive_gex_features(
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(strike_frames, ignore_index=True),
    )


def enrich_replay(replay: pd.DataFrame, gex: pd.DataFrame) -> pd.DataFrame:
    out = replay.copy()
    for column in ["asof", "entry_day", "exit_day"]:
        out[column] = pd.to_datetime(out.get(column), errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    _numeric(
        out,
        [
            "entry_credit",
            "entry_width",
            "entry_credit_pct_width",
            "entry_quote_width_pct",
            "expected_move_ratio",
            "entry_dte",
            "combined_flow_bias",
            "pnl_1x",
            "short_strike_eod",
            "long_strike_eod",
            "iv_hv_ratio",
        ],
    )
    merged = out.merge(gex, on=["asof", "ticker"], how="inner", validate="many_to_one")
    merged["flow_not_contra"] = (
        merged["strategy"].eq("Bull Put Credit Spread") & merged["combined_flow_bias"].ge(-0.05)
    ) | (
        merged["strategy"].eq("Bear Call Credit Spread") & merged["combined_flow_bias"].le(0.05)
    )
    merged["short_outside_wall"] = (
        merged["strategy"].eq("Bull Put Credit Spread")
        & merged["short_strike_eod"].le(merged["gex_put_wall"])
    ) | (
        merged["strategy"].eq("Bear Call Credit Spread")
        & merged["short_strike_eod"].ge(merged["gex_call_wall"])
    )
    merged["wall_buffer_pct"] = np.where(
        merged["strategy"].eq("Bull Put Credit Spread"),
        (merged["gex_put_wall"] - merged["short_strike_eod"]) / merged["gex_spot"],
        (merged["short_strike_eod"] - merged["gex_call_wall"]) / merged["gex_spot"],
    )
    return merged


def _base_credit_quality(frame: pd.DataFrame) -> pd.Series:
    exact = _truthy(frame.get("exact_evaluated", pd.Series(False, index=frame.index)))
    earnings = _truthy(frame.get("earnings_crosses", pd.Series(False, index=frame.index)))
    return (
        frame["strategy"].isin({"Bear Call Credit Spread", "Bull Put Credit Spread"})
        & exact
        & frame["entry_credit"].gt(0)
        & frame["entry_width"].gt(frame["entry_credit"])
        & ~earnings
        & frame["entry_credit_pct_width"].between(0.15, 0.45, inclusive="both")
        & frame["entry_quote_width_pct"].le(0.35)
        & frame["expected_move_ratio"].le(0.90)
        & frame["entry_dte"].between(21, 44, inclusive="both")
    )


def build_vertical_shadow(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = (
        _base_credit_quality(enriched)
        & enriched["regime"].astype(str).eq("range")
        & enriched["gamma_oi_per_1pct"].gt(0)
        & enriched["gex_spot_between_walls"].astype(bool)
        & enriched["gex_wall_concentration"].ge(0.15)
        & enriched["short_outside_wall"].astype(bool)
        & enriched["flow_not_contra"].astype(bool)
    )
    qualified = enriched[policy].copy()
    qualified["range_gex_rank"] = (
        2.0 * qualified["entry_credit_pct_width"]
        - 0.8 * qualified["expected_move_ratio"]
        - 0.5 * qualified["entry_quote_width_pct"]
        + 0.5 * qualified["gex_wall_concentration"]
        + 0.25 * qualified["wall_buffer_pct"].clip(lower=0, upper=0.20)
    )
    selected = (
        qualified.sort_values(["asof", "range_gex_rank"], ascending=[True, False])
        .groupby("asof", as_index=False)
        .head(1)
        .copy()
    )
    return qualified, selected


def build_condor_shadow(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = enriched[_base_credit_quality(enriched)].copy()
    components["component_rank"] = (
        2.0 * components["entry_credit_pct_width"]
        - 0.8 * components["expected_move_ratio"]
        - 0.5 * components["entry_quote_width_pct"]
    )
    components = (
        components.sort_values(
            ["asof", "ticker", "expiry", "strategy", "component_rank"],
            ascending=[True, True, True, True, False],
        )
        .groupby(["asof", "ticker", "expiry", "strategy"], as_index=False)
        .head(1)
    )
    puts = components[components["strategy"].eq("Bull Put Credit Spread")]
    calls = components[components["strategy"].eq("Bear Call Credit Spread")]
    pairs = puts.merge(calls, on=["asof", "ticker", "expiry"], suffixes=("_put", "_call"), how="inner")
    if pairs.empty:
        return pairs, pairs.copy(), pairs.copy()
    for column in [
        "gex_spot",
        "gex_put_wall",
        "gex_call_wall",
        "gex_wall_concentration",
        "gamma_oi_per_1pct",
        "gex_capture_timing",
        "gex_capture_lag_days",
    ]:
        pairs[column] = pairs[f"{column}_put"]
    pairs["entry_day"] = pairs[["entry_day_put", "entry_day_call"]].max(axis=1)
    pairs["exit_day"] = pairs[["exit_day_put", "exit_day_call"]].max(axis=1)
    pairs["total_credit"] = pairs["entry_credit_put"] + pairs["entry_credit_call"]
    pairs["max_wing_width"] = pairs[["entry_width_put", "entry_width_call"]].max(axis=1)
    pairs["max_loss_1x"] = (pairs["max_wing_width"] - pairs["total_credit"]) * 100.0
    pairs["credit_pct_width"] = pairs["total_credit"] / pairs["max_wing_width"]
    pairs["pnl_1x"] = pairs["pnl_1x_put"] + pairs["pnl_1x_call"]
    pairs["neutral_flow"] = pairs[["combined_flow_bias_put", "combined_flow_bias_call"]].mean(axis=1).abs()
    pairs["iv_hv"] = pairs[["iv_hv_ratio_put", "iv_hv_ratio_call"]].mean(axis=1)
    pairs["shorts_outside_walls"] = (
        pairs["short_strike_eod_put"].le(pairs["gex_put_wall"])
        & pairs["short_strike_eod_call"].ge(pairs["gex_call_wall"])
    )
    pairs["gex_spot_between_walls"] = (
        pairs["gex_put_wall"].lt(pairs["gex_spot"])
        & pairs["gex_spot"].lt(pairs["gex_call_wall"])
    )
    base_quality = (
        pairs["entry_day_put"].eq(pairs["entry_day_call"])
        & pairs["short_strike_eod_put"].lt(pairs["gex_spot"])
        & pairs["gex_spot"].lt(pairs["short_strike_eod_call"])
        & pairs["credit_pct_width"].between(0.25, 0.70, inclusive="both")
        & pairs["max_loss_1x"].gt(0)
    )
    policy = (
        base_quality
        & pairs["regime_put"].astype(str).eq("range")
        & pairs["regime_call"].astype(str).eq("range")
        & pairs["gamma_oi_per_1pct"].gt(0)
        & pairs["gex_spot_between_walls"]
        & pairs["gex_wall_concentration"].ge(0.15)
        & pairs["shorts_outside_walls"]
        & pairs["neutral_flow"].le(0.10)
        & pairs["iv_hv"].ge(1.05)
    )
    qualified = pairs[policy].copy()
    qualified["range_gex_rank"] = (
        2.0 * qualified["credit_pct_width"]
        - 0.8 * qualified[["expected_move_ratio_put", "expected_move_ratio_call"]].max(axis=1)
        - 0.5 * qualified[["entry_quote_width_pct_put", "entry_quote_width_pct_call"]].max(axis=1)
        + 0.5 * qualified["gex_wall_concentration"]
        - 0.25 * qualified["neutral_flow"]
    )
    selected = (
        qualified.sort_values(["asof", "range_gex_rank"], ascending=[True, False])
        .groupby("asof", as_index=False)
        .head(1)
        .copy()
    )
    return pairs, qualified, selected


def _segment_metrics(frame: pd.DataFrame, *, credit_column: str, stress: float) -> dict[str, Any]:
    pnl = pd.to_numeric(frame.get("pnl_1x"), errors="coerce")
    credit = pd.to_numeric(frame.get(credit_column), errors="coerce").fillna(0.0)
    stressed = (pnl - credit * 100.0 * stress).dropna()
    wins = int(stressed.gt(0).sum())
    return {
        "trades": int(len(stressed)),
        "wins": wins,
        "losses": int(stressed.le(0).sum()),
        "win_rate": wins / len(stressed) if len(stressed) else math.nan,
        "wilson_90": _wilson_lower(wins, len(stressed)),
        "profit_factor": _profit_factor(stressed),
        "pnl_1x": float(stressed.sum()),
        "max_drawdown_1x": _max_drawdown(stressed),
    }


def evaluate_shadow_book(
    selected: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    credit_column: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected = selected.copy()
    selected["entry_day"] = pd.to_datetime(
        selected.get("entry_day", pd.Series(index=selected.index, dtype="datetime64[ns]")),
        errors="coerce",
    )
    selected["exit_day"] = pd.to_datetime(
        selected.get("exit_day", pd.Series(index=selected.index, dtype="datetime64[ns]")),
        errors="coerce",
    )
    rows: list[dict[str, Any]] = []
    segments = {
        "all": selected,
        "train_matured": selected[selected.get("exit_day", pd.Series(index=selected.index)).lt(cutoff)],
        "holdout": selected[selected.get("entry_day", pd.Series(index=selected.index)).ge(cutoff)],
    }
    for segment, part in segments.items():
        for stress in (0.0, 0.05, 0.10):
            rows.append({"segment": segment, "fill_stress": stress, **_segment_metrics(part, credit_column=credit_column, stress=stress)})
    metrics = pd.DataFrame(rows)
    if selected.empty:
        monthly = pd.DataFrame(columns=["month", "trades", "wins", "pnl_1x"])
    else:
        monthly_frame = selected.copy()
        monthly_frame["month"] = pd.to_datetime(monthly_frame["entry_day"]).dt.to_period("M").astype(str)
        monthly = monthly_frame.groupby("month", as_index=False).agg(
            trades=("pnl_1x", "size"),
            wins=("pnl_1x", lambda values: int((values > 0).sum())),
            pnl_1x=("pnl_1x", "sum"),
        )
    base = metrics[(metrics["segment"].eq("all")) & metrics["fill_stress"].eq(0.10)]
    train = metrics[(metrics["segment"].eq("train_matured")) & metrics["fill_stress"].eq(0.10)]
    holdout = metrics[(metrics["segment"].eq("holdout")) & metrics["fill_stress"].eq(0.10)]
    timing = set(selected.get("gex_capture_timing", pd.Series(dtype=str)).astype(str))
    reasons: list[str] = []
    if "historical_api_reconstruction" in timing:
        reasons.append("historical GEX was reconstructed after the source date")
    if base.empty or int(base.iloc[0]["trades"]) < PRODUCTION_GATES["minimum_total_trades"]:
        reasons.append("total sample below production threshold")
    if train.empty or int(train.iloc[0]["trades"]) < PRODUCTION_GATES["minimum_train_trades"]:
        reasons.append("maturity-safe training sample below threshold")
    if holdout.empty or int(holdout.iloc[0]["trades"]) < PRODUCTION_GATES["minimum_holdout_trades"]:
        reasons.append("untouched holdout sample below threshold")
    if not base.empty and float(base.iloc[0]["profit_factor"]) < PRODUCTION_GATES["minimum_stress_pf"]:
        reasons.append("10% fill-stress PF below threshold")
    if not train.empty and float(train.iloc[0]["profit_factor"]) < PRODUCTION_GATES["minimum_train_stress_pf"]:
        reasons.append("training 10% fill-stress PF below threshold")
    if not holdout.empty and float(holdout.iloc[0]["profit_factor"]) < PRODUCTION_GATES["minimum_holdout_stress_pf"]:
        reasons.append("holdout 10% fill-stress PF below threshold")
    if not base.empty and float(base.iloc[0]["wilson_90"]) < PRODUCTION_GATES["minimum_wilson_90"]:
        reasons.append("Wilson lower bound below threshold")
    positive_month_ratio = float(monthly["pnl_1x"].gt(0).mean()) if not monthly.empty else 0.0
    if positive_month_ratio < PRODUCTION_GATES["minimum_positive_month_ratio"]:
        reasons.append("positive-month ratio below threshold")
    if not base.empty and abs(float(base.iloc[0]["max_drawdown_1x"])) > PRODUCTION_GATES["maximum_drawdown_1x"]:
        reasons.append("drawdown regresses versus frozen V4.21 book")
    status = "PASS" if not reasons else "RESEARCH_ONLY"
    return (
        {
            "policy_version": POLICY_VERSION,
            "status": status,
            "execution_authorized": status == "PASS",
            "reasons": reasons,
            "positive_month_ratio": positive_month_ratio,
            "production_gates": PRODUCTION_GATES,
        },
        metrics,
        monthly,
    )


def run_research(*, root: Path, replay_path: Path, out_dir: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gex = load_historical_gex(root)
    replay = pd.read_csv(replay_path, low_memory=False)
    enriched = enrich_replay(replay, gex)
    vertical_qualified, vertical_selected = build_vertical_shadow(enriched)
    condor_pairs, condor_qualified, condor_selected = build_condor_shadow(enriched)
    vertical_summary, vertical_metrics, vertical_monthly = evaluate_shadow_book(
        vertical_selected, cutoff=cutoff, credit_column="entry_credit"
    )
    condor_summary, condor_metrics, condor_monthly = evaluate_shadow_book(
        condor_selected, cutoff=cutoff, credit_column="total_credit"
    )
    overall = {
        "policy_version": POLICY_VERSION,
        "status": "PASS" if vertical_summary["status"] == "PASS" or condor_summary["status"] == "PASS" else "RESEARCH_ONLY",
        "execution_authorized": bool(vertical_summary["execution_authorized"] or condor_summary["execution_authorized"]),
        "cutoff": cutoff.date().isoformat(),
        "gex_dates": int(gex["asof"].nunique()) if not gex.empty else 0,
        "gex_tickers": int(gex["ticker"].nunique()) if not gex.empty else 0,
        "joined_replay_rows": int(len(enriched)),
        "vertical": vertical_summary,
        "condor": condor_summary,
    }
    artifacts = {
        "historical_gex_features": gex,
        "range_gex_vertical_qualified": vertical_qualified,
        "range_gex_vertical_selected": vertical_selected,
        "range_gex_vertical_metrics": vertical_metrics,
        "range_gex_vertical_monthly": vertical_monthly,
        "range_gex_condor_pairs": condor_pairs,
        "range_gex_condor_qualified": condor_qualified,
        "range_gex_condor_selected": condor_selected,
        "range_gex_condor_metrics": condor_metrics,
        "range_gex_condor_monthly": condor_monthly,
    }
    for name, frame in artifacts.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "range_gex_validation.json").write_text(json.dumps(overall, indent=2, default=str) + "\n", encoding="utf-8")
    report = [
        "# Range/GEX Income Shadow Book",
        "",
        f"Policy: `{POLICY_VERSION}`",
        f"Status: **{overall['status']}**",
        f"Execution authorized: **{overall['execution_authorized']}**",
        "",
        "This development book cannot change Codex Daily V4.21 production decisions.",
        "",
        "## Data Coverage",
        "",
        f"- GEX dates: {overall['gex_dates']}",
        f"- GEX tickers: {overall['gex_tickers']}",
        f"- Replay rows joined to dated GEX: {overall['joined_replay_rows']}",
        "- Historical API reconstructions are labeled and cannot independently authorize production.",
        "",
        "## Vertical Validation",
        "",
        vertical_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Blockers: " + "; ".join(vertical_summary["reasons"]),
        "",
        "## Condor Validation",
        "",
        condor_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Blockers: " + "; ".join(condor_summary["reasons"]),
        "",
        "## Release Rule",
        "",
        "No range/GEX setup receives execution authority until all stated sample, holdout, fill-stress, monthly consistency, drawdown, and source-timing gates pass.",
    ]
    (out_dir / "range_gex_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated Codex Daily range/GEX income shadow book.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-05-19")
    args = parser.parse_args()
    result = run_research(root=args.root, replay_path=args.replay, out_dir=args.out_dir, cutoff=pd.Timestamp(args.cutoff))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
