"""Prospective EV shadow lane for Pattern Analysis V2.

NON-EXECUTABLE BY DESIGN. This module never approves, sizes, or places a trade
and never relaxes a production gate. It exists to accumulate *prospective*
evidence for the entry-time EV model so a future promotion decision can be made
on forward outcomes instead of in-sample fitting.

Each run:
  1. trains an entry-time-only EV model on scored outcomes STRICTLY BEFORE the
     run's as-of date (point-in-time; no leakage),
  2. scores the current decision board,
  3. records the top-EV picks for two variants -- ungated and regime-gated
     (stand down when the setup conflicts with the prevailing regime) -- into a
     central append-only ledger with ``execution_eligible=false``,
  4. resolves previously PENDING ledger rows whose outcomes are now known.

Promotion still requires the same honest bar used everywhere else:
pooled PF >= 1.2 AND day-clustered bootstrap PF p05 >= 1.2 AND every fold
profitable -- measured on PROSPECTIVE rows only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SHADOW_VERSION = "pattern_analysis_v2_ev_shadow.2-liquidity-variant-20260726"
LEDGER_NAME = "pattern_analysis_v2_ev_shadow_ledger.csv"

# Quoted bid/ask spread ceiling for the liquidity-gated variant.
LIQUID_MAX_SPREAD_PCT = 0.02
VARIANTS = ("ungated", "regime_gated", "liquid_gated", "liquid_regime_gated")

NUMERIC_FEATURES = ["dte", "spread_pct", "ask", "bid", "fees"]
CATEGORICAL_FEATURES = [
    "direction",
    "strategy_kind",
    "market_regime",
    "sector",
    "dte_bucket",
    "moneyness_bucket",
]
LEDGER_FIELDS = [
    "first_observed_run_date",
    "signal_date",
    "variant",
    "rank",
    "ticker",
    "direction",
    "strategy_kind",
    "contract_profile",
    "pattern_family",
    "market_regime",
    "dte",
    "strikes",
    "expiration",
    "entry",
    "predicted_ev_r",
    "model_train_rows",
    "model_train_through",
    "execution_eligible",
    "no_order_placement",
    "shadow_version",
    "status",
    "realized_net_r",
    "resolved_run_date",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _profile_part(profile: pd.Series, index: int) -> pd.Series:
    return profile.fillna("").astype(str).str.split("__").str[index].fillna("NA").replace("", "NA")


def normalize_features(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Produce one identical feature frame from history or a decision board.

    ``source='history'`` reads validation_details.csv columns.
    ``source='board'`` reads decision_board.csv columns, deriving the fields the
    board encodes inside contract_profile / pattern_family.
    """

    out = pd.DataFrame(index=df.index)
    profile = df.get("contract_profile", pd.Series("", index=df.index)).fillna("").astype(str)
    if source == "history":
        out["dte"] = _num(df["dte"])
        out["spread_pct"] = _num(df["bid_ask_spread_pct"])
        out["ask"] = _num(df["entry_ask"])
        out["bid"] = _num(df["entry_bid"])
        out["fees"] = _num(df["round_trip_fees"])
        out["strategy_kind"] = df["strategy_kind"].fillna("NA").astype(str).str.upper()
        out["sector"] = df["sector"].fillna("NO_SECTOR").astype(str).str.upper()
        out["market_regime"] = df["market_regime"].fillna("UNKNOWN").astype(str).str.upper()
    elif source == "board":
        out["dte"] = _num(df["DTE"])
        out["spread_pct"] = _num(df["spread"])
        out["ask"] = _num(df["ask"])
        out["bid"] = _num(df["bid"])
        out["fees"] = _num(df["fees_commissions"])
        out["strategy_kind"] = _profile_part(profile, 0)
        family = df.get("pattern_family", pd.Series("", index=df.index)).fillna("").astype(str)
        out["sector"] = family.str.split("__").str[-1].replace("", "NO_SECTOR").fillna("NO_SECTOR")
        out["market_regime"] = (
            df.get("regime_alignment", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
            .str.extract(r"(RISK_[A-Z]+|NEUTRAL|MIXED)", expand=False)
            .fillna("UNKNOWN")
        )
    else:
        raise ValueError(source)

    out["dte_bucket"] = _profile_part(profile, 1)
    out["moneyness_bucket"] = _profile_part(profile, 2)
    out["direction"] = df["direction"].fillna("NA").astype(str).str.lower()
    for col in NUMERIC_FEATURES:
        out[col] = out[col].fillna(0.0)
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("NA").astype(str)
    return out


def train_ev_model(history: pd.DataFrame, as_of: str):
    """Train on SCORED outcomes strictly before ``as_of`` (point-in-time)."""

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    train = history[
        (history["status"] == "SCORED")
        & (history["signal_date"].astype(str) < str(as_of))
    ].copy()
    train["net_r"] = _num(train["net_r"])
    train = train[train["net_r"].notna()]
    if len(train) < 200:
        return None, 0, ""
    features = normalize_features(train, source="history")
    model = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("num", "passthrough", NUMERIC_FEATURES),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    random_state=0, n_estimators=300, max_depth=3,
                    learning_rate=0.05, subsample=0.8,
                ),
            ),
        ]
    )
    model.fit(features, train["net_r"])
    return model, int(len(train)), str(train["signal_date"].max())


def regime_stand_down(board: pd.DataFrame) -> pd.Series:
    """True where the setup conflicts with the prevailing regime.

    Backtests showed the EV edge survives in clear-direction tape and inverts in
    adverse/conflicted tape, so the gated variant stands those setups down.
    """

    alignment = board.get("regime_alignment", pd.Series("", index=board.index))
    return alignment.fillna("").astype(str).str.contains("conflict", case=False)


def illiquid_stand_down(board: pd.DataFrame, max_spread_pct: float = LIQUID_MAX_SPREAD_PCT) -> pd.Series:
    """True where the quoted bid/ask spread is too wide to execute cleanly.

    Slippage is ~91% of the cost drag on this universe, and profit factor is
    monotonic in quoted spread (<2%: 0.879 ... >40%: 0.110). Spread is known at
    entry, so filtering on it is leakage-free. This is a LEAD under prospective
    test, not a validated edge.
    """

    spread = pd.to_numeric(board.get("spread"), errors="coerce")
    return ~(spread <= max_spread_pct)


def select_shadow_picks(
    board: pd.DataFrame,
    model,
    *,
    top_n: int,
    variant: str,
) -> pd.DataFrame:
    scored = board.copy()
    scored["predicted_ev_r"] = model.predict(normalize_features(scored, source="board"))
    if "regime" in variant:
        scored = scored[~regime_stand_down(scored)]
    if variant.startswith("liquid"):
        scored = scored[~illiquid_stand_down(scored)]
    if scored.empty:
        return scored
    scored = scored.sort_values("predicted_ev_r", ascending=False)
    scored = scored.drop_duplicates(subset=["ticker", "direction"], keep="first")
    return scored.head(top_n)


def _read_ledger(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LEDGER_FIELDS})


def _ledger_key(row: Mapping[str, Any]) -> tuple:
    return (
        str(row.get("signal_date", "")),
        str(row.get("variant", "")),
        str(row.get("ticker", "")),
        str(row.get("direction", "")),
        str(row.get("contract_profile", "")),
    )


def append_shadow_rows(
    ledger_path: Path,
    picks: pd.DataFrame,
    *,
    variant: str,
    signal_date: str,
    run_date: str,
    train_rows: int,
    train_through: str,
) -> int:
    """Append picks; first observation of a key is immutable."""

    existing = _read_ledger(ledger_path)
    seen = {_ledger_key(row) for row in existing}
    added = 0
    for rank, (_, row) in enumerate(picks.iterrows(), 1):
        record = {
            "first_observed_run_date": run_date,
            "signal_date": signal_date,
            "variant": variant,
            "rank": rank,
            "ticker": row.get("ticker", ""),
            "direction": row.get("direction", ""),
            "strategy_kind": str(row.get("contract_profile", "")).split("__")[0],
            "contract_profile": row.get("contract_profile", ""),
            "pattern_family": row.get("pattern_family", ""),
            "market_regime": str(row.get("regime_alignment", "")),
            "dte": row.get("DTE", ""),
            "strikes": row.get("strikes", ""),
            "expiration": row.get("expiration", ""),
            "entry": row.get("entry", ""),
            "predicted_ev_r": round(float(row.get("predicted_ev_r", 0.0)), 5),
            "model_train_rows": train_rows,
            "model_train_through": train_through,
            "execution_eligible": "false",
            "no_order_placement": "true",
            "shadow_version": SHADOW_VERSION,
            "status": "PENDING",
            "realized_net_r": "",
            "resolved_run_date": "",
        }
        if _ledger_key(record) in seen:
            continue
        existing.append(record)
        seen.add(_ledger_key(record))
        added += 1
    _write_ledger(ledger_path, existing)
    return added


def resolve_pending(ledger_path: Path, history: pd.DataFrame, run_date: str) -> int:
    """Resolve PENDING rows whose outcome is now scored. Point-in-time only."""

    rows = _read_ledger(ledger_path)
    if not rows:
        return 0
    scored = history[history["status"] == "SCORED"].copy()
    scored["net_r"] = _num(scored["net_r"])
    lookup: Dict[tuple, float] = {}
    for _, row in scored.iterrows():
        key = (
            str(row.get("signal_date", "")),
            str(row.get("ticker", "")),
            str(row.get("direction", "")),
            str(row.get("contract_profile", "")),
        )
        value = row.get("net_r")
        if pd.notna(value):
            lookup.setdefault(key, float(value))
    resolved = 0
    for row in rows:
        if str(row.get("status")) != "PENDING":
            continue
        key = (
            str(row.get("signal_date", "")),
            str(row.get("ticker", "")),
            str(row.get("direction", "")),
            str(row.get("contract_profile", "")),
        )
        if key in lookup:
            row["status"] = "RESOLVED"
            row["realized_net_r"] = round(lookup[key], 5)
            row["resolved_run_date"] = run_date
            resolved += 1
    _write_ledger(ledger_path, rows)
    return resolved


def summarize_ledger(ledger_path: Path) -> Dict[str, Any]:
    rows = _read_ledger(ledger_path)
    if not rows:
        return {"rows": 0}
    frame = pd.DataFrame(rows)
    frame["realized_net_r"] = _num(frame.get("realized_net_r", pd.Series(dtype=float)))
    summary: Dict[str, Any] = {"rows": int(len(frame))}
    for variant, group in frame.groupby("variant"):
        done = group[group["status"] == "RESOLVED"]
        realized = done["realized_net_r"].dropna().to_numpy()
        gains = realized[realized > 0].sum()
        losses = -realized[realized < 0].sum()
        summary[str(variant)] = {
            "pending": int((group["status"] == "PENDING").sum()),
            "resolved": int(len(realized)),
            "win_rate": round(float((realized > 0).mean()), 3) if realized.size else None,
            "avg_r": round(float(realized.mean()), 4) if realized.size else None,
            "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
            "promotion_ready": False,
        }
    return summary


def run_shadow_lane(
    out_dir: Path,
    history_path: Path,
    ledger_path: Path,
    *,
    as_of: str,
    run_date: str,
    top_n: int = 5,
) -> Dict[str, Any]:
    board_path = out_dir / "decision_board.csv"
    if not board_path.exists():
        return {"status": "SKIPPED_NO_DECISION_BOARD"}
    history = pd.read_csv(history_path, low_memory=False)
    board = pd.read_csv(board_path, low_memory=False)

    model, train_rows, train_through = train_ev_model(history, as_of)
    if model is None:
        return {"status": "SKIPPED_INSUFFICIENT_HISTORY"}

    added: Dict[str, int] = {}
    for variant in VARIANTS:
        picks = select_shadow_picks(board, model, top_n=top_n, variant=variant)
        added[variant] = (
            0
            if picks.empty
            else append_shadow_rows(
                ledger_path,
                picks,
                variant=variant,
                signal_date=as_of,
                run_date=run_date,
                train_rows=train_rows,
                train_through=train_through,
            )
        )
    resolved = resolve_pending(ledger_path, history, run_date)
    return {
        "status": "OK",
        "shadow_version": SHADOW_VERSION,
        "execution_eligible": False,
        "model_train_rows": train_rows,
        "model_train_through": train_through,
        "picks_added": added,
        "rows_resolved": resolved,
        "ledger": str(ledger_path),
        "ledger_summary": summarize_ledger(ledger_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m uwos.pattern_analysis_v2.shadow_lane")
    parser.add_argument("--out-dir", required=True, help="Pattern Analysis V2 run output directory.")
    parser.add_argument("--history", default=None, help="validation_details.csv. Default: <out-dir>/validation_details.csv")
    parser.add_argument("--ledger", default=None, help="Central shadow ledger CSV.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    history_path = Path(args.history).expanduser().resolve() if args.history else out_dir / "validation_details.csv"
    ledger_path = (
        Path(args.ledger).expanduser().resolve()
        if args.ledger
        else out_dir.parent / LEDGER_NAME
    )
    result = run_shadow_lane(
        out_dir,
        history_path,
        ledger_path,
        as_of=args.as_of,
        run_date=args.run_date or args.as_of,
        top_n=args.top_n,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
