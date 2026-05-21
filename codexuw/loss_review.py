from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float
from .performance import setup_family


def _parse_date(value: object) -> dt.date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _ledger_candidates(out_root: Path) -> list[Path]:
    return [
        out_root / "codexdaily_v3_recommendation_outcome_ledger.csv",
        out_root / "codexuw_recommendation_outcome_ledger.csv",
        out_root / "codexuw_execute_outcome_ledger.csv",
    ]


def load_recommendation_ledgers(out_root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in _ledger_candidates(out_root):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["ledger_source"] = str(path)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def review_recent_losses(
    ledger: pd.DataFrame,
    *,
    asof: dt.date,
    lookback_days: int = 30,
) -> dict[str, Any]:
    if ledger.empty:
        return {"status": "unavailable", "reason": "no_recommendation_ledger"}
    df = ledger.copy()
    if "realized_pnl" not in df.columns:
        return {"status": "unavailable", "reason": "missing_realized_pnl"}
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    date_col = "report_date" if "report_date" in df.columns else "asof" if "asof" in df.columns else ""
    if not date_col:
        return {"status": "unavailable", "reason": "missing_report_date"}
    df["_date"] = df[date_col].map(_parse_date)
    cutoff = asof - dt.timedelta(days=lookback_days)
    recent = df[df["_date"].notna() & (df["_date"] < asof) & (df["_date"] >= cutoff)].copy()
    losses = recent[pd.to_numeric(recent["realized_pnl"], errors="coerce") < 0].copy()
    if losses.empty:
        return {
            "status": "ok",
            "lookback_days": lookback_days,
            "recent_loss_count": 0,
            "family_losses": {},
            "message": "no recent realized losing recommendations found",
        }
    if "setup_family" not in losses.columns:
        losses["setup_family"] = losses.apply(lambda row: setup_family(row.get("strategy"), row.get("direction")), axis=1)
    family_losses: dict[str, Any] = {}
    for family, part in losses.groupby("setup_family"):
        pnl = pd.to_numeric(part["realized_pnl"], errors="coerce").dropna()
        if pnl.empty:
            continue
        family_losses[str(family)] = {
            "loss_count": int(len(pnl)),
            "total_loss": float(pnl.sum()),
            "avg_loss": float(pnl.mean()),
            "latest_loss_date": str(max(d for d in part["_date"] if d is not None)),
            "tickers": sorted({str(t).upper() for t in part.get("ticker", pd.Series(dtype=object)).dropna()}),
        }
    return {
        "status": "ok",
        "lookback_days": lookback_days,
        "recent_loss_count": int(len(losses)),
        "family_losses": family_losses,
        "message": "recent losing setup families found; similar candidates must explain difference or downgrade",
    }


def load_recent_loss_review(out_root: Path, *, asof: dt.date, lookback_days: int = 30) -> dict[str, Any]:
    ledger = load_recommendation_ledgers(out_root)
    review = review_recent_losses(ledger, asof=asof, lookback_days=lookback_days)
    review["ledger_rows"] = int(len(ledger))
    return review


def apply_loss_review(scored: pd.DataFrame, loss_review: dict[str, Any] | None) -> pd.DataFrame:
    if scored.empty or not loss_review or loss_review.get("status") != "ok":
        return scored.copy()
    family_losses = loss_review.get("family_losses") or {}
    if not family_losses:
        return scored.copy()
    out = scored.copy()
    if "penalties" not in out.columns:
        out["penalties"] = ""
    if "score" not in out.columns:
        out["score"] = 0.0
    if "loss_review_note" not in out.columns:
        out["loss_review_note"] = ""
    for idx, row in out.iterrows():
        family = setup_family(row.get("strategy"), row.get("direction"))
        loss = family_losses.get(family)
        if not loss:
            continue
        penalty = f"recent_loss_family:{family}"
        existing = str(out.at[idx, "penalties"] or "")
        out.at[idx, "penalties"] = ";".join([item for item in [existing, penalty] if item])
        out.at[idx, "score"] = max(0.0, safe_float(row.get("score"), 0.0) - 1.25)
        out.at[idx, "loss_review_note"] = (
            f"{family} had {loss.get('loss_count')} recent realized loss(es), "
            f"avg {safe_float(loss.get('avg_loss'), 0.0):.2f}; downgrade unless thesis is materially different"
        )
    return out


def write_loss_review(out_dir: Path, asof: dt.date, loss_review: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"codexdaily_v3_loss_review_{asof}.json"
    import json

    json_path.write_text(json.dumps(loss_review, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for family, item in (loss_review.get("family_losses") or {}).items():
        row = {"setup_family": family}
        row.update(item)
        rows.append(row)
    csv_path = out_dir / f"codexdaily_v3_loss_review_{asof}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path
