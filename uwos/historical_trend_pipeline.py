#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from urllib.parse import quote
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RUN_MANIFEST_DATE_RE = re.compile(r"run_manifest_(\d{4}-\d{2}-\d{2})")
DATE_TEXT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OUTPUT_MD_RE = re.compile(
    r"anu-expert-trade-table-(\d{4}-\d{2}-\d{2})(?:-([a-zA-Z0-9._-]+))?\.md$"
)
EXEC_SPLIT_RE = re.compile(
    r"Execution book split:\s*Core=(\d+)\s*,\s*Tactical=(\d+)\s*,\s*Watch=(\d+)",
    flags=re.IGNORECASE,
)
CAT_SPLIT_RE = re.compile(
    r"Category split:\s*Approved-FIRE=(\d+)\s*,\s*Approved-SHIELD=(\d+)\s*,\s*Watch-FIRE=(\d+)\s*,\s*Watch-SHIELD=(\d+)",
    flags=re.IGNORECASE,
)
BOOL_TRUE = {"true", "1", "yes", "y", "pass", "ok"}
BOOL_FALSE = {"false", "0", "no", "n", "fail"}
DEFAULT_ROOT = Path(r"c:\uw_root\out\replay_compare")


@dataclass(frozen=True)
class RunArtifacts:
    live_csv: Optional[Path]
    live_final_csv: Optional[Path]
    setup_likelihood_csv: Optional[Path]
    dropped_csv: Optional[Path]
    output_md: Optional[Path]


@dataclass(frozen=True)
class RunBundle:
    manifest_path: Path
    run_dir: Path
    trade_date: pd.Timestamp
    variant: str
    manifest: Dict[str, Any]
    artifacts: RunArtifacts


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if not s:
                return float("nan")
            return float(s)
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> int:
    v = _safe_float(x)
    if math.isnan(v):
        return 0
    return int(round(v))


def _normalize_text(x: Any, *, upper: bool = False, lower: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    if upper:
        return s.upper()
    if lower:
        return s.lower()
    return s


def _normalize_variant(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name or "").strip())
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "default"


def _path_from_any(raw: Any, run_dir: Path) -> Optional[Path]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    p = Path(txt)
    if p.exists():
        return p
    if not p.is_absolute():
        q = (run_dir / p).resolve()
        if q.exists():
            return q
    return None


def _resolve_artifact_path(
    manifest: Dict[str, Any],
    run_dir: Path,
    keys: Sequence[str],
    fallback_globs: Sequence[str],
) -> Optional[Path]:
    art = manifest.get("artifacts")
    candidates: List[Any] = []
    if isinstance(art, dict):
        for k in keys:
            if k in art:
                candidates.append(art.get(k))
    for k in keys:
        if k in manifest:
            candidates.append(manifest.get(k))
    for raw in candidates:
        p = _path_from_any(raw, run_dir)
        if p is not None:
            return p
    for pattern in fallback_globs:
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _extract_trade_date(manifest_path: Path, manifest: Dict[str, Any]) -> Optional[pd.Timestamp]:
    asof = manifest.get("asof_date")
    if isinstance(asof, str) and DATE_TEXT_RE.fullmatch(asof.strip()):
        return pd.to_datetime(asof.strip(), errors="coerce")
    m = RUN_MANIFEST_DATE_RE.search(manifest_path.name)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")
    return None


def _variant_from_manifest_fields(manifest: Dict[str, Any]) -> str:
    out_md = manifest.get("output_md")
    if isinstance(out_md, str):
        m = OUTPUT_MD_RE.search(Path(out_md).name)
        if m and m.group(2):
            return _normalize_variant(m.group(2))
    config_path = manifest.get("config_path")
    if isinstance(config_path, str) and config_path.strip():
        stem = Path(config_path).stem
        stem = re.sub(r"^rulebook_config_", "", stem)
        return _normalize_variant(stem)
    return "default"


def infer_variant(
    manifest_path: Path,
    search_root: Path,
    trade_date: pd.Timestamp,
    manifest: Dict[str, Any],
) -> str:
    date_txt = trade_date.strftime("%Y-%m-%d")
    parent = manifest_path.parent.name
    grand = manifest_path.parent.parent.name if manifest_path.parent.parent else ""
    variant = ""
    if parent == date_txt:
        variant = grand
    elif grand == date_txt:
        variant = parent
    else:
        variant = parent
    variant = _normalize_variant(variant)
    if variant.lower() in {"out", "replay_compare", "default"}:
        rel_parts = []
        try:
            rel_parts = list(manifest_path.relative_to(search_root).parts)
        except Exception:
            rel_parts = list(manifest_path.parts)
        for i, part in enumerate(rel_parts):
            if part == date_txt:
                if i > 0:
                    left = _normalize_variant(rel_parts[i - 1])
                    if left.lower() not in {"out", "replay_compare"}:
                        variant = left
                        break
                if i + 1 < len(rel_parts):
                    right = _normalize_variant(rel_parts[i + 1])
                    if right.lower() not in {"run_manifest", "default"}:
                        variant = right
                        break
    if variant.lower() in {"out", "replay_compare", "default"}:
        variant = _variant_from_manifest_fields(manifest)
    return _normalize_variant(variant)


def discover_run_bundles(search_root: Path) -> List[RunBundle]:
    if not search_root.exists():
        raise FileNotFoundError(f"Search root not found: {search_root}")
    bundles: List[RunBundle] = []
    for manifest_path in sorted(search_root.rglob("run_manifest_*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                continue
        except Exception:
            continue
        trade_date = _extract_trade_date(manifest_path, manifest)
        if trade_date is None or pd.isna(trade_date):
            continue
        run_dir = manifest_path.parent
        date_txt = trade_date.strftime("%Y-%m-%d")
        artifacts = RunArtifacts(
            live_csv=_resolve_artifact_path(
                manifest,
                run_dir,
                keys=("live_csv",),
                fallback_globs=(f"live_trade_table_{date_txt}.csv", "live_trade_table_*.csv"),
            ),
            live_final_csv=_resolve_artifact_path(
                manifest,
                run_dir,
                keys=("live_final_csv",),
                fallback_globs=(
                    f"live_trade_table_{date_txt}_final.csv",
                    "live_trade_table_*_final.csv",
                ),
            ),
            setup_likelihood_csv=_resolve_artifact_path(
                manifest,
                run_dir,
                keys=("likelihood_csv", "setup_likelihood_csv"),
                fallback_globs=(f"setup_likelihood_{date_txt}.csv", "setup_likelihood_*.csv"),
            ),
            dropped_csv=_resolve_artifact_path(
                manifest,
                run_dir,
                keys=("dropped_csv",),
                fallback_globs=(f"dropped_trades_{date_txt}.csv", "dropped_trades_*.csv"),
            ),
            output_md=_resolve_artifact_path(
                manifest,
                run_dir,
                keys=("output_md",),
                fallback_globs=(f"anu-expert-trade-table-{date_txt}*.md", "anu-expert-trade-table-*.md"),
            ),
        )
        variant = infer_variant(manifest_path, search_root, trade_date, manifest)
        bundles.append(
            RunBundle(
                manifest_path=manifest_path,
                run_dir=run_dir,
                trade_date=trade_date.normalize(),
                variant=variant,
                manifest=manifest,
                artifacts=artifacts,
            )
        )
    return bundles


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _to_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    text = values.astype(str).str.strip().str.lower()
    return text.isin(BOOL_TRUE)


def _bool_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        vals = s.astype(float)
    else:
        text = s.astype(str).str.strip().str.lower()
        mapped = text.map(
            lambda t: True
            if t in BOOL_TRUE
            else (False if t in BOOL_FALSE else (np.nan if t not in {"nan", ""} else np.nan))
        )
        vals = pd.to_numeric(mapped, errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.mean())


def _num_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.mean())


def _num_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.sum())


def _linear_slope(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype="float64")
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return float("nan")
    y = arr[mask]
    x = np.arange(len(y), dtype="float64")
    if len(np.unique(x)) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _prepare_proxy_join(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ticker" in out.columns:
        out["_k_ticker"] = out["ticker"].map(lambda x: _normalize_text(x, upper=True))
    if "strategy" in out.columns:
        out["_k_strategy"] = out["strategy"].map(lambda x: _normalize_text(x, lower=True))
    if "expiry" in out.columns:
        out["_k_expiry"] = out["expiry"].map(lambda x: _normalize_text(x))
    if "entry_gate" in out.columns:
        out["_k_entry_gate"] = out["entry_gate"].map(lambda x: _normalize_text(x, lower=True))
    return out


def _compute_proxy_frame(live_df: pd.DataFrame, setup_df: pd.DataFrame) -> pd.DataFrame:
    if live_df.empty or setup_df.empty:
        return pd.DataFrame()
    lhs = _prepare_proxy_join(live_df)
    rhs = _prepare_proxy_join(setup_df)
    join_cols = [c for c in ("_k_ticker", "_k_strategy", "_k_expiry", "_k_entry_gate") if c in lhs.columns and c in rhs.columns]
    if len(join_cols) < 3:
        return pd.DataFrame()
    rhs_cols = join_cols + [c for c in ("hist_success_pct", "edge_pct", "confidence", "verdict") if c in rhs.columns]
    rhs_small = rhs[rhs_cols].dropna(subset=join_cols).drop_duplicates(subset=join_cols, keep="first")
    merged = lhs.merge(rhs_small, on=join_cols, how="inner")
    if merged.empty:
        return merged
    merged["hist_success_pct"] = pd.to_numeric(merged.get("hist_success_pct"), errors="coerce")
    merged["edge_pct"] = pd.to_numeric(merged.get("edge_pct"), errors="coerce")
    merged["live_max_profit"] = pd.to_numeric(merged.get("live_max_profit"), errors="coerce")
    merged["live_max_loss"] = pd.to_numeric(merged.get("live_max_loss"), errors="coerce")
    merged = merged[
        merged["hist_success_pct"].notna()
        & merged["live_max_profit"].notna()
        & merged["live_max_loss"].notna()
        & (merged["live_max_profit"] >= 0)
        & (merged["live_max_loss"] >= 0)
    ].copy()
    if merged.empty:
        return merged
    p = (merged["hist_success_pct"] / 100.0).clip(lower=0.0, upper=1.0)
    merged["expected_gross_profit"] = p * merged["live_max_profit"]
    merged["expected_gross_loss"] = (1.0 - p) * merged["live_max_loss"]
    merged["expected_net"] = merged["expected_gross_profit"] - merged["expected_gross_loss"]
    return merged


def _build_ticker_rows(
    bundle: RunBundle,
    live_df: pd.DataFrame,
    final_df: pd.DataFrame,
    setup_df: pd.DataFrame,
) -> pd.DataFrame:
    md_rows = _extract_output_md_ranked_rows(bundle.artifacts.output_md)
    if not md_rows.empty and not live_df.empty:
        source = live_df.copy()
        source_label = "live_csv_output_md_bound"
    elif not final_df.empty:
        source = final_df.copy()
        source_label = "live_final_csv"
    elif not live_df.empty:
        source = live_df.copy()
        source_label = "live_csv_gate_pass"
        if "gate_pass_live" in source.columns:
            gate = source["gate_pass_live"].astype(str).str.strip().str.lower().isin(BOOL_TRUE)
            source = source[gate].copy()
    else:
        return pd.DataFrame()

    if source.empty or "ticker" not in source.columns:
        return pd.DataFrame()
    source["ticker"] = source["ticker"].map(lambda x: _normalize_text(x, upper=True))
    if "strategy" in source.columns:
        source["strategy"] = source["strategy"].map(_normalize_text)
    else:
        source["strategy"] = ""
    if "expiry" in source.columns:
        source["expiry"] = source["expiry"].map(_normalize_text)
    else:
        source["expiry"] = ""
    if "entry_gate" in source.columns:
        source["entry_gate"] = source["entry_gate"].map(_normalize_text)
    else:
        source["entry_gate"] = ""
    if "net_type" in source.columns:
        source["net_type"] = source["net_type"].map(lambda x: _normalize_text(x, lower=True))
    else:
        source["net_type"] = ""
    for leg_col in (
        "short_leg",
        "long_leg",
        "short_put_leg",
        "long_put_leg",
        "short_call_leg",
        "long_call_leg",
    ):
        if leg_col in source.columns:
            source[leg_col] = source[leg_col].map(_normalize_text)
        else:
            source[leg_col] = ""
    if "track" in source.columns:
        source["track"] = source["track"].map(lambda x: _normalize_text(x, upper=True))
    else:
        source["track"] = ""
    if "optimal_stage1" in source.columns:
        source["optimal_stage1"] = source["optimal_stage1"].map(_normalize_text)
    else:
        source["optimal_stage1"] = ""
    if "confidence_tier" in source.columns:
        source["confidence_tier"] = source["confidence_tier"].map(_normalize_text)
    else:
        source["confidence_tier"] = ""
    if "thesis" in source.columns:
        source["thesis"] = source["thesis"].map(_normalize_text)
    else:
        source["thesis"] = ""
    if "invalidation" in source.columns:
        source["invalidation"] = source["invalidation"].map(_normalize_text)
    else:
        source["invalidation"] = ""
    if "live_status" in source.columns:
        source["live_status"] = source["live_status"].map(_normalize_text)
    else:
        source["live_status"] = ""
    if "core_ok_stage1" in source.columns:
        source["core_ok_stage1"] = _to_bool_series(source["core_ok_stage1"])
    else:
        source["core_ok_stage1"] = False
    if "gate_pass_live" in source.columns:
        source["gate_pass_live"] = _to_bool_series(source["gate_pass_live"])
    else:
        source["gate_pass_live"] = False
    if "is_final_live_valid" in source.columns:
        source["is_final_live_valid"] = _to_bool_series(source["is_final_live_valid"])
    else:
        source["is_final_live_valid"] = False
    if "conviction" in source.columns:
        source["conviction"] = pd.to_numeric(source["conviction"], errors="coerce")
    source["live_max_profit"] = pd.to_numeric(source.get("live_max_profit"), errors="coerce")
    source["live_max_loss"] = pd.to_numeric(source.get("live_max_loss"), errors="coerce")
    source["live_net_mark"] = pd.to_numeric(source.get("live_net_mark"), errors="coerce")
    source["live_net_bid_ask"] = pd.to_numeric(source.get("live_net_bid_ask"), errors="coerce")
    source["spot_live_last"] = pd.to_numeric(source.get("spot_live_last"), errors="coerce")
    source["rank_md"] = np.nan
    source["track_md"] = ""
    source["execution_book_md"] = ""
    source["expiry_md"] = ""
    opt_lower = source["optimal_stage1"].astype(str).str.lower()
    source["execution_book_inferred"] = np.where(
        opt_lower.str.contains("watch", na=False),
        "Watch",
        np.where(source["core_ok_stage1"], "Core", "Tactical"),
    )
    if not md_rows.empty:
        source = _select_source_rows_by_output_md(source, md_rows)
        if "execution_book_md" in source.columns:
            has_book = source["execution_book_md"].astype(str).str.strip().ne("")
            source.loc[has_book, "execution_book_inferred"] = source.loc[has_book, "execution_book_md"].astype(str)
        if "track_md" in source.columns:
            has_track = source["track_md"].astype(str).str.strip().ne("")
            source.loc[has_track, "track"] = source.loc[has_track, "track_md"].astype(str).str.upper()
        if "expiry_md" in source.columns and "expiry" in source.columns:
            has_exp = source["expiry_md"].astype(str).str.strip().ne("")
            source.loc[has_exp, "expiry"] = source.loc[has_exp, "expiry_md"].astype(str)

    source = _prepare_proxy_join(source)
    if not setup_df.empty:
        setup_keyed = _prepare_proxy_join(setup_df)
        join_cols = [
            c
            for c in ("_k_ticker", "_k_strategy", "_k_expiry", "_k_entry_gate")
            if c in source.columns and c in setup_keyed.columns
        ]
        if len(join_cols) >= 3:
            rhs_cols = join_cols + [c for c in ("hist_success_pct", "edge_pct", "confidence", "verdict") if c in setup_keyed.columns]
            rhs = setup_keyed[rhs_cols].drop_duplicates(subset=join_cols, keep="first")
            source = source.merge(rhs, on=join_cols, how="left")

    source["hist_success_pct"] = pd.to_numeric(source.get("hist_success_pct"), errors="coerce")
    source["edge_pct"] = pd.to_numeric(source.get("edge_pct"), errors="coerce")
    p = (source["hist_success_pct"] / 100.0).clip(lower=0.0, upper=1.0)
    source["expected_net"] = p * source["live_max_profit"] - (1.0 - p) * source["live_max_loss"]
    source["trade_date"] = bundle.trade_date
    source["week_start"] = bundle.trade_date - pd.to_timedelta(bundle.trade_date.weekday(), unit="D")
    source["variant"] = bundle.variant
    source["source_table"] = source_label
    source["manifest_path"] = str(bundle.manifest_path)
    source["output_md_path"] = str(bundle.artifacts.output_md or "")
    cols = [
        "variant",
        "trade_date",
        "week_start",
        "ticker",
        "strategy",
        "expiry",
        "net_type",
        "entry_gate",
        "short_leg",
        "long_leg",
        "short_put_leg",
        "long_put_leg",
        "short_call_leg",
        "long_call_leg",
        "track",
        "confidence_tier",
        "optimal_stage1",
        "execution_book_inferred",
        "rank_md",
        "core_ok_stage1",
        "gate_pass_live",
        "is_final_live_valid",
        "live_status",
        "conviction",
        "hist_success_pct",
        "edge_pct",
        "expected_net",
        "live_net_mark",
        "live_net_bid_ask",
        "live_max_profit",
        "live_max_loss",
        "spot_live_last",
        "thesis",
        "invalidation",
        "source_table",
        "manifest_path",
        "output_md_path",
    ]
    for c in cols:
        if c not in source.columns:
            source[c] = np.nan
    return source[cols]


def _build_drop_reason_rows(bundle: RunBundle, dropped_df: pd.DataFrame) -> pd.DataFrame:
    if dropped_df.empty or "drop_reason" not in dropped_df.columns:
        return pd.DataFrame()
    tmp = dropped_df.copy()
    tmp["drop_reason"] = tmp["drop_reason"].astype(str).str.strip()
    if "stage" in tmp.columns:
        tmp["stage"] = tmp["stage"].astype(str).str.strip()
    else:
        tmp["stage"] = "unknown"
    grp = (
        tmp.groupby(["stage", "drop_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grp["variant"] = bundle.variant
    grp["trade_date"] = bundle.trade_date
    grp["week_start"] = bundle.trade_date - pd.to_timedelta(bundle.trade_date.weekday(), unit="D")
    grp["manifest_path"] = str(bundle.manifest_path)
    return grp[["variant", "trade_date", "week_start", "stage", "drop_reason", "count", "manifest_path"]]


def _parse_output_md_category_counts(path: Optional[Path]) -> Dict[str, float]:
    out = {
        "core_book_rows": float("nan"),
        "tactical_book_rows": float("nan"),
        "watch_book_rows": float("nan"),
        "approved_fire_rows": float("nan"),
        "approved_shield_rows": float("nan"),
        "watch_fire_rows": float("nan"),
        "watch_shield_rows": float("nan"),
    }
    if path is None or not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    m_exec = EXEC_SPLIT_RE.search(text)
    if m_exec:
        out["core_book_rows"] = float(_safe_int(m_exec.group(1)))
        out["tactical_book_rows"] = float(_safe_int(m_exec.group(2)))
        out["watch_book_rows"] = float(_safe_int(m_exec.group(3)))
    m_cat = CAT_SPLIT_RE.search(text)
    if m_cat:
        out["approved_fire_rows"] = float(_safe_int(m_cat.group(1)))
        out["approved_shield_rows"] = float(_safe_int(m_cat.group(2)))
        out["watch_fire_rows"] = float(_safe_int(m_cat.group(3)))
        out["watch_shield_rows"] = float(_safe_int(m_cat.group(4)))

    def _is_data_row(line: str) -> bool:
        s = line.strip()
        if not s.startswith("|"):
            return False
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            return False
        first = cells[0].replace(" ", "")
        if first in {"", "#"}:
            return False
        if re.fullmatch(r"[:\-]+", first):
            return False
        return bool(re.fullmatch(r"\d+", first))

    if any(not math.isfinite(out[k]) for k in out.keys()):
        counts = {
            "core_book_rows": 0,
            "tactical_book_rows": 0,
            "watch_book_rows": 0,
            "approved_fire_rows": 0,
            "approved_shield_rows": 0,
            "watch_fire_rows": 0,
            "watch_shield_rows": 0,
        }
        current_book = ""
        current_cat = ""
        for raw in text.splitlines():
            s = raw.strip()
            sl = s.lower()
            if s.startswith("## "):
                current_book = ""
                current_cat = ""
                continue
            if s.startswith("### "):
                current_cat = ""
                if sl.startswith("### core book"):
                    current_book = "core_book_rows"
                elif sl.startswith("### tactical book"):
                    current_book = "tactical_book_rows"
                elif sl.startswith("### watch book"):
                    current_book = "watch_book_rows"
                else:
                    current_book = ""
                continue
            if s.startswith("#### "):
                if "approved - fire" in sl:
                    current_cat = "approved_fire_rows"
                elif "approved - shield" in sl:
                    current_cat = "approved_shield_rows"
                elif "watch only - fire" in sl:
                    current_cat = "watch_fire_rows"
                elif "watch only - shield" in sl:
                    current_cat = "watch_shield_rows"
                else:
                    current_cat = ""
                continue
            if not _is_data_row(s):
                continue
            if current_book:
                counts[current_book] += 1
            if current_cat:
                counts[current_cat] += 1

        for k in out.keys():
            if not math.isfinite(out[k]):
                out[k] = float(counts[k])
    return out


def _extract_output_md_ranked_rows(path: Optional[Path]) -> pd.DataFrame:
    cols = [
        "rank_md",
        "ticker",
        "strategy",
        "track_md",
        "execution_book_md",
        "conviction_md",
        "expiry_md",
        "short_strike_md",
        "long_strike_md",
        "short_put_strike_md",
        "long_put_strike_md",
        "short_call_strike_md",
        "long_call_strike_md",
    ]
    if path is None or not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(columns=cols)
    def _split_cells(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]
    def _parse_strike_setup_cells(raw: str) -> Dict[str, float]:
        out = {
            "short_strike_md": float("nan"),
            "long_strike_md": float("nan"),
            "short_put_strike_md": float("nan"),
            "long_put_strike_md": float("nan"),
            "short_call_strike_md": float("nan"),
            "long_call_strike_md": float("nan"),
        }
        s = _normalize_text(raw)
        if not s:
            return out
        matches = re.findall(r"\b(Buy|Sell)\s+(\d+(?:\.\d+)?)\s*([CP])\b", s, flags=re.IGNORECASE)
        for side_raw, strike_raw, cp_raw in matches:
            side = str(side_raw).lower()
            cp = str(cp_raw).upper()
            strike = _safe_float(strike_raw)
            if not math.isfinite(strike):
                continue
            if cp == "P":
                if side == "sell":
                    out["short_put_strike_md"] = strike
                    out["short_strike_md"] = strike
                else:
                    out["long_put_strike_md"] = strike
                    out["long_strike_md"] = strike
            elif cp == "C":
                if side == "sell":
                    out["short_call_strike_md"] = strike
                    out["short_strike_md"] = strike
                else:
                    out["long_call_strike_md"] = strike
                    out["long_strike_md"] = strike
        return out
    def _heading_track(h3: str, h4: str) -> str:
        x = (h4 or h3 or "").upper()
        if "SHIELD" in x:
            return "SHIELD"
        if "FIRE" in x:
            return "FIRE"
        return ""
    def _heading_book(h3: str, h4: str) -> str:
        x = f"{h3} {h4}".lower()
        if "core book" in x:
            return "Core"
        if "tactical book" in x:
            return "Tactical"
        if "watch book" in x:
            return "Watch"
        if "watch only" in x:
            return "Watch"
        if "approved" in x:
            return "Tactical"
        return ""
    rows: List[Dict[str, Any]] = []
    h3 = ""
    h4 = ""
    header_cells: List[str] = []
    header_keys: List[str] = []
    in_table = False
    in_expert_table = False
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("## "):
            title = s[3:].strip().lower()
            if title.startswith("anu expert trade table"):
                in_expert_table = True
                h3 = ""
                h4 = ""
                in_table = False
                header_cells = []
                header_keys = []
                continue
            if in_expert_table and title.startswith("watch only reason tables"):
                break
            continue
        if not in_expert_table:
            continue
        if s.startswith("### "):
            h3 = s[4:].strip()
            h4 = ""
            in_table = False
            header_cells = []
            header_keys = []
            continue
        if s.startswith("#### "):
            h4 = s[5:].strip()
            in_table = False
            header_cells = []
            header_keys = []
            continue
        if not s.startswith("|"):
            if in_table:
                in_table = False
                header_cells = []
                header_keys = []
            continue
        cells = _split_cells(s)
        if not cells or len(cells) < 2:
            continue
        keys = [re.sub(r"\s+", " ", c.strip().lower()) for c in cells]
        if "ticker" in keys and "#" in keys:
            header_cells = cells
            header_keys = keys
            in_table = True
            continue
        if in_table and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
            continue
        if not in_table or not header_cells or not header_keys:
            continue
        n = min(len(cells), len(header_cells))
        row_map = {header_keys[i]: cells[i] for i in range(n)}
        rank_txt = str(row_map.get("#", "")).strip()
        if not re.fullmatch(r"\d+", rank_txt):
            continue
        ticker = _normalize_text(row_map.get("ticker"), upper=True)
        strategy = _normalize_text(row_map.get("strategy type"))
        conv_txt = _normalize_text(row_map.get("conviction %")).replace("%", "")
        conv = _safe_float(conv_txt)
        expiry = _normalize_text(row_map.get("expiry"))
        book_col = _normalize_text(row_map.get("execution book"))
        if not book_col:
            book_col = _heading_book(h3, h4)
        track = _heading_track(h3, h4)
        if not track:
            action = _normalize_text(row_map.get("action"), upper=True)
            if "SHIELD" in action:
                track = "SHIELD"
            elif "FIRE" in action:
                track = "FIRE"
        strike_map = _parse_strike_setup_cells(row_map.get("strike setup"))
        rows.append(
            {
                "rank_md": int(rank_txt),
                "ticker": ticker,
                "strategy": strategy,
                "track_md": track,
                "execution_book_md": book_col,
                "conviction_md": conv if math.isfinite(conv) else float("nan"),
                "expiry_md": expiry,
                "short_strike_md": strike_map["short_strike_md"],
                "long_strike_md": strike_map["long_strike_md"],
                "short_put_strike_md": strike_map["short_put_strike_md"],
                "long_put_strike_md": strike_map["long_put_strike_md"],
                "short_call_strike_md": strike_map["short_call_strike_md"],
                "long_call_strike_md": strike_map["long_call_strike_md"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out = (
        out.sort_values("rank_md", kind="mergesort")
        .drop_duplicates(subset=["rank_md"], keep="first")
        .reset_index(drop=True)
    )
    return out
def _select_source_rows_by_output_md(source: pd.DataFrame, md_rows: pd.DataFrame) -> pd.DataFrame:
    if source.empty or md_rows.empty:
        return source
    work = source.copy()
    work["_row_id"] = np.arange(len(work))
    work["_k_ticker"] = work.get("ticker", "").astype(str).str.upper().str.strip()
    work["_k_strategy"] = work.get("strategy", "").astype(str).str.lower().str.strip()
    work["_k_track"] = work.get("track", "").astype(str).str.upper().str.strip()
    work["_k_expiry"] = work.get("expiry", "").astype(str).str.strip()
    work["_k_conv"] = pd.to_numeric(work.get("conviction"), errors="coerce")
    work["_k_final"] = _to_bool_series(work.get("is_final_live_valid", pd.Series([False] * len(work))))
    work["_k_gate"] = _to_bool_series(work.get("gate_pass_live", pd.Series([False] * len(work))))
    for col in (
        "short_strike",
        "long_strike",
        "short_put_strike",
        "long_put_strike",
        "short_call_strike",
        "long_call_strike",
    ):
        work[f"_k_{col}"] = pd.to_numeric(work.get(col), errors="coerce")
    opt_vals = (
        work["optimal_stage1"]
        if "optimal_stage1" in work.columns
        else pd.Series([""] * len(work), index=work.index, dtype="object")
    )
    opt_lower = opt_vals.astype(str).str.lower()
    work["_k_book_hint"] = np.where(opt_lower.str.contains("watch", na=False), "Watch", "")
    md = md_rows.copy()
    md["_k_ticker"] = md.get("ticker", "").astype(str).str.upper().str.strip()
    md["_k_strategy"] = md.get("strategy", "").astype(str).str.lower().str.strip()
    md["_k_track"] = md.get("track_md", "").astype(str).str.upper().str.strip()
    md["_k_expiry"] = md.get("expiry_md", "").astype(str).str.strip()
    md["_k_conv"] = pd.to_numeric(md.get("conviction_md"), errors="coerce")
    for col in (
        "short_strike_md",
        "long_strike_md",
        "short_put_strike_md",
        "long_put_strike_md",
        "short_call_strike_md",
        "long_call_strike_md",
    ):
        md[f"_k_{col}"] = pd.to_numeric(md.get(col), errors="coerce")
    md = (
        md.sort_values("rank_md", kind="mergesort")
        .drop_duplicates(subset=["rank_md"], keep="first")
        .reset_index(drop=True)
    )
    used: set[int] = set()
    picks: List[Dict[str, Any]] = []
    for _, mr in md.iterrows():
        cand = work[
            (work["_k_ticker"] == mr["_k_ticker"])
            & (work["_k_strategy"] == mr["_k_strategy"])
            & (~work["_row_id"].isin(used))
        ].copy()
        if cand.empty:
            continue
        track = str(mr["_k_track"] or "").strip()
        if track:
            cand_track = cand[cand["_k_track"] == track].copy()
            if not cand_track.empty:
                cand = cand_track
        expiry = str(mr["_k_expiry"] or "").strip()
        if expiry:
            cand_exp = cand[cand["_k_expiry"] == expiry].copy()
            if not cand_exp.empty:
                cand = cand_exp
        book = _normalize_text(mr.get("execution_book_md"))
        if book.lower() == "watch":
            cand_watch = cand[cand["_k_book_hint"] == "Watch"].copy()
            if not cand_watch.empty:
                cand = cand_watch
        strike_pairs = (
            ("_k_short_strike", "_k_short_strike_md"),
            ("_k_long_strike", "_k_long_strike_md"),
            ("_k_short_put_strike", "_k_short_put_strike_md"),
            ("_k_long_put_strike", "_k_long_put_strike_md"),
            ("_k_short_call_strike", "_k_short_call_strike_md"),
            ("_k_long_call_strike", "_k_long_call_strike_md"),
        )
        for src_col, md_col in strike_pairs:
            md_val = _safe_float(mr.get(md_col))
            if not math.isfinite(md_val):
                continue
            narrowed = cand[(cand[src_col] - md_val).abs() <= 0.05].copy()
            if not narrowed.empty:
                cand = narrowed
        if math.isfinite(_safe_float(mr["_k_conv"])):
            cand["_conv_dist"] = (cand["_k_conv"] - float(mr["_k_conv"])).abs()
        else:
            cand["_conv_dist"] = np.nan
        cand = cand.sort_values(
            ["_k_final", "_k_gate", "_conv_dist", "_k_conv"],
            ascending=[False, False, True, False],
        )
        pick = cand.iloc[0].copy()
        used.add(int(pick["_row_id"]))
        rec = pick.to_dict()
        rec["rank_md"] = int(_safe_int(mr.get("rank_md")))
        rec["track_md"] = str(mr.get("track_md", "") or "")
        rec["execution_book_md"] = str(mr.get("execution_book_md", "") or "")
        rec["expiry_md"] = str(mr.get("expiry_md", "") or "")
        picks.append(rec)
    if not picks:
        return source
    out = pd.DataFrame(picks)
    drop_cols = [c for c in out.columns if c.startswith("_k_") or c in {"_row_id", "_conv_dist"}]
    out = out.drop(columns=drop_cols, errors="ignore")
    out = out.sort_values("rank_md").reset_index(drop=True)
    return out
def _extract_daily_row(bundle: RunBundle) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    m = bundle.manifest
    counts = m.get("counts") if isinstance(m.get("counts"), dict) else {}
    settings = m.get("settings") if isinstance(m.get("settings"), dict) else {}
    live_df = _read_csv(bundle.artifacts.live_csv)
    final_df = _read_csv(bundle.artifacts.live_final_csv)
    setup_df = _read_csv(bundle.artifacts.setup_likelihood_csv)
    dropped_df = _read_csv(bundle.artifacts.dropped_csv)
    row: Dict[str, Any] = {
        "variant": bundle.variant,
        "trade_date": bundle.trade_date,
        "week_start": bundle.trade_date - pd.to_timedelta(bundle.trade_date.weekday(), unit="D"),
        "manifest_path": str(bundle.manifest_path),
        "run_dir": str(bundle.run_dir),
        "live_csv_path": str(bundle.artifacts.live_csv or ""),
        "live_final_csv_path": str(bundle.artifacts.live_final_csv or ""),
        "setup_likelihood_csv_path": str(bundle.artifacts.setup_likelihood_csv or ""),
        "dropped_csv_path": str(bundle.artifacts.dropped_csv or ""),
        "output_md_path": str(bundle.artifacts.output_md or ""),
        "approved_rows_manifest": _safe_int(counts.get("approved_rows")),
        "watch_rows_manifest": _safe_int(counts.get("watch_rows")),
        "final_output_rows_manifest": _safe_int(
            counts.get("final_output_rows", counts.get("rows_after_final_caps"))
        ),
        "stage1_shortlist_rows_manifest": _safe_int(counts.get("stage1_shortlist_rows")),
        "stage2_live_rows_manifest": _safe_int(counts.get("stage2_live_rows")),
        "top_trades_requested": _safe_int(settings.get("top_trades_requested")),
        "pretrade_caps_status": str(settings.get("pretrade_caps_status", "")),
        "pretrade_caps_error": str(settings.get("pretrade_caps_error", "")),
        "rows_live_csv": int(len(live_df)),
        "rows_live_final_csv": int(len(final_df)),
    }
    row.update(_parse_output_md_category_counts(bundle.artifacts.output_md))
    if row["final_output_rows_manifest"] > 0:
        row["approval_rate_manifest"] = row["approved_rows_manifest"] / row["final_output_rows_manifest"]
    else:
        row["approval_rate_manifest"] = float("nan")

    row["gate_pass_live_rate"] = _bool_rate(live_df, "gate_pass_live")
    row["entry_structure_ok_live_rate"] = _bool_rate(live_df, "entry_structure_ok_live")
    row["invalidation_breached_live_rate"] = _bool_rate(live_df, "invalidation_breached_live")
    row["final_live_valid_rate"] = _bool_rate(live_df, "is_final_live_valid")
    row["avg_conviction_live"] = _num_mean(live_df, "conviction")
    row["avg_conviction_final_live"] = _num_mean(final_df, "conviction")
    row["avg_live_max_profit"] = _num_mean(live_df, "live_max_profit")
    row["avg_live_max_loss"] = _num_mean(live_df, "live_max_loss")
    row["strategy_count_live"] = int(live_df["strategy"].nunique()) if "strategy" in live_df.columns else 0
    row["ticker_count_live"] = int(live_df["ticker"].nunique()) if "ticker" in live_df.columns else 0
    row["fire_share_live"] = float("nan")
    row["shield_share_live"] = float("nan")
    if "track" in live_df.columns and len(live_df) > 0:
        t = live_df["track"].astype(str).str.upper().str.strip()
        row["fire_share_live"] = float((t == "FIRE").mean())
        row["shield_share_live"] = float((t == "SHIELD").mean())

    core = row.get("core_book_rows", float("nan"))
    tactical = row.get("tactical_book_rows", float("nan"))
    watch = row.get("watch_book_rows", float("nan"))
    if all(math.isfinite(x) for x in (core, tactical, watch)):
        total = core + tactical + watch
        row["core_book_share"] = (core / total) if total > 0 else float("nan")
        row["tactical_book_share"] = (tactical / total) if total > 0 else float("nan")
        row["watch_book_share"] = (watch / total) if total > 0 else float("nan")
    else:
        row["core_book_share"] = float("nan")
        row["tactical_book_share"] = float("nan")
        row["watch_book_share"] = float("nan")

    approved_fire = row.get("approved_fire_rows", float("nan"))
    approved_shield = row.get("approved_shield_rows", float("nan"))
    if all(math.isfinite(x) for x in (approved_fire, approved_shield)):
        approved_total = approved_fire + approved_shield
        row["approved_fire_share"] = (
            approved_fire / approved_total if approved_total > 0 else float("nan")
        )
        row["approved_shield_share"] = (
            approved_shield / approved_total if approved_total > 0 else float("nan")
        )
    else:
        row["approved_fire_share"] = float("nan")
        row["approved_shield_share"] = float("nan")

    row["setup_rows"] = int(len(setup_df))
    if not setup_df.empty:
        verdict = setup_df.get("verdict", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
        row["setup_pass_rate"] = float((verdict == "PASS").mean()) if len(verdict) else float("nan")
        row["avg_hist_success_pct"] = _num_mean(setup_df, "hist_success_pct")
        row["avg_edge_pct"] = _num_mean(setup_df, "edge_pct")
    else:
        row["setup_pass_rate"] = float("nan")
        row["avg_hist_success_pct"] = float("nan")
        row["avg_edge_pct"] = float("nan")

    proxy = _compute_proxy_frame(live_df, setup_df)
    row["proxy_rows"] = int(len(proxy))
    row["proxy_expected_gross_profit"] = _num_sum(proxy, "expected_gross_profit")
    row["proxy_expected_gross_loss"] = _num_sum(proxy, "expected_gross_loss")
    row["proxy_expected_net"] = _num_sum(proxy, "expected_net")
    loss = row["proxy_expected_gross_loss"]
    gp = row["proxy_expected_gross_profit"]
    if math.isfinite(loss) and loss > 0 and math.isfinite(gp):
        row["proxy_pf"] = gp / loss
    else:
        row["proxy_pf"] = float("nan")

    top_drop_reason = ""
    top_drop_count = 0
    if not dropped_df.empty and "drop_reason" in dropped_df.columns:
        reason_counts = dropped_df["drop_reason"].astype(str).value_counts()
        if len(reason_counts) > 0:
            top_drop_reason = str(reason_counts.index[0])
            top_drop_count = int(reason_counts.iloc[0])
    row["top_drop_reason"] = top_drop_reason
    row["top_drop_reason_count"] = top_drop_count

    ticker_rows = _build_ticker_rows(bundle, live_df, final_df, setup_df)
    drop_rows = _build_drop_reason_rows(bundle, dropped_df)
    return row, ticker_rows, drop_rows


def _rolling_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df
    out = daily_df.sort_values(["variant", "trade_date"]).copy()
    metrics = [
        "approved_rows_manifest",
        "approval_rate_manifest",
        "setup_pass_rate",
        "avg_edge_pct",
        "proxy_expected_net",
        "proxy_pf",
    ]
    for col in metrics:
        if col not in out.columns:
            continue
        out[f"{col}_ma5"] = (
            out.groupby("variant")[col]
            .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        )
        out[f"{col}_ma10"] = (
            out.groupby("variant")[col]
            .transform(lambda s: s.rolling(window=10, min_periods=1).mean())
        )
    if "approved_rows_manifest" in out.columns:
        out["approved_rows_delta_dod"] = out.groupby("variant")["approved_rows_manifest"].diff()
    return out


def aggregate_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    numeric_fill = daily_df.copy()
    for c in (
        "approved_rows_manifest",
        "watch_rows_manifest",
        "final_output_rows_manifest",
        "proxy_expected_gross_profit",
        "proxy_expected_gross_loss",
        "proxy_expected_net",
        "core_book_rows",
        "tactical_book_rows",
        "watch_book_rows",
        "approved_fire_rows",
        "approved_shield_rows",
        "watch_fire_rows",
        "watch_shield_rows",
    ):
        if c in numeric_fill.columns:
            numeric_fill[c] = pd.to_numeric(numeric_fill[c], errors="coerce").fillna(0.0)
    grp = (
        numeric_fill.groupby(["variant", "week_start"], as_index=False)
        .agg(
            trading_days=("trade_date", "nunique"),
            runs=("manifest_path", "count"),
            approved_rows=("approved_rows_manifest", "sum"),
            watch_rows=("watch_rows_manifest", "sum"),
            final_output_rows=("final_output_rows_manifest", "sum"),
            avg_approved_per_day=("approved_rows_manifest", "mean"),
            avg_approval_rate=("approval_rate_manifest", "mean"),
            avg_gate_pass_live_rate=("gate_pass_live_rate", "mean"),
            avg_setup_pass_rate=("setup_pass_rate", "mean"),
            avg_edge_pct=("avg_edge_pct", "mean"),
            proxy_expected_gross_profit=("proxy_expected_gross_profit", "sum"),
            proxy_expected_gross_loss=("proxy_expected_gross_loss", "sum"),
            proxy_expected_net=("proxy_expected_net", "sum"),
            core_book_rows=("core_book_rows", "sum"),
            tactical_book_rows=("tactical_book_rows", "sum"),
            watch_book_rows=("watch_book_rows", "sum"),
            approved_fire_rows=("approved_fire_rows", "sum"),
            approved_shield_rows=("approved_shield_rows", "sum"),
            watch_fire_rows=("watch_fire_rows", "sum"),
            watch_shield_rows=("watch_shield_rows", "sum"),
        )
        .sort_values(["variant", "week_start"])
        .reset_index(drop=True)
    )
    grp["approval_rate"] = np.where(
        grp["final_output_rows"] > 0,
        grp["approved_rows"] / grp["final_output_rows"],
        np.nan,
    )
    grp["proxy_pf"] = np.where(
        grp["proxy_expected_gross_loss"] > 0,
        grp["proxy_expected_gross_profit"] / grp["proxy_expected_gross_loss"],
        np.nan,
    )
    book_total = grp["core_book_rows"] + grp["tactical_book_rows"] + grp["watch_book_rows"]
    grp["core_book_share"] = np.where(book_total > 0, grp["core_book_rows"] / book_total, np.nan)
    grp["tactical_book_share"] = np.where(
        book_total > 0, grp["tactical_book_rows"] / book_total, np.nan
    )
    grp["watch_book_share"] = np.where(book_total > 0, grp["watch_book_rows"] / book_total, np.nan)
    approved_total = grp["approved_fire_rows"] + grp["approved_shield_rows"]
    grp["approved_fire_share"] = np.where(
        approved_total > 0, grp["approved_fire_rows"] / approved_total, np.nan
    )
    grp["approved_shield_share"] = np.where(
        approved_total > 0, grp["approved_shield_rows"] / approved_total, np.nan
    )
    grp["approved_rows_wow"] = grp.groupby("variant")["approved_rows"].diff()
    grp["approval_rate_wow"] = grp.groupby("variant")["approval_rate"].diff()
    grp["proxy_pf_wow"] = grp.groupby("variant")["proxy_pf"].diff()
    grp["avg_edge_pct_wow"] = grp.groupby("variant")["avg_edge_pct"].diff()
    return grp


def _apply_date_filters(
    daily_df: pd.DataFrame,
    ticker_raw_df: pd.DataFrame,
    drop_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    lookback_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if daily_df.empty:
        return daily_df, ticker_raw_df, drop_df
    out_daily = daily_df.copy()
    out_ticker = ticker_raw_df.copy()
    out_drop = drop_df.copy()
    if end_date is None:
        end_date = out_daily["trade_date"].max()
    if lookback_days > 0 and end_date is not None:
        lb_start = end_date - pd.Timedelta(days=lookback_days - 1)
        start_date = lb_start if start_date is None else max(start_date, lb_start)
    if start_date is not None:
        out_daily = out_daily[out_daily["trade_date"] >= start_date].copy()
        if not out_ticker.empty:
            out_ticker = out_ticker[out_ticker["trade_date"] >= start_date].copy()
        if not out_drop.empty:
            out_drop = out_drop[out_drop["trade_date"] >= start_date].copy()
    if end_date is not None:
        out_daily = out_daily[out_daily["trade_date"] <= end_date].copy()
        if not out_ticker.empty:
            out_ticker = out_ticker[out_ticker["trade_date"] <= end_date].copy()
        if not out_drop.empty:
            out_drop = out_drop[out_drop["trade_date"] <= end_date].copy()
    return out_daily, out_ticker, out_drop


def aggregate_tickers(
    ticker_raw_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ticker_raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    raw = ticker_raw_df.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw["week_start"] = pd.to_datetime(raw["week_start"], errors="coerce")
    raw["conviction"] = pd.to_numeric(raw["conviction"], errors="coerce")
    raw["hist_success_pct"] = pd.to_numeric(raw["hist_success_pct"], errors="coerce")
    raw["edge_pct"] = pd.to_numeric(raw["edge_pct"], errors="coerce")
    raw["expected_net"] = pd.to_numeric(raw["expected_net"], errors="coerce")
    raw["live_max_profit"] = pd.to_numeric(raw["live_max_profit"], errors="coerce")
    raw["live_max_loss"] = pd.to_numeric(raw["live_max_loss"], errors="coerce")

    ticker_daily = (
        raw.groupby(["variant", "trade_date", "week_start", "ticker"], as_index=False)
        .agg(
            setups=("ticker", "size"),
            strategies=("strategy", "nunique"),
            avg_conviction=("conviction", "mean"),
            avg_hist_success_pct=("hist_success_pct", "mean"),
            avg_edge_pct=("edge_pct", "mean"),
            expected_net=("expected_net", "sum"),
            avg_live_max_profit=("live_max_profit", "mean"),
            avg_live_max_loss=("live_max_loss", "mean"),
        )
        .sort_values(["variant", "trade_date", "setups", "ticker"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )

    ticker_weekly = (
        ticker_daily.groupby(["variant", "week_start", "ticker"], as_index=False)
        .agg(
            days_present=("trade_date", "nunique"),
            setups=("setups", "sum"),
            avg_conviction=("avg_conviction", "mean"),
            avg_edge_pct=("avg_edge_pct", "mean"),
            expected_net=("expected_net", "sum"),
        )
        .sort_values(["variant", "week_start", "setups", "ticker"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )

    variant_days = (
        daily_df.groupby("variant")["trade_date"].nunique().to_dict() if not daily_df.empty else {}
    )
    rows: List[Dict[str, Any]] = []
    for (variant, ticker), grp in ticker_daily.groupby(["variant", "ticker"], dropna=False):
        g = grp.sort_values("trade_date")
        rows.append(
            {
                "variant": variant,
                "ticker": ticker,
                "days_present": int(g["trade_date"].nunique()),
                "total_setups": int(g["setups"].sum()),
                "avg_conviction": float(g["avg_conviction"].mean()),
                "avg_edge_pct": float(g["avg_edge_pct"].mean()),
                "total_expected_net": float(g["expected_net"].sum()),
                "first_seen": g["trade_date"].min(),
                "last_seen": g["trade_date"].max(),
                "conviction_slope": _linear_slope(g["avg_conviction"].tolist()),
                "edge_slope": _linear_slope(g["avg_edge_pct"].tolist()),
                "presence_rate": float(g["trade_date"].nunique())
                / float(max(1, variant_days.get(variant, 0))),
            }
        )
    ticker_persistence = pd.DataFrame(rows)
    if not ticker_persistence.empty:
        ticker_persistence = ticker_persistence.sort_values(
            ["days_present", "total_setups", "presence_rate", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return ticker_daily, ticker_weekly, ticker_persistence


def _pick_recommendation_variant(
    ticker_raw_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    recommendation_variant: str,
) -> str:
    if ticker_raw_df.empty:
        return ""
    variants = sorted(
        {
            _normalize_variant(v)
            for v in ticker_raw_df.get("variant", pd.Series(dtype=str)).astype(str).tolist()
            if str(v).strip()
        }
    )
    if not variants:
        return ""
    wanted_txt = str(recommendation_variant or "").strip()
    wanted = _normalize_variant(wanted_txt) if wanted_txt else ""
    if wanted:
        if wanted not in variants:
            raise RuntimeError(
                f"Requested recommendation variant '{recommendation_variant}' is not present after filters."
            )
        return wanted
    if len(variants) == 1:
        return variants[0]

    if not weekly_df.empty and "variant" in weekly_df.columns:
        wk = weekly_df.copy()
        wk["week_start"] = pd.to_datetime(wk.get("week_start"), errors="coerce")
        wk = wk[wk["week_start"].notna()].copy()
        if not wk.empty:
            latest_week = wk["week_start"].max()
            wk = wk[wk["week_start"] == latest_week].copy()
            wk = wk[wk["variant"].map(_normalize_variant).isin(variants)].copy()
            if not wk.empty:
                wk["approval_rate"] = pd.to_numeric(wk.get("approval_rate"), errors="coerce").fillna(0.0)
                wk["proxy_pf"] = pd.to_numeric(wk.get("proxy_pf"), errors="coerce").fillna(0.0)
                wk["avg_edge_pct"] = pd.to_numeric(wk.get("avg_edge_pct"), errors="coerce").fillna(0.0)
                wk["approved_rows"] = pd.to_numeric(wk.get("approved_rows"), errors="coerce").fillna(0.0)
                wk["trading_days"] = pd.to_numeric(wk.get("trading_days"), errors="coerce").fillna(0.0)
                wk["variant_pick_score"] = (
                    (wk["approval_rate"].clip(lower=0.0, upper=1.0) * 0.35)
                    + (wk["proxy_pf"].clip(lower=0.0, upper=3.0) / 3.0 * 0.35)
                    + (((wk["avg_edge_pct"].clip(lower=-10.0, upper=20.0) + 10.0) / 30.0) * 0.15)
                    + ((wk["approved_rows"].clip(lower=0.0, upper=20.0) / 20.0) * 0.10)
                    + ((wk["trading_days"].clip(lower=0.0, upper=5.0) / 5.0) * 0.05)
                )
                wk_multi_day = wk[wk["trading_days"] >= 2].copy()
                if not wk_multi_day.empty:
                    wk = wk_multi_day
                wk = wk.sort_values(
                    ["variant_pick_score", "approved_rows", "trading_days"],
                    ascending=[False, False, False],
                )
                if not wk.empty:
                    return _normalize_variant(str(wk.iloc[0]["variant"]))

    if not daily_df.empty and "variant" in daily_df.columns:
        dd = daily_df.copy()
        dd = dd[dd["variant"].map(_normalize_variant).isin(variants)].copy()
        if not dd.empty:
            dd["approval_rate_manifest"] = pd.to_numeric(
                dd.get("approval_rate_manifest"), errors="coerce"
            ).fillna(0.0)
            dd["proxy_pf"] = pd.to_numeric(dd.get("proxy_pf"), errors="coerce").fillna(0.0)
            by_variant = (
                dd.groupby("variant", as_index=False)
                .agg(
                    avg_approval_rate=("approval_rate_manifest", "mean"),
                    avg_proxy_pf=("proxy_pf", "mean"),
                    days=("trade_date", "nunique"),
                )
                .sort_values(["avg_proxy_pf", "avg_approval_rate", "days"], ascending=[False, False, False])
            )
            if not by_variant.empty:
                return _normalize_variant(str(by_variant.iloc[0]["variant"]))

    counts = (
        ticker_raw_df.assign(_v=ticker_raw_df.get("variant", pd.Series(dtype=str)).map(_normalize_variant))
        .groupby("_v", as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    if not counts.empty:
        return str(counts.iloc[0]["_v"])
    return variants[0]


def build_trend_recommendations(
    ticker_raw_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    run_inventory: pd.DataFrame,
    *,
    recommendation_variant: str,
    recommendation_top_n: int,
    recommendation_min_days: int,
    recommendation_min_proxy_pf: float,
    recommendation_min_shield: int,
    recommendation_live_actionable_only: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if ticker_raw_df.empty:
        return pd.DataFrame(), {"selected_variant": ""}

    selected_variant = _pick_recommendation_variant(
        ticker_raw_df=ticker_raw_df,
        daily_df=daily_df,
        weekly_df=weekly_df,
        recommendation_variant=recommendation_variant,
    )
    raw = ticker_raw_df[ticker_raw_df["variant"].map(_normalize_variant) == selected_variant].copy()
    if raw.empty:
        return pd.DataFrame(), {"selected_variant": selected_variant}

    raw["trade_date"] = pd.to_datetime(raw.get("trade_date"), errors="coerce")
    raw = raw[raw["trade_date"].notna()].copy()
    if raw.empty:
        return pd.DataFrame(), {"selected_variant": selected_variant}

    for c in (
        "conviction",
        "hist_success_pct",
        "edge_pct",
        "expected_net",
        "live_max_profit",
        "live_max_loss",
    ):
        raw[c] = pd.to_numeric(raw.get(c), errors="coerce")
    for c in ("core_ok_stage1", "gate_pass_live", "is_final_live_valid"):
        raw[c] = _to_bool_series(raw[c]) if c in raw.columns else False
    for c in ("confidence_tier", "optimal_stage1", "thesis", "invalidation", "output_md_path"):
        raw[c] = raw[c].map(_normalize_text) if c in raw.columns else ""
    raw["track"] = raw.get("track", "").astype(str).str.strip().str.upper()
    raw.loc[~raw["track"].isin(["FIRE", "SHIELD"]), "track"] = "UNKNOWN"

    if "execution_book_inferred" in raw.columns:
        raw["execution_book_inferred"] = raw["execution_book_inferred"].map(_normalize_text)
    else:
        raw["execution_book_inferred"] = ""
    missing_book = raw["execution_book_inferred"].eq("")
    if missing_book.any():
        opt_lower = raw["optimal_stage1"].astype(str).str.lower()
        raw.loc[missing_book, "execution_book_inferred"] = np.where(
            opt_lower[missing_book].str.contains("watch", na=False),
            "Watch",
            np.where(raw.loc[missing_book, "core_ok_stage1"], "Core", "Tactical"),
        )

    # Normalize strategy case to prevent cross-day case drift from splitting groups
    raw["strategy"] = raw["strategy"].astype(str).str.strip()

    grouped = (
        raw.groupby(["ticker", "strategy", "track"], as_index=False)
        .agg(
            days_present=("trade_date", "nunique"),
            appearances=("trade_date", "size"),
            avg_conviction=("conviction", "mean"),
            avg_hist_success_pct=("hist_success_pct", "mean"),
            avg_edge_pct=("edge_pct", "mean"),
            total_expected_net=("expected_net", "sum"),
            avg_live_max_profit=("live_max_profit", "mean"),
            avg_live_max_loss=("live_max_loss", "mean"),
            gate_pass_rate=("gate_pass_live", "mean"),
            final_valid_rate=("is_final_live_valid", "mean"),
            core_rows=("execution_book_inferred", lambda s: int((s == "Core").sum())),
            tactical_rows=("execution_book_inferred", lambda s: int((s == "Tactical").sum())),
            watch_rows=("execution_book_inferred", lambda s: int((s == "Watch").sum())),
            latest_trade_date=("trade_date", "max"),
        )
        .reset_index(drop=True)
    )
    p_est = (pd.to_numeric(grouped["avg_hist_success_pct"], errors="coerce") / 100.0).clip(lower=0.0, upper=1.0)
    gp_est = p_est * pd.to_numeric(grouped["avg_live_max_profit"], errors="coerce")
    gl_est = (1.0 - p_est) * pd.to_numeric(grouped["avg_live_max_loss"], errors="coerce")
    grouped["proxy_pf_est"] = np.where(gl_est > 0, gp_est / gl_est, np.nan)

    latest_ranked = raw.copy()
    latest_ranked["_priority_live"] = (
        latest_ranked["is_final_live_valid"].astype(int) * 100
        + latest_ranked["gate_pass_live"].astype(int) * 10
    )
    latest_ranked = latest_ranked.sort_values(
        ["trade_date", "_priority_live", "conviction"],
        ascending=[False, False, False],
    )
    latest_per_key = (
        latest_ranked.drop_duplicates(subset=["ticker", "strategy", "track"], keep="first")
        .loc[
            :,
            [
                "ticker",
                "strategy",
                "track",
                "trade_date",
                "expiry",
                "net_type",
                "entry_gate",
                "short_leg",
                "long_leg",
                "short_put_leg",
                "long_put_leg",
                "short_call_leg",
                "long_call_leg",
                "live_net_mark",
                "live_net_bid_ask",
                "live_max_profit",
                "live_max_loss",
                "spot_live_last",
                "gate_pass_live",
                "is_final_live_valid",
                "confidence_tier",
                "optimal_stage1",
                "thesis",
                "invalidation",
                "output_md_path",
            ],
        ]
        .rename(columns={"trade_date": "snapshot_trade_date"})
    )
    grouped = grouped.merge(
        latest_per_key,
        on=["ticker", "strategy", "track"],
        how="left",
    )

    live_actionable_only = bool(recommendation_live_actionable_only)
    live_actionable_rows_before = int(len(grouped))
    if live_actionable_only:
        gate_latest = _to_bool_series(
            grouped.get("gate_pass_live", pd.Series([False] * len(grouped), index=grouped.index))
        )
        final_latest = _to_bool_series(
            grouped.get("is_final_live_valid", pd.Series([False] * len(grouped), index=grouped.index))
        )
        grouped = grouped[gate_latest & final_latest].copy()
    live_actionable_rows_after = int(len(grouped))

    if grouped.empty:
        return pd.DataFrame(), {
            "selected_variant": selected_variant,
            "trading_days": int(raw["trade_date"].nunique()) if not raw.empty else 0,
            "rows_considered": int(len(raw)),
            "date_min": str(raw["trade_date"].min().date()) if not raw.empty else "",
            "date_max": str(raw["trade_date"].max().date()) if not raw.empty else "",
            "min_proxy_pf": max(0.0, float(recommendation_min_proxy_pf)),
            "proxy_pf_filter_applied": False,
            "proxy_pf_rows_before_filter": 0,
            "proxy_pf_rows_after_filter": 0,
            "min_shield": max(0, int(recommendation_min_shield)),
            "shield_rows_selected": 0,
            "live_actionable_only": live_actionable_only,
            "live_actionable_rows_before_filter": live_actionable_rows_before,
            "live_actionable_rows_after_filter": live_actionable_rows_after,
            "live_actionable_filter_applied": live_actionable_only,
        }

    variant_daily = daily_df[daily_df["variant"].map(_normalize_variant) == selected_variant].copy()
    variant_days = int(variant_daily["trade_date"].nunique()) if not variant_daily.empty else int(raw["trade_date"].nunique())
    grouped["presence_rate"] = np.where(
        variant_days > 0,
        grouped["days_present"] / float(variant_days),
        np.nan,
    )
    grouped["watch_share"] = np.where(
        grouped["appearances"] > 0,
        grouped["watch_rows"] / grouped["appearances"],
        np.nan,
    )
    grouped["core_share"] = np.where(
        grouped["appearances"] > 0,
        grouped["core_rows"] / grouped["appearances"],
        np.nan,
    )
    grouped["tactical_share"] = np.where(
        grouped["appearances"] > 0,
        grouped["tactical_rows"] / grouped["appearances"],
        np.nan,
    )

    min_pf = max(0.0, float(recommendation_min_proxy_pf))
    pf_filter_applied = False
    pf_rows_before = int(len(grouped))
    if min_pf > 0.0:
        pf_filtered = grouped[grouped["proxy_pf_est"].fillna(0.0) >= min_pf].copy()
        if not pf_filtered.empty:
            grouped = pf_filtered
            pf_filter_applied = True

    min_days = max(1, int(recommendation_min_days))
    grouped["book_recommendation"] = "Watch"
    tactical_mask = (
        (grouped["avg_conviction"] >= 65.0)
        & (grouped["avg_edge_pct"] >= 4.0)
        & (grouped["watch_share"] <= 0.50)
    )
    grouped.loc[tactical_mask, "book_recommendation"] = "Tactical"
    core_mask = (
        (grouped["days_present"] >= min_days)
        & (grouped["avg_conviction"] >= 78.0)
        & (grouped["avg_edge_pct"] >= 8.0)
        & (grouped["watch_share"] <= 0.25)
    )
    grouped.loc[core_mask, "book_recommendation"] = "Core"
    grouped.loc[grouped["watch_share"] >= 0.80, "book_recommendation"] = "Watch"

    conv_norm = (grouped["avg_conviction"] / 100.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    edge_norm = ((grouped["avg_edge_pct"] + 10.0) / 30.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    persistence_norm = grouped["presence_rate"].clip(lower=0.0, upper=1.0).fillna(0.0)
    quality_norm = (
        (0.4 * grouped["gate_pass_rate"].fillna(0.0))
        + (0.4 * grouped["final_valid_rate"].fillna(0.0))
        + (0.2 * (1.0 - grouped["watch_share"].fillna(0.0).clip(lower=0.0, upper=1.0)))
    ).clip(lower=0.0, upper=1.0)
    exp = grouped["total_expected_net"].fillna(0.0)
    if len(exp) >= 2 and float(exp.max()) > float(exp.min()):
        exp_norm = (exp - float(exp.min())) / float(exp.max() - exp.min())
    else:
        exp_norm = pd.Series([0.5] * len(grouped), index=grouped.index, dtype="float64")
    grouped["trend_score"] = (
        (0.30 * persistence_norm)
        + (0.25 * conv_norm)
        + (0.20 * edge_norm)
        + (0.15 * quality_norm)
        + (0.10 * exp_norm)
    )
    # ── FIX T7: staleness, invalidation, and direction-conflict filters ──────
    #
    # Problem: the pipeline groups by [ticker, strategy, track], so a ticker
    # can have both "Bull Call Debit" (from 2 weeks ago when it was bullish)
    # and "Bear Put Debit" (from today when it turned bearish).  The old
    # bullish group has more persistence days => higher trend_score => gets
    # recommended even though the stock already blew past its invalidation.
    #
    # Fix 1 – snapshot staleness: drop rows whose latest snapshot is older
    #         than 5 trading days (7 calendar days) from the window end.
    # Fix 2 – invalidation breach: parse ALL invalidation conditions and drop
    #         rows where spot_live_last violates ANY of them.
    # Fix 3 – direction conflict: if a ticker has both bullish and bearish
    #         setups, keep only the direction from the most recent day,
    #         but only if the new direction has non-negative avg edge.

    _SNAPSHOT_MAX_CALENDAR_DAYS = 7  # 5 trading days ~ 7 calendar days

    staleness_rows_before = int(len(grouped))
    if not grouped.empty and "snapshot_trade_date" in grouped.columns:
        window_end = pd.to_datetime(grouped["snapshot_trade_date"], errors="coerce").max()
        if pd.notna(window_end):
            cutoff = window_end - pd.Timedelta(days=_SNAPSHOT_MAX_CALENDAR_DAYS)
            snap_ts = pd.to_datetime(grouped["snapshot_trade_date"], errors="coerce")
            # Preserve rows with NaT snapshot dates (don't silently drop)
            grouped = grouped[(snap_ts >= cutoff) | snap_ts.isna()].copy()
    staleness_rows_after = int(len(grouped))

    import re as _re
    _INVALIDATION_RE = _re.compile(
        r"close\s*(<=|>=|<|>)\s*([\d]+(?:\.\d+)?)", _re.IGNORECASE
    )

    def _parse_invalidation_breach(row: pd.Series) -> bool:
        """Return True if spot breaches ANY invalidation condition."""
        inv = str(row.get("invalidation", "") or "")
        spot = pd.to_numeric(row.get("spot_live_last"), errors="coerce")
        if pd.isna(spot) or not inv:
            return False
        # Find ALL conditions (handles IC dual: "close < 280 or close > 320")
        for m in _INVALIDATION_RE.finditer(inv):
            op, level = m.group(1), float(m.group(2))
            if op == "<" and spot < level:
                return True
            if op == "<=" and spot <= level:
                return True
            if op == ">" and spot > level:
                return True
            if op == ">=" and spot >= level:
                return True
        return False

    inval_rows_before = int(len(grouped))
    if not grouped.empty:
        breach_mask = grouped.apply(_parse_invalidation_breach, axis=1)
        grouped = grouped[~breach_mask].copy()
    inval_rows_after = int(len(grouped))

    def _infer_direction(strategy: str) -> str:
        s = str(strategy).lower()
        if "bear" in s:
            return "bearish"
        if "bull" in s:
            return "bullish"
        return "neutral"

    direction_rows_before = int(len(grouped))
    if not grouped.empty:
        grouped["_direction"] = grouped["strategy"].map(_infer_direction)
        # For each ticker, find which direction appeared on the most recent day
        # Break ties by conviction (higher conviction = more reliable signal)
        ticker_latest_dir = (
            grouped.sort_values(
                ["latest_trade_date", "avg_conviction"],
                ascending=[False, False],
            )
            .drop_duplicates(subset=["ticker"], keep="first")
            [["ticker", "_direction", "avg_edge_pct"]]
            .rename(columns={
                "_direction": "_latest_direction",
                "avg_edge_pct": "_latest_edge",
            })
        )
        grouped = grouped.merge(ticker_latest_dir, on="ticker", how="left")

        # Only trust a direction flip if the new direction has non-negative edge.
        # If the latest direction has negative edge, fall back to allowing both
        # directions so we don't throw away strong bullish history for a weak
        # bearish blip (or vice versa).
        direction_ok = (
            (grouped["_direction"] == grouped["_latest_direction"])
            | (grouped["_direction"] == "neutral")
            | (grouped["_latest_direction"] == "neutral")
            | (grouped["_latest_edge"] < 0)  # untrusted flip: keep all
        )
        grouped = grouped[direction_ok].copy()
        grouped.drop(
            columns=["_direction", "_latest_direction", "_latest_edge"],
            inplace=True, errors="ignore",
        )
    direction_rows_after = int(len(grouped))

    # Recompute book_recommendation AFTER T7 filters so labels reflect
    # the filtered dataset, not the pre-filter persistence counts.
    grouped["book_recommendation"] = "Watch"
    tactical_mask_post = (
        (grouped["avg_conviction"] >= 65.0)
        & (grouped["avg_edge_pct"] >= 4.0)
        & (grouped["watch_share"] <= 0.50)
    )
    grouped.loc[tactical_mask_post, "book_recommendation"] = "Tactical"
    core_mask_post = (
        (grouped["days_present"] >= min_days)
        & (grouped["avg_conviction"] >= 78.0)
        & (grouped["avg_edge_pct"] >= 8.0)
        & (grouped["watch_share"] <= 0.25)
    )
    grouped.loc[core_mask_post, "book_recommendation"] = "Core"
    grouped.loc[grouped["watch_share"] >= 0.80, "book_recommendation"] = "Watch"
    grouped["book_priority"] = grouped["book_recommendation"].map(
        {"Core": 0, "Tactical": 1, "Watch": 2}
    ).fillna(3)
    # ── END FIX T7 ───────────────────────────────────────────────────────────

    ordered = grouped.sort_values(
        ["book_priority", "trend_score", "days_present", "avg_conviction", "avg_edge_pct"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    keep_rows: List[Dict[str, Any]] = []
    per_ticker_counts: Dict[str, int] = {}
    max_per_ticker = 2
    top_n = max(1, int(recommendation_top_n))
    min_shield = max(0, int(recommendation_min_shield))

    def _try_add_row(r: pd.Series) -> bool:
        ticker = str(r.get("ticker", "")).strip().upper()
        if not ticker:
            return False
        if per_ticker_counts.get(ticker, 0) >= max_per_ticker:
            return False
        keep_rows.append(r.to_dict())
        per_ticker_counts[ticker] = per_ticker_counts.get(ticker, 0) + 1
        return True

    shield_added = 0
    if min_shield > 0:
        for _, r in ordered.iterrows():
            if str(r.get("track", "")).upper() != "SHIELD":
                continue
            if _try_add_row(r):
                shield_added += 1
            if shield_added >= min_shield or len(keep_rows) >= top_n:
                break

    chosen_keys = {
        (
            str(r.get("ticker", "")).strip().upper(),
            str(r.get("strategy", "")).strip().lower(),
            str(r.get("track", "")).strip().upper(),
            str(r.get("expiry", "")).strip(),
            str(r.get("entry_gate", "")).strip(),
        )
        for r in keep_rows
    }
    for _, r in ordered.iterrows():
        if len(keep_rows) >= top_n:
            break
        k = (
            str(r.get("ticker", "")).strip().upper(),
            str(r.get("strategy", "")).strip().lower(),
            str(r.get("track", "")).strip().upper(),
            str(r.get("expiry", "")).strip(),
            str(r.get("entry_gate", "")).strip(),
        )
        if k in chosen_keys:
            continue
        if _try_add_row(r):
            chosen_keys.add(k)
        if len(keep_rows) >= top_n:
            break
    rec_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    # Re-sort so Core > Tactical > Watch, with trend_score within each tier.
    # SHIELD quota items were pre-inserted but should appear in their correct
    # book tier, not forced to the top.
    if not rec_df.empty and "book_recommendation" in rec_df.columns:
        rec_df["_bp"] = rec_df["book_recommendation"].map(
            {"Core": 0, "Tactical": 1, "Watch": 2}
        ).fillna(3)
        rec_df = rec_df.sort_values(
            ["_bp", "trend_score"],
            ascending=[True, False],
        ).reset_index(drop=True)
        rec_df.drop(columns=["_bp"], inplace=True, errors="ignore")
    if rec_df.empty:
        return rec_df, {
            "selected_variant": selected_variant,
            "min_proxy_pf": min_pf,
            "proxy_pf_filter_applied": pf_filter_applied,
            "proxy_pf_rows_before_filter": pf_rows_before,
            "proxy_pf_rows_after_filter": int(len(grouped)),
            "min_shield": min_shield,
            "shield_rows_selected": 0,
            "live_actionable_only": live_actionable_only,
            "live_actionable_rows_before_filter": live_actionable_rows_before,
            "live_actionable_rows_after_filter": live_actionable_rows_after,
            "live_actionable_filter_applied": live_actionable_only,
        }

    rec_df.insert(0, "rank", np.arange(1, len(rec_df) + 1))
    book_icon = {"Core": "🏛️", "Tactical": "⚔️", "Watch": "👀"}
    track_icon = {"FIRE": "🔥", "SHIELD": "🛡️", "UNKNOWN": "❓"}
    rec_df["book_label"] = rec_df["book_recommendation"].map(
        lambda x: f"{book_icon.get(str(x), '')} {str(x)}".strip()
    )
    rec_df["track_label"] = rec_df["track"].map(lambda x: f"{track_icon.get(str(x), '')} {str(x)}".strip())

    latest_sources: List[str] = []
    if not run_inventory.empty and "variant" in run_inventory.columns:
        inv = run_inventory[run_inventory["variant"].map(_normalize_variant) == selected_variant].copy()
        if not inv.empty and "output_md" in inv.columns:
            inv = inv.sort_values(["trade_date", "variant"], ascending=[False, True])
            for txt in inv["output_md"].astype(str).tolist():
                p = Path(txt)
                if not txt.strip() or not p.exists():
                    continue
                px = str(p.resolve())
                if px not in latest_sources:
                    latest_sources.append(px)
                if len(latest_sources) >= 3:
                    break
    if not latest_sources and "output_md_path" in raw.columns:
        for txt in raw["output_md_path"].astype(str).tolist():
            p = Path(txt)
            if not txt.strip() or not p.exists():
                continue
            px = str(p.resolve())
            if px not in latest_sources:
                latest_sources.append(px)
            if len(latest_sources) >= 3:
                break

    meta: Dict[str, Any] = {
        "selected_variant": selected_variant,
        "trading_days": variant_days,
        "rows_considered": int(len(raw)),
        "date_min": str(raw["trade_date"].min().date()) if not raw.empty else "",
        "date_max": str(raw["trade_date"].max().date()) if not raw.empty else "",
        "latest_output_md_files": latest_sources,
        "min_proxy_pf": min_pf,
        "proxy_pf_filter_applied": pf_filter_applied,
        "proxy_pf_rows_before_filter": pf_rows_before,
        "proxy_pf_rows_after_filter": int(len(grouped)),
        "min_shield": min_shield,
        "shield_rows_selected": int((rec_df.get("track", pd.Series(dtype=str)) == "SHIELD").sum()),
        "live_actionable_only": live_actionable_only,
        "live_actionable_rows_before_filter": live_actionable_rows_before,
        "live_actionable_rows_after_filter": live_actionable_rows_after,
        "live_actionable_filter_applied": live_actionable_only,
    }
    if not variant_daily.empty:
        ar = pd.to_numeric(variant_daily.get("approval_rate_manifest"), errors="coerce")
        pf = pd.to_numeric(variant_daily.get("proxy_pf"), errors="coerce")
        meta["avg_approval_rate"] = float(ar.mean()) if ar.notna().any() else float("nan")
        meta["avg_proxy_pf"] = float(pf.mean()) if pf.notna().any() else float("nan")
        meta["core_rows_sum"] = int(pd.to_numeric(variant_daily.get("core_book_rows"), errors="coerce").fillna(0).sum())
        meta["tactical_rows_sum"] = int(
            pd.to_numeric(variant_daily.get("tactical_book_rows"), errors="coerce").fillna(0).sum()
        )
        meta["watch_rows_sum"] = int(pd.to_numeric(variant_daily.get("watch_book_rows"), errors="coerce").fillna(0).sum())
        meta["approved_fire_rows_sum"] = int(
            pd.to_numeric(variant_daily.get("approved_fire_rows"), errors="coerce").fillna(0).sum()
        )
        meta["approved_shield_rows_sum"] = int(
            pd.to_numeric(variant_daily.get("approved_shield_rows"), errors="coerce").fillna(0).sum()
        )
    return rec_df, meta


def build_final_recommendations_markdown(
    out_dir: Path,
    recommendation_df: pd.DataFrame,
    recommendation_meta: Dict[str, Any],
) -> str:
    selected_variant = str(recommendation_meta.get("selected_variant", "") or "")
    date_min = str(recommendation_meta.get("date_min", "") or "")
    date_max = str(recommendation_meta.get("date_max", "") or "")
    trading_days = int(recommendation_meta.get("trading_days", 0) or 0)

    lines: List[str] = ["# Final Trade Recommendations From Trends", ""]
    lines.append(f"- Date range used: **{date_min} to {date_max}**")
    lines.append(f"- Variant selected: **{selected_variant or 'n/a'}**")
    lines.append(f"- Trading days analyzed: **{trading_days}**")
    lines.append("")

    lines.append("## Open In VS Code (No Browser Links)")
    lines.append(f"- code \"{out_dir / 'final_trade_recommendations_from_trends.md'}\"")
    for src in recommendation_meta.get("latest_output_md_files", []):
        lines.append(f"- code \"{src}\"")
    lines.append("")

    lines.append("## Trend Snapshot")
    avg_approval = _safe_float(recommendation_meta.get("avg_approval_rate"))
    avg_pf = _safe_float(recommendation_meta.get("avg_proxy_pf"))
    min_proxy_pf = _safe_float(recommendation_meta.get("min_proxy_pf"))
    pf_filter_applied = bool(recommendation_meta.get("proxy_pf_filter_applied", False))
    if math.isfinite(avg_approval):
        lines.append(f"- Avg approval rate in window: **{avg_approval:.1%}**")
    if math.isfinite(avg_pf):
        lines.append(f"- Avg proxy PF in window: **{avg_pf:.2f}**")
    if math.isfinite(min_proxy_pf) and min_proxy_pf > 0:
        pf_rows_before = int(_safe_int(recommendation_meta.get("proxy_pf_rows_before_filter")))
        pf_rows_after = int(_safe_int(recommendation_meta.get("proxy_pf_rows_after_filter")))
        status = "applied" if pf_filter_applied else "not applied (would empty set)"
        lines.append(
            f"- Proxy PF filter: **>= {min_proxy_pf:.2f}** ({status}; rows {pf_rows_before} -> {pf_rows_after})"
        )
    live_actionable_only = bool(recommendation_meta.get("live_actionable_only", False))
    if live_actionable_only:
        la_before = int(_safe_int(recommendation_meta.get("live_actionable_rows_before_filter")))
        la_after = int(_safe_int(recommendation_meta.get("live_actionable_rows_after_filter")))
        lines.append(
            f"- Live actionable filter: **enabled** (latest snapshot requires gate PASS + final valid; rows {la_before} -> {la_after})"
        )
    min_shield = int(_safe_int(recommendation_meta.get("min_shield")))
    if min_shield > 0:
        shield_selected = int(_safe_int(recommendation_meta.get("shield_rows_selected")))
        lines.append(f"- Shield quota: **min {min_shield}** (selected {shield_selected})")
    core_rows = int(recommendation_meta.get("core_rows_sum", 0) or 0)
    tactical_rows = int(recommendation_meta.get("tactical_rows_sum", 0) or 0)
    watch_rows = int(recommendation_meta.get("watch_rows_sum", 0) or 0)
    if (core_rows + tactical_rows + watch_rows) > 0:
        lines.append(
            f"- Execution mix across window: Core={core_rows}, Tactical={tactical_rows}, Watch={watch_rows}"
        )
    approved_fire = int(recommendation_meta.get("approved_fire_rows_sum", 0) or 0)
    approved_shield = int(recommendation_meta.get("approved_shield_rows_sum", 0) or 0)
    if (approved_fire + approved_shield) > 0:
        lines.append(
            f"- Approved category mix: FIRE={approved_fire}, SHIELD={approved_shield}"
        )
    lines.append("")

    lines.append("## Actionable Trade Sheet")
    if recommendation_df.empty:
        lines.append("_none_")
        lines.append("")
        return "\n".join(lines)

    def _fmt_money(v: Any) -> str:
        x = _safe_float(v)
        return f"${x:,.2f}" if math.isfinite(x) else ""

    def _fmt_pct(v: Any) -> str:
        x = _safe_float(v)
        return f"{x:.2f}%" if math.isfinite(x) else ""

    def _parse_leg_strike(leg: str) -> str:
        s = str(leg or "").strip()
        if not s or s.lower() == "none":
            return ""
        m = re.search(r"([CP])(\d{8})$", s)
        if not m:
            return s
        cp = m.group(1)
        strike_raw = m.group(2)
        try:
            strike = int(strike_raw) / 1000.0
        except Exception:
            return s
        strike_txt = f"{strike:.2f}".rstrip("0").rstrip(".")
        return f"${strike_txt}{cp}"

    def _build_structure(r: Dict[str, Any]) -> str:
        spp = str(r.get("short_put_leg", "")).strip()
        lpp = str(r.get("long_put_leg", "")).strip()
        spc = str(r.get("short_call_leg", "")).strip()
        lpc = str(r.get("long_call_leg", "")).strip()
        sl = str(r.get("short_leg", "")).strip()
        ll = str(r.get("long_leg", "")).strip()
        spp_txt = _parse_leg_strike(spp)
        lpp_txt = _parse_leg_strike(lpp)
        spc_txt = _parse_leg_strike(spc)
        lpc_txt = _parse_leg_strike(lpc)
        sl_txt = _parse_leg_strike(sl)
        ll_txt = _parse_leg_strike(ll)
        if spp and lpp and spc and lpc:
            return f"Sell {spp_txt} / Buy {lpp_txt} + Sell {spc_txt} / Buy {lpc_txt}"
        if sl and ll:
            return f"Sell {sl_txt} / Buy {ll_txt}"
        if spp and lpp:
            return f"Sell {spp_txt} / Buy {lpp_txt}"
        if spc and lpc:
            return f"Sell {spc_txt} / Buy {lpc_txt}"
        return ""

    actionable_rows: List[Dict[str, Any]] = []
    for _, r in recommendation_df.iterrows():
        snapshot_date = pd.to_datetime(r.get("snapshot_trade_date"), errors="coerce")
        snapshot_txt = snapshot_date.date().isoformat() if pd.notna(snapshot_date) else ""
        source_path = str(r.get("output_md_path", "")).strip()
        gate_pass = str(r.get("gate_pass_live", "")).strip().lower() in BOOL_TRUE
        final_valid = str(r.get("is_final_live_valid", "")).strip().lower() in BOOL_TRUE
        actionable_rows.append(
            {
                "Rank": int(_safe_int(r.get("rank"))),
                "Book": str(r.get("book_label", "")),
                "Track": str(r.get("track_label", "")),
                "Ticker": str(r.get("ticker", "")),
                "Strategy": str(r.get("strategy", "")),
                "Expiry": str(r.get("expiry", "")),
                "Structure": _build_structure(r),
                "Entry Gate": str(r.get("entry_gate", "")),
                "Live Mark": _fmt_money(r.get("live_net_mark")),
                "Gate Eval Px": _fmt_money(r.get("live_net_bid_ask")),
                "Max Profit": _fmt_money(r.get("live_max_profit")),
                "Max Loss": _fmt_money(r.get("live_max_loss")),
                "Gate Pass": "PASS" if gate_pass else "FAIL",
                "Final Valid": "YES" if final_valid else "NO",
                "Proxy PF Est": f"{_safe_float(r.get('proxy_pf_est')):.2f}" if math.isfinite(_safe_float(r.get("proxy_pf_est"))) else "",
                "Avg Edge %": _fmt_pct(r.get("avg_edge_pct")),
                "Invalidation": str(r.get("invalidation", "")),
                "Snapshot": snapshot_txt,
            }
        )
    lines.append(
        _render_markdown_table(
            actionable_rows,
            columns=[
                "Rank",
                "Book",
                "Track",
                "Ticker",
                "Strategy",
                "Expiry",
                "Structure",
                "Entry Gate",
                "Live Mark",
                "Gate Eval Px",
                "Max Profit",
                "Max Loss",
                "Gate Pass",
                "Final Valid",
                "Proxy PF Est",
                "Avg Edge %",
                "Invalidation",
                "Snapshot",
            ],
        )
    )
    lines.append("")
    lines.append("## Consolidated Recommendations (Ranking View)")

    table_rows: List[Dict[str, Any]] = []
    for _, r in recommendation_df.iterrows():
        latest_seen = pd.to_datetime(r.get("latest_trade_date"), errors="coerce")
        latest_txt = latest_seen.date().isoformat() if pd.notna(latest_seen) else ""
        presence = _safe_float(r.get("presence_rate"))
        table_rows.append(
            {
                "Rank": int(_safe_int(r.get("rank"))),
                "Book": str(r.get("book_label", "")),
                "Track": str(r.get("track_label", "")),
                "Ticker": str(r.get("ticker", "")),
                "Strategy": str(r.get("strategy", "")),
                "Days": int(_safe_int(r.get("days_present"))),
                "Presence": f"{presence:.0%}" if math.isfinite(presence) else "",
                "Avg Conviction": f"{_safe_float(r.get('avg_conviction')):.1f}",
                "Avg Edge %": f"{_safe_float(r.get('avg_edge_pct')):.2f}",
                "Trend Score": f"{_safe_float(r.get('trend_score')):.3f}",
                "Latest Seen": latest_txt,
            }
        )
    lines.append(
        _render_markdown_table(
            table_rows,
            columns=[
                "Rank",
                "Book",
                "Track",
                "Ticker",
                "Strategy",
                "Days",
                "Presence",
                "Avg Conviction",
                "Avg Edge %",
                "Trend Score",
                "Latest Seen",
            ],
        )
    )
    lines.append("")
    lines.append("## Thesis + Invalidation (Top Picks)")
    for _, r in recommendation_df.head(12).iterrows():
        rank = int(_safe_int(r.get("rank")))
        ticker = str(r.get("ticker", ""))
        strategy = str(r.get("strategy", ""))
        thesis = str(r.get("thesis", "")).strip()
        invalidation = str(r.get("invalidation", "")).strip()
        summary = thesis if thesis else "No thesis text available in source row."
        invalidation_txt = invalidation if invalidation else "No invalidation text available in source row."
        lines.append(f"- #{rank} {ticker} ({strategy}): {summary} Invalidation: {invalidation_txt}")
    lines.append("")
    lines.append("## Data Files")
    lines.append(f"- {out_dir / 'final_trade_recommendations_from_trends.md'}")
    lines.append(f"- {out_dir / 'final_trade_recommendations_from_trends.csv'}")
    lines.append(f"- {out_dir / 'summary.md'}")
    lines.append("")
    return "\n".join(lines)


def _frame_preview(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_none_"
    present = [c for c in columns if c in df.columns]
    if not present:
        present = list(df.columns[: min(8, len(df.columns))])
    view = df[present].head(max_rows).copy()
    return "```text\n" + view.to_string(index=False) + "\n```"


def _md_link(target: Path, base_dir: Path, label: str, link_style: str) -> str:
    resolved = target.resolve()
    if str(link_style).strip().lower() == "vscode":
        raw = resolved.as_posix()
        href = "vscode://file/" + quote(raw, safe="/:-._~")
        return f"[{label}]({href})"
    if str(link_style).strip().lower() == "absolute":
        href = resolved.as_posix()
        return f"[{label}]({href})"
    try:
        rel = resolved.relative_to(base_dir.resolve())
        href = rel.as_posix()
    except Exception:
        href = Path(os.path.relpath(resolved, base_dir.resolve())).as_posix()
    return f"[{label}]({href})"


def _render_markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    def _cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            return f"{v:.6g}"
        s = str(v)
        s = s.replace("\r", " ").replace("\n", " ").replace("|", "\\|")
        return s

    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(_cell(r.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep, *body])


def _df_to_markdown_doc(
    df: pd.DataFrame,
    *,
    title: str,
    columns: Optional[Sequence[str]] = None,
    max_rows: int = 300,
    full_csv_name: str = "",
) -> str:
    lines: List[str] = [f"# {title}", ""]
    if df.empty:
        lines.append("_none_")
        return "\n".join(lines)
    cols = [c for c in (list(columns) if columns is not None else list(df.columns)) if c in df.columns]
    view = df[cols].copy() if cols else df.copy()
    total = len(view)
    shown = min(total, max_rows) if max_rows > 0 else total
    view = view.head(shown)
    rows = view.to_dict(orient="records")
    lines.append(_render_markdown_table(rows, list(view.columns)))
    if total > shown:
        lines.append("")
        tail = f"Showing first {shown} of {total} rows."
        if full_csv_name:
            tail += f" Full data: [{full_csv_name}](./{full_csv_name})."
        lines.append(tail)
    return "\n".join(lines)


def build_summary_markdown(
    search_root: Path,
    out_dir: Path,
    run_inventory: pd.DataFrame,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    ticker_persistence: pd.DataFrame,
    drop_daily_df: pd.DataFrame,
    link_style: str,
) -> str:
    lines: List[str] = []
    lines.append("# Historical Trend Analysis")
    if not daily_df.empty:
        d0 = pd.to_datetime(daily_df["trade_date"]).min().date().isoformat()
        d1 = pd.to_datetime(daily_df["trade_date"]).max().date().isoformat()
        lines.append("")
        lines.append(f"- Date range used: **{d0} to {d1}**")
    lines.append(f"- Source root: {search_root}")
    lines.append(f"- Runs analyzed: {len(run_inventory)}")
    lines.append(f"- Variants: {daily_df['variant'].nunique() if not daily_df.empty else 0}")
    lines.append("")

    lines.append("## Outputs (Absolute Paths)")
    lines.append(f"- {out_dir / 'run_inventory.md'}")
    lines.append(f"- {out_dir / 'daily_variant_metrics.md'}")
    lines.append(f"- {out_dir / 'weekly_variant_metrics.md'}")
    lines.append(f"- {out_dir / 'ticker_daily_metrics.md'}")
    lines.append(f"- {out_dir / 'ticker_weekly_metrics.md'}")
    lines.append(f"- {out_dir / 'ticker_persistence.md'}")
    lines.append(f"- {out_dir / 'drop_reason_daily.md'}")
    lines.append(f"- {out_dir / 'final_trade_recommendations_from_trends.md'}")
    lines.append(f"- {out_dir / 'metadata.json'}")
    lines.append("")
    lines.append("Raw CSVs:")
    lines.append(f"- {out_dir / 'run_inventory.csv'}")
    lines.append(f"- {out_dir / 'daily_variant_metrics.csv'}")
    lines.append(f"- {out_dir / 'weekly_variant_metrics.csv'}")
    lines.append(f"- {out_dir / 'ticker_daily_metrics.csv'}")
    lines.append(f"- {out_dir / 'ticker_weekly_metrics.csv'}")
    lines.append(f"- {out_dir / 'ticker_persistence.csv'}")
    lines.append(f"- {out_dir / 'drop_reason_daily.csv'}")
    lines.append(f"- {out_dir / 'final_trade_recommendations_from_trends.csv'}")
    lines.append("")
    lines.append("Open in VS Code (no browser):")
    lines.append(f"- code \"{out_dir / 'final_trade_recommendations_from_trends.md'}\"")
    lines.append(f"- code \"{out_dir / 'summary.md'}\"")
    lines.append(f"- powershell -ExecutionPolicy Bypass -File \"{out_dir / 'open_trend_files.ps1'}\"")
    lines.append("")
    lines.append("## Legend")
    lines.append("- `Core`, `Tactical`, `Watch`")
    lines.append("- `FIRE`, `SHIELD`")
    lines.append("")

    if not run_inventory.empty:
        manifest_set = set(daily_df["manifest_path"].astype(str)) if not daily_df.empty else set()
        inv = run_inventory.copy()
        if manifest_set:
            inv = inv[inv["manifest_path"].astype(str).isin(manifest_set)].copy()
        inv = inv.sort_values(["trade_date", "variant"], ascending=[False, True]).head(20)
        rows: List[Dict[str, Any]] = []
        for _, r in inv.iterrows():
            trade_date = str(r.get("trade_date", ""))[:10]
            variant = str(r.get("variant", ""))
            output_md = str(r.get("output_md", "")).strip()
            manifest = str(r.get("manifest_path", "")).strip()
            output_link = ""
            manifest_link = ""
            if output_md:
                p = Path(output_md)
                if p.exists():
                    output_link = str(p.resolve())
            if manifest:
                p = Path(manifest)
                if p.exists():
                    manifest_link = str(p.resolve())
            rows.append(
                {
                    "date": trade_date,
                    "variant": variant,
                    "trade_table": output_link,
                    "manifest": manifest_link,
                }
            )
        lines.append("## Recent Runs")
        lines.append(_render_markdown_table(rows, ["date", "variant", "trade_table", "manifest"]))
        lines.append("")

    latest_week = weekly_df["week_start"].max() if not weekly_df.empty else pd.NaT
    if pd.notna(latest_week):
        lines.append("## Latest Week Snapshot")
        latest = weekly_df[weekly_df["week_start"] == latest_week].copy()
        latest = latest.sort_values("approval_rate", ascending=False)
        lines.append(
            _frame_preview(
                latest,
                columns=[
                    "variant",
                    "week_start",
                    "trading_days",
                    "approved_rows",
                    "final_output_rows",
                    "approval_rate",
                    "proxy_pf",
                    "avg_edge_pct",
                ],
            )
        )
        lines.append("")
        lines.append("## Latest Week Mix (Core/Tactical/Watch, FIRE/SHIELD)")
        lines.append(
            _frame_preview(
                latest,
                columns=[
                    "variant",
                    "core_book_rows",
                    "tactical_book_rows",
                    "watch_book_rows",
                    "approved_fire_rows",
                    "approved_shield_rows",
                    "watch_fire_rows",
                    "watch_shield_rows",
                ],
            )
        )
        lines.append("")

    if not weekly_df.empty:
        lines.append("## Variant Momentum")
        momentum = (
            weekly_df.sort_values(["variant", "week_start"])
            .groupby("variant", as_index=False)
            .tail(1)
            .sort_values("approved_rows_wow", ascending=False)
        )
        lines.append(
            _frame_preview(
                momentum,
                columns=[
                    "variant",
                    "week_start",
                    "approved_rows",
                    "approved_rows_wow",
                    "approval_rate",
                    "approval_rate_wow",
                    "proxy_pf",
                    "proxy_pf_wow",
                    "avg_edge_pct",
                    "avg_edge_pct_wow",
                ],
            )
        )
        lines.append("")

    lines.append("## Daily Trend Tail")
    daily_tail = (
        daily_df.sort_values(["variant", "trade_date"])
        .groupby("variant", as_index=False)
        .tail(5)
        .sort_values(["variant", "trade_date"])
    )
    lines.append(
        _frame_preview(
            daily_tail,
            columns=[
                "variant",
                "trade_date",
                "approved_rows_manifest",
                "final_output_rows_manifest",
                "approval_rate_manifest",
                "core_book_rows",
                "tactical_book_rows",
                "watch_book_rows",
                "approved_fire_rows",
                "approved_shield_rows",
                "avg_edge_pct",
                "proxy_pf",
            ],
            max_rows=30,
        )
    )
    lines.append("")

    lines.append("## Persistent Tickers")
    lines.append(
        _frame_preview(
            ticker_persistence,
            columns=[
                "variant",
                "ticker",
                "days_present",
                "total_setups",
                "presence_rate",
                "avg_conviction",
                "avg_edge_pct",
                "conviction_slope",
                "total_expected_net",
            ],
            max_rows=20,
        )
    )
    lines.append("")

    lines.append("## Drop Reason Concentration")
    if drop_daily_df.empty:
        lines.append("_none_")
    else:
        dr = (
            drop_daily_df.groupby(["variant", "drop_reason"], as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
        )
        lines.append(
            _frame_preview(
                dr,
                columns=["variant", "drop_reason", "count"],
                max_rows=20,
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze replay/manifests across days and build daily+weekly trend artifacts."
    )
    ap.add_argument(
        "--search-root",
        default=str(DEFAULT_ROOT),
        help="Root folder to recursively scan for run_manifest_*.json files.",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output folder for trend CSV/MD artifacts (default: <search-root>/trend_analysis).",
    )
    ap.add_argument(
        "--start-date",
        default="",
        help="Inclusive start date (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--end-date",
        default="",
        help="Inclusive end date (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="If >0, keeps only the trailing N calendar days (relative to end-date or latest run date).",
    )
    ap.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Optional variant whitelist.",
    )
    ap.add_argument(
        "--min-ticker-days",
        type=int,
        default=2,
        help="Minimum days for a ticker to appear in ticker_persistence.csv.",
    )
    ap.add_argument(
        "--link-style",
        choices=["relative", "absolute", "vscode"],
        default="relative",
        help="Link style used in summary.md for run/trade-table hyperlinks.",
    )
    ap.add_argument(
        "--recommendation-variant",
        default="",
        help="Variant to use for consolidated final recommendations (default: auto-pick best variant in filtered window).",
    )
    ap.add_argument(
        "--recommendation-top-n",
        type=int,
        default=20,
        help="Top N consolidated recommendations to emit in final_trade_recommendations_from_trends.md/csv.",
    )
    ap.add_argument(
        "--recommendation-min-days",
        type=int,
        default=2,
        help="Minimum distinct days before a setup is eligible for Core recommendation classification.",
    )
    ap.add_argument(
        "--recommendation-min-proxy-pf",
        type=float,
        default=1.15,
        help="Minimum proxy PF estimate for setups to be eligible in final recommendations (default: 1.15).",
    )
    ap.add_argument(
        "--recommendation-min-shield",
        type=int,
        default=2,
        help="Minimum SHIELD rows to force into final top-N recommendations if available.",
    )
    ap.add_argument(
        "--recommendation-live-actionable-only",
        action="store_true",
        help="If set, final recommendations require latest snapshot gate PASS and final-valid = YES.",
    )
    return ap.parse_args(list(argv) if argv is not None else None)


def _parse_opt_date(txt: str) -> Optional[pd.Timestamp]:
    s = str(txt or "").strip()
    if not s:
        return None
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid date: {txt}")
    return dt.normalize()


def run_pipeline(args: argparse.Namespace) -> Dict[str, Path]:
    search_root = Path(args.search_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (search_root / "trend_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = discover_run_bundles(search_root)
    if args.variants:
        wanted = {_normalize_variant(v) for v in args.variants}
        bundles = [b for b in bundles if b.variant in wanted]
    if not bundles:
        raise RuntimeError(f"No run manifests discovered under {search_root}")

    daily_rows: List[Dict[str, Any]] = []
    ticker_raw_parts: List[pd.DataFrame] = []
    drop_parts: List[pd.DataFrame] = []
    inv_rows: List[Dict[str, Any]] = []
    for b in bundles:
        inv_rows.append(
            {
                "variant": b.variant,
                "trade_date": b.trade_date,
                "manifest_path": str(b.manifest_path),
                "run_dir": str(b.run_dir),
                "live_csv": str(b.artifacts.live_csv or ""),
                "live_final_csv": str(b.artifacts.live_final_csv or ""),
                "setup_likelihood_csv": str(b.artifacts.setup_likelihood_csv or ""),
                "dropped_csv": str(b.artifacts.dropped_csv or ""),
                "output_md": str(b.artifacts.output_md or ""),
            }
        )
        row, ticker_rows, drop_rows = _extract_daily_row(b)
        daily_rows.append(row)
        if not ticker_rows.empty:
            ticker_raw_parts.append(ticker_rows)
        if not drop_rows.empty:
            drop_parts.append(drop_rows)

    run_inventory = pd.DataFrame(inv_rows).sort_values(["trade_date", "variant", "manifest_path"])
    daily_df = pd.DataFrame(daily_rows)
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"], errors="coerce")
    daily_df["week_start"] = pd.to_datetime(daily_df["week_start"], errors="coerce")
    daily_df = daily_df.sort_values(["trade_date", "variant", "manifest_path"]).reset_index(drop=True)
    ticker_raw_df = (
        pd.concat(ticker_raw_parts, ignore_index=True)
        if ticker_raw_parts
        else pd.DataFrame()
    )
    drop_daily_df = pd.concat(drop_parts, ignore_index=True) if drop_parts else pd.DataFrame()

    start_date = _parse_opt_date(args.start_date)
    end_date = _parse_opt_date(args.end_date)
    daily_df, ticker_raw_df, drop_daily_df = _apply_date_filters(
        daily_df=daily_df,
        ticker_raw_df=ticker_raw_df,
        drop_df=drop_daily_df,
        start_date=start_date,
        end_date=end_date,
        lookback_days=max(0, int(args.lookback_days)),
    )
    if daily_df.empty:
        raise RuntimeError("No rows left after date/variant filters.")

    daily_df = _rolling_features(daily_df)
    weekly_df = aggregate_weekly(daily_df)
    ticker_daily_df, ticker_weekly_df, ticker_persistence_df = aggregate_tickers(
        ticker_raw_df=ticker_raw_df,
        daily_df=daily_df,
    )
    if not ticker_persistence_df.empty and args.min_ticker_days > 1:
        ticker_persistence_df = ticker_persistence_df[
            ticker_persistence_df["days_present"] >= int(args.min_ticker_days)
        ].copy()

    recommendation_df, recommendation_meta = build_trend_recommendations(
        ticker_raw_df=ticker_raw_df,
        daily_df=daily_df,
        weekly_df=weekly_df,
        run_inventory=run_inventory,
        recommendation_variant=str(getattr(args, "recommendation_variant", "")),
        recommendation_top_n=int(getattr(args, "recommendation_top_n", 20)),
        recommendation_min_days=int(getattr(args, "recommendation_min_days", 2)),
        recommendation_min_proxy_pf=float(getattr(args, "recommendation_min_proxy_pf", 1.15)),
        recommendation_min_shield=int(getattr(args, "recommendation_min_shield", 2)),
        recommendation_live_actionable_only=bool(
            getattr(args, "recommendation_live_actionable_only", False)
        ),
    )
    recommendation_text = build_final_recommendations_markdown(
        out_dir=out_dir,
        recommendation_df=recommendation_df,
        recommendation_meta=recommendation_meta,
    )

    summary_text = build_summary_markdown(
        search_root=search_root,
        out_dir=out_dir,
        run_inventory=run_inventory,
        daily_df=daily_df,
        weekly_df=weekly_df,
        ticker_persistence=ticker_persistence_df,
        drop_daily_df=drop_daily_df,
        link_style=str(args.link_style),
    )

    out_paths = {
        "run_inventory_csv": out_dir / "run_inventory.csv",
        "run_inventory_md": out_dir / "run_inventory.md",
        "daily_variant_metrics_csv": out_dir / "daily_variant_metrics.csv",
        "daily_variant_metrics_md": out_dir / "daily_variant_metrics.md",
        "weekly_variant_metrics_csv": out_dir / "weekly_variant_metrics.csv",
        "weekly_variant_metrics_md": out_dir / "weekly_variant_metrics.md",
        "ticker_raw_rows_csv": out_dir / "ticker_raw_rows.csv",
        "ticker_daily_metrics_csv": out_dir / "ticker_daily_metrics.csv",
        "ticker_daily_metrics_md": out_dir / "ticker_daily_metrics.md",
        "ticker_weekly_metrics_csv": out_dir / "ticker_weekly_metrics.csv",
        "ticker_weekly_metrics_md": out_dir / "ticker_weekly_metrics.md",
        "ticker_persistence_csv": out_dir / "ticker_persistence.csv",
        "ticker_persistence_md": out_dir / "ticker_persistence.md",
        "drop_reason_daily_csv": out_dir / "drop_reason_daily.csv",
        "drop_reason_daily_md": out_dir / "drop_reason_daily.md",
        "final_trade_recommendations_csv": out_dir / "final_trade_recommendations_from_trends.csv",
        "final_trade_recommendations_md": out_dir / "final_trade_recommendations_from_trends.md",
        "summary_md": out_dir / "summary.md",
        "metadata_json": out_dir / "metadata.json",
        "open_files_ps1": out_dir / "open_trend_files.ps1",
    }
    run_inventory.to_csv(out_paths["run_inventory_csv"], index=False)
    daily_df.to_csv(out_paths["daily_variant_metrics_csv"], index=False)
    weekly_df.to_csv(out_paths["weekly_variant_metrics_csv"], index=False)
    ticker_raw_df.to_csv(out_paths["ticker_raw_rows_csv"], index=False)
    ticker_daily_df.to_csv(out_paths["ticker_daily_metrics_csv"], index=False)
    ticker_weekly_df.to_csv(out_paths["ticker_weekly_metrics_csv"], index=False)
    ticker_persistence_df.to_csv(out_paths["ticker_persistence_csv"], index=False)
    drop_daily_df.to_csv(out_paths["drop_reason_daily_csv"], index=False)
    recommendation_df.to_csv(out_paths["final_trade_recommendations_csv"], index=False)

    out_paths["run_inventory_md"].write_text(
        _df_to_markdown_doc(
            run_inventory,
            title="Run Inventory",
            columns=[
                "trade_date",
                "variant",
                "manifest_path",
                "output_md",
                "live_csv",
                "live_final_csv",
                "setup_likelihood_csv",
                "dropped_csv",
            ],
            max_rows=500,
            full_csv_name="run_inventory.csv",
        ),
        encoding="utf-8",
    )
    out_paths["daily_variant_metrics_md"].write_text(
        _df_to_markdown_doc(
            daily_df,
            title="Daily Variant Metrics",
            columns=[
                "trade_date",
                "variant",
                "approved_rows_manifest",
                "final_output_rows_manifest",
                "approval_rate_manifest",
                "core_book_rows",
                "tactical_book_rows",
                "watch_book_rows",
                "approved_fire_rows",
                "approved_shield_rows",
                "watch_fire_rows",
                "watch_shield_rows",
                "avg_edge_pct",
                "proxy_pf",
            ],
            max_rows=800,
            full_csv_name="daily_variant_metrics.csv",
        ),
        encoding="utf-8",
    )
    out_paths["weekly_variant_metrics_md"].write_text(
        _df_to_markdown_doc(
            weekly_df,
            title="Weekly Variant Metrics",
            columns=[
                "week_start",
                "variant",
                "trading_days",
                "approved_rows",
                "final_output_rows",
                "approval_rate",
                "core_book_rows",
                "tactical_book_rows",
                "watch_book_rows",
                "approved_fire_rows",
                "approved_shield_rows",
                "avg_edge_pct",
                "proxy_pf",
                "approved_rows_wow",
                "approval_rate_wow",
                "proxy_pf_wow",
            ],
            max_rows=400,
            full_csv_name="weekly_variant_metrics.csv",
        ),
        encoding="utf-8",
    )
    ticker_daily_for_md = ticker_daily_df.copy()
    if not ticker_daily_for_md.empty:
        ticker_daily_for_md = ticker_daily_for_md.sort_values(
            ["trade_date", "variant", "setups", "ticker"],
            ascending=[False, True, False, True],
        )
    out_paths["ticker_daily_metrics_md"].write_text(
        _df_to_markdown_doc(
            ticker_daily_for_md,
            title="Ticker Daily Metrics",
            columns=[
                "trade_date",
                "variant",
                "ticker",
                "setups",
                "strategies",
                "avg_conviction",
                "avg_hist_success_pct",
                "avg_edge_pct",
                "expected_net",
            ],
            max_rows=600,
            full_csv_name="ticker_daily_metrics.csv",
        ),
        encoding="utf-8",
    )
    ticker_weekly_for_md = ticker_weekly_df.copy()
    if not ticker_weekly_for_md.empty:
        ticker_weekly_for_md = ticker_weekly_for_md.sort_values(
            ["week_start", "variant", "setups", "ticker"],
            ascending=[False, True, False, True],
        )
    out_paths["ticker_weekly_metrics_md"].write_text(
        _df_to_markdown_doc(
            ticker_weekly_for_md,
            title="Ticker Weekly Metrics",
            columns=[
                "week_start",
                "variant",
                "ticker",
                "days_present",
                "setups",
                "avg_conviction",
                "avg_edge_pct",
                "expected_net",
            ],
            max_rows=600,
            full_csv_name="ticker_weekly_metrics.csv",
        ),
        encoding="utf-8",
    )
    out_paths["ticker_persistence_md"].write_text(
        _df_to_markdown_doc(
            ticker_persistence_df,
            title="Ticker Persistence",
            columns=[
                "variant",
                "ticker",
                "days_present",
                "total_setups",
                "presence_rate",
                "avg_conviction",
                "avg_edge_pct",
                "conviction_slope",
                "edge_slope",
                "total_expected_net",
                "first_seen",
                "last_seen",
            ],
            max_rows=800,
            full_csv_name="ticker_persistence.csv",
        ),
        encoding="utf-8",
    )
    drop_daily_for_md = drop_daily_df.copy()
    if not drop_daily_for_md.empty:
        drop_daily_for_md = drop_daily_for_md.sort_values(
            ["trade_date", "variant", "count"],
            ascending=[False, True, False],
        )
    out_paths["drop_reason_daily_md"].write_text(
        _df_to_markdown_doc(
            drop_daily_for_md,
            title="Drop Reason Daily",
            columns=["trade_date", "variant", "stage", "drop_reason", "count", "manifest_path"],
            max_rows=800,
            full_csv_name="drop_reason_daily.csv",
        ),
        encoding="utf-8",
    )
    out_paths["final_trade_recommendations_md"].write_text(recommendation_text, encoding="utf-8")
    out_paths["summary_md"].write_text(summary_text, encoding="utf-8")
    metadata = {
        "search_root": str(search_root),
        "out_dir": str(out_dir),
        "runs_discovered": int(len(bundles)),
        "runs_after_filters": int(len(daily_df)),
        "variants_after_filters": int(daily_df["variant"].nunique()),
        "date_min": str(pd.to_datetime(daily_df["trade_date"]).min().date()),
        "date_max": str(pd.to_datetime(daily_df["trade_date"]).max().date()),
        "lookback_days": int(max(0, int(args.lookback_days))),
        "start_date": str(start_date.date()) if start_date is not None else "",
        "end_date": str(end_date.date()) if end_date is not None else "",
        "recommendation_variant_selected": str(recommendation_meta.get("selected_variant", "")),
        "recommendation_rows": int(len(recommendation_df)),
        "recommendation_date_min": str(recommendation_meta.get("date_min", "")),
        "recommendation_date_max": str(recommendation_meta.get("date_max", "")),
        "recommendation_latest_output_md_files": recommendation_meta.get("latest_output_md_files", []),
        "recommendation_min_proxy_pf": float(
            _safe_float(recommendation_meta.get("min_proxy_pf"))
            if math.isfinite(_safe_float(recommendation_meta.get("min_proxy_pf")))
            else 0.0
        ),
        "recommendation_proxy_pf_filter_applied": bool(
            recommendation_meta.get("proxy_pf_filter_applied", False)
        ),
        "recommendation_proxy_pf_rows_before_filter": int(
            _safe_int(recommendation_meta.get("proxy_pf_rows_before_filter"))
        ),
        "recommendation_proxy_pf_rows_after_filter": int(
            _safe_int(recommendation_meta.get("proxy_pf_rows_after_filter"))
        ),
        "recommendation_min_shield": int(_safe_int(recommendation_meta.get("min_shield"))),
        "recommendation_shield_rows_selected": int(
            _safe_int(recommendation_meta.get("shield_rows_selected"))
        ),
        "recommendation_live_actionable_only": bool(
            recommendation_meta.get("live_actionable_only", False)
        ),
        "recommendation_live_actionable_rows_before_filter": int(
            _safe_int(recommendation_meta.get("live_actionable_rows_before_filter"))
        ),
        "recommendation_live_actionable_rows_after_filter": int(
            _safe_int(recommendation_meta.get("live_actionable_rows_after_filter"))
        ),
    }
    out_paths["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    open_targets: List[Path] = [
        out_paths["final_trade_recommendations_md"],
        out_paths["summary_md"],
        out_paths["daily_variant_metrics_md"],
        out_paths["weekly_variant_metrics_md"],
        out_paths["ticker_persistence_md"],
    ]
    inv_for_open = run_inventory.sort_values(["trade_date", "variant"], ascending=[False, True]).copy()
    extra_count = 0
    for _, r in inv_for_open.iterrows():
        p_txt = str(r.get("output_md", "")).strip()
        if not p_txt:
            continue
        p = Path(p_txt)
        if p.exists():
            open_targets.append(p.resolve())
            extra_count += 1
        if extra_count >= 3:
            break

    # Deduplicate while preserving order.
    dedup: List[Path] = []
    seen: set[str] = set()
    for p in open_targets:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p.resolve())

    ps_lines = [
        "$ErrorActionPreference = 'Stop'",
        "$files = @(",
    ]
    for p in dedup:
        safe = str(p).replace("'", "''")
        ps_lines.append(f"  '{safe}'")
    ps_lines.append(")")
    ps_lines.append("foreach ($f in $files) {")
    ps_lines.append("  if (Test-Path $f) { code $f }")
    ps_lines.append("}")
    out_paths["open_files_ps1"].write_text("\n".join(ps_lines) + "\n", encoding="utf-8")
    return out_paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_paths = run_pipeline(args)
    print("Historical trend pipeline completed.")
    for k, p in out_paths.items():
        print(f"- {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

