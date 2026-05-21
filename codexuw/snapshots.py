from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float
from .provenance import build_run_environment, build_schwab_snapshot_provenance, file_fingerprint
from .schwab_live import chain_spot, chain_to_contracts


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit(workdir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workdir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _chain_timestamp(chain: dict[str, Any], path: Path) -> str:
    for key in ["timestamp", "quoteTime", "tradeTime"]:
        value = chain.get(key)
        if value:
            return str(value)
    underlying = chain.get("underlying", {}) if isinstance(chain, dict) else {}
    for key in ["quoteTime", "tradeTime"]:
        value = underlying.get(key)
        if value:
            return str(value)
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat()


def summarize_schwab_chain_snapshots(out_dir: Path, asof: dt.date) -> Path:
    chain_dir = out_dir / "schwab_chains"
    rows: list[dict[str, Any]] = []
    manifest = {}
    manifest_path = chain_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    for path in sorted(chain_dir.glob("*.json")) if chain_dir.is_dir() else []:
        if path.name in {"errors.json", "manifest.json"}:
            continue
        try:
            chain = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ticker = path.stem.upper()
        spot = chain_spot(chain)
        timestamp = _chain_timestamp(chain, path)
        contracts = chain_to_contracts(chain)
        if contracts.empty:
            rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "spot": spot if math.isfinite(spot) else "",
                    "contract_symbol": "",
                    "right": "",
                    "expiry": "",
                    "strike": "",
                    "bid": "",
                    "ask": "",
                    "mid": "",
                    "natural": "",
                    "delta": "",
                    "iv": "",
                    "open_interest": "",
                    "volume": "",
                    "chain_request_metadata": (manifest.get("sources") or {}).get(ticker, ""),
                }
            )
            continue
        for _, contract in contracts.iterrows():
            bid = safe_float(contract.get("bid"))
            ask = safe_float(contract.get("ask"))
            mid = (bid + ask) / 2.0 if math.isfinite(bid) and math.isfinite(ask) else safe_float(contract.get("mark"))
            rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "spot": spot if math.isfinite(spot) else "",
                    "contract_symbol": contract.get("symbol", ""),
                    "right": contract.get("right", ""),
                    "expiry": contract.get("expiry", ""),
                    "strike": contract.get("strike", ""),
                    "bid": bid if math.isfinite(bid) else "",
                    "ask": ask if math.isfinite(ask) else "",
                    "mid": mid if math.isfinite(mid) else "",
                    "natural": bid if math.isfinite(bid) else "",
                    "delta": contract.get("delta", ""),
                    "iv": contract.get("iv", ""),
                    "open_interest": contract.get("open_interest", ""),
                    "volume": contract.get("volume", ""),
                    "chain_request_metadata": (manifest.get("sources") or {}).get(ticker, ""),
                }
            )
    path = out_dir / f"codexdaily_v3_schwab_snapshot_summary_{asof}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_reproducibility_artifacts(
    *,
    out_dir: Path,
    asof: dt.date,
    repo_root: Path,
    run_config: dict[str, Any],
    input_provenance: dict[str, Any],
    data_quality: dict[str, Any] | None,
    portfolio: dict[str, Any] | None,
    regime: dict[str, Any] | None,
    loss_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_summary = summarize_schwab_chain_snapshots(out_dir, asof)
    artifacts = {
        "pipeline": "Codex Daily V3",
        "asof": str(asof),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_hash": config_hash(run_config),
        "git_commit": git_commit(repo_root),
        "environment": build_run_environment(),
        "input_files": input_provenance,
        "schwab_snapshot": build_schwab_snapshot_provenance(out_dir),
        "schwab_snapshot_summary": str(snapshot_summary),
        "portfolio_status": {
            "status": (portfolio or {}).get("status", "not_checked"),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "position_count": (portfolio or {}).get("position_count", 0),
        },
        "data_quality": data_quality or {},
        "regime": regime or {},
        "loss_review": loss_review or {},
        "run_config": run_config,
    }
    path = out_dir / f"codexdaily_v3_reproducibility_{asof}.json"
    path.write_text(json.dumps(artifacts, indent=2, sort_keys=True, default=str), encoding="utf-8")
    artifacts["reproducibility_artifact"] = str(path)
    try:
        artifacts["reproducibility_fingerprint"] = file_fingerprint(path)
    except OSError:
        pass
    return artifacts
