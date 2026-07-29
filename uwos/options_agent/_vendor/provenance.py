from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import zipfile
from pathlib import Path
from typing import Any

from .data import find_export, find_export_bundle


EXPORT_PREFIXES = {
    "stock_screener": "stock-screener-",
    "hot_chains": "hot-chains-",
    "chain_oi_changes": "chain-oi-changes-",
    "bot_eod_report": "bot-eod-report-",
    "dp_eod_report": "dp-eod-report-",
}


def _cache_path(path: Path) -> Path:
    return path.parent / ".codexuw_hash_cache.json"


def _cache_key(path: Path) -> str:
    return str(path.resolve())


def _load_hash_cache(path: Path) -> dict[str, Any]:
    cache_file = _cache_path(path)
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_hash_cache(path: Path, cache: dict[str, Any]) -> None:
    cache_file = _cache_path(path)
    try:
        cache_file.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path, *, use_cache: bool = True) -> dict[str, Any]:
    stat = path.stat()
    key = _cache_key(path)
    signature = {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    cache = _load_hash_cache(path) if use_cache else {}
    cached = cache.get(key)
    if cached and cached.get("signature") == signature and cached.get("fingerprint"):
        payload = dict(cached["fingerprint"])
        payload["hash_cache"] = "hit"
        return payload
    payload: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat(),
        "sha256": file_sha256(path),
        "hash_cache": "miss",
    }
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                payload["zip_members"] = [
                    {
                        "name": info.filename,
                        "size_bytes": int(info.file_size),
                        "crc": f"{info.CRC:08x}",
                    }
                    for info in zf.infolist()
                    if not info.is_dir()
                ][:25]
        except zipfile.BadZipFile:
            payload["zip_error"] = "bad_zip_file"
    if use_cache:
        cache[key] = {"signature": signature, "fingerprint": payload}
        _save_hash_cache(path, cache)
    return payload


def build_input_provenance(base_dir: Path) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for label, prefix in EXPORT_PREFIXES.items():
        try:
            if label == "bot_eod_report":
                bundle = find_export_bundle(base_dir, prefix)
            else:
                bundle = [find_export(base_dir, prefix)]
        except FileNotFoundError:
            continue
        if len(bundle) == 1:
            exports[label] = file_fingerprint(bundle[0])
        else:
            exports[label] = {
                "bundle": "complete_split_export",
                "part_count": len(bundle),
                "parts": [file_fingerprint(path) for path in bundle],
            }

    browser_texts: list[dict[str, Any]] = []
    browser_dir = base_dir / "browser_text"
    if browser_dir.is_dir():
        for path in sorted(browser_dir.glob("browser-text-capture-*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".json"}:
                browser_texts.append(file_fingerprint(path))

    return {
        "base_dir": str(base_dir),
        "exports": exports,
        "browser_texts": browser_texts,
        "export_count": len(exports),
        "browser_text_count": len(browser_texts),
    }


def build_schwab_snapshot_provenance(out_dir: Path) -> dict[str, Any]:
    chain_dir = out_dir / "schwab_chains"
    if not chain_dir.is_dir():
        return {"status": "not_available", "snapshot_dir": str(chain_dir), "chains": {}}
    chains: dict[str, Any] = {}
    for path in sorted(chain_dir.glob("*.json")):
        if path.name in {"errors.json", "manifest.json"}:
            continue
        chains[path.stem.upper()] = file_fingerprint(path)
    errors = {}
    errors_path = chain_dir / "errors.json"
    if errors_path.exists():
        try:
            errors = json.loads(errors_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors = {"_error": "invalid_errors_json"}
    return {
        "status": "ok" if chains else "empty",
        "snapshot_dir": str(chain_dir),
        "chain_count": len(chains),
        "chains": chains,
        "errors": errors,
    }


def build_run_environment() -> dict[str, Any]:
    return {
        "cwd": os.getcwd(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
