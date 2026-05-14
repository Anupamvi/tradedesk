from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


HARD_FINAL_TOKENS = {
    "news_catalyst_caution",
    "final_guard_near_term_news_caution",
    "earnings_news_risk",
    "negative_replay_edge",
    "no_usable_liquidity",
    "bid_ask_too_wide",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _asof_from_run_dir(run_dir: Path, explicit: str = "") -> str:
    if explicit:
        return explicit
    manifests = sorted(run_dir.glob("codexuw_manifest_*.json"))
    if manifests:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", manifests[0].name)
        if match:
            return match.group(1)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", run_dir.name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot infer asof date from {run_dir}; pass --asof YYYY-MM-DD")


def _token_blob(row: pd.Series, columns: list[str]) -> str:
    return ";".join(str(row.get(col, "") or "") for col in columns)


def _manifest_count_checks(run_dir: Path, asof: str, manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    checks = {
        "execute_rows": run_dir / f"codexuw_final_trades_{asof}.csv",
        "watch_rows": run_dir / f"codexuw_watch_trades_{asof}.csv",
        "research_rows": run_dir / f"codexuw_research_candidates_{asof}.csv",
        "avoid_rows": run_dir / f"codexuw_avoid_trades_{asof}.csv",
    }
    for key, path in checks.items():
        actual = len(_read_csv(path))
        expected = manifest.get(key)
        if expected != actual:
            issues.append(f"{key} manifest={expected} csv={actual} path={path}")
    return issues


def _final_trade_checks(run_dir: Path, asof: str) -> list[str]:
    issues: list[str] = []
    final = _read_csv(run_dir / f"codexuw_final_trades_{asof}.csv")
    if final.empty:
        return issues
    for _, row in final.iterrows():
        ticker = row.get("ticker", "UNKNOWN")
        if str(row.get("hard_rejects") or "").strip() and str(row.get("hard_rejects")).lower() != "nan":
            issues.append(f"final {ticker} has hard_rejects={row.get('hard_rejects')}")
        blob = _token_blob(row, ["penalties", "confirmations_failed", "trade_status_reason", "risk_notes"])
        bad = sorted(token for token in HARD_FINAL_TOKENS if token in blob)
        if bad:
            issues.append(f"final {ticker} contains hard-block token(s): {bad}")
        max_loss = pd.to_numeric(pd.Series([row.get("max_loss")]), errors="coerce").iloc[0]
        if not math.isfinite(float(max_loss)) or float(max_loss) <= 0:
            issues.append(f"final {ticker} has invalid max_loss={row.get('max_loss')}")
    return issues


def _ledger_checks(run_dir: Path, asof: str) -> list[str]:
    issues: list[str] = []
    final = _read_csv(run_dir / f"codexuw_final_trades_{asof}.csv")
    ledger = _read_csv(run_dir / f"codexuw_execute_outcome_ledger_{asof}.csv")
    if len(final) != len(ledger):
        issues.append(f"ledger rows={len(ledger)} final rows={len(final)}")
    if final.empty:
        return issues
    if ledger.empty:
        return issues
    for _, row in final.iterrows():
        match = ledger[(ledger.get("ticker") == row.get("ticker")) & (ledger.get("strategy") == row.get("strategy"))]
        if match.empty:
            issues.append(f"ledger missing {row.get('ticker')} {row.get('strategy')}")
            continue
        status = str(match.iloc[0].get("outcome_status") or "")
        if status != "OPEN_REVIEW_REQUIRED":
            issues.append(f"ledger status for {row.get('ticker')} is {status}")
    return issues


def _provenance_checks(manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    provenance = manifest.get("run_provenance") or {}
    exports = ((provenance.get("input_files") or {}).get("exports") or {})
    for key in ["stock_screener", "hot_chains"]:
        if key not in exports:
            issues.append(f"missing required input provenance: {key}")
        elif len(str(exports[key].get("sha256") or "")) != 64:
            issues.append(f"bad sha256 in input provenance: {key}")
    snapshot = provenance.get("schwab_snapshot") or {}
    if manifest.get("execute_rows", 0) and snapshot.get("status") != "ok":
        issues.append(f"execute run missing ok Schwab snapshot provenance: {snapshot.get('status')}")
    return issues


def _catalyst_checks(run_dir: Path, asof: str) -> list[str]:
    issues: list[str] = []
    catalysts = _read_csv(run_dir / f"codexuw_catalysts_{asof}.csv")
    if catalysts.empty:
        return issues
    allowed = {"supportive", "mixed", "caution", "unknown"}
    for _, row in catalysts.iterrows():
        ticker = row.get("ticker", "UNKNOWN")
        status = str(row.get("catalyst_status") or "")
        if status not in allowed:
            issues.append(f"catalyst {ticker} invalid status={status}")
        date_value = row.get("catalyst_earnings_date")
        if pd.notna(date_value) and str(date_value).strip():
            parsed = pd.to_datetime(date_value, errors="coerce")
            if pd.isna(parsed):
                issues.append(f"catalyst {ticker} invalid earnings date={date_value}")
    return issues


def _report_checks(run_dir: Path, asof: str) -> list[str]:
    issues: list[str] = []
    report = run_dir / f"codexuw_trade_report_{asof}.md"
    if not report.exists():
        return [f"missing report: {report}"]
    text = report.read_text(encoding="utf-8")
    if "## Action Board" not in text:
        issues.append("report missing Action Board")
        return issues
    section = text.split("## Action Board", 1)[1]
    section = section.split("\n## ", 1)[0]
    tickers = re.findall(r"\| [🟢🔵🟡🔴][^|]*\|\s*([A-Z]{1,5})\s*\|", section)
    counts = Counter(tickers)
    duplicates = {ticker: count for ticker, count in counts.items() if count > 1}
    if duplicates:
        issues.append(f"action board duplicate tickers: {duplicates}")
    return issues


def audit_run(run_dir: Path, *, asof: str = "") -> list[str]:
    run_dir = run_dir.expanduser().resolve()
    asof = _asof_from_run_dir(run_dir, asof)
    manifest_path = run_dir / f"codexuw_manifest_{asof}.json"
    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    issues.extend(_manifest_count_checks(run_dir, asof, manifest))
    issues.extend(_final_trade_checks(run_dir, asof))
    issues.extend(_ledger_checks(run_dir, asof))
    issues.extend(_provenance_checks(manifest))
    issues.extend(_catalyst_checks(run_dir, asof))
    issues.extend(_report_checks(run_dir, asof))
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postflight QA for a codexuw.daily run directory")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--asof", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues = audit_run(Path(args.run_dir), asof=args.asof)
    if issues:
        print("PIPELINE FAILED QA")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("PIPELINE PASSED QA")


if __name__ == "__main__":
    main()
