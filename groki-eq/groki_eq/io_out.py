import csv
import json
from pathlib import Path
from typing import Dict, Sequence

BOARD_COLUMNS = [
    "asof_date",
    "ticker",
    "structure",
    "close",
    "high_20",
    "pct_above",
    "atr14",
    "stop",
    "shares",
    "risk_dollars",
    "notional",
    "reasons",
    "decision_pass",
    "action",
    "sleeve",
]

REJECTION_COLUMNS = ["asof_date", "ticker", "structure", "reasons", "stage"]


def day_dir(out_dir: Path, date: str) -> Path:
    path = Path(out_dir) / date
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _fmt(value, digits=4) -> str:
    try:
        return ("%." + str(digits) + "f") % float(value)
    except (TypeError, ValueError):
        return ""


def write_board_md(
    path: Path,
    date: str,
    blocker: str,
    execute_count: int,
    watch_count: int,
    execute_rows: Sequence[dict],
    watch_rows: Sequence[dict],
) -> None:
    lines = [
        "# groki-eq %s" % date,
        "",
        "selector: breakout_eq",
        "execute_count: %s" % execute_count,
        "watch_count: %s" % watch_count,
        "blocker: %s" % (blocker or ""),
        "",
        "## EXECUTE",
        "",
    ]
    if not execute_rows:
        lines.append("Empty board. Valid.")
        lines.append("")
    else:
        lines.append("| ticker | close | 20d high | % above | ATR14 | stop | shares | action |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in execute_rows:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    row.get("ticker") or "",
                    _fmt(row.get("close"), 2),
                    _fmt(row.get("high_20"), 2),
                    _fmt(row.get("pct_above"), 4),
                    _fmt(row.get("atr14"), 2),
                    _fmt(row.get("stop"), 2),
                    row.get("shares") or "",
                    row.get("action") or "",
                )
            )
        lines.append("")
    lines.extend(["## WATCH", ""])
    if not watch_rows:
        lines.append("None.")
        lines.append("")
    else:
        lines.append("| ticker | close | % above | reasons |")
        lines.append("|---|---:|---:|---|")
        for row in watch_rows:
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    row.get("ticker") or "",
                    _fmt(row.get("close"), 2),
                    _fmt(row.get("pct_above"), 4),
                    row.get("reasons") or "",
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_manifest(path: Path, manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
