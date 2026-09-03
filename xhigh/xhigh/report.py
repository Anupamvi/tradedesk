from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from xhigh.rec import decorate, render_recommendation, sort_clicks


def _md_table(rows: List[dict], cols: List[tuple]) -> List[str]:
    lines = ["| " + " | ".join(h for h, _ in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    if not rows:
        lines.append("| " + " | ".join("—" for _ in cols) + " |")
        return lines
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(k) if row.get(k) not in (None, "") else "DATA UNAVAILABLE") for _, k in cols)
            + " |"
        )
    return lines


CLICK_COLS = [
    ("do", "do_s"),
    ("ticker", "ticker"),
    ("last", "last"),
    ("strategy", "strategy"),
    ("expiry", "expiry_s"),
    ("pay / collect", "target_s"),
    ("need", "need_s"),
    ("risk", "risk_s"),
    ("P:R", "rr_s"),
    ("POP", "pop_s"),
]

SKIP_COLS = [
    ("do", "do_s"),
    ("ticker", "ticker"),
    ("last", "last"),
    ("strategy", "strategy"),
    ("expiry", "expiry_s"),
    ("target", "target_s"),
    ("P:R", "rr_s"),
    ("why skip", "why_s"),
]


def write_run(
    dest: Path,
    *,
    date: str,
    tickets: List[dict],
    watch: List[dict],
    x_queue: List[str],
    gates: dict,
    skips: List[dict],
    manifest: dict,
    macro: Optional[dict] = None,
    skip: Optional[List[dict]] = None,
) -> Dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    skip = skip if skip is not None else []
    pooled = [decorate(r, gates) for r in list(tickets) + list(skip) + list(watch)]
    click = sort_clicks([r for r in pooled if r.get("action") == "CLICK"])
    skip = [r for r in pooled if r.get("action") == "SKIP"]
    watch = [r for r in pooled if r.get("action") == "WATCH"]

    files = {}
    (dest / "tickets.json").write_text(json.dumps(click, indent=2) + "\n", encoding="utf-8")
    files["tickets"] = str(dest / "tickets.json")
    (dest / "skip.json").write_text(json.dumps(skip, indent=2) + "\n", encoding="utf-8")
    (dest / "watch.json").write_text(json.dumps(watch, indent=2) + "\n", encoding="utf-8")
    (dest / "skips.json").write_text(json.dumps(skips, indent=2) + "\n", encoding="utf-8")
    (dest / "x_queue.json").write_text(
        json.dumps({"date": date, "names": sorted({str(r.get("ticker")) for r in click if r.get("ticker")})}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    files["x_queue"] = str(dest / "x_queue.json")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    files["manifest"] = str(dest / "manifest.json")
    if gates:
        (dest / "gates.md").write_text(
            "# xhigh gates %s\n\n```json\n%s\n```\n" % (date, json.dumps(gates, indent=2)),
            encoding="utf-8",
        )
    macro = macro if macro is not None else (manifest.get("macro") if isinstance(manifest, dict) else {}) or {}
    rec = render_recommendation(date, click, skip, watch, macro)
    blines = list(rec)
    blines.extend(["## 🟢 CLICK", ""])
    if not click:
        blines.append("None. Empty is valid.")
    else:
        blines.extend(_md_table(click, CLICK_COLS))
    if skip:
        blines.extend(["", "## 🔴 SKIP", "", *_md_table(skip, SKIP_COLS)])
    if watch:
        blines.extend(["", "## 🟡 WATCH", "", *_md_table(watch, SKIP_COLS)])
    board = dest / "board.md"
    board.write_text("\n".join(blines) + "\n", encoding="utf-8")
    files["board"] = str(board)
    rec_path = dest / "recommendation.md"
    rec_path.write_text("\n".join(rec) + "\n", encoding="utf-8")
    files["recommendation"] = str(rec_path)
    return files


def overlay_x(dest: Path, hot: dict) -> None:
    _restamp(dest, xhot=hot, intel=None)


def overlay_intel(dest: Path, intel: dict) -> None:
    _restamp(dest, xhot=None, intel=intel)


def _load_rows(dest: Path, name: str) -> List[dict]:
    path = dest / name
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _restamp(dest: Path, xhot: Optional[dict], intel: Optional[dict]) -> None:
    click = _load_rows(dest, "tickets.json")
    skip = _load_rows(dest, "skip.json")
    watch = _load_rows(dest, "watch.json")
    if not click and not skip:
        # legacy: everything in tickets
        pass
    names = {}
    if xhot:
        for row in xhot.get("names") or []:
            if isinstance(row, dict) and row.get("ticker"):
                names[str(row["ticker"]).upper()] = row
    intel_names = {}
    if intel:
        for row in intel.get("names") or []:
            if isinstance(row, dict) and row.get("ticker"):
                intel_names[str(row["ticker"]).upper()] = row
    from xhigh.gates import load_gates
    from xhigh.score import confidence

    gates = load_gates()
    all_rows = click + skip + watch
    for row in all_rows:
        key = str(row.get("ticker") or "").upper()
        info = names.get(key)
        if info:
            row["x_tag"] = info.get("tag") or row.get("x_tag") or "DATA UNAVAILABLE"
        iv = intel_names.get(key)
        if iv:
            if iv.get("tag"):
                row["x_tag"] = iv.get("tag")
            if iv.get("conf_delta") is not None:
                row["conf_delta"] = iv.get("conf_delta")
            row["conf"] = confidence(row, {"source": row.get("earnings_source"), "usable": True}, gates)
            if iv.get("kill"):
                row["action"] = "WATCH"
                row["note"] = (row.get("note") or "") + " intel KILL"
        row.update(decorate(row, gates))
        if iv and iv.get("kill"):
            row["action"] = "WATCH"
    man = json.loads((dest / "manifest.json").read_text(encoding="utf-8")) if (dest / "manifest.json").is_file() else {}
    skips = _load_rows(dest, "skips.json")
    write_run(
        dest,
        date=str((xhot or intel or {}).get("asof") or man.get("date") or ""),
        tickets=[r for r in all_rows if r.get("action") == "CLICK"],
        skip=[r for r in all_rows if r.get("action") == "SKIP"],
        watch=[r for r in all_rows if r.get("action") == "WATCH"],
        x_queue=[],
        gates=gates,
        skips=skips,
        manifest=man,
        macro=man.get("macro") if isinstance(man, dict) else {},
    )
