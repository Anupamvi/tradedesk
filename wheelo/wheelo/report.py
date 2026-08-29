"""Markdown / JSON artifacts. Never write secrets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from wheelo.num import fmt


def day_dir(out_dir: Path, date: str) -> Path:
    path = Path(out_dir) / date
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items() if k != "raw"}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in columns})


BOARD_COLUMNS = [
    "ticker",
    "tier",
    "allocated",
    "spot",
    "csp_strike",
    "csp_bid",
    "expiry",
    "dte",
    "csp_yield_ann",
    "quality",
    "premium",
    "composite",
    "conf",
    "conf_label",
    "credit_pct",
    "otm_pct",
    "capital",
    "contracts",
    "x_status",
]


def _conf_row(cand: dict) -> str:
    prem = cand.get("premium") or {}
    cr = cand.get("credit_pct")
    otm = cand.get("otm_pct")
    return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        cand.get("conf") if cand.get("conf") is not None else "",
        cand.get("conf_label") or "",
        cand.get("ticker"),
        fmt(cand.get("spot")),
        fmt(prem.get("csp_strike")),
        fmt(prem.get("csp_bid")),
        (fmt(100.0 * cr, 2) + "%") if cr is not None else "",
        (fmt(100.0 * otm, 1) + "%") if otm is not None else "",
        prem.get("expiry") or "",
        prem.get("dte") or "",
        fmt(prem.get("iv_rank"), 0),
        cand.get("x_status") or "DATA UNAVAILABLE",
    )


def rotation_pick(candidates: List[dict]) -> Optional[dict]:
    trades = [c for c in candidates if c.get("conf_label") == "TRADE"]
    if not trades:
        return None
    trades.sort(key=lambda c: (c.get("conf") or 0, c.get("credit_pct") or 0), reverse=True)
    return trades[0]


def render_board(asof: str, candidates: List[dict], capital: float, manifest: dict) -> str:
    pick = rotation_pick(candidates)
    lines = [
        "# Wheelo %s" % asof,
        "",
        "orats_http %s | shortlist A/B/C %s/%s/%s"
        % (
            manifest.get("orats_http") or 0,
            manifest.get("shortlist_a") or 0,
            manifest.get("shortlist_b") or 0,
            manifest.get("shortlist_c") or 0,
        ),
        "",
        "Credits are **put bid**. **conf** is structure/research quality 0-85, not P(win). TRADE requires known earnings after expiry, 2-15% OTM, credit >=1.5% of strike, and not cheap vol.",
        "",
    ]
    if pick:
        prem = pick.get("premium") or {}
        cr = pick.get("credit_pct")
        otm = pick.get("otm_pct")
        lines.append(
            "**Rotation pick:** %s  %sP %s @ %s bid  |  conf %s TRADE  |  Cr %s  OTM %s  IVR %s"
            % (
                pick.get("ticker"),
                fmt(prem.get("csp_strike")),
                prem.get("expiry") or "",
                fmt(prem.get("csp_bid")),
                pick.get("conf"),
                (fmt(100.0 * cr, 2) + "%") if cr is not None else "DATA UNAVAILABLE",
                (fmt(100.0 * otm, 1) + "%") if otm is not None else "DATA UNAVAILABLE",
                fmt(prem.get("iv_rank"), 0),
            )
        )
        drivers = pick.get("conf_drivers") or []
        if drivers:
            lines.append("Why: %s" % "; ".join(str(d) for d in drivers[:8]))
        lines.append("")
    else:
        lines.append("**Rotation pick:** none. Empty TRADE board is valid. Do not loosen filters.")
        lines.append("")
    by_label = {"TRADE": [], "WATCH": [], "NO_TRADE": []}
    for cand in candidates:
        label = str(cand.get("conf_label") or "NO_TRADE")
        if label not in by_label:
            label = "NO_TRADE"
        by_label[label].append(cand)
    for label in ("TRADE", "WATCH", "NO_TRADE"):
        rows = by_label.get(label) or []
        rows.sort(key=lambda c: (c.get("conf") or 0, c.get("credit_pct") or 0), reverse=True)
        if not rows:
            continue
        lines.append("## %s" % label)
        lines.append("| Conf | Label | Ticker | Spot | Put | Bid | Cr% | OTM | Expiry | DTE | IVR | X |")
        lines.append("|------|-------|--------|------|-----|-----|-----|-----|--------|-----|-----|---|")
        for cand in rows:
            lines.append(_conf_row(cand))
        lines.append("")
    return "\n".join(lines)


def render_report(asof: str, candidates: List[dict], rejections: List[dict], manifest: dict, capital: float) -> str:
    lines = [render_board(asof, candidates, capital, manifest)]
    lines.append("## Rejections")
    if not rejections:
        lines.append("None.")
    else:
        lines.append("| Ticker | Stage | Reason |")
        lines.append("|--------|-------|--------|")
        for row in rejections[:80]:
            lines.append("| %s | %s | %s |" % (row.get("ticker"), row.get("stage"), row.get("reason")))
    lines.append("")
    lines.append("## Manifest")
    lines.append("- orats_http: %s" % (manifest.get("orats_http") or 0))
    lines.append("- orats_planned: %s" % (manifest.get("orats_planned") or 0))
    lines.append("- cache_hits: %s" % (manifest.get("cache_hits") or 0))
    lines.append("- schwab: %s" % ("on" if manifest.get("schwab") else "off"))
    lines.append("- error: %s" % (manifest.get("error") or ""))
    lines.append("")
    return "\n".join(lines)


def render_daily(asof: str, actions: List[dict], positions: List[dict]) -> str:
    lines = ["# Wheelo daily %s" % asof, ""]
    if not positions:
        lines.append("No open wheelo positions.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| Ticker | Phase | Action | P/L | Detail |")
    lines.append("|--------|-------|--------|-----|--------|")
    for act in actions:
        pnl = act.get("pnl_pct")
        pnl_s = fmt(100.0 * pnl, 0) + "%" if pnl is not None else ""
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (act.get("ticker"), act.get("phase"), act.get("action"), pnl_s, act.get("detail") or "")
        )
    lines.append("")
    return "\n".join(lines)
