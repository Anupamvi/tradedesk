"""Manage existing breakout_eq. HOLD / CLOSE. No orders."""

from typing import List, Optional

from groki_eq.config import TIME_STOP_SESSIONS


def render_manage_md(date: str, live: bool, rows: Optional[List[dict]] = None, note: str = "") -> str:
    lines = ["# manage-existing %s" % date, ""]
    if not live:
        lines.append("No live broker on this historical as-of.")
        lines.append("Exit rules: stop = entry − 2×ATR(14); time stop = %d sessions." % TIME_STOP_SESSIONS)
        if note:
            lines.append(note)
        lines.append("")
        return "\n".join(lines)
    if note:
        lines.append(note)
        lines.append("")
    rows = rows or []
    if not rows:
        lines.append("No open breakout_eq positions.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| ticker | entry | stop | sessions | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                row.get("ticker") or "",
                row.get("entry") or "",
                row.get("stop") or "",
                row.get("sessions") or "",
                row.get("verdict") or "HOLD",
            )
        )
    lines.append("")
    lines.append("User clicks every Schwab order. No submit/cancel/replace.")
    lines.append("")
    return "\n".join(lines)
