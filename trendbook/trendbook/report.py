"""Action board: ADD / NEW / HOLD / LATE / OUT. Grade is structure, not P(win)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from trendbook.num import fmt, fmt_pct, to_float

ADD_WHY = {"breakout": 0, "pullback_and_tight": 1, "pullback": 2, "pullback_30w": 3}
GRADE_ORDER = {"A": 0, "B": 1, "C": 2}


def _why(row: dict) -> str:
    return str(row.get("entry_reason") or row.get("entry") or "")


def sort_rows(rows: List[dict]) -> List[dict]:
    def key(r):
        act = str(r.get("action") or r.get("status") or "OUT")
        order = {"ADD": 0, "NEW": 1, "HOLD": 2, "LATE": 3, "OUT": 4}.get(act, 9)
        weeks = to_float(r.get("weeks_in_trend")) or 0
        pct = to_float(r.get("pct_from_start")) or -1
        why = ADD_WHY.get(str(r.get("entry_reason") or ""), 9)
        g = GRADE_ORDER.get(str(r.get("grade") or ""), 9)
        if act == "ADD":
            return (order, g, why, -weeks, -pct)
        return (order, g, -weeks, -pct)

    return sorted(rows, key=key)


def _grade(row: dict) -> str:
    g = row.get("grade")
    return str(g) if g else "—"


def render_board(asof: str, rows: List[dict], meta: dict) -> str:
    add = [r for r in rows if r.get("action") == "ADD"]
    new = [r for r in rows if r.get("action") == "NEW"]
    hold = [r for r in rows if r.get("action") == "HOLD"]
    late = [r for r in rows if r.get("action") == "LATE"]
    out = [r for r in rows if r.get("action") == "OUT" and r.get("had_trend")]
    spy_stage = meta.get("spy_stage")
    lines = [
        "# Trendbook %s" % asof,
        "",
        "SPY stage %s. ADD %s · NEW %s · HOLD %s · LATE %s · OUT %s"
        % (
            spy_stage if spy_stage is not None else "DATA UNAVAILABLE",
            len(add),
            len(new),
            len(hold),
            len(late),
            len(out),
        ),
        "",
        "Grade is campaign structure, not P(win). A = long, still expanding, near highs. C = weak tag, stalled, or messy.",
        "ADD = first ticket this campaign: a Stage 2 breakout from a base (Stage 1/4, or a 1–2 week Stage 3 poke still near the 30-week) with break-week volume, or the first early pullback. A Stage 3 MA-turn after the stock already ran is NEW, not a buy. One ADD per campaign. NEW = weak tag. HOLD = in trend or entry already used. LATE = old and dying.",
        "Empty ADD is valid.",
        "",
        "## ADD — buy stock this week",
        "",
    ]
    if not add:
        lines.append("No ADD rows.")
        lines.append("")
    else:
        lines.extend(
            [
                "| ticker | last | weeks | resid 3m | grade | size | stop | why |",
                "|---|---:|---:|---:|---|---|---:|---|",
            ]
        )
        for r in add:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    r.get("ticker"),
                    fmt(r.get("close")),
                    r.get("weeks_in_trend") or "—",
                    fmt_pct(r.get("residual_63"), 0),
                    _grade(r),
                    r.get("size") or "full",
                    fmt((r.get("invalidation") or {}).get("stop")),
                    _why(r),
                )
            )
        lines.append("")
        lines.append("Buy the stock. One ticket per campaign. Stop = weekly close below the 30-week. Prefer full-size A/B. Half-size in a tight regime.")
        lines.append("")

    lines.extend(["## NEW — weak Stage 2 tag, not yet a trade", ""])
    if not new:
        lines.append("No NEW rows.")
        lines.append("")
    else:
        lines.extend(
            [
                "| ticker | last | weeks | from start | vs SPY 6m | source |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for r in new:
            src = "probe" if r.get("discovered") else "universe"
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    r.get("ticker"),
                    fmt(r.get("close")),
                    r.get("weeks_in_trend") or "—",
                    fmt_pct(r.get("pct_from_start"), 0),
                    fmt_pct(r.get("rs_126"), 0),
                    src,
                )
            )
        lines.append("")
        lines.append("Watch. Becomes ADD if the next weeks still Stage 2 with RS rising, or on a real pullback. Not an 8-week wait.")
        lines.append("")

    lines.extend(["## HOLD — in trend, do not chase", ""])
    if not hold:
        lines.append("No HOLD rows.")
        lines.append("")
    else:
        lines.extend(
            [
                "| ticker | last | weeks | from start | vs SPY 6m | grade | note |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for r in hold:
            note = _why(r)
            if r.get("entry") == "broken":
                note = "daily damage — do not add"
            elif r.get("entry") == "extended":
                note = "extended — do not chase"
            elif str(r.get("entry_reason") or "") == "rest_20ema":
                note = "pause at 20 EMA — not a dip"
            elif str(r.get("entry_reason") or "") == "tightness":
                note = "coiled — wait for a pullback"
            elif str(r.get("setup") or "") == "entry_window_used":
                note = "already bought this campaign"
            elif str(r.get("setup") or "") == "regime_off":
                note = "index not Stage 2 — no new money"
            elif str(r.get("setup") or "") == "regime_tight":
                note = "tight regime — residual too small"
            elif str(r.get("setup") or "") == "earnings":
                note = "earnings inside 10 days"
            elif note in ("held", "in_trend_wait"):
                note = "no setup this week"
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    r.get("ticker"),
                    fmt(r.get("close")),
                    r.get("weeks_in_trend") or "—",
                    fmt_pct(r.get("pct_from_start"), 0),
                    fmt_pct(r.get("rs_126"), 0),
                    _grade(r),
                    note,
                )
            )
        lines.append("")

    lines.extend(["## LATE — old and dying, no new money", ""])
    if not late:
        lines.append("No LATE rows.")
        lines.append("")
    else:
        lines.extend(
            [
                "| ticker | last | weeks | from start | last 13w | off high |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for r in late:
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    r.get("ticker"),
                    fmt(r.get("close")),
                    r.get("weeks_in_trend") or "—",
                    fmt_pct(r.get("pct_from_start"), 0),
                    fmt_pct(r.get("ret_13w"), 0),
                    fmt_pct(r.get("off_high"), 0),
                )
            )
        lines.append("")

    lines.extend(["## OUT — left the trend", ""])
    if not out:
        lines.append("No names left a prior trend this lookback.")
        lines.append("")
    else:
        lines.extend(
            [
                "| ticker | last | last trend start | last trend end |",
                "|---|---:|---|---|",
            ]
        )
        for r in out[:20]:
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    r.get("ticker"),
                    fmt(r.get("close")),
                    r.get("last_trend_start") or "—",
                    r.get("last_trend_end") or "—",
                )
            )
        lines.append("")
    return "\n".join(lines)


def render_regime(asof: str, meta: dict, spy_feat: dict) -> str:
    st = (spy_feat.get("stage") or {}) if spy_feat else {}
    daily = (spy_feat.get("daily") or {}) if spy_feat else {}
    lines = [
        "# Regime %s" % asof,
        "",
        "| field | value |",
        "|---|---|",
        "| SPY stage | %s |" % (st.get("stage") if st.get("stage") is not None else "DATA UNAVAILABLE"),
        "| TLT stage | %s |" % (meta.get("tlt_stage") if meta.get("tlt_stage") is not None else "DATA UNAVAILABLE"),
        "| UUP stage | %s |" % (meta.get("uup_stage") if meta.get("uup_stage") is not None else "DATA UNAVAILABLE"),
        "| regime | %s |" % (meta.get("regime") or "DATA UNAVAILABLE"),
        "| SPY 30w MA | %s |" % fmt(st.get("ma30")),
        "| SPY close | %s |" % fmt(daily.get("close")),
        "| live Schwab | %s |" % ("yes" if meta.get("live_schwab") else "no"),
        "| ORATS | %s |" % ("yes" if meta.get("orats") else "missing"),
        "| Schwab HTTP | %s |" % (meta.get("schwab_http") or 0),
        "| ORATS HTTP | %s |" % (meta.get("orats_http") or 0),
        "| universe | %s |" % (meta.get("universe") or 0),
        "| probe | %s |" % (meta.get("n_probe") or 0),
        "",
    ]
    missing = meta.get("missing_bars") or []
    if missing:
        lines.append("Missing bars: %s" % ", ".join(missing))
        lines.append("")
    return "\n".join(lines)


def render_evidence(asof: str, rows: List[dict]) -> str:
    lines = [
        "# Evidence %s" % asof,
        "",
        "Action board is ADD/NEW/HOLD/LATE/OUT. Grade is structure. RS% is sort order only. Resid 3m is vs SPY after beta.",
        "",
        "| ticker | action | grade | stage | weeks | from start | off high | RS% | resid 3m | vol | Mansfield | start |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        if r.get("action") == "OUT" and not r.get("had_trend"):
            continue
        vol = r.get("break_vol_expand")
        if vol is None:
            vol = r.get("vol_expand")
        vol_s = "yes" if vol is True else ("no" if vol is False else "—")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r.get("ticker"),
                r.get("action"),
                _grade(r),
                r.get("stage") if r.get("stage") is not None else "—",
                r.get("weeks_in_trend") if r.get("weeks_in_trend") is not None else "—",
                fmt_pct(r.get("pct_from_start"), 0),
                fmt_pct(r.get("off_high"), 0),
                fmt(r.get("rs_pctile"), 0),
                fmt_pct(r.get("residual_63"), 0),
                vol_s,
                fmt_pct(r.get("mansfield_spy"), 1),
                r.get("trend_start") or "—",
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_replay(asof: str, rows: List[dict]) -> str:
    tagged = [r for r in rows if r.get("tagged")]
    lines = [
        "# Universe replay %s" % asof,
        "",
        "First week of the latest campaign (current or last ended) that tagged Stage 2 and was beating SPY, then 4/8/13-week forward return from that close. Not the first tag in the five-year tape. No future bars in the tag.",
        "",
        "Tagged %s of %s names." % (len(tagged), len(rows)),
        "",
        "| ticker | first tag | px then | now | 4w | 8w | 13w | weeks now |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    tagged_sorted = sorted(tagged, key=lambda r: to_float(r.get("fwd_13w")) or -99, reverse=True)
    for r in tagged_sorted:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r.get("ticker"),
                r.get("first_tag") or "—",
                fmt(r.get("first_px")),
                fmt(r.get("now_px")),
                fmt_pct(r.get("fwd_4w"), 0),
                fmt_pct(r.get("fwd_8w"), 0),
                fmt_pct(r.get("fwd_13w"), 0),
                r.get("weeks_in_trend") or 0,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_replay_csv(path: Path, rows: List[dict]) -> None:
    fields = [
        "ticker",
        "tagged",
        "first_tag",
        "first_px",
        "now_px",
        "fwd_4w",
        "fwd_8w",
        "fwd_13w",
        "weeks_in_trend",
        "trend_start",
        "on_board_now",
        "campaign_high",
        "off_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_run(
    day: Path,
    asof: str,
    rows: List[dict],
    meta: dict,
    spy_feat: dict,
    replay_rows: Optional[List[dict]] = None,
    outcomes_md: str = "",
) -> Dict[str, str]:
    day.mkdir(parents=True, exist_ok=True)
    ordered = sort_rows(rows)
    (day / "board.md").write_text(render_board(asof, ordered, meta), encoding="utf-8")
    (day / "regime.md").write_text(render_regime(asof, meta, spy_feat), encoding="utf-8")
    (day / "evidence.md").write_text(render_evidence(asof, ordered), encoding="utf-8")
    slim = []
    for r in ordered:
        slim.append({k: v for k, v in r.items() if k not in ("template", "core", "weekly", "flags")})
    (day / "board.json").write_text(json.dumps({"meta": meta, "rows": slim}, indent=2) + "\n", encoding="utf-8")
    files = {
        "board": str(day / "board.md"),
        "regime": str(day / "regime.md"),
        "evidence": str(day / "evidence.md"),
        "json": str(day / "board.json"),
    }
    if replay_rows is not None:
        (day / "replay.md").write_text(render_replay(asof, replay_rows), encoding="utf-8")
        write_replay_csv(day / "replay.csv", replay_rows)
        files["replay"] = str(day / "replay.md")
        files["replay_csv"] = str(day / "replay.csv")
    if outcomes_md:
        (day / "outcomes.md").write_text(outcomes_md, encoding="utf-8")
        files["outcomes"] = str(day / "outcomes.md")
    return files
