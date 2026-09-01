"""Markdown / JSON artifacts. Never write secrets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from groat.num import fmt, fmt_pct
from groat.regime import render_regime
from groat.rotation import render_rotation
from groat.evidence import render_evidence, render_evidence_file
from groat.picks import render_desk_picks
from groat.setups import SETUP_GUIDE, SETUP_NAMES


BOARD_COLUMNS = [
    "asof_date",
    "ticker",
    "action",
    "lane",
    "choice",
    "direction",
    "primary",
    "score",
    "opt_conf",
    "naive_pop",
    "target_debit",
    "target_credit",
    "close",
    "trend",
    "rs_20",
    "iv30",
    "hv20",
    "vrp",
    "group",
    "group_status",
]


def day_dir(out_dir: Path, date: str) -> Path:
    path = Path(out_dir) / date
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _cell(row.get(col)) for col in columns})


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return fmt(value, 4, unavailable="")
    return str(value)


def _pop_cell(row: dict) -> str:
    picked = row.get("picked") or {}
    pop = None
    if isinstance(picked, dict):
        pop = picked.get("naive_pop")
    if pop is None:
        pop = row.get("naive_pop")
    if pop is None:
        return "n/a"
    return fmt_pct(pop, 0)


def _premium_cell(row: dict) -> str:
    picked = row.get("picked") or {}
    debit = picked.get("target_debit") if isinstance(picked, dict) else row.get("target_debit")
    credit = picked.get("target_credit") if isinstance(picked, dict) else row.get("target_credit")
    if debit is not None:
        return "debit %s" % fmt(debit)
    if credit is not None:
        return "credit %s" % fmt(credit)
    return ""


def _evidence_cell(row: dict) -> str:
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    stock = ev.get("stock") or {}
    opt = ev.get("options") or {}
    n = int(stock.get("n") or 0)
    if n:
        bits = [
            "stock %sW/%sL/%st n=%s avg R %s"
            % (
                stock.get("wins") or 0,
                stock.get("losses") or 0,
                stock.get("time") or 0,
                n,
                fmt(stock.get("avg_r"), 2),
            )
        ]
    else:
        bits = [ev.get("note") or "no same-setup analog"]
    on = int(opt.get("n") or 0)
    if on:
        bits.append(
            "options n=%s P&L/risk %s (spot-delta, not a live mark)"
            % (on, fmt(opt.get("avg_pnl_per_risk"), 2))
        )
    if ev.get("weak"):
        bits.append("weak analog")
    return "; ".join(bits)


def _instrument_line(row: dict) -> str:
    picked = row.get("picked") or {}
    choice = row.get("choice") or "NO TRADE"
    if choice == "STOCK" and picked:
        return "STOCK %s entry %s stop %s target %s R/R %s shares %s" % (
            picked.get("side") or "",
            fmt(picked.get("entry")),
            fmt(picked.get("stop")),
            fmt(picked.get("target")),
            fmt(picked.get("rr")),
            picked.get("shares") or "",
        )
    if choice == "OPTIONS" and picked:
        return "%s %s" % (picked.get("instrument") or "options", picked.get("legs") or "")
    return "NO TRADE"


def _kv(rows: Sequence[tuple]) -> List[str]:
    lines = ["| | |", "|---|---|"]
    for key, value in rows:
        lines.append("| **%s** | %s |" % (key, value))
    lines.append("")
    return lines


def _card(row: dict) -> List[str]:
    earn = row.get("earnings") or {}
    picked = row.get("picked") if isinstance(row.get("picked"), dict) else {}
    thesis = row.get("thesis") or {}
    reviews = row.get("reviews") or []
    setup = row.get("primary") or "—"
    setup_name = SETUP_NAMES.get(setup, "no setup")
    pop_note = picked.get("naive_pop_note") or row.get("naive_pop_note") or "n/a for stock; no delta-based POP"
    earn_txt = "%s (%s)" % (earn.get("date") or "DATA UNAVAILABLE", earn.get("source") or "DATA UNAVAILABLE")
    if earn.get("overlaps_hold"):
        earn_txt += " — ordinary options blocked"
    x_txt = row.get("x") or "DATA UNAVAILABLE"
    if row.get("x_notes"):
        x_txt += " — %s" % row.get("x_notes")
    invalid = picked.get("invalidation") or (thesis.get("invalidation") if thesis else None) or "thesis/setup break"
    paras = thesis.get("paragraphs") or []
    headline = thesis.get("headline") or "%s — %s" % (row.get("ticker"), setup_name)
    lines = [
        "---",
        "",
        "### %s · %s · **%s**" % (row.get("ticker"), row.get("action"), row.get("choice") or "NO TRADE"),
        "",
        headline,
        "",
    ]
    if paras:
        lines.append(paras[0])
        lines.append("")
    lines.extend(
        _kv(
            [
                ("Setup", "%s — %s" % (setup, setup_name)),
                ("Lane", "%s / %s" % (row.get("lane") or "SWING", row.get("direction") or "")),
                ("Last", fmt(row.get("close"))),
                ("Trade", _instrument_line(row)),
                ("Pay / collect", _premium_cell(row) or "stock (no option premium)"),
                ("Naive POP", "%s — %s" % (_pop_cell(row), pop_note)),
                ("Conf", "%s %s (quality of the *structure*, not P(win))"
                 % (row.get("opt_conf") if row.get("opt_conf") is not None else "n/a",
                    row.get("opt_conf_label") or "")),
                ("Score", fmt(row.get("score"), 1)),
                ("Evidence", _evidence_cell(row)),
                ("Stop / invalidation", invalid),
                ("RS 20d vs SPY", fmt_pct(row.get("rs_20"))),
                ("Trend / MAs", "%s · 20 %s / 50 %s / 200 %s"
                 % (row.get("trend") or "", fmt(row.get("ema20")), fmt(row.get("sma50")), fmt(row.get("sma200")))),
                ("Group", "%s (%s) · %s" % (row.get("group"), row.get("etf"), row.get("group_status"))),
                ("AVWAP", "year %s · swing-low %s" % (fmt(row.get("avwap_year")), fmt(row.get("avwap_swing_low")))),
                ("ORATS IV/HV", "IV30 %s · HV20 %s · VRP %s" % (fmt(row.get("iv30")), fmt(row.get("hv20")), fmt(row.get("vrp")))),
                ("Earnings", earn_txt),
                ("X", x_txt),
                ("Fill", picked.get("fill_assumption") or "stock last; options never mid"),
                ("Fill as-of", picked.get("fill_asof") or "n/a — revalidate at the open"),
                ("Greeks", "Δ %s · Γ %s · Θ %s · ν %s"
                 % (
                     fmt(picked.get("delta")),
                     fmt(picked.get("gamma"), 4),
                     fmt(picked.get("theta"), 4),
                     fmt(picked.get("vega"), 4),
                 )),
            ]
        )
    )
    if len(paras) > 1:
        lines.append("More context:")
        lines.append("")
        for para in paras[1:]:
            lines.append(para)
            lines.append("")
    if row.get("setup_notes"):
        lines.append("Tape notes:")
        lines.append("")
        for note in row.get("setup_notes") or []:
            lines.append("- %s" % note)
        lines.append("")
    if reviews:
        lines.append("All structures reviewed:")
        lines.append("")
        lines.append("| structure | result | debit | credit | why |")
        lines.append("|---|---|---:|---:|---|")
        for rev in reviews:
            lines.append(
                "| %s | %s | %s | %s | %s |"
                % (
                    rev.get("strategy") or "",
                    rev.get("status") or "",
                    fmt(rev.get("target_debit")) if rev.get("target_debit") is not None else "—",
                    fmt(rev.get("target_credit")) if rev.get("target_credit") is not None else "—",
                    (rev.get("reason") or "")[:80],
                )
            )
        lines.append("")
    macros = row.get("macros") or []
    if macros:
        lines.append("Macro in hold window: " + "; ".join("%s %s" % (ev.get("date"), ev.get("event")) for ev in macros))
        lines.append("")
    return lines


def render_board(asof: str, built: dict, include_desk_pick: bool = True) -> str:
    trades = built.get("trades") or []
    watch = built.get("watch") or []
    fire = built.get("fire") or []
    lines = [
        "# Groat %s" % asof,
        "",
        "Regime **%s** · TRADE %s · WATCH %s · FIRE %s · X-HOT %s"
        % (
            ((built.get("regime") or {}).get("regime") or "unknown"),
            len(trades),
            len(watch),
            len(fire),
            len(built.get("xhot") or []),
        ),
        "",
        "You click every Schwab order. Empty board is valid. Prefer 1–3 names.",
        "",
    ]
    if include_desk_pick:
        lines.extend(render_desk_picks(built.get("picks") or {}))
        lines.extend(render_evidence(built.get("evidence") or {}))
    lines.extend(
        [
        "## How to read this",
        "",
        "| field | what it is | what it is not |",
        "|---|---|---|",
        "| **setup C** | Post-earnings drift: print already out; price holds earnings AVWAP. | Not an earnings lottery. |",
        "| **setup E** | Emerging sector rotation: the *group* is attracting capital; pick a leader in it. | Not “the stock is a great company.” |",
        "| **conf** | 0–85 quality of the *option structure* (quotes, OI, IV, earnings distance, X). | **Not** probability of profit. |",
        "| **naive POP** | P(spot beyond breakeven) from ORATS call deltas. | **Not** a backtested win rate. Stock = n/a. |",
        "| **score** | Rank of the underlying idea. | Not POP. |",
        "| **FIRE** | Tape first: volume + 1–2 day shock, then X confirms or vetoes. | Not “it’s loud on X.” |",
        "| **X-HOT** | Conversation first: loud on X, then tape says dipped / will_rise / will_dip. | Not a trade by itself. Heat without a trigger is Watch. |",
        "| **evidence** | Same ticker + same setup on cached tape; options via hist/strikes if cached/capped. | **Not** a system win rate. Does not change today’s gates. |",
        "",
        "Other setups: " + "; ".join("**%s** %s" % (k, SETUP_GUIDE[k].split("—")[0].strip()) for k in ("A", "B", "D", "F", "G", "H")),
        "",
        "## TRADE — index",
        "",
        ]
    )
    if not trades:
        lines.append("Empty board. Valid.")
        lines.append("")
    else:
        lines.append("| ticker | setup | vehicle | pay/collect | naive POP | conf | last | book |")
        lines.append("|---|---|---|---|---:|---:|---:|---|")
        for row in trades:
            held = "IN BOOK" if row.get("in_book") else ("held" if row.get("held") else "")
            lines.append(
                "| **%s** | %s %s | **%s** | %s | %s | %s | %s | %s |"
                % (
                    row.get("ticker"),
                    row.get("primary") or "",
                    SETUP_NAMES.get(row.get("primary") or "", ""),
                    row.get("choice"),
                    _premium_cell(row) or "—",
                    _pop_cell(row),
                    row.get("opt_conf") if row.get("opt_conf") is not None else "—",
                    fmt(row.get("close")),
                    held or "—",
                )
            )
        lines.append("")
        lines.append("| ticker | exact structure | stop / invalidation |")
        lines.append("|---|---|---|")
        for row in trades:
            picked = row.get("picked") if isinstance(row.get("picked"), dict) else {}
            lines.append(
                "| **%s** | %s | %s |"
                % (
                    row.get("ticker"),
                    _instrument_line(row),
                    picked.get("invalidation") or "thesis/setup break",
                )
            )
        lines.append("")
    lines.extend(["## WATCH", ""])
    if not watch:
        lines.append("None.")
        lines.append("")
    else:
        lines.append("| ticker | setup | last | RS20 | score | parked because |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in watch:
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    row.get("ticker"),
                    row.get("primary") or "",
                    fmt(row.get("close")),
                    fmt_pct(row.get("rs_20")),
                    fmt(row.get("score"), 1),
                    "; ".join(row.get("reasons") or [])[:70] or "below trade score",
                )
            )
        lines.append("")
    lines.extend(["## FIRE — spike / dip", ""])
    lines.append("Needs volume + a 1–2 day shock **first**. X only confirms or vetoes.")
    lines.append("")
    if not fire:
        lines.append("No FIRE names. Valid.")
        lines.append("")
    else:
        lines.append("| ticker | kind | 1d | rvol | vehicle | pay/collect | board | X |")
        lines.append("|---|---|---:|---:|---|---|---|---|")
        for row in fire:
            fire_info = row.get("fire") or {}
            board_state = "%s" % (row.get("action") or "")
            if row.get("reasons"):
                board_state += " · " + "; ".join(row.get("reasons") or [])[:40]
            lines.append(
                "| **%s** | %s | %s | %s | **%s** | %s | %s | %s |"
                % (
                    row.get("ticker"),
                    fire_info.get("kind") or "",
                    fmt_pct(row.get("ret_1")),
                    fmt(row.get("rvol"), 1),
                    row.get("choice"),
                    _premium_cell(row) or "—",
                    board_state,
                    row.get("x") or "DATA UNAVAILABLE",
                )
            )
        lines.append("")
        lines.append("FIRE names are not auto-TRADE. Parked/IGNORE rows still show the ticket for visibility.")
        lines.append("")
    errors = built.get("schwab_chain_errors") or []
    if errors:
        lines.extend(["## Schwab chain errors", ""])
        for err in errors[:12]:
            lines.append("- **%s**: %s" % (err.get("ticker"), err.get("error")))
        lines.append("")
    xhot = built.get("xhot") or []
    lines.extend(["## X-HOT — conversation first", ""])
    lines.append(
        "Starts from what is loud on X, then asks the tape. "
        "**dipped** = already red with heat (buy-the-dip only if 20 EMA/AVWAP holds). "
        "**will_rise** = bullish X and tape allows continuation or a later entry. "
        "**will_dip** = bearish X, or a spike already extended (the trade is the pullback, not the chase). "
        "Heat without a volume/price trigger is Watch, not a trade."
    )
    lines.append("")
    if not xhot:
        lines.append("X-HOT DATA UNAVAILABLE until `var/xhot/DATE/hot.json` is written (skill searches X).")
        lines.append("")
    else:
        lines.append("| ticker | move | tape | 1d | rvol | X | vehicle |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for row in xhot:
            xh = row.get("xhot") or {}
            lines.append(
                "| **%s** | **%s** | %s | %s | %s | %s / %s | %s |"
                % (
                    row.get("ticker"),
                    xh.get("move") or "noise",
                    xh.get("kind") or "",
                    fmt_pct(row.get("ret_1")),
                    fmt(row.get("rvol"), 1),
                    xh.get("tag") or "",
                    xh.get("bias") or "",
                    row.get("choice") or "",
                )
            )
        lines.append("")
        lines.append("| ticker | play | X narrative |")
        lines.append("|---|---|---|")
        for row in xhot:
            xh = row.get("xhot") or {}
            narr = str(xh.get("narrative") or "").replace("|", "/")
            lines.append(
                "| **%s** | %s | %s |"
                % (row.get("ticker"), xh.get("play") or "", narr)
            )
        lines.append("")
    lines.extend(["## Tickets", ""])
    seen = set()
    for row in list(trades) + list(fire) + list(xhot):
        t = row.get("ticker")
        if t in seen:
            continue
        seen.add(t)
        lines.extend(_card(row))
    return "\n".join(lines)


def render_report(asof: str, built: dict) -> str:
    lines = [
        "# Groat full scan %s" % asof,
        "",
        "You click every Schwab order. Empty board is valid. Prefer 1–3 names.",
        "",
    ]
    lines.extend(render_desk_picks(built.get("picks") or {}))
    lines.extend(render_evidence(built.get("evidence") or {}))
    lines.extend(
        [
            "Hierarchy: market regime → underlying thesis → price/AVWAP/volume → catalyst → relative strength → risk/reward → ORATS vol + structure → positioning → X.",
            "X excitement cannot rescue a bad chart. Never invent ORATS, prices, posts, or news.",
            "",
        ]
    )
    lines.extend(render_regime(built.get("regime") or {}))
    lines.extend(render_rotation(built.get("groups") or []))
    lines.append(render_board(asof, built, include_desk_pick=False))
    lines.extend(
        [
            "",
            "## Data caveats",
            "",
            "- ORATS rows: %s · error: %s"
            % (built.get("orats_rows") or 0, built.get("orats_error") or "none"),
            "- Option chains pulled only for top underlying theses: %s"
            % ", ".join(built.get("option_names") or [])
            or "none",
            "- X is confirm/veto only. Write `var/xintel/DATE/TICKER.json` after searching $TICKER. Missing file → DATA UNAVAILABLE. FIRE needs volume+price first.",
            "- Dealer GEX is not computed. Do not present estimated GEX as fact.",
            "- Conservative option fills: debit at ask, credit at short bid minus long ask. Never assume midpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def render_analyze(asof: str, row: dict) -> str:
    lines = [
        "# Groat analyze %s — %s" % (row.get("ticker"), asof),
        "",
        "Compare **STOCK** vs **BEST OPTIONS** vs **NO TRADE**.",
        "",
    ]
    lines.extend(_card(row))
    stock = row.get("stock") or {}
    opt = row.get("options") or {}
    lines.extend(
        [
            "## Stock trade",
            "",
            json.dumps(json_safe(stock), indent=2) if stock else "DATA UNAVAILABLE / not computed",
            "",
            "## Best options trade",
            "",
            json.dumps(json_safe(opt), indent=2) if opt else "DATA UNAVAILABLE / rejected",
            "",
            "## Decision",
            "",
            "**%s**" % (row.get("choice") or "NO TRADE"),
            "",
        ]
    )
    return "\n".join(lines)


def render_delta(asof: str, delta: dict) -> str:
    lines = [
        "# Groat delta scan %s" % asof,
        "",
        "What changed vs the prior scan. Not a full rewrite.",
        "",
        "## Newly actionable / changed",
        "",
    ]
    if not delta.get("changes"):
        lines.append("None.")
        lines.append("")
    else:
        for row in delta.get("changes") or []:
            lines.append("- **%s** (%s): %s" % (row.get("ticker"), row.get("kind"), row.get("detail")))
        lines.append("")
    lines.extend(["## Removed / invalidated", ""])
    if not delta.get("removed"):
        lines.append("None.")
        lines.append("")
    else:
        for row in delta.get("removed") or []:
            lines.append("- **%s**: %s" % (row.get("ticker"), row.get("detail")))
        lines.append("")
    return "\n".join(lines)


def render_review(asof: str, rows: List[dict]) -> str:
    lines = [
        "# Groat review open trades %s" % asof,
        "",
        "Re-evaluate the ORIGINAL thesis. Do not rationalize a loser with unrelated new information.",
        "",
    ]
    if not rows:
        lines.append("No open Groat positions on file (`configs/book.json`).")
        lines.append("")
        return "\n".join(lines)
    lines.append("| ticker | instrument | entry | stop | last | verdict | why |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | **%s** | %s |"
            % (
                row.get("ticker"),
                row.get("instrument") or "",
                fmt(row.get("entry")),
                fmt(row.get("stop")),
                fmt(row.get("last")),
                row.get("verdict") or "HOLD",
                row.get("why") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_scan_artifacts(day: Path, asof: str, built: dict) -> None:
    slim = []
    for row in built.get("candidates") or []:
        slim.append({k: v for k, v in row.items() if k not in ("stock", "options") or row.get("action") != "IGNORE"})
    write_json(day / "candidates.json", {"asof": asof, "regime": (built.get("regime") or {}).get("regime"), "candidates": slim, "board": built.get("board")})
    write_csv(day / "board.csv", BOARD_COLUMNS, built.get("board") or [])
    write_csv(day / "rejections.csv", ["asof_date", "ticker", "reasons", "stage"], built.get("rejections") or [])
    write_text(day / "board.md", render_board(asof, built))
    queue = []
    for row in list(built.get("trades") or []) + list(built.get("fire") or []) + list(built.get("xhot") or []):
        t = row.get("ticker")
        if t and t not in queue:
            queue.append(t)
    write_json(
        day / "x_queue.json",
        {
            "asof": asof,
            "tickers": queue,
            "note": "1) Market heat → var/xhot/DATE/hot.json. 2) $TICKER → var/xintel/DATE/TICKER.json tag Quiet|Informed|Crowded. Do not invent posts.",
        },
    )
    write_text(day / "regime.md", "\n".join(render_regime(built.get("regime") or {})))
    write_text(day / "sectors.md", "\n".join(render_rotation(built.get("groups") or [])))
    write_text(day / "report.md", render_report(asof, built))
    write_text(day / "evidence.md", render_evidence_file(asof, built.get("evidence") or {}))
    write_json(day / "evidence.json", built.get("evidence") or {})
