"""Markdown / JSON artifacts. Never write secrets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from groat.num import fmt, fmt_pct, to_float
from groat.regime import render_regime
from groat.rotation import render_rotation
from groat.evidence import render_evidence_file
from groat.picks import render_desk_picks
from groat.setups import SETUP_LINE, setup_label
from groat.xintel import missing_x_tickers


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


def _picked(row: dict) -> dict:
    p = row.get("picked")
    return p if isinstance(p, dict) else {}


def _pop_cell(row: dict) -> str:
    picked = _picked(row)
    pop = picked.get("naive_pop")
    if pop is None:
        pop = row.get("naive_pop")
    if pop is None:
        return "n/a"
    return fmt_pct(pop, 0)


def _premium_cell(row: dict) -> str:
    picked = _picked(row)
    if row.get("choice") == "STOCK":
        side = picked.get("side") or "stock"
        return "%s @ %s" % (side, fmt(picked.get("entry") or row.get("close")))
    debit = picked.get("target_debit") if picked else row.get("target_debit")
    credit = picked.get("target_credit") if picked else row.get("target_credit")
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


def _strategy_cell(row: dict) -> str:
    inst = str(_picked(row).get("instrument") or row.get("choice") or "")
    names = {
        "debit_call_spread": "call debit",
        "debit_put_spread": "put debit",
        "put_credit_spread": "put credit",
        "call_credit_spread": "call credit",
        "long_call": "long call",
        "long_put": "long put",
        "stock": "stock",
    }
    return names.get(inst, inst.replace("_", " ") or "—")


def _strikes_cell(row: dict) -> str:
    p = _picked(row)
    if row.get("choice") == "STOCK":
        return "stock"
    long_k = p.get("long_strike") if p.get("long_strike") is not None else p.get("strike")
    short_k = p.get("short_strike")
    if long_k is not None and short_k is not None:
        return "%s / %s" % (fmt(long_k, 1), fmt(short_k, 1))
    if long_k is not None:
        return fmt(long_k, 1)
    return "—"


def _exp_cell(row: dict) -> str:
    exp = str(_picked(row).get("expiry") or "")[:10]
    return exp or "—"


def _enter_band(row: dict) -> dict:
    """Stock level you must not cross to click this ticket."""
    p = _picked(row)
    guard = row.get("fill_guard") if isinstance(row.get("fill_guard"), dict) else {}
    if not guard:
        guard = p.get("fill_guard") if isinstance(p.get("fill_guard"), dict) else {}
    last = to_float(row.get("close"))
    direction = str(row.get("direction") or "")
    floor = to_float(guard.get("stock_min"))
    ceil = to_float(guard.get("stock_max"))
    if row.get("choice") == "STOCK":
        stop = to_float(p.get("stop"))
        if stop is None:
            return {"icon": "⚪", "text": "—", "ok": None}
        if "bear" in direction or str(p.get("side") or "") == "short":
            ok = last is None or last < stop
            return {
                "icon": "🟢" if ok else "🔴",
                "text": "skip if last > **%s**" % fmt(stop),
                "ok": ok,
            }
        ok = last is None or last > stop
        return {
            "icon": "🟢" if ok else "🔴",
            "text": "skip if last < **%s**" % fmt(stop),
            "ok": ok,
        }
    if floor is None and direction != "bearish":
        floor = to_float(row.get("ema20"))
    if ceil is None and direction == "bearish":
        ceil = to_float(row.get("ema20"))
    if floor is not None:
        text = "skip if last < **%s**" % fmt(floor)
        if last is None:
            return {"icon": "⚪", "text": text, "ok": None}
        ok = last >= floor
        near = ok and floor > 0 and (last - floor) / floor <= 0.005
        icon = "🟢" if ok and not near else ("🟡" if near else "🔴")
        return {"icon": icon, "text": text, "ok": ok}
    if ceil is not None:
        text = "skip if last > **%s**" % fmt(ceil)
        if last is None:
            return {"icon": "⚪", "text": text, "ok": None}
        ok = last <= ceil
        near = ok and ceil > 0 and (ceil - last) / ceil <= 0.005
        icon = "🟢" if ok and not near else ("🟡" if near else "🔴")
        return {"icon": icon, "text": text, "ok": ok}
    return {"icon": "⚪", "text": "—", "ok": None}


def _x_cell(row: dict) -> str:
    tag = str(row.get("x") or "DATA UNAVAILABLE")
    if tag in ("", "DATA UNAVAILABLE"):
        return "⚪ X missing"
    if tag == "Crowded":
        return "🔴 Crowded"
    if tag == "Informed":
        return "🟢 Informed"
    if tag == "Quiet":
        return "🟡 Quiet"
    return "⚪ %s" % tag


def _park_label(reason: str) -> str:
    labels = {
        "analog_0win_veto": "analog 0-win",
        "analog_fast_stop_veto": "analog fast-stop",
        "already_held_calls": "already hold calls",
        "already_held_puts": "already hold puts",
        "below_20ema": "below 20 EMA",
        "below_trade_score": "score short",
        "score_below_watch": "score too low",
        "same_group_in_book": "same group as book",
        "already_in_book": "already in book",
        "already_recommended": "already recommended",
        "crowded_no_dip": "Crowded, no dip",
        "session_incomplete": "session incomplete",
        "regime_unknown": "regime unknown",
        "analog_persist": "analog persist",
        "setup_B_replay_park": "breakout parked",
        "setup_C_replay_park": "post-earnings parked",
        "setup_G_replay_park": "breakdown parked",
        "setup_H_replay_park": "FIRE parked",
        "setup_D_post_rip": "too extended",
        "setup_E_post_rip": "too extended",
    }
    return labels.get(reason, reason.replace("_", " ") if reason else "—")


def _ticket_cell(row: dict) -> str:
    if row.get("choice") == "STOCK":
        return _premium_cell(row) or "stock"
    return "%s %s · %s" % (_strategy_cell(row), _strikes_cell(row), _exp_cell(row))


def _ticket_table(rows: List[dict], parked: bool = False) -> List[str]:
    head = "| | ticker | setup | ticket | pay | last | click | X |"
    rule = "|---|---|---|---|---|---:|---|---|"
    if parked:
        head += " why not |"
        rule += "---|"
    lines = [head, rule]
    for row in rows:
        band = _enter_band(row)
        action = str(row.get("action") or "")
        mark = "🟢" if action == "TRADE" else ("🟡" if action == "WATCH" else "⚪")
        cells = [
            mark,
            "**%s**" % (row.get("ticker") or ""),
            setup_label(row.get("primary")),
            _ticket_cell(row),
            _premium_cell(row) or ("stock" if row.get("choice") == "STOCK" else "—"),
            fmt(row.get("close")),
            "%s %s" % (band["icon"], band["text"]),
            _x_cell(row),
        ]
        if parked:
            cells.append(_park_label((row.get("reasons") or ["—"])[0]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _kv(rows: Sequence[tuple]) -> List[str]:
    lines = ["| | |", "|---|---|"]
    for key, value in rows:
        lines.append("| **%s** | %s |" % (key, value))
    lines.append("")
    return lines


def _card(row: dict) -> List[str]:
    picked = _picked(row)
    setup_name = setup_label(row.get("primary")) if row.get("primary") else "no setup"
    band = _enter_band(row)
    debit = picked.get("target_debit")
    credit = picked.get("target_credit")
    pay = _premium_cell(row) or ("stock" if row.get("choice") == "STOCK" else "—")
    lines = [
        "---",
        "",
        "### %s %s · **%s** · %s" % (
            "🟢" if row.get("action") == "TRADE" else "🟡",
            row.get("ticker"),
            _strategy_cell(row),
            _x_cell(row),
        ),
        "",
        "%s. Last **%s**."
        % (setup_name, fmt(row.get("close"))),
        "",
    ]
    kv = [
        ("Setup", setup_name),
        ("Strategy", _strategy_cell(row)),
        ("Strikes", _strikes_cell(row)),
        ("Expiry", _exp_cell(row)),
        ("Pay", pay),
        ("Last", fmt(row.get("close"))),
        ("Don't enter", "%s %s" % (band["icon"], band["text"])),
        ("Debit max" if debit is not None else "Credit min", fmt(debit) if debit is not None else fmt(credit) if credit is not None else "—"),
        ("Naive POP / conf", "%s / %s" % (_pop_cell(row), row.get("opt_conf") if row.get("opt_conf") is not None else "—")),
    ]
    if row.get("book_group_note"):
        kv.append(("Book overlap", row.get("book_group_note")))
    lines.extend(_kv(kv))
    return lines


def _counts_line(built: dict) -> str:
    return "Regime **%s** · TRADE %s · WATCH %s · FIRE %s · X-HOT %s" % (
        ((built.get("regime") or {}).get("regime") or "unknown"),
        len(built.get("trades") or []),
        len(built.get("watch") or []),
        len(built.get("fire") or []),
        len(built.get("xhot") or []),
    )


def _legend() -> List[str]:
    return [
        "You click every Schwab order. Empty TRADE is valid.",
        "",
        "Click: 🟢 last is clear · 🟡 within 0.5% · 🔴 already through — do not click. **Pay** is max debit / min credit.",
        "X: 🟢 Informed · 🟡 Quiet · 🔴 Crowded · ⚪ missing (do not treat missing as Quiet).",
        "",
        SETUP_LINE,
        "",
    ]


def _alerts(built: dict) -> List[str]:
    lines: List[str] = []
    session = str(built.get("session") or "")
    if session == "open":
        lines.append("Open auction (before 9:45 ET). New TRADE is blocked. Re-run after the open.")
        lines.append("")
    elif built.get("session_incomplete"):
        lines.append(
            "Session volume is incomplete (median rvol %s). FIRE / 1d ranks are not final. TRADE is still allowed."
            % (fmt(built.get("median_rvol"), 2) if built.get("median_rvol") is not None else "n/a")
        )
        lines.append("")
    if str((built.get("regime") or {}).get("regime") or "") == "unknown":
        lines.append("Regime **unknown** — default NO TRADE. Names below are WATCH if they cleared structure.")
        lines.append("")
    if built.get("session"):
        lines.append("Session **%s**." % built.get("session"))
        lines.append("")
    missing_x = missing_x_tickers(built.get("trades") or [])
    if missing_x:
        lines.append("⚠️ X missing on TRADE: **%s**. Search $TICKER and write `var/xintel/` before clicking." % ", ".join(missing_x))
        lines.append("")
    if built.get("analog_options_unpriced"):
        lines.append("Analog option hist/strikes were not priced this run. Stock analog still stands. Do not invent option P&L.")
        lines.append("")
    if built.get("schwab_pos_error"):
        lines.append("Schwab positions: %s" % built.get("schwab_pos_error"))
        lines.append("")
    empty = list(built.get("chain_empty") or [])
    if empty:
        lines.append("Chain fetch empty: " + ", ".join(empty[:20]) + ".")
        lines.append("")
    for err in (built.get("schwab_chain_errors") or [])[:8]:
        lines.append("- chain **%s**: %s" % (err.get("ticker"), err.get("error")))
    tape_errors = built.get("tape_errors") or []
    for err in tape_errors[:8]:
        lines.append("- tape **%s**: %s" % (err.get("ticker"), err.get("error")))
    if (built.get("schwab_chain_errors") or []) or tape_errors:
        lines.append("")
    return lines


def _board_body(built: dict) -> List[str]:
    trades = built.get("trades") or []
    watch = built.get("watch") or []
    fire = built.get("fire") or []
    xhot = built.get("xhot") or []
    lines: List[str] = ["## TRADE", ""]
    if not trades:
        lines.append("Empty. Valid.")
        lines.append("")
    else:
        lines.extend(_ticket_table(trades))
        overlap = [r.get("ticker") for r in trades if r.get("book_group_held") and r.get("ticker")]
        if overlap:
            lines.append(
                "Caveat: **%s** — same group as an open book name. TRADE. Your call whether to add a lot."
                % ", ".join(overlap)
            )
            lines.append("")
        for row in trades:
            lines.extend(_card(row))
            lines.append("")
    lines.extend(["## WATCH", ""])
    if not watch:
        lines.append("None.")
        lines.append("")
    else:
        lines.extend(_ticket_table(watch, parked=True))
    lines.extend(["## FIRE", ""])
    lines.append("Tape first. Not auto-TRADE.")
    lines.append("")
    if not fire:
        lines.append("None.")
        lines.append("")
    else:
        lines.extend(_ticket_table(fire, parked=True))
    lines.extend(["## X-HOT", ""])
    lines.append("Conversation first. Not a trade by itself.")
    lines.append("")
    if not xhot:
        lines.append("X-HOT DATA UNAVAILABLE until `var/xhot/DATE/hot.json` is written.")
        lines.append("")
    else:
        lines.append("| ticker | move | last | 1d | rvol | board |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in xhot:
            xh = row.get("xhot") or {}
            lines.append(
                "| **%s** | **%s** | %s | %s | %s | %s |"
                % (
                    row.get("ticker"),
                    xh.get("move") or "noise",
                    fmt(row.get("close")),
                    fmt_pct(row.get("ret_1")),
                    fmt(row.get("rvol"), 1),
                    row.get("action") or "",
                )
            )
        lines.append("")
    return lines


def render_board(asof: str, built: dict, include_desk_pick: bool = True) -> str:
    lines = [
        "# Groat %s" % asof,
        "",
        _counts_line(built),
        "",
    ]
    lines.extend(_legend())
    lines.extend(_alerts(built))
    if include_desk_pick:
        lines.extend(render_desk_picks(built.get("picks") or {}))
    lines.extend(_board_body(built))
    return "\n".join(lines)


def render_report(asof: str, built: dict) -> str:
    lines = [
        "# Groat full scan %s" % asof,
        "",
        _counts_line(built),
        "",
    ]
    lines.extend(_legend())
    lines.extend(_alerts(built))
    lines.extend(render_desk_picks(built.get("picks") or {}))
    lines.extend(_board_body(built))
    lines.extend(
        [
            "## Data",
            "",
            "- ORATS rows: %s · error: %s"
            % (built.get("orats_rows") or 0, built.get("orats_error") or "none"),
            "- Chain fetch empty: %s" % (", ".join(built.get("chain_empty") or []) or "none"),
            "- Debit at ask, credit at short bid minus long ask. Never mid.",
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
    write_json(
        day / "candidates.json",
        {
            "asof": asof,
            "regime": (built.get("regime") or {}).get("regime"),
            "session": built.get("session") or "",
            "session_incomplete": bool(built.get("session_incomplete")),
            "candidates": slim,
            "board": built.get("board"),
        },
    )
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
