"""CLI: allocate dollars, project paths, write the HTML calculator."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from compoundcore.allocate import distribution
from compoundcore.projections import path_table
from compoundcore.report import calc_markdown
from compoundcore.sleeve import SLEEVE_NAMES, public_snapshot
from compoundcore.webcalc import DEFAULT_HTML, write_calculator


PLAYBOOK = Path(__file__).resolve().parent.parent / "docs" / "PLAYBOOK.md"


def _normalize(argv: List[str]) -> List[str]:
    if not argv:
        return ["calc"]
    if argv[0] in ("calc", "allocate", "project", "playbook", "calculator", "dashboard", "-h", "--help"):
        return argv
    if re.match(r"^\$?\d", argv[0]):
        rest = argv[1:]
        out = ["calc", "--amount", argv[0].lstrip("$")]
        # allow bare --monthly 1000 after the number
        i = 0
        while i < len(rest):
            out.append(rest[i])
            i += 1
        return out
    return argv


def _add_money_args(p: argparse.ArgumentParser, amount_required: bool) -> None:
    p.add_argument("--amount", "-a", required=amount_required, help="Dollars to allocate")
    p.add_argument("--weekly", "-w", default="0", help="Weekly contribution dollars")
    p.add_argument("--monthly", "-m", default="0", help="Monthly contribution for projections")
    p.add_argument(
        "--sleeve",
        choices=["default", "aggressive", "both"],
        default="both",
        help="Which sleeve(s) to print",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable output")


def _money(raw: Optional[str]) -> float:
    if raw is None or raw == "":
        return 0.0
    text = str(raw).strip().replace(",", "").replace("$", "")
    if text == "":
        return 0.0
    return float(text)


def _payload(amount: float, weekly: float, monthly: float, sleeve: str):
    names = list(SLEEVE_NAMES) if sleeve == "both" else [sleeve]
    return {
        "amount": amount,
        "weekly": weekly,
        "monthly": monthly,
        "sleeves": {
            name: {
                "allocation": distribution(amount, name, weekly),
                "projections": path_table(amount, monthly, name),
            }
            for name in names
        },
        "snapshot": public_snapshot(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    raw = _normalize(raw)
    parser = argparse.ArgumentParser(
        prog="compoundcore",
        description="Compound Core long-term index sleeve. No stock-picking, no options, no orders.",
    )
    sub = parser.add_subparsers(dest="cmd")

    calc = sub.add_parser("calc", help="Allocate dollars and project both sleeves")
    _add_money_args(calc, amount_required=False)

    alloc = sub.add_parser("allocate", help="Dollar split only")
    _add_money_args(alloc, amount_required=True)

    proj = sub.add_parser("project", help="5y/10y paths only")
    _add_money_args(proj, amount_required=True)

    book = sub.add_parser("playbook", help="Print the playbook path")
    book.add_argument("--print", dest="dump", action="store_true", help="Print the playbook markdown")

    html = sub.add_parser("calculator", help="Write the raw HTML calculator")
    html.add_argument("--out", default=str(DEFAULT_HTML), help="Output path")

    dash = sub.add_parser("dashboard", help="Persistent local dashboard (both sleeves + my book)")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument("--state", default="", help="JSON state path (default: var/dashboard.json)")

    args = parser.parse_args(raw)
    cmd = args.cmd or "calc"

    if cmd == "playbook":
        if not PLAYBOOK.exists():
            print("playbook missing: %s" % PLAYBOOK, file=sys.stderr)
            return 1
        if getattr(args, "dump", False):
            sys.stdout.write(PLAYBOOK.read_text(encoding="utf-8"))
        else:
            print(PLAYBOOK)
        return 0

    if cmd == "calculator":
        path = write_calculator(Path(args.out))
        print(path)
        return 0

    if cmd == "dashboard":
        from compoundcore.dashboard import serve

        return serve(args.host, args.port, args.state or None)

    amount = _money(getattr(args, "amount", None))
    if cmd in ("calc", "allocate", "project") and amount <= 0:
        if cmd == "calc" and getattr(args, "amount", None) in (None, ""):
            sys.stdout.write(
                "Compound Core — long-term index sleeve. No stock-picking, no options, no orders.\n"
                "\n"
                "  python3 -m compoundcore 100000\n"
                "  python3 -m compoundcore 250000 --weekly 500 --monthly 1000\n"
                "  python3 -m compoundcore calc --amount 100000 --sleeve both\n"
                "  python3 -m compoundcore playbook\n"
                "  python3 -m compoundcore dashboard\n"
                "  python3 -m compoundcore calculator\n"
                "\n"
                "Playbook: %s\n" % PLAYBOOK
            )
            return 0
        parser.error("--amount must be a positive dollar figure (or pass it as python3 -m compoundcore 100000)")

    weekly = _money(getattr(args, "weekly", "0"))
    monthly = _money(getattr(args, "monthly", "0"))
    sleeve = getattr(args, "sleeve", "both")

    if getattr(args, "json", False):
        json.dump(_payload(amount, weekly, monthly, sleeve), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(calc_markdown(amount, weekly, monthly, sleeve))
    return 0
