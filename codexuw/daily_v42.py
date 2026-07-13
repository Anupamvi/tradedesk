from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from .integrity_v42 import build_full_overlay_workspace, write_integrity_decision_book


def _date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _value(args: list[str], name: str, default: str = "") -> str:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _run_v4(arguments: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, "-m", "codexuw.daily_v4", *arguments], cwd=str(cwd), check=True)


def run_command(arguments: list[str]) -> int:
    root = Path(_value(arguments, "--root", "/Users/anuppamvi/uw_root/tradedesk")).expanduser().resolve()
    date_text = _value(arguments, "--date") or Path(_value(arguments, "--base-dir")).name
    if not date_text:
        raise SystemExit("V4.2 run requires --date or a YYYY-MM-DD --base-dir")
    asof = _date(date_text)
    out_text = _value(arguments, "--out-dir")
    out_dir = Path(out_text).expanduser().resolve() if out_text else root / "out" / f"codexdaily_v42_{asof}"
    forwarded = list(arguments)
    if "--out-dir" not in forwarded:
        forwarded.extend(["--out-dir", str(out_dir)])
    _run_v4(["run", *forwarded], root)
    manifest = write_integrity_decision_book(
        out_dir=out_dir, asof=asof, offline="--offline" in forwarded,
        source_dates={"eod_discovery": str(asof), "oi_confirmation": str(asof), "live_quotes": str(asof)},
    )
    print(f"V4.2 report: {manifest['report']}")
    print(f"V4.2 decision book: {manifest['decision_book']}")
    return 0


def overlay_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Full-discovery V4.2 EOD plus next-session OI overlay")
    parser.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--date", required=True, help="Base EOD discovery date")
    parser.add_argument("--overlay-file", required=True)
    parser.add_argument("--overlay-date", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(arguments)
    root, base_date, overlay_date = Path(args.root).expanduser().resolve(), _date(args.date), _date(args.overlay_date)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "out" / f"codexdaily_v42_full_overlay_{base_date}_oi_{overlay_date}"
    stage, source_dates = build_full_overlay_workspace(
        root=root, base_date=base_date, overlay_file=Path(args.overlay_file), overlay_date=overlay_date, out_dir=out_dir
    )
    forwarded = ["run", "--root", str(root), "--base-dir", str(stage), "--out-dir", str(out_dir), "--max-tickers", "0", "--max-candidates", "0", "--risk-budget", "0"]
    if args.offline:
        forwarded.append("--offline")
    _run_v4(forwarded, root)
    manifest = write_integrity_decision_book(out_dir=out_dir, asof=overlay_date, offline=args.offline, source_dates=source_dates)
    print(f"V4.2 full-overlay report: {manifest['report']}")
    print(f"V4.2 decision book: {manifest['decision_book']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"run", "overlay", "validate"}:
        print("usage: python3 -m codexuw.daily_v42 {run|overlay|validate} ...", file=sys.stderr)
        return 2
    command = args.pop(0)
    if command == "run":
        return run_command(args)
    if command == "overlay":
        return overlay_command(args)
    _run_v4(["validate", *args], Path("/Users/anuppamvi/uw_root/tradedesk"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
