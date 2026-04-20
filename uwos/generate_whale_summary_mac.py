from __future__ import annotations

import sys
from pathlib import Path

from uwos.generate_whale_summary import main


DOWNLOAD_PATTERNS = [
    "bot-eod-report-*.csv",
    "bot-eod-report-*.zip",
    "dp-eod-report-*.csv",
    "dp-eod-report-*.zip",
]


def find_download_report() -> Path:
    downloads = Path.home() / "Downloads"
    candidates = []
    for pattern in DOWNLOAD_PATTERNS:
        candidates.extend(p for p in downloads.glob(pattern) if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            "No UW report found in ~/Downloads. Expected one of: "
            "bot-eod-report-YYYY-MM-DD.csv/.zip or dp-eod-report-YYYY-MM-DD.csv/.zip. "
            "Pass --input /path/to/report.csv if it lives somewhere else."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def has_arg(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def arg_value(flag: str) -> str:
    args = sys.argv[1:]
    for idx, arg in enumerate(args):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
    return ""


if __name__ == "__main__":
    if not has_arg("--input"):
        sys.argv[1:1] = ["--input", str(find_download_report())]

    output = arg_value("--output")
    if output:
        Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)

    main()
