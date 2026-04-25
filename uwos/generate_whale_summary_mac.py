from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uwos.generate_whale_summary import main


DOWNLOAD_PATTERNS = [
    "bot-eod-report-*.csv",
    "bot-eod-report-*.zip",
]


def find_download_report(date_str: str = "") -> Path:
    downloads = Path.home() / "Downloads"
    candidates = []
    patterns = (
        [
            f"bot-eod-report-{date_str}.csv",
            f"bot-eod-report-{date_str}.zip",
        ]
        if date_str
        else DOWNLOAD_PATTERNS
    )
    for pattern in patterns:
        candidates.extend(p for p in downloads.glob(pattern) if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            "No UW report found in ~/Downloads. Expected one of: "
            "bot-eod-report-YYYY-MM-DD.csv/.zip. "
            "This whale summary generator does not accept chain-oi-changes or dp-eod-report files. "
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


def pop_positional_date() -> str:
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        if len(arg) == 10 and arg[4] == "-" and arg[7] == "-":
            sys.argv.remove(arg)
            return arg
    return ""


if __name__ == "__main__":
    date_arg = pop_positional_date()
    if not has_arg("--input"):
        sys.argv[1:1] = ["--input", str(find_download_report(date_arg))]

    if not has_arg("--config"):
        sys.argv[1:1] = [
            "--config",
            str(REPO_ROOT / "uwos" / "rulebook_config_goal_holistic_claude.yaml"),
        ]

    if date_arg and not has_arg("--output"):
        sys.argv[1:1] = [
            "--output",
            str(REPO_ROOT / date_arg / f"whale-{date_arg}.md"),
        ]

    output = arg_value("--output")
    if output:
        Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)

    main()
