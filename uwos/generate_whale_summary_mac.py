from __future__ import annotations

import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uwos.generate_whale_summary import main
from uwos.whale_source import BOT_EOD_PREFIX, is_split_bot_eod_part

DATE_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _report_patterns(date_str: str = "") -> list[str]:
    return (
        [
            f"{BOT_EOD_PREFIX}{date_str}.csv",
            f"{BOT_EOD_PREFIX}{date_str}.zip",
        ]
        if date_str
        else [f"{BOT_EOD_PREFIX}*.csv", f"{BOT_EOD_PREFIX}*.zip"]
    )


def _report_candidates(directory: Path, date_str: str = "") -> list[Path]:
    if not directory.exists():
        return []
    candidates: list[Path] = []
    for pattern in _report_patterns(date_str):
        candidates.extend(
            path
            for path in directory.glob(pattern)
            if path.is_file() and not is_split_bot_eod_part(path)
        )
    return candidates


def _dated_directories(date_str: str = "") -> list[Path]:
    cwd = Path.cwd().resolve()
    directories: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved.exists() and resolved not in directories:
            directories.append(resolved)

    if date_str:
        add(REPO_ROOT / date_str)
        if cwd.name == date_str:
            add(cwd)
        add(cwd / date_str)
        if cwd.parent.name == date_str:
            add(cwd.parent)
    else:
        for dated_dir in sorted(REPO_ROOT.glob("20??-??-??")):
            if dated_dir.is_dir():
                add(dated_dir)

    return directories


def _dated_report_candidates(date_str: str = "") -> list[Path]:
    candidates: list[Path] = []
    for dated_dir in _dated_directories(date_str):
        candidates.extend(_report_candidates(dated_dir, date_str))
        for pattern in _report_patterns(date_str):
            candidates.extend(
                path
                for path in dated_dir.rglob(pattern)
                if path.is_file() and not is_split_bot_eod_part(path)
            )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def find_download_report(date_str: str = "") -> Path:
    downloads = Path.home() / "Downloads"
    downloads_candidates = _report_candidates(downloads, date_str)
    if downloads_candidates:
        return max(downloads_candidates, key=lambda p: p.stat().st_mtime)

    dated_candidates = _dated_report_candidates(date_str)
    if dated_candidates:
        return max(dated_candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "No UW report found in ~/Downloads or the repo dated workspace. Expected one of: "
        "bot-eod-report-YYYY-MM-DD.csv/.zip. Split part files are not accepted. "
        "This whale summary generator does not accept chain-oi-changes or dp-eod-report files. "
        "Pass --input /path/to/report.csv if it lives somewhere else."
    )


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


def infer_date_from_text(text: str) -> str:
    match = DATE_TOKEN_RE.search(text)
    return match.group(0) if match else ""


def default_output_path(input_path: Path, date_str: str = "") -> Path:
    resolved_date = date_str or infer_date_from_text(str(input_path))
    if not resolved_date:
        return Path("whale-Unknown Date.md")
    return REPO_ROOT / resolved_date / f"whale-{resolved_date}.md"


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

    if not has_arg("--output"):
        input_path = Path(arg_value("--input"))
        sys.argv[1:1] = [
            "--output",
            str(default_output_path(input_path, date_arg)),
        ]

    output = arg_value("--output")
    if output:
        Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)

    main()
