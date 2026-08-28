"""Run the Options Agent regular-session rerun from an audit plan."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from groko import core


@dataclass(frozen=True)
class MarketOpenRunnerResult:
    status: str
    errors: tuple[str, ...]
    rerun_command: tuple[str, ...]
    audit_command: tuple[str, ...]
    rerun_out_dir: str
    post_rerun_status: str = ""
    can_mark_goal_complete: bool = False
    update_goal_action: str = ""


def read_first_csv_row(path: Path) -> dict[str, str]:
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {str(key): str(value or "") for key, value in row.items()}
    return {}


def validate_market_open_plan(
    plan_row: Mapping[str, str],
    *,
    allow_closed_market: bool = False,
    allow_existing_out_dir: bool = False,
) -> list[str]:
    errors: list[str] = []
    command = str(plan_row.get("rerun_command") or "").strip()
    if not command:
        errors.append("rerun_command_missing")
        parts: list[str] = []
    else:
        parts = shlex.split(command)
    for flag in ("--live-schwab", "--live-portfolio", "--agent-reviews-json"):
        if flag not in parts:
            errors.append(f"rerun_command_missing_{flag.removeprefix('--').replace('-', '_')}")
    out_dir_text = str(plan_row.get("rerun_out_dir") or "").strip()
    if not out_dir_text:
        errors.append("rerun_out_dir_missing")
    else:
        out_dir = Path(out_dir_text).expanduser()
        if out_dir.exists() and not allow_existing_out_dir:
            errors.append("rerun_out_dir_exists")
    if not allow_closed_market and not core.is_regular_market_session_open():
        errors.append("regular_market_session_open=false")
    return errors


def evaluate_post_rerun_packet(path: Path) -> tuple[str, list[str], bool, str]:
    row = read_first_csv_row(path)
    if not row:
        return "MISSING_POST_RERUN_PACKET", ["post_rerun_packet_empty"], False, ""
    status = str(row.get("status") or "").strip()
    can_complete = _truthy_text(row.get("can_mark_goal_complete"))
    update_action = str(row.get("update_goal_action") or "").strip()
    errors: list[str] = []
    if status != "PASS_READY_TO_COMPLETE_GOAL":
        errors.append(f"post_rerun_status={status or 'missing'}")
    if not can_complete:
        errors.append("can_mark_goal_complete=false")
    if update_action != "call_update_goal_complete":
        errors.append(f"update_goal_action={update_action or 'missing'}")
    return status, errors, can_complete, update_action


def _truthy_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def run_from_plan(
    *,
    plan_csv: Path,
    post_rerun_csv: Path | None = None,
    cwd: Path = Path("/Users/anuppamvi/uw_root/tradedesk"),
    dry_run: bool = False,
    allow_closed_market: bool = False,
    allow_existing_out_dir: bool = False,
) -> MarketOpenRunnerResult:
    plan_row = read_first_csv_row(plan_csv)
    if not plan_row:
        return MarketOpenRunnerResult(
            status="INVALID_PLAN",
            errors=("plan_csv_empty",),
            rerun_command=(),
            audit_command=(),
            rerun_out_dir="",
        )
    errors = validate_market_open_plan(
        plan_row,
        allow_closed_market=allow_closed_market,
        allow_existing_out_dir=allow_existing_out_dir,
    )
    rerun_command = tuple(shlex.split(str(plan_row.get("rerun_command") or "")))
    rerun_out_dir = str(plan_row.get("rerun_out_dir") or "")
    audit_command: tuple[str, ...] = ()
    post_rerun_status = ""
    can_mark_goal_complete = False
    update_goal_action = ""
    if post_rerun_csv:
        post_row = read_first_csv_row(post_rerun_csv)
        audit_text = str(post_row.get("audit_regeneration_command") or "").strip()
        if audit_text:
            audit_command = tuple(shlex.split(audit_text))
        else:
            errors.append("audit_regeneration_command_missing")
    if errors:
        return MarketOpenRunnerResult(
            status="BLOCKED",
            errors=tuple(errors),
            rerun_command=rerun_command,
            audit_command=audit_command,
            rerun_out_dir=rerun_out_dir,
            post_rerun_status=post_rerun_status,
            can_mark_goal_complete=can_mark_goal_complete,
            update_goal_action=update_goal_action,
        )
    if dry_run:
        return MarketOpenRunnerResult(
            status="DRY_RUN_READY",
            errors=(),
            rerun_command=rerun_command,
            audit_command=audit_command,
            rerun_out_dir=rerun_out_dir,
            post_rerun_status=post_rerun_status,
            can_mark_goal_complete=can_mark_goal_complete,
            update_goal_action=update_goal_action,
        )
    rerun_completed = subprocess.run(rerun_command, cwd=str(cwd.expanduser()))
    if rerun_completed.returncode != 0:
        return MarketOpenRunnerResult(
            status=f"RERUN_FAILED_{rerun_completed.returncode}",
            errors=(f"rerun_exit_code={rerun_completed.returncode}",),
            rerun_command=rerun_command,
            audit_command=audit_command,
            rerun_out_dir=rerun_out_dir,
            post_rerun_status=post_rerun_status,
            can_mark_goal_complete=can_mark_goal_complete,
            update_goal_action=update_goal_action,
        )
    if audit_command:
        audit_completed = subprocess.run(audit_command, cwd=str(cwd.expanduser()))
        if audit_completed.returncode != 0:
            return MarketOpenRunnerResult(
                status=f"AUDIT_FAILED_{audit_completed.returncode}",
                errors=(f"audit_exit_code={audit_completed.returncode}",),
                rerun_command=rerun_command,
                audit_command=audit_command,
                rerun_out_dir=rerun_out_dir,
                post_rerun_status=post_rerun_status,
                can_mark_goal_complete=can_mark_goal_complete,
                update_goal_action=update_goal_action,
            )
    if post_rerun_csv:
        post_rerun_status, post_errors, can_mark_goal_complete, update_goal_action = evaluate_post_rerun_packet(post_rerun_csv)
        if post_errors:
            return MarketOpenRunnerResult(
                status="COMPLETED_NOT_READY",
                errors=tuple(post_errors),
                rerun_command=rerun_command,
                audit_command=audit_command,
                rerun_out_dir=rerun_out_dir,
                post_rerun_status=post_rerun_status,
                can_mark_goal_complete=can_mark_goal_complete,
                update_goal_action=update_goal_action,
            )
        return MarketOpenRunnerResult(
            status="COMPLETED_READY_TO_COMPLETE_GOAL",
            errors=(),
            rerun_command=rerun_command,
            audit_command=audit_command,
            rerun_out_dir=rerun_out_dir,
            post_rerun_status=post_rerun_status,
            can_mark_goal_complete=can_mark_goal_complete,
            update_goal_action=update_goal_action,
        )
    return MarketOpenRunnerResult(
        status="COMPLETED",
        errors=(),
        rerun_command=rerun_command,
        audit_command=audit_command,
        rerun_out_dir=rerun_out_dir,
    )


def _format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Groko market-open rerun from the audit plan.")
    parser.add_argument("--plan-csv", required=True, type=Path)
    parser.add_argument("--post-rerun-csv", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path("/Users/anuppamvi/uw_root/tradedesk"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-closed-market", action="store_true")
    parser.add_argument("--allow-existing-out-dir", action="store_true")
    args = parser.parse_args(argv)

    result = run_from_plan(
        plan_csv=args.plan_csv,
        post_rerun_csv=args.post_rerun_csv,
        cwd=args.cwd,
        dry_run=args.dry_run,
        allow_closed_market=args.allow_closed_market,
        allow_existing_out_dir=args.allow_existing_out_dir,
    )
    print(f"status={result.status}")
    if result.rerun_out_dir:
        print(f"rerun_out_dir={result.rerun_out_dir}")
    if result.rerun_command:
        print("rerun_command=" + _format_command(result.rerun_command))
    if result.audit_command:
        print("audit_regeneration_command=" + _format_command(result.audit_command))
    if result.post_rerun_status:
        print(f"post_rerun_status={result.post_rerun_status}")
        print(f"can_mark_goal_complete={str(result.can_mark_goal_complete).lower()}")
    if result.update_goal_action:
        print(f"update_goal_action={result.update_goal_action}")
    for error in result.errors:
        print(f"error={error}")
    return 0 if result.status in {"COMPLETED", "COMPLETED_READY_TO_COMPLETE_GOAL", "DRY_RUN_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
