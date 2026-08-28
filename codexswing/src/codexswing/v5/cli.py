"""Read-only CLI for describing and planning the inactive v0.5 lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from codexswing.schemas.source import canonical_json
from codexswing.v5.budget import CacheInventory, plan_cache_only
from codexswing.v5.ledger import ProspectiveLedger
from codexswing.v5.replay_plan import ReplayPathSample, cache_requirements_for_paths
from codexswing.v5.spec import V5ResearchSpec


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = PROJECT_ROOT / "research_specs" / "ORATS_SWING_RESEARCH_V5.json"


def _spec(path: str) -> V5ResearchSpec:
    return V5ResearchSpec.from_json_file(Path(path))


def _write_json(payload: Mapping[str, Any]) -> None:
    print(canonical_json(payload))


def _describe(args: argparse.Namespace) -> int:
    spec = _spec(args.spec)
    _write_json(spec.public_summary())
    return 0


def _plan_cache(args: argparse.Namespace) -> int:
    spec = _spec(args.spec)
    inventory = (
        CacheInventory.from_json_file(Path(args.inventory))
        if args.inventory
        else CacheInventory.from_store(Path(args.store_root))
    )
    payload = json.loads(Path(args.paths).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("paths"), list):
        raise ValueError("replay path input must contain a paths list")
    paths = tuple(ReplayPathSample.from_dict(item) for item in payload["paths"])
    plan = plan_cache_only(spec, inventory, cache_requirements_for_paths(paths))
    _write_json(plan.to_dict(include_missing=args.include_missing))
    return 0


def _verify_ledger(args: argparse.Namespace) -> int:
    spec = _spec(args.spec)
    ledger = ProspectiveLedger(Path(args.ledger), spec.model_version, spec.spec_sha256)
    result = ledger.verify()
    _write_json(result.to_dict())
    return 0 if result.valid else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codexswing-v5",
        description="Cache-only planning tools; this command has no API client.",
    )
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    commands = parser.add_subparsers(dest="command", required=True)

    describe = commands.add_parser("describe", help="show frozen v0.5 status and quota guard")
    describe.set_defaults(handler=_describe)

    plan = commands.add_parser("plan-cache", help="compare replay paths with a local cache inventory")
    inventory_source = plan.add_mutually_exclusive_group(required=True)
    inventory_source.add_argument("--inventory", help="previously generated local inventory JSON")
    inventory_source.add_argument("--store-root", help="immutable local CodexSwing store to inspect")
    plan.add_argument("--paths", required=True)
    plan.add_argument("--include-missing", action="store_true")
    plan.set_defaults(handler=_plan_cache)

    verify = commands.add_parser("ledger-verify", help="verify a local shadow-ledger hash chain")
    verify.add_argument("--ledger", required=True)
    verify.set_defaults(handler=_verify_ledger)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
