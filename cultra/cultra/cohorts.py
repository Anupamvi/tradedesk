"""Point-in-time universe and deterministic historical-cohort contracts.

This module is deliberately data-source agnostic and network free.  It rejects
current-constituent projection into the past and freezes a rotating research
sample without becoming the daily production universe or an output top-N cap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


class CohortError(ValueError):
    """A point-in-time universe or cohort freeze is not auditable."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _iso(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise CohortError("%s must use YYYY-MM-DD" % label) from exc


@dataclass(frozen=True)
class PointInTimeMember:
    ticker: str
    asset_type: str
    eligible_from: date
    eligible_through: date
    observed_at: date
    optionable: bool
    sampling_stratum: str
    liquidity_rank: int

    def __post_init__(self) -> None:
        ticker = str(self.ticker).strip().upper()
        if not ticker or len(ticker) > 12:
            raise CohortError("point-in-time ticker is invalid")
        object.__setattr__(self, "ticker", ticker)
        asset_type = str(self.asset_type).strip().upper()
        if asset_type not in {
            "STOCK",
            "ETF",
            "INELIGIBLE_OTHER_SECURITY",
            "UNRESOLVED_STOCK_OR_ETP",
        }:
            raise CohortError(
                "asset_type must be STOCK, ETF, INELIGIBLE_OTHER_SECURITY, "
                "or UNRESOLVED_STOCK_OR_ETP"
            )
        object.__setattr__(self, "asset_type", asset_type)
        if self.eligible_from > self.eligible_through:
            raise CohortError("member eligibility dates are reversed")
        if not self.eligible_from <= self.observed_at <= self.eligible_through:
            raise CohortError("member observation must fall inside its eligibility interval")
        if not isinstance(self.optionable, bool):
            raise CohortError("optionable must be boolean")
        if not str(self.sampling_stratum).strip():
            raise CohortError("sampling_stratum is required")
        if (
            isinstance(self.liquidity_rank, bool)
            or not isinstance(self.liquidity_rank, int)
            or self.liquidity_rank <= 0
        ):
            raise CohortError("liquidity_rank must be a positive integer")


@dataclass(frozen=True)
class PointInTimeUniverse:
    universe_id: str
    provider: str
    source_uri: str
    source_sha256: str
    coverage: str
    members: Tuple[PointInTimeMember, ...]

    def __post_init__(self) -> None:
        for name in ("universe_id", "provider", "source_uri", "coverage"):
            if not str(getattr(self, name)).strip():
                raise CohortError("%s is required" % name)
        digest = self.source_sha256.lower().removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise CohortError("source_sha256 is invalid")
        if self.coverage != (
            "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_"
            "VOLUME_ACROSS_2_CBOE_VENUES"
        ):
            raise CohortError("universe coverage is not the approved broad scope")
        identities = [(item.ticker, item.observed_at) for item in self.members]
        if not self.members or len(identities) != len(set(identities)):
            raise CohortError("point-in-time members are empty or duplicated")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "universe_id": self.universe_id,
                    "provider": self.provider,
                    "source_uri": self.source_uri,
                    "source_sha256": self.source_sha256,
                    "coverage": self.coverage,
                    "members": [
                        {
                            "ticker": item.ticker,
                            "asset_type": item.asset_type,
                            "eligible_from": item.eligible_from.isoformat(),
                            "eligible_through": item.eligible_through.isoformat(),
                            "observed_at": item.observed_at.isoformat(),
                            "optionable": item.optionable,
                            "sampling_stratum": item.sampling_stratum,
                            "liquidity_rank": item.liquidity_rank,
                        }
                        for item in self.members
                    ],
                }
            )
        ).hexdigest()


def load_point_in_time_universe(path: Path) -> PointInTimeUniverse:
    """Load an explicit broad-universe manifest; never infer one from SPY."""

    supplied = Path(path).expanduser().resolve()
    try:
        supplied.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise CohortError("point-in-time universe manifest must be Cultra-owned") from exc
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortError("point-in-time universe manifest is unreadable") from exc
    if value.get("schema") != "cultra.point-in-time-universe.v1":
        raise CohortError("point-in-time universe schema is unsupported")
    allowed_root_fields = {
        "schema",
        "universe_id",
        "provider",
        "source_uri",
        "source_sha256",
        "coverage",
        "members",
    }
    if set(value) != allowed_root_fields:
        raise CohortError("point-in-time universe contains unfrozen fields")
    raw_members = value.get("members")
    if not isinstance(raw_members, list):
        raise CohortError("point-in-time universe members are missing")
    members = []
    allowed_member_fields = {
        "ticker",
        "asset_type",
        "eligible_from",
        "eligible_through",
        "observed_at",
        "optionable",
        "sampling_stratum",
        "liquidity_rank",
    }
    for raw in raw_members:
        if not isinstance(raw, Mapping):
            raise CohortError("point-in-time universe member is malformed")
        if set(raw) != allowed_member_fields:
            raise CohortError("point-in-time universe member contains unfrozen fields")
        members.append(
            PointInTimeMember(
                ticker=str(raw.get("ticker", "")),
                asset_type=str(raw.get("asset_type", "")),
                eligible_from=_iso(raw.get("eligible_from"), "eligible_from"),
                eligible_through=_iso(raw.get("eligible_through"), "eligible_through"),
                observed_at=_iso(raw.get("observed_at"), "observed_at"),
                optionable=raw.get("optionable"),
                sampling_stratum=str(raw.get("sampling_stratum", "")),
                liquidity_rank=raw.get("liquidity_rank"),
            )
        )
    return PointInTimeUniverse(
        universe_id=str(value.get("universe_id", "")),
        provider=str(value.get("provider", "")),
        source_uri=str(value.get("source_uri", "")),
        source_sha256=str(value.get("source_sha256", "")),
        coverage=str(value.get("coverage", "")),
        members=tuple(members),
    )


def save_rotating_cohorts(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Durably save one frozen cohort manifest inside Cultra/out."""

    supplied = Path(path).expanduser().resolve()
    allowed = (PROJECT_ROOT / "out").resolve()
    try:
        supplied.relative_to(allowed)
    except ValueError as exc:
        raise CohortError("cohort manifest output must remain inside Cultra/out") from exc
    if supplied.exists():
        raise CohortError("frozen cohort manifest already exists")
    supplied.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(supplied.parent, 0o700)
    temporary = supplied.with_name(".%s.tmp-%d" % (supplied.name, os.getpid()))
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, supplied)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return supplied


def eligible_members(
    universe: PointInTimeUniverse,
    *,
    selection_date: date,
) -> Tuple[PointInTimeMember, ...]:
    """Return members knowable and eligible on the selection date.

    Eligibility after ``selection_date`` is deliberately ignored.  Requiring a
    name to survive a later holding window would use future membership and
    create survivorship bias.  A later delisting or corporate action therefore
    remains part of the outcome-resolution ledger.
    """

    return tuple(
        sorted(
            (
                item
                for item in universe.members
                if item.optionable
                and item.asset_type in {"STOCK", "ETF"}
                # A liquidity rank carried forward from an older snapshot is
                # not equivalent to a point-in-time rank. Every block requires
                # a universe snapshot observed on that exact selection date.
                and item.observed_at == selection_date
                and item.eligible_from <= selection_date
                and item.eligible_through >= selection_date
            ),
            key=lambda item: (
                item.sampling_stratum,
                item.liquidity_rank,
                item.ticker,
            ),
        )
    )


def _sample_block(
    members: Sequence[PointInTimeMember],
    *,
    count: int,
    minimum_stock_count: int,
    seed: str,
    previously_used: Iterable[str],
) -> Tuple[PointInTimeMember, ...]:
    used = set(previously_used)
    available = [item for item in members if item.ticker not in used]
    ranked = sorted(
        available,
        key=lambda item: (
            item.liquidity_rank,
            hashlib.sha256(
                (seed + "|" + item.sampling_stratum + "|" + item.ticker).encode(
                    "utf-8"
                )
            ).hexdigest(),
            item.liquidity_rank,
            item.ticker,
        ),
    )
    # Round-robin strata prevents the sample from collapsing into one market
    # segment without imposing a portfolio or trade-level sector gate.
    by_stratum: Dict[str, list] = {}
    for item in ranked:
        by_stratum.setdefault(item.sampling_stratum, []).append(item)
    ordered_candidates = []
    strata = sorted(by_stratum)
    while any(by_stratum.values()):
        for stratum in strata:
            values = by_stratum[stratum]
            if values:
                ordered_candidates.append(values.pop(0))
    stock_candidates = [
        item for item in ordered_candidates if item.asset_type == "STOCK"
    ]
    if len(stock_candidates) < minimum_stock_count:
        raise CohortError(
            "insufficient verified stock members for the frozen cohort floor"
        )
    chosen = {
        (item.ticker, item.observed_at)
        for item in stock_candidates[:minimum_stock_count]
    }
    for item in ordered_candidates:
        if len(chosen) >= count:
            break
        chosen.add((item.ticker, item.observed_at))
    selected = [
        item
        for item in ordered_candidates
        if (item.ticker, item.observed_at) in chosen
    ]
    if len(selected) != count:
        raise CohortError("insufficient eligible point-in-time members for cohort")
    return tuple(selected)


def freeze_rotating_cohorts(
    universe: PointInTimeUniverse,
    sessions: Sequence[date],
    *,
    cohort_size: int = 10,
    block_sessions: int = 120,
    maximum_holding_sessions: int = 60,
    minimum_point_in_time_universe: int = 100,
    minimum_stock_fraction: float = 0.80,
    seed: str = "CULTRA_ROTATING_COHORT_V1",
) -> Mapping[str, Any]:
    """Freeze dynamic research cohorts without looking at later outcomes."""

    ordered = tuple(sessions)
    if not ordered or ordered != tuple(sorted(set(ordered))):
        raise CohortError("sessions must be non-empty, sorted, and unique")
    for name, value in (
        ("cohort_size", cohort_size),
        ("block_sessions", block_sessions),
        ("maximum_holding_sessions", maximum_holding_sessions),
        ("minimum_point_in_time_universe", minimum_point_in_time_universe),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CohortError("%s must be a positive integer" % name)
    if maximum_holding_sessions > block_sessions:
        raise CohortError("maximum holding window exceeds the block length")
    if not 0.0 <= float(minimum_stock_fraction) <= 1.0:
        raise CohortError("minimum_stock_fraction must be between zero and one")
    blocks = []
    previously_used = set()
    for block_index, offset in enumerate(range(0, len(ordered), block_sessions)):
        block = ordered[offset : offset + block_sessions]
        eligible_signal_count = max(0, len(block) - maximum_holding_sessions - 1)
        last_eligible_signal = (
            block[eligible_signal_count - 1] if eligible_signal_count else None
        )
        observed_population = tuple(
            item
            for item in universe.members
            if item.optionable
            and item.observed_at == block[0]
            and item.eligible_from <= block[0] <= item.eligible_through
        )
        if len(observed_population) < minimum_point_in_time_universe:
            raise CohortError(
                "point-in-time selection universe is below the frozen broad-coverage floor"
            )
        candidates = eligible_members(
            universe,
            selection_date=block[0],
        )
        if len(candidates) < cohort_size:
            raise CohortError(
                "insufficient point-in-time asset classifications for cohort selection"
            )
        required_stocks = int(math.ceil(cohort_size * minimum_stock_fraction))
        selected = _sample_block(
            candidates,
            count=cohort_size,
            minimum_stock_count=required_stocks,
            seed="%s|%s|%d" % (seed, universe.fingerprint, block_index),
            previously_used=previously_used,
        )
        previously_used.update(item.ticker for item in selected)
        if sum(item.asset_type == "STOCK" for item in selected) < required_stocks:
            raise CohortError("frozen cohort does not contain enough stock evidence")
        blocks.append(
            {
                "block_index": block_index,
                "selection_date": block[0].isoformat(),
                "block_start": block[0].isoformat(),
                "block_end": block[-1].isoformat(),
                "required_coverage_through": block[-1].isoformat(),
                "eligible_signal_session_count": eligible_signal_count,
                "last_eligible_signal_date": (
                    None
                    if last_eligible_signal is None
                    else last_eligible_signal.isoformat()
                ),
                "future_membership_used_for_selection": False,
                "point_in_time_population_count": len(observed_population),
                "resolved_classification_count": len(candidates),
                "tickers": [item.ticker for item in selected],
                "strata": [item.sampling_stratum for item in selected],
            }
        )
    payload = {
        "schema": "cultra.rotating-historical-cohorts.v1",
        "selection_policy": "POINT_IN_TIME_STRATIFIED_DETERMINISTIC_SAMPLE",
        "daily_production_universe_cap": None,
        "research_sample_is_not_a_ticket_output_cap": True,
        "universe_id": universe.universe_id,
        "universe_fingerprint": universe.fingerprint,
        "session_start": ordered[0].isoformat(),
        "session_end": ordered[-1].isoformat(),
        "session_count": len(ordered),
        "cohort_size": cohort_size,
        "block_sessions": block_sessions,
        "maximum_holding_sessions": maximum_holding_sessions,
        "minimum_point_in_time_universe": minimum_point_in_time_universe,
        "minimum_stock_fraction": minimum_stock_fraction,
        "stock_floor_enforced_during_selection": True,
        "transition_policy": "CENSOR_ENTRIES_BEFORE_COHORT_ROTATION",
        "blocks": blocks,
    }
    return dict(payload, freeze_hash=hashlib.sha256(_canonical(payload)).hexdigest())


__all__ = [
    "CohortError",
    "PointInTimeMember",
    "PointInTimeUniverse",
    "eligible_members",
    "freeze_rotating_cohorts",
    "load_point_in_time_universe",
    "save_rotating_cohorts",
]
