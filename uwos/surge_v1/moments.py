"""Moment-matched payoff gate, and the structure choice that follows from it.

The failure that produced years of false leads in this repo: a signal was
validated on one moment of the return distribution and then traded with a
structure exposed to a different one. E|move| was used to justify short
verticals; a cross-sectional |move| median was used to justify long premium.

Here every structure declares the moment it is exposed to, and a signal may only
be routed to a structure whose moment that signal was actually validated on.

Crossings matter as much as the moment. Measured round-trip friction on this
data is ~8.5% of premium per crossing pair, so the structure with the fewest
crossings per unit of the wanted exposure wins on identical signal quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Moment(str, Enum):
    SIGNED_RIGHT_TAIL = "signed_right_tail"    # needs P(return >= +x)
    SIGNED_LEFT_TAIL = "signed_left_tail"      # needs P(return <= -x)
    ABS_MEAN = "abs_mean"                      # needs E|move| (linear payoff)
    ABS_TAIL = "abs_tail"                      # needs P(|move| >= x)
    ABS_BODY = "abs_body"                      # needs P(|move| < x), short premium


@dataclass(frozen=True)
class Structure:
    key: str
    moment: Moment
    legs: int
    entry_crossings: int
    exit_crossings: int          # 0 when the position is carried to expiry
    defined_risk: bool
    direction: str               # bullish / bearish / neutral
    notes: str = ""

    @property
    def total_crossings(self) -> int:
        return self.entry_crossings + self.exit_crossings

    def friction_estimate(self, half_spread_pct_of_premium: float) -> float:
        """Round-trip cost as a fraction of premium at a given half-spread."""
        return self.total_crossings * half_spread_pct_of_premium


STRUCTURES: dict[str, Structure] = {
    s.key: s for s in [
        Structure("long_call", Moment.SIGNED_RIGHT_TAIL, 1, 1, 1, True, "bullish",
                  "cheapest expression of a signed right tail; 2 crossings"),
        Structure("long_put", Moment.SIGNED_LEFT_TAIL, 1, 1, 1, True, "bearish",
                  "cheapest expression of a signed left tail; 2 crossings"),
        Structure("call_debit_spread", Moment.SIGNED_RIGHT_TAIL, 2, 2, 2, True, "bullish",
                  "caps the tail the signal is being traded FOR; 4 crossings"),
        Structure("put_debit_spread", Moment.SIGNED_LEFT_TAIL, 2, 2, 2, True, "bearish",
                  "caps the tail the signal is being traded FOR; 4 crossings"),
        Structure("long_strangle", Moment.ABS_MEAN, 2, 2, 2, True, "neutral",
                  "payoff linear in |move|; the ONLY structure E|move| validates"),
        Structure("long_straddle", Moment.ABS_MEAN, 2, 2, 2, True, "neutral", ""),
        Structure("call_credit_spread", Moment.ABS_BODY, 2, 2, 0, True, "bearish",
                  "concave and capped; held to expiry so only 2 crossings"),
        Structure("put_credit_spread", Moment.ABS_BODY, 2, 2, 0, True, "bullish", ""),
        Structure("iron_condor", Moment.ABS_BODY, 4, 4, 0, True, "neutral",
                  "4 crossings even held to expiry; needs a large body edge"),
    ]
}


@dataclass
class SignalEvidence:
    """What a lane has actually PROVEN, not what it hopes."""
    lane: str
    validated_moments: set = field(default_factory=set)
    directional_edge: float = 0.0        # P(own direction) - P(opposite)
    directional_folds: int = 0           # folds where that edge was positive
    magnitude_ratio: float = 1.0         # |move| of picks / |move| of universe
    permutation_p: float = 1.0           # equity-level, vs random selection
    executable_permutation_p: float = 1.0  # OPTION book vs random-name option book
    folds_beating_null: int = 0
    n_folds: int = 0


MIN_DIRECTIONAL_EDGE = 0.05
MIN_DIRECTIONAL_FOLD_FRAC = 1.0      # a sign that flips in any fold is not a sign
MIN_MAGNITUDE_RATIO = 1.25
MAX_PERMUTATION_P = 0.05


def validated_moments(ev: SignalEvidence) -> set:
    """Derive, from measurements only, which moments this lane is entitled to trade.

    An equity-level effect is necessary but not sufficient: the option book must
    also have beaten a random-name book using the same structure and exit rules.
    A purged run of this engine produced an equity effect at p=0.000 whose option
    expression was indistinguishable from random, which is what this gate exists
    to catch.
    """
    proven = set()
    if ev.permutation_p > MAX_PERMUTATION_P:
        return proven
    if ev.executable_permutation_p > MAX_PERMUTATION_P:
        return proven
    if ev.magnitude_ratio >= MIN_MAGNITUDE_RATIO:
        proven.add(Moment.ABS_MEAN)
        proven.add(Moment.ABS_TAIL)
    fold_frac = ev.directional_folds / ev.n_folds if ev.n_folds else 0.0
    if ev.directional_edge >= MIN_DIRECTIONAL_EDGE and fold_frac >= MIN_DIRECTIONAL_FOLD_FRAC:
        proven.add(Moment.SIGNED_RIGHT_TAIL if ev.lane.startswith("up")
                   else Moment.SIGNED_LEFT_TAIL)
    return proven


def eligible_structures(ev: SignalEvidence, half_spread_pct: float = 0.045) -> list[Structure]:
    """Structures whose moment is proven, cheapest friction first."""
    proven = validated_moments(ev)
    out = [s for s in STRUCTURES.values() if s.moment in proven]
    # A magnitude-only lane must never be routed to a short-premium structure:
    # high |move| is exactly what ABS_BODY loses to.
    out = [s for s in out if s.moment is not Moment.ABS_BODY]
    return sorted(out, key=lambda s: s.friction_estimate(half_spread_pct))


def explain(ev: SignalEvidence, half_spread_pct: float = 0.045) -> str:
    proven = validated_moments(ev)
    fold_frac = ev.directional_folds / ev.n_folds if ev.n_folds else 0.0
    lines = [
        f"lane={ev.lane}  equity_perm_p={ev.permutation_p:.3f}  "
        f"executable_perm_p={ev.executable_permutation_p:.3f}  "
        f"folds_beating_null={ev.folds_beating_null}/{ev.n_folds}",
        f"  directional_edge {ev.directional_edge:+.3f} (bar {MIN_DIRECTIONAL_EDGE:+.2f})"
        f" in {ev.directional_folds}/{ev.n_folds} folds"
        f" -> {'PROVEN' if ev.directional_edge >= MIN_DIRECTIONAL_EDGE and fold_frac >= MIN_DIRECTIONAL_FOLD_FRAC else 'NOT PROVEN'}",
        f"  magnitude_ratio  {ev.magnitude_ratio:.3f} (bar {MIN_MAGNITUDE_RATIO:.2f})"
        f" -> {'PROVEN' if ev.magnitude_ratio >= MIN_MAGNITUDE_RATIO else 'NOT PROVEN'}",
        f"  validated moments: {sorted(m.value for m in proven) or 'NONE'}",
    ]
    elig = eligible_structures(ev, half_spread_pct)
    if not elig:
        lines.append("  eligible structures: NONE -> lane is research/risk-screen only")
    else:
        lines.append("  eligible structures (cheapest friction first):")
        for s in elig:
            lines.append(f"    {s.key:<20} moment={s.moment.value:<18} "
                         f"crossings={s.total_crossings}  "
                         f"friction~{s.friction_estimate(half_spread_pct):.1%} of premium")
    return "\n".join(lines)
