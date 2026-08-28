"""Exact-chain ORATS EOD replay for six directional swing strategies."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from codexswing.backtest.labels import DailyBar
from codexswing.schemas.source import SourceRecord


REPLAY_SCHEMA_VERSION = "codexswing.orats_option_replay.v3"
HOLDING_SESSIONS = 5
ROUND_TRIP_VERTICAL_COMMISSIONS = 2.60
ROUND_TRIP_SINGLE_COMMISSIONS = 1.30
STRATEGIES = (
    "LONG_CALL",
    "LONG_PUT",
    "BULL_CALL_DEBIT",
    "BEAR_PUT_DEBIT",
    "BULL_PUT_CREDIT",
    "BEAR_CALL_CREDIT",
)


class OptionReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplaySample:
    ticker: str
    side: str
    decision_date: str
    entry_date: str
    exit_date: str
    decision_high: float
    decision_low: float
    entry_open: float
    entry_high: float
    entry_low: float
    entry_close: float
    entry_unadjusted_close: float
    entry_session_index: int


@dataclass(frozen=True)
class HistoricalVertical:
    strategy: str
    ticker: str
    side: str
    right: str
    entry_date: str
    expiration: str
    dte: int
    spot: float
    long_strike: float
    short_strike: float
    long_delta: float
    short_delta: float
    long_bid: float
    long_ask: float
    short_bid: float
    short_ask: float
    long_open_interest: int
    short_open_interest: int
    long_volume: int
    short_volume: int
    width: float
    natural_open_signed_debit: float
    opposite_natural_signed_debit: float
    midpoint_signed_debit: float
    modeled_entry_signed_debit: float
    maximum_leg_relative_spread: float
    maximum_risk_dollars: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalSingleOption:
    strategy: str
    ticker: str
    side: str
    right: str
    entry_date: str
    expiration: str
    dte: int
    spot: float
    long_strike: float
    long_delta: float
    long_bid: float
    long_ask: float
    long_open_interest: int
    long_volume: int
    modeled_entry_signed_debit: float
    maximum_leg_relative_spread: float
    maximum_risk_dollars: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayObservation:
    ticker: str
    side: str
    strategy: str
    decision_date: str
    entry_date: str
    exit_date: str
    entry_session_index: int
    disposition: str
    reason: str
    trigger_price: float
    vertical: Optional[Union[HistoricalVertical, HistoricalSingleOption]]
    exit_position_value_signed_debit: Optional[float]
    net_pnl_dollars: Optional[float]
    return_on_risk: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.vertical is not None:
            value["vertical"] = self.vertical.to_dict()
        return value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _ticker(row: Mapping[str, Any]) -> str:
    return str(row.get("ticker") or row.get("symbol") or "").strip().upper()


def _row_date(row: Mapping[str, Any]) -> str:
    return str(row.get("tradeDate") or row.get("date") or "")[:10]


def _expiry(row: Mapping[str, Any]) -> str:
    return str(row.get("expirDate") or row.get("expiration") or "")[:10]


def _strike(row: Mapping[str, Any]) -> float:
    return _number(row.get("strike"))


def _quote(row: Mapping[str, Any], right: str) -> Tuple[float, float, int, int]:
    prefix = "call" if right == "call" else "put"
    return (
        _number(row.get(prefix + "BidPrice"), -1.0),
        _number(row.get(prefix + "AskPrice"), -1.0),
        int(_number(row.get(prefix + "Volume"))),
        int(_number(row.get(prefix + "OpenInterest"))),
    )


def _absolute_delta(row: Mapping[str, Any], right: str) -> float:
    raw = _number(row.get("delta"), float("nan"))
    if not math.isfinite(raw):
        return -1.0
    if right == "call":
        return abs(raw)
    # ORATS historical strikes commonly expose call-style delta in [0,1].
    return abs(raw - 1.0) if -0.05 <= raw <= 1.05 else abs(raw)


def _relative_spread(bid: float, ask: float) -> float:
    if bid < 0 or ask < bid:
        return float("inf")
    return (ask - bid) / max((ask + bid) / 2.0, 0.05)


def _strategy_geometry(strategy: str) -> Tuple[str, str, str]:
    mapping = {
        "BULL_CALL_DEBIT": ("LONG", "call", "debit"),
        "BEAR_PUT_DEBIT": ("SHORT", "put", "debit"),
        "BULL_PUT_CREDIT": ("LONG", "put", "credit"),
        "BEAR_CALL_CREDIT": ("SHORT", "call", "credit"),
    }
    if strategy not in mapping:
        raise ValueError("unsupported replay strategy")
    return mapping[strategy]


def select_historical_vertical(
    rows: Sequence[Mapping[str, Any]],
    ticker: str,
    side: str,
    entry_date: str,
    strategy: str,
) -> Tuple[Optional[HistoricalVertical], str]:
    expected_side, right, kind = _strategy_geometry(strategy)
    if side.upper() != expected_side:
        return None, "strategy_direction_mismatch"
    chain = [
        row
        for row in rows
        if _ticker(row) == ticker.upper() and _row_date(row) == entry_date
    ]
    if not chain:
        return None, "missing_entry_chain"
    spots = [_number(row.get("stockPrice")) for row in chain if _number(row.get("stockPrice")) > 0]
    if not spots:
        return None, "missing_entry_spot"
    spot = statistics.median(spots)
    by_expiry: Dict[str, List[Mapping[str, Any]]] = {}
    for row in chain:
        expiration = _expiry(row)
        try:
            dte = (date.fromisoformat(expiration) - date.fromisoformat(entry_date)).days
        except ValueError:
            continue
        if not 21 <= dte <= 60:
            continue
        bid, ask, _, _ = _quote(row, right)
        if bid <= 0 or ask < bid or _absolute_delta(row, right) < 0:
            continue
        by_expiry.setdefault(expiration, []).append(row)

    candidates: List[Tuple[float, HistoricalVertical]] = []
    for expiration, expiration_rows in by_expiry.items():
        dte = (date.fromisoformat(expiration) - date.fromisoformat(entry_date)).days
        for long_row in expiration_rows:
            long_strike = _strike(long_row)
            long_delta = _absolute_delta(long_row, right)
            long_bid, long_ask, long_volume, long_oi = _quote(long_row, right)
            for short_row in expiration_rows:
                short_strike = _strike(short_row)
                if long_strike == short_strike:
                    continue
                short_delta = _absolute_delta(short_row, right)
                short_bid, short_ask, short_volume, short_oi = _quote(short_row, right)
                if kind == "debit":
                    geometry = (
                        long_strike < short_strike if right == "call" else long_strike > short_strike
                    )
                    delta_ok = 0.43 <= long_delta <= 0.72 and 0.18 <= short_delta <= 0.45
                else:
                    geometry = (
                        long_strike < short_strike if right == "put" else long_strike > short_strike
                    )
                    delta_ok = 0.05 <= long_delta <= 0.18 and 0.18 <= short_delta <= 0.32
                if not geometry or not delta_ok:
                    continue
                width = abs(long_strike - short_strike)
                if width < max(1.0, spot * 0.005) or width > spot * 0.065:
                    continue
                leg_spread = max(
                    _relative_spread(long_bid, long_ask),
                    _relative_spread(short_bid, short_ask),
                )
                if leg_spread > 0.25 or min(long_oi, short_oi) < 100 or min(long_volume, short_volume) < 10:
                    continue
                natural = long_ask - short_bid
                opposite = long_bid - short_ask
                midpoint = (natural + opposite) / 2.0
                modeled_entry = opposite + 0.66 * (natural - opposite)
                if kind == "debit" and not 0 < modeled_entry < width:
                    continue
                if kind == "credit" and not -width < modeled_entry < 0:
                    continue
                maximum_risk = (
                    modeled_entry * 100.0 + ROUND_TRIP_VERTICAL_COMMISSIONS
                    if kind == "debit"
                    else (width + modeled_entry) * 100.0 + ROUND_TRIP_VERTICAL_COMMISSIONS
                )
                score = (
                    abs(dte - 36) / 30.0
                    + 2.0 * leg_spread
                    + (
                        4.0 * abs(long_delta - 0.58) + 3.0 * abs(short_delta - 0.30)
                        if kind == "debit"
                        else 4.0 * abs(short_delta - 0.25) + 3.0 * abs(long_delta - 0.10)
                    )
                )
                candidates.append(
                    (
                        score,
                        HistoricalVertical(
                            strategy=strategy,
                            ticker=ticker.upper(),
                            side=side.upper(),
                            right=right,
                            entry_date=entry_date,
                            expiration=expiration,
                            dte=dte,
                            spot=spot,
                            long_strike=long_strike,
                            short_strike=short_strike,
                            long_delta=long_delta,
                            short_delta=short_delta,
                            long_bid=long_bid,
                            long_ask=long_ask,
                            short_bid=short_bid,
                            short_ask=short_ask,
                            long_open_interest=long_oi,
                            short_open_interest=short_oi,
                            long_volume=long_volume,
                            short_volume=short_volume,
                            width=width,
                            natural_open_signed_debit=natural,
                            opposite_natural_signed_debit=opposite,
                            midpoint_signed_debit=midpoint,
                            modeled_entry_signed_debit=modeled_entry,
                            maximum_leg_relative_spread=leg_spread,
                            maximum_risk_dollars=maximum_risk,
                        ),
                    )
                )
    if not candidates:
        return None, "no_vertical_passed_entry_gates"
    candidates.sort(key=lambda item: (item[0], item[1].expiration, item[1].long_strike))
    return candidates[0][1], "selected"


def select_historical_single_option(
    rows: Sequence[Mapping[str, Any]],
    ticker: str,
    side: str,
    entry_date: str,
    strategy: str,
) -> Tuple[Optional[HistoricalSingleOption], str]:
    expected_side = "LONG" if strategy == "LONG_CALL" else "SHORT"
    right = "call" if strategy == "LONG_CALL" else "put"
    if strategy not in {"LONG_CALL", "LONG_PUT"}:
        raise ValueError("unsupported single-option replay strategy")
    if side.upper() != expected_side:
        return None, "strategy_direction_mismatch"
    chain = [
        row
        for row in rows
        if _ticker(row) == ticker.upper() and _row_date(row) == entry_date
    ]
    if not chain:
        return None, "missing_entry_chain"
    spots = [_number(row.get("stockPrice")) for row in chain if _number(row.get("stockPrice")) > 0]
    if not spots:
        return None, "missing_entry_spot"
    spot = statistics.median(spots)
    candidates: List[Tuple[float, HistoricalSingleOption]] = []
    for row in chain:
        expiration = _expiry(row)
        try:
            dte = (date.fromisoformat(expiration) - date.fromisoformat(entry_date)).days
        except ValueError:
            continue
        if not 21 <= dte <= 60:
            continue
        bid, ask, volume, open_interest = _quote(row, right)
        delta = _absolute_delta(row, right)
        relative_spread = _relative_spread(bid, ask)
        if (
            bid <= 0
            or ask < bid
            or not 0.42 <= delta <= 0.62
            or relative_spread > 0.25
            or open_interest < 100
            or volume < 10
        ):
            continue
        entry_debit = bid + 0.75 * (ask - bid)
        score = (
            abs(dte - 36) / 30.0
            + 2.0 * relative_spread
            + 4.0 * abs(delta - 0.52)
        )
        candidates.append(
            (
                score,
                HistoricalSingleOption(
                    strategy=strategy,
                    ticker=ticker.upper(),
                    side=side.upper(),
                    right=right,
                    entry_date=entry_date,
                    expiration=expiration,
                    dte=dte,
                    spot=spot,
                    long_strike=_strike(row),
                    long_delta=delta,
                    long_bid=bid,
                    long_ask=ask,
                    long_open_interest=open_interest,
                    long_volume=volume,
                    modeled_entry_signed_debit=entry_debit,
                    maximum_leg_relative_spread=relative_spread,
                    maximum_risk_dollars=(
                        entry_debit * 100.0 + ROUND_TRIP_SINGLE_COMMISSIONS
                    ),
                ),
            )
        )
    if not candidates:
        return None, "no_single_option_passed_entry_gates"
    candidates.sort(key=lambda item: (item[0], item[1].expiration, item[1].long_strike))
    return candidates[0][1], "selected"


def close_historical_vertical(
    vertical: HistoricalVertical,
    rows: Sequence[Mapping[str, Any]],
    exit_date: str,
) -> Tuple[Optional[float], str]:
    chain = {
        (_expiry(row), _strike(row)): row
        for row in rows
        if _ticker(row) == vertical.ticker and _row_date(row) == exit_date
    }
    long_row = chain.get((vertical.expiration, vertical.long_strike))
    short_row = chain.get((vertical.expiration, vertical.short_strike))
    if long_row is None or short_row is None:
        return None, "missing_exact_exit_leg"
    long_bid, long_ask, _, _ = _quote(long_row, vertical.right)
    short_bid, short_ask, _, _ = _quote(short_row, vertical.right)
    if long_bid < 0 or long_ask < long_bid or short_bid < 0 or short_ask < short_bid:
        return None, "invalid_exact_exit_quote"
    # Natural liquidation is deliberately conservative; no midpoint fill.
    return long_bid - short_ask, "closed"


def close_historical_single_option(
    option: HistoricalSingleOption,
    rows: Sequence[Mapping[str, Any]],
    exit_date: str,
) -> Tuple[Optional[float], str]:
    row = next(
        (
            item
            for item in rows
            if _ticker(item) == option.ticker
            and _row_date(item) == exit_date
            and _expiry(item) == option.expiration
            and _strike(item) == option.long_strike
        ),
        None,
    )
    if row is None:
        return None, "missing_exact_exit_leg"
    bid, ask, _, _ = _quote(row, option.right)
    if bid < 0 or ask < bid:
        return None, "invalid_exact_exit_quote"
    return bid, "closed"


def build_replay_samples(
    current_ideas: Mapping[str, Any],
    bars_by_ticker: Mapping[str, Sequence[DailyBar]],
) -> Tuple[ReplaySample, ...]:
    samples = []
    for idea in current_ideas.get("ideas") or ():
        if not isinstance(idea, Mapping):
            continue
        ticker = str(idea.get("ticker") or "").upper()
        side = str(idea.get("direction") or "").upper()
        analog = idea.get("analog_evidence")
        analog_values = analog if isinstance(analog, Mapping) else {}
        dates = analog_values.get("analog_dates")
        if not isinstance(dates, list):
            dates = list(dates or ())
        bars = tuple(sorted(bars_by_ticker.get(ticker, ()), key=lambda item: item.trade_date))
        indexes = {bar.trade_date: index for index, bar in enumerate(bars)}
        for decision_date in dates:
            index = indexes.get(str(decision_date))
            if index is None or index + HOLDING_SESSIONS >= len(bars):
                continue
            decision = bars[index]
            entry = bars[index + 1]
            exit_bar = bars[index + HOLDING_SESSIONS]
            samples.append(
                ReplaySample(
                    ticker=ticker,
                    side=side,
                    decision_date=decision.trade_date,
                    entry_date=entry.trade_date,
                    exit_date=exit_bar.trade_date,
                    decision_high=decision.high,
                    decision_low=decision.low,
                    entry_open=entry.open,
                    entry_high=entry.high,
                    entry_low=entry.low,
                    entry_close=entry.close,
                    entry_unadjusted_close=entry.unadjusted_close or entry.close,
                    entry_session_index=index + 1,
                )
            )
    if not samples:
        raise OptionReplayError("no current-regime historical samples were available")
    unique = {(sample.ticker, sample.decision_date): sample for sample in samples}
    return tuple(sorted(unique.values(), key=lambda item: (item.entry_date, item.ticker)))


def required_chain_slices(samples: Sequence[ReplaySample]) -> Mapping[str, Sequence[str]]:
    result: Dict[str, set] = {}
    for sample in samples:
        result.setdefault(sample.entry_date, set()).add(sample.ticker)
        result.setdefault(sample.exit_date, set()).add(sample.ticker)
    return {
        trade_date: tuple(sorted(tickers))
        for trade_date, tickers in sorted(result.items())
    }


def _trigger(sample: ReplaySample) -> Tuple[str, float]:
    if sample.side == "LONG":
        trigger = sample.decision_high * 1.001
        if sample.entry_high < trigger:
            return "NO_TRIGGER", trigger
        if sample.entry_open > trigger * 1.01:
            return "GAP_SKIP", trigger
        if sample.entry_close <= sample.decision_low:
            return "INVALIDATED", trigger
    else:
        trigger = sample.decision_low * 0.999
        if sample.entry_low > trigger:
            return "NO_TRIGGER", trigger
        if sample.entry_open < trigger * 0.99:
            return "GAP_SKIP", trigger
        if sample.entry_close >= sample.decision_high:
            return "INVALIDATED", trigger
    return "TRIGGERED", trigger


def _strategies_for_side(side: str) -> Tuple[str, str, str]:
    return (
        ("LONG_CALL", "BULL_CALL_DEBIT", "BULL_PUT_CREDIT")
        if side == "LONG"
        else ("LONG_PUT", "BEAR_PUT_DEBIT", "BEAR_CALL_CREDIT")
    )


def _wilson(wins: int, total: int) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _effective_nonoverlap(observations: Sequence[ReplayObservation]) -> int:
    indexes = sorted({item.entry_session_index for item in observations if item.net_pnl_dollars is not None})
    selected: List[int] = []
    for index in indexes:
        if not selected or index - selected[-1] >= HOLDING_SESSIONS:
            selected.append(index)
    return len(selected)


def _bootstrap_lower(observations: Sequence[ReplayObservation], seed_text: str) -> Optional[float]:
    closed = [item for item in observations if item.net_pnl_dollars is not None]
    if not closed:
        return None
    clusters: Dict[int, List[float]] = {}
    for item in closed:
        clusters.setdefault(item.entry_session_index // HOLDING_SESSIONS, []).append(float(item.net_pnl_dollars))
    cluster_means = [statistics.fmean(values) for _, values in sorted(clusters.items())]
    generator = random.Random(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16))
    estimates = []
    for _ in range(4_000):
        estimates.append(statistics.fmean(generator.choice(cluster_means) for _ in cluster_means))
    estimates.sort()
    return estimates[max(0, int(0.025 * len(estimates)) - 1)]


def _metrics(observations: Sequence[ReplayObservation], seed_text: str) -> Mapping[str, Any]:
    closed = [item for item in observations if item.net_pnl_dollars is not None]
    pnl = [float(item.net_pnl_dollars) for item in closed]
    wins = sum(value > 0 for value in pnl)
    lower, upper = _wilson(wins, len(pnl))
    gains = sum(value for value in pnl if value > 0)
    losses = abs(sum(value for value in pnl if value < 0))
    rejection_counts: Dict[str, int] = {}
    for item in observations:
        if item.disposition != "CLOSED":
            rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
    return {
        "sample_count": len(observations),
        "closed_count": len(closed),
        "effective_nonoverlapping_trade_count": _effective_nonoverlap(observations),
        "probability_of_profit": wins / len(pnl) if pnl else None,
        "wilson_95_lower_bound": lower,
        "wilson_95_upper_bound": upper,
        "mean_net_pnl_dollars": statistics.fmean(pnl) if pnl else None,
        "median_net_pnl_dollars": statistics.median(pnl) if pnl else None,
        "bootstrap_2_5_percent_mean_net_pnl_dollars": _bootstrap_lower(observations, seed_text),
        "profit_factor": gains / losses if losses > 0 else (999.0 if gains > 0 else None),
        "profit_factor_is_capped_infinite": bool(losses == 0 and gains > 0),
        "mean_return_on_risk": statistics.fmean(
            float(item.return_on_risk) for item in closed if item.return_on_risk is not None
        ) if closed else None,
        "worst_realized_pnl_dollars": min(pnl) if pnl else None,
        "mean_maximum_risk_dollars": statistics.fmean(
            float(item.vertical.maximum_risk_dollars)
            for item in closed
            if item.vertical is not None
        ) if closed else None,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def _split_group(observations: Sequence[ReplayObservation], seed_text: str) -> Mapping[str, Any]:
    ordered_dates = sorted({item.decision_date for item in observations})
    if len(ordered_dates) < 30:
        train_dates = set(ordered_dates[: max(1, len(ordered_dates) // 2)])
        validation_dates = set(ordered_dates[max(1, len(ordered_dates) // 2) : -max(1, len(ordered_dates) // 5)])
        holdout_dates = set(ordered_dates[-max(1, len(ordered_dates) // 5) :])
    else:
        train_end = int(len(ordered_dates) * 0.50)
        validation_end = int(len(ordered_dates) * 0.70)
        train_dates = set(ordered_dates[:train_end])
        validation_dates = set(ordered_dates[train_end:validation_end])
        holdout_dates = set(ordered_dates[validation_end:])
    partitions = {
        "train": [item for item in observations if item.decision_date in train_dates],
        "validation": [item for item in observations if item.decision_date in validation_dates],
        "holdout": [item for item in observations if item.decision_date in holdout_dates],
    }
    result = {}
    for name, values in partitions.items():
        result[name] = {
            "start_date": min((item.decision_date for item in values), default=None),
            "end_date": max((item.decision_date for item in values), default=None),
            "metrics": _metrics(values, "{}:{}".format(seed_text, name)),
        }
    validation_mean = result["validation"]["metrics"].get("mean_net_pnl_dollars")
    holdout_mean = result["holdout"]["metrics"].get("mean_net_pnl_dollars")
    temporal_stability = bool(
        validation_mean is not None
        and holdout_mean is not None
        and validation_mean > 0
        and holdout_mean > 0
    )
    result["holdout"]["metrics"]["validation_pass"] = bool(
        validation_mean is not None and validation_mean > 0
    )
    result["holdout"]["metrics"]["parameter_stability_pass"] = temporal_stability
    result["holdout"]["metrics"].update(
        {
            "train_mean_net_pnl_dollars": result["train"]["metrics"].get(
                "mean_net_pnl_dollars"
            ),
            "train_profit_factor": result["train"]["metrics"].get("profit_factor"),
            "validation_mean_net_pnl_dollars": result["validation"]["metrics"].get(
                "mean_net_pnl_dollars"
            ),
            "validation_profit_factor": result["validation"]["metrics"].get(
                "profit_factor"
            ),
        }
    )
    return result


def run_orats_option_replay(
    *,
    current_ideas: Mapping[str, Any],
    samples: Sequence[ReplaySample],
    chain_records: Sequence[SourceRecord],
    spec_sha256: str,
) -> Mapping[str, Any]:
    chain_map: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for record in chain_records:
        if record.source != "orats_hist_strikes":
            raise OptionReplayError("chain batch contains an unexpected source")
        chain_map.setdefault((_row_date(record.payload), _ticker(record.payload)), []).append(record.payload)
    observations: List[ReplayObservation] = []
    for sample in samples:
        trigger_disposition, trigger_price = _trigger(sample)
        for strategy in _strategies_for_side(sample.side):
            if trigger_disposition != "TRIGGERED":
                observations.append(
                    ReplayObservation(
                        ticker=sample.ticker,
                        side=sample.side,
                        strategy=strategy,
                        decision_date=sample.decision_date,
                        entry_date=sample.entry_date,
                        exit_date=sample.exit_date,
                        entry_session_index=sample.entry_session_index,
                        disposition=trigger_disposition,
                        reason=trigger_disposition.lower(),
                        trigger_price=trigger_price,
                        vertical=None,
                        exit_position_value_signed_debit=None,
                        net_pnl_dollars=None,
                        return_on_risk=None,
                    )
                )
                continue
            if strategy in {"LONG_CALL", "LONG_PUT"}:
                vertical, reason = select_historical_single_option(
                    chain_map.get((sample.entry_date, sample.ticker), ()),
                    sample.ticker,
                    sample.side,
                    sample.entry_date,
                    strategy,
                )
            else:
                vertical, reason = select_historical_vertical(
                    chain_map.get((sample.entry_date, sample.ticker), ()),
                    sample.ticker,
                    sample.side,
                    sample.entry_date,
                    strategy,
                )
            if vertical is None:
                observations.append(
                    ReplayObservation(
                        ticker=sample.ticker,
                        side=sample.side,
                        strategy=strategy,
                        decision_date=sample.decision_date,
                        entry_date=sample.entry_date,
                        exit_date=sample.exit_date,
                        entry_session_index=sample.entry_session_index,
                        disposition="NO_ENTRY",
                        reason=reason,
                        trigger_price=trigger_price,
                        vertical=None,
                        exit_position_value_signed_debit=None,
                        net_pnl_dollars=None,
                        return_on_risk=None,
                    )
                )
                continue
            if abs(vertical.spot / sample.entry_unadjusted_close - 1.0) > 0.02:
                reason = "adjusted_close_spot_mismatch"
                exit_value = None
            else:
                if isinstance(vertical, HistoricalSingleOption):
                    exit_value, reason = close_historical_single_option(
                        vertical,
                        chain_map.get((sample.exit_date, sample.ticker), ()),
                        sample.exit_date,
                    )
                else:
                    exit_value, reason = close_historical_vertical(
                        vertical,
                        chain_map.get((sample.exit_date, sample.ticker), ()),
                        sample.exit_date,
                    )
            if exit_value is None:
                observations.append(
                    ReplayObservation(
                        ticker=sample.ticker,
                        side=sample.side,
                        strategy=strategy,
                        decision_date=sample.decision_date,
                        entry_date=sample.entry_date,
                        exit_date=sample.exit_date,
                        entry_session_index=sample.entry_session_index,
                        disposition="UNRESOLVED",
                        reason=reason,
                        trigger_price=trigger_price,
                        vertical=vertical,
                        exit_position_value_signed_debit=None,
                        net_pnl_dollars=None,
                        return_on_risk=None,
                    )
                )
                continue
            commissions = (
                ROUND_TRIP_SINGLE_COMMISSIONS
                if isinstance(vertical, HistoricalSingleOption)
                else ROUND_TRIP_VERTICAL_COMMISSIONS
            )
            net_pnl = (exit_value - vertical.modeled_entry_signed_debit) * 100.0 - commissions
            observations.append(
                ReplayObservation(
                    ticker=sample.ticker,
                    side=sample.side,
                    strategy=strategy,
                    decision_date=sample.decision_date,
                    entry_date=sample.entry_date,
                    exit_date=sample.exit_date,
                    entry_session_index=sample.entry_session_index,
                    disposition="CLOSED",
                    reason=(
                        "closed_at_exact_bid"
                        if isinstance(vertical, HistoricalSingleOption)
                        else "closed_at_exact_natural_liquidation"
                    ),
                    trigger_price=trigger_price,
                    vertical=vertical,
                    exit_position_value_signed_debit=exit_value,
                    net_pnl_dollars=net_pnl,
                    return_on_risk=net_pnl / max(vertical.maximum_risk_dollars, 0.01),
                )
            )
    groups = []
    for ticker in sorted({sample.ticker for sample in samples}):
        side = next(sample.side for sample in samples if sample.ticker == ticker)
        for strategy in _strategies_for_side(side):
            group = [
                item
                for item in observations
                if item.ticker == ticker and item.strategy == strategy
            ]
            splits = _split_group(group, "{}:{}:{}".format(spec_sha256, ticker, strategy))
            holdout_metrics = splits["holdout"]["metrics"]
            groups.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "train": splits["train"],
                    "validation": splits["validation"],
                    "holdout": splits["holdout"],
                    "holdout_gate_candidate": True,
                    "holdout_pass": bool(
                        holdout_metrics.get("closed_count", 0) >= 20
                        and holdout_metrics.get("effective_nonoverlapping_trade_count", 0) >= 8
                        and (holdout_metrics.get("mean_net_pnl_dollars") or 0) > 0
                        and (holdout_metrics.get("bootstrap_2_5_percent_mean_net_pnl_dollars") or 0) > 0
                        and (holdout_metrics.get("profit_factor") or 0) >= 1.10
                        and (holdout_metrics.get("wilson_95_lower_bound") or 0) >= 0.40
                        and holdout_metrics.get("validation_pass") is True
                        and holdout_metrics.get("parameter_stability_pass") is True
                    ),
                }
            )
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "HISTORICAL_EVIDENCE_ONLY",
        "as_of_date": current_ideas.get("as_of_date"),
        "spec_sha256": spec_sha256,
        "source_attribution": {
            "signal_and_trigger_path": "ORATS hist/dailies adjusted OHLCV",
            "option_entry_and_exit": "ORATS hist/strikes exact EOD bids and asks",
            "current_contract": "not used in historical outcomes; Schwab is evaluated separately",
        },
        "fill_model": {
            "entry": {
                "single_leg": "75% from bid toward ask",
                "two_leg_vertical": "66% of package width from favorable side",
            },
            "exit": {
                "single_leg": "exact bid",
                "two_leg_vertical": "exact natural liquidation, no midpoint",
            },
            "round_trip_commissions_dollars": {
                "single_leg": ROUND_TRIP_SINGLE_COMMISSIONS,
                "two_leg_vertical": ROUND_TRIP_VERTICAL_COMMISSIONS,
            },
            "holding_sessions": HOLDING_SESSIONS,
            "intraday_data_required": False,
        },
        "split_policy": "chronological 50% train, 20% validation, 30% untouched holdout",
        "sample_count": len(samples),
        "group_count": len(groups),
        "evaluated_hypothesis_count": len(groups),
        "multiple_testing_adjusted": False,
        "selection_bias_notice": (
            "All six strategy rules are reported. Tactical evidence is exploratory and risk-capped; "
            "it is not promoted as full evidence without a positive bootstrap lower bound."
        ),
        "holdout_pass_count": sum(bool(group["holdout_pass"]) for group in groups),
        "groups": groups,
        "observations": [item.to_dict() for item in observations],
        "broker_order_authorized": False,
    }
