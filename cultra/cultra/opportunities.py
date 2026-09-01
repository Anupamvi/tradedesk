"""Broad, payoff-first Cultra opportunity construction.

This is an exploratory present-day research surface.  It never promotes a
manual ticket and never labels its forecast-distribution probability as POP.
Exact contract identifiers remain in the machine artifact; the human board
uses expiration, strike, type, action, and ratio.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"
COMMISSION_AND_FEE_PER_CONTRACT_SIDE = 0.68
SLIPPAGE_FRACTION_OF_SPREAD = 0.10
MIN_SLIPPAGE_PER_SHARE = 0.01
CONTRACT_MULTIPLIER = 100
SIMULATIONS = 12_000


class OpportunityError(RuntimeError):
    """The broad opportunity set could not be reproduced safely."""


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpportunityError("required input artifact is unavailable") from exc


def _private_write(path: Path, data: bytes) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise OpportunityError("opportunity artifacts must remain inside Cultra/out") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(resolved.parent, 0o700)
    temporary = resolved.with_name(".%s.tmp-%d" % (resolved.name, os.getpid()))
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return resolved


def _private_json(path: Path, value: Any) -> Path:
    return _private_write(
        path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _relative_spread(option: Mapping[str, Any]) -> float:
    bid = float(option["bid"])
    ask = float(option["ask"])
    midpoint = (bid + ask) / 2.0
    return math.inf if midpoint <= 0 else (ask - bid) / midpoint


def _viable(option: Mapping[str, Any]) -> bool:
    return bool(
        float(option.get("ask") or 0.0) > 0.0
        and float(option.get("bid") or 0.0) >= 0.01
        and _relative_spread(option) <= 0.30
        and int(option.get("open_interest") or 0) >= 100
        and option.get("delta") is not None
    )


def _nearest_delta(
    options: Sequence[Mapping[str, Any]], option_type: str, target: float
) -> Optional[Mapping[str, Any]]:
    values = [item for item in options if item["option_type"] == option_type and _viable(item)]
    if not values:
        return None
    return min(
        values,
        key=lambda item: (
            abs(float(item["delta"]) - target),
            _relative_spread(item),
            float(item["strike"]),
        ),
    )


def _nearest_strike(
    options: Sequence[Mapping[str, Any]], option_type: str, strike: float
) -> Optional[Mapping[str, Any]]:
    values = [item for item in options if item["option_type"] == option_type and _viable(item)]
    if not values:
        return None
    return min(values, key=lambda item: (abs(float(item["strike"]) - strike), _relative_spread(item)))


def _leg(action: str, ratio: int, option: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "action": action,
        "ratio": int(ratio),
        "occ_symbol": option["occ_symbol"],
        "expiration": option["expiration"],
        "strike": float(option["strike"]),
        "option_type": option["option_type"],
        "bid": float(option["bid"]),
        "ask": float(option["ask"]),
        "quote_timestamp": option["timestamp"],
        "volume": option.get("volume"),
        "open_interest": option.get("open_interest"),
        "delta_market_heuristic_not_pop": option.get("delta"),
        "relative_spread": _relative_spread(option),
    }


def _same_contracts(legs: Sequence[Mapping[str, Any]]) -> bool:
    values = [str(item["occ_symbol"]) for item in legs]
    return len(values) != len(set(values))


def _catalog(
    options: Sequence[Mapping[str, Any]], *, direction: str, iv: float, forecast: float
) -> Tuple[Mapping[str, Any], ...]:
    result: List[Mapping[str, Any]] = []
    call40 = _nearest_delta(options, "CALL", 0.40)
    put40 = _nearest_delta(options, "PUT", -0.40)
    if direction == "BULLISH":
        long_type, long40 = "CALL", call40
        debit_long = _nearest_delta(options, "CALL", 0.55)
        debit_short = _nearest_delta(options, "CALL", 0.25)
        fly_one = _nearest_delta(options, "CALL", 0.58)
        fly_two = _nearest_delta(options, "CALL", 0.30)
        back_short = _nearest_delta(options, "CALL", 0.48)
        back_long = _nearest_delta(options, "CALL", 0.22)
        credit_short = _nearest_delta(options, "PUT", -0.25)
        credit_long = _nearest_delta(options, "PUT", -0.10)
        names = {
            "long": "LONG_CALL",
            "debit": "CALL_DEBIT_SPREAD",
            "fly": "CALL_BUTTERFLY",
            "broken": "CALL_BROKEN_WING_BUTTERFLY",
            "back": "CALL_BACKSPREAD",
            "credit": "PUT_CREDIT_SPREAD",
        }
    else:
        long_type, long40 = "PUT", put40
        debit_long = _nearest_delta(options, "PUT", -0.55)
        debit_short = _nearest_delta(options, "PUT", -0.25)
        fly_one = _nearest_delta(options, "PUT", -0.58)
        fly_two = _nearest_delta(options, "PUT", -0.30)
        back_short = _nearest_delta(options, "PUT", -0.48)
        back_long = _nearest_delta(options, "PUT", -0.22)
        credit_short = _nearest_delta(options, "CALL", 0.25)
        credit_long = _nearest_delta(options, "CALL", 0.10)
        names = {
            "long": "LONG_PUT",
            "debit": "PUT_DEBIT_SPREAD",
            "fly": "PUT_BUTTERFLY",
            "broken": "PUT_BROKEN_WING_BUTTERFLY",
            "back": "PUT_BACKSPREAD",
            "credit": "CALL_CREDIT_SPREAD",
        }
    if long40:
        result.append({"family": names["long"], "legs": [_leg("BUY", 1, long40)]})
    if debit_long and debit_short:
        right_order = (
            float(debit_short["strike"]) > float(debit_long["strike"])
            if long_type == "CALL"
            else float(debit_short["strike"]) < float(debit_long["strike"])
        )
        if right_order:
            result.append(
                {
                    "family": names["debit"],
                    "legs": [_leg("BUY", 1, debit_long), _leg("SELL", 1, debit_short)],
                }
            )
    if fly_one and fly_two:
        k1, k2 = float(fly_one["strike"]), float(fly_two["strike"])
        if (long_type == "CALL" and k2 > k1) or (long_type == "PUT" and k2 < k1):
            equal_target = k2 + (k2 - k1)
            broken_target = k2 + 1.5 * (k2 - k1)
            equal = _nearest_strike(options, long_type, equal_target)
            broken = _nearest_strike(options, long_type, broken_target)
            if equal:
                legs = [_leg("BUY", 1, fly_one), _leg("SELL", 2, fly_two), _leg("BUY", 1, equal)]
                if not _same_contracts(legs):
                    result.append({"family": names["fly"], "legs": legs})
            if broken:
                legs = [_leg("BUY", 1, fly_one), _leg("SELL", 2, fly_two), _leg("BUY", 1, broken)]
                if not _same_contracts(legs):
                    result.append({"family": names["broken"], "legs": legs})
    if back_short and back_long:
        right_order = (
            float(back_long["strike"]) > float(back_short["strike"])
            if long_type == "CALL"
            else float(back_long["strike"]) < float(back_short["strike"])
        )
        if right_order:
            result.append(
                {
                    "family": names["back"],
                    "legs": [_leg("SELL", 1, back_short), _leg("BUY", 2, back_long)],
                }
            )
    if credit_short and credit_long:
        right_order = (
            float(credit_long["strike"]) < float(credit_short["strike"])
            if credit_short["option_type"] == "PUT"
            else float(credit_long["strike"]) > float(credit_short["strike"])
        )
        if right_order:
            result.append(
                {
                    "family": names["credit"],
                    "legs": [_leg("SELL", 1, credit_short), _leg("BUY", 1, credit_long)],
                }
            )
    call20 = _nearest_delta(options, "CALL", 0.20)
    put20 = _nearest_delta(options, "PUT", -0.20)
    call_atm = _nearest_delta(options, "CALL", 0.50)
    put_atm = _nearest_delta(options, "PUT", -0.50)
    if forecast > iv and call20 and put20:
        result.append(
            {
                "family": "LONG_STRANGLE",
                "legs": [_leg("BUY", 1, put20), _leg("BUY", 1, call20)],
            }
        )
    if forecast > iv * 1.05 and call_atm and put_atm:
        result.append(
            {
                "family": "LONG_STRADDLE",
                "legs": [_leg("BUY", 1, put_atm), _leg("BUY", 1, call_atm)],
            }
        )
    call25 = _nearest_delta(options, "CALL", 0.25)
    call10 = _nearest_delta(options, "CALL", 0.10)
    put25 = _nearest_delta(options, "PUT", -0.25)
    put10 = _nearest_delta(options, "PUT", -0.10)
    if iv > forecast and all((call25, call10, put25, put10)):
        legs = [
            _leg("BUY", 1, put10),
            _leg("SELL", 1, put25),
            _leg("SELL", 1, call25),
            _leg("BUY", 1, call10),
        ]
        if not _same_contracts(legs):
            result.append({"family": "IRON_CONDOR", "legs": legs})
    return tuple(result)


def _natural_debit(legs: Sequence[Mapping[str, Any]]) -> float:
    return math.fsum(
        int(item["ratio"])
        * (float(item["ask"]) if item["action"] == "BUY" else -float(item["bid"]))
        for item in legs
    )


def _economic_debit(legs: Sequence[Mapping[str, Any]]) -> Tuple[float, float, float]:
    natural = _natural_debit(legs)
    total_contracts = sum(int(item["ratio"]) for item in legs)
    slippage_one_side = math.fsum(
        int(item["ratio"])
        * max(
            MIN_SLIPPAGE_PER_SHARE,
            (float(item["ask"]) - float(item["bid"])) * SLIPPAGE_FRACTION_OF_SPREAD,
        )
        for item in legs
    )
    round_trip_cost_per_share = (
        2.0 * slippage_one_side
        + 2.0 * total_contracts * COMMISSION_AND_FEE_PER_CONTRACT_SIDE / CONTRACT_MULTIPLIER
    )
    return natural, natural + round_trip_cost_per_share, round_trip_cost_per_share


def _payoff_per_share(spot: float, legs: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for item in legs:
        strike = float(item["strike"])
        intrinsic = max(spot - strike, 0.0) if item["option_type"] == "CALL" else max(strike - spot, 0.0)
        sign = 1.0 if item["action"] == "BUY" else -1.0
        total += sign * int(item["ratio"]) * intrinsic
    return total


def _economics(legs: Sequence[Mapping[str, Any]], spot: float) -> Optional[Mapping[str, Any]]:
    natural, economic, costs = _economic_debit(legs)
    strikes = sorted({float(item["strike"]) for item in legs})
    upper_slope = sum(
        (1 if item["action"] == "BUY" else -1) * int(item["ratio"])
        for item in legs
        if item["option_type"] == "CALL"
    )
    if upper_slope < 0:
        return None
    points = [0.0] + strikes + [max(max(strikes) * 3.0, spot * 4.0)]
    pnls = [(_payoff_per_share(value, legs) - economic) * CONTRACT_MULTIPLIER for value in points]
    maximum_loss = max(0.0, -min(pnls))
    if maximum_loss <= 0.0 or not math.isfinite(maximum_loss):
        return None
    maximum_profit = None if upper_slope > 0 else max(pnls)
    if maximum_profit is not None and maximum_profit <= 0.0:
        return None
    return {
        "natural_entry_per_share": natural,
        "round_trip_costs_per_share": costs,
        "economic_entry_per_share": economic,
        "maximum_loss": maximum_loss,
        "maximum_profit": maximum_profit,
    }


def _seed(*values: str) -> int:
    digest = hashlib.sha256("\x00".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _scenario_metrics(
    *,
    ticker: str,
    family: str,
    expiry: date,
    spot: float,
    legs: Sequence[Mapping[str, Any]],
    economic_debit: float,
    drift: float,
    volatility: float,
) -> Mapping[str, Any]:
    days = max(1, (expiry - date(2026, 8, 28)).days)
    horizon = days / 365.0
    generator = random.Random(_seed(ticker, family, expiry.isoformat(), "%.8f" % drift, "%.8f" % volatility))
    values = []
    for _ in range(SIMULATIONS):
        shock = generator.gauss(0.0, 1.0)
        terminal = spot * math.exp(
            (drift - 0.5 * volatility * volatility) * horizon
            + volatility * math.sqrt(horizon) * shock
        )
        values.append(
            (_payoff_per_share(terminal, legs) - economic_debit) * CONTRACT_MULTIPLIER
        )
    values.sort()
    wins = sum(value > 0.0 for value in values)
    probability = wins / len(values)
    standard_error = math.sqrt(max(probability * (1.0 - probability), 0.0) / len(values))
    return {
        "net_expected_value": math.fsum(values) / len(values),
        "scenario_probability_positive_not_pop": probability,
        "monte_carlo_95_interval_not_model_uncertainty": [
            max(0.0, probability - 1.96 * standard_error),
            min(1.0, probability + 1.96 * standard_error),
        ],
        "expected_shortfall_10pct": math.fsum(values[: max(1, len(values) // 10)])
        / max(1, len(values) // 10),
        "terminal_median_pnl": values[len(values) // 2],
        "terminal_90th_percentile_pnl": values[int(len(values) * 0.90)],
        "simulation_count": len(values),
        "annual_drift_assumption": drift,
        "annual_volatility_assumption": volatility,
        "horizon_calendar_days": days,
    }


def _plain_leg(item: Mapping[str, Any]) -> str:
    month = date.fromisoformat(str(item["expiration"])).strftime("%b %d")
    strike = "%g" % float(item["strike"])
    return "%s %dx %s $%s %s" % (
        "Buy" if item["action"] == "BUY" else "Sell",
        int(item["ratio"]),
        month,
        strike,
        str(item["option_type"]).lower(),
    )


def _money(value: Optional[float]) -> str:
    if value is None:
        return "Unlimited"
    return "$%.0f" % value


def _entry_label(value: float) -> str:
    return "$%.2f debit" % value if value >= 0 else "$%.2f credit" % abs(value)


def _practical_profit_text(item: Mapping[str, Any]) -> str:
    bounded_target_families = {
        "CALL_DEBIT_SPREAD",
        "PUT_DEBIT_SPREAD",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
        "CALL_BUTTERFLY",
        "PUT_BUTTERFLY",
        "CALL_BROKEN_WING_BUTTERFLY",
        "PUT_BROKEN_WING_BUTTERFLY",
        "IRON_CONDOR",
    }
    maximum = item["economics"]["maximum_profit"]
    if item["family"] in bounded_target_families and maximum is not None:
        return "%s max" % _money(float(maximum))
    return "$%.0f conservative 90th-percentile P/L" % float(
        item["conservative_scenario"]["terminal_90th_percentile_pnl"]
    )


def _append_setup_table(
    lines: List[str], values: Sequence[Mapping[str, Any]], *, full: bool
) -> None:
    if not values:
        lines.append("None.")
        return
    if full:
        lines.extend(
            [
                "| Ticker | Thesis | Structure and plain-English legs | Fri natural | Max loss | Practical profit potential | Point EV | Conservative EV | Scenario P+ | Conservative EV/risk |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Ticker | Structure and legs | Max loss | Practical profit potential | Conservative EV | Scenario P+ | Conservative EV/risk |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
    for item in values:
        structure = "%s — %s" % (
            item["family"].replace("_", " ").title(),
            "; ".join(item["human_legs"]),
        )
        economics = item["economics"]
        if full:
            thesis = "%s; 20d %+.1f%%; IV %.1f%% vs forecast %.1f%%; forecast R2 %.2f" % (
                item["direction"].title(),
                100.0 * float(item["thesis"]["momentum_20"]),
                100.0 * float(item["thesis"]["orats_iv30d"]),
                100.0 * float(item["thesis"]["orats_forecast20d"]),
                float(item["thesis"]["orats_forecast_r2"] or 0.0),
            )
            lines.append(
                "| %s | %s | %s | %s | %s | %s | $%.0f | $%.0f | %.1f%% | %.1f%% |"
                % (
                    item["ticker"],
                    thesis,
                    structure,
                    _entry_label(float(economics["natural_entry_per_share"])),
                    _money(float(economics["maximum_loss"])),
                    _practical_profit_text(item),
                    float(item["point_scenario"]["net_expected_value"]),
                    float(item["conservative_scenario"]["net_expected_value"]),
                    100.0 * float(item["conservative_scenario"]["scenario_probability_positive_not_pop"]),
                    100.0 * float(item["conservative_ev_per_max_loss"]),
                )
            )
        else:
            lines.append(
                "| %s | %s | %s | %s | $%.0f | %.1f%% | %.1f%% |"
                % (
                    item["ticker"],
                    structure,
                    _money(float(economics["maximum_loss"])),
                    _practical_profit_text(item),
                    float(item["conservative_scenario"]["net_expected_value"]),
                    100.0 * float(item["conservative_scenario"]["scenario_probability_positive_not_pop"]),
                    100.0 * float(item["conservative_ev_per_max_loss"]),
                )
            )


def _append_primary_cards(
    lines: List[str], values: Sequence[Mapping[str, Any]]
) -> None:
    """Render the decision-first human view without contract identifiers."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    order: List[str] = []
    for item in values:
        ticker = str(item["ticker"])
        if ticker not in grouped:
            grouped[ticker] = []
            order.append(ticker)
        grouped[ticker].append(item)
    for rank, ticker in enumerate(order, start=1):
        alternatives = grouped[ticker]
        item = alternatives[0]
        economics = item["economics"]
        quote = item["underlying_quote"]
        thesis = item["thesis"]
        lines.extend(
            [
                "### %d. %s — %s" % (rank, ticker, str(item["direction"]).lower()),
                "",
                "**Trade:** %s — %s."
                % (
                    item["family"].replace("_", " ").title(),
                    "; ".join(item["human_legs"]),
                ),
                "",
                "**Friday reference:** stock $%.2f; %s; maximum loss %s."
                % (
                    float(quote["last"]),
                    _entry_label(float(economics["natural_entry_per_share"])),
                    _money(float(economics["maximum_loss"])),
                ),
                "",
                "**Why it surfaced:** 20-session move %+.1f%%; 30-day IV %.1f%% versus ORATS 20-day forecast %.1f%%; forecast R2 %.2f."
                % (
                    100.0 * float(thesis["momentum_20"]),
                    100.0 * float(thesis["orats_iv30d"]),
                    100.0 * float(thesis["orats_forecast20d"]),
                    float(thesis["orats_forecast_r2"] or 0.0),
                ),
                "",
                "**Modeled economics:** conservative EV $%.0f (%0.1f%% of maximum loss); point EV $%.0f; upside scenario %s; modeled chance of profit %.1f%% (**scenario estimate, not POP**)."
                % (
                    float(item["conservative_scenario"]["net_expected_value"]),
                    100.0 * float(item["conservative_ev_per_max_loss"]),
                    float(item["point_scenario"]["net_expected_value"]),
                    _practical_profit_text(item),
                    100.0
                    * float(
                        item["conservative_scenario"][
                            "scenario_probability_positive_not_pop"
                        ]
                    ),
                ),
            ]
        )
        for alternative in alternatives[1:]:
            alt_economics = alternative["economics"]
            lines.extend(
                [
                    "",
                    "**Alternative expression:** %s — %s; %s; maximum loss %s; conservative EV $%.0f (%0.1f%% of risk); modeled chance %.1f%%, not POP."
                    % (
                        alternative["family"].replace("_", " ").title(),
                        "; ".join(alternative["human_legs"]),
                        _entry_label(
                            float(alt_economics["natural_entry_per_share"])
                        ),
                        _money(float(alt_economics["maximum_loss"])),
                        float(
                            alternative["conservative_scenario"][
                                "net_expected_value"
                            ]
                        ),
                        100.0
                        * float(alternative["conservative_ev_per_max_loss"]),
                        100.0
                        * float(
                            alternative["conservative_scenario"][
                                "scenario_probability_positive_not_pop"
                            ]
                        ),
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "**Monday decision:** recompute with a fresh Schwab stock and every-leg quote; retain only if conservative EV remains positive and upside-to-risk remains at least 1.0x.",
                "",
            ]
        )


def build_opportunity_run(
    *,
    broad_screen: Path,
    history_screen: Path,
    orats_enrichment: Path,
    finalist_chains: Path,
    run_id: str,
    related_orats_ledgers: Sequence[Path] = (),
) -> Mapping[str, Any]:
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    broad = _load(broad_screen)
    histories = {item["ticker"]: item for item in _load(history_screen)["rows"]}
    orats = {item["ticker"]: item for item in _load(orats_enrichment)["rows"]}
    chains = {item["ticker"]: item for item in _load(finalist_chains)["chains"]}
    candidates = []
    rejected = []
    for ticker in sorted(chains):
        if ticker not in histories or ticker not in orats:
            rejected.append({"ticker": ticker, "reason": "missing history or ORATS Core row"})
            continue
        history = histories[ticker]
        analytics = orats[ticker]
        chain = chains[ticker]
        spot = (float(chain["underlying_quote"]["bid"]) + float(chain["underlying_quote"]["ask"])) / 2.0
        market_date = date.fromisoformat(str(analytics["tradeDate"]))
        expiry_values = {
            date.fromisoformat(str(item["expiration"]))
            for item in chain["contracts"]
            if 25 <= (date.fromisoformat(str(item["expiration"])) - market_date).days <= 52
        }
        viable_by_expiry = {
            value: sum(
                _viable(item)
                for item in chain["contracts"]
                if item["expiration"] == value.isoformat()
            )
            for value in expiry_values
        }
        expiries = sorted(
            expiry_values,
            key=lambda value: (
                0 if viable_by_expiry[value] >= 4 else 1,
                abs((value - market_date).days - 30),
                -viable_by_expiry[value],
                value,
            ),
        )
        if not expiries:
            rejected.append({"ticker": ticker, "reason": "no 30-65 DTE expiration"})
            continue
        expiry = expiries[0]
        options = [item for item in chain["contracts"] if item["expiration"] == expiry.isoformat()]
        direction = "BULLISH" if float(history["trend_score"]) >= 0.0 else "BEARISH"
        horizon_days = max(1, (expiry - market_date).days)
        month_vols = []
        for month in range(1, 5):
            dte_value = analytics.get("dtExM%d" % month)
            iv_value = analytics.get("atmIvM%d" % month)
            if dte_value is not None and iv_value is not None:
                month_vols.append(
                    (abs(float(dte_value) - horizon_days), float(iv_value) / 100.0)
                )
        iv = min(month_vols)[1] if month_vols else float(analytics["iv30d"]) / 100.0
        forecast = float(analytics["orFcst20d"]) / 100.0
        longer_realized = float(
            analytics.get("orHvXern60d")
            or analytics.get("orHv60d")
            or analytics["iv60d"]
        ) / 100.0
        forecast_days = min(28, horizon_days)
        point_base_vol = math.sqrt(
            (
                forecast * forecast * forecast_days
                + longer_realized * longer_realized * (horizon_days - forecast_days)
            )
            / horizon_days
        )
        weeks_to_earnings = float(analytics.get("wksNextErn") or 0.0)
        event_in_horizon = weeks_to_earnings * 7.0 <= horizon_days
        event_move = (
            float(analytics.get("absAvgErnMv") or 0.0) / 100.0
            if event_in_horizon
            else 0.0
        )
        point_vol = math.sqrt(
            point_base_vol * point_base_vol
            + (event_move * event_move) / (horizon_days / 365.0)
        )
        forecast_reliability = max(
            0.0,
            min(
                1.0,
                float(analytics.get("fcstR2") or 0.0)
                * float(analytics.get("confidence") or 0.0)
                / 100.0,
            ),
        )
        conservative_vol = iv + forecast_reliability * (point_vol - iv)
        raw_20_return = math.log1p(float(history["momentum_20"]))
        annual_drift = max(-0.30, min(0.30, 0.20 * raw_20_return * 252.0 / 20.0))
        if direction == "BULLISH":
            annual_drift = abs(annual_drift)
        else:
            annual_drift = -abs(annual_drift)
        for hypothesis in _catalog(options, direction=direction, iv=iv, forecast=forecast):
            family = str(hypothesis["family"])
            legs = hypothesis["legs"]
            economics = _economics(legs, spot)
            if economics is None:
                rejected.append({"ticker": ticker, "family": family, "reason": "no finite positive maximum loss/profit geometry"})
                continue
            neutral = family in {"LONG_STRANGLE", "LONG_STRADDLE", "IRON_CONDOR"}
            point_drift = 0.0 if neutral else annual_drift
            conservative_drift = 0.0 if neutral else annual_drift * 0.50
            point = _scenario_metrics(
                ticker=ticker,
                family=family,
                expiry=expiry,
                spot=spot,
                legs=legs,
                economic_debit=float(economics["economic_entry_per_share"]),
                drift=point_drift,
                volatility=point_vol,
            )
            conservative = _scenario_metrics(
                ticker=ticker,
                family=family,
                expiry=expiry,
                spot=spot,
                legs=legs,
                economic_debit=float(economics["economic_entry_per_share"]),
                drift=conservative_drift,
                volatility=conservative_vol,
            )
            max_profit = economics["maximum_profit"]
            if max_profit is None:
                upside_multiple = max(0.0, float(conservative["terminal_90th_percentile_pnl"])) / float(economics["maximum_loss"])
                upside_basis = "conservative terminal 90th percentile / max loss"
            else:
                upside_multiple = float(max_profit) / float(economics["maximum_loss"])
                upside_basis = "maximum profit / max loss"
            point_ev = float(point["net_expected_value"])
            conservative_ev = float(conservative["net_expected_value"])
            positive = point_ev > 0.0 and conservative_ev > 0.0
            strong_upside = upside_multiple >= 1.0
            if positive and strong_upside:
                disposition = "MONDAY_REPRICE_REQUIRED_HIGH_POTENTIAL"
            elif positive:
                disposition = "POSITIVE_MODEL_EV_LOW_UPSIDE"
            else:
                disposition = "REJECT_NONPOSITIVE_MODEL_EV"
            candidate_id = "%s-%s-%s" % (ticker, family, expiry.strftime("%Y%m%d"))
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "ticker": ticker,
                    "direction": direction,
                    "family": family,
                    "thesis": {
                        "trend_score": history["trend_score"],
                        "momentum_20": history["momentum_20"],
                        "momentum_60": history["momentum_60"],
                        "orats_iv30d": iv,
                        "orats_forecast20d": forecast,
                        "orats_confidence": analytics["confidence"],
                        "orats_contango": analytics["contango"],
                        "orats_slope": analytics["slope"],
                        "orats_slope_forecast": analytics.get("slopeFcst"),
                        "orats_forecast_r2": analytics.get("fcstR2"),
                        "selected_market_atm_iv": iv,
                        "forecast_reliability_weight": forecast_reliability,
                        "event_in_horizon": event_in_horizon,
                        "event_move_assumption": event_move,
                    },
                    "underlying_quote": chain["underlying_quote"],
                    "expiry": expiry.isoformat(),
                    "legs": legs,
                    "human_legs": [_plain_leg(item) for item in legs],
                    "economics": economics,
                    "point_scenario": point,
                    "conservative_scenario": conservative,
                    "upside_multiple": upside_multiple,
                    "upside_basis": upside_basis,
                    "conservative_ev_per_max_loss": conservative_ev / float(economics["maximum_loss"]),
                    "disposition": disposition,
                    "probability_label": "FORECAST_SCENARIO_PROBABILITY_NOT_CALIBRATED_POP",
                    "evidence_state": "EXPLORATORY_BROAD_EQUITY_NOT_HISTORICALLY_VALIDATED",
                    "quantity": "USER DETERMINED",
                    "manual_ticket_enabled": False,
                    "broker_submission_enabled": False,
                }
            )
    candidates.sort(
        key=lambda item: (
            0 if item["disposition"] == "MONDAY_REPRICE_REQUIRED_HIGH_POTENTIAL" else 1,
            -float(item["conservative_ev_per_max_loss"]),
            -float(item["upside_multiple"]),
            item["ticker"],
            item["family"],
        )
    )
    high = [item for item in candidates if item["disposition"] == "MONDAY_REPRICE_REQUIRED_HIGH_POTENTIAL"]
    primary = [item for item in high if float(item["conservative_ev_per_max_loss"]) >= 0.10]
    secondary = [
        item
        for item in high
        if 0.05 <= float(item["conservative_ev_per_max_loss"]) < 0.10
    ]
    marginal = [item for item in high if float(item["conservative_ev_per_max_loss"]) < 0.05]
    for item in primary:
        item["board_tier"] = "PRIMARY_AT_LEAST_10PCT_CONSERVATIVE_EV_PER_RISK"
    for item in secondary:
        item["board_tier"] = "SECONDARY_5_TO_10PCT_CONSERVATIVE_EV_PER_RISK"
    for item in marginal:
        item["board_tier"] = "MARGINAL_POSITIVE_BELOW_5PCT_CONSERVATIVE_EV_PER_RISK"
    low = [item for item in candidates if item["disposition"] == "POSITIVE_MODEL_EV_LOW_UPSIDE"]
    negative = [item for item in candidates if item["disposition"] == "REJECT_NONPOSITIVE_MODEL_EV"]
    ledger_records = []
    for ledger_path in related_orats_ledgers:
        payload = _load(ledger_path)
        summary = payload.get("summary", {})
        ledger_records.append(
            {
                "path": str(Path(ledger_path).resolve()),
                "sha256": hashlib.sha256(Path(ledger_path).read_bytes()).hexdigest(),
                "run_id": summary.get("run_id"),
                "logical_requests": summary.get("planned_logical_requests"),
                "charged_attempts": summary.get("charged_attempts"),
                "outbound_http_attempts": summary.get("outbound_http_attempts"),
                "state": summary.get("state"),
            }
        )
    request_reconciliation = {
        "schema": "cultra.related-request-reconciliation.v1",
        "ledgers": ledger_records,
        "total_planned_logical_requests": sum(
            int(item["logical_requests"] or 0) for item in ledger_records
        ),
        "total_charged_attempts": sum(
            int(item["charged_attempts"] or 0) for item in ledger_records
        ),
        "total_outbound_http_attempts": sum(
            int(item["outbound_http_attempts"] or 0) for item in ledger_records
        ),
    }
    result = {
        "schema": "cultra.broad-opportunities.v1",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_profit_confidence": "UNPROVEN",
        "coverage": {
            "source_constituents": broad["counts"]["source_constituents"],
            "schwab_quotes": broad["counts"]["schwab_quotes"],
            "orats_core_admitted": broad["counts"]["orats_admitted"],
            "orats_core_resolved": len(orats),
            "exact_chain_finalists": len(chains),
            "budget_unresolved": broad["counts"]["budget_unresolved"],
            "related_orats_charged_attempts": request_reconciliation[
                "total_charged_attempts"
            ],
        },
        "counts": {
            "high_potential": len(high),
            "primary": len(primary),
            "secondary": len(secondary),
            "marginal_positive": len(marginal),
            "positive_low_upside": len(low),
            "nonpositive_model_ev": len(negative),
            "construction_rejections": len(rejected),
            "manual_tickets": 0,
        },
        "high_potential": high,
        "primary": primary,
        "secondary": secondary,
        "marginal_positive": marginal,
        "positive_low_upside": low,
        "nonpositive_model_ev": negative,
        "construction_rejections": rejected,
        "budget_unresolved": broad["budget_unresolved"],
        "data_unavailable": broad["data_unavailable"],
        "model_limitations": [
            "Scenario probability is not calibrated POP.",
            "Broad-equity exact-leg historical validation has not yet been run.",
            "Friday closing quotes require a complete Schwab reprice before Monday entry.",
            "Scenario EV is exploratory and depends on explicitly saved drift and volatility assumptions.",
        ],
        "manual_ticket_enabled": False,
        "broker_submission_enabled": False,
    }
    json_path = _private_json(run_dir / "opportunities.json", result)
    reconciliation_path = _private_json(
        run_dir / "request_reconciliation.json", request_reconciliation
    )
    lines = [
        "# Cultra — Broad Profit-Potential Board",
        "",
        "**Status: exploratory Monday setups; no order is executable until every leg is repriced.**",
        "",
        "This run covered **%d SPY equity holdings from State Street's 27-Aug-2026 daily file**, enriched **%d names with ORATS Core**, and pulled exact Schwab chains for **%d finalists**. It is not the earlier ten-ETF pilot."
        % (
            result["coverage"]["source_constituents"],
            result["coverage"]["orats_core_resolved"],
            result["coverage"]["exact_chain_finalists"],
        ),
        "",
        "ORATS accounting for the correction: **%d planned logical requests, %d charged outbound attempts**."
        % (
            request_reconciliation["total_planned_logical_requests"],
            request_reconciliation["total_charged_attempts"],
        ),
        "",
        "## Start here — primary repricing queue",
        "",
        "These have positive point and conservative scenario EV, at least 1.0x upside-to-risk, and at least **10% conservative scenario EV per max-risk dollar**. Repeated structures for one ticker are grouped as alternatives. This is an emphasis tier, not suppression; every other positive appears below. `Scenario P+` is a model scenario and **not calibrated POP**.",
        "",
    ]
    _append_primary_cards(lines, primary)
    lines.extend(
        [
            "",
            "## Secondary positives — 5% to 10% conservative EV/risk",
            "",
        ]
    )
    _append_setup_table(lines, secondary, full=False)
    lines.extend(
        [
            "",
            "## Marginal positives — below 5% conservative EV/risk",
            "",
        ]
    )
    _append_setup_table(lines, marginal, full=False)
    lines.extend(
        [
            "",
            "Multiple structures on the same ticker are alternative expressions of one signal, not independent bets.",
            "",
            "## What to do with this board",
            "",
            "- Reprice every displayed leg from Schwab after Monday's open; Friday prices are reference prices only.",
            "- Discard a setup if refreshed maximum loss rises, conservative EV turns non-positive, or upside/risk falls below 1.0x.",
            "- Quantity remains **USER DETERMINED**. Cultra creates no broker order.",
            "",
            "## Positive model EV but weak payoff",
            "",
        ]
    )
    if low:
        lines.extend(
            [
                "| Ticker | Structure | Max loss | Max profit / 90th percentile | Conservative EV | Upside/risk | Why not headline it |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in low:
            economics = item["economics"]
            profit = (
                _money(economics["maximum_profit"])
                if economics["maximum_profit"] is not None
                else _money(float(item["conservative_scenario"]["terminal_90th_percentile_pnl"]))
            )
            lines.append(
                "| %s | %s | %s | %s | $%.0f | %.2fx | Positive exploratory EV, but less than 1.0x upside per risk dollar |"
                % (
                    item["ticker"],
                    item["family"].replace("_", " ").title(),
                    _money(float(economics["maximum_loss"])),
                    profit,
                    float(item["conservative_scenario"]["net_expected_value"]),
                    float(item["upside_multiple"]),
                )
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "There are **zero qualified tickets** here. The scenario probabilities are not POP, and the broad-equity structures have not passed exact-leg historical validation. The machine artifact retains OCC identifiers, all quotes, assumptions, rejected structures, and all %d budget-unresolved names; the human board intentionally does not lead with contract codes."
            % result["coverage"]["budget_unresolved"],
            "",
        ]
    )
    board_path = _private_write(run_dir / "BOARD.md", ("\n".join(lines) + "\n").encode("utf-8"))
    manifest = {
        "schema": "cultra.broad-opportunity-manifest.v1",
        "run_id": run_id,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (json_path, board_path, reconciliation_path)
        ],
        "inputs": [
            {
                "path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in (
                broad_screen,
                history_screen,
                orats_enrichment,
                finalist_chains,
                *related_orats_ledgers,
            )
        ],
        "manual_ticket_enabled": False,
        "broker_submission_enabled": False,
    }
    _private_json(run_dir / "manifest.json", manifest)
    return result


def verify_opportunity_run(run_dir: Path) -> Tuple[str, ...]:
    """Reproduce saved economics/scenarios and verify the plain-English board."""

    root = Path(run_dir).expanduser().resolve()
    errors: List[str] = []
    try:
        root.relative_to(OUT_ROOT.resolve())
        result = _load(root / "opportunities.json")
        manifest = _load(root / "manifest.json")
        board = (root / "BOARD.md").read_text(encoding="utf-8")
    except (ValueError, OSError, UnicodeError, OpportunityError) as exc:
        return ("opportunity artifact set is unavailable: %s" % exc,)
    if result.get("manual_ticket_enabled") is not False:
        errors.append("broad research improperly enables manual tickets")
    if result.get("broker_submission_enabled") is not False:
        errors.append("broad research improperly enables broker submission")
    if "occ_symbol" in board or re.search(r"\b[A-Z0-9.]{1,6}\d{6}[CP]\d{8}\b", board):
        errors.append("human board exposes machine contract identifiers")
    if "Scenario P+" not in board or "not calibrated POP" not in board:
        errors.append("human board does not distinguish scenario probability from POP")
    for record in manifest.get("files", []):
        path = root / str(record.get("path", ""))
        if not path.is_file():
            errors.append("manifest-listed output is missing")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            errors.append("manifest-listed output hash changed: %s" % path.name)
    for record in manifest.get("inputs", []):
        path = Path(str(record.get("path", "")))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            errors.append("opportunity input changed: %s" % path.name)
    categories = {
        "high_potential": result.get("high_potential", []),
        "positive_low_upside": result.get("positive_low_upside", []),
        "nonpositive_model_ev": result.get("nonpositive_model_ev", []),
    }
    seen = set()
    for category, values in categories.items():
        expected_count = result.get("counts", {}).get(category)
        if expected_count != len(values):
            errors.append("%s count does not reconcile" % category)
        for item in values:
            identifier = str(item.get("candidate_id", ""))
            if not identifier or identifier in seen:
                errors.append("candidate identity is missing or duplicated")
                continue
            seen.add(identifier)
            legs = item.get("legs", [])
            if not legs or any(not leg.get("occ_symbol") for leg in legs):
                errors.append("%s has incomplete exact legs" % identifier)
                continue
            spot_quote = item.get("underlying_quote", {})
            spot = (float(spot_quote["bid"]) + float(spot_quote["ask"])) / 2.0
            reproduced = _economics(legs, spot)
            if reproduced is None:
                errors.append("%s no longer has finite-risk economics" % identifier)
                continue
            for key in (
                "natural_entry_per_share",
                "round_trip_costs_per_share",
                "economic_entry_per_share",
                "maximum_loss",
            ):
                if not math.isclose(
                    float(reproduced[key]),
                    float(item["economics"][key]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append("%s %s cannot be reproduced" % (identifier, key))
            if reproduced["maximum_profit"] is None:
                if item["economics"]["maximum_profit"] is not None:
                    errors.append("%s maximum profit classification changed" % identifier)
            elif not math.isclose(
                float(reproduced["maximum_profit"]),
                float(item["economics"]["maximum_profit"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append("%s maximum profit cannot be reproduced" % identifier)
            expiry = date.fromisoformat(str(item["expiry"]))
            for scenario_name in ("point_scenario", "conservative_scenario"):
                saved = item[scenario_name]
                replayed = _scenario_metrics(
                    ticker=str(item["ticker"]),
                    family=str(item["family"]),
                    expiry=expiry,
                    spot=spot,
                    legs=legs,
                    economic_debit=float(item["economics"]["economic_entry_per_share"]),
                    drift=float(saved["annual_drift_assumption"]),
                    volatility=float(saved["annual_volatility_assumption"]),
                )
                for key in (
                    "net_expected_value",
                    "scenario_probability_positive_not_pop",
                    "expected_shortfall_10pct",
                    "terminal_90th_percentile_pnl",
                ):
                    if not math.isclose(
                        float(saved[key]), float(replayed[key]), rel_tol=0.0, abs_tol=1e-9
                    ):
                        errors.append("%s %s %s cannot be reproduced" % (identifier, scenario_name, key))
            point_ev = float(item["point_scenario"]["net_expected_value"])
            conservative_ev = float(item["conservative_scenario"]["net_expected_value"])
            upside = float(item["upside_multiple"])
            if category == "high_potential" and not (
                point_ev > 0.0 and conservative_ev > 0.0 and upside >= 1.0
            ):
                errors.append("%s does not satisfy the high-potential gate" % identifier)
            if category == "positive_low_upside" and not (
                point_ev > 0.0 and conservative_ev > 0.0 and upside < 1.0
            ):
                errors.append("%s does not satisfy the low-upside gate" % identifier)
            if category == "nonpositive_model_ev" and point_ev > 0.0 and conservative_ev > 0.0:
                errors.append("%s is incorrectly classified as nonpositive" % identifier)
    high_ids = {item["candidate_id"] for item in categories["high_potential"]}
    tier_ids = {
        item["candidate_id"]
        for name in ("primary", "secondary", "marginal_positive")
        for item in result.get(name, [])
    }
    if high_ids != tier_ids:
        errors.append("positive high-potential tier reconciliation failed")
    return tuple(errors)


__all__ = ["OpportunityError", "build_opportunity_run", "verify_opportunity_run"]
