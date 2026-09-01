"""Cultra V6: calibrated development audit and production-readiness gates.

This module replaces the disconnected ETF/current heuristic surfaces.  It
learns from Cultra's exact-leg development outcomes, evaluates the learner with
chronological embargoed folds and overlapping-exposure clusters, then applies
the same feature contract to saved current chains.  The current equity domain
does not match the ten-ETF development domain, so current model outputs remain
diagnostics and can never be promoted to POP, edge, or a ticket.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .backfill import load_recent_sessions
from .domain import parse_occ_symbol
from .learning import TARGETS, build_walk_forward_models, public_model_evidence
from .readiness import assess_production_readiness, render_readiness_markdown
from .research import DEFAULT_CHAIN_DB, generate_historical_trades


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"
CONFIG_PATH = PROJECT_ROOT / "configs" / "pattern_model.v2.json"
EVENTS_PATH = PROJECT_ROOT / "configs" / "confirmed_events.2026-08-30.json"
DEFAULT_SCREEN = OUT_ROOT / "cultra-broad-screen-2026-08-30-v1" / "schwab_screen.json"
DEFAULT_HISTORY = OUT_ROOT / "cultra-broad-screen-2026-08-30-v1" / "history_screen.json"
DEFAULT_ORATS = OUT_ROOT / "cultra-eod-core-full-2026-08-30-v2" / "orats_enrichment.json"
DEFAULT_CHAINS = OUT_ROOT / "cultra-broad-screen-2026-08-30-v1" / "finalist_chains.json"
DEFAULT_SELECTION = (
    OUT_ROOT / "cultra-chain-selection-all-2026-08-30-v1" / "selection.json"
)
PATTERN_SCHEMA = "cultra.pattern-run.v6"


class PatternError(RuntimeError):
    """The unified pattern run cannot be produced without hiding uncertainty."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PatternError("required Cultra artifact is unavailable: %s" % Path(path).name) from exc


def _private_write(path: Path, raw: bytes) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise PatternError("pattern outputs must remain inside Cultra/out") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(resolved.parent, 0o700)
    temporary = resolved.with_name(".%s.tmp-%d" % (resolved.name, os.getpid()))
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(raw)
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
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n",
    )


def _private_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> Path:
    raw = b"".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for value in values
    )
    return _private_write(path, raw)


def _source_fingerprint() -> Mapping[str, Any]:
    paths = tuple(sorted((PROJECT_ROOT / "cultra").glob("*.py"))) + (
        CONFIG_PATH,
        EVENTS_PATH,
    )
    files = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
        if path.is_file()
    }
    return {"tree_sha256": _sha256_value(files), "files": files}


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PatternError("%s is not numeric" % name) from exc
    if not math.isfinite(result):
        raise PatternError("%s is not finite" % name)
    return result


def _relative_spread(option: Mapping[str, Any]) -> float:
    bid = _number(option.get("bid"), "option bid")
    ask = _number(option.get("ask"), "option ask")
    midpoint = (bid + ask) / 2.0
    return math.inf if midpoint <= 0.0 else (ask - bid) / midpoint


def _viable(option: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    try:
        bid = _number(option.get("bid"), "option bid")
        ask = _number(option.get("ask"), "option ask")
        interest = int(option.get("open_interest") or 0)
        delta = _number(option.get("delta"), "option delta")
    except (PatternError, TypeError, ValueError):
        return False
    return bool(
        bid >= float(policy["minimum_bid"])
        and ask >= bid
        and interest >= int(policy["minimum_open_interest"])
        and _relative_spread(option) <= float(policy["maximum_relative_spread"])
        and -1.0 <= delta <= 1.0
    )


def _nearest_delta(
    options: Sequence[Mapping[str, Any]],
    option_type: str,
    target: float,
    policy: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    eligible = tuple(
        item
        for item in options
        if str(item.get("option_type")) == option_type and _viable(item, policy)
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            abs(_number(item["delta"], "option delta") - float(target)),
            _relative_spread(item),
            str(item["occ_symbol"]),
        ),
    )


def _expiration_options(
    chain: Mapping[str, Any], provider_date: date, policy: Mapping[str, Any]
) -> Tuple[str, Tuple[Mapping[str, Any], ...]]:
    by_expiration: Dict[str, List[Mapping[str, Any]]] = {}
    for item in chain.get("contracts", []):
        try:
            expiration = date.fromisoformat(str(item["expiration"]))
        except (KeyError, TypeError, ValueError):
            continue
        dte = (expiration - provider_date).days
        if int(policy["minimum_calendar_dte"]) <= dte <= int(
            policy["maximum_calendar_dte"]
        ):
            by_expiration.setdefault(expiration.isoformat(), []).append(item)
    if not by_expiration:
        raise PatternError("no expiration satisfies the frozen 20-60-session proxy")
    selected = min(
        by_expiration,
        key=lambda value: (
            abs(
                (date.fromisoformat(value) - provider_date).days
                - int(policy["preferred_calendar_dte"])
            ),
            value,
        ),
    )
    return selected, tuple(by_expiration[selected])


def _leg(action: str, option: Mapping[str, Any]) -> Mapping[str, Any]:
    symbol = str(option["occ_symbol"])
    _root, parsed_expiration, parsed_type, parsed_strike = parse_occ_symbol(symbol)
    if parsed_expiration.isoformat() != str(option["expiration"]):
        raise PatternError("OCC expiration does not match Schwab contract")
    if parsed_type.value != str(option["option_type"]):
        raise PatternError("OCC option type does not match Schwab contract")
    if not math.isclose(parsed_strike, float(option["strike"]), abs_tol=1e-9):
        raise PatternError("OCC strike does not match Schwab contract")
    return {
        "action": action,
        "ratio": 1,
        "occ_symbol": symbol,
        "expiration": str(option["expiration"]),
        "strike": _number(option["strike"], "strike"),
        "option_type": str(option["option_type"]),
        "bid": _number(option["bid"], "bid"),
        "ask": _number(option["ask"], "ask"),
        "delta_market_heuristic_not_pop": _number(option["delta"], "delta"),
        "relative_spread": _relative_spread(option),
        "open_interest": int(option.get("open_interest") or 0),
        "volume": int(option.get("volume") or 0),
        "quote_timestamp": str(option["timestamp"]),
    }


def _select_structure(
    family: str,
    options: Sequence[Mapping[str, Any]],
    family_policy: Mapping[str, Any],
    contract_policy: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    if family == "LONG_CALL":
        item = _nearest_delta(
            options, "CALL", float(family_policy["long_delta"]), contract_policy
        )
        if item is None:
            raise PatternError("no liquid call near the frozen long delta")
        return (_leg("BUY", item),)
    if family == "LONG_PUT":
        item = _nearest_delta(
            options,
            "PUT",
            -float(family_policy["long_absolute_delta"]),
            contract_policy,
        )
        if item is None:
            raise PatternError("no liquid put near the frozen long delta")
        return (_leg("BUY", item),)
    if family == "CALL_DEBIT_VERTICAL":
        long_leg = _nearest_delta(
            options, "CALL", float(family_policy["long_delta"]), contract_policy
        )
        short_candidates = tuple(
            item
            for item in options
            if str(item.get("option_type")) == "CALL"
            and _viable(item, contract_policy)
            and long_leg is not None
            and float(item["strike"]) > float(long_leg["strike"])
        )
        if long_leg is None or not short_candidates:
            raise PatternError("no liquid call debit vertical satisfies the frozen deltas")
        short_leg = min(
            short_candidates,
            key=lambda item: (
                abs(float(item["delta"]) - float(family_policy["short_delta"])),
                _relative_spread(item),
                str(item["occ_symbol"]),
            ),
        )
        return (_leg("BUY", long_leg), _leg("SELL", short_leg))
    if family == "PUT_DEBIT_VERTICAL":
        long_leg = _nearest_delta(
            options,
            "PUT",
            -float(family_policy["long_absolute_delta"]),
            contract_policy,
        )
        short_candidates = tuple(
            item
            for item in options
            if str(item.get("option_type")) == "PUT"
            and _viable(item, contract_policy)
            and long_leg is not None
            and float(item["strike"]) < float(long_leg["strike"])
        )
        if long_leg is None or not short_candidates:
            raise PatternError("no liquid put debit vertical satisfies the frozen deltas")
        short_leg = min(
            short_candidates,
            key=lambda item: (
                abs(abs(float(item["delta"])) - float(family_policy["short_absolute_delta"])),
                _relative_spread(item),
                str(item["occ_symbol"]),
            ),
        )
        return (_leg("BUY", long_leg), _leg("SELL", short_leg))
    raise PatternError("unsupported historically modeled strategy family")


def _economics(
    family: str,
    legs: Sequence[Mapping[str, Any]],
    cost_policy: Mapping[str, Any],
    exit_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not legs or len({str(item["occ_symbol"]) for item in legs}) != len(legs):
        raise PatternError("structure legs are empty or duplicated")
    multiplier = int(cost_policy["contract_multiplier"])
    natural = math.fsum(
        float(item["ask"]) if item["action"] == "BUY" else -float(item["bid"])
        for item in legs
    )
    if natural <= 0.0:
        raise PatternError("structure does not have a positive natural debit")
    one_side_slippage = math.fsum(
        max(
            float(cost_policy["minimum_slippage_per_share_per_leg_per_side"]),
            (float(item["ask"]) - float(item["bid"]))
            * float(cost_policy["additional_slippage_fraction_of_spread"]),
        )
        * multiplier
        for item in legs
    )
    commissions = len(legs) * 2 * (
        float(cost_policy["commission_per_contract_per_side"])
        + float(cost_policy["fee_per_contract_per_side"])
    )
    entry_debit = natural * multiplier
    maximum_loss = entry_debit + 2.0 * one_side_slippage + commissions
    maximum_profit: Optional[float] = None
    if family in {"CALL_DEBIT_VERTICAL", "PUT_DEBIT_VERTICAL"}:
        width = abs(float(legs[0]["strike"]) - float(legs[1]["strike"])) * multiplier
        maximum_profit = width - maximum_loss
        if maximum_profit <= 0.0:
            raise PatternError("vertical maximum profit is not positive after costs")
    if family in {"LONG_CALL", "CALL_DEBIT_VERTICAL"}:
        breakeven = float(legs[0]["strike"]) + maximum_loss / multiplier
    else:
        breakeven = float(legs[0]["strike"]) - maximum_loss / multiplier
    return {
        "natural_debit_per_share": natural,
        "proposed_limit_debit_per_share": round(natural, 2),
        "entry_debit_before_costs": entry_debit,
        "modeled_round_trip_slippage": 2.0 * one_side_slippage,
        "commissions_and_fees": commissions,
        "maximum_loss": maximum_loss,
        "maximum_profit": maximum_profit,
        "reward_to_risk": None if maximum_profit is None else maximum_profit / maximum_loss,
        "breakevens_at_expiration": [breakeven],
        "target_pnl": maximum_loss
        * float(exit_policy["profit_target_fraction_of_maximum_loss"]),
        "stop_pnl": -maximum_loss
        * float(exit_policy["stop_loss_fraction_of_maximum_loss"]),
        "time_exit_sessions": int(exit_policy["time_exit_sessions"]),
        "adverse_gap_stress_loss": -maximum_loss,
    }


def _human_legs(legs: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        "%s %dx %s %s $%g %s"
        % (
            "Buy" if item["action"] == "BUY" else "Sell",
            int(item["ratio"]),
            date.fromisoformat(str(item["expiration"])).strftime("%b %d"),
            date.fromisoformat(str(item["expiration"])).year,
            float(item["strike"]),
            str(item["option_type"]).lower(),
        )
        for item in legs
    )


def _historical_rows(database: Path) -> Tuple[Mapping[str, Any], ...]:
    trades, _unresolved, _counts = generate_historical_trades(database)
    return tuple(item.to_dict() for item in trades)


def _orats_snapshot_map(orats: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    snapshots = orats.get("snapshots", {})
    if not isinstance(snapshots, Mapping):
        raise PatternError("ORATS enrichment snapshots are missing")
    for logical_request_id, snapshot in snapshots.items():
        if not isinstance(snapshot, Mapping) or not snapshot.get("snapshot_id"):
            raise PatternError("ORATS snapshot provenance is incomplete")
        provenance = {
            "logical_request_id": str(logical_request_id),
            "snapshot_id": str(snapshot["snapshot_id"]),
            "field_profile": str(snapshot.get("field_profile") or ""),
            "provider_trade_dates": list(snapshot.get("provider_trade_dates", ())),
            "updated_at_min": snapshot.get("updated_at_min"),
            "updated_at_max": snapshot.get("updated_at_max"),
            "raw_sha256": snapshot.get("raw_sha256"),
        }
        for ticker in snapshot.get("returned_entities", ()):
            normalized = str(ticker)
            if normalized in result:
                raise PatternError("ORATS ticker is duplicated across saved snapshots")
            result[normalized] = provenance
    row_tickers = {str(item["ticker"]) for item in orats.get("rows", ())}
    if set(result) != row_tickers:
        raise PatternError("ORATS rows and snapshot entity provenance do not reconcile")
    return result


def _input_maps(
    screen: Mapping[str, Any],
    history: Mapping[str, Any],
    orats: Mapping[str, Any],
    chains: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    universe = tuple(screen.get("quotes", ()))
    tickers = tuple(str(item["ticker"]) for item in universe)
    if not tickers or len(tickers) != len(set(tickers)):
        raise PatternError("broad universe is empty or duplicated")
    maps = {
        "universe": {str(item["ticker"]): item for item in universe},
        "admitted": {str(item["ticker"]): item for item in screen.get("admitted", ())},
        "legacy_local": {
            str(item["ticker"]): item for item in screen.get("locally_rejected", ())
        },
        "orats_budget": {
            str(item["ticker"]): item for item in screen.get("budget_unresolved", ())
        },
        "screen_unavailable": {
            str(item["ticker"]): item for item in screen.get("data_unavailable", ())
        },
        "history": {str(item["ticker"]): item for item in history.get("rows", ())},
        "orats": {str(item["ticker"]): item for item in orats.get("rows", ())},
        "orats_snapshots": _orats_snapshot_map(orats),
        "chains": {str(item["ticker"]): item for item in chains.get("chains", ())},
        "selection": (
            {str(item) for item in selection.get("selected_symbols", ())}
            if selection is not None
            else None
        ),
    }
    initial_sets = (
        set(maps["admitted"]),
        set(maps["legacy_local"]),
        set(maps["orats_budget"]),
        set(maps["screen_unavailable"]),
    )
    if any(left.intersection(right) for index, left in enumerate(initial_sets) for right in initial_sets[index + 1 :]):
        raise PatternError("broad screen categories overlap")
    if set().union(*initial_sets) != set(tickers):
        raise PatternError("broad screen does not reconcile every source ticker")
    if not set(maps["history"]).issubset(set(maps["admitted"])):
        raise PatternError("history rows leave the admitted universe")
    if not set(maps["orats"]).issubset(set(maps["admitted"])):
        raise PatternError("ORATS rows leave the admitted universe")
    if not set(maps["chains"]).issubset(set(maps["admitted"])):
        raise PatternError("exact chains leave the admitted universe")
    if selection is not None:
        if selection.get("schema") != "cultra.chain-finalist-selection.v2":
            raise PatternError("chain selection manifest schema is invalid")
        selected = maps["selection"]
        if not selected or len(selected) != len(selection.get("selected_symbols", ())):
            raise PatternError("chain selection symbols are empty or duplicated")
        if not selected.issubset(set(maps["admitted"])):
            raise PatternError("chain selection leaves the admitted universe")
    return maps


def _provider_trade_date(orats_rows: Mapping[str, Mapping[str, Any]]) -> date:
    values = {str(item.get("tradeDate")) for item in orats_rows.values()}
    if len(values) != 1:
        raise PatternError("ORATS Core rows do not share one provider trade date")
    try:
        return date.fromisoformat(next(iter(values)))
    except ValueError as exc:
        raise PatternError("ORATS provider trade date is invalid") from exc


def _current_feature_row(
    *,
    family: str,
    history: Mapping[str, Any],
    analytics: Mapping[str, Any],
    economics: Mapping[str, Any],
    legs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    implied = _number(
        analytics.get("orIvXern20d")
        or analytics.get("iv20d")
        or analytics.get("orFcst20d"),
        "ORATS implied volatility",
    )
    if implied > 3.0:
        implied /= 100.0
    return {
        "strategy_family": family,
        "momentum_20": _number(history["momentum_20"], "momentum_20"),
        "realized_volatility_20": _number(
            history["realized_volatility_20"], "realized_volatility_20"
        ),
        "smv_vol": implied,
        "relative_spread": max(float(item["relative_spread"]) for item in legs),
        "entry_debit": float(economics["entry_debit_before_costs"]),
        "maximum_loss": float(economics["maximum_loss"]),
        "maximum_profit": economics["maximum_profit"],
    }


def _build_current_candidates(
    maps: Mapping[str, Any],
    models: Mapping[str, Any],
    config: Mapping[str, Any],
    provider_date: date,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], Mapping[str, str]]:
    candidates: List[Mapping[str, Any]] = []
    construction: List[Mapping[str, Any]] = []
    ticker_status: Dict[str, str] = {}
    expected_chains = (
        set(maps["selection"])
        if maps.get("selection") is not None
        else set(maps["orats"])
    )
    chain_coverage_provenance = (
        "COMPLETE_SELECTION_MANIFEST_CHAIN_SNAPSHOT"
        if maps.get("selection") is not None
        and set(maps["chains"]) == expected_chains
        else "PARTIAL_SAVED_CHAIN_SNAPSHOT_%d_OF_%d_SELECTION_MANIFEST"
        % (len(maps["chains"]), len(expected_chains))
        if maps.get("selection") is not None
        else "PARTIAL_SAVED_CHAIN_SNAPSHOT_%d_OF_%d_NO_SELECTION_MANIFEST"
        % (len(maps["chains"]), len(expected_chains))
    )
    for ticker, chain in sorted(maps["chains"].items()):
        history = maps["history"].get(ticker)
        analytics = maps["orats"].get(ticker)
        if history is None or analytics is None:
            ticker_status[ticker] = "DATA_UNAVAILABLE_MISSING_HISTORY_OR_ORATS"
            continue
        momentum = _number(history["momentum_20"], "momentum_20")
        if momentum >= float(config["signal_policy"]["bullish_momentum_threshold"]):
            families = ("LONG_CALL", "CALL_DEBIT_VERTICAL")
            direction = "BULLISH"
        elif momentum <= float(config["signal_policy"]["bearish_momentum_threshold"]):
            families = ("LONG_PUT", "PUT_DEBIT_VERTICAL")
            direction = "BEARISH"
        else:
            ticker_status[ticker] = "NO_FROZEN_DIRECTIONAL_PATTERN_SIGNAL"
            continue
        try:
            expiration, options = _expiration_options(
                chain, provider_date, config["contract_policy"]
            )
        except PatternError as exc:
            ticker_status[ticker] = "DATA_UNAVAILABLE_NO_ELIGIBLE_EXPIRATION"
            construction.append({"ticker": ticker, "family": None, "reason": str(exc)})
            continue
        built = 0
        for family in families:
            family_result = models.get(family, {})
            runtime = family_result.get("_runtime_models")
            runtime_calibrators = family_result.get("_runtime_calibrators", {})
            try:
                legs = _select_structure(
                    family,
                    options,
                    config["families"][family],
                    config["contract_policy"],
                )
                economics = _economics(
                    family, legs, config["cost_policy"], config["exit_policy"]
                )
                feature_row = _current_feature_row(
                    family=family,
                    history=history,
                    analytics=analytics,
                    economics=economics,
                    legs=legs,
                )
                if bool(config["families"][family]["requires_long_vol_value"]):
                    iv_to_realized = float(feature_row["smv_vol"]) / max(
                        1e-6, float(feature_row["realized_volatility_20"])
                    )
                    if iv_to_realized > float(
                        config["signal_policy"]["long_option_max_iv_to_realized_ratio"]
                    ):
                        raise PatternError(
                            "long option IV/realized ratio %.3f exceeds %.3f"
                            % (
                                iv_to_realized,
                                float(
                                    config["signal_policy"][
                                        "long_option_max_iv_to_realized_ratio"
                                    ]
                                ),
                            )
                        )
                metrics = family_result.get("metrics")
                if runtime is not None and metrics is not None:
                    raw_probabilities = {
                        target: runtime["probabilities"][target].predict(feature_row)
                        for target in TARGETS
                    }
                    calibrated_probabilities = {
                        target: calibrator.predict_one(raw_probabilities[target])
                        for target, calibrator in runtime_calibrators.items()
                    }
                    predicted_return = runtime["return_on_risk"].predict(feature_row)
                    residual_lower = float(
                        metrics["return_model"]["residual_bias_95_lower"]
                    )
                    conservative_return = predicted_return + residual_lower
                    point_dollars = predicted_return * float(economics["maximum_loss"])
                    conservative_dollars = conservative_return * float(
                        economics["maximum_loss"]
                    )
                    model_diagnostics = {
                        "publishable_as_profit_estimate": False,
                        "status": "DEVELOPMENT_CALIBRATED_BUT_FAILED_OR_OUT_OF_DOMAIN",
                        "raw_probabilities_not_pop": raw_probabilities,
                        "calibrated_probabilities_not_transferable": calibrated_probabilities,
                        "predicted_return_on_maximum_loss": predicted_return,
                        "residual_bias_adjusted_95_lower_return": conservative_return,
                        "predicted_net_dollars": point_dollars,
                        "residual_adjusted_lower_net_dollars": conservative_dollars,
                    }
                    if (
                        metrics["ev_gate_pass"]
                        and point_dollars > 0.0
                        and conservative_dollars > 0.0
                    ):
                        disposition = (
                            "RESEARCH_PATTERN_OUT_OF_DOMAIN_REQUIRES_BROAD_VALIDATION"
                        )
                    elif not metrics["ev_gate_pass"]:
                        disposition = "WATCHLIST_EV_MODEL_GATE_FAILED"
                    else:
                        disposition = "REJECTED_NONPOSITIVE_DEVELOPMENT_MODEL"
                    pop_reasons = list(metrics["pop_gate_reasons"])
                    edge_reasons = list(metrics["ev_gate_reasons"])
                else:
                    model_diagnostics = {
                        "publishable_as_profit_estimate": False,
                        "status": "HISTORICAL_MODEL_UNAVAILABLE",
                        "raw_probabilities_not_pop": None,
                        "calibrated_probabilities_not_transferable": None,
                        "predicted_return_on_maximum_loss": None,
                        "residual_bias_adjusted_95_lower_return": None,
                        "predicted_net_dollars": None,
                        "residual_adjusted_lower_net_dollars": None,
                    }
                    disposition = "RESEARCH_STRUCTURE_MODEL_UNAVAILABLE"
                    pop_reasons = ["HISTORICAL_MODEL_UNAVAILABLE"]
                    edge_reasons = ["HISTORICAL_MODEL_UNAVAILABLE"]
                identity = {
                    "ticker": ticker,
                    "family": family,
                    "expiration": expiration,
                    "legs": [str(item["occ_symbol"]) for item in legs],
                }
                candidates.append(
                    {
                        "candidate_id": "pattern-" + _sha256_value(identity)[:24],
                        "ticker": ticker,
                        "direction": direction,
                        "strategy_family": family,
                        "expiration": expiration,
                        "human_legs": _human_legs(legs),
                        "legs": list(legs),
                        "underlying_quote": chain["underlying_quote"],
                        "orats_provider_trade_date": provider_date.isoformat(),
                        "orats_snapshot_provenance": dict(
                            maps["orats_snapshots"][ticker]
                        ),
                        "signal": {
                            "momentum_20": float(history["momentum_20"]),
                            "momentum_60": float(history["momentum_60"]),
                            "realized_volatility_20": float(
                                history["realized_volatility_20"]
                            ),
                            "implied_volatility_20": float(feature_row["smv_vol"]),
                            "iv_to_realized_ratio": float(feature_row["smv_vol"])
                            / max(1e-6, float(feature_row["realized_volatility_20"])),
                            "weeks_to_next_earnings": (
                                None
                                if analytics.get("wksNextErn") is None
                                else float(analytics["wksNextErn"])
                            ),
                            "orats_confidence": (
                                None
                                if analytics.get("confidence") is None
                                else float(analytics["confidence"])
                            ),
                        },
                        "economics": economics,
                        "development_model_diagnostics_not_pop_or_edge": model_diagnostics,
                        "POP_net": {
                            "status": "UNAVAILABLE_OUT_OF_DOMAIN_AND_MODEL_GATE",
                            "point": None,
                            "reason": pop_reasons
                            + ["TEN_ETF_TO_BROAD_EQUITY_DOMAIN_TRANSFER"],
                        },
                        "P_target": {"status": "UNAVAILABLE_OUT_OF_DOMAIN", "point": None},
                        "P_stop": {"status": "UNAVAILABLE_OUT_OF_DOMAIN", "point": None},
                        "P_max_loss": {"status": "UNAVAILABLE_OUT_OF_DOMAIN", "point": None},
                        "net_edge": {
                            "status": "UNAVAILABLE_AS_VALIDATED_EDGE",
                            "point": None,
                            "conservative": None,
                            "reason": edge_reasons
                            + ["TEN_ETF_TO_BROAD_EQUITY_DOMAIN_TRANSFER"],
                        },
                        "evidence_state": "RESEARCH_ONLY_OUT_OF_DOMAIN",
                        "chain_coverage_provenance": chain_coverage_provenance,
                        "disposition": disposition,
                        "quantity": "USER DETERMINED",
                        "manual_ticket_enabled": False,
                        "broker_submission_enabled": False,
                    }
                )
                built += 1
            except (PatternError, ValueError) as exc:
                construction.append(
                    {"ticker": ticker, "family": family, "reason": str(exc)}
                )
        ticker_status[ticker] = (
            "EVALUATED_RESEARCH_ONLY" if built else "DATA_UNAVAILABLE_NO_ELIGIBLE_STRUCTURE"
        )
    # No failed or out-of-domain model may establish the presentation order.
    # A deterministic ticker/family order prevents an unvalidated score from
    # becoming a de facto recommendation.
    candidates.sort(key=lambda item: (item["ticker"], item["strategy_family"]))
    construction.sort(key=lambda item: (item["ticker"], str(item.get("family"))))
    return candidates, construction, ticker_status


def _next_weekday(value: date) -> date:
    result = value + timedelta(days=1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result


def _direction_is_aligned(candidate: Mapping[str, Any]) -> bool:
    direction = str(candidate["direction"])
    momentum_20 = float(candidate["signal"]["momentum_20"])
    momentum_60 = float(candidate["signal"]["momentum_60"])
    if direction == "BULLISH":
        return momentum_20 > 0.0 and momentum_60 > 0.0
    if direction == "BEARISH":
        return momentum_20 < 0.0 and momentum_60 < 0.0
    return False


def _profit_evidence_gate(
    candidate: Mapping[str, Any],
) -> Tuple[bool, Tuple[str, ...]]:
    """Require calibrated POP and positive point/conservative edge for action.

    Candidate retention and action eligibility are deliberately separate.  An
    exact structure remains reviewable when this gate fails, but it cannot be
    converted into ENTER or WAIT instructions.
    """

    reasons: List[str] = []
    evidence_state = str(candidate.get("evidence_state") or "")
    if evidence_state not in {
        "HOLDOUT_PASS",
        "SHADOW_PASS",
        "MANUAL_TICKET_ENABLED",
    }:
        reasons.append("UNTOUCHED_HOLDOUT_NOT_PASSED")

    probability_versions = set()
    for target in ("POP_net", "P_target", "P_stop", "P_max_loss"):
        estimate = candidate.get(target) or {}
        point = estimate.get("point")
        if point is None:
            reasons.extend(str(item) for item in estimate.get("reason", ()))
            reasons.append("%s_UNAVAILABLE" % target.upper())
            continue
        try:
            numeric_point = float(point)
            interval = estimate.get("interval_95")
            if (
                not 0.0 <= numeric_point <= 1.0
                or not isinstance(interval, (list, tuple))
                or len(interval) != 2
                or not 0.0 <= float(interval[0]) <= numeric_point <= float(interval[1]) <= 1.0
            ):
                raise ValueError
            if not math.isclose(
                float(estimate.get("confidence_level")),
                0.95,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError
            sample_size = estimate.get("sample_size")
            if (
                isinstance(sample_size, bool)
                or not isinstance(sample_size, int)
                or sample_size <= 0
            ):
                raise ValueError
            period = estimate.get("calibration_period")
            if (
                not isinstance(period, Mapping)
                or not period.get("start")
                or not period.get("end")
            ):
                raise ValueError
            model_version = str(estimate.get("model_version") or "")
            if not model_version:
                raise ValueError
            probability_versions.add(model_version)
        except (TypeError, ValueError):
            reasons.append("%s_CALIBRATION_PROVENANCE_INCOMPLETE" % target.upper())
    if len(probability_versions) > 1:
        reasons.append("PROBABILITY_MODEL_VERSION_MISMATCH")

    pop = candidate.get("POP_net") or {}
    edge = candidate.get("net_edge") or {}
    if pop.get("point") is None:
        reasons.append("CALIBRATED_POP_UNAVAILABLE")
    if edge.get("point") is None:
        reasons.extend(str(item) for item in edge.get("reason", ()))
        reasons.append("POINT_NET_EDGE_UNAVAILABLE")
    elif float(edge["point"]) <= 0.0:
        reasons.append("POINT_NET_EDGE_NOT_POSITIVE")
    if edge.get("conservative") is None:
        reasons.append("CONSERVATIVE_NET_EDGE_UNAVAILABLE")
    elif float(edge["conservative"]) <= 0.0:
        reasons.append("CONSERVATIVE_NET_EDGE_NOT_POSITIVE")
    return not reasons, tuple(dict.fromkeys(reasons))


def _build_manual_research_actions(
    candidates: Sequence[Mapping[str, Any]],
    provider_date: date,
    policy: Mapping[str, Any],
    confirmed_events: Mapping[str, Mapping[str, Any]],
) -> Tuple[Tuple[Mapping[str, Any], ...], Tuple[Mapping[str, Any], ...]]:
    """Retain exact trade candidates while failing closed on actionability."""

    preference = {
        str(family): index
        for index, family in enumerate(policy["structure_preference"])
    }
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["ticker"]), []).append(candidate)
    actions: List[Mapping[str, Any]] = []
    exclusions: List[Mapping[str, Any]] = []
    next_session = _next_weekday(provider_date).isoformat()
    require_alignment = bool(
        policy["require_20_and_60_session_direction_alignment"]
    )
    event_window = float(policy["earnings_confirmation_window_weeks"])
    for ticker, structures in sorted(grouped.items()):
        aligned = [
            item
            for item in structures
            if not require_alignment or _direction_is_aligned(item)
        ]
        if not aligned:
            exclusions.append(
                {
                    "ticker": ticker,
                    "reason": "20_AND_60_SESSION_DIRECTION_NOT_ALIGNED",
                    "candidate_ids": [str(item["candidate_id"]) for item in structures],
                }
            )
            continue
        chosen = min(
            aligned,
            key=lambda item: (
                preference.get(str(item["strategy_family"]), len(preference)),
                str(item["strategy_family"]),
                str(item["candidate_id"]),
            ),
        )
        economics = chosen["economics"]
        maximum_loss = float(economics["maximum_loss"])
        target_pnl = float(economics["target_pnl"])
        stop_pnl = float(economics["stop_pnl"])
        maximum_profit = economics["maximum_profit"]
        if maximum_profit is not None and target_pnl > float(maximum_profit) + 1e-9:
            exclusions.append(
                {
                    "ticker": ticker,
                    "reason": "FROZEN_TARGET_EXCEEDS_MAXIMUM_PROFIT",
                    "candidate_ids": [str(chosen["candidate_id"])],
                }
            )
            continue
        weeks_to_earnings = chosen["signal"].get("weeks_to_next_earnings")
        confirmed_event = confirmed_events.get(ticker)
        confirmed_event_date: Optional[date] = None
        if confirmed_event is not None:
            try:
                confirmed_event_date = date.fromisoformat(str(confirmed_event["date"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise PatternError("confirmed event date is invalid for %s" % ticker) from exc
        event_inside_holding_window = bool(
            confirmed_event_date is not None
            and provider_date <= confirmed_event_date
            <= provider_date + timedelta(days=int(event_window * 7.0))
        )
        event_record_stale = bool(
            confirmed_event_date is not None and confirmed_event_date < provider_date
        )
        quote_timestamps = [str(chosen["underlying_quote"]["timestamp"])] + [
            str(leg["quote_timestamp"]) for leg in chosen["legs"]
        ]
        underlying_reference = float(chosen["underlying_quote"]["last"])
        reference_limit = float(economics["proposed_limit_debit_per_share"])
        reward_to_risk = economics.get("reward_to_risk")
        total_costs = float(economics["modeled_round_trip_slippage"]) + float(
            economics["commissions_and_fees"]
        )
        # An ORATS week estimate is analytical context, not authoritative
        # event clearance.  A candidate needs a dated confirmed record (which
        # may prove the event is outside the holding window); an absent record
        # always fails closed and is never rendered as "no earnings".
        event_blocked = (
            event_inside_holding_window
            or event_record_stale
            or confirmed_event_date is None
        )
        evidence_gate_passed, evidence_gate_reasons = _profit_evidence_gate(chosen)
        if reward_to_risk is not None and float(reward_to_risk) > 1.0:
            geometry_color = "GREEN"
            geometry_symbol = "🟢"
            geometry_rating = "FAVORABLE_PAYOFF_GEOMETRY"
        else:
            geometry_color = "AMBER"
            geometry_symbol = "🟡"
            geometry_rating = "PAYOFF_GEOMETRY_NEEDS_BETTER_DEBIT"
        if event_blocked:
            action_color = "RED"
            action_symbol = "🔴"
            setup_rating = "BLOCKED"
            action = (
                "AVOID_UNTIL_POST_EARNINGS"
                if event_inside_holding_window
                else "AVOID_STALE_EVENT_RECORD"
                if event_record_stale
                else "AVOID_EVENT_DATE_UNAVAILABLE"
            )
            action_limit: Optional[float] = None
        elif not evidence_gate_passed:
            action_color = "RED"
            action_symbol = "🔴"
            setup_rating = "EVIDENCE_BLOCKED"
            action = "BLOCKED_PROFIT_EVIDENCE_GATE"
            action_limit = None
        elif reward_to_risk is not None and float(reward_to_risk) > 1.0:
            action_color = "GREEN"
            action_symbol = "🟢"
            setup_rating = "ACTIONABLE"
            action = "ENTER_AT_OR_BELOW_LIMIT"
            action_limit = reference_limit
        else:
            action_color = "AMBER"
            action_symbol = "🟡"
            setup_rating = "PRICE_BLOCKED"
            action = "WAIT_FOR_BETTER_DEBIT"
            if maximum_profit is not None and len(chosen["legs"]) == 2:
                width = abs(
                    float(chosen["legs"][0]["strike"])
                    - float(chosen["legs"][1]["strike"])
                )
                one_to_one_limit = width / 2.0 - total_costs / 100.0
                action_limit = max(0.01, math.floor(one_to_one_limit * 100.0) / 100.0)
            else:
                action_limit = reference_limit
        economics_limit = reference_limit if action_limit is None else action_limit
        action_maximum_loss = economics_limit * 100.0 + total_costs
        if maximum_profit is None:
            action_maximum_profit: Optional[float] = None
        else:
            width_dollars = (
                abs(
                    float(chosen["legs"][0]["strike"])
                    - float(chosen["legs"][1]["strike"])
                )
                * 100.0
            )
            action_maximum_profit = width_dollars - action_maximum_loss
        target_fraction = target_pnl / maximum_loss
        stop_fraction = abs(stop_pnl) / maximum_loss
        action_target_pnl = action_maximum_loss * target_fraction
        action_stop_pnl = -action_maximum_loss * stop_fraction
        action_target_value = (action_maximum_loss + action_target_pnl) / 100.0
        action_stop_value = max(
            0.0, (action_maximum_loss + action_stop_pnl) / 100.0
        )
        action_breakeven = (
            float(chosen["legs"][0]["strike"]) + action_maximum_loss / 100.0
            if chosen["direction"] == "BULLISH"
            else float(chosen["legs"][0]["strike"]) - action_maximum_loss / 100.0
        )
        breakeven_move = (
            (action_breakeven / underlying_reference - 1.0) * 100.0
            if chosen["direction"] == "BULLISH"
            else (1.0 - action_breakeven / underlying_reference) * 100.0
        )
        action_economics = {
            "maximum_net_debit_per_share": action_limit,
            "reference_net_debit_per_share": reference_limit,
            "maximum_loss": action_maximum_loss,
            "maximum_profit": action_maximum_profit,
            "reward_to_risk": (
                None
                if action_maximum_profit is None
                else action_maximum_profit / action_maximum_loss
            ),
            "breakeven_at_expiration": action_breakeven,
            "profit_target_net_pnl": action_target_pnl,
            "target_structure_value_per_share": round(action_target_value, 2),
            "stop_net_pnl": action_stop_pnl,
            "stop_structure_value_per_share": round(action_stop_value, 2),
            "time_exit_sessions": int(economics["time_exit_sessions"]),
            "modeled_total_costs": total_costs,
        }
        diagnostic = chosen.get(
            "development_model_diagnostics_not_pop_or_edge", {}
        )
        raw_probabilities = diagnostic.get("raw_probabilities_not_pop") or {}
        diagnostic_pop = raw_probabilities.get("POP_NET")
        other_ids = sorted(
            str(item["candidate_id"])
            for item in structures
            if item["candidate_id"] != chosen["candidate_id"]
        )
        identity = {
            "candidate_id": chosen["candidate_id"],
            "review_date": next_session,
            "policy": "CULTRA_MANUAL_RESEARCH_ACTION_V2",
        }
        actions.append(
            {
                "action_id": "action-" + _sha256_value(identity)[:24],
                "status": (
                    "CONDITIONAL_RESEARCH_TRADE_PLAN"
                    if evidence_gate_passed and not event_blocked
                    else "PRESERVED_EXACT_TRADE_CANDIDATE_NOT_ACTIONABLE"
                ),
                "candidate_list_status": "PRESERVED_EXACT_TRADE_CANDIDATE",
                "candidate_symbol": "🔵",
                "action": action,
                "action_color": action_color,
                "action_symbol": action_symbol,
                "setup_confidence_rating": setup_rating,
                "setup_gate_score": (
                    "4/4"
                    if action_color == "GREEN"
                    else "3/4" if action_color == "AMBER" else "BLOCKED"
                ),
                "payoff_geometry_color": geometry_color,
                "payoff_geometry_symbol": geometry_symbol,
                "payoff_geometry_rating": geometry_rating,
                "ticker": ticker,
                "direction": chosen["direction"],
                "strategy_family": chosen["strategy_family"],
                "source_candidate_id": chosen["candidate_id"],
                "alternative_structure_candidate_ids": other_ids,
                "selection_basis": (
                    "retained because the saved exact-chain input contained the "
                    "ticker, the frozen 20-session directional signal fired, the "
                    "20/60-session directions aligned, and an exact finite-loss "
                    "structure could be built; debit vertical won the frozen "
                    "structure preference"
                ),
                "admission_audit": {
                    "chain_coverage_provenance": chosen.get(
                        "chain_coverage_provenance"
                    ),
                    "upstream_candidate_disposition": chosen.get("disposition"),
                    "directional_signal_passed": True,
                    "momentum_20_and_60_direction_aligned": True,
                    "exact_structure_built": True,
                    "selected_by_structure_preference": str(
                        chosen["strategy_family"]
                    ),
                    "alternative_candidate_ids": other_ids,
                    "profit_evidence_gate_passed": evidence_gate_passed,
                    "profit_evidence_gate_reasons": list(evidence_gate_reasons),
                },
                "human_legs": chosen["human_legs"],
                "legs": chosen["legs"],
                "expiration": chosen["expiration"],
                "signal": chosen["signal"],
                "underlying_reference": chosen["underlying_quote"],
                "directional_move_to_expiration_breakeven_percent": breakeven_move,
                "reference_quote_timestamp": max(quote_timestamps),
                "reference_quote_status": "LATEST_SAVED_COMPLETED_SESSION",
                "next_review_date": (
                    _next_weekday(confirmed_event_date).isoformat()
                    if event_inside_holding_window and confirmed_event_date is not None
                    else next_session
                ),
                "entry": {
                    "decision": action,
                    "maximum_net_debit_per_share": action_limit,
                    "maximum_entry_cash_before_costs": (
                        None if action_limit is None else action_limit * 100.0
                    ),
                    "pricing_owner": "CULTRA_PIPELINE",
                    "user_reprice_instruction": False,
                },
                "exit": {
                    "profit_target_net_pnl": action_target_pnl,
                    "target_structure_value_per_share": round(
                        action_target_value, 2
                    ),
                    "stop_net_pnl": action_stop_pnl,
                    "stop_structure_value_per_share": round(action_stop_value, 2),
                    "time_exit_sessions_after_fill": int(economics["time_exit_sessions"]),
                    "close_before_expiration": True,
                    "recalculate_values_from_actual_fill": True,
                },
                "reference_economics": economics,
                "action_economics": action_economics,
                "event_gate": {
                    "weeks_to_next_earnings_from_saved_orats": weeks_to_earnings,
                    "confirmed_date": (
                        None
                        if confirmed_event_date is None
                        else confirmed_event_date.isoformat()
                    ),
                    "market_timing": (
                        None
                        if confirmed_event is None
                        else confirmed_event.get("market_timing")
                    ),
                    "source": (
                        None if confirmed_event is None else confirmed_event.get("source")
                    ),
                    "source_url": (
                        None
                        if confirmed_event is None
                        else confirmed_event.get("source_url")
                    ),
                    "status": (
                        "CONFIRMED_EARNINGS_INSIDE_HOLDING_WINDOW"
                        if event_inside_holding_window
                        else "STALE_EVENT_RECORD_AVOID"
                        if event_record_stale
                        else "EVENT_DATE_UNVERIFIED_AVOID"
                        if confirmed_event_date is None
                        else "CONFIRMED_EARNINGS_OUTSIDE_HOLDING_WINDOW"
                    ),
                },
                "invalidation": [
                    "live net debit exceeds the maximum entry debit",
                    "a leg is unavailable or fails liquidity rules",
                    "20-session and 60-session direction no longer agree",
                    "confirmed earnings falls inside the holding period",
                    "do not leg into a partial fill; use one complex order",
                ],
                "POP_net": dict(chosen.get("POP_net") or {}),
                "raw_classifier_score_not_pop": {
                    "point": diagnostic_pop,
                    "status": "UNPUBLISHABLE_UNCALIBRATED_OUT_OF_DOMAIN",
                    "used_for_action": False,
                },
                "orats_data_confidence_not_trade_confidence": chosen["signal"].get(
                    "orats_confidence"
                ),
                "validated_net_edge": dict(chosen.get("net_edge") or {}),
                "profit_evidence_gate_passed": evidence_gate_passed,
                "profit_evidence_gate_reasons": list(evidence_gate_reasons),
                "failed_model_outputs_used": False,
                "market_open_quote_refresh_owner": "CULTRA_PIPELINE",
                "user_reprice_instruction": False,
                "quantity": "USER DETERMINED",
                "manual_ticket_enabled": False,
                "broker_submission_enabled": False,
            }
        )
    actions.sort(key=lambda item: str(item["ticker"]))
    exclusions.sort(key=lambda item: str(item["ticker"]))
    return tuple(actions), tuple(exclusions)


def _universe_disposition(
    maps: Mapping[str, Any], ticker_status: Mapping[str, str]
) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    for ticker in sorted(maps["universe"]):
        source = maps["universe"][ticker]
        if ticker in maps["legacy_local"]:
            disposition = "NOT_FULLY_EVALUATED_LEGACY_LOCAL_SCREEN"
            reasons = list(maps["legacy_local"][ticker].get("reasons", ()))
        elif ticker in maps["orats_budget"]:
            disposition = "NOT_FULLY_EVALUATED_ORATS_BUDGET"
            reasons = [str(maps["orats_budget"][ticker].get("reason", "ORATS capacity"))]
        elif ticker in maps["screen_unavailable"]:
            disposition = "DATA_UNAVAILABLE_SCHWAB_SCREEN"
            reasons = [str(maps["screen_unavailable"][ticker].get("reason", "screen unavailable"))]
        elif ticker not in maps["orats"]:
            disposition = "DATA_UNAVAILABLE_ORATS_CORE"
            reasons = ["admitted symbol has no saved ORATS Core row"]
        elif ticker not in maps["chains"]:
            disposition = "NOT_FULLY_EVALUATED_CHAIN_NOT_COLLECTED"
            reasons = [
                "saved exact-chain artifact covers %d of %d expected selected symbols; selection manifest %s"
                % (
                    len(maps["chains"]),
                    (
                        len(maps["selection"])
                        if maps.get("selection") is not None
                        else len(maps["orats"])
                    ),
                    "available" if maps.get("selection") is not None else "missing",
                )
            ]
        else:
            disposition = ticker_status.get(ticker, "DATA_UNAVAILABLE_UNKNOWN_CHAIN_RESULT")
            reasons = []
        rows.append(
            {
                "ticker": ticker,
                "name": source.get("name"),
                "disposition": disposition,
                "reasons": reasons,
                "current_quote_timestamp": source.get("quote_timestamp"),
            }
        )
    if len(rows) != len(maps["universe"]) or len({item["ticker"] for item in rows}) != len(rows):
        raise PatternError("universe disposition is not one row per source ticker")
    return tuple(rows)


def _counts_by(values: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, int]:
    result: Dict[str, int] = {}
    for item in values:
        value = str(item[key])
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _board(
    *,
    run_id: str,
    as_of: date,
    provider_date: date,
    universe: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    action_exclusions: Sequence[Mapping[str, Any]],
    construction: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> str:
    counts = _counts_by(universe, "disposition")
    retained_ids = {str(item["source_candidate_id"]) for item in actions}
    enter_count = sum(item["action"] == "ENTER_AT_OR_BELOW_LIMIT" for item in actions)
    wait_count = sum(item["action"] == "WAIT_FOR_BETTER_DEBIT" for item in actions)
    evidence_blocked_count = sum(
        not bool(item["profit_evidence_gate_passed"]) for item in actions
    )
    event_blocked_count = sum(
        str(item["action"]).startswith("AVOID_") for item in actions
    )
    profit_evidence_passed_count = sum(
        bool(item["profit_evidence_gate_passed"]) for item in actions
    )
    exact_chain_count = sum(
        item["disposition"]
        in {
            "EVALUATED_RESEARCH_ONLY",
            "NO_FROZEN_DIRECTIONAL_PATTERN_SIGNAL",
            "DATA_UNAVAILABLE_NO_ELIGIBLE_STRUCTURE",
        }
        for item in universe
    )
    chain_not_collected = counts.get("NOT_FULLY_EVALUATED_CHAIN_NOT_COLLECTED", 0)
    lines = [
        "# Cultra V6 Candidate Audit and Action Board",
        "",
        "- Exact trade candidates retained: **🔵 %d**" % len(actions),
        "- Actionable entries: **🟢 ENTER %d · 🟡 PRICE-WAIT %d**"
        % (enter_count, wait_count),
        "- Independent blockers (overlap allowed): **🔴 profit evidence %d · event records/timing %d**"
        % (evidence_blocked_count, event_blocked_count),
        "- Profit-evidence confidence: **%s %s**"
        % (
            "🟢" if readiness["profit_confidence"] != "UNPROVEN" else "🔴",
            readiness["profit_confidence"],
        ),
        "- Complete calibrated POP/edge bundles: **%d of %d retained candidates**"
        % (profit_evidence_passed_count, len(actions)),
        "- Validated manual tickets: **0**",
        "- Source universe: **%d symbols; every symbol reconciled**" % len(universe),
        "- Saved exact-chain coverage: **%d symbols**; another **%d** Core/history names were not collected"
        % (exact_chain_count, chain_not_collected),
        "- Price reference: **%s completed session**"
        % provider_date.isoformat(),
        "- Quantity: **USER DETERMINED**; broker submission: **disabled**",
        "",
        "🔵 means the exact trade is retained for audit. 🟢/🟡 in the payoff column describes geometry only. It is never a profit-confidence rating. 🔴 in the action column means no entry instruction exists.",
        "",
        "## Why these trades are on the list",
        "",
        "The list did not come from passing POP or edge. The saved exact-chain input is partial relative to the fingerprinted all-resolved selection. From that snapshot, Cultra admitted structures when the 20-session momentum threshold fired, required 20/60-session direction agreement, built liquid 28–84 DTE exact legs, and then chose one structure per ticker using the frozen preference for debit verticals. The previous action builder ignored each candidate's upstream `WATCHLIST_EV_MODEL_GATE_FAILED` disposition and treated payoff ratio above 1.0 as an ENTER signal. This run preserves the rows and blocks that promotion bypass.",
        "",
        "## Retained exact trade candidates",
        "",
    ]
    if actions:
        ranked_actions = sorted(actions, key=lambda item: str(item["ticker"]))
        lines.extend(
            [
                "| Candidate | Ticker | Exact trade | Why admitted | Payoff geometry | POP / edge | Actionability |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for item in ranked_actions:
            economics = item["action_economics"]
            if item["action"] == "ENTER_AT_OR_BELOW_LIMIT":
                decision = "🟢 ENTER ≤ $%.2f" % float(
                    item["entry"]["maximum_net_debit_per_share"]
                )
            elif item["action"] == "WAIT_FOR_BETTER_DEBIT":
                decision = "🟡 PRICE-WAIT ≤ $%.2f" % float(
                    item["entry"]["maximum_net_debit_per_share"]
                )
            elif item["action"] == "BLOCKED_PROFIT_EVIDENCE_GATE":
                decision = "🔴 NOT ACTIONABLE — POP/edge gate"
            elif item["action"] == "AVOID_UNTIL_POST_EARNINGS":
                decision = "🔴 AVOID through %s" % item["event_gate"]["confirmed_date"]
            elif item["action"] == "AVOID_STALE_EVENT_RECORD":
                decision = "🔴 AVOID — event record stale"
            else:
                decision = "🔴 AVOID — event date unavailable"
            signal = item["signal"]
            admission = "%+.1f%% / %+.1f%% 20d/60d; %s preference" % (
                float(signal["momentum_20"]) * 100.0,
                float(signal["momentum_60"]) * 100.0,
                item["strategy_family"],
            )
            reward_to_risk = economics.get("reward_to_risk")
            geometry = "%s %s" % (
                item["payoff_geometry_symbol"],
                (
                    "unlimited"
                    if reward_to_risk is None
                    else "%.2f:1 max-profit/max-loss" % float(reward_to_risk)
                ),
            )
            pop = item["POP_net"]
            edge = item["validated_net_edge"]
            if pop.get("point") is None or edge.get("point") is None:
                confidence_display = "unavailable / unavailable"
            else:
                interval = pop["interval_95"]
                confidence_display = (
                    "POP %.1f%% [%.1f%%, %.1f%%] / EV $%.2f (cons. $%.2f)"
                    % (
                        float(pop["point"]) * 100.0,
                        float(interval[0]) * 100.0,
                        float(interval[1]) * 100.0,
                        float(edge["point"]),
                        float(edge["conservative"]),
                    )
                )
            lines.append(
                "| 🔵 | **%s** | %s | %s | %s | %s | **%s** |"
                % (
                    item["ticker"],
                    item["human_legs"],
                    admission,
                    geometry,
                    confidence_display,
                    decision,
                )
            )
        lines.extend(["", "## Candidate details and reference economics", ""])
        for rank, item in enumerate(ranked_actions, 1):
            economics = item["action_economics"]
            signal = item["signal"]
            maximum_profit_text = (
                "unlimited"
                if economics["maximum_profit"] is None
                else "$%.2f" % float(economics["maximum_profit"])
            )
            if item["event_gate"]["status"] == "CONFIRMED_EARNINGS_INSIDE_HOLDING_WINDOW":
                event_instruction = (
                    "**%s %s** ([company IR](%s)); avoid through the announcement and reassess **%s**."
                    % (
                        item["event_gate"]["confirmed_date"],
                        item["event_gate"]["market_timing"],
                        item["event_gate"]["source_url"],
                        item["next_review_date"],
                    )
                )
            elif item["event_gate"]["status"] == "STALE_EVENT_RECORD_AVOID":
                event_instruction = "**STALE RECORD — AVOID**. The saved earnings date is before the quote session and does not establish the next event."
            elif item["event_gate"]["status"] == "EVENT_DATE_UNVERIFIED_AVOID":
                event_instruction = "**DATE UNVERIFIED — AVOID**. Cultra has no authoritative earnings/no-event record; the ORATS week estimate is context only and cannot clear this gate."
            else:
                event_instruction = (
                    "Confirmed earnings date **%s** is outside the 20-session holding window."
                    % item["event_gate"]["confirmed_date"]
                )
            if item["action"] == "ENTER_AT_OR_BELOW_LIMIT":
                entry_instruction = "**🟢 ENTER** only at a net debit of **$%.2f or less**." % float(
                    item["entry"]["maximum_net_debit_per_share"]
                )
            elif item["action"] == "WAIT_FOR_BETTER_DEBIT":
                entry_instruction = (
                    "**🟡 WAIT**. The saved $%.2f debit has sub-1:1 payoff; enter only if the net debit improves to **$%.2f or less**."
                    % (
                        float(economics["reference_net_debit_per_share"]),
                        float(item["entry"]["maximum_net_debit_per_share"]),
                    )
                )
            elif item["action"] == "BLOCKED_PROFIT_EVIDENCE_GATE":
                entry_instruction = (
                    "**🔴 NOT ACTIONABLE**. The exact trade stays on the list, but no entry is permitted because calibrated POP and positive point/conservative net edge are unavailable."
                )
            else:
                entry_instruction = "**🔴 AVOID**. No entry before the event gate clears."
            pop = item["POP_net"]
            edge = item["validated_net_edge"]
            if pop.get("point") is None or edge.get("point") is None:
                profit_confidence_text = (
                    "**unavailable**. POP model reasons: `%s`. Edge model reasons: `%s`. Complete action-gate reasons: `%s`. No raw classifier score is published as POP."
                    % (
                        "; ".join(pop.get("reason", ())),
                        "; ".join(edge.get("reason", ()))
                        if edge.get("reason")
                        else "; ".join(item["profit_evidence_gate_reasons"]),
                        "; ".join(item["profit_evidence_gate_reasons"]),
                    )
                )
            else:
                bundle = []
                for target in ("POP_net", "P_target", "P_stop", "P_max_loss"):
                    estimate = item[target]
                    interval = estimate["interval_95"]
                    bundle.append(
                        "%s %.1f%% [%.1f%%, %.1f%%]"
                        % (
                            target,
                            float(estimate["point"]) * 100.0,
                            float(interval[0]) * 100.0,
                            float(interval[1]) * 100.0,
                        )
                    )
                profit_confidence_text = (
                    "**available after untouched holdout** — %s; net EV **$%.2f**, conservative net EV **$%.2f**; n=%d, model `%s`, calibration %s through %s."
                    % (
                        "; ".join(bundle),
                        float(edge["point"]),
                        float(edge["conservative"]),
                        int(pop["sample_size"]),
                        pop["model_version"],
                        pop["calibration_period"]["start"],
                        pop["calibration_period"]["end"],
                    )
                )
            lines.extend(
                [
                    "### %d. %s — %s %s — 🔵 retained candidate"
                    % (
                        rank,
                        item["ticker"],
                        str(item["direction"]).lower(),
                        str(item["strategy_family"]).lower().replace("_", " "),
                    ),
                    "",
                    "- **Trade:** %s." % item["human_legs"],
                    "- **Actionability:** %s Saved underlying reference: **$%.2f**."
                    % (entry_instruction, float(item["underlying_reference"]["last"])),
                    "- **Reference target scenario:** structure value **$%.2f**, approximately **+$%.2f net** under the modeled costs."
                    % (
                        float(item["exit"]["target_structure_value_per_share"]),
                        float(item["exit"]["profit_target_net_pnl"]),
                    ),
                    "- **Reference stop scenario:** structure value **$%.2f**, approximately **-$%.2f net**; reference time exit **%d sessions**."
                    % (
                        float(item["exit"]["stop_structure_value_per_share"]),
                        abs(float(item["exit"]["stop_net_pnl"])),
                        int(item["exit"]["time_exit_sessions_after_fill"]),
                    ),
                    "- **Payoff:** maximum profit **%s**; maximum loss **$%.2f**; expiration breakeven **$%.2f** (%+.2f%% directional move from the saved underlying)."
                    % (
                        maximum_profit_text,
                        float(economics["maximum_loss"]),
                        float(economics["breakeven_at_expiration"]),
                        float(item["directional_move_to_expiration_breakeven_percent"]),
                    ),
                    "- **Pattern:** 20-session momentum **%+.1f%%**, 60-session momentum **%+.1f%%**, IV / realized volatility **%.2f**."
                    % (
                        float(signal["momentum_20"]) * 100.0,
                        float(signal["momentum_60"]) * 100.0,
                        float(signal["iv_to_realized_ratio"]),
                    ),
                    "- **Earnings:** %s" % event_instruction,
                    "- **Why listed:** saved partial-chain membership; directional threshold; 20/60 alignment; exact liquid legs; frozen **%s** structure preference. Upstream disposition: `%s`."
                    % (
                        item["strategy_family"],
                        item["admission_audit"]["upstream_candidate_disposition"],
                    ),
                    "- **Profit confidence:** %s" % profit_confidence_text,
                    "- **Data quality:** saved ORATS confidence **%s/100**; this is provider data confidence, not trade confidence. Quantity: **USER DETERMINED**."
                    % item["orats_data_confidence_not_trade_confidence"],
                    "",
                ]
            )
    else:
        lines.append("None from the saved exact-chain coverage.")
    lines.extend(
        [
            "",
            "Targets, stops, maximum profit, and maximum loss above are reference scenario economics, not entry instructions. Maximum profit is payoff potential, not expected profit.",
            "",
        ]
    )
    if action_exclusions:
        lines.extend(["### Exact-chain action exclusions", ""])
        for item in action_exclusions:
            lines.append("- **%s** — `%s`" % (item["ticker"], item["reason"]))
        lines.append("")
    lines.extend(
        [
            "## Full-universe reconciliation",
            "",
            "| Disposition | Symbols |",
            "|---|---:|",
        ]
    )
    for name, count in counts.items():
        lines.append("| `%s` | %d |" % (name, count))
    lines.extend(
        [
            "| **TOTAL** | **%d** |" % len(universe),
            "",
            "The complete one-row-per-symbol disposition is in `universe_disposition.jsonl`; no name is silently omitted.",
            "",
            "## Leakage-safe development model",
            "",
            "The current learner uses ETF-only exact-leg history. Nested chronological logistic-versus-isotonic calibration now runs offline, but the calibrated family POP gates still fail and the outputs remain out of domain for the retained equities. Raw classifier scores and non-transferable calibrated diagnostics stay machine-only and are not published as POP.",
            "",
            "| Family | Hist. rows | OOF rows | POP Brier / base | ECE | EV MSE / base | Selected exposure clusters | Selected return 95% LCB | Gates |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for family in sorted(evidence):
        result = evidence[family]
        metrics = result.get("metrics")
        if not metrics:
            lines.append("| %s | — | — | — | — | — | — | — | no model |" % family)
            continue
        pop = metrics["probabilities"]["POP_NET"]
        returns = metrics["return_model"]
        gates = list(metrics["pop_gate_reasons"]) + list(metrics["ev_gate_reasons"])
        lines.append(
            "| %s | %d | %d | %.4f / %.4f | %.4f | %.4f / %.4f | %d | %s | %s |"
            % (
                family,
                int(result["historical_rows"]),
                int(result["oof_prediction_count"]),
                float(pop["oof_brier"]) if pop["oof_brier"] is not None else math.nan,
                float(pop["base_rate_brier"])
                if pop["base_rate_brier"] is not None
                else math.nan,
                float(pop["expected_calibration_error"])
                if pop["expected_calibration_error"] is not None
                else math.nan,
                float(returns["oof_mse"]),
                float(returns["base_mean_mse"]),
                int(returns["selected_independent_exposure_clusters"]),
                (
                    "%.4f" % float(returns["selected_oof_95_lower_return_on_risk"])
                    if returns["selected_oof_95_lower_return_on_risk"] is not None
                    else "—"
                ),
                "; ".join(gates) or "development gates pass; new holdout still required",
            )
        )
    lines.extend(["", render_readiness_markdown(readiness), ""])
    lines.extend(
        [
            "",
            "## Current exact structures",
            "",
            "Every constructed alternative remains visible. `RETAINED PRIMARY CANDIDATE` identifies the one-per-ticker structure selected by the frozen preference; it does not mean validated edge. No fixed output count is applied.",
            "",
        ]
    )
    if candidates:
        lines.extend(
            [
                "| Role | Ticker | Family | Exact structure | Debit | Max loss | Max profit | IV / realized | POP | Edge | Disposition |",
                "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for item in candidates:
            economics = item["economics"]
            maximum_profit = economics["maximum_profit"]
            lines.append(
                "| %s | %s | %s | %s | $%.2f | $%.2f | %s | %.2f | unavailable | unavailable | `%s` |"
                % (
                    (
                        "**RETAINED PRIMARY CANDIDATE**"
                        if str(item["candidate_id"]) in retained_ids
                        else "ALTERNATIVE"
                    ),
                    item["ticker"],
                    item["strategy_family"],
                    item["human_legs"],
                    float(economics["proposed_limit_debit_per_share"]),
                    float(economics["maximum_loss"]),
                    "unlimited" if maximum_profit is None else "$%.2f" % float(maximum_profit),
                    float(item["signal"]["iv_to_realized_ratio"]),
                    item["disposition"],
                )
            )
    else:
        lines.append("None from the saved exact-chain coverage.")
    lines.extend(["", "## Construction and data failures", ""])
    if construction:
        for item in construction:
            lines.append(
                "- **%s %s** — %s"
                % (item["ticker"], item.get("family") or "", item["reason"])
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Universe symbols by disposition", ""])
    grouped: Dict[str, List[str]] = {}
    for item in universe:
        grouped.setdefault(str(item["disposition"]), []).append(str(item["ticker"]))
    for disposition, symbols in sorted(grouped.items()):
        lines.extend(
            [
                "### `%s` (%d)" % (disposition, len(symbols)),
                "",
                ", ".join(symbols),
                "",
            ]
        )
    lines.extend(
        [
            "## Run identity",
            "",
            "- Run ID: `%s`" % run_id,
            "- Requested as-of: `%s`" % as_of.isoformat(),
            "- Network requests made by this run: **0**",
            "",
        ]
    )
    return "\n".join(lines)


def build_pattern_run(
    *,
    as_of: date,
    run_id: str,
    broad_screen: Path = DEFAULT_SCREEN,
    history_screen: Path = DEFAULT_HISTORY,
    orats_enrichment: Path = DEFAULT_ORATS,
    finalist_chains: Path = DEFAULT_CHAINS,
    selection_manifest: Optional[Path] = None,
    database: Path = DEFAULT_CHAIN_DB,
    confirmed_events_path: Path = EVENTS_PATH,
    output_root: Path = OUT_ROOT,
) -> Mapping[str, Any]:
    """Build the corrected pipeline entirely from saved Cultra inputs."""

    config = _load(CONFIG_PATH)
    if config.get("schema") != "cultra.pattern-model.v2":
        raise PatternError("pattern configuration schema is invalid")
    primary_paths = tuple(
        Path(item).expanduser().resolve()
        for item in (
            broad_screen,
            history_screen,
            orats_enrichment,
            finalist_chains,
            database,
            confirmed_events_path,
        )
    )
    selection_path = (
        None
        if selection_manifest is None
        else Path(selection_manifest).expanduser().resolve()
    )
    paths = primary_paths + (() if selection_path is None else (selection_path,))
    for path in paths:
        if not path.exists() or not path.is_file():
            raise PatternError("required saved input does not exist: %s" % path.name)
    screen = _load(primary_paths[0])
    history = _load(primary_paths[1])
    orats = _load(primary_paths[2])
    chains = _load(primary_paths[3])
    event_artifact = _load(primary_paths[5])
    selection = None if selection_path is None else _load(selection_path)
    if selection is not None:
        fingerprints = selection.get("input_fingerprints", {})
        if fingerprints.get("history_screen_sha256") != _sha256(primary_paths[1]):
            raise PatternError("chain selection history fingerprint does not match")
        if fingerprints.get("orats_enrichment_sha256") != _sha256(primary_paths[2]):
            raise PatternError("chain selection ORATS fingerprint does not match")
    if event_artifact.get("schema") != "cultra.confirmed-events.v1":
        raise PatternError("confirmed event artifact schema is invalid")
    event_rows = tuple(event_artifact.get("events", ()))
    event_map = {str(item.get("ticker")): item for item in event_rows}
    if len(event_rows) != len(event_map) or any(
        item.get("event") != "EARNINGS"
        or item.get("confirmed") is not True
        or not item.get("source_url")
        for item in event_rows
    ):
        raise PatternError("confirmed event artifact is duplicated or incomplete")
    maps = _input_maps(screen, history, orats, chains, selection)
    provider_date = _provider_trade_date(maps["orats"])
    development_rows = _historical_rows(primary_paths[4])
    sessions = load_recent_sessions()
    models = build_walk_forward_models(development_rows, sessions, config)
    candidates, construction, ticker_status = _build_current_candidates(
        maps, models, config, provider_date
    )
    actions, action_exclusions = _build_manual_research_actions(
        candidates,
        provider_date,
        config["manual_research_action_policy"],
        event_map,
    )
    universe = _universe_disposition(maps, ticker_status)
    public_evidence = public_model_evidence(models)
    readiness = assess_production_readiness(
        screen=screen,
        history=history,
        orats=orats,
        chains=chains,
        selection=selection,
        config=config,
        models=models,
        candidates=candidates,
        confirmed_events=event_map,
        database=primary_paths[4],
    )
    disposition_counts = _counts_by(universe, "disposition")
    candidate_counts = _counts_by(candidates, "disposition") if candidates else {}
    summary = {
        "schema": PATTERN_SCHEMA,
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_verdict": "CANDIDATES_RETAINED_PROMOTION_BYPASS_BLOCKED_NO_VALIDATED_EDGE",
        "profit_confidence": "UNPROVEN",
        "prior_holdout_status": "INVALIDATED_EXPOSED_AS_DEVELOPMENT_DATA",
        "development_model_version": config["version"],
        "historical_domain": config["historical_domain"],
        "provider_trade_date": provider_date.isoformat(),
        "coverage": {
            "source_universe": len(universe),
            "orats_core_saved": len(maps["orats"]),
            "exact_chain_saved": len(maps["chains"]),
            "every_symbol_reconciled": len(universe) == len(maps["universe"]),
            "dispositions": disposition_counts,
        },
        "historical_development_rows": len(development_rows),
        "current_candidate_counts": candidate_counts,
        "current_candidates": len(candidates),
        "retained_exact_trade_candidates": len(actions),
        "conditional_manual_research_actions": len(actions),
        "actionable_entries": sum(
            item["action"] == "ENTER_AT_OR_BELOW_LIMIT" for item in actions
        ),
        "profit_evidence_blocked_candidates": sum(
            item["action"] == "BLOCKED_PROFIT_EVIDENCE_GATE" for item in actions
        ),
        "research_action_counts": {
            color: sum(item["action_color"] == color for item in actions)
            for color in ("GREEN", "AMBER", "RED")
        },
        "model_pop_reliability": "UNAVAILABLE_CALIBRATED_GATES_FAILED_AND_OUT_OF_DOMAIN",
        "candidate_chain_input": {
            "saved_symbols": len(maps["chains"]),
            "expected_selected_symbols": (
                len(maps["selection"])
                if maps.get("selection") is not None
                else len(maps["orats"])
            ),
            "core_resolved_symbols": len(maps["orats"]),
            "selection_manifest_available": maps.get("selection") is not None,
            "complete": (
                maps.get("selection") is not None
                and set(maps["chains"]) == set(maps["selection"])
            ),
        },
        "action_exclusions": len(action_exclusions),
        "construction_failures": len(construction),
        "manual_tickets": [],
        "manual_ticket_enabled": False,
        "broker_submission_enabled": False,
        "quantity": "USER DETERMINED",
        "network_attempted": False,
        "production_readiness": {
            "status": readiness["status"],
            "blocker_count": readiness["blocker_count"],
            "historically_validated_action_enabled": readiness[
                "historically_validated_action_enabled"
            ],
        },
    }
    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise PatternError("pattern output root must remain inside Cultra/out") from exc
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    artifacts = []
    artifacts.append(_private_json(run_dir / "pattern_run.json", summary))
    artifacts.append(_private_json(run_dir / "model_evidence.json", public_evidence))
    artifacts.append(_private_json(run_dir / "production_readiness.json", readiness))
    artifacts.append(
        _private_json(
            run_dir / "current_candidates.json",
            {
                "schema": "cultra.current-pattern-candidates.v6",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "conditional_manual_research_action_count": len(actions),
                "conditional_manual_research_actions": actions,
                "action_exclusions": action_exclusions,
                "construction_failures": construction,
                "manual_ticket_enabled": False,
                "broker_submission_enabled": False,
            },
        )
    )
    artifacts.append(
        _private_jsonl(run_dir / "universe_disposition.jsonl", universe)
    )
    artifacts.append(
        _private_jsonl(run_dir / "development_outcomes.jsonl", development_rows)
    )
    artifacts.append(
        _private_write(
            run_dir / "BOARD.md",
            (
                _board(
                    run_id=run_id,
                    as_of=as_of,
                    provider_date=provider_date,
                    universe=universe,
                    candidates=candidates,
                    actions=actions,
                    action_exclusions=action_exclusions,
                    construction=construction,
                    evidence=public_evidence,
                    readiness=readiness,
                )
                + "\n"
            ).encode("utf-8"),
        )
    )
    source = _source_fingerprint()
    manifest = {
        "schema": "cultra.pattern-manifest.v6",
        "run_id": run_id,
        "model_version": config["version"],
        "config_sha256": _sha256(CONFIG_PATH),
        "source_tree_sha256": source["tree_sha256"],
        "source_files": source["files"],
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths
        ],
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifacts
        ],
        "reconciliation": {
            "universe_count": len(universe),
            "universe_disposition_count": sum(disposition_counts.values()),
            "candidate_count": len(candidates),
            "conditional_manual_research_action_count": len(actions),
            "manual_ticket_count": 0,
            "production_readiness_blocker_count": int(readiness["blocker_count"]),
        },
        "network_attempted": False,
        "broker_submission_enabled": False,
    }
    _private_json(run_dir / "manifest.json", manifest)
    return summary


def verify_pattern_run(run_dir: Path) -> Tuple[str, ...]:
    root = Path(run_dir).expanduser().resolve()
    try:
        root.relative_to(OUT_ROOT.resolve())
    except ValueError:
        return ("pattern run leaves Cultra/out",)
    errors: List[str] = []
    try:
        manifest = _load(root / "manifest.json")
        summary = _load(root / "pattern_run.json")
        candidates_artifact = _load(root / "current_candidates.json")
        readiness = _load(root / "production_readiness.json")
    except PatternError as exc:
        return (str(exc),)
    expected_files = {
        "pattern_run.json",
        "model_evidence.json",
        "production_readiness.json",
        "current_candidates.json",
        "universe_disposition.jsonl",
        "development_outcomes.jsonl",
        "BOARD.md",
    }
    listed = {str(item.get("path")) for item in manifest.get("artifacts", ())}
    if listed != expected_files:
        errors.append("manifest artifact set is incomplete or unexpected")
    for item in manifest.get("artifacts", ()):
        path = root / str(item.get("path", ""))
        try:
            if path.stat().st_size != int(item["bytes"]):
                errors.append("artifact byte count changed: %s" % path.name)
            if _sha256(path) != str(item["sha256"]):
                errors.append("artifact sha256 changed: %s" % path.name)
        except (OSError, TypeError, ValueError):
            errors.append("artifact is unavailable: %s" % path.name)
    for item in manifest.get("inputs", ()):
        path = Path(str(item.get("path", "")))
        try:
            if path.stat().st_size != int(item["bytes"]):
                errors.append("input byte count changed: %s" % path.name)
            if _sha256(path) != str(item["sha256"]):
                errors.append("input sha256 changed: %s" % path.name)
        except (OSError, TypeError, ValueError):
            errors.append("input is unavailable: %s" % path.name)
    if manifest.get("config_sha256") != _sha256(CONFIG_PATH):
        errors.append("pattern configuration fingerprint changed")
    if manifest.get("source_tree_sha256") != _source_fingerprint()["tree_sha256"]:
        errors.append("pattern source tree fingerprint changed")
    try:
        universe_rows = tuple(
            json.loads(line)
            for line in (root / "universe_disposition.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("universe disposition JSONL is invalid")
        universe_rows = ()
    if len(universe_rows) != int(summary.get("coverage", {}).get("source_universe", -1)):
        errors.append("universe disposition count does not reconcile")
    if len({str(item.get("ticker")) for item in universe_rows}) != len(universe_rows):
        errors.append("universe disposition contains duplicate tickers")
    candidates = tuple(candidates_artifact.get("candidates", ()))
    actions = tuple(
        candidates_artifact.get("conditional_manual_research_actions", ())
    )
    action_exclusions = tuple(candidates_artifact.get("action_exclusions", ()))
    if len(candidates) != int(summary.get("current_candidates", -1)):
        errors.append("current candidate count does not reconcile")
    if len({str(item.get("candidate_id")) for item in candidates}) != len(candidates):
        errors.append("current candidate ids are duplicated")
    if len(actions) != int(
        summary.get("conditional_manual_research_actions", -1)
    ):
        errors.append("conditional manual research action count does not reconcile")
    if len({str(item.get("action_id")) for item in actions}) != len(actions):
        errors.append("conditional manual research action ids are duplicated")
    if len({str(item.get("ticker")) for item in actions}) != len(actions):
        errors.append("more than one primary action exists for a ticker")
    board_text = ""
    try:
        board_text = (root / "BOARD.md").read_text(encoding="utf-8")
        required_markers = [
            "Why these trades are on the list",
            "Exact trade candidates retained",
            "Complete calibrated POP/edge bundles:",
            "🔵",
        ]
        for color, symbol in (("GREEN", "🟢"), ("AMBER", "🟡"), ("RED", "🔴")):
            if any(item.get("action_color") == color for item in actions):
                required_markers.append(symbol)
        for marker in required_markers:
            if marker not in board_text:
                errors.append("decision board is missing %s" % marker)
        if "Model POP*" in board_text:
            errors.append("decision board publishes an uncalibrated classifier as POP")
        if "REPRICE" in board_text.upper() or "CHECK EARNINGS" in board_text.upper():
            errors.append("decision board delegates a vague price or event check")
    except (OSError, UnicodeError):
        errors.append("decision board cannot be read")
    for item in candidates:
        label = "%s:%s" % (item.get("ticker"), item.get("strategy_family"))
        legs = tuple(item.get("legs", ()))
        if not legs:
            errors.append("%s has no exact legs" % label)
            continue
        try:
            for leg in legs:
                parse_occ_symbol(str(leg["occ_symbol"]))
            maximum_loss = float(item["economics"]["maximum_loss"])
            if not math.isfinite(maximum_loss) or maximum_loss <= 0.0:
                errors.append("%s maximum loss is not finite and positive" % label)
            if item["POP_net"].get("point") is not None:
                errors.append("%s publishes an out-of-domain POP point" % label)
            if item["net_edge"].get("point") is not None:
                errors.append("%s publishes an out-of-domain edge point" % label)
            if item.get("development_model_diagnostics_not_pop_or_edge", {}).get(
                "publishable_as_profit_estimate"
            ) is not False:
                errors.append("%s exposes diagnostics as a profit estimate" % label)
            if item.get("manual_ticket_enabled") is not False:
                errors.append("%s enables a manual ticket" % label)
            if item.get("broker_submission_enabled") is not False:
                errors.append("%s enables broker submission" % label)
        except (KeyError, TypeError, ValueError):
            errors.append("%s has malformed economics or probability fields" % label)
    candidate_map = {str(item.get("candidate_id")): item for item in candidates}
    for item in actions:
        label = "action:%s" % item.get("ticker")
        try:
            candidate = candidate_map[str(item["source_candidate_id"])]
            reference_economics = item["reference_economics"]
            action_economics = item["action_economics"]
            if item["legs"] != candidate["legs"]:
                errors.append("%s exact legs differ from its source candidate" % label)
            if reference_economics != candidate["economics"]:
                errors.append("%s reference economics differ from its source candidate" % label)
            maximum_loss = float(action_economics["maximum_loss"])
            target_pnl = float(item["exit"]["profit_target_net_pnl"])
            stop_pnl = float(item["exit"]["stop_net_pnl"])
            entry_limit = item["entry"]["maximum_net_debit_per_share"]
            economics_limit = (
                float(reference_economics["proposed_limit_debit_per_share"])
                if entry_limit is None
                else float(entry_limit)
            )
            total_costs = float(reference_economics["modeled_round_trip_slippage"]) + float(
                reference_economics["commissions_and_fees"]
            )
            if not math.isclose(
                maximum_loss,
                economics_limit * 100.0 + total_costs,
                abs_tol=1e-9,
            ):
                errors.append("%s action maximum loss is not reproducible" % label)
            expected_target_value = round((maximum_loss + target_pnl) / 100.0, 2)
            expected_stop_value = round(
                max(0.0, (maximum_loss + stop_pnl) / 100.0), 2
            )
            if not math.isclose(
                float(item["exit"]["target_structure_value_per_share"]),
                expected_target_value,
                abs_tol=1e-9,
            ):
                errors.append("%s target value is not reproducible" % label)
            if not math.isclose(
                float(item["exit"]["stop_structure_value_per_share"]),
                expected_stop_value,
                abs_tol=1e-9,
            ):
                errors.append("%s stop value is not reproducible" % label)
            expected_gate, expected_gate_reasons = _profit_evidence_gate(candidate)
            if item.get("profit_evidence_gate_passed") is not expected_gate:
                errors.append("%s profit evidence gate does not reproduce" % label)
            if tuple(item.get("profit_evidence_gate_reasons", ())) != tuple(
                expected_gate_reasons
            ):
                errors.append("%s profit evidence reasons do not reproduce" % label)
            if not expected_gate and item.get("action") in {
                "ENTER_AT_OR_BELOW_LIMIT",
                "WAIT_FOR_BETTER_DEBIT",
            }:
                errors.append("%s bypasses failed profit evidence" % label)
            if (
                not expected_gate
                and not str(item.get("action", "")).startswith("AVOID_")
                and item.get("action") != "BLOCKED_PROFIT_EVIDENCE_GATE"
            ):
                errors.append("%s has an invalid evidence-blocked action" % label)
            if item.get("candidate_list_status") != "PRESERVED_EXACT_TRADE_CANDIDATE":
                errors.append("%s is not retained as an exact trade candidate" % label)
            if item.get("candidate_symbol") != "🔵":
                errors.append("%s candidate color is not blue" % label)
            admission = item.get("admission_audit") or {}
            if admission.get("upstream_candidate_disposition") != candidate.get(
                "disposition"
            ):
                errors.append("%s hides its upstream disposition" % label)
            if admission.get("profit_evidence_gate_passed") is not expected_gate:
                errors.append("%s admission audit hides the evidence gate" % label)
            expected_probabilities = candidate.get(
                "development_model_diagnostics_not_pop_or_edge", {}
            ).get("raw_probabilities_not_pop") or {}
            expected_diagnostic_pop = expected_probabilities.get("POP_NET")
            raw_score = item["raw_classifier_score_not_pop"]
            if raw_score.get("point") != expected_diagnostic_pop:
                errors.append("%s raw classifier score is not reproducible" % label)
            if raw_score.get("status") != "UNPUBLISHABLE_UNCALIBRATED_OUT_OF_DOMAIN":
                errors.append("%s mislabels its raw classifier score" % label)
            if raw_score.get("used_for_action") is not False:
                errors.append("%s uses a raw classifier score for its action" % label)
            if item.get("failed_model_outputs_used") is not False:
                errors.append("%s uses failed model output" % label)
            if item.get("market_open_quote_refresh_owner") != "CULTRA_PIPELINE":
                errors.append("%s does not assign quote refresh to Cultra" % label)
            if item.get("user_reprice_instruction") is not False:
                errors.append("%s delegates repricing to the user" % label)
            expected_symbol = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(
                str(item.get("action_color"))
            )
            if item.get("action_symbol") != expected_symbol:
                errors.append("%s action color and symbol disagree" % label)
            if item.get("manual_ticket_enabled") is not False:
                errors.append("%s improperly enables a validated ticket" % label)
            if item.get("broker_submission_enabled") is not False:
                errors.append("%s improperly enables broker submission" % label)
            if item.get("quantity") != "USER DETERMINED":
                errors.append("%s supplies quantity" % label)
            confirmed_date = item["event_gate"].get("confirmed_date")
            if confirmed_date is not None and str(confirmed_date) not in board_text:
                errors.append("%s omits its exact confirmed event date" % label)
        except (KeyError, TypeError, ValueError):
            errors.append("%s is malformed" % label)
    try:
        config = _load(CONFIG_PATH)
        event_artifact = _load(EVENTS_PATH)
        event_map = {
            str(item["ticker"]): item for item in event_artifact.get("events", ())
        }
        expected_actions, expected_exclusions = _build_manual_research_actions(
            candidates,
            date.fromisoformat(str(summary["provider_trade_date"])),
            config["manual_research_action_policy"],
            event_map,
        )
        if _canonical(actions) != _canonical(expected_actions):
            errors.append(
                "conditional manual research actions do not reproduce from all candidates"
            )
        if _canonical(action_exclusions) != _canonical(expected_exclusions):
            errors.append("action exclusions do not reproduce from all candidates")
    except (KeyError, TypeError, ValueError, PatternError):
        errors.append("conditional manual research action policy is not reproducible")
    if summary.get("manual_tickets") != [] or summary.get("manual_ticket_enabled") is not False:
        errors.append("pattern run improperly enables manual tickets")
    if summary.get("network_attempted") is not False:
        errors.append("offline pattern run claims a network attempt")
    if summary.get("broker_submission_enabled") is not False:
        errors.append("pattern run improperly enables broker submission")
    if readiness.get("status") not in {"READY", "BLOCKED"}:
        errors.append("production readiness status is invalid")
    if (readiness.get("status") == "READY") != (
        int(readiness.get("blocker_count", -1)) == 0
    ):
        errors.append("production readiness status and blocker count disagree")
    if bool(readiness.get("historically_validated_action_enabled")) != (
        readiness.get("status") == "READY"
    ):
        errors.append("historical action state disagrees with production readiness")
    if "Production readiness" not in board_text:
        errors.append("decision board omits production readiness")
    reconciliation = manifest.get("reconciliation", {})
    if int(reconciliation.get("universe_count", -1)) != len(universe_rows):
        errors.append("manifest universe count does not reconcile")
    if int(reconciliation.get("candidate_count", -1)) != len(candidates):
        errors.append("manifest candidate count does not reconcile")
    if int(
        reconciliation.get("conditional_manual_research_action_count", -1)
    ) != len(actions):
        errors.append("manifest action count does not reconcile")
    if int(reconciliation.get("production_readiness_blocker_count", -1)) != int(
        readiness.get("blocker_count", -2)
    ):
        errors.append("manifest production blocker count does not reconcile")
    return tuple(errors)


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_CHAINS",
    "DEFAULT_HISTORY",
    "DEFAULT_ORATS",
    "DEFAULT_SCREEN",
    "DEFAULT_SELECTION",
    "EVENTS_PATH",
    "PatternError",
    "build_pattern_run",
    "verify_pattern_run",
]
