"""Leakage-safe exact-leg outcome generation for Cultra V2."""

from __future__ import annotations

import hashlib
import functools
import json
import math
import os
import sqlite3
import statistics
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .domain import LegAction, LegQuote, OptionType
from .edge import CostBreakdown
from .economics import round_trip_costs
from .historical_events import load_historical_event_manifest
from .historical_v2 import HISTORICAL_ROOT
from .hypotheses import FROZEN_HYPOTHESIS_REGISTRY, HypothesisDefinition
from .protocol import load_historical_campaign_protocol
from .sessions import HistoricalSessionCalendar, load_historical_session_calendar
from .structures import (
    ContractQuote,
    HistoricalStructureOutcome,
    SelectedStructure,
    StructureError,
    resolve_historical_structure_path,
    select_frozen_structure,
    structure_risk_envelope,
)


class OutcomeV2Error(RuntimeError):
    """Historical candidates or outcomes cannot be reproduced exactly."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _load_manifest(database: Path) -> Mapping[str, Any]:
    path = database.with_suffix(database.suffix + ".manifest.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutcomeV2Error("normalized historical V2 manifest is unavailable") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "cultra.normalized-historical-v2-manifest.v1"
        or Path(str(value.get("database", ""))).resolve() != database
        or int(value.get("database_bytes", -1)) != database.stat().st_size
        or value.get("database_sha256") != _sha256(database)
    ):
        raise OutcomeV2Error("normalized historical V2 manifest does not reconcile")
    return value


def _open_verified(database_path: Path) -> Tuple[sqlite3.Connection, Mapping[str, Any]]:
    database = Path(database_path).expanduser().resolve()
    try:
        database.relative_to(HISTORICAL_ROOT)
    except ValueError as exc:
        raise OutcomeV2Error("historical V2 database is outside Cultra") from exc
    if not database.is_file():
        raise OutcomeV2Error("historical V2 database is unavailable")
    manifest = _load_manifest(database)
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % database, uri=True)
        connection.row_factory = sqlite3.Row
        check = connection.execute("PRAGMA integrity_check").fetchone()
        if check is None or check[0] != "ok":
            raise OutcomeV2Error("historical V2 database failed integrity check")
        metadata = {
            str(row[0]): json.loads(str(row[1]))
            for row in connection.execute("SELECT key, value FROM metadata")
        }
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise OutcomeV2Error("historical V2 database cannot be verified") from exc
    if (
        metadata.get("schema") != "cultra.normalized-historical-v2.v1"
        or metadata.get("campaign_freeze_hash") != manifest.get("campaign_freeze_hash")
    ):
        connection.close()
        raise OutcomeV2Error("historical V2 metadata does not reconcile")
    return connection, manifest


def historical_costs(
    selection: SelectedStructure, policy: Mapping[str, Any]
) -> CostBreakdown:
    """Freeze round-trip commissions, fees, and spread-dependent slippage."""

    return round_trip_costs(selection.legs, selection.entry_quotes, policy)


def _contract_from_row(row: sqlite3.Row, close_at: datetime) -> ContractQuote:
    return ContractQuote(
        ticker=str(row["ticker"]),
        trade_date=date.fromisoformat(str(row["trade_date"])),
        expiration=date.fromisoformat(str(row["expiration"])),
        dte=int(row["dte"]),
        strike=float(row["strike"]),
        call_delta=float(row["call_delta"]),
        call_bid=None if row["call_bid"] is None else float(row["call_bid"]),
        call_ask=None if row["call_ask"] is None else float(row["call_ask"]),
        put_bid=None if row["put_bid"] is None else float(row["put_bid"]),
        put_ask=None if row["put_ask"] is None else float(row["put_ask"]),
        call_open_interest=(
            None
            if row["call_open_interest"] is None
            else int(row["call_open_interest"])
        ),
        put_open_interest=(
            None
            if row["put_open_interest"] is None
            else int(row["put_open_interest"])
        ),
        observed_at=close_at,
        snapshot_id=str(row["snapshot_id"]),
        stock_price=(
            None if row["stock_price"] is None else float(row["stock_price"])
        ),
    )


def _chain(
    connection: sqlite3.Connection,
    calendar: HistoricalSessionCalendar,
    ticker: str,
    trade_date: date,
) -> Tuple[ContractQuote, ...]:
    close_at = calendar.close_for(trade_date)
    rows = tuple(
        connection.execute(
            """
            SELECT * FROM chain_quotes
             WHERE ticker = ? AND trade_date = ?
             ORDER BY expiration, strike
            """,
            (ticker, trade_date.isoformat()),
        )
    )
    return tuple(_contract_from_row(row, close_at) for row in rows)


def _leg_quotes_from_chain(
    selection: SelectedStructure,
    contracts: Sequence[ContractQuote],
) -> Tuple[LegQuote, ...]:
    contract_map = {(item.expiration, item.strike): item for item in contracts}
    quotes = []
    for leg in selection.legs:
        contract = contract_map.get((leg.expiration, leg.strike))
        if contract is None:
            raise StructureError("historical exact-contract path is missing a selected leg")
        bid, ask = contract.bid_ask(leg.option_type)
        if bid is None or ask is None:
            raise StructureError("historical exact-contract quote side is missing")
        quotes.append(
            LegQuote(leg.occ_symbol, float(bid), float(ask), contract.observed_at)
        )
    return tuple(quotes)


def _early_exercise_risk(
    selection: SelectedStructure, contracts: Sequence[ContractQuote]
) -> Optional[str]:
    """Fail closed when a short American option has negligible extrinsic value.

    Daily EOD chains cannot prove whether an assignment occurred between marks.
    Rather than silently treating that path as an ordinary close, Cultra marks
    the outcome unresolved whenever a short in-the-money option's quoted
    extrinsic value is five cents or less.
    """

    contract_map = {(item.expiration, item.strike): item for item in contracts}
    for leg in selection.legs:
        if leg.action is not LegAction.SELL:
            continue
        contract = contract_map.get((leg.expiration, leg.strike))
        if contract is None or contract.stock_price is None:
            return "EARLY_EXERCISE_STRESS_UNAVAILABLE"
        bid, ask = contract.bid_ask(leg.option_type)
        if bid is None or ask is None:
            return "EARLY_EXERCISE_STRESS_QUOTE_UNAVAILABLE"
        intrinsic = (
            max(0.0, contract.stock_price - leg.strike)
            if leg.option_type is OptionType.CALL
            else max(0.0, leg.strike - contract.stock_price)
        )
        extrinsic = max(0.0, (float(bid) + float(ask)) / 2.0 - intrinsic)
        if intrinsic > 0.0 and extrinsic <= 0.05:
            return "EARLY_EXERCISE_RISK_SHORT_%s" % leg.option_type.value
    return None


def _market_features(
    connection: sqlite3.Connection,
    session_dates: Sequence[date],
    signal_index: int,
    ticker: str,
) -> Mapping[str, Any]:
    signal_date = session_dates[signal_index]
    row = connection.execute(
        "SELECT * FROM core_features WHERE ticker = ? AND trade_date = ?",
        (ticker, signal_date.isoformat()),
    ).fetchone()
    if row is None or signal_index < 20:
        raise OutcomeV2Error("signal-time Core or 20-session history is unavailable")
    prices = []
    for selected in session_dates[signal_index - 20 : signal_index + 1]:
        value = connection.execute(
            "SELECT priorCls FROM core_features WHERE ticker = ? AND trade_date = ?",
            (ticker, selected.isoformat()),
        ).fetchone()
        if value is None or value[0] is None or float(value[0]) <= 0.0:
            raise OutcomeV2Error("signal-time price history is incomplete")
        prices.append(float(value[0]))
    returns = tuple(math.log(right / left) for left, right in zip(prices[:-1], prices[1:]))
    realized = statistics.pstdev(returns) * math.sqrt(252.0)
    result = {key: row[key] for key in row.keys()}
    result["momentum_20"] = prices[-1] / prices[0] - 1.0
    result["realized_volatility_20"] = realized
    return result


def _chain_shape(chain: Sequence[ContractQuote], selection: SelectedStructure) -> Mapping[str, float]:
    quote_map = {item.occ_symbol: quote for item, quote in zip(selection.legs, selection.entry_quotes)}
    weighted_mid = 0.0
    weighted_spread = 0.0
    for leg in selection.legs:
        quote = quote_map[leg.occ_symbol]
        weighted_mid += quote.midpoint * leg.ratio
        weighted_spread += quote.spread * leg.ratio
    if weighted_mid <= 0.0:
        raise OutcomeV2Error("selected structure midpoint is not positive")
    return {"relative_spread": weighted_spread / weighted_mid}


def _iv_shape(
    connection: sqlite3.Connection,
    ticker: str,
    trade_date: date,
    selection: SelectedStructure,
) -> Mapping[str, float]:
    front_expiration = min(item.expiration for item in selection.legs)
    rows = tuple(
        connection.execute(
            """
            SELECT call_delta, call_mid_iv
              FROM chain_quotes
             WHERE ticker = ? AND trade_date = ?
               AND expiration = ?
               AND call_mid_iv IS NOT NULL
             ORDER BY strike
            """,
            (ticker, trade_date.isoformat(), front_expiration.isoformat()),
        )
    )
    if not rows:
        raise OutcomeV2Error("entry chain IV surface is unavailable")

    def value(target: float) -> float:
        row = min(rows, key=lambda item: (abs(float(item[0]) - target), float(item[0])))
        return float(row[1])

    low, atm, high = value(0.25), value(0.50), value(0.75)
    return {
        "chain_skew_slope": (low - high) / 0.50,
        "chain_skew_curvature": low + high - 2.0 * atm,
    }


def _features(
    *,
    hypothesis: HypothesisDefinition,
    market: Mapping[str, Any],
    chain: Sequence[ContractQuote],
    selection: SelectedStructure,
    risk_reference: float,
    iv_shape: Mapping[str, float],
    feature_names: Sequence[str],
) -> Mapping[str, float]:
    values: Dict[str, Any] = dict(market)
    values.update(_chain_shape(chain, selection))
    values.update(iv_shape)
    values["entry_price_on_risk"] = (
        abs(selection.signed_entry_debit) * 100.0 / risk_reference
    )
    result = {}
    for name in feature_names:
        value = values.get(name)
        if value is None:
            raise OutcomeV2Error("required signal feature is unavailable: %s" % name)
        converted = float(value)
        if not math.isfinite(converted):
            raise OutcomeV2Error("required signal feature is non-finite: %s" % name)
        result[name] = converted
    return result


def _selection_payload(selection: SelectedStructure) -> Mapping[str, Any]:
    return {
        "hypothesis_id": selection.hypothesis_id,
        "strategy_id": selection.strategy_id,
        "holding_sessions": selection.holding_sessions,
        "template_hash": selection.template_hash,
        "signed_entry_debit": selection.signed_entry_debit,
        "entry_snapshot_ids": list(selection.entry_snapshot_ids),
        "legs": [
            {
                "occ_symbol": item.occ_symbol,
                "action": item.action.value,
                "option_type": item.option_type.value,
                "expiration": item.expiration.isoformat(),
                "strike": item.strike,
                "ratio": item.ratio,
                "bid": quote.bid,
                "ask": quote.ask,
                "quote_timestamp": quote.timestamp.isoformat(),
            }
            for item, quote in zip(selection.legs, selection.entry_quotes)
        ],
    }


def _outcome_payload(outcome: HistoricalStructureOutcome) -> Mapping[str, Any]:
    value = asdict(outcome)
    value["exit_date"] = outcome.exit_date.isoformat()
    return value


def _event_payload(events: Sequence[Any]) -> Sequence[Mapping[str, Any]]:
    return [
        {
            "ticker": item.ticker,
            "event_type": item.event_type,
            "effective_date": item.effective_date.isoformat(),
            "available_at": item.available_at.isoformat(),
            "source_event_id": item.source_event_id,
            "status": item.status,
            "cash_amount": item.cash_amount,
            "split_ratio": item.split_ratio,
            "adjustment_reference": item.adjustment_reference,
        }
        for item in events
    ]


def _record_id(
    campaign_id: str, ticker: str, signal_date: date, hypothesis_id: str
) -> str:
    return hashlib.sha256(
        ("|".join((campaign_id, ticker, signal_date.isoformat(), hypothesis_id))).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_candidate(
    connection: sqlite3.Connection,
    *,
    record_id: str,
    hypothesis: HypothesisDefinition,
    ticker: str,
    signal_date: date,
    signal_close_at: datetime,
    entry_date: date,
    planned_exit_date: date,
    status: str,
    reason: str,
    selection: Optional[Mapping[str, Any]] = None,
    features: Optional[Mapping[str, Any]] = None,
    costs: Optional[Mapping[str, Any]] = None,
    risk: Optional[Mapping[str, Any]] = None,
    known_events: Sequence[Mapping[str, Any]] = (),
    path_events: Sequence[Mapping[str, Any]] = (),
    outcome: Optional[Mapping[str, Any]] = None,
) -> None:
    connection.execute(
        """
        INSERT INTO candidate_ledger VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            record_id,
            hypothesis.hypothesis_id,
            hypothesis.strategy_id,
            hypothesis.signal_profile,
            hypothesis.signal_bias,
            hypothesis.holding_sessions,
            ticker,
            signal_date.isoformat(),
            signal_close_at.isoformat(),
            entry_date.isoformat(),
            planned_exit_date.isoformat(),
            status,
            reason,
            None if selection is None else _canonical(selection),
            None if features is None else _canonical(features),
            None if costs is None else _canonical(costs),
            None if risk is None else _canonical(risk),
            _canonical(list(known_events)),
            _canonical(list(path_events)),
            None if outcome is None else _canonical(outcome),
        ),
    )


def generate_historical_v2_outcomes(
    *, normalized_database: Path, output_database: Path
) -> Mapping[str, Any]:
    """Generate every geometrically executable candidate without outcome screening."""

    source, source_manifest = _open_verified(normalized_database)
    metadata = {
        str(row[0]): json.loads(str(row[1]))
        for row in source.execute("SELECT key, value FROM metadata")
    }
    campaign_completion = json.loads(
        Path(str(source_manifest["campaign_completion"])).read_text(encoding="utf-8")
    )
    campaign_freeze = json.loads(
        Path(str(campaign_completion["campaign_freeze_path"])).read_text(encoding="utf-8")
    )
    inputs = campaign_freeze["inputs"]
    calendar = load_historical_session_calendar(Path(inputs["sessions"]["path"]))
    events = load_historical_event_manifest(Path(inputs["events"]["path"]))
    cohorts = json.loads(Path(inputs["cohorts"]["path"]).read_text(encoding="utf-8"))
    protocol = load_historical_campaign_protocol()
    destination = Path(output_database).expanduser().resolve()
    try:
        destination.relative_to(HISTORICAL_ROOT)
    except ValueError as exc:
        source.close()
        raise OutcomeV2Error("historical outcome store must remain inside Cultra") from exc
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        source.close()
        raise OutcomeV2Error("historical outcome output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(".%s.tmp-%d" % (destination.name, os.getpid()))
    output = sqlite3.connect(str(temporary))
    try:
        output.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE candidate_ledger(
                record_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                signal_profile TEXT NOT NULL,
                signal_bias TEXT NOT NULL,
                holding_sessions INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                signal_close_at TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                planned_exit_date TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('RESOLVED','DATA_UNAVAILABLE','UNRESOLVED')),
                reason TEXT NOT NULL,
                selection_json TEXT,
                features_json TEXT,
                costs_json TEXT,
                risk_json TEXT,
                known_events_json TEXT NOT NULL,
                path_events_json TEXT NOT NULL,
                outcome_json TEXT
            );
            CREATE INDEX candidates_by_hypothesis_date
                ON candidate_ledger(hypothesis_id, signal_date);
            CREATE INDEX candidates_by_status
                ON candidate_ledger(status);
            """
        )
        session_dates = tuple(item.session_date for item in calendar.sessions)
        date_index = {value: index for index, value in enumerate(session_dates)}
        feature_profiles = protocol["learning_policy"]["feature_profiles"]

        @functools.lru_cache(maxsize=700)
        def cached_chain(
            ticker: str, trade_date_iso: str
        ) -> Tuple[ContractQuote, ...]:
            return _chain(
                source, calendar, ticker, date.fromisoformat(trade_date_iso)
            )

        counts = {"RESOLVED": 0, "DATA_UNAVAILABLE": 0, "UNRESOLVED": 0}
        for block in cohorts["blocks"]:
            start_index = date_index[date.fromisoformat(str(block["block_start"]))]
            signal_count = int(block["eligible_signal_session_count"])
            for local_index in range(signal_count):
                signal_index = start_index + local_index
                signal_date = session_dates[signal_index]
                entry_date = session_dates[signal_index + 1]
                for ticker in block["tickers"]:
                    market: Optional[Mapping[str, Any]] = None
                    market_error: Optional[str] = None
                    try:
                        market = _market_features(source, session_dates, signal_index, ticker)
                    except OutcomeV2Error as exc:
                        market_error = str(exc)
                    entry_chain = cached_chain(ticker, entry_date.isoformat())
                    for hypothesis in FROZEN_HYPOTHESIS_REGISTRY:
                        planned_exit = session_dates[
                            signal_index + 1 + hypothesis.holding_sessions
                        ]
                        record_id = _record_id(
                            str(metadata["campaign_id"]), ticker, signal_date, hypothesis.hypothesis_id
                        )
                        try:
                            selection = select_frozen_structure(
                                hypothesis_id=hypothesis.hypothesis_id,
                                strategy_id=hypothesis.strategy_id,
                                holding_sessions=hypothesis.holding_sessions,
                                contracts=entry_chain,
                                required_path_end=planned_exit,
                            )
                            costs = historical_costs(selection, protocol["cost_policy"])
                            envelope = structure_risk_envelope(selection, costs)
                            risk_reference = (
                                float(envelope.maximum_loss)
                                if envelope is not None
                                else max(
                                    1.0,
                                    abs(selection.signed_entry_debit) * 100.0
                                    + costs.total,
                                )
                            )
                        except (StructureError, ValueError) as exc:
                            _write_candidate(
                                output,
                                record_id=record_id,
                                hypothesis=hypothesis,
                                ticker=ticker,
                                signal_date=signal_date,
                                signal_close_at=calendar.close_for(signal_date),
                                entry_date=entry_date,
                                planned_exit_date=planned_exit,
                                status="DATA_UNAVAILABLE",
                                reason=str(exc),
                            )
                            counts["DATA_UNAVAILABLE"] += 1
                            continue

                        selection_payload = _selection_payload(selection)
                        costs_payload = asdict(costs)
                        risk_payload = {
                            "risk_reference": risk_reference,
                            "maximum_loss": (
                                None if envelope is None else float(envelope.maximum_loss)
                            ),
                            "maximum_profit": (
                                None
                                if envelope is None or envelope.maximum_profit is None
                                else float(envelope.maximum_profit)
                            ),
                            "defined_risk": envelope is not None,
                        }
                        if market is None:
                            # Preserve the exact geometrically selected legs
                            # and finite risk even when signal features are
                            # missing. Holdout evaluation can then charge the
                            # row as selected at worst-case loss instead of
                            # letting missing features remove it from evidence.
                            _write_candidate(
                                output,
                                record_id=record_id,
                                hypothesis=hypothesis,
                                ticker=ticker,
                                signal_date=signal_date,
                                signal_close_at=calendar.close_for(signal_date),
                                entry_date=entry_date,
                                planned_exit_date=planned_exit,
                                status="DATA_UNAVAILABLE",
                                reason=market_error or "SIGNAL_FEATURES_UNAVAILABLE",
                                selection=selection_payload,
                                costs=costs_payload,
                                risk=risk_payload,
                            )
                            counts["DATA_UNAVAILABLE"] += 1
                            continue
                        try:
                            iv_shape = _iv_shape(
                                source, ticker, entry_date, selection
                            )
                            features = _features(
                                hypothesis=hypothesis,
                                market=market,
                                chain=entry_chain,
                                selection=selection,
                                risk_reference=risk_reference,
                                iv_shape=iv_shape,
                                feature_names=feature_profiles[
                                    hypothesis.signal_profile
                                ],
                            )
                        except (OutcomeV2Error, ValueError) as exc:
                            _write_candidate(
                                output,
                                record_id=record_id,
                                hypothesis=hypothesis,
                                ticker=ticker,
                                signal_date=signal_date,
                                signal_close_at=calendar.close_for(signal_date),
                                entry_date=entry_date,
                                planned_exit_date=planned_exit,
                                status="DATA_UNAVAILABLE",
                                reason=str(exc),
                                selection=selection_payload,
                                costs=costs_payload,
                                risk=risk_payload,
                            )
                            counts["DATA_UNAVAILABLE"] += 1
                            continue
                        known = events.known_events(
                            ticker=ticker,
                            signal_timestamp=calendar.close_for(signal_date),
                            through_date=planned_exit,
                        )
                        path_events = events.events_in_window(
                            ticker=ticker,
                            start_date=entry_date,
                            end_date=planned_exit,
                        )
                        unsupported = {
                            item.event_type
                            for item in path_events
                            if item.event_type
                            in {"SPLIT", "CONTRACT_ADJUSTMENT", "DELISTING"}
                        }
                        if any(item.event_type == "DIVIDEND" for item in path_events) and any(
                            leg.action is LegAction.SELL for leg in selection.legs
                        ):
                            unsupported.add("SHORT_OPTION_DIVIDEND_ASSIGNMENT")
                        if unsupported:
                            _write_candidate(
                                output,
                                record_id=record_id,
                                hypothesis=hypothesis,
                                ticker=ticker,
                                signal_date=signal_date,
                                signal_close_at=calendar.close_for(signal_date),
                                entry_date=entry_date,
                                planned_exit_date=planned_exit,
                                status="UNRESOLVED",
                                reason="UNSUPPORTED_PATH_EVENT:" + ",".join(sorted(unsupported)),
                                selection=selection_payload,
                                features=features,
                                costs=costs_payload,
                                risk=risk_payload,
                                known_events=_event_payload(known),
                                path_events=_event_payload(path_events),
                            )
                            counts["UNRESOLVED"] += 1
                            continue
                        try:
                            path_rows = []
                            early_exercise_reason = None
                            for offset in range(
                                signal_index + 2,
                                signal_index + 2 + hypothesis.holding_sessions,
                            ):
                                path_date = session_dates[offset]
                                path_chain = cached_chain(
                                    ticker, path_date.isoformat()
                                )
                                early_exercise_reason = _early_exercise_risk(
                                    selection, path_chain
                                )
                                if early_exercise_reason is not None:
                                    break
                                path_rows.append(
                                    (
                                        path_date,
                                        _leg_quotes_from_chain(selection, path_chain),
                                    )
                                )
                            if early_exercise_reason is not None:
                                raise StructureError(early_exercise_reason)
                            path = tuple(path_rows)
                            outcome = resolve_historical_structure_path(selection, path, costs)
                        except (StructureError, ValueError) as exc:
                            _write_candidate(
                                output,
                                record_id=record_id,
                                hypothesis=hypothesis,
                                ticker=ticker,
                                signal_date=signal_date,
                                signal_close_at=calendar.close_for(signal_date),
                                entry_date=entry_date,
                                planned_exit_date=planned_exit,
                                status="UNRESOLVED",
                                reason=str(exc),
                                selection=selection_payload,
                                features=features,
                                costs=costs_payload,
                                risk=risk_payload,
                                known_events=_event_payload(known),
                                path_events=_event_payload(path_events),
                            )
                            counts["UNRESOLVED"] += 1
                            continue
                        _write_candidate(
                            output,
                            record_id=record_id,
                            hypothesis=hypothesis,
                            ticker=ticker,
                            signal_date=signal_date,
                            signal_close_at=calendar.close_for(signal_date),
                            entry_date=entry_date,
                            planned_exit_date=planned_exit,
                            status="RESOLVED",
                            reason="EXACT_PATH_RESOLVED",
                            selection=selection_payload,
                            features=features,
                            costs=costs_payload,
                            risk=risk_payload,
                            known_events=_event_payload(known),
                            path_events=_event_payload(path_events),
                            outcome=_outcome_payload(outcome),
                        )
                        counts["RESOLVED"] += 1
        total = sum(counts.values())
        expected = sum(
            int(block["eligible_signal_session_count"]) * len(block["tickers"])
            for block in cohorts["blocks"]
        ) * len(FROZEN_HYPOTHESIS_REGISTRY)
        if total != expected:
            raise OutcomeV2Error("candidate ledger does not cover the frozen hypothesis grid")
        output_metadata = {
            "schema": "cultra.historical-outcomes-v2.v1",
            "campaign_id": metadata["campaign_id"],
            "campaign_freeze_hash": metadata["campaign_freeze_hash"],
            "normalized_database_sha256": _sha256(normalized_database),
            "candidate_generation": "EVERY_GEOMETRICALLY_EXECUTABLE_TICKER_DATE_STRUCTURE",
            "expected_candidate_grid": expected,
            "network_attempted": False,
        }
        for key, value in output_metadata.items():
            output.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True)),
            )
        output.commit()
        check = output.execute("PRAGMA integrity_check").fetchone()
        if check is None or check[0] != "ok":
            raise OutcomeV2Error("historical outcome database failed integrity check")
    except BaseException:
        output.rollback()
        output.close()
        source.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    output.close()
    source.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    result = {
        "schema": "cultra.historical-outcomes-v2-manifest.v1",
        "campaign_id": metadata["campaign_id"],
        "campaign_freeze_hash": metadata["campaign_freeze_hash"],
        "normalized_database": str(Path(normalized_database).expanduser().resolve()),
        "normalized_database_sha256": _sha256(normalized_database),
        "database": str(destination),
        "database_bytes": destination.stat().st_size,
        "database_sha256": _sha256(destination),
        "counts": counts,
        "candidate_grid": total,
        "network_attempted": False,
    }
    with open(manifest_path, "xb") as handle:
        os.chmod(manifest_path, 0o600)
        handle.write(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return dict(result, manifest=str(manifest_path))


__all__ = [
    "OutcomeV2Error",
    "generate_historical_v2_outcomes",
    "historical_costs",
]
