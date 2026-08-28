"""Prospective, append-only shadow outcome scoring for Options Agent."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from ._vendor.schwab_auth import compact_occ_to_schwab_symbol


SCHEMA_VERSION = "options_agent.shadow_outcome.v1"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MIN_OUTCOME_OBSERVATION_TIME = dt.time(15, 45)
OUTCOME_COLUMNS = [
    "schema_version",
    "outcome_id",
    "logical_recommendation_id",
    "recommendation_date",
    "evaluation_due_date",
    "observed_at",
    "observation_session_date",
    "ticker",
    "strategy",
    "strategy_route",
    "strategy_family",
    "entry_type",
    "direction_bucket",
    "regime",
    "dte",
    "dte_bucket",
    "iv_rank_bucket",
    "economics_bucket",
    "liquidity_bucket",
    "entry_limit",
    "expiry",
    "opened_at",
    "closed_at",
    "entry_order_ids",
    "realized_pnl",
    "liquidation_value",
    "outcome_status",
    "exact_evaluated",
    "contributes_to_expectancy",
    "source",
    "quote_source",
    "quote_snapshot_json",
    "legs_json",
    "recommendation_pipeline_version",
    "selector_policy_id",
    "recommendation_code_git_sha",
]
ATTEMPT_COLUMNS = [
    "logical_recommendation_id",
    "recommendation_date",
    "evaluation_due_date",
    "observation_session_date",
    "ticker",
    "status",
    "blocker",
    "required_symbols",
    "outcome_id",
]


def read_shadow_outcomes(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"corrupt shadow outcome registry line {line_number}: {exc}") from exc
                if not isinstance(row, Mapping):
                    raise RuntimeError(f"corrupt shadow outcome registry line {line_number}: object required")
                rows.append(dict(row))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def collect_due_shadow_outcomes(
    shadow_rows: pd.DataFrame,
    *,
    outcome_registry_path: Path,
    observation_session_date: dt.date,
    live_schwab: bool,
    quote_fetcher: Optional[Callable[[Sequence[str]], Mapping[str, Any]]] = None,
    historical_quote_fetcher: Optional[
        Callable[[dt.date, Sequence[str]], Mapping[str, Any]]
    ] = None,
    observed_at: Optional[dt.datetime] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score only the fixed fifth-session cohort using conservative exact-leg quotes."""

    observed_at = _aware_utc(observed_at or dt.datetime.now(dt.timezone.utc))
    observed_market_time = observed_at.astimezone(MARKET_TIMEZONE)
    existing = read_shadow_outcomes(outcome_registry_path)
    existing_ids = set(existing.get("logical_recommendation_id", pd.Series(dtype=str)).astype(str))
    attempts: list[dict[str, Any]] = []
    due: list[dict[str, Any]] = []
    frame = shadow_rows.copy() if shadow_rows is not None else pd.DataFrame()
    for _, row in frame.iterrows():
        logical_id = _text(row.get("logical_recommendation_id"))
        recommendation_date = _date(row.get("recommendation_date"))
        due_date = _date(row.get("evaluation_due_date"))
        symbols = _leg_symbols(row.get("legs_json"))
        attempt = {
            "logical_recommendation_id": logical_id,
            "recommendation_date": recommendation_date.isoformat() if recommendation_date else "",
            "evaluation_due_date": due_date.isoformat() if due_date else "",
            "observation_session_date": observation_session_date.isoformat(),
            "ticker": _text(row.get("ticker")).upper(),
            "status": "",
            "blocker": "",
            "required_symbols": ";".join(symbols),
            "outcome_id": "",
        }
        if logical_id in existing_ids:
            match = existing[existing["logical_recommendation_id"].astype(str).eq(logical_id)].iloc[0]
            attempt.update(status="ALREADY_SCORED", outcome_id=_text(match.get("outcome_id")))
        elif _text(row.get("registration_status")) != "VALID_PROSPECTIVE":
            attempt.update(status="EXCLUDED", blocker="invalid_prospective_registration")
        elif not recommendation_date or not due_date or not logical_id:
            attempt.update(status="EXCLUDED", blocker="shadow_row_missing_identity_or_dates")
        elif observation_session_date < due_date:
            attempt.update(status="NOT_DUE", blocker="fixed_evaluation_session_not_reached")
        elif observation_session_date > due_date:
            if not symbols:
                attempt.update(status="BLOCKED", blocker="directed_legs_missing")
            elif historical_quote_fetcher is None:
                attempt.update(status="MISSED", blocker="fixed_evaluation_session_missed")
            else:
                attempt.update(
                    status="DUE_HISTORICAL",
                    blocker="",
                    observation_session_date=due_date.isoformat(),
                )
                due.append(
                    {
                        "row": row.to_dict(),
                        "attempt": attempt,
                        "symbols": symbols,
                        "quote_mode": "historical",
                        "quote_date": due_date,
                    }
                )
        elif (
            observed_market_time.date() == observation_session_date
            and observed_market_time.time() < MIN_OUTCOME_OBSERVATION_TIME
        ):
            attempt.update(
                status="NOT_DUE",
                blocker="wait_until_near_close_for_same_session_quotes",
            )
        elif not live_schwab:
            attempt.update(status="BLOCKED", blocker="live_schwab_required_for_exact_outcome")
        elif not symbols:
            attempt.update(status="BLOCKED", blocker="directed_legs_missing")
        else:
            attempt["status"] = "DUE"
            due.append(
                {
                    "row": row.to_dict(),
                    "attempt": attempt,
                    "symbols": symbols,
                    "quote_mode": "live",
                    "quote_date": observation_session_date,
                }
            )
        attempts.append(attempt)

    quote_payloads: dict[tuple[str, dt.date], Mapping[str, Any]] = {}
    fetch_errors: dict[tuple[str, dt.date], str] = {}
    live_groups = {
        (str(item["quote_mode"]), item["quote_date"])
        for item in due
        if item["quote_mode"] == "live"
    }
    for group_key in live_groups:
        try:
            if quote_fetcher is None:
                from ._vendor.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

                service = SchwabLiveDataService(
                    SchwabAuthConfig.from_env(load_dotenv_file=True),
                    interactive_login=False,
                )
                quote_fetcher = service.get_quotes
            requested = sorted(
                {
                    compact_occ_to_schwab_symbol(symbol)
                    for item in due
                    if item["quote_mode"] == "live"
                    for symbol in item["symbols"]
                }
            )
            fetched: dict[str, Any] = {}
            for start in range(0, len(requested), 100):
                fetched.update(quote_fetcher(requested[start : start + 100]))
            quote_payloads[group_key] = fetched
        except Exception as exc:
            fetch_errors[group_key] = str(exc)

    historical_dates = sorted(
        {
            item["quote_date"]
            for item in due
            if item["quote_mode"] == "historical"
        }
    )
    for quote_date in historical_dates:
        group_key = ("historical", quote_date)
        try:
            requested = sorted(
                {
                    symbol
                    for item in due
                    if item["quote_mode"] == "historical" and item["quote_date"] == quote_date
                    for symbol in item["symbols"]
                }
            )
            quote_payloads[group_key] = historical_quote_fetcher(quote_date, requested)  # type: ignore[misc]
        except Exception as exc:
            fetch_errors[group_key] = str(exc)

    new_rows: list[dict[str, Any]] = []
    for item in due:
        attempt = item["attempt"]
        row = item["row"]
        group_key = (str(item["quote_mode"]), item["quote_date"])
        fetch_error = fetch_errors.get(group_key, "")
        if fetch_error:
            attempt.update(status="BLOCKED", blocker=f"quote_fetch_failed:{fetch_error}")
            continue
        scored, blocker = _score_row(
            row,
            quote_payloads.get(group_key, {}),
            observed_at=observed_at,
            observation_session_date=item["quote_date"],
            quote_source=(
                "dated_uw_exact_option_quotes"
                if item["quote_mode"] == "historical"
                else "schwab_exact_option_quotes"
            ),
        )
        if scored is None:
            attempt.update(status="BLOCKED", blocker=blocker)
            continue
        new_rows.append(scored)
        attempt.update(status="SCORED", outcome_id=scored["outcome_id"])

    appended = _append_new_outcomes(outcome_registry_path, new_rows)
    outcomes = read_shadow_outcomes(outcome_registry_path)
    summary = {
        "registry_path": str(outcome_registry_path),
        "outcome_rows": int(len(outcomes)),
        "new_outcome_rows": int(appended),
        "due_rows": int(
            sum(item.get("status") in {"DUE", "DUE_HISTORICAL", "SCORED", "BLOCKED"} for item in attempts)
        ),
        "scored_rows": int(sum(item.get("status") == "SCORED" for item in attempts)),
        "missed_rows": int(sum(item.get("status") == "MISSED" for item in attempts)),
        "blocked_rows": int(sum(item.get("status") == "BLOCKED" for item in attempts)),
        "contributing_rows": int(
            outcomes.get("contributes_to_expectancy", pd.Series(False, index=outcomes.index))
            .map(_truthy)
            .sum()
        ),
        "policy": "exact_conservative_liquidation_on_fifth_regular_session_v2",
    }
    return outcomes, pd.DataFrame(attempts, columns=ATTEMPT_COLUMNS), summary


def _score_row(
    row: Mapping[str, Any],
    quotes: Mapping[str, Any],
    *,
    observed_at: dt.datetime,
    observation_session_date: dt.date,
    quote_source: str = "schwab_exact_option_quotes",
) -> tuple[Optional[dict[str, Any]], str]:
    entry_type = _text(row.get("entry_type")).upper()
    entry_limit = _float(row.get("entry_limit"))
    if entry_type not in {"DEBIT", "CREDIT"} or entry_limit is None or entry_limit <= 0:
        return None, "invalid_frozen_entry_assumption"
    try:
        legs = json.loads(_text(row.get("legs_json")))
    except json.JSONDecodeError:
        return None, "invalid_legs_json"
    if not isinstance(legs, list) or not legs:
        return None, "directed_legs_missing"

    liquidation = 0.0
    snapshots: list[dict[str, Any]] = []
    for leg in legs:
        if not isinstance(leg, Mapping):
            return None, "invalid_directed_leg"
        side = _text(leg.get("side")).upper()
        ratio = _float(leg.get("ratio"))
        compact_symbol = _text(leg.get("occ_symbol")).upper()
        if side not in {"BUY", "SELL"} or ratio is None or ratio <= 0 or not compact_symbol:
            return None, "invalid_directed_leg"
        schwab_symbol = compact_occ_to_schwab_symbol(compact_symbol)
        payload = _quote_for_symbol(quotes, schwab_symbol)
        bid, ask, quote_time = _quote_fields(payload)
        if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
            return None, f"invalid_or_missing_bid_ask:{compact_symbol}"
        if quote_time is None or quote_time.astimezone(MARKET_TIMEZONE).date() != observation_session_date:
            return None, f"quote_not_from_evaluation_session:{compact_symbol}"
        leg_cashflow = bid * ratio if side == "BUY" else -ask * ratio
        liquidation += leg_cashflow
        snapshots.append(
            {
                "occ_symbol": compact_symbol,
                "schwab_symbol": schwab_symbol,
                "side": side,
                "ratio": ratio,
                "bid": bid,
                "ask": ask,
                "quote_time": quote_time.isoformat(),
                "liquidation_cashflow": round(leg_cashflow, 6),
            }
        )

    initial_cashflow = entry_limit if entry_type == "CREDIT" else -entry_limit
    realized_pnl = round((initial_cashflow + liquidation) * 100.0, 2)
    selected_for_expectancy = _truthy(row.get("selected_for_expectancy"))
    logical_id = _text(row.get("logical_recommendation_id"))
    observed_key = observed_at.isoformat()
    outcome_id = hashlib.sha256(f"{logical_id}|{observation_session_date.isoformat()}".encode()).hexdigest()[:32]
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome_id": outcome_id,
        "logical_recommendation_id": logical_id,
        "recommendation_date": _text(row.get("recommendation_date")),
        "evaluation_due_date": _text(row.get("evaluation_due_date")),
        "observed_at": observed_key,
        "observation_session_date": observation_session_date.isoformat(),
        "ticker": _text(row.get("ticker")).upper(),
        "strategy": _text(row.get("trade_plan")),
        "strategy_route": _text(row.get("strategy_route")),
        "strategy_family": _text(row.get("strategy_family")),
        "entry_type": entry_type,
        "direction_bucket": _text(row.get("direction_bucket")),
        "regime": _text(row.get("regime")),
        "dte": row.get("dte", ""),
        "dte_bucket": _text(row.get("dte_bucket")),
        "iv_rank_bucket": _text(row.get("iv_rank_bucket")),
        "economics_bucket": _text(row.get("economics_bucket")),
        "liquidity_bucket": _text(row.get("liquidity_bucket")),
        "entry_limit": round(entry_limit, 6),
        "expiry": _text(row.get("expiry")),
        "opened_at": _text(row.get("recommendation_date")),
        "closed_at": observation_session_date.isoformat(),
        "entry_order_ids": logical_id,
        "realized_pnl": realized_pnl,
        "liquidation_value": round(liquidation, 6),
        "outcome_status": (
            "SCORED_EXACT_FIXED_HORIZON_SELECTED"
            if selected_for_expectancy
            else "SCORED_EXACT_FIXED_HORIZON_DIAGNOSTIC"
        ),
        "exact_evaluated": True,
        "contributes_to_expectancy": selected_for_expectancy,
        "source": "options_agent_shadow_outcomes",
        "quote_source": quote_source,
        "quote_snapshot_json": json.dumps(snapshots, sort_keys=True, separators=(",", ":")),
        "legs_json": _text(row.get("legs_json")),
        "recommendation_pipeline_version": _text(row.get("pipeline_version")),
        "selector_policy_id": _text(row.get("selector_policy_id")),
        "recommendation_code_git_sha": _text(row.get("code_git_sha")),
    }, ""


def _append_new_outcomes(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing_ids = {
                _text(json.loads(line).get("logical_recommendation_id"))
                for line in handle
                if line.strip()
            }
            handle.seek(0, os.SEEK_END)
            for row in rows:
                logical_id = _text(row.get("logical_recommendation_id"))
                if not logical_id or logical_id in existing_ids:
                    continue
                handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
                existing_ids.add(logical_id)
                appended += 1
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def _quote_for_symbol(quotes: Mapping[str, Any], symbol: str) -> Mapping[str, Any]:
    wanted = symbol.replace(" ", "").upper()
    for key, value in quotes.items():
        if _text(key).replace(" ", "").upper() == wanted and isinstance(value, Mapping):
            return value
    return {}


def _quote_fields(payload: Mapping[str, Any]) -> tuple[Optional[float], Optional[float], Optional[dt.datetime]]:
    body = payload.get("quote", payload)
    if not isinstance(body, Mapping):
        return None, None, None
    bid = _float(body.get("bidPrice", body.get("bid")))
    ask = _float(body.get("askPrice", body.get("ask")))
    raw_time = next(
        (
            body.get(key)
            for key in (
                "quoteTimeInLong",
                "quoteTime",
                "tradeTimeInLong",
                "tradeTime",
                "regularMarketTradeTimeInLong",
            )
            if body.get(key) not in (None, "")
        ),
        None,
    )
    return bid, ask, _timestamp(raw_time)


def _timestamp(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if abs(float(value)) > 10_000_000_000 else float(value)
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _aware_utc(parsed)


def _leg_symbols(value: Any) -> list[str]:
    try:
        legs = json.loads(_text(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(legs, list):
        return []
    return [_text(leg.get("occ_symbol")).upper() for leg in legs if isinstance(leg, Mapping) and _text(leg.get("occ_symbol"))]


def _aware_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _date(value: Any) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(_text(value)[:10])
    except ValueError:
        return None


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text
