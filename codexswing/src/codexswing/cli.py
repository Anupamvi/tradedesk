"""Command line entrypoint for the ORATS-first CodexSwing pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from codexswing.backtest.labels import parse_orats_daily_rows
from codexswing.backtest.orats_option_replay import (
    build_replay_samples,
    required_chain_slices,
    run_orats_option_replay,
)
from codexswing.clock import iso_utc, parse_timestamp, session_close_utc, utc_now
from codexswing.config import (
    DEFAULT_ENV_FILE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHWAB_ENV_FILE,
    CodexSwingConfig,
)
from codexswing.report.idea_html import render_idea_html
from codexswing.research.current_ideas import build_current_ideas
from codexswing.research.universe import UniverseCandidate, discover_optionable_universe
from codexswing.schemas.source import SourceRecord, canonical_json
from codexswing.secrets import SecretBundle
from codexswing.sources.events import (
    GDELTClient,
    GDELTError,
    GoogleNewsRSSClient,
    GoogleNewsRSSError,
)
from codexswing.sources.orats import ORATSClient, ORATSError, ORATSHTTPError
from codexswing.sources.schwab import SchwabError, SchwabReadOnlyClient
from codexswing.sources.schwab_auth import SchwabOAuthRefresher, SchwabTokenRefreshError
from codexswing.store.immutable import (
    ContentAddressedStore,
    audit_store,
    read_batch,
    sha256_file,
    write_once_bytes,
    write_once_json,
)
from codexswing.store.manifest import RunManifest, code_tree_sha256


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _progress(message: str) -> None:
    print("[codexswing] {}".format(message), file=sys.stderr, flush=True)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--schwab-env-file", type=Path, default=DEFAULT_SCHWAB_ENV_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout", type=int, default=30)


def _config(args: argparse.Namespace) -> CodexSwingConfig:
    return CodexSwingConfig(
        env_file=args.env_file,
        schwab_env_file=args.schwab_env_file,
        output_root=args.output_root,
        request_timeout_seconds=args.timeout,
    ).validated()


def _secrets(config: CodexSwingConfig, *, refresh_schwab: bool = False) -> SecretBundle:
    bundle = SecretBundle.from_files(config.env_file, config.schwab_env_file)
    if refresh_schwab and bundle.schwab_access_token_expired():
        _progress("Schwab access token is stale; refreshing the authorized token cache")
        SchwabOAuthRefresher(config.request_timeout_seconds).refresh(bundle)
        bundle = SecretBundle.from_files(config.env_file, config.schwab_env_file)
    return bundle


def _store_batch(
    store: ContentAddressedStore,
    records: Sequence[SourceRecord],
) -> Optional[Path]:
    if not records:
        return None
    artifact = store.put_batch(records)
    return artifact.path


def _doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    secrets = _secrets(config)
    payload: Dict[str, Any] = {
        "status": "READY_FOR_RESEARCH" if secrets.orats_token else "BLOCKED",
        "architecture": "ORATS_FIRST_SCHWAB_EXECUTION_TRUTH",
        "configured_sources": {
            "ORATS": bool(secrets.orats_token),
            "Schwab": bool(secrets.schwab_access_token),
            "public_context": True,
        },
        "output_root": str(config.output_root),
        "broker_mutation_surface": False,
        "order_submission": "USER_ONLY",
    }
    if args.online:
        if not secrets.orats_token:
            raise RuntimeError("ORATS token is unavailable")
        payload["orats_probe"] = ORATSClient(
            secrets.orats_token, config.request_timeout_seconds
        ).probe()
        try:
            refreshed = _secrets(config, refresh_schwab=True)
            payload["schwab_probe"] = SchwabReadOnlyClient(
                refreshed.schwab_access_token, config.request_timeout_seconds
            ).probe()
        except (SchwabError, SchwabTokenRefreshError) as exc:
            payload["schwab_probe"] = {"status": "unavailable", "reason": str(exc)}
    _print(payload)
    return 0 if secrets.orats_token else 2


def _discover(args: argparse.Namespace) -> int:
    config = _config(args)
    secrets = _secrets(config)
    client = ORATSClient(secrets.orats_token, config.request_timeout_seconds)
    rows = client.fetch_rows("cores", {})
    candidates, funnel = discover_optionable_universe(
        rows,
        limit=args.limit,
        minimum_average_option_volume=args.minimum_option_volume,
        minimum_option_open_interest=args.minimum_open_interest,
    )
    payload = {
        "status": "DISCOVERY_COMPLETE",
        "source": "ORATS cores full universe",
        "raw_underlying_count": len(rows),
        "funnel": funnel,
        "candidates": [item.to_dict() for item in candidates],
        "broker_order_authorized": False,
    }
    if args.write:
        records = client.rows_to_records("cores", rows)
        store = ContentAddressedStore(config.output_root, secrets.values())
        batch = _store_batch(store, records)
        rendered = (canonical_json(payload) + "\n").encode("utf-8")
        digest = hashlib.sha256(rendered).hexdigest()
        path = config.output_root / "universe" / str(rows[0].get("tradeDate") or "current") / "{}.json".format(digest)
        write_once_bytes(path, rendered, secrets.values())
        payload["artifact_path"] = str(path)
        payload["source_batch_path"] = str(batch) if batch else None
    _print(payload)
    return 0


def _parse_params(values: Sequence[str]) -> Mapping[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--param values must be key=value")
        key, raw = value.split("=", 1)
        if not key.strip():
            raise ValueError("empty ORATS parameter key")
        result[key.strip()] = raw.strip()
    return result


def _fetch_orats(args: argparse.Namespace) -> int:
    config = _config(args)
    secrets = _secrets(config)
    client = ORATSClient(secrets.orats_token, config.request_timeout_seconds)
    params = dict(_parse_params(args.param))
    if args.tickers:
        rows = client.fetch_tickers(args.endpoint, args.tickers.split(","), params)
    else:
        rows = client.fetch_rows(args.endpoint, params)
    records = client.rows_to_records(args.endpoint, rows)
    payload: Dict[str, Any] = {
        "status": "FETCH_COMPLETE",
        "endpoint": args.endpoint,
        "row_count": len(rows),
        "session_dates": sorted({record.session_date for record in records}),
    }
    if args.write and records:
        store = ContentAddressedStore(config.output_root, secrets.values())
        payload["batch_path"] = str(store.put_batch(records).path)
    _print(payload)
    return 0


def _account_hashes(payload: Any) -> Sequence[str]:
    rows = payload if isinstance(payload, list) else []
    return tuple(
        str(item.get("hashValue") or "")
        for item in rows
        if isinstance(item, Mapping) and item.get("hashValue")
    )


def _fetch_portfolio(
    client: SchwabReadOnlyClient,
    as_of_date: str,
) -> Optional[SourceRecord]:
    try:
        hashes = _account_hashes(client.account_numbers())
        if not hashes:
            return None
        account_payloads = []
        order_payloads = []
        start = "{}T00:00:00.000Z".format(
            (date.fromisoformat(as_of_date) - timedelta(days=60)).isoformat()
        )
        end = "{}T23:59:59.999Z".format(
            (date.fromisoformat(as_of_date) + timedelta(days=1)).isoformat()
        )
        for account_hash in hashes:
            payload = client.positions(account_hash)
            if isinstance(payload, Mapping):
                account_payloads.append(payload)
            order_payloads.append(client.working_orders(account_hash, start, end))
        return client.portfolio_record(account_payloads, order_payloads)
    except (SchwabError, ValueError) as exc:
        _progress("Portfolio snapshot unavailable: {}".format(exc))
        return None


def _fetch_schwab(args: argparse.Namespace) -> int:
    config = _config(args)
    secrets = _secrets(config, refresh_schwab=True)
    client = SchwabReadOnlyClient(secrets.schwab_access_token, config.request_timeout_seconds)
    tickers = tuple(value.strip().upper() for value in args.tickers.split(",") if value.strip())
    quote_records = client.quote_records(client.quotes(tickers))
    chain_records = []
    for ticker in tickers:
        payload = client.option_chain(
            ticker,
            contractType="ALL",
            strategy="SINGLE",
            includeUnderlyingQuote="TRUE",
            fromDate=args.from_date,
            toDate=args.to_date,
        )
        chain_records.append(client.option_chain_record(ticker, payload))
    portfolio = _fetch_portfolio(client, args.as_of_date) if args.include_portfolio else None
    records = tuple(quote_records) + tuple(chain_records) + ((portfolio,) if portfolio else ())
    payload: Dict[str, Any] = {
        "status": "SCHWAB_READ_COMPLETE",
        "quote_count": len(quote_records),
        "chain_count": len(chain_records),
        "portfolio_snapshot": portfolio is not None,
        "broker_mutation_surface": False,
    }
    if args.write:
        store = ContentAddressedStore(config.output_root, secrets.values())
        payload["record_paths"] = [str(store.put(record)) for record in records]
    _print(payload)
    return 0


def _select_finalists(
    candidates: Sequence[UniverseCandidate], limit: int
) -> Tuple[UniverseCandidate, ...]:
    etf_cap = max(2, int(round(limit * 0.35)))
    short_floor = min(max(2, int(round(limit * 0.25))), limit // 2)
    selected = list(
        item for item in candidates if item.direction == "SHORT"
    )[:short_floor]
    selected_tickers = {item.ticker for item in selected}
    etfs = sum(item.asset_class == "ETF" for item in selected)
    for item in candidates:
        if item.ticker in selected_tickers:
            continue
        if item.asset_class == "ETF" and etfs >= etf_cap:
            continue
        selected.append(item)
        selected_tickers.add(item.ticker)
        if item.asset_class == "ETF":
            etfs += 1
        if len(selected) == limit:
            break
    if len(selected) < limit:
        selected.extend(
            item for item in candidates if item.ticker not in selected_tickers
        )
    return tuple(selected[:limit])


def _fetch_public_context(
    tickers: Sequence[str], as_of_date: str, timeout: int
) -> Tuple[Mapping[str, Any], Sequence[SourceRecord]]:
    end = datetime.combine(date.fromisoformat(as_of_date), time(23, 59), tzinfo=timezone.utc)
    start = end - timedelta(days=3)
    query = "({}) OR (geopolitics markets)".format(" OR ".join(tickers[:8]))
    failures = []
    records: Sequence[SourceRecord] = ()
    provider = "GDELT"
    try:
        client = GDELTClient(timeout)
        articles = client.fetch_articles(query, start, end, max_records=75)
        records = client.articles_to_records(articles, ingested_at=datetime.now(timezone.utc))
    except (GDELTError, ValueError) as exc:
        failures.append(str(exc))
    if not records:
        provider = "Google News RSS fallback"
        try:
            fallback = GoogleNewsRSSClient(timeout)
            articles = fallback.fetch_articles(query, start, end, max_records=75)
            records = fallback.articles_to_records(
                articles, ingested_at=datetime.now(timezone.utc)
            )
        except (GoogleNewsRSSError, ValueError) as exc:
            failures.append(str(exc))
    if not records:
        return {
            "items": [],
            "status": "SHADOW_SOURCE_UNAVAILABLE",
            "reason": "; ".join(failures) or "public context sources returned no in-window articles",
            "providers_attempted": ["GDELT", "Google News RSS"],
            "numeric_vote": False,
        }, ()
    items = []
    for record in records[:30]:
        items.append(
            {
                "title": record.payload.get("title"),
                "url": record.payload.get("url"),
                "source_name": record.payload.get("source") or record.payload.get("domain"),
                "source_country": record.payload.get("sourcecountry"),
                "published_at_utc": record.published_at_utc,
                "provider": provider,
            }
        )
    return {
        "items": items,
        "status": "SHADOW_ONLY_FALLBACK" if failures else "SHADOW_ONLY",
        "provider": provider,
        "upstream_failures": failures,
        "numeric_vote": False,
        "reason": "Time-aligned public context is displayed but cannot alter rank until incremental ablation passes.",
    }, records


def _cached_hist_strikes(
    root: Path,
    requirements: Mapping[str, Sequence[str]],
) -> Tuple[List[SourceRecord], List[SourceRecord], Mapping[str, Sequence[str]]]:
    cached_chains: List[SourceRecord] = []
    cached_coverage: List[SourceRecord] = []
    missing: Dict[str, Sequence[str]] = {}
    negative_cache_cutoff = utc_now() - timedelta(hours=24)
    for trade_date, tickers in requirements.items():
        found = set()
        for path in sorted((root / "batches" / "orats_hist_strikes" / trade_date).glob("*.jsonl.gz")):
            try:
                records = read_batch(path)
            except (OSError, ValueError):
                continue
            for record in records:
                ticker = str(record.payload.get("ticker") or "").upper()
                if ticker in tickers:
                    cached_chains.append(record)
                    found.add(ticker)
        for path in sorted(
            (root / "records" / "orats_hist_strikes_unavailable" / trade_date).glob("*.json")
        ):
            try:
                record = SourceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if parse_timestamp(record.ingested_at_utc) < negative_cache_cutoff:
                    continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            ticker = str(record.payload.get("ticker") or "").upper()
            if ticker in tickers:
                cached_coverage.append(record)
                found.add(ticker)
        absent = tuple(ticker for ticker in tickers if ticker not in found)
        if absent:
            missing[trade_date] = absent
    return cached_chains, cached_coverage, missing


def _hist_strikes_unavailable_record(
    ticker: str,
    trade_date: str,
    http_status: int,
) -> SourceRecord:
    checked_at = utc_now()
    checked_at_text = iso_utc(checked_at)
    reason_code = "HTTP_404_NO_ARCHIVED_SLICE" if http_status == 404 else "EMPTY_DATA_ARRAY"
    return SourceRecord(
        source="orats_hist_strikes_unavailable",
        source_id="hist/strikes:{}:{}:{}:{}".format(
            ticker.upper(), trade_date, http_status, checked_at_text
        ),
        session_date=trade_date,
        event_time_utc=iso_utc(session_close_utc(trade_date)),
        published_at_utc=checked_at_text,
        first_seen_at_utc=checked_at_text,
        available_at_utc=checked_at_text,
        ingested_at_utc=checked_at_text,
        source_uri="https://api.orats.io/datav2/hist/strikes",
        revision=reason_code,
        payload={
            "ticker": ticker.upper(),
            "requestedTradeDate": trade_date,
            "endpoint": "hist/strikes",
            "httpStatus": http_status,
            "availability": "NO_ARCHIVED_CHAIN_RETURNED",
            "reasonCode": reason_code,
        },
    )


def _fetch_hist_strikes(
    client: ORATSClient,
    store: ContentAddressedStore,
    requirements: Mapping[str, Sequence[str]],
    workers: int,
) -> Tuple[Sequence[SourceRecord], Sequence[SourceRecord]]:
    cached_chains, cached_coverage, missing = _cached_hist_strikes(
        store.root, requirements
    )
    if not missing:
        _progress("Historical strike cache covers all required entry/exit slices")
        return tuple(cached_chains), tuple(cached_coverage)
    _progress(
        "Fetching {} missing ORATS historical chain dates with {} workers".format(
            len(missing), workers
        )
    )

    def fetch(
        item: Tuple[str, Sequence[str]],
    ) -> Tuple[str, Sequence[SourceRecord], Sequence[SourceRecord]]:
        trade_date, tickers = item
        chain_records: List[SourceRecord] = []
        unavailable_records: List[SourceRecord] = []
        # Fetch one ticker at a time. A missing ticker must not turn a valid
        # same-date ticker into a combined-request 404.
        for ticker in tickers:
            try:
                rows = client.fetch_tickers(
                    "hist/strikes",
                    [ticker],
                    {"tradeDate": trade_date, "dte": "14,60", "delta": ".03,.97"},
                )
            except ORATSHTTPError as exc:
                if exc.endpoint != "hist/strikes" or exc.status_code != 404:
                    raise
                unavailable_records.append(
                    _hist_strikes_unavailable_record(ticker, trade_date, exc.status_code)
                )
                continue
            if not rows:
                unavailable_records.append(
                    _hist_strikes_unavailable_record(ticker, trade_date, 200)
                )
                continue
            chain_records.extend(client.rows_to_records("hist/strikes", rows))
        return trade_date, tuple(chain_records), tuple(unavailable_records)

    fetched_chains: List[SourceRecord] = []
    fetched_coverage: List[SourceRecord] = []
    unavailable_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in missing.items()}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            trade_date, records, unavailable_records = future.result()
            if records:
                store.put_batch(records)
                fetched_chains.extend(records)
            for unavailable_record in unavailable_records:
                store.put(unavailable_record)
                fetched_coverage.append(unavailable_record)
            unavailable_count += len(unavailable_records)
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                _progress("Historical chain slices {}/{}".format(completed, len(futures)))
    if unavailable_count:
        _progress(
            "Recorded {} unavailable historical chain slices; they remain replay rejections".format(
                unavailable_count
            )
        )
    return (
        tuple(cached_chains + fetched_chains),
        tuple(cached_coverage + fetched_coverage),
    )


def _write_report(
    payload: Mapping[str, Any],
    root: Path,
    secrets: SecretBundle,
) -> Tuple[Path, Path]:
    rendered_json = (canonical_json(payload) + "\n").encode("utf-8")
    digest = hashlib.sha256(rendered_json).hexdigest()
    report_root = root / "idea_reports" / str(payload.get("as_of_date"))
    json_path = report_root / "{}.json".format(digest)
    html_path = report_root / "{}.html".format(digest)
    write_once_bytes(json_path, rendered_json, secrets.values())
    write_once_bytes(
        html_path,
        render_idea_html(payload).encode("utf-8"),
        secrets.values(),
    )
    return json_path, html_path


def _run_daily(args: argparse.Namespace) -> int:
    config = _config(args)
    secrets = _secrets(config, refresh_schwab=True)
    orats = ORATSClient(secrets.orats_token, config.request_timeout_seconds)
    schwab = SchwabReadOnlyClient(secrets.schwab_access_token, config.request_timeout_seconds)
    store = ContentAddressedStore(config.output_root, secrets.values())
    all_inputs: List[SourceRecord] = []
    batch_paths: List[Path] = []

    _progress("1/7 ORATS full-universe discovery")
    core_rows = orats.fetch_rows("cores", {})
    core_records_all = orats.rows_to_records("cores", core_rows)
    batch = _store_batch(store, core_records_all)
    if batch:
        batch_paths.append(batch)
    broad, funnel = discover_optionable_universe(core_rows, limit=args.discovery_limit)
    finalists = _select_finalists(broad, args.finalists)
    finalist_tickers = tuple(item.ticker for item in finalists)
    _progress("Selected finalists: {}".format(", ".join(finalist_tickers)))
    core_records = tuple(record for record in core_records_all if str(record.payload.get("ticker") or "").upper() in finalist_tickers)
    all_inputs.extend(core_records_all)

    _progress("2/7 ORATS adjusted history, IV history, and current exact strikes")
    daily_rows = orats.fetch_tickers("hist/dailies", finalist_tickers)
    iv_rows = orats.fetch_tickers("hist/ivrank", finalist_tickers)
    strike_rows = orats.fetch_tickers("strikes", finalist_tickers)
    daily_records = orats.rows_to_records("hist/dailies", daily_rows)
    iv_records = orats.rows_to_records("hist/ivrank", iv_rows)
    strike_records = orats.rows_to_records("strikes", strike_rows)
    for records in (daily_records, iv_records, strike_records):
        batch = _store_batch(store, records)
        if batch:
            batch_paths.append(batch)
        all_inputs.extend(records)

    _progress("3/7 Schwab current quotes, chains, balances, positions, and working orders")
    quote_records = schwab.quote_records(schwab.quotes(finalist_tickers))
    chain_records = []
    from_date = (date.fromisoformat(args.date) + timedelta(days=21)).isoformat()
    to_date = (date.fromisoformat(args.date) + timedelta(days=60)).isoformat()
    for ticker in finalist_tickers:
        chain = schwab.option_chain(
            ticker,
            contractType="ALL",
            strategy="SINGLE",
            includeUnderlyingQuote="TRUE",
            fromDate=from_date,
            toDate=to_date,
        )
        chain_records.append(schwab.option_chain_record(ticker, chain))
    portfolio_record = _fetch_portfolio(schwab, args.date)
    for record in tuple(quote_records) + tuple(chain_records) + ((portfolio_record,) if portfolio_record else ()):
        store.put(record)
        all_inputs.append(record)

    _progress("4/7 Public trend/geopolitical context (shadow-only)")
    if args.no_public_context:
        context, context_records = {"items": [], "status": "DISABLED", "numeric_vote": False}, ()
    else:
        context, context_records = _fetch_public_context(finalist_tickers, args.date, config.request_timeout_seconds)
        if context_records:
            _store_batch(store, context_records)
            all_inputs.extend(context_records)

    _progress("5/7 Provisional six-strategy current-contract screen")
    provisional = build_current_ideas(
        as_of_date=args.date,
        universe=finalists,
        core_records=core_records,
        daily_records=daily_records,
        ivrank_records=iv_records,
        schwab_quote_records=quote_records,
        schwab_chain_records=chain_records,
        orats_strike_records=strike_records,
        portfolio_record=portfolio_record,
        context=context,
    )

    replay: Optional[Mapping[str, Any]] = None
    if args.backtest_top > 0 and provisional.get("ideas"):
        _progress("6/7 ORATS exact-chain historical replay")
        replay_seed = dict(provisional)
        replay_seed["ideas"] = list(provisional["ideas"][: args.backtest_top])
        bars = parse_orats_daily_rows(
            [record.payload for record in daily_records],
            tickers=[item["ticker"] for item in replay_seed["ideas"]],
        )
        samples = build_replay_samples(replay_seed, bars)
        requirements = required_chain_slices(samples)
        historical_chain_records, historical_coverage_records = _fetch_hist_strikes(
            orats,
            store,
            requirements,
            args.backtest_workers,
        )
        all_inputs.extend(historical_chain_records)
        all_inputs.extend(historical_coverage_records)
        spec = {
            "schema": "ORATS_EXACT_DIRECTIONAL_CURRENT_REGIME_V3",
            "as_of_date": args.date,
            "tickers": [item["ticker"] for item in replay_seed["ideas"]],
            "strategies": [
                "LONG_CALL",
                "LONG_PUT",
                "BULL_CALL_DEBIT",
                "BEAR_PUT_DEBIT",
                "BULL_PUT_CREDIT",
                "BEAR_CALL_CREDIT",
            ],
            "fill": (
                "single leg: 75% bid-to-ask entry, exact bid exit, $1.30 round trip; "
                "vertical: 66% package entry, natural exit, $2.60 round trip"
            ),
            "current_regime_analog_count": 250,
            "split": "50/20/30 chronological; at least 20 closed holdout trades required",
            "tactical_policy": (
                "30 closed/15 effective; positive train-validation-holdout expectancy; "
                "holdout PF>=1.20; bootstrap lower no worse than 5% current defined risk; "
                "one contract <=0.05% NAV/$500"
            ),
            "multiple_testing": "all evaluated ticker-strategy groups reported; no winner-only suppression",
            "spot_scale": "adjusted signal path; contemporaneous unadjusted close for option-chain alignment",
            "code_tree_sha256": code_tree_sha256(),
        }
        spec_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
        replay = dict(run_orats_option_replay(
            current_ideas=replay_seed,
            samples=samples,
            chain_records=historical_chain_records,
            spec_sha256=spec_hash,
        ))
        replay["coverage"] = {
            "required_ticker_date_slices": sum(len(values) for values in requirements.values()),
            "unavailable_ticker_date_slices": len(historical_coverage_records),
            "unavailable_slices_are_rejections": True,
        }
        replay_path = config.output_root / "backtests" / args.date / "{}.json".format(spec_hash)
        write_once_json(replay_path, replay, secrets.values())
    else:
        _progress("6/7 Historical replay disabled for this run")

    _progress("7/7 Final promotion pass and immutable report")
    final_payload = build_current_ideas(
        as_of_date=args.date,
        universe=finalists,
        core_records=core_records,
        daily_records=daily_records,
        ivrank_records=iv_records,
        schwab_quote_records=quote_records,
        schwab_chain_records=chain_records,
        orats_strike_records=strike_records,
        option_replay=replay,
        portfolio_record=portfolio_record,
        context=context,
    )
    final_payload = dict(final_payload)
    final_payload["universe_funnel"] = funnel
    final_payload["raw_orats_underlying_count"] = len(core_rows)
    final_payload["historical_replay_summary"] = (
        {
            "available": True,
            "sample_count": replay.get("sample_count"),
            "group_count": replay.get("group_count"),
            "holdout_pass_count": replay.get("holdout_pass_count"),
            "evaluated_hypothesis_count": replay.get("evaluated_hypothesis_count"),
            "multiple_testing_adjusted": replay.get("multiple_testing_adjusted"),
            "fill_model": replay.get("fill_model"),
            "split_policy": replay.get("split_policy"),
            "coverage": replay.get("coverage"),
        }
        if replay
        else {"available": False}
    )
    json_path, html_path = _write_report(final_payload, config.output_root, secrets)
    manifest = RunManifest.create(
        mode="run_daily_v4",
        configuration={
            "as_of_date": args.date,
            "discovery_limit": args.discovery_limit,
            "finalists": args.finalists,
            "backtest_top": args.backtest_top,
            "backtest_workers": args.backtest_workers,
            "public_context_numeric_vote": False,
            "broker_order_authorized": False,
        },
        input_records=all_inputs,
        input_file_hashes={str(path): sha256_file(path) for path in batch_paths},
        output_paths=(json_path, html_path),
        warnings=(
            "No profitability guarantee",
            "Manual submission only",
            "Public context is shadow-only",
        ),
    )
    manifest_path = manifest.write(config.output_root, secrets.values())
    _print(
        {
            "status": final_payload["status"],
            "manual_ready_trade_count": final_payload["manual_ready_trade_count"],
            "tactical_ready_trade_count": final_payload["tactical_ready_trade_count"],
            "actionable_trade_count": final_payload["actionable_trade_count"],
            "top_candidate": final_payload["top_candidate"],
            "screened_underlyings": len(core_rows),
            "finalists": list(finalist_tickers),
            "report_json": str(json_path),
            "report_html": str(html_path),
            "manifest": str(manifest_path),
            "broker_order_authorized": False,
            "broker_order_submitted": False,
        }
    )
    return 0


def _audit(args: argparse.Namespace) -> int:
    config = _config(args)
    secrets = _secrets(config)
    result = audit_store(config.output_root, secrets.values())
    _print(result.public_dict())
    return 0 if result.valid else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codexswing",
        description="ORATS-first stock/options swing research; every broker order remains user-submitted.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check credentials and read-only source readiness")
    _common(doctor)
    doctor.add_argument("--online", action="store_true")
    doctor.set_defaults(handler=_doctor)

    discover = commands.add_parser("discover-universe", help="rank the complete liquid ORATS universe")
    _common(discover)
    discover.add_argument("--limit", type=int, default=250)
    discover.add_argument("--minimum-option-volume", type=float, default=5_000)
    discover.add_argument("--minimum-open-interest", type=float, default=10_000)
    discover.add_argument("--write", action="store_true")
    discover.set_defaults(handler=_discover)

    orats = commands.add_parser("fetch-orats", help="fetch an allowed ORATS research endpoint")
    _common(orats)
    orats.add_argument("--endpoint", required=True)
    orats.add_argument("--tickers", default="")
    orats.add_argument("--param", action="append", default=[])
    orats.add_argument("--write", action="store_true")
    orats.set_defaults(handler=_fetch_orats)

    schwab = commands.add_parser("fetch-schwab", help="fetch exact current quotes/chains and optional portfolio snapshot")
    _common(schwab)
    schwab.add_argument("--tickers", required=True)
    schwab.add_argument("--as-of-date", required=True)
    schwab.add_argument("--from-date", required=True)
    schwab.add_argument("--to-date", required=True)
    schwab.add_argument("--include-portfolio", action="store_true")
    schwab.add_argument("--write", action="store_true")
    schwab.set_defaults(handler=_fetch_schwab)

    daily = commands.add_parser("run-daily", help="run discovery, current screen, ORATS replay, and promotion gates")
    _common(daily)
    daily.add_argument("--date", required=True, help="completed market session YYYY-MM-DD")
    daily.add_argument("--discovery-limit", type=int, default=250)
    daily.add_argument("--finalists", type=int, default=8)
    daily.add_argument("--backtest-top", type=int, default=1)
    daily.add_argument("--backtest-workers", type=int, default=6)
    daily.add_argument("--no-public-context", action="store_true")
    daily.set_defaults(handler=_run_daily)

    audit = commands.add_parser("audit-store", help="verify immutable artifacts and secret exclusion")
    _common(audit)
    audit.set_defaults(handler=_audit)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        try:
            config = _config(args)
            message = _secrets(config).redact(str(exc))
        except Exception:
            message = str(exc)
        print("codexswing error: {}".format(message), file=sys.stderr)
        return 2
