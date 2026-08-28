"""Fail-closed Schwab client exposing GET-only research reads."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from codexswing.clock import NEW_YORK, iso_utc, utc_now
from codexswing.schemas.source import SourceRecord, canonical_json


MARKETDATA_BASE_URL = "https://api.schwabapi.com/marketdata/v1"
TRADER_BASE_URL = "https://api.schwabapi.com/trader/v1"
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9./-]{0,31}$")


class SchwabError(RuntimeError):
    pass


class SchwabCredentialUnavailable(SchwabError):
    pass


Transport = Callable[[str, str, Mapping[str, str]], Any]


class SchwabReadOnlyClient:
    def __init__(
        self,
        access_token: Optional[str],
        timeout_seconds: int = 30,
        transport: Optional[Transport] = None,
    ) -> None:
        if not access_token:
            raise SchwabCredentialUnavailable(
                "SCHWAB_ACCESS_TOKEN is unavailable in the authorized .env; Schwab reads are disabled"
            )
        self._access_token = access_token
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_transport

    def _default_transport(self, base_url: str, path: str, params: Mapping[str, str]) -> Any:
        query = urllib.parse.urlencode(dict(params))
        url = "{}{}{}".format(base_url, path, "?{}".format(query) if query else "")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer {}".format(self._access_token),
                "User-Agent": "codexswing/0.2.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise SchwabError("Schwab GET {} returned HTTP {}".format(path, exc.code)) from None
        except urllib.error.URLError as exc:
            raise SchwabError("Schwab GET {} failed: {}".format(path, exc.reason)) from None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SchwabError("Schwab GET {} returned invalid JSON".format(path)) from None

    @staticmethod
    def _symbols(symbols: Iterable[str]) -> Tuple[str, ...]:
        normalized = []
        seen = set()
        for raw in symbols:
            symbol = raw.strip().upper()
            if not SYMBOL_RE.fullmatch(symbol):
                raise ValueError("invalid symbol: {}".format(raw))
            if symbol not in seen:
                normalized.append(symbol)
                seen.add(symbol)
        if not normalized:
            raise ValueError("at least one symbol is required")
        return tuple(normalized)

    def quotes(self, symbols: Iterable[str]) -> Any:
        normalized = self._symbols(symbols)
        return self._transport(MARKETDATA_BASE_URL, "/quotes", {"symbols": ",".join(normalized)})

    def option_chain(self, symbol: str, **filters: str) -> Any:
        normalized = self._symbols([symbol])[0]
        params: Dict[str, str] = {"symbol": normalized}
        params.update({str(key): str(value) for key, value in filters.items() if value is not None})
        return self._transport(MARKETDATA_BASE_URL, "/chains", params)

    def account_numbers(self) -> Any:
        return self._transport(TRADER_BASE_URL, "/accounts/accountNumbers", {})

    def positions(self, account_hash: str) -> Any:
        if not account_hash or not re.fullmatch(r"[A-Za-z0-9_-]{4,128}", account_hash):
            raise ValueError("invalid Schwab account hash")
        return self._transport(
            TRADER_BASE_URL,
            "/accounts/{}".format(account_hash),
            {"fields": "positions"},
        )

    def working_orders(
        self,
        account_hash: str,
        from_entered_time: str,
        to_entered_time: str,
    ) -> Any:
        """Read working orders so the research desk cannot suggest conflicts."""

        if not account_hash or not re.fullmatch(r"[A-Za-z0-9_-]{4,128}", account_hash):
            raise ValueError("invalid Schwab account hash")
        return self._transport(
            TRADER_BASE_URL,
            "/accounts/{}/orders".format(account_hash),
            {
                "fromEnteredTime": from_entered_time,
                "toEnteredTime": to_entered_time,
                "status": "WORKING",
            },
        )

    def probe(self, symbol: str = "SPY") -> Dict[str, Any]:
        payload = self.quotes([symbol])
        row_count = len(payload) if isinstance(payload, Mapping) else 0
        return {"status": "available", "endpoint": "quotes", "symbol": symbol, "row_count": row_count}

    @staticmethod
    def _epoch_time(value: Any) -> Optional[datetime]:
        try:
            epoch = float(value)
        except (TypeError, ValueError):
            return None
        if epoch > 10_000_000_000:
            epoch /= 1000.0
        try:
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    @classmethod
    def _quote_event_time(cls, payload: Mapping[str, Any]) -> Optional[datetime]:
        quote = payload.get("quote")
        quote_values = quote if isinstance(quote, Mapping) else {}
        candidates = (
            quote_values.get("quoteTime"),
            quote_values.get("tradeTime"),
            quote_values.get("regularMarketTradeTime"),
        )
        for value in candidates:
            parsed = cls._epoch_time(value)
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _chain_event_time(cls, payload: Mapping[str, Any]) -> Optional[datetime]:
        """Derive chain session from embedded Schwab quotes, never wall-clock ingestion."""

        option_times = []
        for map_name in ("callExpDateMap", "putExpDateMap"):
            expiration_map = payload.get(map_name)
            if not isinstance(expiration_map, Mapping):
                continue
            for strike_map in expiration_map.values():
                if not isinstance(strike_map, Mapping):
                    continue
                for contracts in strike_map.values():
                    if not isinstance(contracts, list):
                        continue
                    for contract in contracts:
                        if not isinstance(contract, Mapping):
                            continue
                        parsed = cls._epoch_time(
                            contract.get("quoteTimeInLong") or contract.get("tradeTimeInLong")
                        )
                        if parsed is not None:
                            option_times.append(parsed)
        if option_times:
            return max(option_times)
        underlying = payload.get("underlying")
        underlying_values = underlying if isinstance(underlying, Mapping) else {}
        return cls._epoch_time(
            underlying_values.get("quoteTime") or underlying_values.get("tradeTime")
        )

    def quote_records(
        self,
        payload: Mapping[str, Any],
        ingested_at: Optional[datetime] = None,
    ) -> Tuple[SourceRecord, ...]:
        ingestion_time = ingested_at or utc_now()
        records = []
        for raw_symbol, raw_row in sorted(payload.items()):
            symbol = self._symbols([str(raw_symbol)])[0]
            if not isinstance(raw_row, Mapping):
                raise SchwabError("Schwab quote row for {} is not an object".format(symbol))
            event_time = self._quote_event_time(raw_row)
            session_time = event_time or ingestion_time
            session_date = session_time.astimezone(NEW_YORK).date().isoformat()
            digest = hashlib.sha256(canonical_json(raw_row).encode("utf-8")).hexdigest()[:20]
            records.append(
                SourceRecord(
                    source="schwab_quotes",
                    source_id="quotes:{}:{}:{}".format(symbol, session_date, digest),
                    session_date=session_date,
                    event_time_utc=iso_utc(event_time) if event_time else None,
                    published_at_utc=iso_utc(ingestion_time),
                    first_seen_at_utc=iso_utc(ingestion_time),
                    available_at_utc=iso_utc(ingestion_time),
                    ingested_at_utc=iso_utc(ingestion_time),
                    source_uri="{}/quotes".format(MARKETDATA_BASE_URL),
                    revision=digest,
                    payload=dict(raw_row),
                )
            )
        return tuple(records)

    def option_chain_record(
        self,
        symbol: str,
        payload: Mapping[str, Any],
        ingested_at: Optional[datetime] = None,
    ) -> SourceRecord:
        normalized = self._symbols([symbol])[0]
        if not isinstance(payload, Mapping):
            raise SchwabError("Schwab option chain payload is not an object")
        ingestion_time = ingested_at or utc_now()
        event_time = self._chain_event_time(payload)
        session_time = event_time or ingestion_time
        session_date = session_time.astimezone(NEW_YORK).date().isoformat()
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:20]
        return SourceRecord(
            source="schwab_option_chain",
            source_id="chain:{}:{}:{}".format(normalized, session_date, digest),
            session_date=session_date,
            event_time_utc=iso_utc(event_time) if event_time else None,
            published_at_utc=iso_utc(ingestion_time),
            first_seen_at_utc=iso_utc(ingestion_time),
            available_at_utc=iso_utc(ingestion_time),
            ingested_at_utc=iso_utc(ingestion_time),
            source_uri="{}/chains".format(MARKETDATA_BASE_URL),
            revision=digest,
            payload=dict(payload),
        )

    def portfolio_record(
        self,
        account_payloads: Sequence[Mapping[str, Any]],
        working_order_payloads: Sequence[Any],
        ingested_at: Optional[datetime] = None,
    ) -> SourceRecord:
        """Create one account-id-free snapshot for portfolio gating.

        Account numbers and hashes are intentionally discarded. The stored
        record contains only balances, positions, and working-order legs needed
        to detect risk and duplicate exposure.
        """

        ingestion_time = ingested_at or utc_now()
        accounts = []
        for raw in account_payloads:
            securities = raw.get("securitiesAccount") if isinstance(raw, Mapping) else None
            values = securities if isinstance(securities, Mapping) else raw
            if not isinstance(values, Mapping):
                continue
            balances = values.get("currentBalances")
            positions = values.get("positions")
            clean_positions = []
            if isinstance(positions, list):
                for position in positions:
                    if not isinstance(position, Mapping):
                        continue
                    instrument = position.get("instrument")
                    instrument_values = instrument if isinstance(instrument, Mapping) else {}
                    clean_positions.append(
                        {
                            "symbol": instrument_values.get("symbol"),
                            "assetType": instrument_values.get("assetType"),
                            "underlyingSymbol": instrument_values.get("underlyingSymbol"),
                            "longQuantity": position.get("longQuantity"),
                            "shortQuantity": position.get("shortQuantity"),
                            "marketValue": position.get("marketValue"),
                            "averagePrice": position.get("averagePrice"),
                        }
                    )
            balance_values = balances if isinstance(balances, Mapping) else {}
            accounts.append(
                {
                    "balances": {
                        key: balance_values.get(key)
                        for key in (
                            "cashAvailableForTrading",
                            "cashBalance",
                            "buyingPower",
                            "liquidationValue",
                            "availableFunds",
                        )
                    },
                    "positions": clean_positions,
                }
            )
        clean_orders = []
        for payload in working_order_payloads:
            rows = payload if isinstance(payload, list) else []
            for order in rows:
                if not isinstance(order, Mapping):
                    continue
                legs = []
                for leg in order.get("orderLegCollection") or ():
                    instrument = leg.get("instrument") if isinstance(leg, Mapping) else None
                    values = instrument if isinstance(instrument, Mapping) else {}
                    legs.append(
                        {
                            "symbol": values.get("symbol"),
                            "underlyingSymbol": values.get("underlyingSymbol"),
                            "assetType": values.get("assetType"),
                            "instruction": leg.get("instruction") if isinstance(leg, Mapping) else None,
                            "quantity": leg.get("quantity") if isinstance(leg, Mapping) else None,
                        }
                    )
                clean_orders.append(
                    {
                        "status": order.get("status"),
                        "orderType": order.get("orderType"),
                        "price": order.get("price"),
                        "legs": legs,
                    }
                )
        payload = {"accounts": accounts, "workingOrders": clean_orders}
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:20]
        session_date = ingestion_time.astimezone(NEW_YORK).date().isoformat()
        return SourceRecord(
            source="schwab_portfolio",
            source_id="portfolio:{}:{}".format(session_date, digest),
            session_date=session_date,
            published_at_utc=iso_utc(ingestion_time),
            first_seen_at_utc=iso_utc(ingestion_time),
            available_at_utc=iso_utc(ingestion_time),
            ingested_at_utc=iso_utc(ingestion_time),
            source_uri="{}/accounts".format(TRADER_BASE_URL),
            revision=digest,
            payload=payload,
        )
