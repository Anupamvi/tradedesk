"""Strictly read-only Schwab market-data boundary.

Cultra does not implement brokerage actions.  A concrete adapter may satisfy
``SchwabMarketDataProvider`` later, but this facade intentionally exposes only
quotes, option chains, and price history.  Tests use in-memory providers and
never touch the network or token file.
"""

from __future__ import annotations

import json
import math
import re
import stat
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from .domain import OptionType, parse_occ_symbol
from .gateway import (
    SchwabSafeTransportError,
    SchwabUrllibTransport,
    TransportResponse,
)


DEFAULT_SCHWAB_TOKEN_PATH = Path(
    "/Users/anuppamvi/tradedesk/tokens/schwab_token.json"
)
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9./^-]{0,14}$")


class SchwabBoundaryError(ValueError):
    """Raised for unsafe or malformed market-data requests/responses."""


class MarketDataOperation(str, Enum):
    QUOTES = "quotes"
    OPTION_CHAIN = "option_chain"
    PRICE_HISTORY = "price_history"


class SchwabAccessTokenSource(Protocol):
    def access_token(self, *, force_refresh: bool = False) -> str:
        ...


def _private_regular_file(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise SchwabBoundaryError("%s cannot be a symlink" % label)
        metadata = path.stat()
    except SchwabBoundaryError:
        raise
    except OSError as exc:
        raise SchwabBoundaryError("%s is unavailable" % label) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SchwabBoundaryError("%s must be a regular file" % label)
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise SchwabBoundaryError("%s permissions must be 0600" % label)


class SharedSchwabTokenSource:
    """Read the established token file without refreshing or mutating it."""

    __slots__ = ("_token_path",)

    def __init__(self, token_path: Path = DEFAULT_SCHWAB_TOKEN_PATH) -> None:
        resolved = Path(token_path).expanduser().resolve()
        if resolved != DEFAULT_SCHWAB_TOKEN_PATH.resolve():
            raise SchwabBoundaryError("only the established Schwab token is allowed")
        self._token_path = resolved

    def access_token(self, *, force_refresh: bool = False) -> str:
        if force_refresh:
            raise SchwabBoundaryError(
                "Schwab access token must be refreshed outside Cultra"
            )
        _private_regular_file(self._token_path, "Schwab token file")
        try:
            payload = json.loads(self._token_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SchwabBoundaryError("Schwab token file is invalid") from exc
        token = payload.get("token") if isinstance(payload, Mapping) else None
        if not isinstance(token, Mapping):
            raise SchwabBoundaryError("Schwab token payload is incomplete")
        access = str(token.get("access_token", ""))
        expires_at = float(token.get("expires_at", 0.0) or 0.0)
        if not access or expires_at <= time.time() + 60.0:
            raise SchwabBoundaryError(
                "Schwab access token is expired; refresh it outside Cultra"
            )
        return access


def _aware(timestamp: datetime, label: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise SchwabBoundaryError("%s must be timezone-aware" % label)


def normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if not _SYMBOL_RE.fullmatch(normalized):
        raise SchwabBoundaryError("invalid market symbol: %r" % symbol)
    return normalized


def _finite_nonnegative(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise SchwabBoundaryError("%s must be finite and nonnegative" % label)


@dataclass(frozen=True)
class TokenFileReference:
    """A path reference only; construction never opens or parses the token."""

    path: Path = DEFAULT_SCHWAB_TOKEN_PATH

    def __post_init__(self) -> None:
        resolved = Path(self.path).expanduser().resolve()
        allowed = DEFAULT_SCHWAB_TOKEN_PATH.resolve()
        if resolved != allowed:
            raise SchwabBoundaryError(
                "Cultra may reference only the established Schwab token path"
            )
        object.__setattr__(self, "path", resolved)

    def stat_status(self) -> Tuple[bool, str]:
        """Inspect metadata without reading token contents."""

        try:
            mode = self.path.stat().st_mode & 0o777
        except OSError as exc:
            return False, "token file unavailable: %s" % exc
        if mode & 0o077:
            return False, "token file permissions are too broad (%03o)" % mode
        return True, "token file exists with restricted permissions"


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    last: Optional[float]
    timestamp: datetime
    total_volume: Optional[int] = None
    close: Optional[float] = None
    net_percent_change: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _finite_nonnegative(self.bid, "bid")
        _finite_nonnegative(self.ask, "ask")
        if self.ask < self.bid:
            raise SchwabBoundaryError("ask cannot be below bid")
        if self.last is not None:
            _finite_nonnegative(self.last, "last")
        if self.total_volume is not None and self.total_volume < 0:
            raise SchwabBoundaryError("total_volume cannot be negative")
        for label, value in (
            ("close", self.close),
            ("week52_high", self.week52_high),
            ("week52_low", self.week52_low),
        ):
            if value is not None:
                _finite_nonnegative(value, label)
        if self.net_percent_change is not None and not math.isfinite(
            self.net_percent_change
        ):
            raise SchwabBoundaryError("net_percent_change must be finite")
        _aware(self.timestamp, "quote timestamp")


@dataclass(frozen=True)
class OptionQuote:
    occ_symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str
    bid: float
    ask: float
    timestamp: datetime
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    delta: Optional[float] = None

    def __post_init__(self) -> None:
        normalized_occ = str(self.occ_symbol).strip().upper()
        try:
            occ_root, occ_expiration, occ_type, occ_strike = parse_occ_symbol(
                normalized_occ
            )
        except ValueError as exc:
            raise SchwabBoundaryError(str(exc)) from exc
        object.__setattr__(self, "occ_symbol", normalized_occ)
        object.__setattr__(self, "underlying", normalize_symbol(self.underlying))
        if normalize_symbol(occ_root) != self.underlying:
            raise SchwabBoundaryError("OCC root does not match option underlying")
        normalized_type = self.option_type.strip().upper()
        if normalized_type not in {"CALL", "PUT"}:
            raise SchwabBoundaryError("option_type must be CALL or PUT")
        object.__setattr__(self, "option_type", normalized_type)
        _finite_nonnegative(self.strike, "strike")
        _finite_nonnegative(self.bid, "option bid")
        _finite_nonnegative(self.ask, "option ask")
        if self.ask < self.bid:
            raise SchwabBoundaryError("option ask cannot be below bid")
        if self.volume is not None and self.volume < 0:
            raise SchwabBoundaryError("volume cannot be negative")
        if self.open_interest is not None and self.open_interest < 0:
            raise SchwabBoundaryError("open interest cannot be negative")
        if self.delta is not None and (not math.isfinite(self.delta) or abs(self.delta) > 1):
            raise SchwabBoundaryError("delta must be finite and between -1 and 1")
        _aware(self.timestamp, "option quote timestamp")
        expected_type = OptionType.CALL if normalized_type == "CALL" else OptionType.PUT
        if occ_expiration != self.expiration:
            raise SchwabBoundaryError("OCC expiration does not match option quote")
        if occ_type is not expected_type:
            raise SchwabBoundaryError("OCC type does not match option quote")
        if not math.isclose(occ_strike, self.strike, rel_tol=0.0, abs_tol=0.0005):
            raise SchwabBoundaryError("OCC strike does not match option quote")


@dataclass(frozen=True)
class OptionChain:
    underlying: str
    underlying_quote: Quote
    contracts: Tuple[OptionQuote, ...]
    timestamp: datetime

    def __post_init__(self) -> None:
        normalized = normalize_symbol(self.underlying)
        object.__setattr__(self, "underlying", normalized)
        if self.underlying_quote.symbol != normalized:
            raise SchwabBoundaryError("chain underlying quote does not match request")
        _aware(self.timestamp, "chain timestamp")
        seen = set()
        for contract in self.contracts:
            if contract.underlying != normalized:
                raise SchwabBoundaryError("chain contains a different underlying")
            if contract.occ_symbol in seen:
                raise SchwabBoundaryError("chain contains duplicate OCC symbols")
            seen.add(contract.occ_symbol)


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _aware(self.timestamp, "bar timestamp")
        for label in ("open", "high", "low", "close"):
            _finite_nonnegative(getattr(self, label), label)
        if self.high < max(self.open, self.low, self.close):
            raise SchwabBoundaryError("bar high is inconsistent")
        if self.low > min(self.open, self.high, self.close):
            raise SchwabBoundaryError("bar low is inconsistent")
        if self.volume < 0:
            raise SchwabBoundaryError("bar volume cannot be negative")


class SchwabMarketDataProvider(Protocol):
    """Narrow provider protocol; deliberately contains no brokerage methods."""

    def fetch_quotes(self, symbols: Sequence[str]) -> Mapping[str, Quote]:
        ...

    def fetch_option_chain(
        self,
        symbol: str,
        *,
        from_date: date,
        to_date: date,
    ) -> OptionChain:
        ...

    def fetch_price_history(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
    ) -> Sequence[PriceBar]:
        ...


class SchwabMarketDataBoundary:
    """Validated facade over an injected, read-only market-data provider."""

    __slots__ = ("_provider", "_token_reference")

    def __init__(
        self,
        provider: SchwabMarketDataProvider,
        token_reference: Optional[TokenFileReference] = None,
    ) -> None:
        self._provider = provider
        self._token_reference = token_reference or TokenFileReference()

    @property
    def token_path(self) -> Path:
        """Return the approved path without reading it."""

        return self._token_reference.path

    def quotes(self, symbols: Sequence[str]) -> Mapping[str, Quote]:
        normalized = tuple(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols))
        if not normalized:
            raise SchwabBoundaryError("at least one quote symbol is required")
        result = dict(self._provider.fetch_quotes(normalized))
        unexpected = set(result).difference(normalized)
        if unexpected:
            raise SchwabBoundaryError("provider returned unrequested quote symbols")
        for key, quote in result.items():
            if normalize_symbol(key) != quote.symbol:
                raise SchwabBoundaryError("provider quote key and symbol disagree")
        return result

    def option_chain(
        self,
        symbol: str,
        *,
        from_date: date,
        to_date: date,
    ) -> OptionChain:
        normalized = normalize_symbol(symbol)
        if to_date < from_date:
            raise SchwabBoundaryError("option-chain date range is reversed")
        result = self._provider.fetch_option_chain(
            normalized,
            from_date=from_date,
            to_date=to_date,
        )
        if result.underlying != normalized:
            raise SchwabBoundaryError("provider returned the wrong option chain")
        for contract in result.contracts:
            if contract.expiration < from_date or contract.expiration > to_date:
                raise SchwabBoundaryError(
                    "provider returned a contract outside the requested expiration range"
                )
        return result

    def price_history(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
    ) -> Tuple[PriceBar, ...]:
        normalized = normalize_symbol(symbol)
        if end < start:
            raise SchwabBoundaryError("history date range is reversed")
        bars = tuple(
            self._provider.fetch_price_history(normalized, start=start, end=end)
        )
        previous: Optional[datetime] = None
        for bar in bars:
            if bar.symbol != normalized:
                raise SchwabBoundaryError("provider returned history for another symbol")
            if previous is not None and bar.timestamp <= previous:
                raise SchwabBoundaryError("history bars must be strictly chronological")
            previous = bar.timestamp
        return bars


def _number(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _integer(value: Any) -> Optional[int]:
    converted = _number(value)
    if converted is None or converted < 0:
        return None
    return int(converted)


def _timestamp_from_millis(value: Any, fallback: datetime) -> datetime:
    converted = _number(value)
    if converted is None or converted <= 0:
        return fallback
    return datetime.fromtimestamp(converted / 1000.0, timezone.utc)


def _canonical_occ(value: Any) -> str:
    return "".join(str(value).strip().upper().split())


class SchwabHTTPProvider:
    """Concrete read-only provider over the restricted Schwab transport."""

    __slots__ = ("_tokens", "_transport")

    def __init__(
        self,
        token_source: SchwabAccessTokenSource,
        transport: SchwabUrllibTransport,
    ) -> None:
        self._tokens = token_source
        self._transport = transport

    @classmethod
    def production(cls, *, timeout_seconds: float = 30.0) -> "SchwabHTTPProvider":
        transport = SchwabUrllibTransport(timeout_seconds=timeout_seconds)
        return cls(SharedSchwabTokenSource(), transport)

    def _authorized_get(
        self,
        path: str,
        query: Mapping[str, Any],
        *,
        max_response_bytes: int = 25_000_000,
    ) -> Tuple[Mapping[str, Any], TransportResponse]:
        token = self._tokens.access_token(force_refresh=False)
        try:
            response = self._transport.send(
                method="GET",
                path=path,
                query=query,
                headers={
                    "Accept": "application/json",
                    "Authorization": "Bearer " + token,
                    "User-Agent": "Cultra/1",
                },
                max_response_bytes=max_response_bytes,
            )
        except SchwabSafeTransportError as exc:
            raise SchwabBoundaryError("Schwab market-data request failed safely") from exc
        if response.status_code == 401:
            raise SchwabBoundaryError(
                "Schwab access token was rejected; refresh it outside Cultra"
            )
        if response.status_code < 200 or response.status_code >= 300:
            raise SchwabBoundaryError(
                "Schwab market data returned HTTP %d" % response.status_code
            )
        try:
            parsed = json.loads(response.body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SchwabBoundaryError("Schwab market data returned invalid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise SchwabBoundaryError("Schwab market data returned the wrong shape")
        return parsed, response

    def fetch_quotes(self, symbols: Sequence[str]) -> Mapping[str, Quote]:
        normalized = tuple(dict.fromkeys(normalize_symbol(item) for item in symbols))
        if not normalized or len(normalized) > 100:
            raise SchwabBoundaryError("Schwab quote batch must contain 1-100 symbols")
        payload, _response = self._authorized_get(
            "/marketdata/v1/quotes",
            {"symbols": ",".join(normalized), "fields": "quote,reference"},
            max_response_bytes=5_000_000,
        )
        now = datetime.now(timezone.utc)
        result: Dict[str, Quote] = {}
        for symbol in normalized:
            raw = payload.get(symbol)
            if not isinstance(raw, Mapping):
                continue
            quote = raw.get("quote")
            if not isinstance(quote, Mapping):
                continue
            bid = _number(quote.get("bidPrice"))
            ask = _number(quote.get("askPrice"))
            last = _number(quote.get("lastPrice"))
            if bid is None or ask is None:
                continue
            timestamp = _timestamp_from_millis(
                quote.get("quoteTime") or quote.get("tradeTime"), now
            )
            reference = raw.get("reference")
            if not isinstance(reference, Mapping):
                reference = {}
            result[symbol] = Quote(
                symbol,
                bid,
                ask,
                last,
                timestamp,
                total_volume=_integer(quote.get("totalVolume")),
                close=_number(quote.get("closePrice")),
                net_percent_change=_number(quote.get("netPercentChange")),
                week52_high=_number(
                    quote.get("52WeekHigh") or reference.get("52WeekHigh")
                ),
                week52_low=_number(
                    quote.get("52WeekLow") or reference.get("52WeekLow")
                ),
            )
        return result

    def fetch_option_chain(
        self,
        symbol: str,
        *,
        from_date: date,
        to_date: date,
    ) -> OptionChain:
        normalized = normalize_symbol(symbol)
        payload, _response = self._authorized_get(
            "/marketdata/v1/chains",
            {
                "symbol": normalized,
                "contractType": "ALL",
                "strategy": "SINGLE",
                "strikeCount": 100,
                "includeUnderlyingQuote": "true",
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
            },
            max_response_bytes=25_000_000,
        )
        now = datetime.now(timezone.utc)
        underlying_raw = payload.get("underlying")
        if not isinstance(underlying_raw, Mapping):
            underlying_raw = {}
        underlying_price = _number(payload.get("underlyingPrice"), default=0.0) or 0.0
        bid = _number(underlying_raw.get("bid"), default=underlying_price)
        ask = _number(underlying_raw.get("ask"), default=underlying_price)
        last = _number(underlying_raw.get("last"), default=underlying_price)
        if bid is None or ask is None or ask < bid:
            raise SchwabBoundaryError("Schwab chain has no valid underlying quote")
        chain_timestamp = _timestamp_from_millis(
            underlying_raw.get("quoteTime") or underlying_raw.get("tradeTime"), now
        )
        underlying_quote = Quote(normalized, bid, ask, last, chain_timestamp)
        contracts = []
        for map_name, option_type in (
            ("callExpDateMap", "CALL"),
            ("putExpDateMap", "PUT"),
        ):
            expiration_map = payload.get(map_name)
            if not isinstance(expiration_map, Mapping):
                continue
            for expiration_key, strike_map in expiration_map.items():
                try:
                    expiration = date.fromisoformat(str(expiration_key).split(":", 1)[0])
                except ValueError:
                    continue
                if not isinstance(strike_map, Mapping):
                    continue
                for _strike_key, values in strike_map.items():
                    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                        continue
                    for raw in values:
                        if not isinstance(raw, Mapping):
                            continue
                        strike = _number(raw.get("strikePrice"))
                        option_bid = _number(raw.get("bid"))
                        option_ask = _number(raw.get("ask"))
                        if strike is None or option_bid is None or option_ask is None:
                            continue
                        timestamp = _timestamp_from_millis(
                            raw.get("quoteTimeInLong") or raw.get("tradeTimeInLong"),
                            chain_timestamp,
                        )
                        try:
                            contracts.append(
                                OptionQuote(
                                    occ_symbol=_canonical_occ(raw.get("symbol", "")),
                                    underlying=normalized,
                                    expiration=expiration,
                                    strike=strike,
                                    option_type=option_type,
                                    bid=option_bid,
                                    ask=option_ask,
                                    timestamp=timestamp,
                                    volume=_integer(raw.get("totalVolume")),
                                    open_interest=_integer(raw.get("openInterest")),
                                    delta=_number(raw.get("delta")),
                                )
                            )
                        except SchwabBoundaryError:
                            continue
        return OptionChain(
            normalized,
            underlying_quote,
            tuple(sorted(contracts, key=lambda item: item.occ_symbol)),
            chain_timestamp,
        )

    def fetch_price_history(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
    ) -> Sequence[PriceBar]:
        normalized = normalize_symbol(symbol)
        start_ms = int(
            datetime.combine(start, datetime_time.min, tzinfo=timezone.utc).timestamp()
            * 1000
        )
        end_ms = int(
            datetime.combine(end, datetime_time.max, tzinfo=timezone.utc).timestamp()
            * 1000
        )
        payload, _response = self._authorized_get(
            "/marketdata/v1/pricehistory",
            {
                "symbol": normalized,
                "periodType": "year",
                "period": 1,
                "frequencyType": "daily",
                "frequency": 1,
                "startDate": start_ms,
                "endDate": end_ms,
                "needExtendedHoursData": "false",
                "needPreviousClose": "false",
            },
            max_response_bytes=5_000_000,
        )
        candles = payload.get("candles")
        if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
            return ()
        bars = []
        for raw in candles:
            if not isinstance(raw, Mapping):
                continue
            values = {
                key: _number(raw.get(key)) for key in ("open", "high", "low", "close")
            }
            timestamp = _number(raw.get("datetime"))
            volume = _integer(raw.get("volume"))
            if any(value is None for value in values.values()) or timestamp is None or volume is None:
                continue
            bars.append(
                PriceBar(
                    normalized,
                    datetime.fromtimestamp(timestamp / 1000.0, timezone.utc),
                    float(values["open"]),
                    float(values["high"]),
                    float(values["low"]),
                    float(values["close"]),
                    volume,
                )
            )
        return tuple(sorted(bars, key=lambda item: item.timestamp))


PUBLIC_MARKET_DATA_METHODS = frozenset(
    {"quotes", "option_chain", "price_history"}
)
