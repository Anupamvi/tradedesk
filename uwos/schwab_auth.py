from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from authlib.oauth2.rfc6749.errors import MismatchingStateException
from dotenv import load_dotenv
from schwab.auth import (
    RedirectServerExitedError,
    RedirectTimeoutError,
    client_from_manual_flow,
    client_from_token_file,
    easy_client,
)

DEFAULT_CALLBACK_URL = "https://127.0.0.1"
DEFAULT_TOKEN_PATH = "./tokens/schwab_token.json"
DEFAULT_SYMBOLS = ["AAPL", "SPY"]
COMPACT_OCC_RE = re.compile(r"^([A-Z\.]{1,6})(\d{6})([CP])(\d{8})$")
SCHWAB_OCC_RE = re.compile(r"^([A-Z\. ]{6})(\d{6})([CP])(\d{8})$")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_refresh_token_error(exc: Exception) -> bool:
    text = str(exc)
    return ("refresh_token_authentication_error" in text) or ("unsupported_token_type" in text)


def _to_date(value: Any) -> Optional[dt.date]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in your shell or in a .env file."
        )
    return value


def normalize_symbols(raw_symbols: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for item in raw_symbols:
        for token in str(item).split(","):
            symbol = token.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                cleaned.append(symbol)
    return cleaned


def parse_symbols(
    symbols_csv: str,
    symbols: Iterable[str],
    fallback: Sequence[str],
    env_name: str = "SCHWAB_SYMBOLS",
) -> List[str]:
    if symbols_csv:
        raw = [symbols_csv]
    elif symbols:
        raw = list(symbols)
    else:
        raw = [os.environ.get(env_name, "")] if os.environ.get(env_name, "") else list(fallback)

    cleaned = normalize_symbols(raw)
    if not cleaned:
        raise RuntimeError("No symbols provided.")
    return cleaned


def compact_occ_to_schwab_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        raise ValueError("Empty option symbol")

    candidate = text.upper()
    if SCHWAB_OCC_RE.match(candidate):
        return candidate

    compact = candidate.replace(" ", "")
    match = COMPACT_OCC_RE.match(compact)
    if not match:
        raise ValueError(f"Unsupported OCC symbol format: {symbol}")
    root, yymmdd, right, strike8 = match.groups()
    return f"{root:<6}{yymmdd}{right}{strike8}"


def schwab_occ_to_compact_symbol(symbol: str) -> str:
    text = str(symbol or "").upper()
    match = SCHWAB_OCC_RE.match(text)
    if not match:
        return text.replace(" ", "")
    root, yymmdd, right, strike8 = match.groups()
    return f"{root.strip()}{yymmdd}{right}{strike8}"


def occ_underlying_symbol(symbol: str) -> Optional[str]:
    text = str(symbol or "").upper()
    match = SCHWAB_OCC_RE.match(text)
    if match:
        return match.group(1).strip() or None
    match = COMPACT_OCC_RE.match(text.replace(" ", ""))
    if match:
        return match.group(1).strip() or None
    return None


def _iter_contracts(exp_map: Dict[str, Dict[str, List[Dict[str, Any]]]]):
    for exp_key, strike_map in exp_map.items():
        expiry = exp_key.split(":")[0]
        for strike_key, contracts in strike_map.items():
            strike = _safe_float(strike_key)
            for contract in contracts:
                yield expiry, strike, contract


def _contract_brief(
    contract: Dict[str, Any], expiry: Optional[str] = None, strike: Optional[float] = None
) -> Dict[str, Any]:
    return {
        "symbol": contract.get("symbol"),
        "expiry": expiry or str(contract.get("expirationDate", ""))[:10],
        "strike": _safe_float(contract.get("strikePrice")) or strike,
        "bid": _safe_float(contract.get("bid")),
        "ask": _safe_float(contract.get("ask")),
        "last": _safe_float(contract.get("last")),
        "mark": _safe_float(contract.get("mark")),
        "delta": _safe_float(contract.get("delta")),
        "gamma": _safe_float(contract.get("gamma")),
        "theta": _safe_float(contract.get("theta")),
        "vega": _safe_float(contract.get("vega")),
        "open_interest": _safe_float(contract.get("openInterest")),
        "volume": _safe_float(contract.get("totalVolume")),
        "in_the_money": contract.get("inTheMoney"),
    }


def extract_quote_fields(quote_payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    body = quote_payload.get("quote", quote_payload)
    last_price = _safe_float(body.get("lastPrice", body.get("mark")))
    bid_price = _safe_float(body.get("bidPrice"))
    ask_price = _safe_float(body.get("askPrice"))
    return last_price, bid_price, ask_price


@dataclass(frozen=True)
class SchwabAuthConfig:
    api_key: str
    app_secret: str
    callback_url: str = DEFAULT_CALLBACK_URL
    token_path: str = DEFAULT_TOKEN_PATH

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "SchwabAuthConfig":
        if load_dotenv_file:
            load_dotenv()
        return cls(
            api_key=require_env("SCHWAB_API_KEY"),
            app_secret=require_env("SCHWAB_APP_SECRET"),
            callback_url=os.environ.get("SCHWAB_CALLBACK_URL", DEFAULT_CALLBACK_URL),
            token_path=os.environ.get("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH),
        )


class SchwabLiveDataService:
    def __init__(
        self,
        config: SchwabAuthConfig,
        manual_auth: bool = False,
        interactive_login: bool = True,
    ) -> None:
        self.config = config
        self.manual_auth = manual_auth
        self.interactive_login = interactive_login
        self._client = None
        self.auth_mode = "unknown"

    @property
    def token_path(self) -> Path:
        token_path = Path(self.config.token_path).expanduser().resolve()
        token_path.parent.mkdir(parents=True, exist_ok=True)
        return token_path

    def _load_client_from_token(self):
        return client_from_token_file(
            str(self.token_path),
            self.config.api_key,
            self.config.app_secret,
        )

    def _manual_client_from_login(self):
        try:
            return client_from_manual_flow(
                self.config.api_key,
                self.config.app_secret,
                self.config.callback_url,
                str(self.token_path),
            )
        except MismatchingStateException as exc:
            raise RuntimeError(
                "Manual OAuth failed due to mismatched state. "
                "Re-run and paste the callback URL from that same login attempt."
            ) from exc

    def connect(self):
        if self._client is not None:
            return self._client

        parsed = urlparse(self.config.callback_url)

        if self.manual_auth:
            self._client = self._manual_client_from_login()
            self.auth_mode = "manual_login"
            return self._client

        has_token_file = self.token_path.is_file()

        if has_token_file:
            try:
                self._client = self._load_client_from_token()
                self.auth_mode = "token_file"
                return self._client
            except Exception:
                self._client = None

        if parsed.port in (None, 443):
            self._client = self._manual_client_from_login()
            self.auth_mode = "manual_login"
            return self._client

        try:
            self._client = easy_client(
                api_key=self.config.api_key,
                app_secret=self.config.app_secret,
                callback_url=self.config.callback_url,
                token_path=str(self.token_path),
                interactive=self.interactive_login,
            )
            self.auth_mode = "browser_login"
            return self._client
        except (RedirectServerExitedError, RedirectTimeoutError, ValueError):
            self._client = self._manual_client_from_login()
            self.auth_mode = "manual_login"
            return self._client

    def get_account_hash(self, account_index: int = 0) -> str:
        """Return the account hash for the given account index (default: first account)."""
        client = self.connect()
        response = client.get_account_numbers()
        response.raise_for_status()
        accounts = response.json()
        if not accounts:
            raise RuntimeError("No accounts found for this Schwab token.")
        if account_index >= len(accounts):
            raise RuntimeError(
                f"Account index {account_index} out of range "
                f"(found {len(accounts)} account(s))."
            )
        return accounts[account_index]["hashValue"]

    def get_account_positions(self, account_index: int = 0) -> Dict[str, Any]:
        """Fetch current account positions and balances from Schwab."""
        from schwab.client import Client

        account_hash = self.get_account_hash(account_index)
        client = self.connect()
        try:
            response = client.get_account(
                account_hash, fields=[Client.Account.Fields.POSITIONS]
            )
        except Exception as exc:
            if _is_refresh_token_error(exc):
                raise RuntimeError(
                    "Schwab token refresh failed (stale/revoked refresh token). "
                    "Re-auth once with: python -m uwos.schwab_quotes --manual-auth "
                    "--symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2"
                ) from exc
            raise
        response.raise_for_status()
        data = response.json()

        acct = data.get("securitiesAccount", data)
        balances = acct.get("currentBalances", {})
        raw_positions = acct.get("positions", [])

        positions = []
        for pos in raw_positions:
            instrument = pos.get("instrument", {})
            positions.append({
                "symbol": instrument.get("symbol", ""),
                "asset_type": instrument.get("assetType", ""),
                "underlying": instrument.get("underlyingSymbol", ""),
                "put_call": instrument.get("putCall", ""),
                "qty": pos.get("longQuantity", 0) - pos.get("shortQuantity", 0),
                "short_qty": pos.get("shortQuantity", 0),
                "long_qty": pos.get("longQuantity", 0),
                "avg_cost": pos.get("averagePrice"),
                "market_value": pos.get("marketValue"),
                "day_pnl": pos.get("currentDayProfitLoss"),
                "day_pnl_pct": pos.get("currentDayProfitLossPercentage"),
            })

        return {
            "balances": {
                "total_value": _safe_float(balances.get("liquidationValue")),
                "cash": _safe_float(balances.get("cashBalance")),
            },
            "positions": positions,
        }

    def get_transactions(
        self,
        account_hash: str,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        transaction_types: Optional[Any] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch transactions for a single date window (max 60 days per Schwab API)."""
        client = self.connect()
        kwargs: Dict[str, Any] = {}
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        if transaction_types is not None:
            kwargs["transaction_types"] = transaction_types
        if symbol is not None:
            kwargs["symbol"] = symbol
        try:
            response = client.get_transactions(account_hash, **kwargs)
        except Exception as exc:
            if _is_refresh_token_error(exc):
                raise RuntimeError(
                    "Schwab token refresh failed (stale/revoked refresh token). "
                    "Re-auth once with: python -m uwos.schwab_quotes --manual-auth --symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2"
                ) from exc
            raise
        response.raise_for_status()
        return response.json()

    def get_trade_history(
        self,
        days: int = 90,
        account_index: int = 0,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch TRADE transactions for the last *days* days, chunking into 60-day windows."""
        from schwab.client import Client

        account_hash = self.get_account_hash(account_index)
        today = dt.date.today()
        start = today - dt.timedelta(days=days)
        trade_type = Client.Transactions.TransactionType.TRADE

        all_txns: List[Dict[str, Any]] = []
        chunk_start = start
        while chunk_start < today:
            chunk_end = min(chunk_start + dt.timedelta(days=59), today)
            txns = self.get_transactions(
                account_hash=account_hash,
                start_date=chunk_start,
                end_date=chunk_end,
                transaction_types=[trade_type],
                symbol=symbol,
            )
            all_txns.extend(txns)
            chunk_start = chunk_end + dt.timedelta(days=1)

        all_txns.sort(key=lambda t: t.get("transactionDate", ""), reverse=True)
        return all_txns

    def get_quotes(self, symbols: Sequence[str]) -> Dict[str, Any]:
        sym_list = normalize_symbols(symbols)
        if not sym_list:
            raise RuntimeError("No quote symbols provided.")

        client = self.connect()
        try:
            response = client.get_quotes(sym_list)
        except Exception as exc:
            if _is_refresh_token_error(exc):
                raise RuntimeError(
                    "Schwab token refresh failed (stale/revoked refresh token). "
                    "Re-auth once with: python -m uwos.schwab_quotes --manual-auth --symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2"
                ) from exc
            raise
        response.raise_for_status()
        return response.json()

    def get_option_chain(
        self,
        symbol: str,
        strike_count: Optional[int] = 8,
        include_underlying_quote: bool = True,
        from_date: Any = None,
        to_date: Any = None,
    ) -> Dict[str, Any]:
        client = self.connect()
        kwargs: Dict[str, Any] = {
            "include_underlying_quote": include_underlying_quote,
        }
        if strike_count is not None:
            kwargs["strike_count"] = int(strike_count)
        parsed_from = _to_date(from_date)
        parsed_to = _to_date(to_date)
        if parsed_from is not None:
            kwargs["from_date"] = parsed_from
        if parsed_to is not None:
            kwargs["to_date"] = parsed_to

        try:
            response = client.get_option_chain(symbol, **kwargs)
        except Exception as exc:
            if _is_refresh_token_error(exc):
                raise RuntimeError(
                    "Schwab token refresh failed (stale/revoked refresh token). "
                    "Re-auth once with: python -m uwos.schwab_quotes --manual-auth --symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2"
                ) from exc
            raise
        response.raise_for_status()
        return response.json()

    def get_option_chains(
        self,
        symbols: Sequence[str],
        strike_count: Optional[int] = 8,
        include_underlying_quote: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for symbol in normalize_symbols(symbols):
            out[symbol] = self.get_option_chain(
                symbol=symbol,
                strike_count=strike_count,
                include_underlying_quote=include_underlying_quote,
            )
        return out

    def summarize_option_chain(self, symbol: str, chain_payload: Dict[str, Any]) -> Dict[str, Any]:
        call_contracts = list(_iter_contracts(chain_payload.get("callExpDateMap", {})))
        put_contracts = list(_iter_contracts(chain_payload.get("putExpDateMap", {})))
        underlying = chain_payload.get("underlying", {})
        underlying_price = (
            _safe_float(chain_payload.get("underlyingPrice"))
            or _safe_float(underlying.get("mark"))
            or _safe_float(underlying.get("last"))
        )

        sample_call = (
            _contract_brief(call_contracts[0][2], call_contracts[0][0], call_contracts[0][1])
            if call_contracts
            else None
        )
        sample_put = (
            _contract_brief(put_contracts[0][2], put_contracts[0][0], put_contracts[0][1])
            if put_contracts
            else None
        )

        atm_call = None
        atm_put = None
        if underlying_price is not None:
            if call_contracts:
                nearest_call = min(
                    call_contracts,
                    key=lambda x: abs((x[1] if x[1] is not None else underlying_price) - underlying_price),
                )
                atm_call = _contract_brief(nearest_call[2], nearest_call[0], nearest_call[1])
            if put_contracts:
                nearest_put = min(
                    put_contracts,
                    key=lambda x: abs((x[1] if x[1] is not None else underlying_price) - underlying_price),
                )
                atm_put = _contract_brief(nearest_put[2], nearest_put[0], nearest_put[1])

        expiries = sorted(
            set(expiry for expiry, _, _ in call_contracts) | set(expiry for expiry, _, _ in put_contracts)
        )

        return {
            "symbol": symbol,
            "status": chain_payload.get("status", "UNKNOWN"),
            "underlying_price": underlying_price,
            "calls": len(call_contracts),
            "puts": len(put_contracts),
            "expiries": expiries,
            "sample_call": sample_call,
            "sample_put": sample_put,
            "atm_call": atm_call,
            "atm_put": atm_put,
        }

    def build_trading_query_context(
        self,
        quotes_payload: Dict[str, Any],
        chains_payload: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        quote_context: Dict[str, Any] = {}
        for symbol, payload in quotes_payload.items():
            body = payload.get("quote", payload)
            quote_context[symbol] = {
                "last": _safe_float(body.get("lastPrice", body.get("mark"))),
                "bid": _safe_float(body.get("bidPrice")),
                "ask": _safe_float(body.get("askPrice")),
                "mark": _safe_float(body.get("mark")),
                "change": _safe_float(body.get("netChange")),
                "percent_change": _safe_float(body.get("netPercentChangeInDouble")),
                "volume": _safe_float(body.get("totalVolume")),
                "quote_time": body.get("quoteTime"),
                "trade_time": body.get("tradeTime"),
            }

        chain_context = {
            symbol: self.summarize_option_chain(symbol, payload)
            for symbol, payload in chains_payload.items()
        }

        return {
            "asof_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "quotes": quote_context,
            "option_chains": chain_context,
        }

    def snapshot(
        self,
        symbols: Sequence[str],
        chain_symbols: Optional[Sequence[str]] = None,
        strike_count: int = 8,
    ) -> Dict[str, Any]:
        quote_symbols = normalize_symbols(symbols)
        option_symbols = normalize_symbols(chain_symbols or quote_symbols)

        quotes_payload = self.get_quotes(quote_symbols)
        chain_payloads = self.get_option_chains(
            symbols=option_symbols,
            strike_count=strike_count,
            include_underlying_quote=True,
        )
        chain_summary = {
            symbol: self.summarize_option_chain(symbol, payload)
            for symbol, payload in chain_payloads.items()
        }
        query_context = self.build_trading_query_context(quotes_payload, chain_payloads)

        return {
            "asof_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "auth_mode": self.auth_mode,
            "quote_symbols": quote_symbols,
            "chain_symbols": option_symbols,
            "quotes": quotes_payload,
            "option_chains": chain_payloads,
            "option_chain_summary": chain_summary,
            "trading_query_context": query_context,
        }

    def save_snapshot(self, snapshot: Dict[str, Any], out_dir: Path) -> None:
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "quotes.json").write_text(
            json.dumps(snapshot.get("quotes", {}), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out_dir / "trading_query_context.json").write_text(
            json.dumps(snapshot.get("trading_query_context", {}), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        chains = snapshot.get("option_chains", {})
        for symbol, payload in chains.items():
            (out_dir / f"option_chain_{symbol}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
