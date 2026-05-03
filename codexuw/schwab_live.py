from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float


def _iter_contracts(exp_map: dict[str, Any], right: str):
    for exp_key, strike_map in (exp_map or {}).items():
        expiry_text = str(exp_key).split(":")[0]
        try:
            expiry = dt.datetime.strptime(expiry_text[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        for strike_key, contracts in (strike_map or {}).items():
            strike = safe_float(strike_key)
            for contract in contracts or []:
                yield {
                    "expiry": expiry,
                    "right": right,
                    "strike": safe_float(contract.get("strikePrice"), strike),
                    "symbol": contract.get("symbol", ""),
                    "bid": safe_float(contract.get("bid")),
                    "ask": safe_float(contract.get("ask")),
                    "mark": safe_float(contract.get("mark")),
                    "delta": safe_float(contract.get("delta")),
                    "iv": safe_float(contract.get("volatility")),
                    "open_interest": safe_float(contract.get("openInterest"), 0.0),
                    "volume": safe_float(contract.get("totalVolume"), 0.0),
                }


def chain_to_contracts(chain: dict[str, Any]) -> pd.DataFrame:
    rows = list(_iter_contracts(chain.get("callExpDateMap", {}), "C"))
    rows.extend(_iter_contracts(chain.get("putExpDateMap", {}), "P"))
    return pd.DataFrame(rows)


def chain_spot(chain: dict[str, Any]) -> float:
    underlying = chain.get("underlying", {}) if isinstance(chain, dict) else {}
    for value in [
        chain.get("underlyingPrice"),
        underlying.get("mark"),
        underlying.get("last"),
        underlying.get("lastPrice"),
    ]:
        number = safe_float(value)
        if math.isfinite(number) and number > 0:
            return number
    return math.nan


def option_mid(row: pd.Series) -> float:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    mark = safe_float(row.get("mark"))
    if math.isfinite(bid) and math.isfinite(ask) and bid >= 0 and ask > 0:
        return (bid + ask) / 2.0
    return mark


def price_width_bucket(spot: float) -> float:
    if spot < 60:
        return 2.5
    if spot < 180:
        return 5.0
    if spot < 500:
        return 5.0
    return 10.0


def _same_expiry_contracts(contracts: pd.DataFrame, expiry: dt.date, right: str) -> pd.DataFrame:
    if contracts.empty:
        return contracts
    out = contracts[(contracts["expiry"] == expiry) & (contracts["right"] == right)].copy()
    for col in ["strike", "bid", "ask", "mark", "delta", "open_interest", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["mid"] = out.apply(option_mid, axis=1)
    out["quote_width"] = out["ask"] - out["bid"]
    out["quote_width_pct"] = out["quote_width"] / out["mid"].where(out["mid"].abs() > 0)
    return out.sort_values("strike")


def find_best_credit_spread(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
) -> dict[str, Any]:
    right = "P" if direction == "Bull Put" else "C"
    chain = _same_expiry_contracts(contracts, expiry, right)
    if chain.empty:
        return {"live_status": "missing_expiry_or_right", "live_blocker": f"no {right} contracts for {expiry}"}

    width = float(preferred_width or price_width_bucket(spot))
    rows: list[dict[str, Any]] = []
    strikes = sorted(float(x) for x in chain["strike"].dropna().unique())
    by_strike = {float(r["strike"]): r for _, r in chain.iterrows()}

    for _, short in chain.iterrows():
        short_strike = safe_float(short.get("strike"))
        if not math.isfinite(short_strike):
            continue
        if direction == "Bull Put":
            if short_strike >= spot:
                continue
            long_candidates = [s for s in strikes if s < short_strike and abs((short_strike - s) - width) <= max(0.51, width * 0.35)]
            distance_pct = (spot - short_strike) / spot
            delta_abs = abs(safe_float(short.get("delta")))
        else:
            if short_strike <= spot:
                continue
            long_candidates = [s for s in strikes if s > short_strike and abs((s - short_strike) - width) <= max(0.51, width * 0.35)]
            distance_pct = (short_strike - spot) / spot
            delta_abs = abs(safe_float(short.get("delta")))
        if not long_candidates:
            continue
        long_strike = min(long_candidates, key=lambda s: abs(abs(s - short_strike) - width))
        long = by_strike[long_strike]
        actual_width = abs(long_strike - short_strike)
        short_bid = safe_float(short.get("bid"))
        short_ask = safe_float(short.get("ask"))
        long_bid = safe_float(long.get("bid"))
        long_ask = safe_float(long.get("ask"))
        short_mid = option_mid(short)
        long_mid = option_mid(long)
        natural_credit = short_bid - long_ask
        mid_credit = short_mid - long_mid
        realistic_credit = max(natural_credit, mid_credit * 0.90)
        if not math.isfinite(realistic_credit) or realistic_credit <= 0:
            continue
        credit_pct = realistic_credit / actual_width if actual_width > 0 else math.nan
        pop = 1.0 - delta_abs if math.isfinite(delta_abs) and delta_abs > 0 else math.nan
        short_liq = safe_float(short.get("open_interest"), 0.0) + safe_float(short.get("volume"), 0.0)
        long_liq = safe_float(long.get("open_interest"), 0.0) + safe_float(long.get("volume"), 0.0)
        short_qwp = safe_float(short.get("quote_width_pct"))
        long_qwp = safe_float(long.get("quote_width_pct"))
        quote_penalty = max(short_qwp if math.isfinite(short_qwp) else 0.0, long_qwp if math.isfinite(long_qwp) else 0.0)
        if distance_pct < 0.015:
            continue
        if math.isfinite(delta_abs) and not (0.08 <= delta_abs <= 0.35):
            continue
        rows.append(
            {
                "live_status": "PASS",
                "short_leg": short.get("symbol", ""),
                "long_leg": long.get("symbol", ""),
                "short_strike": short_strike,
                "long_strike": long_strike,
                "spread_width": actual_width,
                "credit": round(realistic_credit, 2),
                "mid_credit": round(mid_credit, 2),
                "natural_credit": round(natural_credit, 2),
                "credit_pct_width": credit_pct,
                "sell_leg_bid": short_bid,
                "sell_leg_ask": short_ask,
                "sell_leg_mid": short_mid,
                "buy_leg_bid": long_bid,
                "buy_leg_ask": long_ask,
                "buy_leg_mid": long_mid,
                "pop_delta_proxy": pop,
                "short_delta": safe_float(short.get("delta")),
                "distance_pct": distance_pct,
                "short_oi": safe_float(short.get("open_interest"), 0.0),
                "short_volume": safe_float(short.get("volume"), 0.0),
                "long_oi": safe_float(long.get("open_interest"), 0.0),
                "long_volume": safe_float(long.get("volume"), 0.0),
                "quote_width_pct": quote_penalty,
                "liq_score": min(short_liq, long_liq),
            }
        )

    if not rows:
        return {"live_status": "no_realistic_spread", "live_blocker": "no OTM spread with positive realistic credit/delta/liquidity"}
    df = pd.DataFrame(rows)
    delta_target_penalty = (df["short_delta"].abs().fillna(0.22) - 0.22).abs().clip(upper=0.20)
    df["_rank"] = (
        df["credit_pct_width"].clip(upper=0.45) * 4.0
        + df["distance_pct"].clip(upper=0.18) * 24.0
        + df["pop_delta_proxy"].fillna(0.55)
        + (df["liq_score"].clip(upper=5000) / 5000.0)
        - df["quote_width_pct"].fillna(0.0).clip(upper=2.0)
        - delta_target_penalty * 2.0
    )
    best = df.sort_values("_rank", ascending=False).iloc[0].drop(labels=["_rank"]).to_dict()
    return best


class SchwabChainValidator:
    def __init__(self, out_dir: Path, *, strike_count: int = 80) -> None:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

        self.out_dir = out_dir
        self.strike_count = strike_count
        self.service = SchwabLiveDataService(SchwabAuthConfig.from_env(load_dotenv_file=True), interactive_login=False)
        self.chains: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}

    def get_chain(self, ticker: str, from_date: dt.date, to_date: dt.date) -> dict[str, Any] | None:
        symbol = str(ticker).upper().strip()
        if symbol in self.chains:
            return self.chains[symbol]
        try:
            chain = self.service.get_option_chain(
                symbol,
                strike_count=self.strike_count,
                include_underlying_quote=True,
                from_date=from_date,
                to_date=to_date,
            )
            self.chains[symbol] = chain
            return chain
        except Exception as exc:
            dated_error = str(exc)
        try:
            chain = self.service.get_option_chain(
                symbol,
                strike_count=self.strike_count,
                include_underlying_quote=True,
            )
            self.chains[symbol] = chain
            return chain
        except Exception as exc:
            self.errors[symbol] = f"dated chain failed: {dated_error}; undated fallback failed: {exc}"
            return None

    def save(self) -> None:
        chain_dir = self.out_dir / "schwab_chains"
        chain_dir.mkdir(parents=True, exist_ok=True)
        for ticker, chain in self.chains.items():
            (chain_dir / f"{ticker}.json").write_text(json.dumps(chain, indent=2, sort_keys=True), encoding="utf-8")
        if self.errors:
            (chain_dir / "errors.json").write_text(json.dumps(self.errors, indent=2, sort_keys=True), encoding="utf-8")
