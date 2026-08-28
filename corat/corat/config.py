"""Configuration and universe loading."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@dataclass(frozen=True)
class UniverseItem:
    ticker: str
    name: str
    sector: str
    theme: str
    kind: str
    sector_etf: str


DISCOVERY_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _sector_etf(sector: str, supplied: str) -> str:
    candidate = str(supplied or "").strip().upper()
    if DISCOVERY_TICKER_RE.fullmatch(candidate) and candidate not in {"N/A", "NA", "NULL"}:
        return candidate
    lowered = str(sector or "").lower()
    mapping = (
        ("technology", "XLK"),
        ("comm", "XLC"),
        ("cyclical", "XLY"),
        ("discretion", "XLY"),
        ("defensive", "XLP"),
        ("staple", "XLP"),
        ("energy", "XLE"),
        ("financial", "XLF"),
        ("health", "XLV"),
        ("industrial", "XLI"),
        ("material", "XLB"),
        ("real estate", "XLRE"),
        ("utilit", "XLU"),
    )
    return next((etf for token, etf in mapping if token in lowered), "SPY")


def discover_universe(
    config: Mapping[str, Any],
    core_rows: Iterable[Mapping[str, Any]],
    configured_items: Sequence[UniverseItem],
) -> tuple[List[UniverseItem], Dict[str, Any]]:
    """Build a broad, traceable liquid equity universe from ORATS cores.

    The configured benchmarks/theme ETFs are preserved. Equity slots combine
    market capitalization, stock dollar volume, and average option volume so a
    pure mega-cap list does not erase liquid emerging mid-cap names.
    """

    rows = list(core_rows)
    discovery = config.get("discovery") if isinstance(config.get("discovery"), Mapping) else {}
    maximum_equities = max(1, int(discovery.get("maximum_equities") or 500))
    minimum_market_cap = float(discovery.get("minimum_market_cap_thousands") or 2_000_000)
    minimum_price = float((config.get("liquidity") or {}).get("minimum_stock_price") or 5.0)
    configured_by_ticker = {item.ticker: item for item in configured_items}
    eligible: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not DISCOVERY_TICKER_RE.fullmatch(ticker) or "_" in ticker:
            continue
        asset_type = str(row.get("assetType") if row.get("assetType") is not None else "")
        if asset_type not in {"0", "1", "2", "3"}:
            continue
        price = _number(row.get("pxCls")) or _number(row.get("pxAtmIv"))
        market_cap = _number(row.get("mktCap"))
        stock_dollar_volume = price * _number(row.get("stkVolu"))
        option_volume = _number(row.get("avgOptVolu20d"))
        if price < minimum_price or market_cap < minimum_market_cap:
            continue
        eligible[ticker] = {
            "row": row,
            "market_cap": market_cap,
            "stock_dollar_volume": stock_dollar_volume,
            "option_volume": option_volume,
        }
    market_cap_slots = min(maximum_equities, int(discovery.get("market_cap_slots") or 350))
    option_volume_slots = min(maximum_equities, int(discovery.get("option_volume_slots") or 125))
    stock_volume_slots = min(maximum_equities, int(discovery.get("stock_volume_slots") or 125))
    ranked_lists = [
        sorted(eligible, key=lambda ticker: (eligible[ticker]["market_cap"], ticker), reverse=True)[:market_cap_slots],
        sorted(eligible, key=lambda ticker: (eligible[ticker]["option_volume"], ticker), reverse=True)[:option_volume_slots],
        sorted(eligible, key=lambda ticker: (eligible[ticker]["stock_dollar_volume"], ticker), reverse=True)[:stock_volume_slots],
    ]
    rank: Dict[str, int] = {}
    for names in ranked_lists:
        for index, ticker in enumerate(names):
            rank[ticker] = min(rank.get(ticker, maximum_equities * 10), index)
    configured_equities = [
        item.ticker for item in configured_items
        if item.kind == "equity" and item.ticker in eligible
    ]
    selected_names = list(dict.fromkeys(configured_equities))
    ranked_union = sorted(
        set().union(*[set(names) for names in ranked_lists]),
        key=lambda ticker: (
            rank.get(ticker, maximum_equities * 10),
            -eligible[ticker]["market_cap"],
            ticker,
        ),
    )
    for ticker in ranked_union:
        if ticker not in selected_names and len(selected_names) < maximum_equities:
            selected_names.append(ticker)
    # Fill any overlap left by the three discovery sleeves with the next
    # largest eligible companies so `maximum_equities` is an actual breadth
    # target rather than an accidental upper bound.
    for ticker in sorted(
        eligible,
        key=lambda name: (eligible[name]["market_cap"], eligible[name]["option_volume"], name),
        reverse=True,
    ):
        if ticker not in selected_names and len(selected_names) < maximum_equities:
            selected_names.append(ticker)
    equities: List[UniverseItem] = []
    for ticker in selected_names:
        if ticker in configured_by_ticker:
            equities.append(configured_by_ticker[ticker])
            continue
        row = eligible[ticker]["row"]
        sector = str(row.get("sectorName") or row.get("sector") or "Unknown").strip()
        equities.append(
            UniverseItem(
                ticker=ticker,
                name=ticker,
                sector=sector or "Unknown",
                theme=sector or "Broad liquid equity",
                kind="equity",
                sector_etf=_sector_etf(sector, str(row.get("bestEtf") or "")),
            )
        )
    supporting = [item for item in configured_items if item.kind != "equity"]
    result = list({item.ticker: item for item in supporting + equities}.values())
    return result, {
        "source": "ORATS complete core universe",
        "orats_core_rows": len(rows),
        "eligible_discovery_equities": len(eligible),
        "selected_equities": len(equities),
        "configured_equities_preserved": len(configured_equities),
        "selection": "configured themes plus leading market-cap, stock-dollar-volume, and average-option-volume names",
    }


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    target = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CORAT config must be a JSON object")
    payload["_config_path"] = str(target)
    payload["_project_root"] = str(PROJECT_ROOT)
    if overrides:
        payload = _deep_merge(payload, overrides)
    return payload


def load_universe(config: Mapping[str, Any], tickers: Optional[Iterable[str]] = None) -> List[UniverseItem]:
    configured = Path(str(config["universe_file"]))
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    wanted = None
    if tickers is not None:
        wanted = {str(value).strip().upper() for value in tickers if str(value).strip()}
    items: List[UniverseItem] = []
    seen = set()
    with configured.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            if wanted is not None and ticker not in wanted:
                continue
            seen.add(ticker)
            items.append(
                UniverseItem(
                    ticker=ticker,
                    name=str(row.get("name") or ticker).strip(),
                    sector=str(row.get("sector") or "Unknown").strip(),
                    theme=str(row.get("theme") or "").strip(),
                    kind=str(row.get("kind") or "equity").strip(),
                    sector_etf=str(row.get("sector_etf") or "SPY").strip().upper(),
                )
            )
    if wanted is not None:
        missing = wanted - seen
        for ticker in sorted(missing):
            items.append(UniverseItem(ticker, ticker, "Unknown", "", "equity", "SPY"))
    if not items:
        raise ValueError("CORAT universe is empty")
    return items


def supporting_tickers(config: Mapping[str, Any], selected: Iterable[UniverseItem]) -> List[str]:
    names = {str(value).upper() for value in config.get("regime", {}).get("benchmarks", [])}
    names.update(str(value).upper() for value in config.get("regime", {}).get("macro_proxies", []))
    names.update(item.sector_etf for item in selected if item.sector_etf)
    names.update(item.ticker for item in selected)
    return sorted(name for name in names if name)
