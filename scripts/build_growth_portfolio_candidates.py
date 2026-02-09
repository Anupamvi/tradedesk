#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build growth portfolio candidates markdown from UW capture files.")
    parser.add_argument(
        "--trade-date",
        default=dt.date.today().isoformat(),
        help="Day folder to process (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root folder that contains daily YYYY-MM-DD folders.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="How many ranked candidates to include in the report.",
    )
    parser.add_argument(
        "--portfolio-size",
        type=int,
        default=12,
        help="Target number of names in diversified basket proposal.",
    )
    parser.add_argument(
        "--max-per-sector",
        type=int,
        default=2,
        help="Max names per sector in diversified basket.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=10.0,
        help="Minimum score for inclusion in ranked candidates.",
    )
    parser.add_argument(
        "--min-sources",
        type=int,
        default=2,
        help="Minimum number of independent source groups required for ranked list.",
    )
    parser.add_argument(
        "--require-stock-screener",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the run if stock-screener data is unavailable.",
    )
    parser.add_argument(
        "--stock-lookback-days",
        type=int,
        default=7,
        help="If today's stock-screener is missing, fallback to latest prior file within this many days.",
    )
    return parser.parse_args()


def parse_trade_date(text: str) -> dt.date:
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--trade-date must be YYYY-MM-DD, got: {text}") from exc


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


def clean_ticker(raw: str) -> str:
    if raw is None:
        return ""
    ticker = str(raw).strip().upper().replace("/", ".").replace("$", "")
    if ticker in {"", "TICKER"}:
        return ""
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", ticker):
        return ""
    return ticker


def parse_number(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    s = s.replace("\u2013", "-").replace("\u2212", "-")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        return None
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        neg = True
        s = s[1:]
    mult = 1.0
    if s.endswith("%"):
        s = s[:-1]
    if s.endswith("K"):
        mult = 1_000.0
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1_000_000.0
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1_000_000_000.0
        s = s[:-1]
    elif s.endswith("T"):
        mult = 1_000_000_000_000.0
        s = s[:-1]
    if not s:
        return None
    try:
        value = float(s) * mult
    except ValueError:
        return None
    return -value if neg else value


def parse_percent(text: str) -> Optional[float]:
    v = parse_number(text)
    return None if v is None else float(v)


def read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    data = rows[1:]
    return header, data


def data_row_count(path: Path) -> int:
    try:
        header, rows = read_csv_rows(path)
    except Exception:
        return 0
    if not header:
        return 0
    head_key = [normalize_text(x) for x in header]
    count = 0
    for row in rows:
        if not row:
            continue
        row_key = [normalize_text(x) for x in row]
        if row_key == head_key:
            continue
        if not any(cell.strip() for cell in row):
            continue
        count += 1
    return count


def choose_best_csv(candidates: Sequence[Path]) -> Optional[Path]:
    existing = [p for p in candidates if p.exists() and p.is_file()]
    if not existing:
        return None

    def rank(path: Path) -> Tuple[int, int, int]:
        rows = data_row_count(path)
        st = path.stat()
        return rows, int(st.st_size), int(st.st_mtime_ns)

    return max(existing, key=rank)


def find_best_csv(day_dir: Path, glob_patterns: Sequence[str], include_unzipped: bool = True) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in glob_patterns:
        candidates.extend(day_dir.glob(pat))
    if include_unzipped:
        uz = day_dir / "_unzipped_mode_a"
        if uz.exists():
            for pat in glob_patterns:
                candidates.extend(uz.glob(pat))
    deduped = list(dict.fromkeys(candidates))
    return choose_best_csv(deduped)


def extract_day_from_stock_path(path: Path) -> Optional[dt.date]:
    # Expected layouts:
    #   <base>/<YYYY-MM-DD>/stock-screener-YYYY-MM-DD.csv
    #   <base>/<YYYY-MM-DD>/_unzipped_mode_a/stock-screener-YYYY-MM-DD.csv
    for parent in path.parents:
        name = parent.name
        if re.match(r"^\d{4}-\d{2}-\d{2}$", name):
            try:
                return dt.datetime.strptime(name, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def find_recent_stock_screener(base_dir: Path, trade_date: dt.date, lookback_days: int) -> Optional[Path]:
    candidates: List[Path] = []
    candidates.extend(base_dir.glob("*/stock-screener-*.csv"))
    candidates.extend(base_dir.glob("*/_unzipped_mode_a/stock-screener-*.csv"))
    valid: List[Tuple[int, int, int, Path]] = []
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        day = extract_day_from_stock_path(p)
        if day is None:
            continue
        age = (trade_date - day).days
        if age < 0 or age > max(0, int(lookback_days)):
            continue
        rows = data_row_count(p)
        st = p.stat()
        valid.append((rows, -age, int(st.st_mtime_ns), p))
    if not valid:
        return None
    valid.sort(reverse=True)
    return valid[0][3]


def col_index(header: Sequence[str], names: Sequence[str]) -> Optional[int]:
    norm_map: Dict[str, int] = {}
    for i, col in enumerate(header):
        k = normalize_text(col)
        if k and k not in norm_map:
            norm_map[k] = i
    for name in names:
        k = normalize_text(name)
        if k in norm_map:
            return norm_map[k]
    for name in names:
        nk = normalize_text(name)
        if not nk:
            continue
        for k, idx in norm_map.items():
            if nk in k or k in nk:
                return idx
    return None


def canonical_sector(raw: str) -> str:
    t = (raw or "").strip().lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    synonyms = {
        "health care": "healthcare",
        "healthcare": "healthcare",
        "consumer cyclical": "consumercyclical",
        "consumer discretionary": "consumercyclical",
        "consumer defensive": "consumerstaples",
        "consumer staples": "consumerstaples",
        "financial services": "financialservices",
        "financials": "financialservices",
        "communication services": "communicationservices",
        "real estate": "realestate",
        "basic materials": "materials",
    }
    if t in synonyms:
        return synonyms[t]
    return re.sub(r"[^a-z0-9]+", "", t)


def display_sector(raw: str) -> str:
    return raw.strip() if raw and raw.strip() else "Unknown"


def ensure_signal(store: Dict[str, dict], ticker: str) -> dict:
    if ticker not in store:
        store[ticker] = {
            "ticker": ticker,
            "sector": "",
            "sector_key": "",
            "marketcap": None,
            "issue_type": "",
            "is_index": False,
            "source_stock": False,
            "source_insider": False,
            "source_sec": False,
            "source_analyst": False,
            "put_call_ratio": None,
            "bullish_premium": 0.0,
            "bearish_premium": 0.0,
            "net_call_premium": 0.0,
            "net_put_premium": 0.0,
            "insider_buys": 0,
            "insider_sells": 0,
            "insider_buy_value": 0.0,
            "insider_sell_value": 0.0,
            "sec_filings": 0,
            "sec_13d": 0,
            "sec_13g": 0,
            "sec_13f": 0,
            "sec_other": 0,
            "sec_new_holdings": 0,
            "sec_shares_total": 0.0,
            "analyst_trades": 0,
            "analyst_bull": 0,
            "analyst_bear": 0,
            "analyst_neutral": 0,
            "analyst_premium": 0.0,
        }
    return store[ticker]


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def fmt_money_short(value: float) -> str:
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000_000:
        return f"{sign}${a / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{sign}${a / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{sign}${a / 1_000:.1f}K"
    return f"{sign}${a:.0f}"


def parse_stock_screener(path: Optional[Path], signals: Dict[str, dict]) -> int:
    if not path:
        return 0
    header, rows = read_csv_rows(path)
    if not header:
        return 0
    i_ticker = col_index(header, ["ticker"])
    i_sector = col_index(header, ["sector"])
    i_marketcap = col_index(header, ["marketcap"])
    i_issue_type = col_index(header, ["issue_type", "issuetype"])
    i_is_index = col_index(header, ["is_index", "isindex"])
    i_put_call = col_index(header, ["put_call_ratio", "putcallratio"])
    i_bull_prem = col_index(header, ["bullish_premium", "bullishpremium"])
    i_bear_prem = col_index(header, ["bearish_premium", "bearishpremium"])
    i_net_call = col_index(header, ["net_call_premium", "netcallpremium"])
    i_net_put = col_index(header, ["net_put_premium", "netputpremium"])

    parsed = 0
    for row in rows:
        if i_ticker is None or i_ticker >= len(row):
            continue
        ticker = clean_ticker(row[i_ticker])
        if not ticker:
            continue
        sig = ensure_signal(signals, ticker)
        sig["source_stock"] = True
        if i_sector is not None and i_sector < len(row):
            sec = row[i_sector].strip()
            if sec:
                sig["sector"] = sec
                sig["sector_key"] = canonical_sector(sec)
        if i_marketcap is not None and i_marketcap < len(row):
            mc = parse_number(row[i_marketcap])
            if mc is not None and mc > 0:
                sig["marketcap"] = mc
        if i_issue_type is not None and i_issue_type < len(row):
            sig["issue_type"] = row[i_issue_type].strip()
        if i_is_index is not None and i_is_index < len(row):
            v = row[i_is_index].strip().lower()
            sig["is_index"] = v in {"t", "true", "1", "yes", "y"}
        if i_put_call is not None and i_put_call < len(row):
            sig["put_call_ratio"] = parse_number(row[i_put_call])
        if i_bull_prem is not None and i_bull_prem < len(row):
            sig["bullish_premium"] = parse_number(row[i_bull_prem]) or 0.0
        if i_bear_prem is not None and i_bear_prem < len(row):
            sig["bearish_premium"] = parse_number(row[i_bear_prem]) or 0.0
        if i_net_call is not None and i_net_call < len(row):
            sig["net_call_premium"] = parse_number(row[i_net_call]) or 0.0
        if i_net_put is not None and i_net_put < len(row):
            sig["net_put_premium"] = parse_number(row[i_net_put]) or 0.0
        parsed += 1
    return parsed


def parse_insiders(path: Optional[Path], signals: Dict[str, dict]) -> int:
    if not path:
        return 0
    header, rows = read_csv_rows(path)
    if not header:
        return 0
    i_ticker = col_index(header, ["ticker"])

    parsed = 0
    for row in rows:
        if i_ticker is None or i_ticker >= len(row):
            continue
        ticker = clean_ticker(row[i_ticker])
        if not ticker:
            continue
        sig = ensure_signal(signals, ticker)
        sig["source_insider"] = True

        joined = " ".join(str(x) for x in row).upper()
        is_buy = "PURCHASE" in joined or " P - " in f" {joined} "
        is_sell = "SALE" in joined or " S - " in f" {joined} "

        numeric_cells: List[Tuple[int, str, float]] = []
        for idx, raw in enumerate(row):
            v = parse_number(raw)
            if v is None:
                continue
            numeric_cells.append((idx, str(raw), float(v)))

        price_idx = None
        shares_idx = None
        for idx, raw, v in numeric_cells:
            raw_s = raw.strip()
            av = abs(v)
            if price_idx is None and "$" in raw_s and 0.05 <= av <= 2_000:
                price_idx = idx
                continue
            if shares_idx is None:
                has_suffix = bool(re.search(r"[KMBT]\s*$", raw_s.upper()))
                if "$" not in raw_s and not has_suffix and av >= 100 and av <= 10_000_000 and float(int(av)) == av:
                    shares_idx = idx

        value = None
        if price_idx is not None and shares_idx is not None:
            price = parse_number(row[price_idx])
            shares = parse_number(row[shares_idx])
            if price is not None and shares is not None:
                target = abs(price * shares)
                best_err = float("inf")
                for idx, _, v in numeric_cells:
                    if idx in {price_idx, shares_idx}:
                        continue
                    av = abs(v)
                    if av < 100:
                        continue
                    err = abs(av - target)
                    if err < best_err:
                        best_err = err
                        value = v

        if value is None:
            # Fallback: prefer K/M/B style values and avoid trans-count-like integers.
            candidates: List[Tuple[float, float]] = []
            for _, raw, v in numeric_cells:
                av = abs(v)
                if av <= 20 and float(int(av)) == av:
                    continue
                raw_s = raw.strip().upper()
                has_suffix = bool(re.search(r"[KMBT]\s*$", raw_s))
                if has_suffix and av < 1_000_000_000:
                    candidates.append((av, v))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                value = candidates[0][1]

        if value is None:
            # Last fallback: smallest non-trivial numeric value excluding obvious price-like cells.
            candidates2: List[Tuple[float, float]] = []
            for _, raw, v in numeric_cells:
                av = abs(v)
                raw_s = raw.strip()
                if av < 100:
                    continue
                if "$" in raw_s and av <= 2_000:
                    continue
                if av > 1_000_000_000:
                    continue
                candidates2.append((av, v))
            if candidates2:
                candidates2.sort(key=lambda x: x[0])
                value = candidates2[0][1]

        if value is None:
            value = 0.0

        if is_buy:
            sig["insider_buys"] += 1
            sig["insider_buy_value"] += abs(value)
        elif is_sell:
            sig["insider_sells"] += 1
            sig["insider_sell_value"] += abs(value)
        else:
            if value >= 0:
                sig["insider_buys"] += 1
                sig["insider_buy_value"] += abs(value)
            else:
                sig["insider_sells"] += 1
                sig["insider_sell_value"] += abs(value)
        parsed += 1
    return parsed


def parse_sec_filings(path: Optional[Path], signals: Dict[str, dict]) -> int:
    if not path:
        return 0
    header, rows = read_csv_rows(path)
    if not header:
        return 0
    i_ticker = col_index(header, ["ticker"])
    i_filing_type = col_index(header, ["filing type", "filingtype"])
    i_shares = col_index(header, ["shares held", "sharesheld"])
    i_new_holdings = col_index(header, ["new holdings", "newholdings"])

    parsed = 0
    for row in rows:
        if i_ticker is None or i_ticker >= len(row):
            continue
        ticker = clean_ticker(row[i_ticker])
        if not ticker:
            continue
        sig = ensure_signal(signals, ticker)
        sig["source_sec"] = True
        sig["sec_filings"] += 1

        filing = ""
        if i_filing_type is not None and i_filing_type < len(row):
            filing = row[i_filing_type].upper().strip()
        if "13D" in filing:
            sig["sec_13d"] += 1
        elif "13G" in filing:
            sig["sec_13g"] += 1
        elif "13F" in filing:
            sig["sec_13f"] += 1
        else:
            sig["sec_other"] += 1

        if i_new_holdings is not None and i_new_holdings < len(row):
            nh = row[i_new_holdings].strip()
            if nh and nh not in {"0", "-", "--"}:
                sig["sec_new_holdings"] += 1
        if i_shares is not None and i_shares < len(row):
            sh = parse_number(row[i_shares])
            if sh is not None and sh > 0:
                sig["sec_shares_total"] += sh
        parsed += 1
    return parsed


def parse_analysts(path: Optional[Path], signals: Dict[str, dict]) -> int:
    if not path:
        return 0
    header, rows = read_csv_rows(path)
    if not header:
        return 0
    i_ticker = col_index(header, ["ticker"])
    i_premium = col_index(header, ["premium"])
    i_tags = col_index(header, ["tags unusual whales", "tags"])

    parsed = 0
    for row in rows:
        if i_ticker is None or i_ticker >= len(row):
            continue
        ticker = clean_ticker(row[i_ticker])
        if not ticker:
            continue
        sig = ensure_signal(signals, ticker)
        sig["source_analyst"] = True
        sig["analyst_trades"] += 1

        if i_premium is not None and i_premium < len(row):
            prem = parse_number(row[i_premium])
            if prem is not None:
                sig["analyst_premium"] += abs(prem)

        tags = ""
        if i_tags is not None and i_tags < len(row):
            tags = row[i_tags].lower()
        row_text = " ".join(str(x) for x in row).lower()
        if row_text:
            tags = f"{tags} {row_text}".strip()
        if "bullish" in tags:
            sig["analyst_bull"] += 1
        elif "bearish" in tags:
            sig["analyst_bear"] += 1
        else:
            sig["analyst_neutral"] += 1
        parsed += 1
    return parsed


def parse_flow_sector_changes(path: Optional[Path]) -> Dict[str, float]:
    if not path:
        return {}
    header, rows = read_csv_rows(path)
    if not header:
        return {}
    i_sector = col_index(header, ["sector"])
    i_change = col_index(header, ["change"])
    if i_sector is None or i_change is None:
        return {}
    out: Dict[str, float] = {}
    for row in rows:
        if i_sector >= len(row) or i_change >= len(row):
            continue
        sec = row[i_sector].strip()
        chg = parse_percent(row[i_change])
        if not sec or chg is None:
            continue
        key = canonical_sector(sec)
        if key:
            out[key] = chg
    return out


def score_candidate(sig: dict, sector_changes: Dict[str, float]) -> dict:
    bullish_premium = float(sig["bullish_premium"])
    bearish_premium = float(sig["bearish_premium"])
    net_call = float(sig["net_call_premium"])
    net_put = float(sig["net_put_premium"])
    put_call = sig["put_call_ratio"]

    insider_buys = int(sig["insider_buys"])
    insider_sells = int(sig["insider_sells"])
    insider_buy_val = float(sig["insider_buy_value"])
    insider_sell_val = float(sig["insider_sell_value"])
    insider_net = insider_buy_val - insider_sell_val

    sec_13d = int(sig["sec_13d"])
    sec_13g = int(sig["sec_13g"])
    sec_13f = int(sig["sec_13f"])
    sec_new = int(sig["sec_new_holdings"])
    sec_shares_total = float(sig["sec_shares_total"])

    analyst_bull = int(sig["analyst_bull"])
    analyst_bear = int(sig["analyst_bear"])
    analyst_premium = float(sig["analyst_premium"])

    stock_score = 0.0
    if sig["source_stock"]:
        total_prem = bullish_premium + bearish_premium
        if total_prem > 0:
            balance = (bullish_premium - bearish_premium) / total_prem
            stock_score += balance * 18.0
        if put_call is not None:
            if put_call <= 0.80:
                stock_score += 6.0
            elif put_call <= 1.00:
                stock_score += 3.0
            elif put_call <= 1.20:
                stock_score += 0.0
            elif put_call <= 1.50:
                stock_score -= 4.0
            else:
                stock_score -= 8.0
        if abs(net_call) > 0:
            stock_score += math.copysign(min(8.0, math.log10(abs(net_call) + 1.0) * 1.8), net_call)
        if abs(net_put) > 0:
            stock_score += -math.copysign(min(8.0, math.log10(abs(net_put) + 1.0) * 1.8), net_put)
        stock_score = clip(stock_score, -30.0, 30.0)

    insider_score = 0.0
    if sig["source_insider"]:
        insider_score += clip((insider_buys - insider_sells) * 1.8, -14.0, 14.0)
        if abs(insider_net) > 0:
            insider_score += math.copysign(min(14.0, math.log10(abs(insider_net) + 1.0) * 3.2), insider_net)
        insider_score += min(6.0, math.log10(insider_buy_val + insider_sell_val + 1.0) * 1.2)
        insider_score = clip(insider_score, -25.0, 25.0)

    sec_score = 0.0
    if sig["source_sec"]:
        if (sec_13d + sec_13g + sec_new) == 0 and sec_13f > 0:
            sec_score += min(6.0, 0.02 * sec_13f)
            sec_score += min(2.0, math.log10(sec_shares_total + 1.0) * 0.35)
        else:
            sec_score += 6.0 * sec_13d
            sec_score += 3.0 * sec_13g
            sec_score += 0.02 * sec_13f
            sec_score += 2.0 * sec_new
            sec_score = min(20.0, sec_score)
            sec_score += min(5.0, math.log10(sec_shares_total + 1.0) * 0.6)
        sec_score = clip(sec_score, 0.0, 25.0)

    analyst_score = 0.0
    if sig["source_analyst"]:
        analyst_score += clip((analyst_bull - analyst_bear) * 2.0, -12.0, 12.0)
        prem_scale = min(8.0, math.log10(analyst_premium + 1.0) * 1.8) if analyst_premium > 0 else 0.0
        if analyst_bull > analyst_bear:
            analyst_score += prem_scale
        elif analyst_bear > analyst_bull:
            analyst_score -= prem_scale
        analyst_score = clip(analyst_score, -20.0, 20.0)

    sector_score = 0.0
    sector_key = sig.get("sector_key") or ""
    if sector_key and sector_key in sector_changes:
        chg = sector_changes[sector_key]
        if chg >= 2.0:
            sector_score = 4.0
        elif chg >= 0.5:
            sector_score = 2.0
        elif chg <= -2.0:
            sector_score = -4.0
        elif chg <= -0.5:
            sector_score = -2.0

    source_count = int(bool(sig["source_stock"])) + int(bool(sig["source_insider"])) + int(bool(sig["source_sec"])) + int(
        bool(sig["source_analyst"])
    )
    coverage_score = source_count * 2.0 - 4.0

    total_score = stock_score + insider_score + sec_score + analyst_score + sector_score + coverage_score

    if total_score >= 45.0:
        action = "Build / Increase"
    elif total_score >= 30.0:
        action = "Accumulate"
    elif total_score >= 15.0:
        action = "Watchlist"
    elif total_score <= -10.0:
        action = "Trim / Avoid"
    else:
        action = "Hold"

    if source_count >= 3 and abs(total_score) >= 25.0:
        confidence = "High"
    elif source_count >= 2 and abs(total_score) >= 12.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    flags: List[str] = []
    positives: List[str] = []

    if insider_sells > insider_buys and insider_sell_val > insider_buy_val * 1.5 and insider_sell_val > 1_000_000:
        flags.append("insider selling pressure")
    if analyst_bear > analyst_bull + 2:
        flags.append("bearish analyst/flow skew")
    if put_call is not None and put_call > 1.20:
        flags.append("elevated put/call ratio")
    if source_count < 2:
        flags.append("thin signal coverage")

    if sec_13d > 0:
        positives.append("13D activity")
    if insider_buys > insider_sells and insider_buy_val > insider_sell_val:
        positives.append("net insider buying")
    if analyst_bull > analyst_bear:
        positives.append("bullish analyst/flow skew")
    if not positives and sec_13g > 0:
        positives.append("13G accumulation")

    return {
        "ticker": sig["ticker"],
        "sector": display_sector(sig.get("sector", "")),
        "sector_key": sig.get("sector_key", ""),
        "marketcap": sig.get("marketcap"),
        "issue_type": sig.get("issue_type", ""),
        "is_index": bool(sig.get("is_index", False)),
        "source_stock": bool(sig.get("source_stock", False)),
        "source_count": source_count,
        "stock_score": round(stock_score, 2),
        "insider_score": round(insider_score, 2),
        "sec_score": round(sec_score, 2),
        "analyst_score": round(analyst_score, 2),
        "sector_score": round(sector_score, 2),
        "coverage_score": round(coverage_score, 2),
        "score": round(total_score, 2),
        "action": action,
        "confidence": confidence,
        "flags": "; ".join(flags),
        "positives": "; ".join(positives),
        "insider_buys": insider_buys,
        "insider_sells": insider_sells,
        "insider_net_value": round(insider_net, 2),
        "sec_13d": sec_13d,
        "sec_13g": sec_13g,
        "sec_13f": sec_13f,
        "sec_new_holdings": sec_new,
        "analyst_bull": analyst_bull,
        "analyst_bear": analyst_bear,
        "analyst_trades": int(sig["analyst_trades"]),
        "put_call_ratio": None if put_call is None else round(float(put_call), 3),
        "bullish_premium": round(bullish_premium, 2),
        "bearish_premium": round(bearish_premium, 2),
    }


def is_growth_equity(row: dict) -> bool:
    if not bool(row.get("source_stock", False)):
        return False
    if str(row.get("sector", "Unknown")) == "Unknown":
        return False
    if row.get("is_index"):
        return False
    issue_type = str(row.get("issue_type", "") or "").lower()
    if "etf" in issue_type:
        return False
    if row["ticker"] in {"SPX", "SPXW", "VIX", "NDX"}:
        return False
    return True


def build_diversified_basket(
    ranked: Sequence[dict],
    portfolio_size: int,
    max_per_sector: int,
) -> List[dict]:
    basket: List[dict] = []
    sector_counts: Dict[str, int] = {}
    eligible = [r for r in ranked if r["score"] >= 25.0 and r["action"] in {"Build / Increase", "Accumulate"}]

    for row in eligible:
        if len(basket) >= portfolio_size:
            break
        key = row.get("sector_key") or "unknown"
        if sector_counts.get(key, 0) >= max_per_sector:
            continue
        basket.append(row)
        sector_counts[key] = sector_counts.get(key, 0) + 1

    if len(basket) < portfolio_size:
        already = {r["ticker"] for r in basket}
        for row in eligible:
            if len(basket) >= portfolio_size:
                break
            if row["ticker"] in already:
                continue
            basket.append(row)
            already.add(row["ticker"])

    return basket


def write_candidates_csv(path: Path, rows: Sequence[dict]) -> None:
    cols = [
        "ticker",
        "sector",
        "score",
        "action",
        "confidence",
        "source_count",
        "stock_score",
        "insider_score",
        "sec_score",
        "analyst_score",
        "sector_score",
        "coverage_score",
        "insider_buys",
        "insider_sells",
        "insider_net_value",
        "sec_13d",
        "sec_13g",
        "sec_13f",
        "sec_new_holdings",
        "analyst_bull",
        "analyst_bear",
        "analyst_trades",
        "put_call_ratio",
        "bullish_premium",
        "bearish_premium",
        "positives",
        "flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def write_markdown(
    path: Path,
    trade_date: dt.date,
    day_dir: Path,
    source_files: Dict[str, Optional[Path]],
    source_rows: Dict[str, int],
    stock_day_used: Optional[dt.date],
    scored_all: Sequence[dict],
    ranked: Sequence[dict],
    basket: Sequence[dict],
    min_score: float,
    min_sources: int,
    top_n: int,
) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append(f"# Growth Portfolio Candidates - {trade_date.isoformat()}")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Folder: `{day_dir}`")
    lines.append(f"- Ranking floor (`--min-score`): {min_score}")
    lines.append(f"- Ranking source floor (`--min-sources`): {min_sources}")
    if stock_day_used is not None:
        lines.append(f"- Stock screener date used: {stock_day_used.isoformat()}")
    lines.append("")
    lines.append("## Data Sources Used")
    for key in ["stock_screener", "insiders", "sec_filings", "analysts", "flow_sectors"]:
        p = source_files.get(key)
        n = source_rows.get(key, 0)
        if p:
            lines.append(f"- `{key}`: `{p}` ({n} rows)")
        else:
            lines.append(f"- `{key}`: not found")
    lines.append("")

    lines.append("## Diversified Basket Proposal")
    if basket:
        lines.append("")
        lines.append("| Rank | Ticker | Sector | Score | Action | Confidence | Primary Drivers | Risk Flags |")
        lines.append("|---:|---|---|---:|---|---|---|---|")
        for i, row in enumerate(basket, start=1):
            drivers = row["positives"] or "-"
            flags = row["flags"] or "-"
            lines.append(
                f"| {i} | {row['ticker']} | {row['sector']} | {row['score']:.2f} | {row['action']} | {row['confidence']} | {drivers} | {flags} |"
            )
    else:
        lines.append("")
        lines.append("- No names met diversified basket thresholds yet. Capture more routes (include `stock-screener`) and rerun.")
    lines.append("")

    lines.append("## Ranked Candidates")
    lines.append("")
    lines.append("| Rank | Ticker | Sector | Score | Insider Net | 13D | 13G | Analyst B/B | Put/Call | Sources | Action |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, row in enumerate(ranked[:top_n], start=1):
        pcr = "-" if row["put_call_ratio"] is None else f"{row['put_call_ratio']:.2f}"
        lines.append(
            f"| {i} | {row['ticker']} | {row['sector']} | {row['score']:.2f} | {fmt_money_short(row['insider_net_value'])} | {row['sec_13d']} | {row['sec_13g']} | {row['analyst_bull']}/{row['analyst_bear']} | {pcr} | {row['source_count']} | {row['action']} |"
        )
    lines.append("")

    trim_rows = [r for r in scored_all if r["score"] <= -10.0 and int(r.get("source_count", 0)) >= min_sources]
    lines.append("## Trim / Reduce Watchlist")
    if trim_rows:
        lines.append("")
        lines.append("| Ticker | Sector | Score | Main Flags |")
        lines.append("|---|---|---:|---|")
        for row in trim_rows[:15]:
            lines.append(f"| {row['ticker']} | {row['sector']} | {row['score']:.2f} | {row['flags'] or '-'} |")
    else:
        lines.append("")
        lines.append("- No strong trim signals in current capture.")
    lines.append("")

    lines.append("## Monthly Rebalance Rules")
    lines.append("1. Run capture + finalize + this report on the first trading weekend each month.")
    lines.append("2. Add or increase names only from `Build / Increase` or `Accumulate` with `confidence` >= `Medium`.")
    lines.append("3. Keep sector balance by capping to 2 names per sector unless score spread is very large.")
    lines.append("4. Trim when a name stays `Trim / Avoid` for 2 consecutive monthly runs.")
    lines.append("5. Re-check earnings/SEC catalysts before placing final orders.")
    lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    trade_date = parse_trade_date(args.trade_date)
    base_dir = Path(args.base_dir).resolve()
    day_dir = base_dir / trade_date.isoformat()
    if not day_dir.exists():
        print(f"Day folder does not exist: {day_dir}")
        return 2

    stock_today = find_best_csv(day_dir, ["stock-screener-*.csv"], include_unzipped=True)
    stock_fallback = None
    if stock_today is None:
        stock_fallback = find_recent_stock_screener(
            base_dir=base_dir,
            trade_date=trade_date,
            lookback_days=int(args.stock_lookback_days),
        )
    stock_source = stock_today or stock_fallback
    if stock_source is None and bool(args.require_stock_screener):
        print("Missing stock-screener data. Capture `stock-screener` route (or provide recent file) and rerun.")
        return 2

    source_files: Dict[str, Optional[Path]] = {
        "stock_screener": stock_source,
        "insiders": find_best_csv(day_dir, ["insiders-trades-scrape-*.csv"], include_unzipped=False),
        "sec_filings": find_best_csv(day_dir, ["sec-filings-scrape-*.csv"], include_unzipped=False),
        "analysts": find_best_csv(day_dir, ["analysts-scrape-*.csv"], include_unzipped=False),
        "flow_sectors": find_best_csv(day_dir, ["flow-sectors-scrape-*.csv"], include_unzipped=False),
    }
    stock_day_used = extract_day_from_stock_path(stock_source) if stock_source else None

    signals: Dict[str, dict] = {}
    source_rows: Dict[str, int] = {}
    source_rows["stock_screener"] = parse_stock_screener(source_files["stock_screener"], signals)
    source_rows["insiders"] = parse_insiders(source_files["insiders"], signals)
    source_rows["sec_filings"] = parse_sec_filings(source_files["sec_filings"], signals)
    source_rows["analysts"] = parse_analysts(source_files["analysts"], signals)
    source_rows["flow_sectors"] = data_row_count(source_files["flow_sectors"]) if source_files["flow_sectors"] else 0

    sector_changes = parse_flow_sector_changes(source_files["flow_sectors"])

    scored: List[dict] = []
    for sig in signals.values():
        row = score_candidate(sig, sector_changes)
        if not is_growth_equity(row):
            continue
        scored.append(row)

    scored.sort(key=lambda x: (x["score"], x["source_count"]), reverse=True)
    ranked = [
        r
        for r in scored
        if r["score"] >= float(args.min_score)
        and int(r["source_count"]) >= int(args.min_sources)
        and bool(r.get("source_stock", False))
        and str(r.get("sector", "Unknown")) != "Unknown"
    ]
    basket = build_diversified_basket(
        ranked=ranked,
        portfolio_size=int(args.portfolio_size),
        max_per_sector=int(args.max_per_sector),
    )

    out_csv = day_dir / f"growth_portfolio_candidates_{trade_date.isoformat()}.csv"
    out_md = day_dir / f"growth_portfolio_candidates_{trade_date.isoformat()}.md"
    write_candidates_csv(out_csv, scored)
    write_markdown(
        path=out_md,
        trade_date=trade_date,
        day_dir=day_dir,
        source_files=source_files,
        source_rows=source_rows,
        stock_day_used=stock_day_used,
        scored_all=scored,
        ranked=ranked,
        basket=basket,
        min_score=float(args.min_score),
        min_sources=int(args.min_sources),
        top_n=int(args.top_n),
    )

    print(f"Processed day folder: {day_dir}")
    print(f"Scored tickers: {len(scored)}")
    print(f"Ranked (score >= {float(args.min_score):.1f}, sources >= {int(args.min_sources)}): {len(ranked)}")
    print(f"Diversified basket size: {len(basket)}")
    print(f"Candidates CSV: {out_csv}")
    print(f"Candidates MD:  {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
