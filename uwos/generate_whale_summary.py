import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from uwos.whale_source import BOT_EOD_DTYPE, bot_eod_usecols, is_split_bot_eod_part, open_bot_eod

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Please install pyyaml.") from exc


def resolve_config_path(path: Path) -> Path:
    raw = Path(path)
    candidates = [raw]

    if raw.is_absolute():
        candidates.append(raw.parent / "uwos" / raw.name)
    else:
        cwd = Path.cwd()
        module_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                cwd / raw,
                cwd / "uwos" / raw.name,
                module_dir / raw,
                module_dir / raw.name,
            ]
        )

    seen = set()
    ordered = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)

    for candidate in ordered:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(p) for p in ordered)
    raise FileNotFoundError(f"Config file not found. Tried: {tried}")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_date_from_path(path):
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(path))
    return match.group(0) if match else "Unknown Date"


def build_width(prices, tiers):
    width = np.full(len(prices), np.nan, dtype="float64")
    for i, tier in enumerate(tiers):
        min_price = float(tier["min_price"])
        max_price = float(tier["max_price"])
        default_width = float(tier["default_width"])
        if i == len(tiers) - 1:
            mask = (prices >= min_price) & (prices <= max_price)
        else:
            mask = (prices >= min_price) & (prices < max_price)
        width[mask] = default_width
    return width


def format_markdown_table(df):
    if df.empty:
        return "(no rows)"
    try:
        return df.to_markdown(index=False)
    except ImportError as exc:
        if "tabulate" not in str(exc).lower():
            raise

    def cell(value):
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    cols = [str(col).replace("|", "\\|") for col in df.columns]
    rows = ["| " + " | ".join(cols) + " |"]
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(cell(row[col]) for col in df.columns) + " |")
    return "\n".join(rows)


def update_symbol_stats(symbol_stats, df):
    grouped = df.groupby("underlying_symbol", dropna=False)["premium"].agg(["count", "sum"]).reset_index()
    for _, row in grouped.iterrows():
        symbol = row["underlying_symbol"]
        if symbol not in symbol_stats:
            symbol_stats[symbol] = [0, 0.0]
        symbol_stats[symbol][0] += int(row["count"])
        symbol_stats[symbol][1] += float(row["sum"])


def symbol_stats_frame(symbol_stats, limit):
    if not symbol_stats:
        return pd.DataFrame(columns=["underlying_symbol", "count", "total_premium"])
    df = pd.DataFrame(
        [
            {"underlying_symbol": k, "count": v[0], "total_premium": v[1]}
            for k, v in symbol_stats.items()
        ]
    )
    return df.sort_values("total_premium", ascending=False).head(limit)


def keep_top_premium(existing, candidate, limit):
    if candidate.empty:
        return existing
    candidate = candidate.nlargest(limit, "premium")
    out = pd.concat([existing, candidate], ignore_index=True)
    return out.nlargest(limit, "premium").reset_index(drop=True)


def infer_price_and_premium(chunk):
    price = pd.to_numeric(chunk["price"], errors="coerce")
    premium = pd.to_numeric(chunk["premium"], errors="coerce")
    size = pd.to_numeric(chunk["size"], errors="coerce")

    inferred_price = premium / (size * 100.0)
    price = price.where(price.notna(), inferred_price.where((premium > 0) & (size > 0)))

    inferred_premium = price * size * 100.0
    premium = premium.where(premium.notna(), inferred_premium.where((price > 0) & (size > 0)))

    chunk["price"] = price
    chunk["premium"] = premium


def annotate_reject_reasons(rejected, failure_masks):
    if rejected.empty:
        return rejected
    fail_df = pd.DataFrame(
        {reason: mask.reindex(rejected.index, fill_value=False) for reason, mask in failure_masks.items()},
        index=rejected.index,
    )
    rejected["reject_reasons"] = fail_df.apply(
        lambda row: "; ".join(reason for reason, failed in row.items() if bool(failed)),
        axis=1,
    )
    return rejected


def main():
    parser = argparse.ArgumentParser(description="Generate Yes-Prime summary from bot-eod report.")
    parser.add_argument(
        "--input",
        default=r"C:\Users\anupamvi\Downloads\bot-eod-report-2026-01-30.csv",
        help="Path to bot-eod-report CSV",
    )
    parser.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parent / "rulebook_config.yaml")),
        help="Path to rulebook_config.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: whale-{date}.md from input)",
    )
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--top-symbols", type=int, default=200)
    parser.add_argument("--top-trades", type=int, default=500)
    parser.add_argument("--top-raw-trades", type=int, default=500)
    parser.add_argument("--top-rejected", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    if is_split_bot_eod_part(input_path):
        raise SystemExit(
            "generate_whale_summary requires the full bot-eod-report-YYYY-MM-DD.csv/.zip; "
            f"split part files are not accepted: {input_path}"
        )
    config_path = resolve_config_path(Path(args.config))
    if args.output is None:
        date_str = infer_date_from_path(input_path)
        output_path = Path(f"whale-{date_str}.md")
    else:
        output_path = Path(args.output)

    config = load_config(config_path)
    top_symbols_limit = max(1, int(args.top_symbols))
    top_trades_limit = max(200, int(args.top_trades))
    top_raw_trades_limit = max(1, int(args.top_raw_trades))
    top_rejected_limit = max(1, int(args.top_rejected))

    exclude_etfs = bool(config["gates"].get("exclude_etfs", True))
    exclude_issue_types = {t.upper() for t in config["gates"].get("exclude_issue_types", ["ETF"])}
    min_credit_pct = float(config["gates"].get("min_credit_pct_width", 0.25))
    max_credit_pct = float(config["gates"].get("max_credit_pct_width", 0.55))
    max_debit_pct = float(config["gates"].get("max_debit_pct_width", 0.55))
    width_tiers = config["gates"].get("width_tiers", [])
    min_open_interest = int(config["gates"].get("min_leg_open_interest", 0))
    max_strike_dist_pct = float(config["gates"].get("max_strike_distance_pct", 1.0))
    min_premium = float(config["gates"].get("min_whale_premium", 0))

    shield_dte_min, shield_dte_max = config["shield"]["dte_range"]
    fire_dte_min, fire_dte_max = config["fire"]["dte_range"]

    use_anchor = bool(config["shield"].get("use_anchor_whitelist", False))
    anchor_whitelist = [sym.upper() for sym in config["shield"].get("anchor_whitelist", [])]
    anchor_set = set(anchor_whitelist)

    total_rows = 0
    yes_prime_rows = 0

    track_counter = Counter()
    option_type_counter = Counter()
    side_counter = Counter()
    symbol_stats = {}
    raw_symbol_stats = {}
    reject_counter = Counter()

    top_trades = pd.DataFrame()
    raw_top_trades = pd.DataFrame()
    rejected_top_trades = pd.DataFrame()
    report_date = infer_date_from_path(input_path)
    report_date_ts = pd.to_datetime(report_date, errors="coerce")

    usecols = bot_eod_usecols(input_path)
    dtype = {col: BOT_EOD_DTYPE[col] for col in usecols}

    with open_bot_eod(input_path) as (input_handle, input_label):
        for chunk in pd.read_csv(
            input_handle,
            chunksize=args.chunksize,
            usecols=usecols,
            dtype=dtype,
        ):
            total_rows += len(chunk)

            chunk["side"] = chunk["side"].fillna("no_side").astype("string").str.lower()
            chunk["option_type"] = chunk["option_type"].fillna("unknown").astype("string").str.lower()
            chunk["underlying_symbol"] = chunk["underlying_symbol"].fillna("").astype("string").str.upper()
            chunk["equity_type"] = chunk["equity_type"].fillna("").astype("string")
            if "canceled" in chunk.columns:
                chunk["canceled"] = chunk["canceled"].fillna("").astype("string").str.lower()
            infer_price_and_premium(chunk)

            executed_date = pd.to_datetime(
                chunk["executed_at"].str.slice(0, 10), errors="coerce"
            )
            if not pd.isna(report_date_ts):
                executed_date = executed_date.fillna(report_date_ts)
            expiry_date = pd.to_datetime(chunk["expiry"], errors="coerce")
            dte = (expiry_date - executed_date).dt.days

            underlying_price = pd.to_numeric(chunk["underlying_price"], errors="coerce").to_numpy()
            price = pd.to_numeric(chunk["price"], errors="coerce").to_numpy()
            width = build_width(underlying_price, width_tiers)
            pct_width = price / width

            side = chunk["side"].to_numpy()
            side_ok = np.isin(side, ["bid", "ask"])
            net_type = np.where(side == "bid", "credit", np.where(side == "ask", "debit", "unknown"))
            track = np.where(net_type == "credit", "SHIELD", np.where(net_type == "debit", "FIRE", "UNKNOWN"))

            top_cols = [
                "underlying_symbol",
                "option_chain_id",
                "track",
                "net_type",
                "option_type",
                "side",
                "executed_at",
                "expiry",
                "dte",
                "underlying_price",
                "nbbo_bid",
                "nbbo_ask",
                "ewma_nbbo_bid",
                "ewma_nbbo_ask",
                "strike",
                "price",
                "width",
                "pct_width",
                "size",
                "volume",
                "premium",
                "open_interest",
                "implied_volatility",
                "delta",
                "theta",
                "gamma",
                "vega",
                "rho",
                "theo",
                "sector",
                "exchange",
                "report_flags",
                "canceled",
                "upstream_condition_detail",
                "equity_type",
            ]

            enriched = chunk.copy()
            enriched["track"] = track
            enriched["net_type"] = net_type
            enriched["dte"] = dte
            enriched["width"] = width
            enriched["pct_width"] = pct_width
            top_cols = [col for col in top_cols if col in enriched.columns]

            update_symbol_stats(raw_symbol_stats, enriched)
            raw_top_trades = keep_top_premium(raw_top_trades, enriched[top_cols].copy(), top_raw_trades_limit)

            valid_market_data = ~np.isnan(width) & ~np.isnan(pct_width) & dte.notna().to_numpy()
            issue_ok = np.ones(len(chunk), dtype=bool)

            if exclude_etfs:
                eq_type_upper = chunk["equity_type"].str.upper().to_numpy()
                for issue_type in exclude_issue_types:
                    issue_ok &= eq_type_upper != issue_type

            canceled_ok = np.ones(len(chunk), dtype=bool)
            if "canceled" in chunk.columns:
                canceled_ok &= ~chunk["canceled"].isin({"t", "true", "1", "yes", "y"}).to_numpy()

            width_price_ok = (
                ((net_type == "credit") & (pct_width >= min_credit_pct) & (pct_width <= max_credit_pct))
                | ((net_type == "debit") & (pct_width <= max_debit_pct))
            )

            dte_arr = dte.to_numpy()
            dte_ok = (
                ((track == "SHIELD") & (dte_arr >= shield_dte_min) & (dte_arr <= shield_dte_max))
                | ((track == "FIRE") & (dte_arr >= fire_dte_min) & (dte_arr <= fire_dte_max))
            )

            anchor_ok = np.ones(len(chunk), dtype=bool)
            if use_anchor:
                anchor_ok &= (track != "SHIELD") | (chunk["underlying_symbol"].isin(anchor_set).to_numpy())

            premium_ok = np.ones(len(chunk), dtype=bool)
            if min_premium > 0:
                premium_ok &= chunk["premium"].to_numpy() >= min_premium

            oi_ok = np.ones(len(chunk), dtype=bool)
            if min_open_interest > 0:
                oi_vals = chunk["open_interest"].to_numpy()
                oi_ok &= (oi_vals >= min_open_interest) | np.isnan(oi_vals)

            strike_ok = np.ones(len(chunk), dtype=bool)
            if max_strike_dist_pct < 1.0:
                strike_vals = chunk["strike"].to_numpy()
                strike_dist = np.abs(strike_vals - underlying_price) / np.where(
                    underlying_price > 0, underlying_price, 1.0
                )
                strike_ok &= (strike_dist <= max_strike_dist_pct) | np.isnan(strike_dist)

            failure_masks = {
                "missing price/width/dte": pd.Series(~valid_market_data, index=chunk.index),
                "unsupported side": pd.Series(~side_ok, index=chunk.index),
                "canceled trade": pd.Series(~canceled_ok, index=chunk.index),
                "excluded issue type": pd.Series(~issue_ok, index=chunk.index),
                "price outside width band": pd.Series(~width_price_ok, index=chunk.index),
                "outside track dte range": pd.Series(~dte_ok, index=chunk.index),
                "not in shield anchor whitelist": pd.Series(~anchor_ok, index=chunk.index),
                "below min premium": pd.Series(~premium_ok, index=chunk.index),
                "below min open interest": pd.Series(~oi_ok, index=chunk.index),
                "strike too far from spot": pd.Series(~strike_ok, index=chunk.index),
            }
            for reason, failed in failure_masks.items():
                reject_counter[reason] += int(failed.sum())

            mask = (
                valid_market_data
                & side_ok
                & canceled_ok
                & issue_ok
                & width_price_ok
                & dte_ok
                & anchor_ok
                & premium_ok
                & oi_ok
                & strike_ok
            )

            rejected = enriched.loc[~mask, top_cols].copy()
            if not rejected.empty:
                rejected = rejected.nlargest(top_rejected_limit, "premium")
                rejected = annotate_reject_reasons(rejected, failure_masks)
                rejected_top_trades = keep_top_premium(rejected_top_trades, rejected, top_rejected_limit)

            if not mask.any():
                continue

            yes_chunk = chunk.loc[mask].copy()
            yes_chunk["track"] = track[mask]
            yes_chunk["net_type"] = net_type[mask]
            yes_chunk["dte"] = dte[mask]
            yes_chunk["width"] = width[mask]
            yes_chunk["pct_width"] = pct_width[mask]

            yes_prime_rows += len(yes_chunk)

            track_counter.update(yes_chunk["track"].tolist())
            option_type_counter.update(yes_chunk["option_type"].tolist())
            side_counter.update(yes_chunk["side"].tolist())

            update_symbol_stats(symbol_stats, yes_chunk)

            candidate = yes_chunk[top_cols].copy()
            top_trades = keep_top_premium(top_trades, candidate, top_trades_limit)

    if total_rows == 0:
        raise SystemExit("No rows read from input file.")

    yes_prime_pct = (yes_prime_rows / total_rows) * 100

    track_df = pd.DataFrame(
        sorted(track_counter.items(), key=lambda x: x[1], reverse=True),
        columns=["track", "count"],
    )
    option_df = pd.DataFrame(
        sorted(option_type_counter.items(), key=lambda x: x[1], reverse=True),
        columns=["option_type", "count"],
    )
    side_df = pd.DataFrame(
        sorted(side_counter.items(), key=lambda x: x[1], reverse=True),
        columns=["side", "count"],
    )

    raw_symbol_df = symbol_stats_frame(raw_symbol_stats, top_symbols_limit)
    symbol_df = symbol_stats_frame(symbol_stats, top_symbols_limit)
    reject_df = pd.DataFrame(
        [
            {"reject_reason": reason, "row_count": count}
            for reason, count in reject_counter.items()
            if count
        ]
    )
    if not reject_df.empty:
        reject_df = reject_df.sort_values("row_count", ascending=False)

    def format_trade_numbers(df):
        if df.empty:
            return df
        df = df.sort_values("premium", ascending=False).copy()
        for int_col in ["dte", "size", "open_interest"]:
            if int_col in df.columns:
                df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

        for col, digits in [
            ("underlying_price", 3),
            ("nbbo_bid", 3),
            ("nbbo_ask", 3),
            ("ewma_nbbo_bid", 3),
            ("ewma_nbbo_ask", 3),
            ("strike", 3),
            ("price", 3),
            ("width", 3),
            ("pct_width", 4),
            ("volume", 0),
            ("premium", 2),
            ("implied_volatility", 4),
            ("delta", 4),
            ("theta", 4),
            ("gamma", 4),
            ("vega", 4),
            ("rho", 4),
            ("theo", 3),
        ]:
            if col in df.columns:
                df[col] = df[col].astype(float).round(digits)
        return df

    raw_top_trades = format_trade_numbers(raw_top_trades)
    top_trades = format_trade_numbers(top_trades)
    rejected_top_trades = format_trade_numbers(rejected_top_trades)
    legacy_top_trades = top_trades.head(200)
    legacy_rejected_top_trades = rejected_top_trades.head(200)

    lines = []
    lines.append(f"# GravityWhale Yes-Prime Summary ({report_date})")
    lines.append("")
    lines.append(f"Source: `{input_label}`")
    lines.append(f"Rulebook: `{config_path}`")
    lines.append("")
    lines.append("Filters applied:")
    lines.append(f"- exclude_issue_types: {sorted(exclude_issue_types)}")
    lines.append(f"- credit_pct_width: {min_credit_pct} - {max_credit_pct}")
    lines.append(f"- debit_pct_width <= {max_debit_pct}")
    lines.append(f"- SHIELD DTE range: {shield_dte_min}-{shield_dte_max}")
    lines.append(f"- FIRE DTE range: {fire_dte_min}-{fire_dte_max}")
    lines.append(f"- min_premium: ${min_premium:,.0f}")
    lines.append(f"- min_open_interest: {min_open_interest}")
    lines.append(f"- max_strike_distance: {max_strike_dist_pct:.0%}")
    lines.append(f"- SHIELD anchor whitelist: {use_anchor} ({len(anchor_whitelist)} tickers)")
    lines.append("")
    lines.append(f"Total rows scanned: {total_rows:,}")
    lines.append(f"Yes-Prime candidates: {yes_prime_rows:,} ({yes_prime_pct:.2f}%)")

    lines.append("")
    lines.append("## Raw Top Symbols by Total Premium (Before Filters)")
    lines.append(format_markdown_table(raw_symbol_df))

    lines.append("")
    lines.append(f"## Top {top_raw_trades_limit} Raw Trades by Premium (Before Filters)")
    if raw_top_trades.empty:
        lines.append("(no rows)")
    else:
        lines.append(format_markdown_table(raw_top_trades))

    lines.append("")
    lines.append("## Filter Reject Counts")
    lines.append(format_markdown_table(reject_df))

    lines.append("")
    lines.append("## Yes-Prime by Track")
    lines.append(format_markdown_table(track_df))

    lines.append("")
    lines.append("## Yes-Prime by Option Type")
    lines.append(format_markdown_table(option_df))

    lines.append("")
    lines.append("## Yes-Prime by Side")
    lines.append(format_markdown_table(side_df))

    lines.append("")
    lines.append("## Top Symbols by Total Premium (Yes-Prime)")
    lines.append(format_markdown_table(symbol_df))

    lines.append("")
    lines.append("## Top 200 Yes-Prime Trades by Premium")
    if legacy_top_trades.empty:
        lines.append("(no rows)")
    else:
        lines.append(format_markdown_table(legacy_top_trades))

    if top_trades_limit > 200:
        lines.append("")
        lines.append(f"## Top {top_trades_limit} Yes-Prime Trades by Premium (Extended Audit)")
        if top_trades.empty:
            lines.append("(no rows)")
        else:
            lines.append(format_markdown_table(top_trades))

    lines.append("")
    lines.append("## Top 200 Rejected Trades by Premium (Audit)")
    if legacy_rejected_top_trades.empty:
        lines.append("(no rows)")
    else:
        lines.append(format_markdown_table(legacy_rejected_top_trades))

    if top_rejected_limit > 200:
        lines.append("")
        lines.append(f"## Top {top_rejected_limit} Rejected Trades by Premium (Extended Audit)")
        if rejected_top_trades.empty:
            lines.append("(no rows)")
        else:
            lines.append(format_markdown_table(rejected_top_trades))

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
