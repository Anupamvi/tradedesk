import argparse
import io
import re
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

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
    return df.to_markdown(index=False)


@contextmanager
def open_input(path):
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        try:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise SystemExit(f"No CSV found inside zip: {path}")
            csv_name = csv_names[0]
            handle = zf.open(csv_name, "r")
            text_handle = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            try:
                yield text_handle, f"{path}::{csv_name}"
            finally:
                text_handle.close()
                handle.close()
        finally:
            zf.close()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            yield handle, str(path)


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
    args = parser.parse_args()

    input_path = Path(args.input)
    config_path = resolve_config_path(Path(args.config))
    if args.output is None:
        date_str = infer_date_from_path(input_path)
        output_path = Path(f"whale-{date_str}.md")
    else:
        output_path = Path(args.output)

    config = load_config(config_path)

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

    top_trades = pd.DataFrame()

    usecols = [
        "executed_at",
        "underlying_symbol",
        "side",
        "strike",
        "option_type",
        "expiry",
        "underlying_price",
        "price",
        "size",
        "premium",
        "open_interest",
        "equity_type",
        "implied_volatility",
        "delta",
    ]

    dtype = {
        "executed_at": "string",
        "underlying_symbol": "string",
        "side": "string",
        "strike": "float64",
        "option_type": "string",
        "expiry": "string",
        "underlying_price": "float64",
        "price": "float64",
        "size": "float64",
        "premium": "float64",
        "open_interest": "float64",
        "equity_type": "string",
        "implied_volatility": "float64",
        "delta": "float64",
    }

    with open_input(input_path) as (input_handle, input_label):
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

            executed_date = pd.to_datetime(
                chunk["executed_at"].str.slice(0, 10), errors="coerce"
            )
            expiry_date = pd.to_datetime(chunk["expiry"], errors="coerce")
            dte = (expiry_date - executed_date).dt.days

            underlying_price = pd.to_numeric(chunk["underlying_price"], errors="coerce").to_numpy()
            price = pd.to_numeric(chunk["price"], errors="coerce").to_numpy()
            width = build_width(underlying_price, width_tiers)
            pct_width = price / width

            side = chunk["side"].to_numpy()
            net_type = np.where(side == "bid", "credit", "debit")
            track = np.where(net_type == "credit", "SHIELD", "FIRE")

            mask = ~np.isnan(width) & ~np.isnan(pct_width) & dte.notna().to_numpy()

            if exclude_etfs:
                eq_type_upper = chunk["equity_type"].str.upper().to_numpy()
                for issue_type in exclude_issue_types:
                    mask &= eq_type_upper != issue_type

            mask &= (
                ((net_type == "credit") & (pct_width >= min_credit_pct) & (pct_width <= max_credit_pct))
                | ((net_type == "debit") & (pct_width <= max_debit_pct))
            )

            mask &= (
                ((track == "SHIELD") & (dte.to_numpy() >= shield_dte_min) & (dte.to_numpy() <= shield_dte_max))
                | ((track == "FIRE") & (dte.to_numpy() >= fire_dte_min) & (dte.to_numpy() <= fire_dte_max))
            )

            if use_anchor:
                mask &= ((track != "SHIELD") | (chunk["underlying_symbol"].isin(anchor_set).to_numpy()))

            if min_premium > 0:
                mask &= chunk["premium"].to_numpy() >= min_premium

            if min_open_interest > 0:
                oi_vals = chunk["open_interest"].to_numpy()
                mask &= (oi_vals >= min_open_interest) | np.isnan(oi_vals)

            if max_strike_dist_pct < 1.0:
                strike_vals = chunk["strike"].to_numpy()
                strike_dist = np.abs(strike_vals - underlying_price) / np.where(
                    underlying_price > 0, underlying_price, 1.0
                )
                mask &= (strike_dist <= max_strike_dist_pct) | np.isnan(strike_dist)

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

            grouped = yes_chunk.groupby("underlying_symbol", dropna=False)["premium"].agg(["count", "sum"]).reset_index()
            for _, row in grouped.iterrows():
                symbol = row["underlying_symbol"]
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = [0, 0.0]
                symbol_stats[symbol][0] += int(row["count"])
                symbol_stats[symbol][1] += float(row["sum"])

            top_cols = [
                "underlying_symbol",
                "track",
                "net_type",
                "option_type",
                "side",
                "expiry",
                "dte",
                "underlying_price",
                "strike",
                "price",
                "width",
                "pct_width",
                "size",
                "premium",
                "open_interest",
                "implied_volatility",
                "delta",
                "equity_type",
            ]

            candidate = yes_chunk[top_cols].copy()
            candidate = candidate.nlargest(200, "premium")
            top_trades = pd.concat([top_trades, candidate], ignore_index=True)
            top_trades = top_trades.nlargest(200, "premium").reset_index(drop=True)

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

    symbol_df = pd.DataFrame(
        [
            {"underlying_symbol": k, "count": v[0], "total_premium": v[1]}
            for k, v in symbol_stats.items()
        ]
    )
    symbol_df = symbol_df.sort_values("total_premium", ascending=False).head(50)

    if not top_trades.empty:
        top_trades = top_trades.sort_values("premium", ascending=False)
        top_trades["dte"] = top_trades["dte"].astype("Int64")
        top_trades["size"] = top_trades["size"].astype("Int64")
        top_trades["open_interest"] = top_trades["open_interest"].astype("Int64")

        for col, digits in [
            ("underlying_price", 3),
            ("strike", 3),
            ("price", 3),
            ("width", 3),
            ("pct_width", 4),
            ("premium", 2),
            ("implied_volatility", 4),
            ("delta", 4),
        ]:
            top_trades[col] = top_trades[col].astype(float).round(digits)

    report_date = infer_date_from_path(input_path)

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
    if top_trades.empty:
        lines.append("(no rows)")
    else:
        lines.append(format_markdown_table(top_trades))

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

