import argparse
import datetime as dt
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eod_trade_scan_mode_a import (
    build_best_candidates,
    build_quotes,
    ensure_cols,
    fnum,
    md_tables,
)


REQ_CSV_PREFIXES = [
    "chain-oi-changes-",
    "dp-eod-report-",
    "hot-chains-",
    "stock-screener-",
]


def unzip_inputs_if_needed(base_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.csv"))
    if existing:
        return
    zips = sorted(base_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No input CSV/ZIP files found in {base_dir}")
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
            if not names:
                continue
            name = names[0]
            target = out_dir / Path(name).name
            with zf.open(name, "r") as src:
                target.write_bytes(src.read())


def detect_asof_from_names(paths):
    pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    vals = []
    for p in paths:
        m = pat.search(p.name)
        if m:
            vals.append(m.group(1))
    if not vals:
        raise ValueError("Could not detect as-of date from filenames.")
    return sorted(vals)[-1]


def pick_csvs(base_dir: Path):
    unz = base_dir / "_unzipped_mode_a"
    unzip_inputs_if_needed(base_dir, unz)
    csvs = sorted(unz.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {unz}")

    out = {}
    for pref in REQ_CSV_PREFIXES:
        matches = [p for p in csvs if p.name.startswith(pref)]
        if not matches:
            raise FileNotFoundError(f"Missing required CSV prefix: {pref}")
        out[pref] = sorted(matches)[-1]
    return out


def round_strike(x):
    try:
        return round(float(x), 3)
    except Exception:
        return math.nan


def build_leg_map(quotes: pd.DataFrame):
    q = quotes.copy()
    q["k"] = list(
        zip(
            q["ticker"].astype(str).str.upper(),
            q["right"].astype(str).str.upper(),
            q["expiry"],
            q["strike"].map(round_strike),
        )
    )
    return q.drop_duplicates("k").set_index("k")["option_symbol"].to_dict()


def strategy_right(strategy: str):
    s = str(strategy).strip()
    if s in {"Bull Call Debit", "Bear Call Credit"}:
        return "C"
    if s in {"Bear Put Debit", "Bull Put Credit"}:
        return "P"
    return ""


def action_cell(strategy: str, track: str, optimal: str):
    if optimal == "Watch Only":
        return "🔥🟨 WATCH ONLY" if str(track).upper() == "FIRE" else "🛡️🟨 WATCH ONLY"
    s = str(strategy)
    if s == "Bull Call Debit":
        return "🔥🟦 BULL CALL DEBIT"
    if s == "Bear Put Debit":
        return "🔥🟧 BEAR PUT DEBIT"
    if s == "Bull Put Credit":
        return "🛡️🟩 BULL PUT CREDIT"
    if s == "Bear Call Credit":
        return "🛡️🟥 BEAR CALL CREDIT"
    return s.upper()


def strike_setup(strategy, long_strike, short_strike, width):
    ls = float(long_strike)
    ss = float(short_strike)
    w = float(width)
    if strategy == "Bull Call Debit":
        return f"Buy {ls:.2f}C / Sell {ss:.2f}C ({w:.2f}w)"
    if strategy == "Bear Put Debit":
        return f"Buy {ls:.2f}P / Sell {ss:.2f}P ({w:.2f}w)"
    if strategy == "Bull Put Credit":
        return f"Sell {ss:.2f}P / Buy {ls:.2f}P ({w:.2f}w)"
    if strategy == "Bear Call Credit":
        return f"Sell {ss:.2f}C / Buy {ls:.2f}C ({w:.2f}w)"
    return "N/A"


def parse_gate_value(entry_gate: str):
    m = re.match(r"^\s*(>=|<=)\s*([0-9]*\.?[0-9]+)\s*(cr|db)\s*$", str(entry_gate).strip(), re.I)
    if not m:
        return None, None, None
    op, val, unit = m.groups()
    return op, float(val), unit.lower()


def calc_target_max(net_type: str, width: float, net: float):
    if net_type == "credit":
        return net * 100.0, (width - net) * 100.0
    return (width - net) * 100.0, net * 100.0


def calc_be(strategy, long_strike, short_strike, net):
    ls = float(long_strike)
    ss = float(short_strike)
    if strategy == "Bull Call Debit":
        return ls + net
    if strategy == "Bear Put Debit":
        return ls - net
    if strategy == "Bull Put Credit":
        return ss - net
    if strategy == "Bear Call Credit":
        return ss + net
    return math.nan


def money(x):
    return "N/A" if not np.isfinite(fnum(x)) else f"${float(x):,.2f}"


def px(x):
    return "N/A" if not np.isfinite(fnum(x)) else f"{float(x):.2f}"


def likelihood_strength(verdict: str, edge_pct: float, signals: float):
    v = str(verdict).strip().upper()
    e = fnum(edge_pct)
    n = fnum(signals)
    if v == "LOW_SAMPLE":
        return "Low Sample"
    if not np.isfinite(e):
        return "N/A"
    if e < 0:
        return "Negative-Strong" if e <= -15 else "Negative"
    if np.isfinite(n) and n < 100:
        return "Low Sample"
    if e >= 25:
        return "Strong"
    if e >= 10:
        return "Moderate"
    return "Weak"


def run():
    ap = argparse.ArgumentParser(description="MODE A two-stage runner (discovery + live execution)")
    ap.add_argument("--base-dir", default=r"c:\uw_root\2026-02-05")
    ap.add_argument("--config", default=r"c:\uw_root\rulebook_config.yaml")
    ap.add_argument("--out-dir", default=r"c:\uw_root\out")
    ap.add_argument("--top-trades", type=int, default=20)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = pick_csvs(base)
    asof_str = detect_asof_from_names(list(csvs.values()))
    asof = dt.datetime.strptime(asof_str, "%Y-%m-%d").date()

    if not args.output:
        output_path = base / f"anu-expert-trade-table-{asof_str}.md"
    else:
        output_path = Path(args.output).resolve()

    whale_md = base / f"whale-{asof_str}.md"
    if not whale_md.exists():
        alt = sorted(base.glob("whale-*.md"))
        whale_md = alt[-1] if alt else whale_md
    if not whale_md.exists():
        raise FileNotFoundError(f"Missing whale markdown in {base}")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hot_df = pd.read_csv(csvs["hot-chains-"], low_memory=False)
    oi_df = pd.read_csv(csvs["chain-oi-changes-"], low_memory=False)
    dp_df = pd.read_csv(csvs["dp-eod-report-"], low_memory=False)
    sc_df = pd.read_csv(csvs["stock-screener-"], low_memory=False)

    ensure_cols(hot_df, csvs["hot-chains-"].name, ["option_symbol", "date", "bid", "ask", "volume", "open_interest"])
    ensure_cols(
        oi_df,
        csvs["chain-oi-changes-"].name,
        ["option_symbol", "curr_date", "last_bid", "last_ask", "curr_oi", "volume"],
    )
    ensure_cols(
        sc_df,
        csvs["stock-screener-"].name,
        [
            "ticker",
            "close",
            "issue_type",
            "is_index",
            "next_earnings_date",
            "bullish_premium",
            "bearish_premium",
            "call_premium",
            "put_premium",
            "put_call_ratio",
        ],
    )
    _ = dp_df  # loaded intentionally; stage-1 model already relies on screener + quotes + whale tables.

    whale_tables = md_tables(whale_md.read_text(encoding="utf-8", errors="replace"))
    quotes = build_quotes(hot_df, oi_df, asof, csvs["hot-chains-"].name, csvs["chain-oi-changes-"].name)
    best = build_best_candidates(asof, cfg, sc_df, quotes, whale_tables, top_trades=args.top_trades)
    if not best:
        raise RuntimeError("No stage-1 candidates produced.")

    leg_map = build_leg_map(quotes)
    shortlist_rows = []
    for r in best:
        ticker = str(r["ticker"]).upper()
        strategy = str(r["strategy"])
        right = strategy_right(strategy)
        expiry = r["expiry"]
        long_strike = round_strike(r["long_strike"])
        short_strike = round_strike(r["short_strike"])

        if strategy in {"Bull Call Debit", "Bear Put Debit"}:
            long_key = (ticker, right, expiry, long_strike)
            short_key = (ticker, right, expiry, short_strike)
        else:
            short_key = (ticker, right, expiry, short_strike)
            long_key = (ticker, right, expiry, long_strike)

        long_leg = leg_map.get(long_key)
        short_leg = leg_map.get(short_key)
        if not long_leg or not short_leg:
            continue

        net = fnum(r.get("net"))
        net_type = str(r.get("net_type", "")).strip().lower()
        if not np.isfinite(net):
            continue
        entry_gate = f">= {net:.2f} cr" if net_type == "credit" else f"<= {net:.2f} db"

        shortlist_rows.append(
            {
                "ticker": ticker,
                "strategy": strategy,
                "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                "short_leg": short_leg,
                "long_leg": long_leg,
                "net_type": net_type,
                "entry_gate": entry_gate,
                "width": float(r["width"]),
                "conviction": int(r["conviction"]),
                "track": str(r.get("track", "")),
                "confidence_tier": str(r.get("tier", "")),
                "optimal_stage1": str(r.get("optimal", "")),
                "notes_stage1": str(r.get("notes", "")),
                "thesis": str(r.get("thesis", "")),
                "invalidation": str(r.get("invalidation", "")),
                "long_strike": float(r["long_strike"]),
                "short_strike": float(r["short_strike"]),
            }
        )

    shortlist = pd.DataFrame(shortlist_rows)
    if shortlist.empty:
        raise RuntimeError("No shortlist rows with valid leg symbols.")
    shortlist = shortlist.sort_values("conviction", ascending=False).head(max(1, int(args.top_trades))).reset_index(
        drop=True
    )
    shortlist_csv = out_dir / f"shortlist_trades_{asof_str}_mode_a.csv"
    shortlist.to_csv(shortlist_csv, index=False)

    likelihood_csv = out_dir / f"setup_likelihood_{asof_str}.csv"
    likelihood_cmd = [
        sys.executable,
        str((Path.cwd() / "setup_likelihood_backtest.py").resolve()),
        "--setups-csv",
        str(shortlist_csv),
        "--asof-date",
        asof_str,
        "--root-dir",
        str(Path.cwd().resolve()),
        "--out-dir",
        str(out_dir),
        "--lookback-years",
        "2",
        "--min-signals",
        "100",
    ]
    subprocess.run(likelihood_cmd, check=True)

    cmd = [
        sys.executable,
        str((Path.cwd() / "build_live_trade_table.py").resolve()),
        "--shortlist-csv",
        str(shortlist_csv),
        "--out-dir",
        str(out_dir),
        "--top",
        str(int(args.top_trades)),
        "--min-conviction",
        "0",
    ]
    subprocess.run(cmd, check=True)

    live_csv = out_dir / f"live_trade_table_{asof_str}.csv"
    live_final_csv = out_dir / f"live_trade_table_{asof_str}_final.csv"
    if not live_csv.exists():
        raise FileNotFoundError(f"Missing live output: {live_csv}")
    if not live_final_csv.exists():
        raise FileNotFoundError(f"Missing live final output: {live_final_csv}")

    live = pd.read_csv(live_csv, low_memory=False)
    key = ["ticker", "strategy", "expiry", "short_leg", "long_leg"]
    live_cols = key + [
        "live_status",
        "is_final_live_valid",
        "invalidation_breached_live",
        "invalidation_rule_op",
        "invalidation_rule_level",
        "invalidation_eval_price_live",
        "live_net_bid_ask",
        "live_max_profit",
        "live_max_loss",
        "gate_pass_live",
        "short_bid_live",
        "short_ask_live",
        "long_bid_live",
        "long_ask_live",
    ]
    mdf = shortlist.merge(live[live_cols], on=key, how="left", suffixes=("", "_live"))

    if likelihood_csv.exists():
        like_df = pd.read_csv(likelihood_csv, low_memory=False)
        like_df["ticker"] = like_df["ticker"].astype(str).str.upper().str.strip()
        like_df["strategy"] = like_df["strategy"].astype(str).str.strip()
        like_df["expiry"] = like_df["expiry"].astype(str).str[:10]
        like_df["entry_gate"] = like_df["entry_gate"].astype(str).str.strip()
        like_keep = [
            "ticker",
            "strategy",
            "expiry",
            "entry_gate",
            "hist_success_pct",
            "edge_pct",
            "signals",
            "verdict",
            "confidence",
        ]
        like_df = like_df[like_keep].drop_duplicates(subset=["ticker", "strategy", "expiry", "entry_gate"])
        mdf["entry_gate"] = mdf["entry_gate"].astype(str).str.strip()
        mdf["expiry"] = mdf["expiry"].astype(str).str[:10]
        mdf = mdf.merge(
            like_df,
            on=["ticker", "strategy", "expiry", "entry_gate"],
            how="left",
            suffixes=("", "_lk"),
        )
    else:
        mdf["hist_success_pct"] = np.nan
        mdf["edge_pct"] = np.nan
        mdf["signals"] = np.nan
        mdf["verdict"] = ""
        mdf["confidence"] = ""

    def is_approved(row):
        ok_live = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        invalidated_live = (
            bool(row.get("invalidation_breached_live"))
            if pd.notna(row.get("invalidation_breached_live"))
            else False
        )
        return (not invalidated_live) and ok_live and str(row.get("optimal_stage1", "")) in {"Yes-Prime", "Yes-Good"}

    mdf["approved"] = mdf.apply(is_approved, axis=1)
    mdf = mdf.sort_values(["approved", "conviction"], ascending=[False, False]).reset_index(drop=True)

    out_rows = []
    for i, r in mdf.iterrows():
        approved = bool(r["approved"])
        strategy = str(r["strategy"])
        net_type = str(r["net_type"]).lower()
        op, gate_val, _ = parse_gate_value(r.get("entry_gate", ""))

        live_net = fnum(r.get("live_net_bid_ask"))
        if np.isfinite(live_net):
            net_txt = f"{'Credit' if net_type == 'credit' else 'Debit'} {live_net:.2f} (Target {r['entry_gate']})"
            max_profit = money(r.get("live_max_profit"))
            max_loss = money(r.get("live_max_loss"))
            be_txt = px(calc_be(strategy, r["long_strike"], r["short_strike"], live_net))
        elif gate_val is not None:
            tgt_max_p, tgt_max_l = calc_target_max(net_type, float(r["width"]), gate_val)
            net_txt = f"Target {r['entry_gate']}"
            max_profit = money(tgt_max_p)
            max_loss = money(tgt_max_l)
            be_txt = px(calc_be(strategy, r["long_strike"], r["short_strike"], gate_val))
        else:
            net_txt = "N/A"
            max_profit = "N/A"
            max_loss = "N/A"
            be_txt = "N/A"

        if approved:
            confidence_tier = str(r.get("confidence_tier", ""))
            optimal = str(r.get("optimal_stage1", ""))
            notes = (
                f"Live executable; gate PASS ({r.get('entry_gate')}); short BID/ASK "
                f"{r.get('short_bid_live')}/{r.get('short_ask_live')}, long BID/ASK "
                f"{r.get('long_bid_live')}/{r.get('long_ask_live')}."
            )
        else:
            confidence_tier = "Watch Only"
            optimal = "Watch Only"
            cur_txt = f"{live_net:.2f}" if np.isfinite(live_net) else "N/A"
            invalidated_live = (
                bool(r.get("invalidation_breached_live"))
                if pd.notna(r.get("invalidation_breached_live"))
                else False
            )
            if invalidated_live:
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_live = fnum(r.get("invalidation_eval_price_live"))
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_live:.2f}" if np.isfinite(px_live) else "n/a"
                notes = (
                    f"Watch Only: live_status={r.get('live_status', 'missing')}; invalidation breached "
                    f"(rule: {inv_text}; level={lvl_txt}; live spot={px_txt})."
                )
            elif np.isfinite(live_net) and gate_val is not None:
                if net_type == "debit":
                    need = f"needs debit <= {gate_val:.2f}"
                else:
                    need = f"needs credit >= {gate_val:.2f}"
                notes = f"Watch Only: live_status={r.get('live_status', 'missing')}; {need}; current live net={cur_txt}."
            else:
                need = f"target {r.get('entry_gate', 'N/A')}"
                notes = f"Watch Only: live_status={r.get('live_status', 'missing')}; {need}; current live net={cur_txt}."

        hist_success = fnum(r.get("hist_success_pct"))
        edge_pct = fnum(r.get("edge_pct"))
        signals = fnum(r.get("signals"))
        verdict = str(r.get("verdict", "")).strip().upper()
        strength = likelihood_strength(verdict, edge_pct, signals)
        if np.isfinite(hist_success):
            n_txt = f"{int(signals)}" if np.isfinite(signals) else "n/a"
            verdict_txt = verdict if verdict else "N/A"
            setup_likelihood = f"{hist_success:.1f}% {verdict_txt} ({strength}, edge {edge_pct:+.1f}%, n={n_txt})"
        else:
            setup_likelihood = "N/A"

        out_rows.append(
            {
                "#": i + 1,
                "Ticker": r["ticker"],
                "Action": action_cell(strategy, str(r.get("track", "")), optimal),
                "Strategy Type": strategy,
                "Strike Setup": strike_setup(strategy, r["long_strike"], r["short_strike"], r["width"]),
                "Expiry": str(r["expiry"])[:10],
                "DTE": (dt.datetime.strptime(str(r["expiry"])[:10], "%Y-%m-%d").date() - asof).days,
                "Net Credit/Debit": net_txt,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Breakeven": be_txt,
                "Conviction %": f"{int(r['conviction'])}%",
                "Setup Likelihood": setup_likelihood,
                "Confidence Tier": confidence_tier,
                "Optimal": optimal,
                "Notes": notes,
                "Source": "Stage1(ChainOI+DP+HotChains+Screener+Whale) + Stage2(build_live_trade_table)",
            }
        )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "#",
            "Ticker",
            "Action",
            "Strategy Type",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Setup Likelihood",
            "Confidence Tier",
            "Optimal",
            "Notes",
            "Source",
        ],
    )

    lines = [
        f"As-of date used: {asof_str}",
        "Files used: "
        + ", ".join(
            [
                csvs["chain-oi-changes-"].name,
                csvs["dp-eod-report-"].name,
                csvs["hot-chains-"].name,
                csvs["stock-screener-"].name,
                whale_md.name,
                shortlist_csv.name,
                likelihood_csv.name,
                live_csv.name,
                live_final_csv.name,
            ]
        ),
        "",
        "## Anu Expert Trade Table",
        out_df.to_markdown(index=False),
        "",
        "Ticker thesis + invalidation (Yes-Prime / Yes-Good):",
    ]

    seen = set()
    for _, r in mdf.iterrows():
        if not bool(r["approved"]):
            continue
        t = str(r["ticker"])
        if t in seen:
            continue
        seen.add(t)
        lines.append(f"- {t}: {str(r.get('thesis', '')).strip()} Invalidation: {str(r.get('invalidation', '')).strip()}")
    if not seen:
        lines.append("- none")

    output_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote: {output_path}")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print("\n".join(lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    run()
