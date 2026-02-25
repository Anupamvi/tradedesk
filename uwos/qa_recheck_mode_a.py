import argparse
import json
from pathlib import Path

import pandas as pd

from uwos.eod_trade_scan_mode_a import md_tables


def parse_args():
    p = argparse.ArgumentParser(
        description="Artifact-level QA checks for MODE A two-stage output."
    )
    p.add_argument("--out-dir", required=True, help="Run output directory")
    p.add_argument("--date", required=False, help="As-of date (YYYY-MM-DD)")
    p.add_argument("--expect-top", type=int, default=20, help="Expected final row count")
    return p.parse_args()


def pick_file(out_dir: Path, prefix: str, asof: str | None):
    if asof:
        exact = sorted(out_dir.glob(f"{prefix}-{asof}*.md"))
        if exact:
            return exact[-1]
    matches = sorted(out_dir.glob(f"{prefix}-*.md"))
    if not matches:
        raise FileNotFoundError(f"Missing {prefix}-*.md in {out_dir}")
    return matches[-1]


def load_markdown_tables(md_path: Path):
    txt = md_path.read_text(encoding="utf-8-sig", errors="replace")
    return md_tables(txt)


def normalize_ticker(x):
    return str(x or "").strip().upper()


def normalize_strategy(x):
    return str(x or "").strip()


def normalize_expiry(x):
    return str(x or "").strip()[:10]


def fnum(x):
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def live_spot_from_row(row: pd.Series) -> float:
    last = fnum(row.get("spot_live_last"))
    bid = fnum(row.get("spot_live_bid"))
    ask = fnum(row.get("spot_live_ask"))
    if pd.notna(last):
        return float(last)
    if pd.notna(bid) and pd.notna(ask):
        return float((bid + ask) / 2.0)
    if pd.notna(bid):
        return float(bid)
    if pd.notna(ask):
        return float(ask)
    return float("nan")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    asof = args.date
    if not asof:
        manifests = sorted(out_dir.glob("run_manifest_*.json"))
        if manifests:
            asof = manifests[-1].stem.replace("run_manifest_", "")

    md_path = pick_file(out_dir, "anu-expert-trade-table", asof)
    live_csv = out_dir / f"live_trade_table_{asof}.csv" if asof else None
    if not (live_csv and live_csv.exists()):
        live_matches = sorted(out_dir.glob("live_trade_table_*.csv"))
        if not live_matches:
            raise FileNotFoundError(f"Missing live_trade_table_*.csv in {out_dir}")
        live_csv = live_matches[-1]

    manifest_path = out_dir / f"run_manifest_{asof}.json" if asof else None
    if not (manifest_path and manifest_path.exists()):
        manifests = sorted(out_dir.glob("run_manifest_*.json"))
        manifest_path = manifests[-1] if manifests else None

    tables = load_markdown_tables(md_path)
    live = pd.read_csv(live_csv, low_memory=False)
    for c in ["ticker", "strategy", "expiry", "live_status", "is_final_live_valid"]:
        if c not in live.columns:
            live[c] = pd.NA

    approved_parts = []
    watch_parts = []
    for k, df in tables.items():
        kk = k.strip().lower()
        if kk in {"approved - fire", "approved - shield"}:
            z = df.copy()
            z["__category"] = k
            approved_parts.append(z)
        if kk in {"watch only - fire", "watch only - shield"}:
            z = df.copy()
            z["__category"] = k
            watch_parts.append(z)

    approved = pd.concat(approved_parts, ignore_index=True) if approved_parts else pd.DataFrame()
    watch = pd.concat(watch_parts, ignore_index=True) if watch_parts else pd.DataFrame()
    total_rows = len(approved) + len(watch)

    findings = []

    def add(sev, msg):
        findings.append({"severity": sev, "message": msg})

    if total_rows != args.expect_top:
        add("HIGH", f"Final markdown rows = {total_rows}, expected = {args.expect_top}")

    if not approved.empty:
        if "Optimal" in approved.columns:
            bad = approved[approved["Optimal"].astype(str).str.contains("Watch", case=False, regex=True)]
            if not bad.empty:
                add("HIGH", f"Approved section contains Watch rows ({len(bad)})")
        if "Setup Likelihood" in approved.columns:
            weak = approved[
                approved["Setup Likelihood"].astype(str).str.contains(
                    r"FAIL|Negative|Weak", case=False, regex=True
                )
            ]
            if not weak.empty:
                add("HIGH", f"Approved section contains weak/negative likelihood rows ({len(weak)})")

        for _, r in approved.iterrows():
            t = normalize_ticker(r.get("Ticker"))
            s = normalize_strategy(r.get("Strategy Type"))
            e = normalize_expiry(r.get("Expiry"))
            sub = live[
                (live["ticker"].astype(str).str.upper() == t)
                & (live["strategy"].astype(str) == s)
                & (live["expiry"].astype(str).str[:10] == e)
            ]
            if sub.empty:
                add("HIGH", f"Approved row missing in live csv: {t} | {s} | {e}")

        if "Ticker" in approved.columns:
            vc = approved["Ticker"].astype(str).str.upper().value_counts()
            crowded = vc[vc >= 3]
            for t, n in crowded.items():
                add("MED", f"Approved concentration: {t} appears {int(n)} times")

    if not watch.empty and "Optimal" in watch.columns:
        non_watch = watch[~watch["Optimal"].astype(str).str.contains("Watch", case=False, regex=True)]
        if not non_watch.empty:
            add("HIGH", f"Watch section contains non-watch rows ({len(non_watch)})")

    manifest_counts = {}
    screener_spot_map = {}
    if manifest_path and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_counts = (manifest.get("counts") or {})
            appr = manifest_counts.get("approved_rows")
            if isinstance(appr, int) and appr != len(approved):
                add("HIGH", f"Manifest approved_rows={appr} but markdown approved={len(approved)}")
            final_rows = manifest_counts.get("final_output_rows")
            if isinstance(final_rows, int) and final_rows != total_rows:
                add("HIGH", f"Manifest final_output_rows={final_rows} but markdown rows={total_rows}")
            screener_csv = str((manifest.get("input_files") or {}).get("stock_screener_csv", "")).strip()
            if screener_csv:
                sp = Path(screener_csv)
                if sp.exists():
                    sc = pd.read_csv(sp, low_memory=False)
                    if "ticker" in sc.columns and "close" in sc.columns:
                        sc["ticker"] = sc["ticker"].astype(str).str.upper().str.strip()
                        sc["close"] = pd.to_numeric(sc["close"], errors="coerce")
                        screener_spot_map = (
                            sc.dropna(subset=["ticker"])
                            .drop_duplicates("ticker")
                            .set_index("ticker")["close"]
                            .to_dict()
                        )
        except Exception as exc:
            add("MED", f"Failed to parse manifest: {exc}")

    if not approved.empty:
        for _, r in approved.iterrows():
            t = normalize_ticker(r.get("Ticker"))
            s = normalize_strategy(r.get("Strategy Type"))
            e = normalize_expiry(r.get("Expiry"))
            sub = live[
                (live["ticker"].astype(str).str.upper() == t)
                & (live["strategy"].astype(str) == s)
                & (live["expiry"].astype(str).str[:10] == e)
            ]
            if sub.empty:
                continue
            lv = sub.iloc[0]
            live_spot = live_spot_from_row(lv)
            asof_spot = fnum(screener_spot_map.get(t))
            if pd.notna(asof_spot) and asof_spot > 0 and pd.notna(live_spot):
                drift = abs(float(live_spot) - float(asof_spot)) / float(asof_spot)
                if drift > 0.35:
                    add("HIGH", f"Approved {t} spot drift too high: asof={asof_spot:.2f}, live={live_spot:.2f}, drift={drift:.1%}")
            if not pd.notna(live_spot):
                add("HIGH", f"Approved {t} missing live spot in live csv")
                continue

            long_strike = fnum(lv.get("long_strike"))
            short_strike = fnum(lv.get("short_strike"))
            short_put = fnum(lv.get("short_put_strike"))
            short_call = fnum(lv.get("short_call_strike"))
            strategy = normalize_strategy(lv.get("strategy"))
            if strategy == "Bull Call Debit" and pd.notna(long_strike):
                if long_strike > float(live_spot) * 1.60:
                    add("HIGH", f"Approved {t} {strategy} long strike too far OTM vs spot ({long_strike:.2f} vs {live_spot:.2f})")
            if strategy == "Bear Put Debit" and pd.notna(long_strike):
                if long_strike < float(live_spot) * 0.40:
                    add("HIGH", f"Approved {t} {strategy} long strike too far below spot ({long_strike:.2f} vs {live_spot:.2f})")
            if strategy == "Bull Put Credit" and pd.notna(short_strike):
                if short_strike >= float(live_spot):
                    add("HIGH", f"Approved {t} {strategy} short put not below spot ({short_strike:.2f} vs {live_spot:.2f})")
            if strategy == "Bear Call Credit" and pd.notna(short_strike):
                if short_strike <= float(live_spot):
                    add("HIGH", f"Approved {t} {strategy} short call not above spot ({short_strike:.2f} vs {live_spot:.2f})")
            if strategy in {"Iron Condor", "Iron Butterfly"} and pd.notna(short_put) and pd.notna(short_call):
                if not (short_put < float(live_spot) < short_call):
                    add("HIGH", f"Approved {t} {strategy} spot not between short strikes ({short_put:.2f}/{short_call:.2f} vs {live_spot:.2f})")

    high = [f for f in findings if f["severity"] == "HIGH"]
    med = [f for f in findings if f["severity"] == "MED"]

    lines = []
    lines.append("# MODE A QA Recheck")
    lines.append("")
    lines.append(f"- Output dir: `{out_dir}`")
    lines.append(f"- Markdown: `{md_path.name}`")
    lines.append(f"- Live CSV: `{live_csv.name}`")
    if manifest_path:
        lines.append(f"- Manifest: `{manifest_path.name}`")
    lines.append(f"- Rows: approved={len(approved)}, watch={len(watch)}, total={total_rows}")
    if manifest_counts:
        lines.append(
            f"- Manifest counts: approved_rows={manifest_counts.get('approved_rows')}, "
            f"final_output_rows={manifest_counts.get('final_output_rows')}"
        )
    lines.append("")
    lines.append("## Findings")
    if not findings:
        lines.append("- PASS: no inconsistencies found in artifact checks.")
    else:
        for f in findings:
            lines.append(f"- [{f['severity']}] {f['message']}")

    report_path = out_dir / f"qa_recheck_{asof or 'latest'}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"\nWrote: {report_path}")

    if high:
        raise SystemExit(2)
    if med:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
