#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analyze_trading_year import (
    option_strategy_name,
    parse_broker_date_value,
    parse_float,
    parse_symbol_meta,
    schwab_delta_from_action,
)
from exact_spread_backtester import (
    HistoricalOptionQuoteStore,
    UnderlyingCloseStore,
    build_occ_symbol,
)


def _safe_float(x: object) -> float:
    try:
        if x is None or pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def _intrinsic(option_type: str, strike: float, spot: float) -> float:
    ot = str(option_type).upper().strip()
    if ot == "CALL":
        return max(0.0, float(spot) - float(strike))
    return max(0.0, float(strike) - float(spot))


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Schwab columns: {missing}")


def reconstruct_closed_trades(input_csv: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    raw = pd.read_csv(input_csv, low_memory=False)
    _require_cols(raw, ["Date", "Action", "Symbol", "Quantity", "Amount"])

    events: List[Dict[str, object]] = []
    rows_total = len(raw)
    rows_ignored = 0
    for i, r in raw.iterrows():
        d = parse_broker_date_value(r["Date"])
        qty = parse_float(r["Quantity"])
        amount = parse_float(r["Amount"])
        price = parse_float(r.get("Price"))
        if pd.isna(d) or not np.isfinite(qty) or qty == 0:
            rows_ignored += 1
            continue
        if not np.isfinite(amount):
            amount = 0.0

        instrument = str(r["Symbol"] or "").strip().upper()
        underlying, is_option, option_type = parse_symbol_meta(instrument)
        delta = schwab_delta_from_action(r["Action"], float(qty), is_option)
        if delta == 0:
            rows_ignored += 1
            continue

        expiry = None
        strike = math.nan
        if is_option:
            # parse_symbol_meta already validated this pattern; split once for expiry/strike.
            parts = instrument.split()
            if len(parts) >= 4:
                try:
                    expiry = pd.to_datetime(parts[1], format="%m/%d/%Y", errors="coerce")
                    expiry = None if pd.isna(expiry) else expiry.date()
                except Exception:
                    expiry = None
                strike = _safe_float(parts[2])

        events.append(
            {
                "row_id": int(i),
                "date": pd.to_datetime(d),
                "action_raw": str(r["Action"]),
                "instrument": instrument,
                "underlying": underlying,
                "is_option": bool(is_option),
                "option_type": option_type,
                "expiry": expiry,
                "strike": strike,
                "qty_raw": float(qty),
                "delta_qty": float(delta),
                "amount": float(amount),
                "price": float(price) if np.isfinite(price) else math.nan,
            }
        )

    events = sorted(events, key=lambda x: (x["date"], x["row_id"]))
    book: Dict[str, Dict[str, deque]] = defaultdict(lambda: {"long": deque(), "short": deque()})
    closed_rows: List[Dict[str, object]] = []

    for ev in events:
        inst = str(ev["instrument"])
        side_book = book[inst]
        qty_abs = abs(float(ev["delta_qty"]))
        if qty_abs <= 0:
            continue

        close_unit_cash = float(ev["amount"]) / qty_abs
        action = str(ev["action_raw"]).strip().lower()
        can_open_long = action in {"buy to open", "buy", "reinvest shares"}
        can_open_short = action in {"sell to open", "sell"}

        def close_one(lot_side: str, remaining_qty: float) -> float:
            q = remaining_qty
            qbook = side_book[lot_side]
            while q > 1e-9 and qbook:
                lot = qbook[0]
                m = min(q, float(lot["qty"]))
                open_unit_cash = float(lot["unit_cash"])
                realized = (open_unit_cash + close_unit_cash) * m

                entry_price = _safe_float(lot["entry_price"])
                if not np.isfinite(entry_price):
                    denom = 100.0 if bool(lot["is_option"]) else 1.0
                    entry_price = abs(open_unit_cash) / max(1e-9, denom)

                exit_price_actual = _safe_float(ev["price"])
                if not np.isfinite(exit_price_actual):
                    if action in {"expired", "assigned", "exchange or exercise"}:
                        exit_price_actual = 0.0
                    else:
                        denom = 100.0 if bool(ev["is_option"]) else 1.0
                        exit_price_actual = abs(close_unit_cash) / max(1e-9, denom)

                closed_rows.append(
                    {
                        "source_file": str(input_csv),
                        "open_date": pd.to_datetime(lot["open_date"]).date(),
                        "close_date": pd.to_datetime(ev["date"]).date(),
                        "symbol": str(lot["underlying"]),
                        "instrument": str(lot["instrument"]),
                        "strategy": str(lot["strategy"]),
                        "side": str(lot["side"]),
                        "qty": float(m),
                        "entry_price": float(entry_price),
                        "exit_price_actual": float(exit_price_actual),
                        "realized_pnl": float(realized),
                        "is_option": bool(lot["is_option"]),
                        "option_type": str(lot["option_type"]),
                        "expiry": lot["expiry"],
                        "strike": _safe_float(lot["strike"]),
                    }
                )
                lot["qty"] = float(lot["qty"]) - float(m)
                q -= float(m)
                if lot["qty"] <= 1e-9:
                    qbook.popleft()
            return q

        delta = float(ev["delta_qty"])
        if delta > 0:
            remaining = close_one("short", qty_abs)
            if remaining > 1e-9 and can_open_long:
                side_book["long"].append(
                    {
                        "qty": remaining,
                        "unit_cash": close_unit_cash,
                        "open_date": ev["date"],
                        "entry_price": ev["price"],
                        "side": "LONG",
                        "is_option": bool(ev["is_option"]),
                        "option_type": str(ev["option_type"]),
                        "underlying": str(ev["underlying"]),
                        "instrument": str(ev["instrument"]),
                        "expiry": ev["expiry"],
                        "strike": ev["strike"],
                        "strategy": option_strategy_name("LONG", str(ev["option_type"]))
                        if bool(ev["is_option"])
                        else "Long Stock",
                    }
                )
        else:
            remaining = close_one("long", qty_abs)
            if remaining > 1e-9 and can_open_short:
                side_book["short"].append(
                    {
                        "qty": remaining,
                        "unit_cash": close_unit_cash,
                        "open_date": ev["date"],
                        "entry_price": ev["price"],
                        "side": "SHORT",
                        "is_option": bool(ev["is_option"]),
                        "option_type": str(ev["option_type"]),
                        "underlying": str(ev["underlying"]),
                        "instrument": str(ev["instrument"]),
                        "expiry": ev["expiry"],
                        "strike": ev["strike"],
                        "strategy": option_strategy_name("SHORT", str(ev["option_type"]))
                        if bool(ev["is_option"])
                        else "Short Stock",
                    }
                )

    out = pd.DataFrame(closed_rows)
    if out.empty:
        meta = {"rows_total": rows_total, "rows_ignored": rows_ignored, "closed_trades": 0}
        return out, meta
    out["open_date"] = pd.to_datetime(out["open_date"], errors="coerce").dt.date
    out["close_date"] = pd.to_datetime(out["close_date"], errors="coerce").dt.date
    out["holding_days"] = (
        pd.to_datetime(out["close_date"], errors="coerce") - pd.to_datetime(out["open_date"], errors="coerce")
    ).dt.days
    out = out.sort_values(["close_date", "symbol", "strategy"]).reset_index(drop=True)
    meta = {"rows_total": rows_total, "rows_ignored": rows_ignored, "closed_trades": int(len(out))}
    return out, meta


def estimate_exit_and_sim_pnl(
    trades: pd.DataFrame,
    quote_store: HistoricalOptionQuoteStore,
    close_store: UnderlyingCloseStore,
    close_lookback_days: int = 7,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in trades.itertuples(index=False):
        symbol = str(r.symbol).upper().strip()
        side = str(r.side).upper().strip()
        is_option = bool(r.is_option)
        qty = float(r.qty) if np.isfinite(_safe_float(r.qty)) else 1.0
        entry = float(r.entry_price) if np.isfinite(_safe_float(r.entry_price)) else math.nan
        close_date = r.close_date
        if not isinstance(close_date, dt.date):
            continue

        sim_exit = math.nan
        exit_source = "missing"

        if is_option:
            expiry = r.expiry if isinstance(r.expiry, dt.date) else None
            strike = _safe_float(r.strike)
            option_type = str(r.option_type).upper().strip()
            right = "C" if option_type == "CALL" else "P"
            if expiry and np.isfinite(strike):
                occ = build_occ_symbol(symbol, expiry, right, float(strike))
                q = quote_store.get_leg_quote(close_date, occ)
                if q is not None and np.isfinite(_safe_float(q.mid)) and q.mid >= 0:
                    sim_exit = float(q.mid)
                    exit_source = "option_quote_mid"
                else:
                    spot = close_store.get_close_on_or_before(symbol, close_date, lookback_days=close_lookback_days)
                    if spot is not None and np.isfinite(spot):
                        sim_exit = _intrinsic(option_type, float(strike), float(spot))
                        exit_source = "intrinsic_fallback"
        else:
            spot = close_store.get_close_on_or_before(symbol, close_date, lookback_days=close_lookback_days)
            if spot is not None and np.isfinite(spot):
                sim_exit = float(spot)
                exit_source = "underlying_close"

        if not np.isfinite(sim_exit) or not np.isfinite(entry):
            continue

        mult = 100.0 if is_option else 1.0
        if side == "LONG":
            sim_pnl = (float(sim_exit) - float(entry)) * mult * qty
        else:
            sim_pnl = (float(entry) - float(sim_exit)) * mult * qty

        rows.append(
            {
                **r._asdict(),
                "sim_exit_price": float(sim_exit),
                "exit_source": exit_source,
                "sim_pnl": float(sim_pnl),
                "actual_profit": bool(float(r.realized_pnl) > 0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["close_date"] = pd.to_datetime(out["close_date"], errors="coerce")
    out = out.sort_values(["close_date", "symbol", "strategy"]).reset_index(drop=True)
    return out


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.5
    tnr = float((y_pred[neg] == 0).mean()) if neg.any() else 0.5
    return 0.5 * (tpr + tnr)


def calibrate_threshold_band(hist: pd.DataFrame, min_history: int) -> Tuple[float, float, int]:
    n = int(len(hist))
    if n < int(min_history):
        return 0.0, 0.0, n

    s = pd.to_numeric(hist["sim_pnl"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(hist["actual_profit"], errors="coerce").astype(int).to_numpy(dtype=int)
    mask = np.isfinite(s)
    s = s[mask]
    y = y[mask]
    if len(s) < int(min_history):
        return 0.0, 0.0, int(len(s))

    quantiles = np.linspace(0.1, 0.9, 17)
    threshold_candidates = sorted(set([0.0] + [float(x) for x in np.quantile(s, quantiles)]))

    best_obj = -1e18
    best_threshold = 0.0
    best_band = 0.0
    for th in threshold_candidates:
        margin = s - th
        base_pred = (margin > 0).astype(int)
        abs_margin = np.abs(margin)
        band_candidates = sorted(set([0.0] + [float(x) for x in np.quantile(abs_margin, [0.1, 0.2, 0.3, 0.4])]))
        for band in band_candidates:
            keep = abs_margin >= band
            if keep.sum() < max(8, int(0.55 * len(s))):
                continue
            acc_keep = _balanced_accuracy(y[keep], base_pred[keep])
            coverage = float(keep.mean())
            # Penalize over-abstaining, but allow mild abstain for higher precision.
            obj = acc_keep - 0.10 * (1.0 - coverage)
            if obj > best_obj:
                best_obj = obj
                best_threshold = float(th)
                best_band = float(band)

    return best_threshold, best_band, int(len(s))


def walk_forward_predict(
    scored: pd.DataFrame,
    min_history: int,
    min_abs_margin: float,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    df = scored.copy()
    df = df.sort_values(["close_date", "symbol", "strategy"]).reset_index(drop=True)

    preds: List[str] = []
    thresholds: List[float] = []
    bands: List[float] = []
    hist_ns: List[int] = []
    reasons: List[str] = []

    for i, r in df.iterrows():
        strat = str(r["strategy"])
        hist = df.iloc[:i]
        hist = hist[hist["strategy"] == strat]
        th, band, n_hist = calibrate_threshold_band(hist, min_history=min_history)
        eff_band = max(float(band), float(min_abs_margin))

        margin = float(r["sim_pnl"]) - th
        if abs(margin) <= eff_band:
            pred = "ABSTAIN"
            reason = "within_confidence_band"
        elif margin > 0:
            pred = "PASS"
            reason = "above_threshold"
        else:
            pred = "FAIL"
            reason = "below_threshold"

        preds.append(pred)
        thresholds.append(float(th))
        bands.append(float(eff_band))
        hist_ns.append(int(n_hist))
        reasons.append(reason)

    df["pred_result"] = preds
    df["threshold"] = thresholds
    df["confidence_band"] = bands
    df["history_n"] = hist_ns
    df["pred_reason"] = reasons
    return df


def summarize_predictions(pred_df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if pred_df.empty:
        out["trades_scored"] = 0
        return out

    out["trades_scored"] = int(len(pred_df))
    out["abstain_total"] = int((pred_df["pred_result"] == "ABSTAIN").sum())

    eff = pred_df[pred_df["pred_result"].isin(["PASS", "FAIL"])].copy()
    out["decision_trades"] = int(len(eff))
    out["decision_coverage"] = float(len(eff) / len(pred_df)) if len(pred_df) else math.nan
    if eff.empty:
        return out

    pass_total = int((eff["pred_result"] == "PASS").sum())
    fail_total = int((eff["pred_result"] == "FAIL").sum())
    pass_profit = int(((eff["pred_result"] == "PASS") & (eff["realized_pnl"] > 0)).sum())
    fail_loss = int(((eff["pred_result"] == "FAIL") & (eff["realized_pnl"] < 0)).sum())
    pass_loss = int(((eff["pred_result"] == "PASS") & (eff["realized_pnl"] < 0)).sum())
    fail_profit = int(((eff["pred_result"] == "FAIL") & (eff["realized_pnl"] > 0)).sum())
    hit = float(
        ((eff["pred_result"] == "PASS") == (pd.to_numeric(eff["realized_pnl"], errors="coerce") > 0)).mean()
    )

    out.update(
        {
            "pass_total": pass_total,
            "pass_profit": pass_profit,
            "pass_loss": pass_loss,
            "pass_profit_rate": float(pass_profit / pass_total) if pass_total else math.nan,
            "fail_total": fail_total,
            "fail_loss": fail_loss,
            "fail_profit": fail_profit,
            "fail_loss_rate": float(fail_loss / fail_total) if fail_total else math.nan,
            "directional_hit_rate": hit,
        }
    )
    return out


def baseline_summary(scored_df: pd.DataFrame) -> Dict[str, object]:
    if scored_df.empty:
        return {"trades_scored": 0}
    df = scored_df.copy()
    df["pred_result"] = np.where(pd.to_numeric(df["sim_pnl"], errors="coerce") > 0, "PASS", "FAIL")
    return summarize_predictions(df)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Full-year Schwab validation with quote-based option exits, per-strategy walk-forward calibration, and PASS/FAIL/ABSTAIN gating."
    )
    ap.add_argument("--input-csv", required=True, help="Schwab transactions CSV path.")
    ap.add_argument("--root-dir", default=r"c:\uw_root", help="Root directory with YYYY-MM-DD market data folders.")
    ap.add_argument("--out-dir", default=r"c:\uw_root\out\full_year_validation_pipeline_v3", help="Output directory.")
    ap.add_argument("--min-history", type=int, default=12, help="Min prior trades per strategy before calibration.")
    ap.add_argument(
        "--min-abs-margin",
        type=float,
        default=150.0,
        help="Minimum absolute simulated PnL margin required to classify PASS/FAIL (otherwise ABSTAIN).",
    )
    ap.add_argument("--close-lookback-days", type=int, default=7, help="Lookback for close lookup on/before close date.")
    ap.add_argument("--no-web-fallback", action="store_true", help="Disable yfinance fallback for missing underlying close.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reconstructed, recon_meta = reconstruct_closed_trades(input_csv)
    if reconstructed.empty:
        raise RuntimeError("No closed trades reconstructed from input.")

    quote_store = HistoricalOptionQuoteStore(root_dir=root_dir, use_hot=True, use_oi=True)
    close_store = UnderlyingCloseStore(root_dir=root_dir, allow_web_fallback=not args.no_web_fallback)
    scored = estimate_exit_and_sim_pnl(
        reconstructed,
        quote_store=quote_store,
        close_store=close_store,
        close_lookback_days=max(1, int(args.close_lookback_days)),
    )
    if scored.empty:
        raise RuntimeError("No trades could be scored (missing option/underlying marks).")

    pred_df = walk_forward_predict(
        scored,
        min_history=max(1, int(args.min_history)),
        min_abs_margin=max(0.0, float(args.min_abs_margin)),
    )
    base = baseline_summary(scored)
    improved = summarize_predictions(pred_df)

    # Strategy-level diagnostics.
    strat_rows = []
    for strat, g in pred_df.groupby("strategy", dropna=False):
        s = summarize_predictions(g)
        strat_rows.append(
            {
                "strategy": str(strat),
                "trades_scored": s.get("trades_scored", 0),
                "decision_trades": s.get("decision_trades", 0),
                "pass_total": s.get("pass_total", 0),
                "pass_profit": s.get("pass_profit", 0),
                "fail_total": s.get("fail_total", 0),
                "fail_loss": s.get("fail_loss", 0),
                "pass_profit_rate": s.get("pass_profit_rate", math.nan),
                "fail_loss_rate": s.get("fail_loss_rate", math.nan),
                "directional_hit_rate": s.get("directional_hit_rate", math.nan),
            }
        )
    by_strategy = pd.DataFrame(strat_rows).sort_values("trades_scored", ascending=False).reset_index(drop=True)

    # Save outputs.
    recon_csv = out_dir / "reconstructed_closed_trades.csv"
    scored_csv = out_dir / "scored_trades_with_features.csv"
    pred_csv = out_dir / "predictions_walkforward.csv"
    strat_csv = out_dir / "strategy_metrics_walkforward.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    reconstructed.to_csv(recon_csv, index=False)
    scored.to_csv(scored_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)
    by_strategy.to_csv(strat_csv, index=False)

    payload = {
        "input_csv": str(input_csv),
        "root_dir": str(root_dir),
        "reconstruction": recon_meta,
        "scored_trades": int(len(scored)),
        "baseline_no_calibration": base,
        "walkforward_strategy_calibrated": improved,
        "config": {
            "min_history": int(args.min_history),
            "min_abs_margin": float(args.min_abs_margin),
            "close_lookback_days": int(args.close_lookback_days),
            "web_fallback": not bool(args.no_web_fallback),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Full-Year Validation (Pipeline v3)",
        "",
        f"- Input: `{input_csv}`",
        f"- Closed trades reconstructed: `{recon_meta['closed_trades']}`",
        f"- Trades scored: `{len(scored)}`",
        "",
        "## Baseline (PASS if sim_pnl > 0)",
        f"- pass->profit: `{base.get('pass_profit', 0)}/{base.get('pass_total', 0)}`",
        f"- fail->loss: `{base.get('fail_loss', 0)}/{base.get('fail_total', 0)}`",
        f"- directional hit: `{base.get('directional_hit_rate', math.nan):.2%}`",
        "",
        "## Walk-Forward Calibrated (PASS/FAIL/ABSTAIN)",
        f"- decision trades: `{improved.get('decision_trades', 0)}/{improved.get('trades_scored', 0)}`",
        f"- pass->profit: `{improved.get('pass_profit', 0)}/{improved.get('pass_total', 0)}`",
        f"- fail->loss: `{improved.get('fail_loss', 0)}/{improved.get('fail_total', 0)}`",
        f"- directional hit: `{improved.get('directional_hit_rate', math.nan):.2%}`",
        "",
        "Artifacts:",
        f"- `{recon_csv}`",
        f"- `{scored_csv}`",
        f"- `{pred_csv}`",
        f"- `{strat_csv}`",
        f"- `{summary_json}`",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Input CSV: {input_csv}")
    print(f"Reconstructed closed trades: {recon_meta['closed_trades']}")
    print(f"Scored trades: {len(scored)}")
    print("Baseline:", base)
    print("Walk-forward:", improved)
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {pred_csv}")


if __name__ == "__main__":
    main()
