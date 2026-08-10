from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ._vendor.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

from .data import safe_float
from .schwab_live import chain_spot, chain_to_contracts


PIPELINE_VERSION_V42 = "v4.2-integrity-20260713"
TECHNICAL_HISTORY_DAYS = 320
TECHNICAL_MAX_TICKERS = 80


def _number(value: object) -> float:
    value = safe_float(value)
    return value if math.isfinite(value) else math.nan


def _append_token(value: object, token: str) -> str:
    tokens = [item.strip() for item in str(value or "").split(";") if item.strip()]
    if token and token not in tokens:
        tokens.append(token)
    return ";".join(tokens)


def _direction(row: pd.Series | dict[str, Any]) -> str:
    text = f"{row.get('direction', '')} {row.get('strategy', '')}".lower()
    if "bull" in text:
        return "bullish"
    if "bear" in text:
        return "bearish"
    return "neutral"


def _is_credit(row: pd.Series | dict[str, Any]) -> bool:
    text = f"{row.get('direction', '')} {row.get('strategy', '')}".lower()
    return "credit" in text or "bull put" in text or "bear call" in text


def _history_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    candles = pd.DataFrame(payload.get("candles") or [])
    if candles.empty:
        return {"technical_data_status": "insufficient_history"}
    for col in ["open", "high", "low", "close", "volume", "datetime"]:
        if col not in candles.columns:
            candles[col] = math.nan
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    candles = candles.dropna(subset=["close"]).sort_values("datetime").drop_duplicates("datetime")
    if len(candles) < 35:
        return {"technical_data_status": "insufficient_history"}

    close = candles["close"]
    log_returns = close.map(math.log).diff().dropna()
    hv30 = log_returns.tail(30).std(ddof=1) * math.sqrt(252.0)
    sma20 = close.tail(20).mean()
    sma50 = close.tail(50).mean() if len(close) >= 50 else math.nan
    sma200 = close.tail(200).mean() if len(close) >= 200 else math.nan
    change = close.diff()
    gain = change.clip(lower=0).tail(14).mean()
    loss = (-change.clip(upper=0)).tail(14).mean()
    rsi14 = 100.0 - 100.0 / (1.0 + gain / loss) if loss > 0 else 100.0 if gain > 0 else 50.0
    previous = close.shift(1)
    true_range = pd.concat(
        [candles["high"] - candles["low"], (candles["high"] - previous).abs(), (candles["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.tail(14).mean()
    last = close.iloc[-1]
    anchor = candles.tail(20)
    typical = (anchor["high"] + anchor["low"] + anchor["close"]) / 3.0
    weights = anchor["volume"].fillna(0.0).clip(lower=0.0)
    avwap20 = (typical * weights).sum() / weights.sum() if weights.sum() > 0 else anchor["close"].mean()
    raw_ts = _number(candles.iloc[-1].get("datetime"))
    last_day = dt.datetime.fromtimestamp(raw_ts / 1000.0, tz=dt.timezone.utc).date().isoformat() if math.isfinite(raw_ts) else ""
    return {
        "technical_data_status": "ok",
        "technical_last_day": last_day,
        "technical_close": last,
        "historical_volatility_30d": hv30,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi14,
        "atr14": atr14,
        "atr14_pct": atr14 / last if last > 0 else math.nan,
        "anchored_vwap_20d": avwap20,
        "return_20d": last / close.iloc[-21] - 1.0 if len(close) >= 21 and close.iloc[-21] > 0 else math.nan,
        "return_60d": last / close.iloc[-61] - 1.0 if len(close) >= 61 and close.iloc[-61] > 0 else math.nan,
    }


def _priority_tickers(scored: pd.DataFrame, limit: int) -> list[str]:
    frame = scored.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["_score"] = pd.to_numeric(frame.get("score", 0.0), errors="coerce").fillna(0.0)
    if "live_status" in frame.columns and frame["live_status"].astype(str).str.upper().eq("PASS").any():
        frame = frame[frame["live_status"].astype(str).str.upper().eq("PASS")]
    tickers = (
        frame[frame["ticker"].ne("")].groupby("ticker", as_index=False)["_score"].max()
        .sort_values("_score", ascending=False)["ticker"].head(max(1, limit)).tolist()
    )
    return list(dict.fromkeys(["SPY", *tickers]))


def _load_history(ticker: str, client: Any, cache_dir: Path, asof: dt.date) -> dict[str, Any]:
    path = cache_dir / f"{ticker}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    end = dt.datetime.combine(asof + dt.timedelta(days=1), dt.time.min, tzinfo=dt.timezone.utc)
    response = client.get_price_history_every_day(
        ticker,
        start_datetime=end - dt.timedelta(days=TECHNICAL_HISTORY_DAYS * 2),
        end_datetime=end,
        need_extended_hours_data=False,
        need_previous_close=True,
    )
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def apply_schwab_price_context(
    scored: pd.DataFrame,
    *,
    out_dir: Path,
    asof: dt.date,
    offline: bool = False,
    max_tickers: int = TECHNICAL_MAX_TICKERS,
) -> pd.DataFrame:
    """Use Schwab daily candles for HV and technicals; never alias UW's generic volatility field."""
    if scored.empty:
        return scored.copy()
    out = scored.copy()
    if "raw_uw_volatility" not in out.columns:
        out["raw_uw_volatility"] = out.get("volatility", math.nan)
    out["realized_volatility_source"] = "missing_schwab_daily_history"
    out["iv_hv_valid"] = False
    if offline:
        out["technical_data_status"] = "offline"
        return _score_integrity(out)

    cache_dir = Path(out_dir) / "schwab_price_history"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    try:
        service = SchwabLiveDataService(SchwabAuthConfig.from_env(load_dotenv_file=True), interactive_login=False)
        client = service.connect()
        for ticker in _priority_tickers(out, max_tickers):
            try:
                metrics[ticker] = _history_metrics(_load_history(ticker, client, cache_dir, asof))
            except Exception as exc:
                errors[ticker] = str(exc)[:500]
    except Exception as exc:
        errors["__connection__"] = str(exc)[:500]
    (cache_dir / "errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8")

    metric_columns = [
        "technical_data_status", "technical_last_day", "technical_close", "historical_volatility_30d",
        "sma20", "sma50", "sma200", "rsi14", "atr14", "atr14_pct", "anchored_vwap_20d",
        "return_20d", "return_60d",
    ]
    tickers = out["ticker"].astype(str).str.upper()
    for col in metric_columns:
        out[col] = tickers.map(lambda ticker: (metrics.get(ticker) or {}).get(col, math.nan))
    spy_return = _number((metrics.get("SPY") or {}).get("return_20d"))
    out["relative_strength_20d_vs_spy"] = pd.to_numeric(out["return_20d"], errors="coerce") - spy_return

    # Schwab daily log returns are the preferred realised-vol source, but Schwab
    # history is missing for a large minority of the universe. Rather than
    # overwrite those rows with NaN -- which silently disables the IV/HV richness
    # gate for them -- fall back to the value computed from the dated-folder
    # close history by codexuw.realized_vol.
    prior_hv = pd.to_numeric(out.get("realized_volatility_30d"), errors="coerce")
    schwab_hv = pd.to_numeric(out["historical_volatility_30d"], errors="coerce")
    combined_hv = schwab_hv.where(schwab_hv.gt(0), prior_hv)
    out["realized_volatility_30d"] = combined_hv

    iv = pd.to_numeric(out.get("iv30d"), errors="coerce")
    hv = pd.to_numeric(out["realized_volatility_30d"], errors="coerce")
    out["iv_hv_ratio"] = iv / hv.where(hv.gt(0))
    out["iv_hv_spread"] = iv - hv
    valid = out["iv_hv_ratio"].replace([math.inf, -math.inf], math.nan).notna()
    out.loc[valid, "iv_hv_valid"] = True
    used_schwab = valid & schwab_hv.gt(0)
    used_local = valid & ~schwab_hv.gt(0)
    out.loc[used_schwab, "realized_volatility_source"] = "schwab_daily_log_returns_30d_annualized"
    out.loc[used_local, "realized_volatility_source"] = "screener_close_history_21d_annualized"
    return _score_integrity(out)


def _score_integrity(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    rows: list[tuple[int, int, int, int, str]] = []
    for _, row in out.iterrows():
        direction = _direction(row)
        sign = 1.0 if direction == "bullish" else -1.0 if direction == "bearish" else 0.0
        bias = _number(row.get("combined_flow_bias"))
        if not math.isfinite(bias):
            bias = _number(row.get("flow_bias"))
        alignment = bias * sign if math.isfinite(bias) else math.nan
        directional = str(row.get("flow_quality") or "").lower() == "directional"
        flow = 3 if directional and alignment >= 0.20 else 2 if directional or alignment >= 0.20 else 1 if alignment > 0 else 0

        close, sma20, sma50 = (_number(row.get(key)) for key in ["technical_close", "sma20", "sma50"])
        avwap, rsi, rs = (_number(row.get(key)) for key in ["anchored_vwap_20d", "rsi14", "relative_strength_20d_vs_spy"])
        finite = [math.isfinite(close) and math.isfinite(sma20), math.isfinite(close) and math.isfinite(sma50),
                  math.isfinite(close) and math.isfinite(avwap), math.isfinite(rsi), math.isfinite(rs)]
        if direction == "bullish":
            checks = [close > sma20, close > sma50, close > avwap, 50 <= rsi <= 75, rs > 0]
        else:
            checks = [close < sma20, close < sma50, close < avwap, 25 <= rsi <= 50, rs < 0]
        passed = sum(bool(check) for check, available in zip(checks, finite) if available)
        available = sum(finite)
        technical = 3 if passed >= 4 else 2 if passed >= 3 else 1 if passed >= 2 else 0
        confirmation = "confirmed" if technical >= 2 else "contra" if available >= 3 and passed <= 1 else "missing"

        iv_hv = _number(row.get("iv_hv_ratio"))
        if _is_credit(row):
            volatility = 2 if iv_hv >= 1.10 else 1 if iv_hv >= 1.00 else 0
        else:
            volatility = 2 if 0 < iv_hv <= 1.00 else 1 if 0 < iv_hv <= 1.25 else 0
        ratio = _number(row.get("expected_move_ratio"))
        if not math.isfinite(ratio):
            ratio = _number(row.get("breakeven_expected_move_ratio"))
        distance = 2 if ratio >= 1.25 else 1 if ratio >= 0.75 else 0
        rows.append((flow, technical, volatility, distance, confirmation))
    out[["flow_score_0_3", "technical_score_0_3", "volatility_score_0_2", "distance_score_0_2", "technical_confirmation"]] = pd.DataFrame(rows, index=out.index)
    out["confidence_model_score"] = out["flow_score_0_3"] + out["technical_score_0_3"] + out["volatility_score_0_2"] + out["distance_score_0_2"]
    out["confidence_model"] = out["confidence_model_score"].map(lambda value: "High" if value >= 7 else "Medium" if value >= 5 else "Reject")
    return out


def _cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _model_pop(row: pd.Series | dict[str, Any]) -> float:
    existing = _number(row.get("pop_delta_proxy"))
    if 0 <= existing <= 1:
        return existing
    spot = _number(row.get("stock_price_live"))
    breakeven = _number(row.get("breakeven"))
    iv = _number(row.get("spread_iv"))
    if not math.isfinite(iv):
        iv = _number(row.get("iv30d"))
    if iv > 3:
        iv /= 100.0
    expiry = pd.to_datetime(row.get("expiry"), errors="coerce")
    dte_value = _number(row.get("dte"))
    dte = int(dte_value) if math.isfinite(dte_value) and dte_value > 0 else 30
    if not pd.isna(expiry):
        dte = max((expiry.date() - dt.date.today()).days, 1)
    if not all(math.isfinite(value) and value > 0 for value in [spot, breakeven, iv]):
        return math.nan
    time = dte / 365.0
    d2 = (math.log(spot / breakeven) + (0.04 - 0.5 * iv * iv) * time) / (iv * math.sqrt(time))
    return _cdf(d2) if _direction(row) == "bullish" else _cdf(-d2)


def _chain_gex(path: Path) -> dict[str, Any]:
    chain = json.loads(path.read_text(encoding="utf-8"))
    spot = chain_spot(chain)
    contracts = chain_to_contracts(chain)
    if contracts.empty or not math.isfinite(spot) or spot <= 0:
        return {"gex_status": "unavailable"}
    contracts = contracts.copy()
    contracts["gamma"] = pd.to_numeric(contracts["gamma"], errors="coerce").fillna(0.0).abs()
    contracts["open_interest"] = pd.to_numeric(contracts["open_interest"], errors="coerce").fillna(0.0).clip(lower=0.0)
    contracts["unsigned_gex"] = contracts["gamma"] * contracts["open_interest"] * 100.0 * spot * spot * 0.01
    contracts["signed_gex"] = contracts["unsigned_gex"] * contracts["right"].map({"C": 1.0, "P": -1.0}).fillna(0.0)
    calls = contracts[contracts["right"].eq("C")].groupby("strike", as_index=False)["unsigned_gex"].sum()
    puts = contracts[contracts["right"].eq("P")].groupby("strike", as_index=False)["unsigned_gex"].sum()
    if calls.empty or puts.empty:
        return {"gex_status": "partial"}
    call_wall = calls.sort_values("unsigned_gex", ascending=False).iloc[0]
    put_wall = puts.sort_values("unsigned_gex", ascending=False).iloc[0]
    gross = contracts["unsigned_gex"].sum()
    net = contracts["signed_gex"].sum()
    concentration = (call_wall["unsigned_gex"] + put_wall["unsigned_gex"]) / gross if gross > 0 else math.nan
    return {
        "gex_status": "proxy_available",
        "gex_method": "schwab_gamma_x_oi_dealer_sign_proxy_not_observed_positioning",
        "gex_spot": spot,
        "gex_net_per_1pct": net,
        "gex_gross_per_1pct": gross,
        "gex_net_to_gross": net / gross if gross > 0 else math.nan,
        "gex_call_wall": _number(call_wall["strike"]),
        "gex_put_wall": _number(put_wall["strike"]),
        "gex_wall_concentration": concentration,
        "gex_materiality": "strong" if concentration >= 0.30 else "moderate" if concentration >= 0.15 else "weak",
    }


def apply_candidate_integrity_context(scored: pd.DataFrame, *, out_dir: Path) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    out = scored.copy()
    chain_dir = Path(out_dir) / "schwab_chains"
    gex: dict[str, dict[str, Any]] = {}
    for ticker in out["ticker"].astype(str).str.upper().drop_duplicates():
        path = chain_dir / f"{ticker}.json"
        if path.exists():
            try:
                gex[ticker] = _chain_gex(path)
            except Exception:
                gex[ticker] = {"gex_status": "error"}
    gex_cols = ["gex_status", "gex_method", "gex_spot", "gex_net_per_1pct", "gex_gross_per_1pct", "gex_net_to_gross", "gex_call_wall", "gex_put_wall", "gex_wall_concentration", "gex_materiality"]
    tickers = out["ticker"].astype(str).str.upper()
    for col in gex_cols:
        out[col] = tickers.map(lambda ticker: (gex.get(ticker) or {}).get(col, math.nan))

    quote_status, quote_reason, relations, pops = [], [], [], []
    for idx, row in out.iterrows():
        credit = _is_credit(row)
        mid = _number(row.get("mid_credit" if credit else "mid_debit"))
        natural = _number(row.get("natural_credit" if credit else "natural_debit"))
        width_pct = _number(row.get("quote_width_pct"))
        ratio = natural / mid if math.isfinite(mid) and mid > 0 and math.isfinite(natural) else math.nan
        reasons: list[str] = []
        severe = not math.isfinite(natural) or natural <= 0
        if severe:
            reasons.append("nonpositive executable natural")
        if width_pct > 0.50:
            severe = True
            reasons.append("quote width above 50%")
        if credit and math.isfinite(ratio) and ratio < 0.35:
            severe = True
            reasons.append("natural credit below 35% of midpoint")
        if not credit and math.isfinite(ratio) and ratio > 2.0:
            severe = True
            reasons.append("natural debit above 200% of midpoint")
        warning = not severe and ((credit and math.isfinite(ratio) and ratio < 0.60) or width_pct > 0.30)
        quote_status.append("reject" if severe else "work_limit" if warning else "acceptable")
        quote_reason.append("; ".join(reasons) if reasons else "wide quote requires patient limit" if warning else "executable quote geometry acceptable")
        if severe:
            out.at[idx, "live_status"] = "FAIL"
            out.at[idx, "live_blocker"] = _append_token(row.get("live_blocker"), "integrity_quote_geometry_reject")
            out.at[idx, "primary_blocker"] = _append_token(row.get("primary_blocker"), "integrity_quote_geometry_reject")

        spot, call_wall, put_wall = (_number(row.get(key)) for key in ["gex_spot", "gex_call_wall", "gex_put_wall"])
        breakeven = _number(row.get("breakeven"))
        if not all(math.isfinite(value) for value in [spot, call_wall, put_wall]):
            relation = "missing candidate-level GEX"
        elif _direction(row) == "bullish":
            relation = "breakeven above call wall; resistance risk" if math.isfinite(breakeven) and breakeven > call_wall else "breakeven below call wall"
        elif _direction(row) == "bearish":
            relation = "breakeven below put wall; support risk" if math.isfinite(breakeven) and breakeven < put_wall else "breakeven above put wall"
        else:
            relation = "spot between walls" if put_wall <= spot <= call_wall else "spot outside primary walls"
        relations.append(relation)
        pops.append(_model_pop(row))
    out["integrity_quote_status"] = quote_status
    out["integrity_quote_reason"] = quote_reason
    out["gex_candidate_relation"] = relations
    out["model_pop"] = pops
    out["oi_evidence_scope"] = "prior_session_cleared_open_interest_confirmation"
    out["oi_is_live_directional_signal"] = False
    oi = out["oi_carryover_status"] if "oi_carryover_status" in out.columns else pd.Series("", index=out.index)
    out["oi_approval_role"] = oi.map(lambda value: "confirming" if str(value) in {"supportive", "matched_unconfirmed"} else "warning_not_standalone_veto")
    return _score_integrity(out)


def _legacy_disposition(row: pd.Series) -> str:
    status = str(row.get("trade_status") or "")
    reason = str(row.get("v4_direct_disposition_reason") or row.get("trade_status_reason") or "").lower()
    if status == "Execute" or "v4 execute" in reason:
        return "ENTER"
    if "work limit" in reason:
        return "WORK LIMIT"
    if "scout" in reason:
        return "SCOUT"
    if status in {"Watch", "Research"}:
        return "REVIEW"
    return "REJECT"


def _edge_supported(row: pd.Series) -> bool:
    verdict = str(row.get("replay_ev_verdict") or row.get("edge_verdict") or "").lower()
    return verdict in {"positive", "acceptable", "pass"} or (_number(row.get("edge_profit_factor")) >= 1.15 and _number(row.get("edge_avg_pnl")) > 0)


def _final_decision(row: pd.Series) -> str:
    legacy = _legacy_disposition(row)
    severe = str(row.get("integrity_quote_status")) == "reject"
    confidence = str(row.get("confidence_model"))
    pop = _number(row.get("model_pop"))
    blocker = str(row.get("primary_blocker") or "").lower()
    earnings = "earnings" in blocker or str(row.get("catalyst_status") or "").lower() == "caution"
    explicit_edge = any(_number(row.get(key)) > 0 for key in ["flow_score_0_3", "technical_score_0_3", "volatility_score_0_2"])
    cap_only = "credit_sleeve_cap" in blocker or "credit sleeve cap" in str(row.get("v4_direct_disposition_reason") or "").lower()
    if severe:
        return "REJECT"
    if legacy == "REJECT":
        oi_only = blocker.strip() in {"oi_carryover_contrary", "oi_carryover_mixed"}
        return "QUALIFIED REVIEW" if oi_only else "REJECT"
    strict = confidence == "High" and pop >= 0.60 and _edge_supported(row) and not earnings and explicit_edge
    if legacy == "ENTER" and strict:
        return "ENTER NOW"
    if cap_only and strict:
        return "ENTER NOW"
    if legacy in {"ENTER", "WORK LIMIT"} and confidence in {"High", "Medium"}:
        return "WORK LIMIT"
    return "QUALIFIED REVIEW"


def _trade_text(row: pd.Series) -> str:
    strategy = str(row.get("strategy") or row.get("direction") or "")
    short, long = _number(row.get("short_strike")), _number(row.get("long_strike"))
    if not math.isfinite(short) or not math.isfinite(long):
        return strategy
    if _is_credit(row):
        return f"{strategy}: sell {short:g} / buy {long:g}"
    return f"{strategy}: buy {long:g} / sell {short:g}"


def _why(row: pd.Series) -> str:
    parts = []
    if str(row.get("integrity_quote_status")) != "acceptable":
        parts.append(str(row.get("integrity_quote_reason") or "quote review"))
    if str(row.get("primary_blocker") or ""):
        parts.append(str(row.get("primary_blocker")).replace("_", " "))
    if str(row.get("technical_confirmation")) in {"contra", "missing"}:
        parts.append(f"technical {row.get('technical_confirmation')}")
    if str(row.get("gex_candidate_relation") or "").startswith("breakeven"):
        parts.append(str(row.get("gex_candidate_relation")))
    if not bool(row.get("iv_hv_valid")):
        parts.append("validated IV/HV unavailable")
    return "; ".join(dict.fromkeys(parts)) or "all recorded integrity checks passed"


def _decision_rows(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["Decision"] = out.apply(_final_decision, axis=1)
    out["Ticker"] = out["ticker"].astype(str)
    out["Trade"] = out.apply(_trade_text, axis=1)
    out["Expiry"] = out["expiry"].astype(str).str[:10]
    out["Confidence"] = out["confidence_model"]
    out["Confidence score"] = pd.to_numeric(out["confidence_model_score"], errors="coerce")
    out["POP"] = pd.to_numeric(out["model_pop"], errors="coerce")
    out["Historical n"] = pd.to_numeric(out.get("edge_sample_size"), errors="coerce")
    out["Historical win"] = pd.to_numeric(out.get("edge_win_rate"), errors="coerce")
    out["Historical PF"] = pd.to_numeric(out.get("edge_profit_factor"), errors="coerce")
    out["Current mid"] = out.apply(lambda row: _number(row.get("mid_credit")) if _is_credit(row) else _number(row.get("mid_debit")), axis=1)
    out["Executable natural"] = out.apply(lambda row: _number(row.get("natural_credit")) if _is_credit(row) else _number(row.get("natural_debit")), axis=1)
    out["Entry limit"] = pd.to_numeric(out.get("target_entry"), errors="coerce")
    out["Technical"] = out["technical_confirmation"]
    out["IV/HV"] = pd.to_numeric(out["iv_hv_ratio"], errors="coerce")
    out["GEX"] = out["gex_candidate_relation"]
    out["OI scope"] = out["oi_evidence_scope"]
    out["Why"] = out.apply(_why, axis=1)
    order = {"ENTER NOW": 0, "WORK LIMIT": 1, "QUALIFIED REVIEW": 2, "REJECT": 3}
    out["_order"] = out["Decision"].map(order).fillna(9)
    return out.sort_values(["_order", "Confidence score", "POP", "Historical PF"], ascending=[True, False, False, False])


def build_condor_shadow_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    credits = scored[scored["strategy"].astype(str).str.contains("Credit Spread", na=False)]
    for (ticker, expiry), part in credits.groupby(["ticker", "expiry"], dropna=False):
        puts = part[part["strategy"].astype(str).str.contains("Bull Put", na=False)].sort_values("score", ascending=False)
        calls = part[part["strategy"].astype(str).str.contains("Bear Call", na=False)].sort_values("score", ascending=False)
        if puts.empty or calls.empty:
            continue
        put, call = puts.iloc[0], calls.iloc[0]
        spot, put_short, call_short = _number(put.get("stock_price_live")), _number(put.get("short_strike")), _number(call.get("short_strike"))
        if not all(math.isfinite(value) for value in [spot, put_short, call_short]) or not put_short < spot < call_short:
            continue
        rows.append({
            "Ticker": ticker,
            "Expiry": str(expiry)[:10],
            "Trade": f"Iron Condor: buy {_number(put.get('long_strike')):g}P / sell {put_short:g}P + sell {call_short:g}C / buy {_number(call.get('long_strike')):g}C",
            "Executable natural": _number(put.get("natural_credit")) + _number(call.get("natural_credit")),
            "POP": max(0.0, min(1.0, _number(put.get("model_pop")) + _number(call.get("model_pop")) - 1.0)),
            "Decision": "QUALIFIED REVIEW",
            "Why": "new strategy family; dedicated condor replay required before promotion",
        })
    return pd.DataFrame(rows).sort_values(["POP", "Executable natural"], ascending=False).head(20) if rows else pd.DataFrame(rows)


def _find_scored(out_dir: Path, asof: dt.date) -> Path:
    preferred = out_dir / f"codexdaily_v4_scored_reference_{asof}.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(out_dir.glob("*scored*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No scored V4 artifact found in {out_dir}")
    return candidates[0]


def write_integrity_decision_book(
    *, out_dir: Path, asof: dt.date, offline: bool = False, source_dates: dict[str, str] | None = None
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    scored_path = _find_scored(out_dir, asof)
    scored = pd.read_csv(scored_path, low_memory=False)
    scored = apply_schwab_price_context(scored, out_dir=out_dir, asof=asof, offline=offline)
    scored = apply_candidate_integrity_context(scored, out_dir=out_dir)
    decisions = _decision_rows(scored)
    condors = build_condor_shadow_candidates(scored)

    enriched_path = out_dir / f"codexdaily_v42_scored_integrity_{asof}.csv"
    decision_path = out_dir / f"codexdaily_v42_decision_book_{asof}.csv"
    condor_path = out_dir / f"codexdaily_v42_condor_shadow_{asof}.csv"
    report_path = out_dir / f"codexdaily_v42_trade_table_{asof}.md"
    scored.to_csv(enriched_path, index=False)
    decisions.drop(columns=["_order"], errors="ignore").to_csv(decision_path, index=False)
    condors.to_csv(condor_path, index=False)

    columns = ["Decision", "Ticker", "Trade", "Expiry", "Current mid", "Executable natural", "Entry limit", "Confidence", "Confidence score", "POP", "Historical n", "Historical win", "Historical PF", "Technical", "IV/HV", "GEX", "Why"]
    active = decisions[decisions["Decision"].ne("REJECT")].head(40)
    counts = decisions["Decision"].value_counts().to_dict()
    source_dates = source_dates or {}
    lines = [
        f"# Codex Daily V4.2 Integrity Trade Table - {asof}", "", f"Release: `{PIPELINE_VERSION_V42}`", "",
        "## Actionable And Qualified Setups", "",
        active[columns].to_markdown(index=False, floatfmt=".2f") if not active.empty else "_No actionable or qualified rows._",
        "", "## Decision Funnel", "", "| Decision | Rows |", "|:--|--:|",
        *[f"| {key} | {value} |" for key, value in sorted(counts.items())],
        "", "## Strategy Expansion Shadow Book", "",
        condors.to_markdown(index=False, floatfmt=".2f") if not condors.empty else "_No structurally valid condor pair was found._",
        "", "## Evidence Contract", "",
        "- OI is prior-session cleared confirmation, not a live directional signal.",
        "- Candidate GEX is a Schwab gamma-times-OI proxy and not observed dealer positioning.",
        "- Historical volatility is calculated from Schwab daily log returns and annualized.",
        "- ENTER NOW requires the underlying V4 approval plus the V4.2 confidence, POP, edge, event, and quote checks.",
        "- Missing technical or GEX data remains visible and cannot silently create confidence.",
        "", "## Source Dates", "", "| Source | Date |", "|:--|:--|",
        *[f"| {key} | {value} |" for key, value in sorted(source_dates.items())], "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    ledger_path = out_dir.parent / "codexdaily_v42_recommendation_ledger.csv"
    ledger = decisions[decisions["Decision"].ne("REJECT")].copy()
    ledger.insert(0, "asof", str(asof))
    ledger.insert(1, "pipeline_version", PIPELINE_VERSION_V42)
    if ledger_path.exists():
        ledger = pd.concat([pd.read_csv(ledger_path, low_memory=False), ledger], ignore_index=True, sort=False)
        ledger = ledger.drop_duplicates(["asof", "Ticker", "Trade", "Expiry"], keep="last")
    ledger.to_csv(ledger_path, index=False)
    outcomes_path = out_dir.parent / "codexdaily_v42_live_outcomes.csv"
    if not outcomes_path.exists():
        pd.DataFrame(columns=["asof", "pipeline_version", "ticker", "trade", "expiry", "decision", "entry_time", "entry_price", "exit_time", "exit_price", "realized_pnl", "outcome_source", "matched_fill_id"]).to_csv(outcomes_path, index=False)

    manifest = {
        "pipeline_version": PIPELINE_VERSION_V42, "asof": str(asof), "source_dates": source_dates,
        "scored_input": str(scored_path), "enriched_scored": str(enriched_path), "decision_book": str(decision_path),
        "condor_shadow": str(condor_path), "report": str(report_path), "recommendation_ledger": str(ledger_path),
        "live_outcome_ledger": str(outcomes_path), "decision_counts": counts,
    }
    manifest_path = out_dir / f"codexdaily_v42_manifest_{asof}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_full_overlay_workspace(
    *, root: Path, base_date: dt.date, overlay_file: Path, overlay_date: dt.date, out_dir: Path
) -> tuple[Path, dict[str, str]]:
    source = Path(root) / str(base_date)
    if not source.is_dir():
        raise FileNotFoundError(f"Missing base EOD folder: {source}")
    stage = Path(out_dir) / "_full_overlay_source" / str(overlay_date)
    stage.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name.startswith("chain-oi-changes-"):
            continue
        target = stage / child.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(child, target_is_directory=child.is_dir())
    target = stage / Path(overlay_file).name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(Path(overlay_file).resolve())
    return stage, {"eod_discovery": str(base_date), "oi_confirmation": str(overlay_date), "live_quotes": str(overlay_date)}
