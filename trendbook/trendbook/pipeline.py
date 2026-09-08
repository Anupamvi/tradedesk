"""Weekly trend book: Schwab tape first, ORATS after Stage 2, stock first."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from trendbook.bars import daily_snapshot, to_weekly
from trendbook.config import (
    INDEX_TICKERS,
    MACRO_TICKERS,
    MKT_CAP_MIN,
    ORATS_HTTP_DEFAULT,
    OUT_DIR,
    SECTOR_ETFS,
    ticker_etf,
)
from trendbook.dates import today_et
from trendbook.entry import classify_entry
from trendbook.envload import load_orats_token
from trendbook.num import to_float
from trendbook.options import expression_for, orats_call_debit, schwab_call_debit
from trendbook.orats import cheap_iv, fetch_cores, fetch_strikes, parse_core, process_http, reset_process_http
from trendbook.persist import update as persist_update
from trendbook.rs import percentile_rank, rs_bundle, rs_ok
from trendbook.schwab import http_count, price_history_bars, quotes_many, reset_http, use_live_schwab
from trendbook.stage import classify_stage
from trendbook.template import eight_point
from trendbook.discover import dynamic_universe, infra_tickers, load_memory, save_memory
from trendbook.trend import (
    EMPTY_RUN,
    classify_regime,
    current_run_from_flags,
    decide,
    earnings_near,
    first_add_dates,
    grade,
    invalidation_levels,
    replay_row,
    week_flags,
)


def _board_skip() -> set:
    return set(INDEX_TICKERS) | set(SECTOR_ETFS) | set(MACRO_TICKERS)


def _load_bars(ticker: str, asof: str, live: bool, injected: Optional[Dict[str, list]]) -> List[dict]:
    if injected is not None and ticker in injected:
        return [b for b in injected[ticker] if str(b.get("date") or "")[:10] <= asof[:10]]
    return price_history_bars(ticker, asof, live=live, use_cache=True)


def classify_row(
    ticker: str,
    daily: dict,
    weekly_stage: dict,
    bundle: dict,
    rs_pctile: Optional[float],
    universe_n: int,
    core: Optional[dict],
    run: Optional[dict] = None,
    already_bought: bool = False,
    regime_risk: str = "on",
    asof: Optional[str] = None,
) -> dict:
    run = run or dict(EMPTY_RUN)
    rs_pass = rs_ok(bundle)
    on_board = bool(run.get("on_board") and daily.get("ok"))
    if core:
        cap = to_float(core.get("mkt_cap"))
        if cap is not None and cap < MKT_CAP_MIN and ticker not in INDEX_TICKERS:
            on_board = False
        tk = core.get("tk_over")
        if tk in (True, "1", "Y", "y") or tk == 1:
            on_board = False
    entry = classify_entry(daily, weekly_stage) if on_board else {"state": "off", "reason": "not_on_board"}
    weeks = int(run.get("weeks_in_trend") or 0)
    info = decide(
        on_board,
        bool(run.get("late")),
        entry.get("state"),
        weeks=weeks,
        entry_reason=entry.get("reason"),
        off_high=run.get("off_high"),
        rs_126=bundle.get("rs_126"),
        rs_63=bundle.get("rs_63"),
        mansfield_spy=bundle.get("mansfield_spy"),
        mansfield_rising=bundle.get("mansfield_spy_rising"),
        hh_hl=weekly_stage.get("hh_hl"),
        residual_63=bundle.get("residual_63"),
        vol_expand=run.get("vol_expand"),
        break_vol_expand=run.get("break_vol_expand"),
        already_bought=already_bought,
        regime_risk=regime_risk,
        rs_sector_126=bundle.get("rs_sector_126"),
        earnings_near_flag=earnings_near(core),
    )
    action = info["action"]
    setup = info.get("setup")
    why = setup if setup else entry.get("reason")
    conf = grade(
        action,
        weeks,
        run.get("ret_13w"),
        run.get("off_high"),
        why,
        rs_pass,
        hh_hl=weekly_stage.get("hh_hl"),
    )
    tmpl = eight_point(daily, rs_pctile)
    return {
        "ticker": ticker,
        "action": action,
        "status": action,
        "grade": conf,
        "on_board": on_board,
        "stage": weekly_stage.get("stage"),
        "stage_reason": weekly_stage.get("reason"),
        "hh_hl": weekly_stage.get("hh_hl"),
        "entry": entry.get("state"),
        "entry_reason": why,
        "setup": setup,
        "rs_pctile": rs_pctile,
        "rs_pass": rs_pass,
        "mansfield_spy": bundle.get("mansfield_spy"),
        "mansfield_spy_rising": bundle.get("mansfield_spy_rising"),
        "rs_126": bundle.get("rs_126"),
        "rs_63": bundle.get("rs_63"),
        "rs_sector_126": bundle.get("rs_sector_126"),
        "beta_126": bundle.get("beta_126"),
        "residual_63": bundle.get("residual_63"),
        "residual_126": bundle.get("residual_126"),
        "vol_expand": run.get("vol_expand"),
        "break_vol_expand": run.get("break_vol_expand"),
        "size": info.get("size"),
        "invalidation": invalidation_levels(weekly_stage) if on_board else None,
        "regime_risk": regime_risk,
        "ibd": bundle.get("ibd"),
        "close": daily.get("close"),
        "ema20": daily.get("ema20"),
        "extension_atr": daily.get("extension_atr"),
        "template_pass": tmpl.get("pass"),
        "template_n": tmpl.get("n_pass"),
        "template": tmpl,
        "core": core,
        "iv_pctile_1y": (core or {}).get("iv_pctile_1y") if core else None,
        "cheap_iv": cheap_iv(core) if core else False,
        "group": None,
        "etf": None,
        "trend_start": run.get("trend_start"),
        "weeks_in_trend": run.get("weeks_in_trend") or 0,
        "px_at_start": run.get("px_at_start"),
        "pct_from_start": run.get("pct_from_start"),
        "ret_13w": run.get("ret_13w"),
        "late": bool(run.get("late")),
        "had_trend": bool(run.get("had_trend")),
        "last_trend_start": run.get("last_trend_start"),
        "last_trend_end": run.get("last_trend_end"),
        "campaign_high": run.get("campaign_high"),
        "off_high": run.get("off_high"),
        "discovered": False,
    }


def _wanted_names(
    tickers: Optional[Sequence[str]],
    asof: Optional[str] = None,
    live: bool = False,
    spy: Optional[Sequence[dict]] = None,
) -> List[str]:
    if tickers:
        names = list(tickers)
    else:
        names = dynamic_universe(asof or today_et(), live=live, spy=spy)
    if "SPY" not in names:
        names = ["SPY"] + names
    wanted = []
    seen = set()
    for name in names:
        key = str(name).upper()
        if key in seen:
            continue
        seen.add(key)
        wanted.append(key)
    return wanted


def _load_universe_bars(
    wanted: Sequence[str],
    asof: str,
    live: bool,
    no_schwab: bool,
    bars_map: Optional[Dict[str, list]],
) -> tuple:
    bars: Dict[str, list] = {}
    missing_bars = []
    for name in wanted:
        series = _load_bars(name, asof, live=live, injected=bars_map)
        if not series:
            missing_bars.append(name)
            continue
        bars[name] = series
    quotes = {}
    if live and not no_schwab:
        quotes = quotes_many(list(bars.keys()), asof)
        for name, q in quotes.items():
            last = q.get("last") or q.get("close")
            if last and bars.get(name):
                last_bar = bars[name][-1]
                last_d = str(last_bar.get("date") or "")[:10]
                weekend = False
                try:
                    weekend = datetime.strptime(asof[:10], "%Y-%m-%d").weekday() >= 5
                except ValueError:
                    weekend = False
                if last_d == asof[:10] or weekend:
                    last_bar["close"] = last
                elif last_d < asof[:10]:
                    bars[name] = bars[name] + [
                        {
                            "date": asof,
                            "open": last,
                            "high": last,
                            "low": last,
                            "close": last,
                            "volume": q.get("volume"),
                        }
                    ]
    return bars, missing_bars


def _features(bars: Dict[str, list], asof: str) -> dict:
    from trendbook.config import ticker_group

    spy_bars = bars.get("SPY") or []
    features = {}
    ibd_map = {}
    skip = _board_skip()
    for name, series in bars.items():
        daily = daily_snapshot(series, asof)
        weekly = to_weekly(series, asof)
        st = classify_stage(weekly)
        etf = ticker_etf(name)
        sector_bars = bars.get(etf) if etf != name else None
        bundle = rs_bundle(name, series, spy_bars, sector_bars, asof)
        if name == "SPY" or name in skip:
            flags = []
            run = dict(EMPTY_RUN)
            adds = {}
        else:
            flags = week_flags(series, spy_bars, asof)
            run = current_run_from_flags(flags)
            adds = first_add_dates(flags, series, spy_bars)
        features[name] = {
            "daily": daily,
            "weekly": weekly,
            "stage": st,
            "bundle": bundle,
            "etf": etf,
            "group": ticker_group(name),
            "flags": flags,
            "run": run,
            "first_adds": adds,
        }
        ibd_map[name] = bundle.get("ibd")
    return features, ibd_map


def build_full(
    asof: str,
    out_dir: Optional[Path] = None,
    live_schwab: bool = False,
    no_schwab: bool = False,
    no_orats: bool = False,
    max_orats_http: int = ORATS_HTTP_DEFAULT,
    orats_token_file: Optional[str] = None,
    tickers: Optional[Sequence[str]] = None,
    bars_map: Optional[Dict[str, list]] = None,
    persist: bool = True,
) -> dict:
    from trendbook import outcomes, report

    today = today_et()
    live = use_live_schwab(asof, live_flag=live_schwab, no_schwab=no_schwab, today=today)
    reset_http()
    reset_process_http()
    spy_seed = _load_bars("SPY", asof, live=live, injected=bars_map)
    wanted = _wanted_names(tickers, asof=asof, live=live and tickers is None, spy=spy_seed)
    discovered: set = set()
    if tickers is None:
        known = set(infra_tickers()) | set(load_memory())
        discovered = {n for n in wanted if n not in known}
    bars, missing_bars = _load_universe_bars(wanted, asof, live, no_schwab, bars_map)
    features, ibd_map = _features(bars, asof)
    pctiles = percentile_rank(ibd_map)
    ranked_n = sum(1 for v in ibd_map.values() if v is not None)
    skip = _board_skip()
    regime = classify_regime(
        ((features.get("SPY") or {}).get("stage") or {}).get("stage"),
        ((features.get("TLT") or {}).get("stage") or {}).get("stage"),
        ((features.get("UUP") or {}).get("stage") or {}).get("stage"),
    )

    def _already(feat: dict) -> bool:
        start = str((feat.get("run") or {}).get("trend_start") or "")
        first = (feat.get("first_adds") or {}).get(start)
        return bool(first and first < asof)

    prelim = []
    for name, feat in features.items():
        if name in skip:
            continue
        row = classify_row(
            name,
            feat["daily"],
            feat["stage"],
            feat["bundle"],
            pctiles.get(name),
            ranked_n,
            None,
            feat["run"],
            already_bought=_already(feat),
            regime_risk=regime["risk"],
            asof=asof,
        )
        row["group"] = feat["group"]
        row["etf"] = feat["etf"]
        prelim.append(row)

    board_names = [r["ticker"] for r in prelim if r["on_board"] or r["stage"] == 2]
    token = None if no_orats else load_orats_token(token_file=orats_token_file)
    cores = {}
    if token and board_names:
        raw = fetch_cores(board_names, token, max_requests=max_orats_http, asof=asof, today=today)
        cores = {k: parse_core(v) for k, v in raw.items()}

    rows = []
    for row in prelim:
        name = row["ticker"]
        feat = features[name]
        core = cores.get(name)
        classified = classify_row(
            name,
            feat["daily"],
            feat["stage"],
            feat["bundle"],
            pctiles.get(name),
            ranked_n,
            core,
            feat["run"],
            already_bought=_already(feat),
            regime_risk=regime["risk"],
            asof=asof,
        )
        classified["group"] = feat["group"]
        classified["etf"] = feat["etf"]
        classified["discovered"] = name in discovered
        if persist:
            book = persist_update(name, asof, classified["on_board"], classified["stage"], feat["run"])
            classified["weeks_in_stage"] = book.get("weeks_in_stage")
            classified["first_on_board"] = book.get("first_on_board")
        else:
            classified["weeks_in_stage"] = classified.get("weeks_in_trend")
            classified["first_on_board"] = classified.get("trend_start")
        rows.append(classified)

    add_names = [r["ticker"] for r in rows if r["action"] == "ADD"]
    strike_rows = {}
    if token and add_names:
        strike_rows = fetch_strikes(add_names, token, max_requests=max_orats_http, asof=asof, today=today)

    for row in rows:
        if row["action"] != "ADD":
            row["expression"] = {"kind": "none", "reason": "not_add"}
            continue
        core = row.get("core")
        schwab_ticket = None
        if live and not no_schwab:
            schwab_ticket = schwab_call_debit(row["ticker"], asof, core)
        orats_ticket = None
        if not schwab_ticket:
            orats_ticket = orats_call_debit(strike_rows.get(row["ticker"]) or [], row.get("close"), core)
        row["expression"] = expression_for(True, core, schwab_ticket, orats_ticket)

    rows = report.sort_rows(rows)
    if persist and tickers is None:
        save_memory(
            [r["ticker"] for r in rows if r.get("on_board") or r.get("stage") == 2 or r.get("had_trend")],
            asof,
        )
    spy_feat = features.get("SPY") or {}
    spy_stage = (spy_feat.get("stage") or {}).get("stage")
    n_add = sum(1 for r in rows if r["action"] == "ADD")
    n_hold = sum(1 for r in rows if r["action"] == "HOLD")
    n_new = sum(1 for r in rows if r["action"] == "NEW")
    n_late = sum(1 for r in rows if r["action"] == "LATE")
    n_out = sum(1 for r in rows if r["action"] == "OUT" and r.get("had_trend"))
    n_board = sum(1 for r in rows if r["on_board"])

    replay_rows = []
    for name, feat in features.items():
        if name in skip:
            continue
        replay_rows.append(replay_row(name, feat["flags"], feat["run"], bars[name]))

    outcomes_md = ""
    if persist:
        flags_map = {n: feat.get("flags") or [] for n, feat in features.items()}
        outcomes.rebuild(bars, asof, skip=skip, flags_map=flags_map)
        outcomes.record(rows, asof)
        outcomes_md = outcomes.render(outcomes.load_ledger(), asof)

    day = Path(out_dir or OUT_DIR) / asof
    meta = {
        "date": asof,
        "spy_stage": spy_stage,
        "n_add": n_add,
        "n_hold": n_hold,
        "n_new": n_new,
        "n_late": n_late,
        "n_out": n_out,
        "n_board": n_board,
        "live_schwab": live,
        "orats": bool(token),
        "schwab_http": http_count(),
        "orats_http": process_http(),
        "missing_bars": missing_bars,
        "universe": len(wanted),
        "n_probe": len(discovered),
        "regime": regime.get("risk"),
        "tlt_stage": regime.get("tlt_stage"),
        "uup_stage": regime.get("uup_stage"),
    }
    files = report.write_run(day, asof, rows, meta, spy_feat, replay_rows, outcomes_md)
    return {
        "mode": "full",
        "date": asof,
        "rows": rows,
        "n_add": n_add,
        "n_hold": n_hold,
        "n_new": n_new,
        "n_late": n_late,
        "n_out": n_out,
        "n_board": n_board,
        "orats_token": bool(token),
        "orats_http": process_http(),
        "schwab_http": http_count(),
        "out_dir": str(day),
        "files": files,
        "missing_bars": missing_bars,
        "spy_stage": spy_stage,
        "replay": replay_rows,
    }


def build_replay(
    asof: str,
    out_dir: Optional[Path] = None,
    live_schwab: bool = False,
    no_schwab: bool = False,
    tickers: Optional[Sequence[str]] = None,
    bars_map: Optional[Dict[str, list]] = None,
) -> dict:
    from trendbook import report
    from trendbook.trend import universe_replay

    today = today_et()
    live = use_live_schwab(asof, live_flag=live_schwab, no_schwab=no_schwab, today=today)
    reset_http()
    spy_seed = _load_bars("SPY", asof, live=live, injected=bars_map)
    wanted = _wanted_names(tickers, asof=asof, live=live and tickers is None, spy=spy_seed)
    bars, missing_bars = _load_universe_bars(wanted, asof, live, no_schwab, bars_map)
    rows = universe_replay(bars, asof, skip=_board_skip())
    day = Path(out_dir or OUT_DIR) / asof
    day.mkdir(parents=True, exist_ok=True)
    (day / "replay.md").write_text(report.render_replay(asof, rows), encoding="utf-8")
    report.write_replay_csv(day / "replay.csv", rows)
    return {
        "mode": "replay",
        "date": asof,
        "rows": rows,
        "n_tagged": sum(1 for r in rows if r.get("tagged")),
        "schwab_http": http_count(),
        "out_dir": str(day),
        "missing_bars": missing_bars,
        "files": {"replay": str(day / "replay.md"), "replay_csv": str(day / "replay.csv")},
    }
