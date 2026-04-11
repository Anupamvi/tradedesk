"""
Trade Monitor — automated position surveillance with push notifications.

Runs schwab_position_analyzer, applies credit/debit verdict rules,
diffs against previous state, and sends alerts for all transitions.

Usage:
    python -m uwos.trade_monitor              # single run
    python -m uwos.trade_monitor --loop 30    # run every 30 min during market hours
    python -m uwos.trade_monitor --test       # send a test notification
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("c:/uw_root")
STATE_FILE = ROOT / "out" / "trade_analysis" / "monitor_state.json"
LOG_FILE = ROOT / "out" / "trade_analysis" / "monitor_log.jsonl"


# ---------------------------------------------------------------------------
# Notification backends
# ---------------------------------------------------------------------------

def load_notify_config() -> Dict[str, str]:
    """Read notification config from .env."""
    from dotenv import dotenv_values
    env = dotenv_values(ROOT / ".env")
    return {
        "ntfy_topic": env.get("NTFY_TOPIC", ""),
    }


def send_ntfy(topic: str, title: str, body: str, priority: str = "default",
              tags: str = "") -> bool:
    """Push notification via ntfy.sh."""
    if not topic:
        return False
    try:
        # Use JSON publish to avoid header encoding issues on Windows
        payload = {
            "topic": topic,
            "title": _strip_emoji(title),
            "message": body,
            "priority": _priority_int(priority),
        }
        if tags:
            payload["tags"] = [t.strip() for t in tags.split(",")]
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        _safe_print(f"  [ntfy] FAILED: {e}")
        return False


def _strip_emoji(text: str) -> str:
    """Remove emoji for header safety; ntfy tags handle emoji display."""
    return text.encode("ascii", "ignore").decode("ascii").strip()


def _priority_int(p: str) -> int:
    return {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}.get(p, 3)


def _safe_print(msg: str) -> None:
    """Print without crashing on Windows cp1252 encoding."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def notify(title: str, body: str, priority: str = "default",
           tags: str = "", critical: bool = False) -> None:
    """Send notification via ntfy push."""
    cfg = load_notify_config()
    sent = send_ntfy(cfg["ntfy_topic"], title, body, priority, tags)
    if not sent:
        _safe_print(f"  [notify] ntfy failed, message: {_strip_emoji(title)}: {body}")


# ---------------------------------------------------------------------------
# Verdict engine — mirrors the trade-history skill rules
# ---------------------------------------------------------------------------

_spy_change_cache = None


def _get_spy_change() -> float:
    """Get SPY 5-day % change. Cached per session to avoid repeated API calls."""
    global _spy_change_cache
    if _spy_change_cache is not None:
        return _spy_change_cache
    try:
        from uwos.eod_trade_scan_mode_a import compute_macro_regime
        macro = compute_macro_regime(dt.date.today())
        _spy_change_cache = macro["spy_5d_ret"] * 100
    except Exception:
        _spy_change_cache = 0.0
    return _spy_change_cache


def classify_position(pos: Dict) -> str:
    """Classify as CREDIT, DEBIT, or EQUITY."""
    if pos["asset_type"] == "EQUITY":
        return "EQUITY"
    qty = pos.get("qty", 0)
    if qty < 0:
        return "CREDIT"
    return "DEBIT"


def safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def compute_verdict(pos: Dict) -> Tuple[str, str]:
    """Return (verdict, reason) for a single position.

    Returns one of: HOLD, CLOSE, ROLL, ASSESS
    """
    c = pos["computed"]
    atype = pos["asset_type"]
    qty = pos.get("qty", 0)
    pct_max = safe(c.get("pct_of_max_profit"))
    pnl_pct = safe(c.get("unrealized_pnl_pct"))
    dte = safe(c.get("dte"), -1)
    delta = safe((pos.get("greeks") or {}).get("delta"))
    strike = safe(pos.get("strike"))
    pc = pos.get("put_call", "")
    ul_price = safe((pos.get("underlying_quote") or {}).get("last"))
    sym = pos.get("symbol", "")

    if atype == "EQUITY":
        pnl = safe(c.get("unrealized_pnl"))
        change_today = safe((pos.get("underlying_quote") or {}).get("change_pct"))
        spy_corr = safe(c.get("spy_correlation_20d"), 0.5)

        # Intraday rapid-drop alert (Schwab netPercentChange)
        if change_today <= -7:
            return ("CLOSE", f"equity CRASHED {change_today:+.1f}% TODAY (${pnl:.0f} total) — emergency")
        if change_today <= -5:
            return ("ASSESS", f"equity dropped {change_today:+.1f}% TODAY (${pnl:.0f} total) — rapid drop")

        # Context-aware thresholds: compare stock drop to market
        # If SPY also dropped hard (via spy_correlation), widen thresholds
        # In a broad crash, stocks dropping WITH the market is normal
        # In a stock-specific drop, tighter thresholds apply
        spy_5d = _get_spy_change()
        in_broad_selloff = spy_5d < -3.0  # SPY down >3% in 5 days

        # Adjust thresholds: in a crash, allow more drawdown before alerting
        close_threshold = -55 if in_broad_selloff else -40
        assess_threshold = -35 if in_broad_selloff else -25
        tax_harvest_threshold = -60  # always harvest at -60%

        # Near worthless: always
        if pnl_pct <= tax_harvest_threshold:
            return ("CLOSE", f"equity down {pnl_pct:.0f}% (${pnl:.0f}) — tax-loss harvest")

        # Deep loss: tighter if stock-specific, wider if market crash
        if pnl_pct <= close_threshold:
            ctx = "broad selloff" if in_broad_selloff else "stock-specific"
            return ("CLOSE", f"equity down {pnl_pct:.0f}% (${pnl:.0f}) — {ctx}, cut or justify")

        # Review: same context logic
        if pnl_pct <= assess_threshold:
            ctx = "market-wide" if in_broad_selloff else "underperforming"
            return ("ASSESS", f"equity down {pnl_pct:.0f}% (${pnl:.0f}) — {ctx}, review thesis")

        # Take profit: up > 100%
        if pnl_pct >= 100:
            return ("ASSESS", f"equity up +{pnl_pct:.0f}% (+${pnl:.0f}) — consider trimming")

        # Strong gain: up > 50%
        if pnl_pct >= 50:
            return ("HOLD", f"equity +{pnl_pct:.0f}% — strong, trail stop")

        return ("HOLD", f"equity {pnl_pct:+.0f}%")

    category = classify_position(pos)

    # ---- CREDIT rules ----
    if category == "CREDIT":
        # Near max profit
        if pct_max >= 85:
            return ("CLOSE", f"{pct_max:.0f}% of max profit — nothing left to harvest")

        # ITM detection
        is_itm = False
        itm_pct = 0.0
        if strike > 0 and ul_price > 0:
            if pc == "PUT" and qty < 0 and ul_price < strike:
                is_itm = True
                itm_pct = (strike - ul_price) / strike * 100
            elif pc == "CALL" and qty < 0 and ul_price > strike:
                is_itm = True
                itm_pct = (ul_price - strike) / strike * 100

        # Assignment risk: deep ITM + DTE < 5 = likely assigned
        if is_itm and dte >= 0 and dte <= 5 and abs(delta) > 0.85:
            return ("CLOSE", f"ASSIGNMENT RISK: ITM {itm_pct:.0f}%%, delta {delta:+.2f}, {dte:.0f} DTE — close or roll NOW")

        # ITM + DTE < 14: ROLL
        if is_itm and dte >= 0 and dte <= 14:
            return ("ROLL", f"ITM by {itm_pct:.1f}%% with {dte:.0f} DTE — roll now")

        # Pin risk: within 1% of strike with DTE < 3
        if not is_itm and dte >= 0 and dte <= 3 and strike > 0 and ul_price > 0:
            dist = abs(ul_price - strike) / strike * 100
            if dist < 1.5:
                return ("CLOSE", f"PIN RISK: {dist:.1f}%% from strike, {dte:.0f} DTE — close to avoid assignment")

        # ITM + deep (>5% or delta > 0.50): ASSESS regardless of DTE
        if is_itm and (itm_pct > 5 or abs(delta) > 0.50):
            return ("ASSESS", f"ITM by {itm_pct:.1f}%% (delta {delta:+.2f}) {dte:.0f} DTE — review, consider rolling")
        # ITM at all: ASSESS
        if is_itm:
            return ("ASSESS", f"ITM by {itm_pct:.1f}%% with {dte:.0f} DTE — monitor closely")

        # Expiration week: DTE < 7 with less than 50% max = gamma risk
        if dte >= 0 and dte <= 7 and pct_max < 50:
            return ("CLOSE", f"{pct_max:.0f}%% max with {dte:.0f} DTE — expiration week gamma risk")

        # Earnings proximity: CLOSE or ASSESS if earnings within 7 days
        earnings_days = safe(c.get("days_to_earnings"), 999)
        if 0 < earnings_days <= 7:
            if pct_max > 0:
                return ("CLOSE", f"EARNINGS in {earnings_days:.0f}d — take {pct_max:.0f}%% profit before binary event")
            else:
                return ("ASSESS", f"EARNINGS in {earnings_days:.0f}d — at {pct_max:.0f}%% max, assess hold vs close")

        # Approaching max (>75%)
        if pct_max >= 75:
            return ("CLOSE", f"{pct_max:.0f}% of max — past 75%% target")
        # Good profit, low DTE
        if pct_max >= 50 and dte >= 0 and dte <= 10:
            return ("CLOSE", f"{pct_max:.0f}% max with {dte:.0f} DTE — diminishing returns")
        # High delta without being ITM (approaching ATM)
        if abs(delta) > 0.45:
            return ("ASSESS", f"delta {delta:+.2f} — approaching ATM, {pct_max:.0f}%% max, {dte:.0f} DTE")
        # Deep loss on credit
        if pct_max <= -80:
            return ("ASSESS", f"{pct_max:.0f}% — deep loss, review thesis")
        return ("HOLD", f"{pct_max:.0f}% max, {dte:.0f} DTE")

    # ---- DEBIT rules ----
    if category == "DEBIT":
        # Calculate OTM distance for long leg
        otm_pct = 0.0
        if strike > 0 and ul_price > 0:
            if pc == "CALL":
                otm_pct = (strike - ul_price) / ul_price * 100  # positive = OTM
            elif pc == "PUT":
                otm_pct = (ul_price - strike) / ul_price * 100  # positive = OTM

        # Down > 60% = CLOSE
        if pnl_pct <= -60:
            return ("CLOSE", f"down {pnl_pct:.0f}%% — debit rule >60%% loss")
        # OTM > 5% with DTE < 35
        if otm_pct > 5 and dte >= 0 and dte < 35:
            return ("CLOSE", f"OTM {otm_pct:.1f}%% with {dte:.0f} DTE — debit rule")
        # Any amount OTM with DTE < 14
        if otm_pct > 0 and dte >= 0 and dte < 14:
            return ("CLOSE", f"OTM with {dte:.0f} DTE — theta acceleration")
        # Down > 40% = ASSESS
        if pnl_pct <= -40:
            return ("ASSESS", f"down {pnl_pct:.0f}%% — escalated review")
        # OTM 3-5% with DTE < 35
        if otm_pct > 3 and dte >= 0 and dte < 35:
            return ("ASSESS", f"OTM {otm_pct:.1f}%% with {dte:.0f} DTE")
        return ("HOLD", f"{'ITM' if otm_pct <= 0 else f'OTM {otm_pct:.1f}%%'}, {dte:.0f} DTE")

    return ("HOLD", "unknown")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> Dict[str, Dict]:
    """Load previous monitor state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Dict]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def position_key(pos: Dict) -> str:
    """Unique key for a position."""
    sym = pos.get("symbol", "unknown")
    return sym.strip()


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_scan() -> List[Dict]:
    """Run position analyzer and return alerts."""
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    from uwos.schwab_position_analyzer import analyze_positions

    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, interactive_login=False)

    result = analyze_positions(svc=svc, days=90)
    positions = result.get("positions", [])
    account = result.get("account_summary", {})

    # Save position data
    out_dir = ROOT / "out" / "trade_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    today_str = dt.date.today().isoformat()
    json_path = out_dir / f"position_data_{today_str}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Compute verdicts
    prev_state = load_state()
    new_state = {}
    alerts = []

    for pos in positions:
        key = position_key(pos)
        verdict, reason = compute_verdict(pos)
        category = classify_position(pos)
        pct_max = safe(pos["computed"].get("pct_of_max_profit"))
        pnl = safe(pos["computed"].get("unrealized_pnl"))
        dte = safe(pos["computed"].get("dte"), -1)
        ul_price = safe((pos.get("underlying_quote") or {}).get("last"))
        underlying = pos.get("underlying", "") or pos.get("symbol", "")

        new_state[key] = {
            "verdict": verdict,
            "reason": reason,
            "category": category,
            "pct_max": pct_max,
            "pnl": pnl,
            "dte": dte,
            "ul_price": ul_price,
            "timestamp": dt.datetime.now().isoformat(),
        }

        prev = prev_state.get(key, {})
        prev_verdict = prev.get("verdict", "NEW")

        # Detect transitions with hysteresis to prevent flip-flopping
        # Once escalated (HOLD->ASSESS or ASSESS->CLOSE), require significant
        # improvement to de-escalate — prevents noisy ASSESS->HOLD->ASSESS cycles
        prev_pnl = prev.get("pnl", 0)

        # Hysteresis: suppress de-escalation unless improvement is significant
        VERDICT_RANK = {"HOLD": 0, "ASSESS": 1, "ROLL": 2, "CLOSE": 3}
        cur_rank = VERDICT_RANK.get(verdict, 0)
        prev_rank = VERDICT_RANK.get(prev_verdict, 0)
        is_escalation = cur_rank > prev_rank
        is_deescalation = cur_rank < prev_rank

        # Allow de-escalation only if P&L improved by $300+ or 10%+ from when it escalated
        if is_deescalation:
            pnl_improvement = pnl - prev_pnl
            if pnl_improvement < 300:
                # Suppress de-escalation — not enough improvement to justify
                verdict = prev_verdict
                reason = prev.get("reason", reason) + " (sticky)"

        # Re-alert if worsening within same ASSESS/CLOSE verdict by $500+
        worsened = (verdict in ("ASSESS", "CLOSE") and
                    prev_verdict == verdict and
                    pnl < prev_pnl - 500)

        if prev_verdict != verdict or worsened:
            alert = {
                "symbol": key,
                "underlying": underlying,
                "transition": f"{prev_verdict} -> {verdict}",
                "verdict": verdict,
                "reason": reason,
                "category": category,
                "pct_max": pct_max,
                "pnl": pnl,
                "dte": dte,
                "ul_price": ul_price,
                "critical": verdict in ("CLOSE", "ROLL"),
            }
            alerts.append(alert)

    # Detect closed positions (in prev but not in new)
    for key, prev in prev_state.items():
        if key not in new_state and prev.get("verdict") != "CLOSED":
            alerts.append({
                "symbol": key,
                "underlying": key,
                "transition": f"{prev.get('verdict', '?')} -> CLOSED",
                "verdict": "CLOSED",
                "reason": "Position no longer in account",
                "category": prev.get("category", "?"),
                "pct_max": prev.get("pct_max", 0),
                "pnl": prev.get("pnl", 0),
                "dte": 0,
                "ul_price": 0,
                "critical": False,
            })

    save_state(new_state)

    # Log
    log_entry = {
        "timestamp": dt.datetime.now().isoformat(),
        "account_value": account.get("total_value", 0),
        "cash": account.get("cash", 0),
        "positions": len(positions),
        "alerts": len(alerts),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return alerts


def format_alert(alert: Dict) -> Tuple[str, str]:
    """Format an alert into (title, body)."""
    v = alert["verdict"]
    sym = alert["symbol"][:20]
    underlying = alert.get("underlying", sym)

    emoji = {"CLOSE": "[CLOSE]", "ROLL": "[ROLL]", "ASSESS": "[ASSESS]",
             "HOLD": "[HOLD]", "CLOSED": "[CLOSED]", "NEW": "[NEW]"}.get(v, "[?]")

    title = f"{emoji} {v}: {underlying}"

    parts = [
        alert["transition"],
        alert["reason"],
    ]
    if alert.get("pct_max"):
        parts.append(f"Max: {alert['pct_max']:.0f}%")
    if alert.get("pnl"):
        parts.append(f"P&L: ${alert['pnl']:.0f}")
    if alert.get("dte") and alert["dte"] > 0:
        parts.append(f"DTE: {alert['dte']:.0f}")
    if alert.get("ul_price"):
        parts.append(f"Price: ${alert['ul_price']:.2f}")

    body = " | ".join(parts)
    return title, body


def is_market_hours() -> bool:
    """Check if we're in US market hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    return market_open <= now <= market_close


IDEAS_STATE_FILE = ROOT / "out" / "trade_ideas" / "ideas_state.json"


def run_trade_ideas_scan() -> List[Dict]:
    """Run unified trade ideas scanner and return alerts for new ideas."""
    from uwos.trade_ideas import scan_trade_ideas, format_results_md, format_alert, find_latest_data_dir
    from uwos.eod_trade_scan_mode_a import compute_macro_regime

    data_dir = find_latest_data_dir()
    results = scan_trade_ideas(data_dir=data_dir, top_n=8, verbose=False)
    if not results:
        return []

    # Save report
    macro = compute_macro_regime(dt.date.today())
    report = format_results_md(results, macro)
    out_dir = ROOT / "out" / "trade_ideas"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trade-ideas-{dt.date.today().isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")

    # Load previous state
    prev = {}
    if IDEAS_STATE_FILE.exists():
        try:
            prev = json.loads(IDEAS_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    alerts = []
    new_state = {}
    for r in results:
        ticker = r["ticker"]
        new_state[ticker] = {
            "strategy": r["strategy"],
            "composite": r["composite"],
            "short_strike": r["short_strike"],
            "expiry": r["expiry"],
            "timestamp": dt.datetime.now().isoformat(),
        }

        # Alert if this is a new ticker or the trade changed
        prev_entry = prev.get(ticker, {})
        if (prev_entry.get("strategy") != r["strategy"] or
                prev_entry.get("short_strike") != r["short_strike"]):
            alerts.append(r)

    IDEAS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    IDEAS_STATE_FILE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return alerts


def should_run_ideas_scan() -> bool:
    """Run trade ideas scan every hour during market hours (on the hour)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    # Run at the top of each hour during market hours: 10, 11, 12, 1, 2, 3 PM
    return 10 <= now.hour <= 15 and now.minute < 30


def _is_market_open_window() -> bool:
    """True during the first 30 min after market open (9:30-10:00 ET)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    return now.hour == 9 and 30 <= now.minute <= 59


def run_once(force: bool = False, manual: bool = False) -> int:
    """Run a single scan and notify. Returns number of alerts.

    manual=True: notify ALL actionable verdicts (CLOSE/ROLL/ASSESS), not just transitions.
    manual=False (scheduled): only notify on state transitions.
    """
    if not force and not manual and not is_market_hours():
        _safe_print(f"  [{dt.datetime.now():%H:%M}] Market closed, skipping scan")
        return 0

    _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] Running position scan...")
    total_alerts = 0

    # Heartbeat: send once at market open (9:30-10:00 ET)
    if _is_market_open_window():
        state = load_state()
        close_count = sum(1 for v in state.values() if v.get("verdict") == "CLOSE")
        assess_count = sum(1 for v in state.values() if v.get("verdict") == "ASSESS")
        notify("Trade Desk Active",
               f"Monitor running. {close_count} CLOSE, {assess_count} ASSESS positions.",
               priority="low", tags="white_check_mark")

    # Mode 1: Position monitor
    try:
        alerts = run_scan()
    except Exception as e:
        err_msg = str(e)[:200]
        _safe_print(f"  [ERROR] Position scan failed: {e}")
        # Specific auth failure detection
        if "token" in err_msg.lower() or "auth" in err_msg.lower() or "401" in err_msg:
            notify("AUTH EXPIRED",
                   "Schwab token expired. Run in terminal: del c:\\uw_root\\tokens\\schwab_token.json && python -m uwos.schwab_position_analyzer --manual-auth",
                   priority="urgent", tags="rotating_light")
        else:
            notify("Monitor Error", err_msg, priority="high", tags="warning")
        alerts = []

    # In manual mode, send ALL current CLOSE/ROLL/ASSESS verdicts (not just transitions)
    if manual:
        state = load_state()  # freshly written by run_scan above
        for key, val in state.items():
            verdict = val.get("verdict", "HOLD")
            if verdict in ("CLOSE", "ROLL", "ASSESS"):
                already_alerted = any(a["symbol"] == key for a in alerts)
                if not already_alerted:
                    alerts.append({
                        "symbol": key,
                        "underlying": key,
                        "transition": f"CURRENT: {verdict}",
                        "verdict": verdict,
                        "reason": val.get("reason", ""),
                        "category": val.get("category", ""),
                        "pct_max": val.get("pct_max", 0),
                        "pnl": val.get("pnl", 0),
                        "dte": val.get("dte", 0),
                        "ul_price": val.get("ul_price", 0),
                        "critical": True,
                    })

    for alert in alerts:
        title, body = format_alert(alert)
        priority = "urgent" if alert.get("critical") else "default"
        tags = "rotating_light" if alert.get("critical") else "chart_with_upwards_trend"
        _safe_print(f"    {title}: {body}")
        notify(title, body, priority=priority, tags=tags, critical=True)
    total_alerts += len(alerts)

    # Mode 2: Trade ideas scanner (runs 2x daily OR on manual)
    if force or manual or should_run_ideas_scan():
        _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] Running trade ideas scanner...")
        try:
            from uwos.trade_ideas import format_alert as fmt_idea
            idea_alerts = run_trade_ideas_scan()

            # In manual mode, send ALL current ideas (not just new transitions)
            if manual and not idea_alerts:
                ideas_state = {}
                if IDEAS_STATE_FILE.exists():
                    try:
                        ideas_state = json.loads(IDEAS_STATE_FILE.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                for ticker, val in ideas_state.items():
                    idea_alerts.append({
                        "ticker": ticker,
                        "strategy": val.get("strategy", "?"),
                        "short_strike": val.get("short_strike", 0),
                        "composite": val.get("composite", 0),
                    })

            for r in idea_alerts:
                title = f"NEW TRADE: {r['ticker']} {r.get('strategy', '?')}"
                try:
                    body = fmt_idea(r)
                except Exception:
                    body = f"{r.get('strategy','?')} | Score: {r.get('composite',0):.0f}"
                _safe_print(f"    IDEA: {title}: {body}")
                notify(title, body, priority="high", tags="chart_with_upwards_trend")
            total_alerts += len(idea_alerts)
            if not idea_alerts:
                _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] No trade ideas")
        except Exception as e:
            _safe_print(f"  [ERROR] Trade ideas scan failed: {e}")

    if total_alerts == 0:
        _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] No alerts")

    return total_alerts


def main():
    parser = argparse.ArgumentParser(description="Trade position monitor with push notifications")
    parser.add_argument("--loop", type=int, default=0,
                        help="Run every N minutes (0 = single run)")
    parser.add_argument("--test", action="store_true",
                        help="Send a test notification and exit")
    parser.add_argument("--force", action="store_true",
                        help="Run even outside market hours")
    parser.add_argument("--manual", action="store_true",
                        help="Manual run — notify ALL current verdicts (not just transitions)")
    args = parser.parse_args()

    if args.test:
        print("Sending test notification...")
        notify(
            "Trade Monitor Test",
            "If you see this, notifications are working! Monitor will send alerts for HOLD->CLOSE, ROLL, and other transitions.",
            priority="default",
            tags="white_check_mark",
        )
        print("Done. Check your ntfy app.")
        return

    if args.loop > 0:
        print(f"Trade Monitor starting — scanning every {args.loop} min during market hours")
        print(f"  ntfy topic: {load_notify_config()['ntfy_topic']}")
        print(f"  State file: {STATE_FILE}")
        print(f"  Press Ctrl+C to stop")
        while True:
            try:
                run_once(force=args.force, manual=args.manual)
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                print("\nMonitor stopped.")
                break
    else:
        run_once(force=args.force, manual=args.manual)


if __name__ == "__main__":
    main()
