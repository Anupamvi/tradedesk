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
import smtplib
import sys
import time
import urllib.request
import urllib.error
from email.mime.text import MIMEText
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
        "sms_phone": env.get("SMS_PHONE", ""),
        "sms_gateway": env.get("SMS_GATEWAY", ""),
        "gmail_user": env.get("GMAIL_USER", ""),
        "gmail_password": env.get("GMAIL_APP_PASSWORD", ""),
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


def send_sms(phone: str, gateway: str, subject: str, body: str) -> bool:
    """Send SMS via T-Mobile email-to-SMS gateway using Gmail SMTP."""
    cfg = load_notify_config()
    gmail_user = cfg.get("gmail_user", "")
    gmail_pass = cfg.get("gmail_password", "")
    if not phone or not gateway or not gmail_user or not gmail_pass:
        _safe_print("  [sms] Missing config (phone/gateway/gmail)")
        return False
    to_addr = f"{phone}@{gateway}"
    sms_body = body[:155] + "..." if len(body) > 160 else body
    try:
        msg = MIMEText(sms_body)
        msg["Subject"] = subject[:30]
        msg["From"] = gmail_user
        msg["To"] = to_addr
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(gmail_user, gmail_pass)
            smtp.sendmail(gmail_user, [to_addr], msg.as_string())
        return True
    except Exception as e:
        _safe_print(f"  [sms] FAILED: {e}")
        return False


def notify(title: str, body: str, priority: str = "default",
           tags: str = "", critical: bool = False) -> None:
    """Send notification via all configured channels. Always sends both ntfy + SMS."""
    cfg = load_notify_config()
    sent = send_ntfy(cfg["ntfy_topic"], title, body, priority, tags)
    if not sent:
        _safe_print(f"  [notify] ntfy failed, message: {_strip_emoji(title)}: {body}")
    send_sms(cfg["sms_phone"], cfg["sms_gateway"], title, body)
    time.sleep(2)  # T-Mobile gateway throttles rapid messages


# ---------------------------------------------------------------------------
# Verdict engine — mirrors the trade-history skill rules
# ---------------------------------------------------------------------------

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
        return ("HOLD", "equity position")

    category = classify_position(pos)

    # ---- CREDIT rules ----
    if category == "CREDIT":
        # Near max profit
        if pct_max >= 85:
            return ("CLOSE", f"{pct_max:.0f}% of max profit — nothing left to harvest")
        # ITM with low DTE
        if dte >= 0 and dte <= 14 and strike > 0 and ul_price > 0:
            is_itm = False
            if pc == "PUT" and qty < 0 and ul_price < strike:
                is_itm = True
            elif pc == "CALL" and qty < 0 and ul_price > strike:
                is_itm = True
            if is_itm:
                return ("ROLL", f"ITM with {dte:.0f} DTE — roll to extend duration")
        # Approaching max (>75%)
        if pct_max >= 75:
            return ("CLOSE", f"{pct_max:.0f}% of max — past 75%% target")
        # Good profit, low DTE
        if pct_max >= 50 and dte >= 0 and dte <= 10:
            return ("CLOSE", f"{pct_max:.0f}% max with {dte:.0f} DTE — diminishing returns")
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
        if pos["asset_type"] == "EQUITY":
            continue  # skip equity monitoring for now

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

        # Detect transitions
        if prev_verdict != verdict:
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


DIP_STATE_FILE = ROOT / "out" / "dip_scanner" / "dip_state.json"


def run_dip_scan() -> List[Dict]:
    """Run dip scanner and return alerts for new opportunities."""
    from uwos.dip_scanner import scan_dips, format_results_md
    from uwos.eod_trade_scan_mode_a import compute_macro_regime

    results = scan_dips(top_n=10, verbose=False)
    if not results:
        return []

    # Save report
    macro = compute_macro_regime(dt.date.today())
    report = format_results_md(results, macro)
    out_dir = ROOT / "out" / "dip_scanner"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dip-scanner-{dt.date.today().isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")

    # Load previous dip state
    prev_dips = {}
    if DIP_STATE_FILE.exists():
        try:
            prev_dips = json.loads(DIP_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    alerts = []
    new_state = {}
    for r in results:
        ticker = r["ticker"]
        new_state[ticker] = {
            "signal": r["signal"],
            "composite": r["composite"],
            "price": r["price"],
            "ret_5d": r["ret_5d"],
            "timestamp": dt.datetime.now().isoformat(),
        }
        prev = prev_dips.get(ticker, {})
        prev_signal = prev.get("signal", "")

        # Alert on new BUY/STRONG BUY entries, or upgrades (WATCH->BUY)
        if r["signal"] in ("STRONG BUY", "BUY"):
            if prev_signal != r["signal"]:
                alerts.append({
                    "symbol": ticker,
                    "underlying": ticker,
                    "transition": f"{prev_signal or 'NEW'} -> {r['signal']}",
                    "verdict": r["signal"],
                    "reason": f"Score {r['composite']:.0f} | 5d {r['ret_5d']:+.1f}% | RSI {r['rsi_14']:.0f} | {r['context']}",
                    "category": "DIP",
                    "pct_max": r["composite"],
                    "pnl": 0,
                    "dte": 0,
                    "ul_price": r["price"],
                    "critical": r["signal"] == "STRONG BUY",
                })

    DIP_STATE_FILE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return alerts


def should_run_dip_scan() -> bool:
    """Run dip scan twice daily: ~10:00 AM and ~1:00 PM ET."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    hour = now.hour
    minute = now.minute
    # Run at 10:00-10:30 and 13:00-13:30 windows
    return (hour == 10 and minute < 30) or (hour == 13 and minute < 30)


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

    # Mode 1: Position monitor
    try:
        alerts = run_scan()
    except Exception as e:
        _safe_print(f"  [ERROR] Position scan failed: {e}")
        notify("Monitor Error", str(e)[:200], priority="high", tags="warning")
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

    # Mode 2: Dip scanner (runs 2x daily)
    if force or should_run_dip_scan():
        _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] Running dip scanner...")
        try:
            dip_alerts = run_dip_scan()
            for alert in dip_alerts:
                title = f"[{alert['verdict']}] {alert['underlying']}"
                body = alert["reason"]
                is_critical = alert.get("critical", False)
                priority = "urgent" if is_critical else "high"
                tags = "money_with_wings" if is_critical else "chart_with_upwards_trend"
                _safe_print(f"    DIP: {title}: {body}")
                notify(title, body, priority=priority, tags=tags, critical=True)
            total_alerts += len(dip_alerts)
            if not dip_alerts:
                _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] No new dip opportunities")
        except Exception as e:
            _safe_print(f"  [ERROR] Dip scan failed: {e}")

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
