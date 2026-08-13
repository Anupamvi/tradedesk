"""
Trade Monitor — automated position surveillance with push notifications.

Runs schwab_position_analyzer, applies credit/debit verdict rules,
diffs against previous state, and sends alerts for all transitions.

Usage:
    python -m uwos.trade_monitor              # single run
    python -m uwos.trade_monitor --loop 5     # run every 5 min during market hours
    python -m uwos.trade_monitor --test       # send a test notification
    python -m uwos.trade_monitor --manual-test # send a manual-monitor style test
"""

import argparse
import datetime as dt
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from uwos.paths import project_root
from uwos.spread_positions import (
    build_position_review_items,
    compute_spread_metrics,
    compute_spread_verdict,
    current_leg_keys,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = project_root()
STATE_FILE = ROOT / "out" / "trade_analysis" / "monitor_state.json"
LOG_FILE = ROOT / "out" / "trade_analysis" / "monitor_log.jsonl"
MANUAL_MONITORS_FILE = ROOT / "out" / "trade_analysis" / "manual_monitors.json"


# ---------------------------------------------------------------------------
# Notification backends
# ---------------------------------------------------------------------------

def load_notify_config() -> Dict[str, str]:
    """Read notification config from cloud env, falling back to local .env."""
    from dotenv import dotenv_values
    env = dotenv_values(ROOT / ".env")
    return {
        "ntfy_server": os.environ.get("NTFY_SERVER") or env.get("NTFY_SERVER", "https://ntfy.sh"),
        "ntfy_topic": os.environ.get("NTFY_TOPIC") or env.get("NTFY_TOPIC", ""),
        "ntfy_token": os.environ.get("NTFY_TOKEN") or env.get("NTFY_TOKEN", ""),
        "ntfy_phone_topic": os.environ.get("NTFY_PHONE_TOPIC") or env.get("NTFY_PHONE_TOPIC", ""),
        "ntfy_manual_topic": os.environ.get("NTFY_MANUAL_TOPIC") or env.get("NTFY_MANUAL_TOPIC", ""),
        "manual_alert_prefix": os.environ.get("MANUAL_ALERT_PREFIX") or env.get("MANUAL_ALERT_PREFIX", "MANUAL MONITOR"),
        "manual_alert_tags": os.environ.get("MANUAL_ALERT_TAGS") or env.get("MANUAL_ALERT_TAGS", "rotating_light,warning"),
        "manual_monitors_path": os.environ.get("MANUAL_MONITORS_PATH") or env.get("MANUAL_MONITORS_PATH", ""),
        "phone_notify_mode": (
            os.environ.get("PHONE_NOTIFY_MODE") or env.get("PHONE_NOTIFY_MODE", "ntfy")
        ).lower(),
        "twilio_account_sid": os.environ.get("TWILIO_ACCOUNT_SID") or env.get("TWILIO_ACCOUNT_SID", ""),
        "twilio_auth_token": os.environ.get("TWILIO_AUTH_TOKEN") or env.get("TWILIO_AUTH_TOKEN", ""),
        "twilio_from": os.environ.get("TWILIO_FROM") or env.get("TWILIO_FROM", ""),
        "sms_to": os.environ.get("SMS_TO") or env.get("SMS_TO", ""),
    }


def send_ntfy(topic: str, title: str, body: str, priority: str = "default",
              tags: str = "", *, server: str = "https://ntfy.sh", token: str = "") -> bool:
    """Push notification via ntfy.sh. Retries once on failure."""
    if not topic:
        return False
    server = (server or "https://ntfy.sh").rstrip("/")
    payload = {
        "topic": topic,
        "title": _strip_emoji(title),
        "message": body,
        "priority": _priority_int(priority),
    }
    if tags:
        payload["tags"] = [t.strip() for t in tags.split(",")]
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(2):  # retry once
        try:
            req = urllib.request.Request(server, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            if attempt == 0:
                time.sleep(2)  # brief pause before retry
            else:
                _safe_print(f"  [ntfy] FAILED after retry: {e}")
    return False


def _strip_emoji(text: str) -> str:
    """Remove emoji for header safety; ntfy tags handle emoji display."""
    return text.encode("ascii", "ignore").decode("ascii").strip()


def _priority_int(p: str) -> int:
    return {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}.get(p, 3)


def _phone_mode_enabled(mode: str, channel: str) -> bool:
    tokens = {part.strip().lower() for part in str(mode or "").split(",") if part.strip()}
    if not tokens:
        return False
    if "off" in tokens or "none" in tokens:
        return False
    return "both" in tokens or "all" in tokens or channel in tokens


def _merge_tags(*tag_groups: str) -> str:
    seen = set()
    merged = []
    for group in tag_groups:
        for tag in str(group or "").split(","):
            cleaned = tag.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                merged.append(cleaned)
    return ",".join(merged)


def _manual_alert_title(title: str, cfg: Dict[str, str]) -> str:
    prefix = (cfg.get("manual_alert_prefix") or "MANUAL MONITOR").strip()
    clean_title = _strip_emoji(title)
    if not prefix:
        return clean_title
    if clean_title.upper().startswith(prefix.upper()):
        return clean_title
    return f"{prefix} - {clean_title}"


def send_sms_twilio(cfg: Dict[str, str], title: str, body: str) -> bool:
    """Send an optional SMS via Twilio. Returns False when not configured."""
    sid = cfg.get("twilio_account_sid", "")
    token = cfg.get("twilio_auth_token", "")
    sender = cfg.get("twilio_from", "")
    recipient = cfg.get("sms_to", "")
    if not all([sid, token, sender, recipient]):
        return False
    text = f"{_strip_emoji(title)}\n{body}"[:1500]
    data = urllib.parse.urlencode({"From": sender, "To": recipient, "Body": text}).encode("utf-8")
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Authorization", f"Basic {auth}")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                _safe_print(f"  [sms] FAILED after retry: {e}")
    return False


def _masked_secret_status(value: str) -> str:
    if not value:
        return "unset"
    if len(value) <= 4:
        return "set"
    return f"set ({len(value)} chars)"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _safe_print(msg: str) -> None:
    """Print without crashing on Windows cp1252 encoding."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _is_schwab_credential_failure(error: Exception) -> bool:
    """Return True only for broker responses that require a fresh OAuth login."""
    text = str(error or "").lower()
    markers = (
        "refresh_token_authentication_error",
        "unsupported_token_type",
        "invalid_grant",
        "invalid token",
        "token expired",
        "token is expired",
        "401 unauthorized",
        "401 client error",
        "403 forbidden",
    )
    return any(marker in text for marker in markers)


def notify(title: str, body: str, priority: str = "default",
           tags: str = "", critical: bool = False, manual: bool = False) -> None:
    """Send notification via ntfy, plus optional phone channels for risk/manual alerts."""
    cfg = load_notify_config()
    send_title = _manual_alert_title(title, cfg) if manual else title
    send_priority = "urgent" if manual else priority
    send_tags = _merge_tags(cfg.get("manual_alert_tags", ""), tags) if manual else tags

    sent = send_ntfy(
        cfg["ntfy_topic"],
        send_title,
        body,
        send_priority,
        send_tags,
        server=cfg["ntfy_server"],
        token=cfg["ntfy_token"],
    )
    if not sent:
        _safe_print(f"  [notify] ntfy failed, message: {_strip_emoji(send_title)}: {body}")
    if not critical and not manual:
        return

    phone_mode = cfg.get("phone_notify_mode", "")
    if _phone_mode_enabled(phone_mode, "ntfy"):
        phone_topic = (
            cfg.get("ntfy_manual_topic", "") if manual and cfg.get("ntfy_manual_topic", "")
            else cfg.get("ntfy_phone_topic", "")
        )
        if phone_topic and phone_topic != cfg.get("ntfy_topic", ""):
            phone_sent = send_ntfy(
                phone_topic,
                send_title if manual else f"PHONE ALERT - {_strip_emoji(send_title)}",
                body,
                "urgent",
                send_tags or "rotating_light",
                server=cfg["ntfy_server"],
                token=cfg["ntfy_token"],
            )
            if not phone_sent:
                _safe_print(f"  [notify] phone ntfy failed: {_strip_emoji(send_title)}: {body}")
    if _phone_mode_enabled(phone_mode, "sms"):
        sms_sent = send_sms_twilio(cfg, send_title, body)
        if not sms_sent:
            _safe_print(f"  [notify] SMS not sent or not configured: {_strip_emoji(send_title)}")


# ---------------------------------------------------------------------------
# Verdict engine — mirrors the trade-history skill rules
# ---------------------------------------------------------------------------

_spy_change_cache = None
_spy_change_ts = None


def _get_spy_change() -> float:
    """Get SPY 5-day % change. Cached for 10 min to avoid repeated API calls."""
    global _spy_change_cache, _spy_change_ts
    now = dt.datetime.now()
    if _spy_change_cache is not None and _spy_change_ts and (now - _spy_change_ts).seconds < 600:
        return _spy_change_cache
    try:
        from uwos.eod_trade_scan_mode_a import compute_macro_regime
        macro = compute_macro_regime(dt.date.today())
        _spy_change_cache = macro["spy_5d_ret"] * 100
        _spy_change_ts = now
    except Exception:
        _spy_change_cache = 0.0
        _spy_change_ts = now
    return _spy_change_cache


def classify_position(pos: Dict) -> str:
    """Classify as CREDIT, DEBIT, EQUITY, or OTHER."""
    asset_type = pos.get("asset_type", "")
    if asset_type == "EQUITY":
        return "EQUITY"
    if asset_type != "OPTION":
        return "OTHER"
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

    if atype != "OPTION":
        return ("HOLD", f"{atype or 'asset'} not covered by option verdict rules")

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
            return ("CLOSE", f"ASSIGNMENT RISK: ITM {itm_pct:.0f}%, delta {delta:+.2f}, {dte:.0f} DTE — close or roll NOW")

        # ITM + DTE < 14: ROLL
        if is_itm and dte >= 0 and dte <= 14:
            return ("ROLL", f"ITM by {itm_pct:.1f}% with {dte:.0f} DTE — roll now")

        # Pin risk: within 1% of strike with DTE < 3
        if not is_itm and dte >= 0 and dte <= 3 and strike > 0 and ul_price > 0:
            dist = abs(ul_price - strike) / strike * 100
            if dist < 1.5:
                return ("CLOSE", f"PIN RISK: {dist:.1f}% from strike, {dte:.0f} DTE — close to avoid assignment")

        # ITM + deep (>5% or delta > 0.50): ASSESS regardless of DTE
        if is_itm and (itm_pct > 5 or abs(delta) > 0.50):
            return ("ASSESS", f"ITM by {itm_pct:.1f}% (delta {delta:+.2f}) {dte:.0f} DTE — review, consider rolling")
        # ITM at all: ASSESS
        if is_itm:
            return ("ASSESS", f"ITM by {itm_pct:.1f}% with {dte:.0f} DTE — monitor closely")

        # Expiration week: DTE < 7 with less than 50% max = gamma risk
        if dte >= 0 and dte <= 7 and pct_max < 50:
            return ("CLOSE", f"{pct_max:.0f}% max with {dte:.0f} DTE — expiration week gamma risk")

        # Earnings proximity: CLOSE or ASSESS if earnings within 7 days
        earnings_days = safe(c.get("days_to_earnings"), 999)
        if 0 < earnings_days <= 7:
            if pct_max >= 25:
                return ("CLOSE", f"EARNINGS in {earnings_days:.0f}d — take {pct_max:.0f}% profit before binary event")
            elif pct_max > 0:
                return ("ASSESS", f"EARNINGS in {earnings_days:.0f}d — only {pct_max:.0f}% profit, assess risk vs reward")
            else:
                return ("ASSESS", f"EARNINGS in {earnings_days:.0f}d — at {pct_max:.0f}% max, assess hold vs close")

        # Approaching max (>75%)
        if pct_max >= 75:
            return ("CLOSE", f"{pct_max:.0f}% of max — past 75% target")
        # Good profit, low DTE
        if pct_max >= 50 and dte >= 0 and dte <= 10:
            return ("CLOSE", f"{pct_max:.0f}% max with {dte:.0f} DTE — diminishing returns")
        # High delta without being ITM (approaching ATM)
        if abs(delta) > 0.45:
            return ("ASSESS", f"delta {delta:+.2f} — approaching ATM, {pct_max:.0f}% max, {dte:.0f} DTE")
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
            return ("CLOSE", f"down {pnl_pct:.0f}% — debit rule >60% loss")
        # OTM > 5% with DTE < 35
        if otm_pct > 5 and dte >= 0 and dte < 35:
            return ("CLOSE", f"OTM {otm_pct:.1f}% with {dte:.0f} DTE — debit rule")
        # Any amount OTM with DTE < 14
        if otm_pct > 0 and dte >= 0 and dte < 14:
            return ("CLOSE", f"OTM with {dte:.0f} DTE — theta acceleration")
        # Down > 40% = ASSESS
        if pnl_pct <= -40:
            return ("ASSESS", f"down {pnl_pct:.0f}% — escalated review")
        # OTM 3-5% with DTE < 35
        if otm_pct > 3 and dte >= 0 and dte < 35:
            return ("ASSESS", f"OTM {otm_pct:.1f}% with {dte:.0f} DTE")
        # Earnings proximity for debit: IV crush risk
        earnings_days = safe(c.get("days_to_earnings"), 999)
        if 0 < earnings_days <= 5:
            if pnl_pct >= 20:
                return ("CLOSE", f"EARNINGS in {earnings_days:.0f}d — take +{pnl_pct:.0f}% profit, IV crush risk after")
            else:
                return ("ASSESS", f"EARNINGS in {earnings_days:.0f}d — IV crush risk, assess exit")
        return ("HOLD", f"{'ITM' if otm_pct <= 0 else f'OTM {otm_pct:.1f}%'}, {dte:.0f} DTE")

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
# Manual brink monitors
# ---------------------------------------------------------------------------

def _manual_monitors_path() -> Path:
    cfg = load_notify_config()
    configured = str(cfg.get("manual_monitors_path") or "").strip()
    return Path(configured).expanduser() if configured else MANUAL_MONITORS_FILE


def load_manual_monitors(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load explicit stop/roll monitors for specific open trades.

    The file may be either a list or an object with a "monitors" list.
    Missing config is fine; the generic monitor still runs.
    """
    path = path or _manual_monitors_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _safe_print(f"  [manual-monitor] failed to read {path}: {exc}")
        return []
    monitors = payload.get("monitors", payload) if isinstance(payload, dict) else payload
    if not isinstance(monitors, list):
        return []
    return [m for m in monitors if isinstance(m, dict) and m.get("enabled", True)]


def _right_suffix(right: str) -> str:
    return "P" if str(right).upper().startswith("P") else "C"


def _same_number(left: Any, right: Any, tolerance: float = 0.01) -> bool:
    if left is None or right is None:
        return False
    return abs(safe(left) - safe(right)) <= tolerance


def _format_price(value: Any) -> str:
    val = safe(value, None)
    if val is None:
        return "n/a"
    return f"${val:.2f}"


def _symbol_underlying(symbol: Any) -> str:
    parts = str(symbol or "").split()
    return parts[0] if parts else ""


def _manual_key(monitor: Dict[str, Any]) -> str:
    raw = str(monitor.get("id") or monitor.get("label") or monitor.get("ticker") or "monitor")
    cleaned = "".join(ch if ch.isalnum() or ch in "._:-" else "-" for ch in raw.strip())
    return f"MANUAL:{cleaned or 'monitor'}"


def _payload_from_review_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a spread/single position into common monitor fields."""
    if item["kind"] == "SPREAD":
        group = item["group"]
        metrics = compute_spread_metrics(group)
        right = str(metrics.get("put_call") or "").upper()
        suffix = _right_suffix(right)
        short_strike = metrics.get("short_strike")
        long_strike = metrics.get("long_strike")
        if metrics.get("net_type") == "credit":
            legs = f"Sell ${safe(short_strike):g}{suffix} / Buy ${safe(long_strike):g}{suffix}"
        else:
            legs = f"Buy ${safe(long_strike):g}{suffix} / Sell ${safe(short_strike):g}{suffix}"
        return {
            "position_key": item["key"],
            "kind": "SPREAD",
            "underlying": metrics.get("underlying"),
            "strategy": metrics.get("strategy"),
            "net_type": metrics.get("net_type"),
            "expiry": metrics.get("expiry"),
            "put_call": right,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "strike": short_strike,
            "legs": f"{legs} {metrics.get('expiry')}",
            "spot": metrics.get("underlying_price"),
            "dte": metrics.get("dte"),
            "close_debit": metrics.get("current_exit_net"),
            "short_delta": abs(safe(metrics.get("short_delta"), 0.0)),
            "pnl": metrics.get("unrealized_pnl"),
            "pct_max": metrics.get("pct_of_max_profit"),
            "entry_net": metrics.get("entry_net_per_contract"),
        }

    pos = item["position"]
    qty = safe(pos.get("qty"), 0.0)
    right = str(pos.get("put_call") or "").upper()
    suffix = _right_suffix(right)
    strike = pos.get("strike")
    side = "Sell" if qty < 0 else "Buy"
    quote = pos.get("live_quote") or {}
    greeks = pos.get("greeks") or {}
    computed = pos.get("computed") or {}
    return {
        "position_key": item["key"],
        "kind": "POSITION",
        "underlying": pos.get("underlying") or _symbol_underlying(pos.get("symbol")),
        "strategy": f"{'Short' if qty < 0 else 'Long'} {right.title()}",
        "net_type": "credit" if qty < 0 else "debit",
        "expiry": pos.get("expiry"),
        "put_call": right,
        "short_strike": strike if qty < 0 else None,
        "long_strike": strike if qty > 0 else None,
        "strike": strike,
        "legs": f"{side} {abs(qty):g} ${safe(strike):g}{suffix} {pos.get('expiry')}",
        "spot": (pos.get("underlying_quote") or {}).get("last"),
        "dte": computed.get("dte"),
        "close_debit": quote.get("ask") if qty < 0 else None,
        "short_delta": abs(safe(greeks.get("delta"), 0.0)) if qty < 0 else None,
        "pnl": computed.get("unrealized_pnl"),
        "pct_max": computed.get("pct_of_max_profit"),
        "entry_net": pos.get("avg_cost"),
    }


def _monitor_matches_payload(monitor: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    ticker = str(monitor.get("ticker") or monitor.get("underlying") or "").upper().strip()
    if ticker and ticker != str(payload.get("underlying") or "").upper().strip():
        return False

    for field in ("expiry", "put_call", "strategy", "position_key"):
        expected = monitor.get(field)
        if expected is not None and str(expected).upper() != str(payload.get(field) or "").upper():
            return False

    for field in ("short_strike", "long_strike", "strike"):
        if field in monitor and not _same_number(monitor.get(field), payload.get(field)):
            return False

    return True


def _risk_direction(monitor: Dict[str, Any], payload: Dict[str, Any]) -> str:
    configured = str(monitor.get("risk_direction") or "").lower().strip()
    if configured in {"down", "below", "<=", "put"}:
        return "down"
    if configured in {"up", "above", ">=", "call"}:
        return "up"
    right = str(payload.get("put_call") or "").upper()
    strategy = str(payload.get("strategy") or "").lower()
    if "bear call" in strategy or (right == "CALL" and payload.get("net_type") == "credit"):
        return "up"
    return "down"


def _spot_crossed(spot: Any, line: Any, direction: str) -> bool:
    spot_val = safe(spot, None)
    line_val = safe(line, None)
    if spot_val is None or line_val is None:
        return False
    return spot_val <= line_val if direction == "down" else spot_val >= line_val


def _metric_at_or_above(value: Any, line: Any) -> bool:
    val = safe(value, None)
    threshold = safe(line, None)
    return val is not None and threshold is not None and val >= threshold


def _metric_at_or_below(value: Any, line: Any) -> bool:
    val = safe(value, None)
    threshold = safe(line, None)
    return val is not None and threshold is not None and val <= threshold


def _manual_monitor_alert_for_payload(monitor: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    label = str(monitor.get("label") or payload.get("strategy") or payload.get("underlying") or "manual monitor")
    action_text = str(monitor.get("action") or "Close or roll the whole position as one order; do not leg out.")
    direction = _risk_direction(monitor, payload)
    triggers: List[str] = []
    severity = "HOLD"

    if "critical_spot" in monitor and _spot_crossed(payload.get("spot"), monitor.get("critical_spot"), direction):
        severity = "CRITICAL"
        op = "<=" if direction == "down" else ">="
        triggers.append(f"spot {_format_price(payload.get('spot'))} {op} {_format_price(monitor.get('critical_spot'))}")
    elif "warning_spot" in monitor and _spot_crossed(payload.get("spot"), monitor.get("warning_spot"), direction):
        severity = "WARNING"
        op = "<=" if direction == "down" else ">="
        triggers.append(f"spot {_format_price(payload.get('spot'))} {op} {_format_price(monitor.get('warning_spot'))}")

    if _metric_at_or_above(payload.get("close_debit"), monitor.get("critical_close_debit")):
        severity = "CRITICAL"
        triggers.append(
            f"debit to close {_format_price(payload.get('close_debit'))} >= {_format_price(monitor.get('critical_close_debit'))}"
        )
    elif severity != "CRITICAL" and _metric_at_or_above(payload.get("close_debit"), monitor.get("warning_close_debit")):
        severity = "WARNING"
        triggers.append(
            f"debit to close {_format_price(payload.get('close_debit'))} >= {_format_price(monitor.get('warning_close_debit'))}"
        )

    if _metric_at_or_above(payload.get("short_delta"), monitor.get("critical_short_delta")):
        severity = "CRITICAL"
        triggers.append(f"short delta {safe(payload.get('short_delta')):.2f} >= {safe(monitor.get('critical_short_delta')):.2f}")
    elif severity != "CRITICAL" and _metric_at_or_above(payload.get("short_delta"), monitor.get("warning_short_delta")):
        severity = "WARNING"
        triggers.append(f"short delta {safe(payload.get('short_delta')):.2f} >= {safe(monitor.get('warning_short_delta')):.2f}")

    if _metric_at_or_below(payload.get("dte"), monitor.get("critical_dte")):
        severity = "CRITICAL"
        triggers.append(f"DTE {safe(payload.get('dte')):.0f} <= {safe(monitor.get('critical_dte')):.0f}")
    elif severity != "CRITICAL" and _metric_at_or_below(payload.get("dte"), monitor.get("warning_dte")):
        severity = "WARNING"
        triggers.append(f"DTE {safe(payload.get('dte')):.0f} <= {safe(monitor.get('warning_dte')):.0f}")

    profit_hit = bool(monitor.get("alert_on_profit")) and _metric_at_or_below(
        payload.get("close_debit"),
        monitor.get("profit_close_debit"),
    )
    if profit_hit and severity == "HOLD":
        severity = "PROFIT"
        triggers.append(
            f"profit target: debit to close {_format_price(payload.get('close_debit'))} <= {_format_price(monitor.get('profit_close_debit'))}"
        )

    verdict = "HOLD"
    critical = False
    if severity == "CRITICAL":
        verdict = str(monitor.get("critical_verdict") or "CLOSE").upper()
        if verdict not in {"CLOSE", "ROLL", "ASSESS"}:
            verdict = "CLOSE"
        critical = True
    elif severity == "WARNING":
        verdict = str(monitor.get("warning_verdict") or "ASSESS").upper()
        if verdict not in {"CLOSE", "ROLL", "ASSESS"}:
            verdict = "ASSESS"
    elif severity == "PROFIT":
        verdict = str(monitor.get("profit_verdict") or "CLOSE").upper()
        if verdict not in {"CLOSE", "ROLL", "ASSESS"}:
            verdict = "CLOSE"

    reason = "thresholds clear"
    if triggers:
        reason = f"{label}: {'; '.join(triggers)}. {action_text}"

    state = {
        "verdict": verdict,
        "reason": reason,
        "category": payload.get("strategy", ""),
        "pct_max": safe(payload.get("pct_max")),
        "pnl": safe(payload.get("pnl")),
        "dte": safe(payload.get("dte"), -1),
        "ul_price": safe(payload.get("spot")),
        "close_debit": safe(payload.get("close_debit"), None),
        "short_delta": safe(payload.get("short_delta"), None),
        "underlying": payload.get("underlying"),
        "timestamp": dt.datetime.now().isoformat(),
    }

    if verdict == "HOLD":
        return state, None

    alert = {
        "symbol": _manual_key(monitor),
        "underlying": payload.get("underlying"),
        "transition": "",
        "verdict": verdict,
        "reason": reason,
        "category": payload.get("strategy", ""),
        "pct_max": safe(payload.get("pct_max")),
        "pnl": safe(payload.get("pnl")),
        "dte": safe(payload.get("dte"), -1),
        "ul_price": safe(payload.get("spot")),
        "close_debit": safe(payload.get("close_debit"), None),
        "short_delta": safe(payload.get("short_delta"), None),
        "legs": payload.get("legs"),
        "critical": critical,
        "manual_monitor": True,
    }
    return state, alert


def evaluate_manual_monitors(review_items: List[Dict[str, Any]], prev_state: Dict[str, Dict]) -> Tuple[List[Dict], Dict[str, Dict]]:
    monitors = load_manual_monitors()
    if not monitors:
        return [], {}

    payloads = [_payload_from_review_item(item) for item in review_items]
    alerts: List[Dict] = []
    state_updates: Dict[str, Dict] = {}

    for monitor in monitors:
        key = _manual_key(monitor)
        payload = next((p for p in payloads if _monitor_matches_payload(monitor, p)), None)
        if not payload:
            state_updates[key] = {
                "verdict": "CLOSED",
                "reason": "Configured manual monitor no longer matches an open Schwab position.",
                "category": "manual-monitor",
                "pct_max": 0,
                "pnl": 0,
                "dte": 0,
                "ul_price": 0,
                "underlying": monitor.get("ticker") or monitor.get("underlying") or key,
                "timestamp": dt.datetime.now().isoformat(),
            }
            continue

        state, alert = _manual_monitor_alert_for_payload(monitor, payload)
        state_updates[key] = state
        if not alert:
            continue

        prev = prev_state.get(key, {})
        prev_verdict = prev.get("verdict", "NEW")
        alert["transition"] = f"{prev_verdict} -> {alert['verdict']}"

        worsened_debit = False
        close_debit = safe(alert.get("close_debit"), None)
        prev_debit = safe(prev.get("close_debit"), None)
        if close_debit is not None and prev_debit is not None:
            worsened_debit = close_debit >= prev_debit + safe(monitor.get("realert_debit_step"), 0.75)

        if prev_verdict != alert["verdict"] or (alert.get("critical") and worsened_debit):
            alerts.append(alert)

    return alerts, state_updates


def manual_suppressed_position_keys(review_items: List[Dict[str, Any]]) -> set[str]:
    """Return spread/leg keys whose generic alerts are replaced by manual monitors."""
    monitors = load_manual_monitors()
    if not monitors:
        return set()

    suppressed = set()
    for item in review_items:
        payload = _payload_from_review_item(item)
        for monitor in monitors:
            if not monitor.get("suppress_generic_alerts", True):
                continue
            if _monitor_matches_payload(monitor, payload):
                key = payload.get("position_key")
                if key:
                    suppressed.add(str(key))
                break
    return suppressed


PROFIT_ALERT_PHRASES = (
    "max profit",
    "nothing left to harvest",
    "of max profit harvested",
    "profit harvested",
    "past 75% target",
    "diminishing returns",
    "take profit",
    "take +",
    "consider trimming",
    "protect gains",
    "strong, trail stop",
    "gives back gains",
)

NOISE_ALERT_PHRASES = (
    "expiration week with limited profit captured",
)


def is_risk_management_alert(alert: Dict[str, Any]) -> bool:
    """True only for alerts that mean close/roll risk is developing.

    Position monitor notifications are for failure prevention, not profit
    harvesting. Trade ideas have their own notification path.
    """
    if alert.get("manual_monitor"):
        return alert.get("verdict") in {"ASSESS", "CLOSE", "ROLL"}
    if alert.get("verdict") == "CLOSED":
        return False

    reason = str(alert.get("reason") or "").lower()
    if any(phrase in reason for phrase in PROFIT_ALERT_PHRASES):
        return False
    if any(phrase in reason for phrase in NOISE_ALERT_PHRASES):
        return False
    return alert.get("verdict") in {"ASSESS", "CLOSE", "ROLL"}


def _is_profit_taking_reason(reason: str) -> bool:
    text = str(reason or "").lower()
    return any(phrase in text for phrase in PROFIT_ALERT_PHRASES)


def _suppress_profit_taking_verdict(verdict: str, reason: str) -> Tuple[str, str, bool]:
    if verdict in {"ASSESS", "CLOSE", "ROLL"} and _is_profit_taking_reason(reason):
        return "HOLD", "profit-taking alert suppressed; monitoring risk/failure only", True
    return verdict, reason, False


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

    review_items = build_position_review_items(positions)
    active_leg_keys = current_leg_keys(positions)

    # Compute verdicts
    prev_state = load_state()
    new_state = {}
    alerts = []
    suppressed_generic_keys = manual_suppressed_position_keys(review_items)

    for item in review_items:
        if item["kind"] == "SPREAD":
            key = item["key"]
            verdict, reason, metrics = compute_spread_verdict(item["group"])
            category = metrics["strategy"]
            pct_max = safe(metrics.get("pct_of_max_profit"))
            pnl = safe(metrics.get("unrealized_pnl"))
            dte = safe(metrics.get("dte"), -1)
            ul_price = safe(metrics.get("underlying_price"))
            underlying = metrics.get("underlying", key)
        else:
            pos = item["position"]
            key = position_key(pos)
            verdict, reason = compute_verdict(pos)
            category = classify_position(pos)
            pct_max = safe(pos["computed"].get("pct_of_max_profit"))
            pnl = safe(pos["computed"].get("unrealized_pnl"))
            dte = safe(pos["computed"].get("dte"), -1)
            ul_price = safe((pos.get("underlying_quote") or {}).get("last"))
            underlying = pos.get("underlying", "") or pos.get("symbol", "")

        verdict, reason, profit_suppressed = _suppress_profit_taking_verdict(verdict, reason)

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

        # Allow de-escalation only if P&L improved by $300+
        if is_deescalation and not profit_suppressed:
            pnl_improvement = pnl - prev_pnl
            if pnl_improvement < 300:
                verdict = prev_verdict
                reason = prev.get("reason", reason) + " (sticky)"

        verdict, reason, _ = _suppress_profit_taking_verdict(verdict, reason)

        # Update state with the FINAL verdict (after hysteresis), not the raw one
        new_state[key]["verdict"] = verdict
        new_state[key]["reason"] = reason

        # Re-alert if worsening within same ASSESS/CLOSE verdict by $500+
        worsened = (verdict in ("ASSESS", "CLOSE") and
                    prev_verdict == verdict and
                    pnl < prev_pnl - 500)

        if key not in suppressed_generic_keys and (prev_verdict != verdict or worsened):
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
            if is_risk_management_alert(alert):
                alerts.append(alert)

    manual_alerts, manual_state = evaluate_manual_monitors(review_items, prev_state)
    new_state.update(manual_state)
    alerts.extend(manual_alerts)

    # Detect closed positions (in prev but not in new)
    for key, prev in prev_state.items():
        if str(key).startswith("MANUAL:"):
            continue
        # If a previous single leg is now represented by a spread-level state
        # row, do not emit a false "closed" alert for that still-open leg.
        if key in active_leg_keys:
            continue
        if key not in new_state and prev.get("verdict") != "CLOSED":
            alert = {
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
            }
            if is_risk_management_alert(alert):
                alerts.append(alert)

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
    if alert.get("close_debit") is not None:
        parts.append(f"Close debit: ${alert['close_debit']:.2f}")
    if alert.get("short_delta") is not None:
        parts.append(f"Short delta: {alert['short_delta']:.2f}")
    if alert.get("legs"):
        parts.append(f"Legs: {alert['legs']}")

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


def is_after_hours_watch_window() -> bool:
    """True from 4:00 PM to 8:00 PM ET on weekdays."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    after_open = now.replace(hour=16, minute=0, second=0)
    after_close = now.replace(hour=20, minute=0, second=0)
    return after_open < now <= after_close


def _state_underlying(key: str, value: Any) -> Optional[str]:
    if isinstance(value, dict) and value.get("underlying"):
        return str(value["underlying"]).strip().upper()
    text = str(key).strip()
    if text.startswith("SPREAD:"):
        parts = text.split(":")
        if len(parts) > 1 and parts[1]:
            return parts[1].strip().upper()
    parts = text.split()
    if parts:
        return parts[0].upper()
    return None


def _watched_underlyings() -> List[str]:
    tickers = {"SPY", "QQQ"}
    try:
        for key, value in load_state().items():
            underlying = _state_underlying(key, value)
            if underlying:
                tickers.add(underlying)
    except Exception:
        pass
    try:
        if IDEAS_STATE_FILE.exists():
            ideas_state = json.loads(IDEAS_STATE_FILE.read_text(encoding="utf-8"))
            tickers.update(str(t).strip().upper() for t in ideas_state if str(t).strip())
    except Exception:
        pass
    max_watch = _env_int("AFTER_HOURS_MAX_WATCH", 60)
    return sorted(tickers)[:max_watch]


def _quote_move_pct(qdata: Dict[str, Any]) -> float:
    q = qdata.get("quote", qdata)
    candidates = []
    for field in (
        "postMarketPercentChange",
        "extendedMarketPercentChange",
        "markPercentChange",
        "netPercentChange",
        "regularMarketPercentChange",
    ):
        if field in q:
            candidates.append(safe(q.get(field)))

    last = safe(q.get("lastPrice") or q.get("mark") or q.get("markPrice"))
    close = safe(q.get("closePrice") or q.get("regularMarketLastPrice"))
    if last > 0 and close > 0:
        candidates.append((last - close) / close * 100)

    if not candidates:
        return 0.0
    return max(candidates, key=lambda x: abs(x))


def _after_hours_movement_from_quotes(quotes: Dict[str, Any],
                                      watched: List[str]) -> Tuple[bool, str]:
    market_threshold = _env_float("AFTER_HOURS_MARKET_MOVE_PCT", 0.6)
    watch_threshold = _env_float("AFTER_HOURS_WATCH_MOVE_PCT", 2.0)
    market_movers = []
    watch_movers = []
    all_moves = []

    for ticker, payload in quotes.items():
        move = _quote_move_pct(payload)
        if abs(move) <= 0:
            continue
        all_moves.append((ticker, move))
        threshold = market_threshold if ticker in {"SPY", "QQQ"} else watch_threshold
        if abs(move) >= threshold:
            if ticker in {"SPY", "QQQ"}:
                market_movers.append((ticker, move))
            elif ticker in watched:
                watch_movers.append((ticker, move))

    movers = market_movers + sorted(watch_movers, key=lambda x: abs(x[1]), reverse=True)[:5]
    if movers:
        summary = ", ".join(f"{ticker} {move:+.1f}%" for ticker, move in movers)
        return True, f"after-hours movement gate passed: {summary}"

    if all_moves:
        ticker, move = max(all_moves, key=lambda x: abs(x[1]))
        return False, f"after-hours quiet: max watched move {ticker} {move:+.1f}%"
    return False, "after-hours quiet: no quote movement available"


def has_after_hours_movement() -> Tuple[bool, str]:
    """Check whether an after-hours scan is justified."""
    if not is_after_hours_watch_window():
        return False, "outside after-hours watch window"
    watched = _watched_underlyings()
    try:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
        config = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(config=config, interactive_login=False)
        quotes = {}
        for i in range(0, len(watched), 100):
            quotes.update(svc.get_quotes(watched[i:i + 100]))
    except Exception as exc:
        return False, f"after-hours movement check failed: {str(exc)[:160]}"
    return _after_hours_movement_from_quotes(quotes, watched)


def should_run_monitor_now(force: bool = False, manual: bool = False) -> Tuple[bool, str]:
    if force:
        return True, "force run"
    if manual:
        return True, "manual run"
    if is_market_hours():
        return True, "regular market hours"
    return has_after_hours_movement()


def run_trade_ideas_scan() -> List[Dict]:
    """Run unified trade ideas scanner and return alerts for new ideas."""
    from uwos.trade_ideas import scan_trade_ideas, format_results_md, format_alert, find_latest_data_dir
    from uwos.eod_trade_scan_mode_a import compute_macro_regime

    data_dir = find_latest_data_dir()

    excluded_underlyings = set()
    for key, state in load_state().items():
        underlying = _state_underlying(key, state)
        if underlying:
            excluded_underlyings.add(underlying)

    results = scan_trade_ideas(
        data_dir=data_dir,
        top_n=8,
        exclude_tickers=excluded_underlyings,
        verbose=False,
    )
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
            "setup_lane": r.get("setup_lane", ""),
            "strategy": r["strategy"],
            "composite": r["composite"],
            "short_strike": r["short_strike"],
            "long_strike": r.get("long_strike"),
            "expiry": r["expiry"],
            "timestamp": dt.datetime.now().isoformat(),
        }

        # Alert if this is a new ticker or the trade changed
        prev_entry = prev.get(ticker, {})
        if (prev_entry.get("setup_lane") != r.get("setup_lane", "") or
                prev_entry.get("strategy") != r["strategy"] or
                prev_entry.get("short_strike") != r["short_strike"] or
                prev_entry.get("long_strike") != r.get("long_strike") or
                prev_entry.get("expiry") != r["expiry"]):
            alerts.append(r)

    IDEAS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    IDEAS_STATE_FILE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return alerts


def should_run_ideas_scan(after_hours_moving: bool = False) -> bool:
    """Run trade ideas scan during market hours, plus moving after-hours windows."""
    if after_hours_moving:
        return True
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
    should_run, run_reason = should_run_monitor_now(force=force, manual=manual)
    if not should_run:
        _safe_print(f"  [{dt.datetime.now():%H:%M}] Market closed, skipping scan ({run_reason})")
        return 0

    _safe_print(f"  [{dt.datetime.now():%H:%M:%S}] Running position scan ({run_reason})...")
    total_alerts = 0

    # Heartbeat: send once at market open (9:30-10:00 ET)
    if _is_market_open_window():
        state = load_state()
        close_count = sum(
            1
            for v in state.values()
            if v.get("verdict") == "CLOSE" and is_risk_management_alert(v)
        )
        assess_count = sum(
            1
            for v in state.values()
            if v.get("verdict") == "ASSESS" and is_risk_management_alert(v)
        )
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
        if _is_schwab_credential_failure(e):
            notify("AUTH EXPIRED",
                   "Schwab token expired. In Codex, run: renew Schwab token. Refresh local auth, sync the token to the GCP VM, and retest trade-monitor auth.",
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

    alerts = [alert for alert in alerts if is_risk_management_alert(alert)]

    for alert in alerts:
        title, body = format_alert(alert)
        priority = "urgent" if alert.get("critical") else "default"
        tags = "rotating_light" if alert.get("critical") else "chart_with_upwards_trend"
        _safe_print(f"    {title}: {body}")
        notify(
            title,
            body,
            priority=priority,
            tags=tags,
            critical=bool(alert.get("critical")),
            manual=manual or bool(alert.get("manual_monitor")),
        )
    total_alerts += len(alerts)

    # Mode 2: Trade ideas scanner during market hours, plus moving after-hours windows.
    after_hours_moving = run_reason.startswith("after-hours movement gate passed")
    if force or manual or should_run_ideas_scan(after_hours_moving=after_hours_moving):
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
                        "setup_lane": val.get("setup_lane", ""),
                        "strategy": val.get("strategy", "?"),
                        "short_strike": val.get("short_strike", 0),
                        "long_strike": val.get("long_strike", 0),
                        "expiry": val.get("expiry", ""),
                        "composite": val.get("composite", 0),
                    })

            for r in idea_alerts:
                lane = r.get("setup_lane", "")
                lane_text = f" {lane}" if lane else ""
                title = f"NEW TRADE: {r['ticker']}{lane_text} {r.get('strategy', '?')}"
                try:
                    body = fmt_idea(r)
                except Exception:
                    strategy = r.get("strategy", "?")
                    short_strike = r.get("short_strike", 0)
                    long_strike = r.get("long_strike", 0)
                    option_suffix = "P" if "Put" in strategy else "C"
                    if "Debit" in strategy:
                        legs = f"Buy ${long_strike:.0f}{option_suffix} / Sell ${short_strike:.0f}{option_suffix}"
                    else:
                        legs = f"Sell ${short_strike:.0f}{option_suffix} / Buy ${long_strike:.0f}{option_suffix}"
                    body = (
                        f"[{r.get('setup_lane') or 'IDEA'}] {strategy}: "
                        f"{legs} {r.get('expiry', '')} | Score: {r.get('composite',0):.0f}"
                    )
                _safe_print(f"    IDEA: {title}: {body}")
                notify(title, body, priority="high", tags="chart_with_upwards_trend", manual=manual)
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
    parser.add_argument("--phone-test", action="store_true",
                        help="Send an urgent test through phone channels and exit")
    parser.add_argument("--manual-test", action="store_true",
                        help="Send a distinct manual-monitor phone alert test and exit")
    parser.add_argument("--force", action="store_true",
                        help="Run even outside market hours")
    parser.add_argument("--manual", action="store_true",
                        help="Manual run — notify ALL current verdicts (not just transitions)")
    args = parser.parse_args()

    if args.test or args.phone_test or args.manual_test:
        print("Sending test notification...")
        critical = bool(args.phone_test or args.manual_test)
        manual_test = bool(args.manual_test)
        notify(
            (
                "TEST ONLY - Manual Monitor"
                if manual_test else
                "TEST ONLY - Trade Monitor" if not critical else "TEST ONLY - Phone Alert"
            ),
            (
                "Notification path OK. Example setup format: "
                "[BREAKOUT] Bull Put Credit: Sell $460P / Buy $450P 2026-06-18 | "
                "Cr $3.40 | MaxP $340 | Prob 66% | Tech 8/10 | TEST ONLY, do not trade."
            ),
            priority="default" if not critical else "urgent",
            tags="white_check_mark" if not critical else "rotating_light",
            critical=critical,
            manual=manual_test,
        )
        if manual_test:
            print("Done. Check your manual-monitor ntfy topic on the phone.")
        else:
            print("Done. Check your ntfy app and phone channel." if critical else "Done. Check your ntfy app.")
        return

    if args.loop > 0:
        print(f"Trade Monitor starting — scanning every {args.loop} min during market hours")
        cfg = load_notify_config()
        print(f"  ntfy topic: {_masked_secret_status(cfg['ntfy_topic'])}")
        print(f"  phone ntfy topic: {_masked_secret_status(cfg['ntfy_phone_topic'])}")
        print(f"  manual ntfy topic: {_masked_secret_status(cfg['ntfy_manual_topic'])}")
        print(f"  phone mode: {cfg['phone_notify_mode'] or 'unset'}")
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
