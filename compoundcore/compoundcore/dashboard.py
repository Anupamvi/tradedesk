"""Local persistent dashboard. Both sleeves + live book. No orders."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from compoundcore.allocate import distribution
from compoundcore.book import (
    book_view,
    default_state_path,
    load_state,
    mark_with_prices,
    now_iso,
    parse_money,
    save_state,
)
from compoundcore import quotes as quotes_mod
from compoundcore.projections import path_table
from compoundcore.sleeve import ASOF, SLEEVE_NAMES, TICKER_ORDER, public_snapshot


WEB = Path(__file__).resolve().parent.parent / "web"
DASHBOARD_HTML = WEB / "dashboard.html"
CALCULATOR_HTML = WEB / "calculator.html"

_LOCK = threading.Lock()


def plan_payload(amount: float, weekly: float, monthly: float) -> Dict[str, Any]:
    return {
        "asof": ASOF,
        "amount": amount,
        "weekly": weekly,
        "monthly": monthly,
        "sleeves": {
            name: {
                "allocation": distribution(amount, name, weekly),
                "projections": path_table(amount, monthly, name),
            }
            for name in SLEEVE_NAMES
        },
    }


def dashboard_payload(path: Path) -> Dict[str, Any]:
    state = load_state(path)
    planner = state["planner"]
    book = state["book"]
    return {
        "asof": ASOF,
        "tickers": list(TICKER_ORDER),
        "snapshot": public_snapshot(),
        "planner": plan_payload(planner["amount"], planner["weekly"], planner["monthly"]),
        "book": book_view(
            holdings=book.get("holdings"),
            monthly_add=book["monthly_add"],
            compare_to=book["compare_to"],
            submitted_at=book.get("submitted_at"),
            positions=book.get("positions"),
            marked_at=book.get("marked_at"),
        ),
        "saved": {
            "planner": planner,
            "book": {
                "positions": book["positions"],
                "holdings": {t: book["positions"][t]["cost"] for t in TICKER_ORDER},
                "monthly_add": book["monthly_add"],
                "compare_to": book["compare_to"],
                "submitted_at": book.get("submitted_at"),
                "marked_at": book.get("marked_at"),
            },
        },
    }


def _apply_planner(state: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    planner = dict(state["planner"])
    if "amount" in body:
        planner["amount"] = parse_money(body.get("amount"), "amount")
    if "weekly" in body:
        planner["weekly"] = parse_money(body.get("weekly"), "weekly")
    if "monthly" in body:
        planner["monthly"] = parse_money(body.get("monthly"), "monthly")
    state["planner"] = planner
    return state


def _positions_from_body(body: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    nested = body.get("positions")
    if isinstance(nested, dict):
        return nested
    holdings = body.get("holdings") if isinstance(body.get("holdings"), dict) else body
    current = body.get("current") if isinstance(body.get("current"), dict) else {}
    shares = body.get("shares") if isinstance(body.get("shares"), dict) else {}
    out = {}
    for ticker in TICKER_ORDER:
        prev = existing.get(ticker) if isinstance(existing.get(ticker), dict) else {}
        out[ticker] = {
            "cost": holdings.get(ticker, prev.get("cost", 0)),
            "current": current.get(ticker, prev.get("current", 0)),
            "shares": shares.get(ticker, prev.get("shares", 0)),
        }
    return out


def _apply_book(state: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    book = dict(state["book"])
    book["positions"] = _positions_from_body(body, book.get("positions") or {})
    if "monthly_add" in body:
        book["monthly_add"] = parse_money(body.get("monthly_add"), "monthly_add")
    if "compare_to" in body:
        book["compare_to"] = body.get("compare_to") or "default"
    stamp = now_iso()
    book["submitted_at"] = stamp
    if any(float((book["positions"].get(t) or {}).get("current") or 0) > 0 for t in TICKER_ORDER):
        book["marked_at"] = stamp
    state["book"] = book
    return state


def _apply_refresh(state: Dict[str, Any], prices: Dict[str, float]) -> Dict[str, Any]:
    book = dict(state["book"])
    book["positions"] = mark_with_prices(book.get("positions") or {}, prices)
    book["marked_at"] = now_iso()
    state["book"] = book
    return state


class DashboardHandler(BaseHTTPRequestHandler):
    state_path: Path

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/dashboard", "/dashboard.html"):
            self._send_file(DASHBOARD_HTML, "text/html; charset=utf-8")
            return
        if path in ("/calculator.html", "/calculator"):
            self._send_file(CALCULATOR_HTML, "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(200, dashboard_payload(self.state_path))
            return
        if path == "/api/plan":
            qs = parse_qs(parsed.query)
            try:
                amount = parse_money((qs.get("amount") or ["0"])[0], "amount")
                weekly = parse_money((qs.get("weekly") or ["0"])[0], "weekly")
                monthly = parse_money((qs.get("monthly") or ["0"])[0], "monthly")
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, plan_payload(amount, weekly, monthly))
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        try:
            with _LOCK:
                state = load_state(self.state_path)
                if parsed.path == "/api/planner":
                    state = _apply_planner(state, body)
                    save_state(state, self.state_path)
                    self._send_json(200, dashboard_payload(self.state_path))
                    return
                if parsed.path == "/api/book":
                    state = _apply_book(state, body)
                    save_state(state, self.state_path)
                    self._send_json(200, dashboard_payload(self.state_path))
                    return
                if parsed.path == "/api/book/refresh":
                    prices = quotes_mod.last_prices(TICKER_ORDER)
                    if not prices:
                        self._send_json(503, {"error": "DATA UNAVAILABLE: no live quotes"})
                        return
                    state = _apply_refresh(state, prices)
                    save_state(state, self.state_path)
                    payload = dashboard_payload(self.state_path)
                    payload["book"]["quote_status"] = "schwab"
                    self._send_json(200, payload)
                    return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self.send_error(404, "Not found")

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("body must be JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404, "Not found")
            return
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def make_server(host: str, port: int, state_path: Optional[Path] = None) -> ThreadingHTTPServer:
    dest = Path(state_path) if state_path is not None else default_state_path()

    class Bound(DashboardHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.state_path = dest
            super().__init__(*args, **kwargs)

    httpd = ThreadingHTTPServer((host, port), Bound)
    httpd.allow_reuse_address = True
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8765, state_path: Optional[str] = None) -> int:
    dest = Path(state_path) if state_path else default_state_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    httpd = make_server(host, port, dest)
    print("Compound Core dashboard  http://%s:%s/" % (host, port), flush=True)
    print("Raw calculator           http://%s:%s/calculator.html" % (host, port), flush=True)
    print("State                    %s" % dest, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
