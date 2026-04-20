#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import websocket


UW_BASE_URL = "https://unusualwhales.com"
COPY_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls", ".tsv"}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_time(value: str) -> dt.datetime:
    raw = str(value or "").strip()
    if not raw:
        return dt.datetime.now(dt.timezone.utc)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def clean_ticker(value: object) -> str:
    ticker = str(value or "").strip().upper()
    ticker = ticker.replace("$", "").replace(".", "").replace("/", "")
    return ticker


def unique_tickers(values: Iterable[object]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        ticker = clean_ticker(value)
        if not ticker or ticker in seen or ticker == "NAN":
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _cdp_json(remote_debugging_url: str, path: str, timeout_sec: float = 2.0) -> object:
    base = str(remote_debugging_url or "").rstrip("/")
    with urllib.request.urlopen(base + path, timeout=float(timeout_sec)) as handle:
        return json.loads(handle.read().decode("utf-8", "replace"))


def cdp_is_available(remote_debugging_url: str) -> bool:
    try:
        _cdp_json(remote_debugging_url, "/json/version", timeout_sec=2.0)
        return True
    except Exception:
        return False


def _remote_port(remote_debugging_url: str) -> str:
    parsed = urllib.parse.urlparse(str(remote_debugging_url or ""))
    return str(parsed.port or 9222)


def _profile_dir_for_app(app_name: str) -> Path:
    slug = "atlas" if "atlas" in str(app_name).lower() else "chrome"
    return Path.home() / f".{slug}-uw-cdp"


def ensure_uw_cdp_browser(
    remote_debugging_url: str,
    browser_app: str = "ChatGPT Atlas",
    profile_dir: str = "",
    wait_sec: float = 8.0,
    start_url: str = UW_BASE_URL,
) -> Dict[str, object]:
    """Ensure a CDP browser exists and has a UW tab.

    Atlas currently accepts remote-debugging flags on some machines but may not
    actually open a listener.  We still try it first because it preserves the
    user's preferred authenticated path, then fall back to a dedicated Chrome
    profile that reliably exposes CDP and persists UW login cookies.
    """
    remote_debugging_url = str(remote_debugging_url or "http://127.0.0.1:9222").rstrip("/")
    attempted: List[Dict[str, object]] = []

    def ensure_page() -> bool:
        try:
            pages = _cdp_json(remote_debugging_url, "/json/list", timeout_sec=2.0)
        except Exception:
            return False
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                if page.get("type") == "page" and "unusualwhales.com" in str(page.get("url", "")):
                    return True
        try:
            urllib.request.urlopen(
                remote_debugging_url + "/json/new?" + urllib.parse.quote(start_url, safe=":/?&=%"),
                data=b"",
                timeout=3.0,
            ).read()
            return True
        except Exception:
            try:
                subprocess.Popen(["open", start_url])
            except Exception:
                pass
            return False

    if cdp_is_available(remote_debugging_url):
        page_ok = ensure_page()
        return {
            "ok": True,
            "browser_app": "existing",
            "remote_debugging_url": remote_debugging_url,
            "uw_page_ok": page_ok,
            "attempted": attempted,
        }

    apps: List[str] = []
    for app in [browser_app, "Google Chrome"]:
        app = str(app or "").strip()
        if app and app not in apps:
            apps.append(app)

    port = _remote_port(remote_debugging_url)
    for app in apps:
        prof = Path(profile_dir).expanduser() if profile_dir.strip() else _profile_dir_for_app(app)
        prof.mkdir(parents=True, exist_ok=True)
        cmd = [
            "open",
            "-na",
            app,
            "--args",
            f"--user-data-dir={prof}",
            f"--remote-debugging-port={port}",
            "--remote-debugging-address=127.0.0.1",
            "--remote-allow-origins=*",
            start_url,
        ]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            attempted.append({"browser_app": app, "profile_dir": str(prof), "ok": False, "error": str(exc)})
            continue
        deadline = time.time() + float(wait_sec)
        opened = False
        while time.time() < deadline:
            if cdp_is_available(remote_debugging_url):
                opened = True
                break
            time.sleep(0.5)
        attempted.append({"browser_app": app, "profile_dir": str(prof), "ok": opened})
        if opened:
            page_ok = ensure_page()
            return {
                "ok": True,
                "browser_app": app,
                "profile_dir": str(prof),
                "remote_debugging_url": remote_debugging_url,
                "uw_page_ok": page_ok,
                "attempted": attempted,
            }

    return {
        "ok": False,
        "remote_debugging_url": remote_debugging_url,
        "uw_page_ok": False,
        "attempted": attempted,
    }


def read_tickers_from_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        ticker_col = next((c for c in reader.fieldnames if str(c).strip().lower() == "ticker"), None)
        if not ticker_col:
            return []
        return unique_tickers(row.get(ticker_col) for row in reader)


def resolve_default_input(repo_root: Path, date_str: str) -> Path:
    live_final = repo_root / "out" / date_str / f"live_trade_table_{date_str}_final.csv"
    if live_final.exists():
        return live_final
    shortlist = repo_root / "out" / date_str / f"shortlist_trades_{date_str}_mode_a.csv"
    if shortlist.exists():
        return shortlist
    return live_final


def dashboard_urls(tickers: Sequence[str]) -> Dict[str, object]:
    ticker_urls = {}
    for ticker in tickers:
        ticker_urls[ticker] = [
            f"{UW_BASE_URL}/flow/tickers/{ticker}",
            f"{UW_BASE_URL}/flow/ticker/{ticker}",
            f"{UW_BASE_URL}/stock/{ticker}",
            f"{UW_BASE_URL}/stock/{ticker}/gex",
        ]
    return {
        "market_urls": [
            f"{UW_BASE_URL}/gex",
            f"{UW_BASE_URL}/flow/ticker/overview",
            f"{UW_BASE_URL}/flow",
        ],
        "ticker_urls": ticker_urls,
    }


def write_plan(out_dir: Path, date_str: str, tickers: Sequence[str], urls: Dict[str, object]) -> Path:
    path = out_dir / f"uw_dashboard_plan_{date_str}.md"
    lines = [
        f"# Unusual Whales enrichment plan for {date_str}",
        "",
        "Goal: collect UW-only context as explicit local files before running approval logic.",
        "",
        "Preferred browser flow:",
        "",
        "1. Open the market GEX dashboard and export/download GEX data if available.",
        "2. Open ticker flow/GEX pages for the tickers below and export/download CSV/XLSX where available.",
        "3. Run this collector again with `--mode ingest` to copy new Downloads files into this folder.",
        "",
        "## Market URLs",
        "",
    ]
    for url in urls.get("market_urls", []):
        lines.append(f"- {url}")
    lines.extend(["", "## Ticker URLs", ""])
    ticker_urls = urls.get("ticker_urls", {})
    if isinstance(ticker_urls, dict):
        for ticker in tickers:
            lines.append(f"### {ticker}")
            for url in ticker_urls.get(ticker, []):
                lines.append(f"- {url}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_manifest(out_dir: Path, date_str: str, payload: Dict[str, object]) -> Path:
    path = out_dir / f"uw_enrichment_manifest_{date_str}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(out_dir: Path, date_str: str) -> Dict[str, object]:
    path = out_dir / f"uw_enrichment_manifest_{date_str}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def open_urls(urls: Sequence[str], browser: str, max_urls: int) -> None:
    count = 0
    for url in urls:
        if count >= max_urls:
            break
        cmd = ["open", "-a", browser, url] if browser else ["open", url]
        subprocess.run(cmd, check=False)
        count += 1


def file_mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)


def looks_relevant(path: Path, tickers: Sequence[str]) -> bool:
    suffix = path.suffix.lower()
    if suffix not in COPY_EXTENSIONS:
        return False
    name = path.name.lower()
    keywords = ["unusual", "whale", "gex", "gamma", "flow", "option", "oi", "open_interest"]
    if any(k in name for k in keywords):
        return True
    ticker_set = {t.lower() for t in tickers}
    return any(t and t in name for t in ticker_set)


def safe_copy_name(path: Path, dest_dir: Path) -> Path:
    candidate = dest_dir / path.name
    if not candidate.exists():
        return candidate
    stem = path.stem
    suffix = path.suffix
    for i in range(2, 1000):
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique copy name for {path}")


def csv_header(path: Path) -> List[str]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            return next(reader, [])
    except Exception:
        return []


def ingest_downloads(
    downloads_dir: Path,
    out_dir: Path,
    tickers: Sequence[str],
    since_utc: dt.datetime,
) -> List[Dict[str, object]]:
    raw_dir = out_dir / "downloads_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    if not downloads_dir.exists():
        return copied
    for path in sorted(downloads_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        if not path.is_file():
            continue
        try:
            mtime = file_mtime_utc(path)
        except Exception:
            continue
        if mtime < since_utc:
            continue
        if not looks_relevant(path, tickers):
            continue
        dest = safe_copy_name(path, raw_dir)
        shutil.copy2(path, dest)
        copied.append(
            {
                "source_path": str(path),
                "copied_path": str(dest),
                "mtime_utc": mtime.replace(microsecond=0).isoformat(),
                "size_bytes": int(dest.stat().st_size),
                "header": csv_header(dest),
            }
        )
    return copied


def debugger_json(remote_debugging_url: str, path: str) -> object:
    base = str(remote_debugging_url).rstrip("/")
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as handle:
        return json.load(handle)


def find_uw_page_ws(remote_debugging_url: str) -> str:
    pages = debugger_json(remote_debugging_url, "/json")
    if not isinstance(pages, list):
        raise RuntimeError(f"Unexpected Chrome debugger /json payload: {type(pages)}")
    for page in pages:
        if page.get("type") == "page" and "unusualwhales.com" in str(page.get("url", "")):
            ws_url = page.get("webSocketDebuggerUrl")
            if ws_url:
                return str(ws_url)
    for page in pages:
        if page.get("type") == "page" and page.get("webSocketDebuggerUrl"):
            return str(page["webSocketDebuggerUrl"])
    raise RuntimeError("No controllable Chrome page found. Launch Chrome with --remote-debugging-port first.")


class CdpClient:
    def __init__(self, websocket_url: str):
        self.ws = websocket.create_connection(websocket_url, timeout=15, suppress_origin=True)
        self.next_id = 1

    def close(self) -> None:
        self.ws.close()

    def send(self, method: str, params: Dict[str, object] | None = None) -> int:
        call_id = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        return call_id

    def call(
        self,
        method: str,
        params: Dict[str, object] | None = None,
        timeout_sec: float = 15,
        response_collector=None,
    ) -> Dict[str, object]:
        call_id = self.send(method, params)
        end = time.time() + float(timeout_sec)
        while time.time() < end:
            msg = json.loads(self.ws.recv())
            if response_collector is not None:
                response_collector(msg)
            if msg.get("id") == call_id:
                return msg
        raise TimeoutError(f"Timed out waiting for CDP response to {method}")


def cdp_collect_ticker_gex(
    client: CdpClient,
    ticker: str,
    date_str: str,
    wait_sec: float,
) -> List[Dict[str, object]]:
    ticker = clean_ticker(ticker)
    captured: Dict[str, Dict[str, object]] = {}

    def collect_event(msg: Dict[str, object]) -> None:
        if msg.get("method") != "Network.responseReceived":
            return
        params = msg.get("params", {})
        if not isinstance(params, dict):
            return
        response = params.get("response", {})
        if not isinstance(response, dict):
            return
        url = str(response.get("url", ""))
        needle = f"greek_exposures/{ticker}/spot"
        if needle not in url:
            return
        request_id = str(params.get("requestId", ""))
        if not request_id:
            return
        captured[request_id] = {
            "request_id": request_id,
            "url": url,
            "status": response.get("status"),
            "mime_type": response.get("mimeType"),
        }

    page_url = f"{UW_BASE_URL}/stock/{ticker}/greek-exposure?type=exposure&greek=gamma&date={date_str}"
    client.call("Page.navigate", {"url": page_url}, timeout_sec=5, response_collector=collect_event)
    end = time.time() + float(wait_sec)
    while time.time() < end:
        try:
            msg = json.loads(client.ws.recv())
        except Exception:
            break
        collect_event(msg)

    out = []
    for request_id, item in list(captured.items()):
        status = item.get("status")
        body = ""
        base64_encoded = False
        if status and int(status) != 204:
            try:
                res = client.call("Network.getResponseBody", {"requestId": request_id}, timeout_sec=5)
                body_info = res.get("result", {})
                if isinstance(body_info, dict):
                    body = str(body_info.get("body", ""))
                    base64_encoded = bool(body_info.get("base64Encoded"))
                    if base64_encoded and body:
                        body = base64.b64decode(body).decode("utf-8", "replace")
            except Exception as exc:
                item["body_error"] = str(exc)
        item["body"] = body
        item["base64_encoded"] = base64_encoded
        out.append(item)
    if not out or not any(response_matches_requested_date(str(item.get("url", "")), date_str) for item in out):
        summary_path = f"/api/greek_exposures/{ticker}/spot?date={date_str}"
        strikes_path = f"/api/greek_exposures/{ticker}/spot/strikes?date={date_str}"
        paths_json = json.dumps([summary_path, strikes_path])
        expr = f"""
(async () => {{
  let req = null;
  if (!window.webpackChunk_N_E) {{
    throw new Error("UW webpack runtime unavailable");
  }}
  window.webpackChunk_N_E.push([[Math.floor(Math.random() * 1000000000)], {{}}, function(r) {{ req = r; }}]);
  if (!req) {{
    throw new Error("UW webpack require unavailable");
  }}
  const mod = req(64146);
  const api = mod && mod.mm;
  if (!api) {{
    throw new Error("UW API wrapper unavailable");
  }}
  const paths = {paths_json};
  const rows = [];
  for (const path of paths) {{
    try {{
      const payload = await api(path);
      rows.push({{
        url: "https://phx.unusualwhales.com" + path,
        status: 200,
        body: JSON.stringify(payload)
      }});
    }} catch (err) {{
      rows.push({{
        url: "https://phx.unusualwhales.com" + path,
        status: 0,
        error: String((err && err.message) || err || "unknown")
      }});
    }}
  }}
  return rows;
}})()
"""
        try:
            res = client.call(
                "Runtime.evaluate",
                {
                    "expression": expr,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
                timeout_sec=30,
            )
            value = (
                res.get("result", {})
                .get("result", {})
                .get("value", [])
            )
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    out.append(
                        {
                            "request_id": "runtime_api",
                            "url": item.get("url", ""),
                            "status": item.get("status", 0),
                            "mime_type": "application/json",
                            "body": item.get("body", ""),
                            "base64_encoded": False,
                            "body_error": item.get("error", ""),
                        }
                    )
        except Exception as exc:
            out.append(
                {
                    "request_id": "runtime_api",
                    "url": f"https://phx.unusualwhales.com{summary_path}",
                    "status": 0,
                    "mime_type": "application/json",
                    "body": "",
                    "base64_encoded": False,
                    "body_error": str(exc),
                }
            )
    return out


def classify_gex_endpoint(url: str) -> str:
    if "/one_minute" in url:
        return "one_minute"
    if "/strikes" in url:
        return "strikes"
    return "summary"


def url_query_date(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(str(url))
        qs = urllib.parse.parse_qs(parsed.query)
        values = qs.get("date") or []
        return str(values[0]) if values else ""
    except Exception:
        return ""


def response_matches_requested_date(url: str, date_str: str) -> bool:
    actual = url_query_date(url)
    return bool(actual and actual == str(date_str))


def payload_time_matches_requested_date(value: object, date_str: str) -> bool:
    raw = str(value or "").strip()
    return bool(raw.startswith(str(date_str)))


def parse_json_body(body: str) -> object:
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_num(value: object) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def normalize_gex_files(
    out_dir: Path,
    date_str: str,
    tickers: Sequence[str] | None = None,
) -> Dict[str, object]:
    raw_dir = out_dir / "uw_gex_raw"
    summary_path = out_dir / f"uw_gex_summary_{date_str}.csv"
    strikes_path = out_dir / f"uw_gex_strikes_{date_str}.csv"
    summary_rows: List[Dict[str, object]] = []
    strike_rows: List[Dict[str, object]] = []
    # Normalize every raw file we have for the date, not just the current
    # request chunk.  Broad historical GEX collection is intentionally batched;
    # filtering to only the latest chunk would overwrite prior rows and make
    # coverage appear to regress after each batch.
    requested = set()

    for path in sorted(raw_dir.glob(f"uw_gex_{date_str}_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ticker = clean_ticker(payload.get("ticker"))
        endpoints = payload.get("endpoints", {})
        if not isinstance(endpoints, dict):
            continue
        if requested and ticker not in requested:
            continue

        summary_payload = endpoints.get("summary", {})
        if isinstance(summary_payload, dict):
            if not response_matches_requested_date(str(summary_payload.get("url", "")), date_str):
                continue
            data = summary_payload.get("payload", {})
            if isinstance(data, dict):
                row = data.get("data", {})
                if isinstance(row, dict) and isinstance(row.get("data"), dict):
                    row = row.get("data", {})
                if isinstance(row, dict) and row:
                    if not payload_time_matches_requested_date(row.get("time"), date_str):
                        continue
                    summary_rows.append(
                        {
                            "ticker": ticker,
                            "date": date_str,
                            "source": "unusual_whales_dashboard_cdp",
                            "source_url": summary_payload.get("url"),
                            "captured_utc": summary_payload.get("captured_utc"),
                            "uw_time": row.get("time"),
                            "spot": row.get("price"),
                            "gamma_oi_per_1pct": row.get("gamma_per_one_percent_move_oi"),
                            "gamma_vol_per_1pct": row.get("gamma_per_one_percent_move_vol"),
                            "gamma_dir_per_1pct": row.get("gamma_per_one_percent_move_dir"),
                            "vanna_oi_per_1pct": row.get("vanna_per_one_percent_move_oi"),
                            "vanna_vol_per_1pct": row.get("vanna_per_one_percent_move_vol"),
                            "vanna_dir_per_1pct": row.get("vanna_per_one_percent_move_dir"),
                            "charm_oi_per_1pct": row.get("charm_per_one_percent_move_oi"),
                            "charm_vol_per_1pct": row.get("charm_per_one_percent_move_vol"),
                            "charm_dir_per_1pct": row.get("charm_per_one_percent_move_dir"),
                        }
                    )

        strikes_payload = endpoints.get("strikes", {})
        if isinstance(strikes_payload, dict):
            if not response_matches_requested_date(str(strikes_payload.get("url", "")), date_str):
                continue
            data = strikes_payload.get("payload", {})
            rows = data.get("data", []) if isinstance(data, dict) else []
            if isinstance(rows, dict) and isinstance(rows.get("data"), list):
                rows = rows.get("data", [])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if not payload_time_matches_requested_date(row.get("time"), date_str):
                        continue
                    strike_rows.append(
                        {
                            "ticker": ticker,
                            "date": row.get("date", date_str),
                            "source": "unusual_whales_dashboard_cdp",
                            "source_url": strikes_payload.get("url"),
                            "uw_time": row.get("time"),
                            "spot": row.get("price"),
                            "strike": row.get("strike"),
                            "call_gamma_oi": row.get("call_gamma_oi"),
                            "put_gamma_oi": row.get("put_gamma_oi"),
                            "call_gamma_vol": row.get("call_gamma_vol"),
                            "put_gamma_vol": row.get("put_gamma_vol"),
                            "call_gamma_ask": row.get("call_gamma_ask"),
                            "put_gamma_ask": row.get("put_gamma_ask"),
                            "call_gamma_bid": row.get("call_gamma_bid"),
                            "put_gamma_bid": row.get("put_gamma_bid"),
                            "call_vanna_oi": row.get("call_vanna_oi"),
                            "put_vanna_oi": row.get("put_vanna_oi"),
                            "call_charm_oi": row.get("call_charm_oi"),
                            "put_charm_oi": row.get("put_charm_oi"),
                        }
                    )

    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    elif summary_path.exists():
        summary_path.unlink()
    if strike_rows:
        with strikes_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(strike_rows[0].keys()))
            writer.writeheader()
            writer.writerows(strike_rows)
    elif strikes_path.exists():
        strikes_path.unlink()

    return {
        "summary_path": str(summary_path) if summary_rows else "",
        "strikes_path": str(strikes_path) if strike_rows else "",
        "summary_rows": len(summary_rows),
        "strike_rows": len(strike_rows),
    }


def collect_gex_with_cdp(
    out_dir: Path,
    date_str: str,
    tickers: Sequence[str],
    remote_debugging_url: str,
    wait_sec: float,
) -> List[Dict[str, object]]:
    raw_dir = out_dir / "uw_gex_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ws_url = find_uw_page_ws(remote_debugging_url)
    client = CdpClient(ws_url)
    collected = []
    try:
        client.call("Runtime.enable", timeout_sec=5)
        client.call("Page.enable", timeout_sec=5)
        client.call("Network.enable", timeout_sec=5)
        client.call("Network.setCacheDisabled", {"cacheDisabled": True}, timeout_sec=5)
        for idx, ticker in enumerate(tickers, start=1):
            print(f"[{idx}/{len(tickers)}] UW GEX {ticker}")
            responses = cdp_collect_ticker_gex(client, ticker, date_str, wait_sec=wait_sec)
            by_kind: Dict[str, Dict[str, object]] = {}
            for response in responses:
                if not response_matches_requested_date(str(response.get("url", "")), date_str):
                    continue
                kind = classify_gex_endpoint(str(response.get("url", "")))
                body = str(response.get("body", ""))
                parsed = parse_json_body(body)
                if parsed is not None:
                    by_kind[kind] = {
                        "url": response.get("url"),
                        "status": response.get("status"),
                        "captured_utc": utc_now_iso(),
                        "payload": parsed,
                    }
            ticker_payload = {
                "ticker": ticker,
                "date": date_str,
                "captured_utc": utc_now_iso(),
                "source": "unusual_whales_dashboard_cdp",
                "page_url": f"{UW_BASE_URL}/stock/{ticker}/greek-exposure?type=exposure&greek=gamma&date={date_str}",
                "endpoints": by_kind,
                "raw_response_count": len(responses),
            }
            path = raw_dir / f"uw_gex_{date_str}_{ticker}.json"
            write_json(path, ticker_payload)
            collected.append(
                {
                    "ticker": ticker,
                    "path": str(path),
                    "endpoint_kinds": sorted(by_kind.keys()),
                    "raw_response_count": len(responses),
                    "ok": bool(by_kind),
                }
            )
            print(f"  -> {'ok' if by_kind else 'missing'} {sorted(by_kind.keys())}", flush=True)
    finally:
        client.close()
    return collected


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mac browser-assisted Unusual Whales enrichment collector.")
    ap.add_argument("--mode", choices=["plan", "ingest", "collect-gex", "status"], default="plan")
    ap.add_argument("--date", required=True, help="Trading date YYYY-MM-DD.")
    ap.add_argument("--repo-root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--base-dir", default="", help="Date folder. Defaults to {repo-root}/{date}.")
    ap.add_argument("--out-dir", default="", help="Enrichment folder. Defaults to {base-dir}/enrichments/uw.")
    ap.add_argument("--input-csv", default="", help="CSV containing a ticker column. Defaults to final live table, then shortlist.")
    ap.add_argument("--tickers", default="", help="Optional comma-separated tickers overriding CSV discovery.")
    ap.add_argument("--browser", default="ChatGPT Atlas", help="macOS browser app name for `open -a`.")
    ap.add_argument("--open-dashboard", action="store_true", help="Open UW dashboard URLs after writing the plan.")
    ap.add_argument("--max-open-urls", type=int, default=8, help="Maximum URLs to open when --open-dashboard is set.")
    ap.add_argument("--downloads-dir", default=str(Path.home() / "Downloads"))
    ap.add_argument("--since", default="", help="UTC timestamp for ingest. Defaults to manifest marker.")
    ap.add_argument("--remote-debugging-url", default="http://127.0.0.1:9222")
    ap.add_argument("--browser-profile-dir", default="", help="Dedicated browser profile directory for CDP collection.")
    ap.add_argument("--wait-sec", type=float, default=8.0, help="Seconds to wait per UW ticker dashboard page.")
    ap.add_argument("--max-tickers", type=int, default=0, help="Optional cap for collection trials.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    date_str = args.date.strip()
    dt.datetime.strptime(date_str, "%Y-%m-%d")
    repo_root = Path(args.repo_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir.strip() else repo_root / date_str
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir.strip() else base_dir / "enrichments" / "uw"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers.strip():
        tickers = unique_tickers(args.tickers.split(","))
        input_csv = None
    else:
        input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv.strip() else resolve_default_input(repo_root, date_str)
        tickers = read_tickers_from_csv(input_csv)

    if not tickers:
        raise RuntimeError("No tickers found. Pass --tickers AAPL,NVDA or --input-csv with a ticker column.")
    if int(args.max_tickers) > 0:
        tickers = tickers[: int(args.max_tickers)]

    urls = dashboard_urls(tickers)
    manifest = load_manifest(out_dir, date_str)

    if args.mode == "plan":
        marker_utc = utc_now_iso()
        plan_path = write_plan(out_dir, date_str, tickers, urls)
        payload = {
            "date": date_str,
            "created_utc": marker_utc,
            "updated_utc": marker_utc,
            "status": "planned",
            "collector": "uwos.collect_uw_enrichments_mac",
            "repo_root": str(repo_root),
            "base_dir": str(base_dir),
            "out_dir": str(out_dir),
            "input_csv": str(input_csv) if input_csv else "manual_tickers",
            "tickers": tickers,
            "download_marker_utc": marker_utc,
            "plan_path": str(plan_path),
            "urls": urls,
            "downloaded_files": [],
            "notes": [
                "Browser/dashboard data is not part of the approval model until exports are ingested and parser integration is added.",
                "This collector copies files from Downloads; it does not delete or move originals.",
            ],
        }
        manifest_path = write_manifest(out_dir, date_str, payload)
        if args.open_dashboard:
            all_urls = list(urls.get("market_urls", []))
            ticker_urls = urls.get("ticker_urls", {})
            if isinstance(ticker_urls, dict):
                for ticker in tickers:
                    all_urls.extend(ticker_urls.get(ticker, [])[:1])
            open_urls(all_urls, args.browser, max(1, int(args.max_open_urls)))
        print(f"Tickers: {len(tickers)}")
        print(f"Wrote: {plan_path}")
        print(f"Wrote: {manifest_path}")
        if args.open_dashboard:
            print(f"Opened up to {args.max_open_urls} URLs in {args.browser}")
        return

    if args.mode == "ingest":
        if args.since.strip():
            since = parse_iso_time(args.since)
        else:
            since = parse_iso_time(str(manifest.get("download_marker_utc", utc_now_iso())))
        copied = ingest_downloads(
            downloads_dir=Path(args.downloads_dir).expanduser().resolve(),
            out_dir=out_dir,
            tickers=tickers,
            since_utc=since,
        )
        payload = dict(manifest)
        payload.update(
            {
                "date": date_str,
                "updated_utc": utc_now_iso(),
                "status": "ingested" if copied else "no_downloads_found",
                "repo_root": str(repo_root),
                "base_dir": str(base_dir),
                "out_dir": str(out_dir),
                "tickers": tickers,
                "ingest_since_utc": since.replace(microsecond=0).isoformat(),
                "downloaded_files": copied,
            }
        )
        manifest_path = write_manifest(out_dir, date_str, payload)
        print(f"Copied files: {len(copied)}")
        for item in copied:
            print(item["copied_path"])
        print(f"Wrote: {manifest_path}")
        return

    if args.mode == "collect-gex":
        browser_state = ensure_uw_cdp_browser(
            remote_debugging_url=str(args.remote_debugging_url),
            browser_app=str(args.browser),
            profile_dir=str(args.browser_profile_dir),
            wait_sec=8.0,
            start_url=UW_BASE_URL,
        )
        if not browser_state.get("ok"):
            raise RuntimeError(f"Could not start UW CDP browser: {browser_state}")
        if browser_state.get("browser_app") not in {"existing", ""}:
            print(
                f"CDP browser ready: {browser_state.get('browser_app')} "
                f"profile={browser_state.get('profile_dir')}"
            )
        collected = collect_gex_with_cdp(
            out_dir=out_dir,
            date_str=date_str,
            tickers=tickers,
            remote_debugging_url=str(args.remote_debugging_url),
            wait_sec=float(args.wait_sec),
        )
        normalized = normalize_gex_files(out_dir, date_str, tickers=tickers)
        payload = dict(manifest)
        existing = payload.get("uw_gex_files", [])
        if not isinstance(existing, list):
            existing = []
        payload.update(
            {
                "date": date_str,
                "updated_utc": utc_now_iso(),
                "status": "gex_collected",
                "repo_root": str(repo_root),
                "base_dir": str(base_dir),
                "out_dir": str(out_dir),
                "tickers": tickers,
                "uw_gex_files": existing + collected,
                "uw_gex_normalized": normalized,
                "gex_source": "unusual_whales_dashboard_cdp",
            }
        )
        manifest_path = write_manifest(out_dir, date_str, payload)
        print(f"Collected GEX tickers: {sum(1 for x in collected if x.get('ok'))}/{len(collected)}")
        print(f"Summary rows: {normalized.get('summary_rows')} -> {normalized.get('summary_path')}")
        print(f"Strike rows: {normalized.get('strike_rows')} -> {normalized.get('strikes_path')}")
        print(f"Wrote: {manifest_path}")
        return

    manifest_path = out_dir / f"uw_enrichment_manifest_{date_str}.json"
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
