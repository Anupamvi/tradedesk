#!/usr/bin/env python3
"""Fetch UW per-ticker daily options state (market_state_all) and build a research panel.

The phx endpoint is Bearer-token authenticated. The token lives in the signed-in
browser session, so it is handed over out-of-band:

  1. python3 scripts/fetch_uw_daily_state.py --wait-token
     -> listens on 127.0.0.1:<port> for a single POST carrying the token
  2. browser posts the captured Authorization header to that port
  3. python3 scripts/fetch_uw_daily_state.py --build
     -> reads the token file, fetches one request per ticker, writes the panel

One request per ticker returns up to 400 trading days, so the whole universe is
~150 requests.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

API = "https://phx.unusualwhales.com/api/market_state_all/{ticker}?limit={limit}"
TOKEN_PATH = Path("/tmp/uw_tok.txt")
DEFAULT_PORT = 8765


# --------------------------------------------------------------------------- token


class _TokenHandler(BaseHTTPRequestHandler):
    received = None

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace").strip()
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(b"ok")
        _TokenHandler.received = body

    def log_message(self, fmt, *args):  # silence; never log the body
        return


def wait_for_token(port: int) -> int:
    srv = HTTPServer(("127.0.0.1", port), _TokenHandler)
    srv.timeout = 180
    print(f"listening on 127.0.0.1:{port} for token (180s)...", flush=True)
    deadline = time.time() + 180
    while _TokenHandler.received is None and time.time() < deadline:
        srv.handle_request()
    srv.server_close()
    tok = _TokenHandler.received
    if not tok:
        print("no token received", file=sys.stderr)
        return 1
    fd = os.open(str(TOKEN_PATH), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(tok)
    print(f"token stored ({len(tok)} chars) -> {TOKEN_PATH}")
    return 0


# --------------------------------------------------------------------------- fetch


def _get(url: str, token: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={
        "Authorization": token,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _rows(payload) -> list:
    if isinstance(payload, dict):
        for key in ("data", "market_state_all", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    return payload if isinstance(payload, list) else []


def build(tickers_file: Path, out_path: Path, cache_dir: Path, limit: int, pause: float) -> int:
    if not TOKEN_PATH.exists():
        print(f"missing {TOKEN_PATH}; run --wait-token first", file=sys.stderr)
        return 1
    token = TOKEN_PATH.read_text().strip()
    if not token.lower().startswith("bearer "):
        token = "Bearer " + token

    tickers = [t.strip().upper() for t in tickers_file.read_text().split() if t.strip()]
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames, failed = [], []
    for i, ticker in enumerate(tickers, 1):
        cached = cache_dir / f"{ticker}.json"
        if cached.exists():
            try:
                rows = _rows(json.loads(cached.read_text()))
            except Exception:
                rows = []
        else:
            try:
                payload = _get(API.format(ticker=ticker, limit=limit), token)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    print(f"\nAUTH FAILED at {ticker} (HTTP {exc.code}) -- token expired. "
                          f"Re-run --wait-token and re-post.", file=sys.stderr)
                    return 2
                failed.append((ticker, f"HTTP {exc.code}"))
                continue
            except Exception as exc:
                failed.append((ticker, str(exc)[:60]))
                continue
            cached.write_text(json.dumps(payload))
            rows = _rows(payload)
            time.sleep(pause)

        if not rows:
            failed.append((ticker, "empty"))
            continue
        for row in rows:
            row["ticker"] = row.get("ticker") or ticker
        frames.extend(rows)
        if i % 25 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {ticker:6s} rows={len(frames):,}", flush=True)

    if not frames:
        print("no rows fetched", file=sys.stderr)
        return 1

    import pandas as pd

    df = pd.DataFrame(frames)
    df = df.rename(columns={"date": "asof"})
    num_cols = [c for c in df.columns if c not in ("asof", "ticker", "market_time")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["asof"] = df["asof"].astype(str)
    df = df.drop_duplicates(["asof", "ticker"]).sort_values(["asof", "ticker"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", newline="") as fh:
        df.to_csv(fh, index=False)

    print(f"\nwrote {out_path}")
    print(f"  rows={len(df):,}  tickers={df['ticker'].nunique()}  "
          f"days={df['asof'].nunique()}  range={df['asof'].min()}..{df['asof'].max()}")
    if failed:
        print(f"  failed ({len(failed)}): " + ", ".join(f"{t}:{e}" for t, e in failed[:12]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-token", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tickers", type=Path, default=Path("/tmp/uw_tickers.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("/Users/anuppamvi/uw_root/tradedesk/out/research/uw_daily_state.csv.gz"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/Users/anuppamvi/uw_root/tradedesk/out/research/uw_state_cache"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--pause", type=float, default=0.35)
    args = ap.parse_args()

    if args.wait_token:
        return wait_for_token(args.port)
    if args.build:
        return build(args.tickers, args.out, args.cache_dir, args.limit, args.pause)
    ap.error("pass --wait-token or --build")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
