#!/usr/bin/env python3
"""Scrape all posts from X handles in the prior N hours via FxTwitter API."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Any

DEFAULT_HANDLES = [
    "DeepValueBagger",
    "TJTheWheelDeal",
    "Jake__Wujastyk",
    "Banana3Stocks",
]
UA = "Mozilla/5.0 (compatible; tradedesk-daily-x/1.0)"


def fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def parse_created(post: dict[str, Any]) -> datetime | None:
    ts = post.get("created_timestamp")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    raw = post.get("created_at")
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def summarize_post(post: dict[str, Any], handle: str) -> str:
    text = re.sub(r"https?://\S+", "", post.get("text") or "").strip()
    text = re.sub(r"\s+", " ", text)
    tickers = sorted(set(re.findall(r"\$[A-Za-z]{1,6}", post.get("text") or "")))
    tags: list[str] = []
    replying = post.get("replying_to")
    if replying:
        if isinstance(replying, list):
            names = [str(x) for x in replying if isinstance(x, str) and not x.startswith("http")]
            if names:
                tags.append(f"reply to {', '.join(names)}")
            else:
                tags.append("reply")
        elif isinstance(replying, dict):
            name = replying.get("screen_name") or replying.get("name")
            tags.append(f"reply to @{name}" if name else "reply")
        else:
            tags.append("reply")
    if post.get("quote"):
        tags.append("quote tweet")
    if post.get("media"):
        kinds = sorted({m.get("type", "media") for m in post["media"] if isinstance(m, dict)})
        tags.append("attached " + ", ".join(kinds))
    if post.get("reposted_by"):
        tags.append("repost")

    parts: list[str] = []
    if text:
        parts.append(text)
    else:
        parts.append("Media-only post with no visible text in the preview.")

    if tickers:
        parts.append(f"Mentions {', '.join(tickers)}.")
    if tags:
        parts.append(f"Type: {', '.join(tags)}.")

    engagement = []
    for key, label in [("likes", "likes"), ("replies", "replies"), ("reposts", "reposts"), ("views", "views")]:
        val = post.get(key)
        if val is not None:
            engagement.append(f"{val:,} {label}")
    if engagement:
        parts.append(f"Engagement: {', '.join(engagement)}.")

    created = parse_created(post)
    if created:
        parts.append(f"Posted {created.strftime('%Y-%m-%d %H:%M UTC')}.")

    return " ".join(parts)


def fetch_handle_posts(handle: str, since_dt: datetime) -> list[dict[str, Any]]:
    since_ts = int(since_dt.timestamp())
    cursor: str | None = None
    collected: dict[str, dict[str, Any]] = {}
    page = 0

    while True:
        page += 1
        params = {
            "count": "100",
            "since": str(since_ts),
            "with_replies": "true",
        }
        if cursor:
            params["cursor"] = cursor
        url = f"https://api.fxtwitter.com/2/profile/{handle}/statuses?{urllib.parse.urlencode(params)}"
        data = fetch_json(url)
        results = data.get("results") or []
        if not results:
            break

        page_in_window = 0
        for post in results:
            if not isinstance(post, dict):
                continue
            author = (post.get("author") or {}).get("screen_name", "")
            if author.lower() != handle.lower():
                continue
            created = parse_created(post)
            if created is None or created < since_dt:
                continue
            page_in_window += 1
            tid = str(post.get("id"))
            post["url"] = post.get("url") or f"https://x.com/{handle}/status/{tid}"
            post["summary"] = summarize_post(post, handle)
            collected[tid] = post

        cursor = (data.get("cursor") or {}).get("bottom")
        oldest = None
        for post in results:
            created = parse_created(post)
            if created and (oldest is None or created < oldest):
                oldest = created
        if oldest and oldest < since_dt:
            break
        if not cursor:
            break
        if page_in_window == 0 and oldest and oldest < since_dt:
            break

    posts = list(collected.values())
    posts.sort(key=lambda p: parse_created(p) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return posts


def scrape_handles(handles: list[str], hours: int = 24) -> dict[str, Any]:
    since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    results: dict[str, list[dict[str, Any]]] = {}
    for handle in handles:
        results[handle] = fetch_handle_posts(handle, since_dt)
    return {"since_utc": since_dt.isoformat(), "results": results}


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# X Handle Posts — prior 24 hours",
        "",
        f"**Window start (UTC):** {payload['since_utc']}",
        "",
    ]
    for handle, posts in payload["results"].items():
        lines.append(f"## @{handle} ({len(posts)} posts)")
        lines.append("")
        for i, post in enumerate(posts, 1):
            lines.append(f"### {i}. {post['summary']}")
            lines.append(f"**Link:** {post['url']}")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--handles", nargs="*", default=DEFAULT_HANDLES)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument(
        "--report-md",
        default="",
        help="Markdown report path (default: reports/x_daily_handles_YYYY-MM-DD.md)",
    )
    args = parser.parse_args()
    payload = scrape_handles(args.handles, hours=args.hours)
    report_path = Path(args.report_md) if args.report_md else Path(
        f"reports/x_daily_handles_{datetime.now(timezone.utc).date().isoformat()}.md"
    )
    write_markdown_report(payload, report_path)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote markdown report: {report_path}")
    for handle, posts in payload["results"].items():
        print(f"  @{handle}: {len(posts)} posts")


if __name__ == "__main__":
    main()
