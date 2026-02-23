#!/usr/bin/env python3
"""
Scrape public X profile posts using a logged-in persistent browser profile.

Outputs:
- posts.csv
- posts.jsonl
- scrape_summary.md
- run_manifest.json
- screenshots/*.png (optional)
- media/* (optional)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape public posts from an X profile via logged-in browser profile.")
    parser.add_argument("--handle", required=True, help="X profile handle, for example: elonmusk")
    parser.add_argument(
        "--run-date",
        default=dt.date.today().isoformat(),
        help="Date folder to write into (YYYY-MM-DD). Default: today.",
    )
    # Backward-compatible alias.
    parser.add_argument("--trade-date", dest="run_date", help=argparse.SUPPRESS)
    parser.add_argument("--base-dir", default=".", help="Root folder for date folders.")
    parser.add_argument("--out-subdir", default="x_scrapes", help="Subdirectory inside the date folder.")
    parser.add_argument(
        "--time-filter",
        choices=["all", "past_7d", "past_30d", "past_month", "this_month", "this_year"],
        default="past_month",
        help=(
            "Tweet time window filter. Default is past_month (rolling 30 days). "
            "Use all to disable time filtering."
        ),
    )
    parser.add_argument(
        "--since-date",
        default="",
        help="Optional inclusive lower bound for tweet publish date (YYYY-MM-DD). Overrides time-filter start.",
    )
    parser.add_argument(
        "--until-date",
        default="",
        help="Optional inclusive upper bound for tweet publish date (YYYY-MM-DD). Default is now (UTC).",
    )
    parser.add_argument(
        "--profile-dir",
        default="tokens/x_playwright_profile",
        help="Persistent Playwright profile dir with your X login session.",
    )
    parser.add_argument(
        "--browser-channel",
        choices=["chromium", "chrome", "msedge"],
        default="msedge",
        help="Browser channel for Playwright persistent profile.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-posts", type=int, default=80, help="Stop after this many unique posts.")
    parser.add_argument("--max-scrolls", type=int, default=80, help="Maximum scroll cycles.")
    parser.add_argument("--scroll-delay-ms", type=int, default=1400, help="Delay between scrolls.")
    parser.add_argument("--wait-after-open-ms", type=int, default=2500, help="Initial wait after opening profile.")
    parser.add_argument(
        "--manual-login-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open login page only so you can log in manually and persist session, then exit.",
    )
    parser.add_argument("--login-url", default="https://x.com/i/flow/login", help="Login URL for manual mode.")
    parser.add_argument(
        "--disable-automation-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --disable-blink-features=AutomationControlled.",
    )
    parser.add_argument(
        "--capture-screenshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture one screenshot per scraped post card.",
    )
    parser.add_argument(
        "--download-media",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download media URLs found in posts.",
    )
    parser.add_argument(
        "--include-replies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include replies. Default is false (best effort filter for top-level posts).",
    )
    return parser.parse_args()


def clean_handle(handle: str) -> str:
    h = str(handle).strip().lstrip("@")
    if not h:
        raise ValueError("Handle cannot be empty.")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,20}", h):
        raise ValueError(f"Invalid handle: {handle}")
    return h


def parse_trade_date(text: str) -> dt.date:
    return dt.datetime.strptime(str(text).strip(), "%Y-%m-%d").date()


def parse_date_optional(text: str) -> Optional[dt.date]:
    raw = str(text or "").strip()
    if not raw:
        return None
    return parse_trade_date(raw)


def parse_published_at(text: str) -> Optional[dt.datetime]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def resolve_time_window(
    time_filter: str,
    since_date_text: str,
    until_date_text: str,
) -> tuple[Optional[dt.datetime], Optional[dt.datetime], str]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    until_date = parse_date_optional(until_date_text)
    until_dt = (
        dt.datetime.combine(until_date, dt.time.max, tzinfo=dt.timezone.utc)
        if until_date is not None
        else now_utc
    )

    since_override = parse_date_optional(since_date_text)
    if since_override is not None:
        since_dt = dt.datetime.combine(since_override, dt.time.min, tzinfo=dt.timezone.utc)
        label = f"custom since {since_override.isoformat()} to {until_dt.date().isoformat()}"
        return since_dt, until_dt, label

    if time_filter == "all":
        return None, until_dt, "all"
    if time_filter == "past_7d":
        since_dt = until_dt - dt.timedelta(days=7)
        return since_dt, until_dt, "past 7 days"
    if time_filter in {"past_30d", "past_month"}:
        since_dt = until_dt - dt.timedelta(days=30)
        return since_dt, until_dt, "past 30 days"
    if time_filter == "this_month":
        since_dt = dt.datetime(until_dt.year, until_dt.month, 1, tzinfo=dt.timezone.utc)
        return since_dt, until_dt, "this month"
    if time_filter == "this_year":
        since_dt = dt.datetime(until_dt.year, 1, 1, tzinfo=dt.timezone.utc)
        return since_dt, until_dt, "this year"

    return None, until_dt, "all"


def in_time_window(
    published_at: Optional[dt.datetime],
    since_dt: Optional[dt.datetime],
    until_dt: Optional[dt.datetime],
) -> bool:
    if published_at is None:
        return False
    if since_dt is not None and published_at < since_dt:
        return False
    if until_dt is not None and published_at > until_dt:
        return False
    return True


def classify_content(text: str, media_urls: List[str]) -> str:
    has_text = bool(str(text or "").strip())
    has_media = bool(media_urls)
    if has_text and has_media:
        return "text_and_media"
    if has_text:
        return "text_only"
    if has_media:
        return "media_only"
    return "empty_or_unparsed"


def parse_compact_count(text: Any) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s:
        return None
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([KMB])?$", s, flags=re.I)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    mul = 1
    if unit == "K":
        mul = 1000
    elif unit == "M":
        mul = 1000000
    elif unit == "B":
        mul = 1000000000
    return int(round(val * mul))


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())[:120] or "item"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "tweet_id",
                    "tweet_url",
                    "author_handle",
                    "author_name",
                    "published_at",
                    "text",
                    "likes",
                    "replies",
                    "reposts",
                    "views",
                    "bookmark_count",
                    "media_urls",
                    "has_text",
                    "has_media",
                    "content_type",
                    "screenshot_file",
                    "is_reply",
                    "is_retweet_or_repost",
                ]
            )
        return

    keys = [
        "index",
        "tweet_id",
        "tweet_url",
        "author_handle",
        "author_name",
        "published_at",
        "text",
        "likes",
        "replies",
        "reposts",
        "views",
        "bookmark_count",
        "media_urls",
        "has_text",
        "has_media",
        "content_type",
        "screenshot_file",
        "is_reply",
        "is_retweet_or_repost",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            media_urls = row.get("media_urls", [])
            row["media_urls"] = ";".join(media_urls) if isinstance(media_urls, list) else str(media_urls or "")
            writer.writerow({k: row.get(k, "") for k in keys})


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_md_summary(
    out_path: Path,
    handle: str,
    profile_url: str,
    rows: List[Dict[str, Any]],
    screenshot_dir: Path,
    media_dir: Path,
    started: dt.datetime,
    finished: dt.datetime,
    time_filter_label: str,
    skipped_counts: Dict[str, int],
) -> None:
    lines: List[str] = []
    lines.append(f"# X Profile Scrape: @{handle}")
    lines.append("")
    lines.append(f"- Profile URL: {profile_url}")
    lines.append(f"- Run started: {started.isoformat(timespec='seconds')}")
    lines.append(f"- Run finished: {finished.isoformat(timespec='seconds')}")
    lines.append(f"- Time filter: {time_filter_label}")
    lines.append(f"- Posts captured: {len(rows)}")
    lines.append(f"- Skipped (reply filter): {int(skipped_counts.get('reply', 0))}")
    lines.append(f"- Skipped (time window): {int(skipped_counts.get('time_window', 0))}")
    lines.append(f"- Skipped (missing publish timestamp): {int(skipped_counts.get('missing_timestamp', 0))}")
    lines.append(f"- Screenshots dir: `{screenshot_dir}`")
    lines.append(f"- Media dir: `{media_dir}`")
    lines.append("")

    if not rows:
        lines.append("_No posts captured._")
    else:
        content_counts = Counter(str(r.get("content_type", "")) for r in rows)
        lines.append(
            "- Content mix: "
            f"text+media={content_counts.get('text_and_media', 0)}, "
            f"text-only={content_counts.get('text_only', 0)}, "
            f"media-only={content_counts.get('media_only', 0)}, "
            f"empty/unparsed={content_counts.get('empty_or_unparsed', 0)}"
        )
        lines.append("")
        table_rows: List[Dict[str, Any]] = []
        for r in rows:
            table_rows.append(
                {
                    "#": r.get("index"),
                    "Published": r.get("published_at", ""),
                    "Tweet URL": r.get("tweet_url", ""),
                    "Likes": r.get("likes"),
                    "Replies": r.get("replies"),
                    "Reposts": r.get("reposts"),
                    "Views": r.get("views"),
                    "Content": r.get("content_type"),
                    "Media Count": len(r.get("media_urls", []) or []),
                    "Text (trimmed)": (str(r.get("text", ""))[:180] + ("..." if len(str(r.get("text", ""))) > 180 else "")),
                }
            )
        try:
            import pandas as pd  # type: ignore

            lines.append(pd.DataFrame(table_rows).to_markdown(index=False))
        except Exception:
            lines.append("Could not render markdown table (pandas missing).")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    handle = clean_handle(args.handle)
    run_date = parse_trade_date(args.run_date)
    profile_url = f"https://x.com/{handle}"
    since_dt, until_dt, time_filter_label = resolve_time_window(
        str(args.time_filter),
        str(args.since_date or ""),
        str(args.until_date or ""),
    )

    base_dir = Path(args.base_dir).resolve()
    day_dir = base_dir / run_date.isoformat()
    out_dir = day_dir / args.out_subdir / handle
    screenshot_dir = out_dir / "screenshots"
    media_dir = out_dir / "media"
    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    print("Starting X profile scrape")
    print(f"- Handle: @{handle}")
    print(f"- Profile URL: {profile_url}")
    print(f"- Time filter: {time_filter_label}")
    print(f"- Output dir: {out_dir}")
    print(f"- Browser profile: {profile_dir}")
    print("Use this only on accounts/content you are allowed to access and analyze.")

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        print("Playwright is required. Install with:")
        print("  pip install playwright")
        print("  python -m playwright install chromium")
        return 2

    started = dt.datetime.now()
    posts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    download_manifest: List[Dict[str, Any]] = []
    skipped_counts: Dict[str, int] = {
        "reply": 0,
        "time_window": 0,
        "missing_timestamp": 0,
    }

    with sync_playwright() as pw:
        launch_kwargs: Dict[str, Any] = {
            "user_data_dir": str(profile_dir),
            "headless": bool(args.headless),
            "accept_downloads": True,
            "viewport": {"width": 1720, "height": 1080},
        }
        if args.browser_channel in {"chrome", "msedge"}:
            launch_kwargs["channel"] = args.browser_channel
        if args.disable_automation_flags:
            launch_kwargs["args"] = ["--disable-blink-features=AutomationControlled"]

        context = pw.chromium.launch_persistent_context(**launch_kwargs)
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(25000)

        if args.manual_login_only:
            print("Manual login mode. Log in to X and then press ENTER here to save session.")
            page.goto(args.login_url, wait_until="domcontentloaded")
            try:
                input("Press ENTER after login/session is ready...")
            except EOFError:
                print("No stdin detected; waiting 90s before exit.")
                page.wait_for_timeout(90000)
            context.close()
            return 0

        page.goto(profile_url, wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.wait_after_open_ms))

        stagnation = 0
        scrolls = 0

        while len(posts) < int(args.max_posts) and scrolls < int(args.max_scrolls):
            cards = page.locator("article[data-testid='tweet']")
            count = cards.count()
            before = len(posts)
            for idx in range(count):
                card = cards.nth(idx)
                try:
                    item = card.evaluate(
                        """(el) => {
                            const getText = (sel) => {
                                const n = el.querySelector(sel);
                                return n ? (n.innerText || n.textContent || '').trim() : '';
                            };
                            const getAttr = (sel, attr) => {
                                const n = el.querySelector(sel);
                                return n ? (n.getAttribute(attr) || '') : '';
                            };
                            const tweetTextNode = el.querySelector('div[data-testid="tweetText"]');
                            const tweetText = tweetTextNode ? (tweetTextNode.innerText || tweetTextNode.textContent || '').trim() : '';
                            const timeNode = el.querySelector('time');
                            const publishedAt = timeNode ? (timeNode.getAttribute('datetime') || '') : '';
                            const timeLink = timeNode ? timeNode.closest('a') : null;
                            const tweetUrl = timeLink ? (timeLink.getAttribute('href') || '') : '';
                            const authorLink = el.querySelector('a[role="link"][href^="/"][href*="/status/"]');
                            const socialContext = getText('div[data-testid="socialContext"]');

                            const replyText = getText('button[data-testid="reply"] span');
                            const repostText = getText('button[data-testid="retweet"] span, button[data-testid="unretweet"] span');
                            const likeText = getText('button[data-testid="like"] span, button[data-testid="unlike"] span');
                            const viewText = getText('a[href$="/analytics"] span');
                            const bookmarkText = getText('button[data-testid="bookmark"] span, button[data-testid="removeBookmark"] span');

                            const mediaUrls = [];
                            el.querySelectorAll('div[data-testid="tweetPhoto"] img, img[src*="twimg.com/media"], video').forEach((m) => {
                                if (m.tagName.toLowerCase() === 'img') {
                                    const src = m.getAttribute('src') || '';
                                    if (src) mediaUrls.push(src);
                                } else {
                                    const poster = m.getAttribute('poster') || '';
                                    if (poster) mediaUrls.push(poster);
                                }
                            });

                            const userNameNode = el.querySelector('div[dir="auto"] span');
                            const authorName = userNameNode ? (userNameNode.textContent || '').trim() : '';
                            const authorHandleNode = Array.from(el.querySelectorAll('a[href^="/"] span')).find(x => (x.textContent || '').trim().startsWith('@'));
                            const authorHandle = authorHandleNode ? (authorHandleNode.textContent || '').trim().replace(/^@/, '') : '';

                            return {
                                tweet_text: tweetText,
                                published_at: publishedAt,
                                tweet_url: tweetUrl,
                                author_name: authorName,
                                author_handle: authorHandle,
                                reply_count_text: replyText,
                                repost_count_text: repostText,
                                like_count_text: likeText,
                                view_count_text: viewText,
                                bookmark_count_text: bookmarkText,
                                media_urls: Array.from(new Set(mediaUrls)),
                                social_context: socialContext,
                            };
                        }"""
                    )
                except Exception:
                    continue

                raw_url = str(item.get("tweet_url", "")).strip()
                if not raw_url:
                    continue
                full_url = raw_url if raw_url.startswith("http") else f"https://x.com{raw_url}"
                tweet_id_match = re.search(r"/status/(\d+)", full_url)
                tweet_id = tweet_id_match.group(1) if tweet_id_match else ""
                unique_key = tweet_id or full_url
                if unique_key in seen:
                    continue
                seen.add(unique_key)

                is_reply = "/status/" in full_url and bool(item.get("social_context"))
                if (not args.include_replies) and is_reply:
                    skipped_counts["reply"] += 1
                    continue

                published_at_raw = str(item.get("published_at", "")).strip()
                published_at_dt = parse_published_at(published_at_raw)
                if (since_dt is not None or str(args.time_filter) != "all") and published_at_dt is None:
                    skipped_counts["missing_timestamp"] += 1
                    continue
                if not in_time_window(published_at_dt, since_dt, until_dt):
                    skipped_counts["time_window"] += 1
                    continue

                media_urls = list(item.get("media_urls", []) or [])
                content_type = classify_content(str(item.get("tweet_text", "")), media_urls)
                record: Dict[str, Any] = {
                    "index": len(posts) + 1,
                    "tweet_id": tweet_id,
                    "tweet_url": full_url,
                    "author_handle": str(item.get("author_handle", "")).strip().lstrip("@"),
                    "author_name": str(item.get("author_name", "")).strip(),
                    "published_at": published_at_raw,
                    "text": str(item.get("tweet_text", "")).strip(),
                    "likes": parse_compact_count(item.get("like_count_text")),
                    "replies": parse_compact_count(item.get("reply_count_text")),
                    "reposts": parse_compact_count(item.get("repost_count_text")),
                    "views": parse_compact_count(item.get("view_count_text")),
                    "bookmark_count": parse_compact_count(item.get("bookmark_count_text")),
                    "media_urls": media_urls,
                    "has_text": content_type in {"text_only", "text_and_media"},
                    "has_media": content_type in {"media_only", "text_and_media"},
                    "content_type": content_type,
                    "screenshot_file": "",
                    "is_reply": is_reply,
                    "is_retweet_or_repost": bool(item.get("social_context")),
                }

                if args.capture_screenshots:
                    shot_name = safe_slug(f"{record['index']:04d}_{record.get('tweet_id') or int(time.time()*1000)}") + ".png"
                    shot_path = screenshot_dir / shot_name
                    try:
                        card.screenshot(path=str(shot_path))
                        record["screenshot_file"] = str(shot_path.relative_to(out_dir))
                    except Exception:
                        pass

                if args.download_media and record["media_urls"]:
                    for media_i, media_url in enumerate(record["media_urls"], start=1):
                        media_ext = ".jpg"
                        m = re.search(r"\.([a-zA-Z0-9]{3,4})(?:[?&#]|$)", media_url)
                        if m:
                            media_ext = "." + m.group(1).lower()
                        media_name = safe_slug(
                            f"{record['index']:04d}_{record.get('tweet_id') or 'noid'}_{media_i}"
                        ) + media_ext
                        media_path = media_dir / media_name
                        try:
                            resp = context.request.get(media_url, timeout=30000)
                            if resp.ok:
                                media_path.write_bytes(resp.body())
                                download_manifest.append(
                                    {
                                        "tweet_id": record.get("tweet_id", ""),
                                        "tweet_url": record.get("tweet_url", ""),
                                        "media_url": media_url,
                                        "file": str(media_path.relative_to(out_dir)),
                                        "bytes": media_path.stat().st_size,
                                    }
                                )
                        except Exception:
                            continue

                posts.append(record)
                if len(posts) >= int(args.max_posts):
                    break

            added = len(posts) - before
            if added == 0:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= 5:
                break

            page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.8)")
            page.wait_for_timeout(int(args.scroll_delay_ms))
            scrolls += 1

        context.close()

    finished = dt.datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)

    posts_csv = out_dir / "posts.csv"
    posts_jsonl = out_dir / "posts.jsonl"
    md_summary = out_dir / "scrape_summary.md"
    run_manifest = out_dir / "run_manifest.json"
    media_manifest = out_dir / "media_manifest.json"

    write_csv(posts_csv, posts)
    write_jsonl(posts_jsonl, posts)
    build_md_summary(
        out_path=md_summary,
        handle=handle,
        profile_url=profile_url,
        rows=posts,
        screenshot_dir=screenshot_dir,
        media_dir=media_dir,
        started=started,
        finished=finished,
        time_filter_label=time_filter_label,
        skipped_counts=skipped_counts,
    )
    manifest_payload = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "handle": handle,
        "profile_url": profile_url,
        "max_posts": int(args.max_posts),
        "max_scrolls": int(args.max_scrolls),
        "time_filter": time_filter_label,
        "since_utc": since_dt.isoformat() if since_dt is not None else None,
        "until_utc": until_dt.isoformat() if until_dt is not None else None,
        "captured_posts": len(posts),
        "skipped": skipped_counts,
        "output_dir": str(out_dir),
        "files": {
            "posts_csv": str(posts_csv),
            "posts_jsonl": str(posts_jsonl),
            "scrape_summary_md": str(md_summary),
            "media_manifest_json": str(media_manifest),
        },
    }
    run_manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    media_manifest.write_text(json.dumps(download_manifest, indent=2), encoding="utf-8")

    print("")
    print(f"Done. Captured posts: {len(posts)}")
    print(f"- CSV: {posts_csv}")
    print(f"- JSONL: {posts_jsonl}")
    print(f"- Summary: {md_summary}")
    print(f"- Manifest: {run_manifest}")
    if args.download_media:
        print(f"- Media files: {len(download_manifest)} in {media_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
