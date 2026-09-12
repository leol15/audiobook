#!/usr/bin/env python3
"""Scrape chapter text from quanben.io for '全球诡异时代' (quanqiuguiyishidai).

Chapters are served at /n/<slug>/<n>.html with sequential numeric ids; each
page carries a rel="next" link to the following chapter, so we just walk
forward from chapter 1 until a page has no next link (end of book) or a
fetch fails (missing chapter / rate limit / ban).

Usage:
    python3 scrape.py                  # scrape all chapters, resume if interrupted
    python3 scrape.py --start 5 --end 10
    python3 scrape.py --delay 2.0      # be gentler on the server
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

BASE = "https://www.quanben.io"
SLUG = "quanqiuguiyishidai"
OUT_DIR = Path(__file__).parent / "output"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class ChapterParser(HTMLParser):
    """Pulls the headline (h1.headline) and article paragraphs (#content p)."""

    def __init__(self) -> None:
        super().__init__()
        self.in_headline = False
        self.in_content = False
        self.content_depth = 0
        self.in_p = False
        self.in_ad_span = False
        self.ad_span_depth = 0
        self.title_parts: list[str] = []
        self.paragraphs: list[str] = []
        self._cur_p: list[str] = []
        self.has_next = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h1" and "headline" in (attrs.get("class") or ""):
            self.in_headline = True
        if tag == "div" and attrs.get("id") == "content":
            self.in_content = True
            self.content_depth = 1
            return
        if self.in_content and tag == "div":
            self.content_depth += 1
        if self.in_content and tag == "span" and attrs.get("id") == "ad":
            self.in_ad_span = True
            self.ad_span_depth = 1
            return
        if self.in_ad_span and tag == "span":
            self.ad_span_depth += 1
        if self.in_content and not self.in_ad_span and tag == "p":
            self.in_p = True
            self._cur_p = []
        if tag == "a" and attrs.get("rel") == "next":
            self.has_next = True

    def handle_endtag(self, tag):
        if tag == "h1" and self.in_headline:
            self.in_headline = False
        if self.in_ad_span:
            if tag == "span":
                self.ad_span_depth -= 1
                if self.ad_span_depth <= 0:
                    self.in_ad_span = False
            return
        if self.in_content and tag == "div":
            self.content_depth -= 1
            if self.content_depth <= 0:
                self.in_content = False
        if tag == "p" and self.in_p:
            self.in_p = False
            text = "".join(self._cur_p).strip()
            if text:
                self.paragraphs.append(text)

    def handle_data(self, data):
        if self.in_headline:
            self.title_parts.append(data)
        elif self.in_p and not self.in_ad_span:
            self._cur_p.append(data)

    @property
    def title(self) -> str:
        return html.unescape("".join(self.title_parts)).strip()


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name.strip()[:120] or "untitled"


def fetch(session: requests.Session, url: str, timeout: float = 20.0) -> requests.Response | None:
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            print(f"  fetch error ({exc}); retrying...", file=sys.stderr)
            time.sleep(2.0 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            return None
        print(f"  HTTP {resp.status_code}; retrying...", file=sys.stderr)
        time.sleep(2.0 * (attempt + 1))
    return None


def scrape(start: int, end: int | None, delay: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    n = start
    while end is None or n <= end:
        dest_glob = list(out_dir.glob(f"{n:04d}_*.txt"))
        if dest_glob:
            print(f"chapter {n}: already downloaded, skipping")
            n += 1
            continue

        url = f"{BASE}/n/{SLUG}/{n}.html"
        print(f"chapter {n}: {url}")
        resp = fetch(session, url)
        if resp is None:
            print(f"chapter {n}: not found, stopping (end of book or gap)")
            break

        parser = ChapterParser()
        parser.feed(resp.text)
        if not parser.paragraphs:
            print(f"chapter {n}: no content found, stopping")
            break

        title = parser.title or f"Chapter {n}"
        dest = out_dir / f"{n:04d}_{sanitize_filename(title)}.txt"
        body = "\n\n".join(parser.paragraphs)
        dest.write_text(f"{title}\n\n{body}\n", encoding="utf-8")
        print(f"  saved: {dest.name} ({len(parser.paragraphs)} paragraphs)")

        if not parser.has_next and end is None:
            print(f"chapter {n}: no next link, this is the last chapter")
            break

        n += 1
        time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None, help="stop after this chapter number (default: until end of book)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to sleep between requests")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    scrape(args.start, args.end, args.delay, args.out)


if __name__ == "__main__":
    main()
