"""Goodreads public-profile scraper for BookMind taste import.

Goodreads shut down its official API in 2020, but public "read" shelves are
still HTML-scrapable and (sometimes) available as RSS. This module tries RSS
first for speed, then falls back to HTML scraping.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15


def _headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _resolve_user_id(user_input: str) -> str | None:
    """Turn whatever the user typed into a numeric Goodreads user id."""
    user_input = user_input.strip()
    if not user_input:
        return None

    if user_input.isdigit():
        return user_input

    if "goodreads.com" in user_input:
        parsed = urlparse(user_input)
        m = re.search(r"/user/show/(\d+)", parsed.path)
        if m:
            return m.group(1)
        m = re.search(r"/user/show/([A-Za-z0-9._-]+)", parsed.path)
        if m:
            user_input = m.group(1)

    url = f"https://www.goodreads.com/user/show/{quote(user_input)}"
    try:
        r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200:
            m = re.search(r"/user/show/(\d+)", r.url)
            if m:
                return m.group(1)
            m = re.search(r'data-user-id="(\d+)"', r.text)
            if m:
                return m.group(1)
            m = re.search(r"/user/show/(\d+)", r.text)
            if m:
                return m.group(1)
    except requests.RequestException:
        pass
    return None


def _parse_html_shelf(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.bookalike.review")
    out: list[dict[str, Any]] = []
    for row in rows:
        title_el = row.select_one("td.field.title a")
        title = (title_el.get("title") or title_el.get_text(strip=True)) if title_el else None

        author_el = row.select_one("td.field.author a")
        author = author_el.get_text(strip=True) if author_el else None
        if author and "," in author:
            last, first = (p.strip() for p in author.split(",", 1))
            author = f"{first} {last}".strip()

        rating = 0
        rating_el = row.select_one("td.field.rating .staticStars")
        if rating_el:
            title_attr = rating_el.get("title") or ""
            rating_map = {
                "did not like it": 1,
                "it was ok": 2,
                "liked it": 3,
                "really liked it": 4,
                "it was amazing": 5,
            }
            rating = rating_map.get(title_attr.strip().lower(), 0)
            if rating == 0:
                rating = len(rating_el.select("span.staticStar.p10"))

        if title:
            out.append(
                {
                    "title": title,
                    "author": author or "",
                    "rating": int(rating),
                    "shelf": "read",
                }
            )
    return out


def _parse_rss_shelf(xml_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    out: list[dict[str, Any]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        author = (item.findtext("author_name") or "").strip()
        rating_s = (item.findtext("user_rating") or "0").strip()
        try:
            rating = int(rating_s)
        except ValueError:
            rating = 0
        if title:
            out.append(
                {
                    "title": title,
                    "author": author,
                    "rating": rating,
                    "shelf": "read",
                }
            )
    return out


def fetch_read_shelf(
    user_input: str,
    shelf: str = "read",
    max_books: int = 100,
) -> list[dict[str, Any]]:
    """Return the user's books on the given shelf (default 'read').

    Tries RSS first (cheapest), falls back to HTML scraping. If the profile
    is private or doesn't exist, returns an empty list.
    """
    user_id = _resolve_user_id(user_input)
    if not user_id:
        return []

    rss_url = f"https://www.goodreads.com/review/list_rss/{user_id}?shelf={shelf}"
    try:
        r = requests.get(rss_url, headers=_headers(), timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and "<item" in r.text:
            books = _parse_rss_shelf(r.text)
            if books:
                return books[:max_books]
    except requests.RequestException:
        pass

    html_url = (
        f"https://www.goodreads.com/review/list/{user_id}"
        f"?shelf={shelf}&per_page={max_books}&sort=date_read&order=d"
    )
    try:
        r = requests.get(html_url, headers=_headers(), timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            books = _parse_html_shelf(r.text)
            return books[:max_books]
    except requests.RequestException:
        pass

    return []
