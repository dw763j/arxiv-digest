from __future__ import annotations

from datetime import date, datetime
from typing import Iterable
from urllib.parse import urlencode

import feedparser
import httpx
from loguru import logger

from .models import Paper


ARXIV_API = "https://export.arxiv.org/api/query"


def _build_query(category: str, target_date: date) -> str:
    date_str = target_date.strftime("%Y%m%d")
    date_from = f"{date_str}0000"
    date_to = f"{date_str}2359"
    return f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]"


def _normalize_title(value: str) -> str:
    return " ".join(value.split())


def _parse_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def fetch_papers(
    categories: Iterable[str],
    target_date: date,
    max_results: int = 200,
    timeout: float = 30.0,
) -> list[Paper]:
    headers = {"User-Agent": "arxiv-digest/0.1 (+https://arxiv.org)"}
    results: list[Paper] = []

    with httpx.Client(timeout=timeout, headers=headers) as client:
        for category in categories:
            query = _build_query(category, target_date)
            params = {
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": 0,
                "max_results": max_results,
            }
            url = f"{ARXIV_API}?{urlencode(params)}"
            logger.info("Fetching arXiv feed: {}", url)
            response = client.get(url)
            response.raise_for_status()
            feed = feedparser.parse(response.text)

            for entry in feed.entries:
                paper = Paper(
                    paper_id=entry.id,
                    title=_normalize_title(entry.title),
                    summary=" ".join(entry.summary.split()),
                    authors=[author.name for author in entry.authors],
                    link=entry.link,
                    category=category,
                    published=_parse_datetime(entry.published),
                    updated=_parse_datetime(entry.updated),
                )
                results.append(paper)

    return results
