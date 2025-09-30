from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

import httpx
from loguru import logger

URL_PATTERN = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)


@dataclass
class Feed:
    name: str
    url: str
    label: int
    limit: int | None = None


async def _fetch_text(url: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.text


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []

    matches = URL_PATTERN.findall(text)
    cleaned = []
    for raw in matches:
        url = raw.strip().strip("'\";,)")
        if url.lower().startswith(("http://", "https://")):
            cleaned.append(url)

    if cleaned:
        return cleaned

    fallback: List[str] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        for part in parts:
            token = part.lower()
            if DOMAIN_PATTERN.match(token):
                fallback.append(f"http://{token}")
    return fallback


async def collect_feeds(feeds: Sequence[Feed], concurrency: int = 5) -> Dict[str, Dict[str, Iterable[str]]]:
    results: Dict[str, Dict[str, Iterable[str]]] = {}
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def _run(feed: Feed):
            async with sem:
                try:
                    logger.info("拉取数据源 %s", feed.name)
                    text = await _fetch_text(feed.url, client)
                    urls = _extract_urls(text)
                    if feed.limit:
                        urls = urls[: feed.limit]
                    results.setdefault(feed.name, {"urls": [], "label": feed.label})
                    results[feed.name]["urls"] = urls
                    logger.info("%s -> %d 条URL", feed.name, len(urls))
                except Exception as exc:
                    logger.warning("拉取 %s 失败: %s", feed.name, exc)
                    results.setdefault(feed.name, {"urls": [], "label": feed.label})

        await asyncio.gather(*[_run(feed) for feed in feeds])

    return results


def merge_results(results: Dict[str, Dict[str, Iterable[str]]]) -> Dict[int, List[str]]:
    merged: Dict[int, Set[str]] = {}
    for info in results.values():
        label = info["label"]
        urls = info.get("urls") or []
        merged.setdefault(label, set()).update(urls)
    return {label: sorted(urls) for label, urls in merged.items()}
