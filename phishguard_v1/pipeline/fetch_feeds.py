from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from loguru import logger

from phishguard_v1.pipeline.url_feeds import Feed, collect_feeds, merge_results

DEFAULT_PHISH_FEEDS: List[Feed] = [
    Feed("PhishTank", "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt", 1, limit=5000),
    Feed("OpenPhish", "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-OPENPHISH.txt", 1, limit=5000),
    Feed("URLHaus", "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-URLHAUS.txt", 1, limit=5000),
]

DEFAULT_BENIGN_FEEDS: List[Feed] = [
    Feed(
        "OpenDNS",
        "https://raw.githubusercontent.com/opendns/public-domain-lists/master/opendns-top-domains.txt",
        0,
        limit=12000,
    ),
    Feed(
        "TrancoBackup",
        "https://tranco-list.eu/top-1m.csv",
        0,
        limit=8000,
    ),
]


def write_urls(urls: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for url in urls:
            f.write(f"{url}\n")
    logger.info("写入 %d 条 URL -> %s", len(urls), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取远程钓鱼/良性URL列表")
    parser.add_argument("--phish-out", type=Path, default=Path("data/phish_auto.txt"))
    parser.add_argument("--benign-out", type=Path, default=Path("data/benign_auto.txt"))
    parser.add_argument("--phish-limit", type=int, default=8000)
    parser.add_argument("--benign-limit", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    phish_feeds = [feed for feed in DEFAULT_PHISH_FEEDS]
    benign_feeds = [feed for feed in DEFAULT_BENIGN_FEEDS]

    # 覆盖每个feed的limit
    for feed in phish_feeds:
        feed.limit = min(feed.limit or args.phish_limit, args.phish_limit)
    for feed in benign_feeds:
        feed.limit = min(feed.limit or args.benign_limit, args.benign_limit)

    async def _run():
        phish_result = await collect_feeds(phish_feeds, concurrency=args.concurrency)
        benign_result = await collect_feeds(benign_feeds, concurrency=args.concurrency)
        merged = merge_results({**phish_result, **benign_result})
        write_urls(list(merged.get(1, []))[: args.phish_limit], args.phish_out)
        write_urls(list(merged.get(0, []))[: args.benign_limit], args.benign_out)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
