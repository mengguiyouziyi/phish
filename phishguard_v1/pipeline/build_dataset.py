from __future__ import annotations
import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd
from loguru import logger

from phishguard_v1.features.fetcher import fetch_many
from phishguard_v1.features.parser import extract_from_html
from phishguard_v1.features.url_features import url_stats
from phishguard_v1.features.feature_engineering import (
    compute_feature_dict,
    ensure_feature_columns,
)


@dataclass
class URLGroup:
    label: int
    urls: List[str]
    source: str
    category: str


def read_urls(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
    return lines


def build_record(raw: Dict[str, Any], label: int, source: str, category: str) -> Dict[str, Any]:
    html = raw.get("html", "") or ""
    final_url = raw.get("final_url") or raw.get("request_url") or ""
    url_features = raw.get("url_feats") or url_stats(final_url)
    html_features = extract_from_html(html, final_url) if html else {}

    record: Dict[str, Any] = {
        "request_url": raw.get("request_url", ""),
        "final_url": final_url,
        "status_code": raw.get("status_code", 0) or 0,
        "content_type": raw.get("content_type", ""),
        "bytes": raw.get("bytes", 0) or 0,
        "html": html,
        "label": int(label),
        "source": source,
        "category": category,
        "title": html_features.get("title", ""),
        "headers_json": json.dumps(raw.get("headers") or {}, ensure_ascii=False),
        "cookies_json": json.dumps(raw.get("cookies") or {}, ensure_ascii=False),
        "set_cookie": raw.get("set_cookie", ""),
        "meta_tags_json": json.dumps(html_features.get("meta_kv") or {}, ensure_ascii=False),
        "script_srcs_json": json.dumps(html_features.get("script_srcs") or [], ensure_ascii=False),
        "stylesheets_json": json.dumps(html_features.get("stylesheets") or [], ensure_ascii=False),
        "kw_hits_json": json.dumps(html_features.get("kw_hits") or {}, ensure_ascii=False),
        "lib_hits_json": json.dumps(html_features.get("lib_hits") or {}, ensure_ascii=False),
        "fingerprint_hash": html_features.get("fingerprint_hash", ""),
    }

    # 打平 URL 数值特征
    for key, value in url_features.items():
        record[key] = value

    # 打平 HTML 衍生特征
    for key, value in html_features.items():
        if key not in record:
            record[key] = value

    enriched_source = {**raw}
    enriched_source["url_feats"] = url_features
    enriched_source["html_feats"] = html_features
    enriched_source["html"] = html

    feature_values = compute_feature_dict(enriched_source, html_features)
    record.update(feature_values)
    ensure_feature_columns(record)

    return record


def collect(groups: Iterable[URLGroup], concurrency: int = 8) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for group in groups:
        if not group.urls:
            continue
        logger.info(f"采集中 ({group.category}) - {len(group.urls)} 条 URL")
        results = asyncio.run(fetch_many(group.urls, concurrency=concurrency))
        for raw in results:
            record = build_record(raw, group.label, group.source, group.category)
            records.append(record)
    if not records:
        raise RuntimeError("未采集到任何数据，请检查 URL 列表或网络代理配置。")
    df = pd.DataFrame(records)
    logger.info(f"采集完成，共 {len(df)} 条记录，正样本 {df['label'].sum()} 条")
    return df


def split_dataset(df: pd.DataFrame, output_dir: Path, test_size: float = 0.2, val_size: float = 0.1) -> None:
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=42,
        stratify=df["label"],
    )
    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        random_state=42,
        stratify=temp_df["label"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")
    logger.info(
        "数据集切分完成 -> train:%d / val:%d / test:%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="采集 URL 数据并构建训练集")
    parser.add_argument("--benign", type=Path, help="良性 URL 列表 (txt，每行一个 URL)")
    parser.add_argument("--phish", type=Path, help="钓鱼 URL 列表 (txt，每行一个 URL)")
    parser.add_argument("--output", type=Path, default=Path("data_custom"), help="输出目录")
    parser.add_argument("--name", type=str, default="custom", help="数据来源标记")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-split", action="store_true", help="仅输出 parquet，不切分 train/val/test")
    args = parser.parse_args()

    groups = [
        URLGroup(label=0, urls=read_urls(args.benign), source=args.name, category="benign"),
        URLGroup(label=1, urls=read_urls(args.phish), source=args.name, category="phish"),
    ]

    df = collect(groups, concurrency=args.concurrency)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"dataset_{args.name}.parquet"
    df.to_parquet(raw_path)
    logger.info(f"原始数据已写入 {raw_path}")

    if not args.no_split:
        split_dataset(df, output_dir)


if __name__ == "__main__":
    main()
