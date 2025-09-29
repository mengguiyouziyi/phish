#!/usr/bin/env python3
"""
增强数据集构建脚本
收集更多高质量、多样化的训练数据
"""

import asyncio
import pandas as pd
from typing import List, Dict, Any
from phishguard_v1.features.fetcher import fetch_one
from phishguard_v1.features.parser import extract_from_html
from phishguard_v1.features.url_features import url_stats
from httpx import AsyncClient
import numpy as np
from pathlib import Path

# 高质量良性网站来源
BENIGN_SOURCES = {
    "知名科技公司": [
        "https://www.google.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.youtube.com",
        "https://www.instagram.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.airbnb.com",
    ],
    "中国知名网站": [
        "https://www.baidu.com",
        "https://www.alibaba.com",
        "https://www.tmall.com",
        "https://www.qq.com",
        "https://www.weibo.com",
        "https://www.zhihu.com",
        "https://www.douban.com",
        "https://www.taobao.com",
        "https://www.jd.com",
        "https://www.meituan.com",
        "https://www.didi.com",
        "https://www.bytedance.com",
        "https://www.xiaohongshu.com",
    ],
    "政府机构": [
        "https://www.gov.cn",
        "https://www.whitehouse.gov",
        "https://www.europa.eu",
        "https://www.parliament.uk",
        "https://www.canada.ca",
    ],
    "金融机构": [
        "https://www.bankofamerica.com",
        "https://www.chase.com",
        "https://www.wellsfargo.com",
        "https://www.citibank.com",
        "https://www.hsbc.com",
        "https://www.icbc.com.cn",
        "https://www.bankofchina.com",
    ]
}

# 钓鱼网站来源（从公开数据源获取）
PHISH_SOURCES = [
    # 可以从这些来源获取更多钓鱼网站数据
    "https://openphish.com/",
    "https://phishtank.org/",
    "https://urlhaus.abuse.ch/browse/",
]

async def collect_benign_data(urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """收集良性网站数据"""
    results = []

    async with AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    ) as client:

        # 分批处理，避免过多并发
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i:i + max_concurrent]
            tasks = [fetch_single_url(url, client) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict) and result.get("ok"):
                    results.append(result)
                elif isinstance(result, Exception):
                    print(f"抓取失败: {result}")

            print(f"已完成 {min(i + max_concurrent, len(urls))}/{len(urls)} 个良性网站")
            await asyncio.sleep(1)  # 避免请求过快

    return results

async def fetch_single_url(url: str, client: AsyncClient) -> Dict[str, Any]:
    """抓取单个URL并提取特征"""
    try:
        # 抓取网站
        item = await fetch_one(url, client)

        if not item or not item.get("ok"):
            return {"url": url, "error": "抓取失败", "ok": False}

        # 提取HTML特征
        html_content = item.get("html", "")
        final_url = item.get("final_url") or url
        item["html_feats"] = extract_from_html(html_content, final_url)

        # 确保有URL特征
        if "url_feats" not in item or not item["url_feats"]:
            item["url_feats"] = url_stats(final_url)

        # 添加标签 (0 = 良性)
        item["label"] = 0
        item["source"] = "benign"
        item["category"] = "trusted"

        return item

    except Exception as e:
        return {"url": url, "error": str(e), "ok": False}

def load_existing_data() -> pd.DataFrame:
    """加载现有数据"""
    data_path = Path("data")
    if not data_path.exists():
        return pd.DataFrame()

    dfs = []
    for file in ["train.parquet", "test.parquet", "val.parquet"]:
        file_path = data_path / file
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

async def main():
    """主函数"""
    print("🚀 开始收集增强训练数据...")

    # 加载现有数据
    existing_df = load_existing_data()
    print(f"现有数据: {len(existing_df)} 条记录")

    # 收集良性数据
    all_benign_urls = []
    for category, urls in BENIGN_SOURCES.items():
        print(f"准备收集 {category} 类别的网站...")
        all_benign_urls.extend(urls)

    print(f"总共需要收集 {len(all_benign_urls)} 个良性网站")

    # 收集新数据
    benign_data = await collect_benign_data(all_benign_urls)
    print(f"成功收集 {len(benign_data)} 个良性网站")

    # 转换为DataFrame
    new_data = []
    for item in benign_data:
        if item.get("ok"):
            row = {
                "request_url": item.get("request_url"),
                "final_url": item.get("final_url"),
                "status_code": item.get("status_code"),
                "content_type": item.get("content_type"),
                "bytes": item.get("bytes"),
                "html": item.get("html", "")[:1000],  # 只保存部分HTML以节省空间
                "label": item.get("label", 0),
                "source": item.get("source", "unknown"),
                "category": item.get("category", "unknown"),
            }

            # 添加URL特征
            url_feats = item.get("url_feats", {})
            for k, v in url_feats.items():
                row[k] = v

            # 添加HTML特征
            html_feats = item.get("html_feats", {})
            for k, v in html_feats.items():
                row[k] = v

            new_data.append(row)

    new_df = pd.DataFrame(new_data)
    print(f"新数据转换完成: {len(new_df)} 条有效记录")

    # 合并数据
    if not existing_df.empty:
        # 去重 - 基于final_url
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["final_url"], keep="last")
    else:
        combined_df = new_df

    print(f"合并后数据: {len(combined_df)} 条记录")

    # 数据统计
    benign_count = len(combined_df[combined_df["label"] == 0])
    phishing_count = len(combined_df[combined_df["label"] == 1])

    print(f"良性网站: {benign_count} ({benign_count/len(combined_df)*100:.1f}%)")
    print(f"钓鱼网站: {phishing_count} ({phishing_count/len(combined_df)*100:.1f}%)")

    # 保存数据
    output_dir = Path("data_enhanced")
    output_dir.mkdir(exist_ok=True)

    # 分割数据集
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42, stratify=combined_df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")

    print(f"✅ 数据已保存到 {output_dir}/")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    # 生成数据质量报告
    generate_data_report(combined_df, output_dir)

def generate_data_report(df: pd.DataFrame, output_dir: Path):
    """生成数据质量报告"""
    report = f"""
# 增强数据集质量报告

## 数据统计
- 总样本数: {len(df)}
- 特征数量: {len([col for col in df.columns if col not in ['request_url', 'final_url', 'html', 'source', 'category', 'label']])}
- 数据大小: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

## 标签分布
- 良性网站: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)
- 钓鱼网站: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)

## 来源分布
"""

    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            report += f"- {source}: {count} ({count/len(df)*100:.1f}%)\n"

    # 保存报告
    with open(output_dir / "data_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📊 数据质量报告已保存: {output_dir}/data_report.md")

if __name__ == "__main__":
    asyncio.run(main())