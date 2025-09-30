#!/usr/bin/env python3
"""
验证GitHub收集的URL并提取特征，构建高质量训练数据集
"""

import asyncio
import httpx
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json
import time
import sys
sys.path.append('.')

from phishguard_v1.features.fetcher import fetch_one
from phishguard_v1.config import settings

async def validate_and_extract_features(urls: List[str], label: int, batch_size: int = 10) -> List[Dict[str, Any]]:
    """验证URL并提取特征"""
    results = []

    # 配置会话
    timeout = httpx.Timeout(30.0)
    headers = {"User-Agent": settings.user_agent}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1} ({len(batch)} 个URL)")

            tasks = []
            for url in batch:
                task = process_single_url(client, url, label)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"  ❌ 处理失败: {result}")
                elif result:
                    results.append(result)
                    status = "✅" if result['success'] else "❌"
                    print(f"  {status} {result['url']} - 状态: {result.get('status_code', 'N/A')}")

            # 批次间延迟，避免请求过快
            if i + batch_size < len(urls):
                await asyncio.sleep(2)

    return results

async def process_single_url(client: httpx.AsyncClient, url: str, label: int) -> Dict[str, Any]:
    """处理单个URL"""
    try:
        # 使用我们的fetcher获取网页内容
        item = await fetch_one(url, client)

        if not item or not item.get('ok', False):
            return {
                'url': url,
                'label': label,
                'success': False,
                'error': item.get('meta', {}).get('error', '抓取失败') if item else '抓取失败',
                'status_code': item.get('status_code') if item else None
            }

        # 特征已经在fetch_one中提取完成
        url_feats = item.get('url_feats', {})
        html_feats = item.get('html_feats', {})

        # 合并特征
        features = {}
        features.update(url_feats)
        features.update(html_feats)
        features.update({
            'status_code': item.get('status_code'),
            'content_type': item.get('content_type'),
            'bytes': item.get('bytes', 0)
        })

        # 合并所有特征
        result = {
            'url': url,
            'final_url': item.get('final_url', url),
            'label': label,
            'success': True,
            'status_code': item.get('status_code'),
            'content_type': item.get('content_type'),
            'bytes': item.get('bytes', 0),
            'features': features,
            'timestamp': time.time()
        }

        return result

    except Exception as e:
        return {
            'url': url,
            'label': label,
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }

def load_github_data() -> tuple[List[str], List[str]]:
    """加载GitHub收集的数据"""
    github_dir = Path("github_data")

    # 读取钓鱼网站URL
    phishing_urls = []
    if (github_dir / "phishing_urls.txt").exists():
        with open(github_dir / "phishing_urls.txt", "r", encoding="utf-8") as f:
            phishing_urls = [line.strip() for line in f if line.strip()]

    # 读取良性网站URL
    benign_urls = []
    if (github_dir / "benign_urls.txt").exists():
        with open(github_dir / "benign_urls.txt", "r", encoding="utf-8") as f:
            benign_urls = [line.strip() for line in f if line.strip()]

    return phishing_urls, benign_urls

def save_validated_data(phishing_results: List[Dict], benign_results: List[Dict]):
    """保存验证后的数据"""
    output_dir = Path("validated_github_data")
    output_dir.mkdir(exist_ok=True)

    # 合并所有结果
    all_results = phishing_results + benign_results

    # 转换为DataFrame
    df = pd.DataFrame(all_results)

    # 保存完整数据
    df.to_parquet(output_dir / "validated_github_data.parquet", index=False)

    # 分别保存成功和失败的数据
    success_df = df[df['success'] == True]
    failed_df = df[df['success'] == False]

    success_df.to_parquet(output_dir / "successful_urls.parquet", index=False)
    failed_df.to_parquet(output_dir / "failed_urls.parquet", index=False)

    # 保存训练数据格式
    if not success_df.empty:
        training_data = []
        for _, row in success_df.iterrows():
            features = row.get('features', {})
            training_row = {
                'url': row['url'],
                'final_url': row['final_url'],
                'label': row['label'],
                'timestamp': row['timestamp']
            }
            # 添加所有特征
            training_row.update(features)
            training_data.append(training_row)

        training_df = pd.DataFrame(training_data)
        training_df.to_parquet(output_dir / "training_data.parquet", index=False)

    # 保存统计信息
    stats = {
        'total_phishing': len(phishing_results),
        'successful_phishing': len([r for r in phishing_results if r['success']]),
        'total_benign': len(benign_results),
        'successful_benign': len([r for r in benign_results if r['success']]),
        'total_urls': len(all_results),
        'successful_urls': len(success_df),
        'failed_urls': len(failed_df),
        'success_rate': len(success_df) / len(all_results) * 100 if all_results else 0
    }

    with open(output_dir / "validation_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n📊 验证统计:")
    print(f"  总URL数: {stats['total_urls']}")
    print(f"  成功验证: {stats['successful_urls']}")
    print(f"  验证失败: {stats['failed_urls']}")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    print(f"  钓鱼网站: {stats['successful_phishing']}/{stats['total_phishing']}")
    print(f"  良性网站: {stats['successful_benign']}/{stats['total_benign']}")

    return output_dir

async def main():
    """主函数"""
    print("🔍 开始验证GitHub收集的URL并提取特征")
    print("=" * 60)

    # 加载数据
    phishing_urls, benign_urls = load_github_data()

    print(f"📦 加载数据:")
    print(f"  钓鱼网站: {len(phishing_urls)} 个")
    print(f"  良性网站: {len(benign_urls)} 个")

    # 验证钓鱼网站
    print(f"\n🎯 验证钓鱼网站...")
    phishing_results = await validate_and_extract_features(phishing_urls, label=1)

    # 验证良性网站
    print(f"\n✅ 验证良性网站...")
    benign_results = await validate_and_extract_features(benign_urls, label=0)

    # 保存数据
    print(f"\n💾 保存验证后的数据...")
    output_dir = save_validated_data(phishing_results, benign_results)

    print(f"\n🎉 验证完成! 数据已保存到: {output_dir}/")
    print("文件列表:")
    print("  - validated_github_data.parquet: 完整验证数据")
    print("  - successful_urls.parquet: 成功验证的URL")
    print("  - failed_urls.parquet: 验证失败的URL")
    print("  - training_data.parquet: 训练数据格式")
    print("  - validation_stats.json: 验证统计信息")

if __name__ == "__main__":
    asyncio.run(main())