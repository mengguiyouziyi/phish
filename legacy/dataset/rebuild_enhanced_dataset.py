#!/usr/bin/env python3
"""
重建增强数据集
使用现有数据但添加新的增强特征
"""

import pandas as pd
import asyncio
from pathlib import Path
from phishguard_v1.features.parser import extract_from_html
from phishguard_v1.features.url_features import url_stats
from httpx import AsyncClient
from typing import Dict, Any
import numpy as np

def reprocess_existing_data():
    """重新处理现有数据，添加增强特征"""
    print("🔄 重新处理现有数据...")

    # 加载现有数据
    data_path = Path("data")
    if not data_path.exists():
        print("❌ 现有数据目录不存在")
        return

    dfs = []
    for file in ["train.parquet", "val.parquet", "test.parquet"]:
        file_path = data_path / file
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)

    if not dfs:
        print("❌ 没有找到现有数据")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 加载了 {len(combined_df)} 条现有记录")

    # 重新处理每条记录
    enhanced_records = []

    for idx, row in combined_df.iterrows():
        if idx % 100 == 0:
            print(f"处理中: {idx}/{len(combined_df)}")

        # 基础数据
        record = {
            "request_url": row.get("request_url", ""),
            "final_url": row.get("final_url", row.get("request_url", "")),
            "status_code": row.get("status_code", 200),
            "content_type": row.get("content_type", ""),
            "bytes": row.get("bytes", 0),
            "html": row.get("html", ""),
            "label": row.get("label", 0),
            "source": row.get("source", "existing"),
            "category": row.get("category", "unknown"),
        }

        # 添加现有的URL特征
        for col in ["url_len", "host_len", "path_len", "num_digits", "num_letters", "num_specials",
                   "num_dots", "num_hyphen", "num_slash", "num_qm", "num_at", "num_pct",
                   "has_ip", "subdomain_depth", "tld_suspicious", "has_punycode", "scheme_https",
                   "query_len", "fragment_len"]:
            record[col] = row.get(col, 0)

        # 添加现有的HTML特征
        for col in ["has_html", "title_len", "num_meta", "num_links", "num_stylesheets",
                   "num_scripts", "num_script_ext", "num_script_inline", "num_iframes",
                   "num_forms", "has_password_input", "has_email_input", "suspicious_js_inline"]:
            record[col] = row.get(col, 0)

        # 如果有HTML内容，重新提取增强特征
        html_content = row.get("html", "")
        final_url = record["final_url"]

        if html_content and final_url:
            try:
                # 使用增强的解析器重新提取特征
                html_feats = extract_from_html(html_content, final_url)

                # 添加增强特征
                record["external_form_actions"] = html_feats.get("external_form_actions", 0)
                record["num_hidden_inputs"] = html_feats.get("num_hidden_inputs", 0)
                record["external_links"] = html_feats.get("external_links", 0)
                record["internal_links"] = html_feats.get("internal_links", 0)
                record["external_images"] = html_feats.get("external_images", 0)
                record["is_subdomain"] = html_feats.get("is_subdomain", 0)
                record["has_www"] = html_feats.get("has_www", 0)
                record["is_common_tld"] = html_feats.get("is_common_tld", 0)

            except Exception as e:
                print(f"处理记录 {idx} 时出错: {e}")
                # 使用默认值
                record["external_form_actions"] = 0
                record["num_hidden_inputs"] = 0
                record["external_links"] = 0
                record["internal_links"] = 0
                record["external_images"] = 0
                record["is_subdomain"] = 0
                record["has_www"] = 0
                record["is_common_tld"] = 0
        else:
            # 没有HTML内容，使用默认值
            record["external_form_actions"] = 0
            record["num_hidden_inputs"] = 0
            record["external_links"] = 0
            record["internal_links"] = 0
            record["external_images"] = 0
            record["is_subdomain"] = 0
            record["has_www"] = 0
            record["is_common_tld"] = 0

        enhanced_records.append(record)

    # 创建新的DataFrame
    enhanced_df = pd.DataFrame(enhanced_records)
    print(f"✅ 重新处理完成，共 {len(enhanced_df)} 条记录")

    # 数据统计
    benign_count = len(enhanced_df[enhanced_df["label"] == 0])
    phishing_count = len(enhanced_df[enhanced_df["label"] == 1])

    print(f"📊 数据统计:")
    print(f"  良性网站: {benign_count} ({benign_count/len(enhanced_df)*100:.1f}%)")
    print(f"  钓鱼网站: {phishing_count} ({phishing_count/len(enhanced_df)*100:.1f}%)")

    # 保存数据
    output_dir = Path("data_enhanced")
    output_dir.mkdir(exist_ok=True)

    # 分割数据集
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(enhanced_df, test_size=0.3, random_state=42, stratify=enhanced_df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")

    print(f"✅ 数据已保存到 {output_dir}/")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    return enhanced_df

if __name__ == "__main__":
    reprocess_existing_data()