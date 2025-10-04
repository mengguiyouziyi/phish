#!/usr/bin/env python3
"""
基于真实钓鱼网站构建训练数据集
"""

import asyncio
import httpx
import pandas as pd
import numpy as np
from pathlib import Path
from phishguard_v1.features.fetcher import fetch_one
from phishguard_v1.features.parser import extract_from_html
from phishguard_v1.features.url_features import url_stats
from phishguard_v1.features.feature_engineering import compute_feature_dict, FEATURE_COLUMNS

async def collect_phishing_data():
    """收集钓鱼网站数据"""
    print("🔍 收集真实钓鱼网站训练数据...")

    # 真实钓鱼网站列表
    phishing_urls = [
        "http://wells-fargo-login.com",
        "http://citibank-online.com",
        "http://paypal-verification.net",
        "http://www.paypal-verification.net",
        "http://paypal-verification.org",
        "http://apple-id-verify.com",
        "http://login.apple.com.id-verify.com",
        "http://gmail-security-alert.com",
        "http://facebook.login-secure.net",
        "http://www.facebook.login-secure.net",
        "http://snapchat-verify.com",
        "https://account-verification.net",
        "http://secure.paypal.com.verify-account.com",
        "http://paypa1.com",
        "http://arnazon.com",
        "http://faceb00k.com",
    ]

    # 合法网站
    legitimate_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.wikipedia.org",
        "https://www.stackoverflow.com",
        "https://www.paypal.com",
        "https://www.facebook.com",
        "https://www.linkedin.com",
        "https://www.amazon.com",
        "https://www.instagram.com",
        "https://www.twitter.com",
        "https://www.netflix.com",
        "https://www.youtube.com",
        "https://www.reddit.com",
    ]

    print(f"📊 目标: {len(phishing_urls)} 钓鱼 + {len(legitimate_urls)} 合法")

    async with httpx.AsyncClient(timeout=15.0) as client:
        phishing_data = []
        legitimate_data = []

        # 收集钓鱼网站数据
        print(f"\n🎣 收集钓鱼网站数据...")
        for i, url in enumerate(phishing_urls):
            print(f"  {i+1}/{len(phishing_urls)}: {url}")
            try:
                item = await fetch_one(url.strip(), client)
                html_feats = extract_from_html(
                    item.get("html", ""),
                    item.get("final_url") or item.get("request_url")
                )
                item["html_feats"] = html_feats
                item["label"] = 1
                phishing_data.append(item)
                print(f"    ✅ 成功")
            except Exception as e:
                print(f"    ❌ 失败: {e}")

        # 收集合法网站数据
        print(f"\n✅ 收集合法网站数据...")
        for i, url in enumerate(legitimate_urls):
            print(f"  {i+1}/{len(legitimate_urls)}: {url}")
            try:
                item = await fetch_one(url.strip(), client)
                html_feats = extract_from_html(
                    item.get("html", ""),
                    item.get("final_url") or item.get("request_url")
                )
                item["html_feats"] = html_feats
                item["label"] = 0
                legitimate_data.append(item)
                print(f"    ✅ 成功")
            except Exception as e:
                print(f"    ❌ 失败: {e}")

        print(f"\n📊 数据收集完成:")
        print(f"  钓鱼网站: {len(phishing_data)} 个")
        print(f"  合法网站: {len(legitimate_data)} 个")

        return phishing_data, legitimate_data

def create_training_dataset(phishing_data, legitimate_data):
    """创建训练数据集"""
    print(f"\n🔧 创建训练数据集...")

    # 合并数据
    all_data = phishing_data + legitimate_data

    # 提取特征
    features_list = []
    labels_list = []

    for item in all_data:
        try:
            # 计算完整特征
            feature_dict = compute_feature_dict(item)

            # 确保所有特征都存在
            feature_vector = [feature_dict.get(col, 0.0) for col in FEATURE_COLUMNS]

            features_list.append(feature_vector)
            labels_list.append(item["label"])
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            continue

    # 转换为DataFrame
    feature_df = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    label_df = pd.DataFrame({"label": labels_list})

    print(f"✅ 特征矩阵形状: {feature_df.shape}")
    print(f"📊 标签分布: {label_df['label'].value_counts().to_dict()}")

    return feature_df, label_df

def augment_dataset(features_df, labels_df, target_size=1000):
    """数据增强以增加训练数据"""
    print(f"\n🔄 数据增强 (目标大小: {target_size})...")

    current_size = len(features_df)
    if current_size >= target_size:
        print(f"✅ 数据量已足够: {current_size}")
        return features_df, labels_df

    # 计算需要增加的样本数
    samples_needed = target_size - current_size
    print(f"📈 需要增加 {samples_needed} 个样本")

    augmented_features = []
    augmented_labels = []

    while len(augmented_features) < samples_needed:
        # 随机选择一个现有样本进行增强
        idx = np.random.randint(0, current_size)
        original_features = features_df.iloc[idx].values
        original_label = labels_df.iloc[idx]['label']

        # 创建增强样本
        noise_level = 0.1 if original_label == 0 else 0.15  # 钓鱼网站增加更多噪声

        # 添加高斯噪声
        noise = np.random.normal(0, noise_level, original_features.shape)
        noisy_features = original_features + noise

        # 确保特征值合理
        noisy_features = np.maximum(noisy_features, 0)  # 非负值

        # 限制某些特征的合理范围
        for i, col in enumerate(FEATURE_COLUMNS):
            if col.endswith("_len"):  # 长度特征
                noisy_features[i] = min(noisy_features[i], 1000)
            elif col.endswith("_ratio"):  # 比率特征
                noisy_features[i] = np.clip(noisy_features[i], 0, 1)
            elif col in ["status_code"]:  # 状态码
                noisy_features[i] = np.clip(noisy_features[i], 100, 600)

        augmented_features.append(noisy_features)
        augmented_labels.append(original_label)

    # 创建增强数据
    augmented_features_df = pd.DataFrame(augmented_features, columns=FEATURE_COLUMNS)
    augmented_labels_df = pd.DataFrame({"label": augmented_labels})

    # 合并原始数据和增强数据
    final_features = pd.concat([features_df, augmented_features_df], ignore_index=True)
    final_labels = pd.concat([labels_df, augmented_labels_df], ignore_index=True)

    print(f"✅ 增强后数据集大小: {len(final_features)}")
    print(f"📊 最终标签分布: {final_labels['label'].value_counts().to_dict()}")

    return final_features, final_labels

def save_dataset(features_df, labels_df, name="real_phishing_dataset"):
    """保存数据集"""
    print(f"\n💾 保存数据集...")

    # 合并特征和标签
    dataset_df = pd.concat([features_df, labels_df], axis=1)

    # 保存完整数据集
    dataset_path = Path(f"data_{name}/dataset.parquet")
    dataset_path.parent.mkdir(exist_ok=True)
    dataset_df.to_parquet(dataset_path, index=False)
    print(f"✅ 完整数据集已保存: {dataset_path}")

    # 分割并保存训练/验证/测试集
    # 打乱数据
    dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 分割
    total_size = len(dataset_df)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    train_df = dataset_df[:train_size]
    val_df = dataset_df[train_size:train_size + val_size]
    test_df = dataset_df[train_size + val_size:]

    # 保存分割后的数据
    train_path = dataset_path.parent / "train.parquet"
    val_path = dataset_path.parent / "val.parquet"
    test_path = dataset_path.parent / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"✅ 训练集: {train_path} ({len(train_df)} 样本)")
    print(f"✅ 验证集: {val_path} ({len(val_df)} 样本)")
    print(f"✅ 测试集: {test_path} ({len(test_df)} 样本)")

    return dataset_path.parent

async def main():
    """主函数"""
    print("🚀 构建基于真实钓鱼网站的训练数据集")
    print("=" * 60)

    # 收集数据
    phishing_data, legitimate_data = await collect_phishing_data()

    if len(phishing_data) == 0 or len(legitimate_data) == 0:
        print("❌ 数据收集失败，无法构建数据集")
        return None

    # 创建数据集
    features_df, labels_df = create_training_dataset(phishing_data, legitimate_data)

    # 数据增强
    augmented_features, augmented_labels = augment_dataset(features_df, labels_df, target_size=1000)

    # 保存数据集
    dataset_dir = save_dataset(augmented_features, augmented_labels, "real_phishing_v2")

    print(f"\n🎉 真实钓鱼网站数据集构建完成!")
    print(f"📁 数据集目录: {dataset_dir}")
    print(f"🔥 现在可以训练更强模型了!")

    return dataset_dir

if __name__ == "__main__":
    asyncio.run(main())