#!/usr/bin/env python3
"""
下载URL模型到本地
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def download_url_model():
    """下载URL模型到本地"""
    print("🔍 开始下载URL模型...")

    model_id = "imanoop7/bert-phishing-detector"
    local_path = "artifacts/url_model"

    # 创建本地目录
    os.makedirs(local_path, exist_ok=True)

    try:
        print(f"📥 下载模型: {model_id}")
        print(f"📁 保存到: {local_path}")

        # 下载tokenizer和model
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)

        # 保存到本地
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)

        print("✅ URL模型下载完成!")

        # 测试本地模型
        print("🧪 测试本地模型...")
        test_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        test_model = AutoModelForSequenceClassification.from_pretrained(local_path, trust_remote_code=True)

        test_url = "https://www.baidu.com"
        inputs = test_tokenizer(test_url, truncation=True, max_length=256, return_tensors="pt")

        with torch.no_grad():
            outputs = test_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            phishing_prob = probs[0, -1].item()

        print(f"📊 测试URL: {test_url}")
        print(f"🎯 钓鱼概率: {phishing_prob:.4f}")
        print(f"🎯 良性概率: {1-phishing_prob:.4f}")

        # 检查文件
        files = os.listdir(local_path)
        print(f"📁 下载的文件: {files}")

        return True

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_url_model()
    if success:
        print("\n🎉 URL模型已成功下载到本地!")
        print("📂 模型位置: artifacts/url_model/")
    else:
        print("\n❌ URL模型下载失败!")