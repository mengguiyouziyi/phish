#!/usr/bin/env python3
"""
测试新训练的模型在UI中的兼容性
"""

import asyncio
import httpx
from phishguard_v1.features.fetcher import fetch_one
from phishguard_v1.features.parser import extract_from_html
from phishguard_v1.models.inference import InferencePipeline

async def test_new_model_ui():
    """测试新模型在UI中的兼容性"""
    print("🔍 测试新模型UI兼容性...")

    try:
        # 加载新模型
        pipeline = InferencePipeline(
            fusion_ckpt_path="artifacts/real_phishing_advanced_20251001_204447.pt",
            enable_fusion=True
        )

        print(f"✅ 新模型加载成功!")
        print(f"🧠 特征数量: {len(pipeline.fusion_feature_names)}")
        print(f"🏗️ 模型架构: {type(pipeline.fusion).__name__}")

    except Exception as e:
        print(f"❌ 新模型加载失败: {e}")
        return False

    # 测试真实钓鱼网站
    test_cases = [
        ("https://www.google.com", "合法"),
        ("http://wells-fargo-login.com", "钓鱼"),
        ("http://paypal-verification.net", "钓鱼"),
        ("http://apple-id-verify.com", "钓鱼"),
        ("http://paypa1.com", "钓鱼"),
    ]

    print(f"\n📊 测试 {len(test_cases)} 个URL...")

    async with httpx.AsyncClient(timeout=15.0) as client:
        results = []

        for url, expected_type in test_cases:
            print(f"\n🔗 测试: {url} (期望: {expected_type})")

            try:
                # 获取数据
                item = await fetch_one(url.strip(), client)
                html_feats = extract_from_html(
                    item.get("html", ""),
                    item.get("final_url") or item.get("request_url")
                )
                item["html_feats"] = html_feats

                # 预测
                pred = pipeline.predict(item)

                actual_type = "钓鱼" if pred['label'] == 1 else "合法"
                is_correct = expected_type == actual_type

                print(f"  ✅ 预测: {actual_type} ({pred['final_prob']:.1%})")
                print(f"  🎯 决策: {pred['details']['decision']}")

                if is_correct:
                    print(f"  🎉 正确!")
                else:
                    print(f"  ❌ 错误! 期望: {expected_type}")

                results.append(is_correct)

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                results.append(False)

    # 统计结果
    correct = sum(results)
    total = len(results)
    accuracy = correct / total

    print(f"\n📊 测试结果:")
    print(f"✅ 准确率: {accuracy:.1%} ({correct}/{total})")

    if accuracy >= 0.8:
        print(f"🎉 新模型UI兼容性测试通过!")
        return True
    else:
        print(f"⚠️ 新模型UI兼容性需要进一步调试")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_new_model_ui())
    if success:
        print(f"\n✅ 可以重新部署到9005端口!")
    else:
        print(f"\n❌ 需要进一步调试模型兼容性")