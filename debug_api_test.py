#!/usr/bin/env python3
"""
通过API测试权重计算问题
"""

import requests
import json

def test_api_weight_calculation():
    """通过API测试权重计算"""
    print("🔍 通过API测试权重计算...")

    # 测试URL
    test_url = "https://www.baidu.com/index.php"
    print(f"\n📊 测试URL: {test_url}")

    # 测试API
    api_urls = [
        "http://localhost:8001/predict",
        "http://localhost:8002/predict"
    ]

    for api_url in api_urls:
        try:
            print(f"\n🔧 测试API: {api_url}")
            response = requests.post(api_url, json={"url": test_url}, timeout=10)
            response.raise_for_status()

            result = response.json()
            print(f"✅ API响应成功")
            print(f"   URL模型良性概率: {result.get('url_prob', 'N/A')}")
            print(f"   FusionDNN良性概率: {result.get('fusion_prob', 'N/A')}")
            print(f"   最终良性概率: {result.get('final_prob', 'N/A')}")
            print(f"   预测标签: {'良性' if result.get('label', 0) == 1 else '钓鱼'}")

            # 分析问题
            final_prob = result.get('final_prob', 0)
            if final_prob < 0.5:
                print(f"   ❌ 问题：最终概率为 {final_prob:.4f}，预测为钓鱼")
                print(f"   📝 分析：")
                url_prob = result.get('url_prob', 0)
                fusion_prob = result.get('fusion_prob', 0)
                print(f"      - URL模型: {url_prob:.4f} ({'良性' if url_prob >= 0.5 else '钓鱼'})")
                print(f"      - FusionDNN: {fusion_prob:.4f} ({'良性' if fusion_prob >= 0.5 else '钓鱼'})")

                if url_prob >= 0.5 and fusion_prob < 0.5:
                    print(f"      - 🔍 问题根源：FusionDNN模型预测为钓鱼")
                elif url_prob < 0.5 and fusion_prob >= 0.5:
                    print(f"      - 🔍 问题根源：URL模型预测为钓鱼")
                else:
                    print(f"      - 🔍 问题根源：两个模型都预测为钓鱼")

        except requests.exceptions.ConnectionError:
            print(f"   ❌ API连接失败: {api_url}")
        except Exception as e:
            print(f"   ❌ API测试失败: {e}")

if __name__ == "__main__":
    test_api_weight_calculation()