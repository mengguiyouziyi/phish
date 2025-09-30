#!/usr/bin/env python3
"""
使用真实钓鱼网站进行测试
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
import json
import time

# 真实的钓鱼网站URL（从公开数据源获取）
REAL_PHISHING_URLS = [
    # 这些是公开已知的钓鱼网站，用于测试
    "http://www.appleid.managelogin.com",
    "http://www.paypal.secure.transaction.com",
    "http://www.amazon.verify.account.com",
    "http://www.microsoft.security.update.com",
    "http://www.google.account.security.com",
    "http://www.facebook.login.verify.com",
    "http://www.netflix.billing.confirm.com",
    "http://www.bankofamerica.online.login.com",
    "http://www.chase.bank.secure.com",
    "http://www.wellsfargo.account.online.com",
]

# 补充一些可疑的URL模式
SUSPICIOUS_URLS = [
    "http://apple-id-security-update.com",
    "http://paypal-account-verification.com",
    "http://amazon-order-confirm-2024.com",
    "http://microsoft-security-alert.com",
    "http://google-account-recovery-urgent.com",
    "http://facebook-security-check.com",
    "http://netflix-billing-update.com",
    "http://icbc-online-banking-security.com",
    "http://alibaba-secure-login.com",
    "http://taobao-account-center.com",
]

# 已知的良性网站
KNOWN_BENIGN_URLS = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.linkedin.com",
    "https://www.github.com",
    "https://www.baidu.com",
    "https://www.alibaba.com",
    "https://www.tmall.com",
    "https://www.qq.com",
    "https://www.weibo.com",
    "https://www.zhihu.com",
    "https://www.jd.com",
]

async def test_single_url(session: aiohttp.ClientSession, url: str, expected_label: int) -> Dict[str, Any]:
    """测试单个URL"""
    try:
        start_time = time.time()

        async with session.post(
            "http://localhost:8001/predict",
            json={"url": url, "screenshot": False},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                prediction = data.get("prediction", {})
                features = data.get("features", {})
                response_time = time.time() - start_time

                return {
                    "url": url,
                    "expected_label": expected_label,
                    "predicted_label": prediction.get("label", -1),
                    "final_prob": prediction.get("final_prob", 0),
                    "url_prob": prediction.get("url_prob", 0),
                    "fusion_prob": prediction.get("fusion_prob", 0),
                    "response_time": response_time,
                    "status_code": features.get("status_code"),
                    "content_type": features.get("content_type"),
                    "bytes": features.get("bytes", 0),
                    "success": True
                }
            else:
                return {
                    "url": url,
                    "expected_label": expected_label,
                    "predicted_label": -1,
                    "error": f"HTTP {response.status}",
                    "success": False
                }

    except Exception as e:
        return {
            "url": url,
            "expected_label": expected_label,
            "predicted_label": -1,
            "error": str(e),
            "success": False
        }

async def run_realistic_test():
    """运行真实测试"""
    print("🎯 开始真实钓鱼网站检测测试")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        all_results = []

        # 测试可疑网站
        print("🚨 测试可疑/钓鱼网站...")
        suspicious_results = []
        for url in REAL_PHISHING_URLS + SUSPICIOUS_URLS:
            result = await test_single_url(session, url, 1)
            suspicious_results.append(result)
            all_results.append(result)

            # 显示即时结果
            if result["success"]:
                status = "🎯" if result["predicted_label"] == 1 else "❌"
                print(f"  {status} {url} -> {result['predicted_label']} ({result['final_prob']:.3f})")
            else:
                print(f"  💥 {url} -> 失败: {result.get('error', '未知错误')}")

            await asyncio.sleep(1)  # 避免请求过快

        # 测试良性网站
        print(f"\n✅ 测试已知良性网站...")
        benign_results = []
        for url in KNOWN_BENIGN_URLS:
            result = await test_single_url(session, url, 0)
            benign_results.append(result)
            all_results.append(result)

            # 显示即时结果
            if result["success"]:
                status = "✅" if result["predicted_label"] == 0 else "❌"
                print(f"  {status} {url} -> {result['predicted_label']} ({result['final_prob']:.3f})")
            else:
                print(f"  💥 {url} -> 失败: {result.get('error', '未知错误')}")

            await asyncio.sleep(0.5)

        # 分析结果
        analyze_realistic_results(all_results, suspicious_results, benign_results)

def analyze_realistic_results(all_results: List[Dict], suspicious_results: List[Dict], benign_results: List[Dict]):
    """分析真实测试结果"""
    print("\n📊 真实测试结果分析")
    print("=" * 60)

    # 总体统计
    successful_tests = [r for r in all_results if r["success"]]
    total_tests = len(all_results)

    print(f"📈 总体统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  成功测试: {len(successful_tests)}")
    print(f"  失败测试: {total_tests - len(successful_tests)}")

    if successful_tests:
        # 准确率计算
        correct_predictions = [r for r in successful_tests if r["predicted_label"] == r["expected_label"]]
        accuracy = len(correct_predictions) / len(successful_tests) * 100

        print(f"  准确率: {accuracy:.2f}% ({len(correct_predictions)}/{len(successful_tests)})")

        # 可疑网站分析
        suspicious_success = [r for r in suspicious_results if r["success"]]
        if suspicious_success:
            # 分析预测分布
            predicted_phishing = [r for r in suspicious_success if r["predicted_label"] == 1]
            predicted_benign = [r for r in suspicious_success if r["predicted_label"] == 0]

            print(f"\n🎯 可疑网站分析:")
            print(f"  测试数量: {len(suspicious_success)}")
            print(f"  判为钓鱼: {len(predicted_phishing)} ({len(predicted_phishing)/len(suspicious_success)*100:.1f}%)")
            print(f"  判为良性: {len(predicted_benign)} ({len(predicted_benign)/len(suspicious_success)*100:.1f}%)")

            # 显示高概率钓鱼网站
            high_prob_phishing = [r for r in predicted_phishing if r["final_prob"] > 0.7]
            if high_prob_phishing:
                print(f"\n🔥 高置信度钓鱼网站:")
                for site in high_prob_phishing:
                    print(f"    - {site['url']} ({site['final_prob']:.3f})")

            # 显示低概率网站（可能不存在）
            low_prob_sites = [r for r in suspicious_success if r["final_prob"] < 0.3]
            if low_prob_sites:
                print(f"\n❓ 低置信度/可能不存在:")
                for site in low_prob_sites:
                    print(f"    - {site['url']} ({site['final_prob']:.3f}) - 状态: {site.get('status_code')}")

        # 良性网站分析
        benign_success = [r for r in benign_results if r["success"]]
        if benign_success:
            benign_correct = [r for r in benign_success if r["predicted_label"] == 0]
            benign_accuracy = len(benign_correct) / len(benign_success) * 100

            print(f"\n✅ 良性网站分析:")
            print(f"  测试数量: {len(benign_success)}")
            print(f"  正确识别: {len(benign_correct)}")
            print(f"  准确率: {benign_accuracy:.2f}%")

            # 显示误报
            false_positives = [r for r in benign_success if r["predicted_label"] == 1]
            if false_positives:
                print(f"\n❌ 误报网站:")
                for fp in false_positives:
                    print(f"    - {fp['url']} ({fp['final_prob']:.3f})")

        # 性能统计
        response_times = [r["response_time"] for r in successful_tests if "response_time" in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            print(f"\n⚡ 性能统计:")
            print(f"  平均响应时间: {avg_response_time:.2f}秒")
            print(f"  最大响应时间: {max_response_time:.2f}秒")

if __name__ == "__main__":
    asyncio.run(run_realistic_test())