#!/usr/bin/env python3
"""
钓鱼网站检测测试用例收集和测试
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
import json
import time
from pathlib import Path

# 测试用例定义

# 钓鱼网站测试用例 (正例)
PHISHING_TEST_CASES = [
    # 银行钓鱼网站
    "https://www.icbc-security.com/login",  # 工商银行钓鱼
    "https://www.bankofamerica-secure.com",  # 美国银行钓鱼
    "https://login.chase-bank.com",  # 大通银行钓鱼
    "https://www.hsbc-online-banking.com",  # 汇丰银行钓鱼
    "https://secure-paypal-account.com",  # PayPal钓鱼
    "https://www.apple-id-support.com",  # Apple ID钓鱼

    # 社交媒体钓鱼
    "https://www.facebook-login-verify.com",  # Facebook钓鱼
    "https://login-twitter-secure.com",  # Twitter钓鱼
    "https://www.linkedin-verification.com",  # LinkedIn钓鱼
    "https://instagram-account-secure.com",  # Instagram钓鱼

    # 电商平台钓鱼
    "https://www.amazon-order-confirm.com",  # Amazon钓鱼
    "https://taobao-login-center.com",  # 淘宝钓鱼
    "https://www.alibaba-secure-login.com",  # 阿里巴巴钓鱼

    # 邮件服务钓鱼
    "https://gmail-account-recovery.com",  # Gmail钓鱼
    "https://outlook-security-check.com",  # Outlook钓鱼
    "https://qq-mail-login.com",  # QQ邮箱钓鱼

    # 政府机构钓鱼
    "https://irs-tax-return-verify.com",  # 美国国税局钓鱼
    "https://social-security-benefits.com",  # 社保钓鱼

    # 知名品牌钓鱼
    "https://microsoft-account-security.com",  # Microsoft钓鱼
    "https://google-account-verify.com",  # Google钓鱼
    "https://netflix-billing-update.com",  # Netflix钓鱼
    "https://spotify-account-confirm.com",  # Spotify钓鱼
]

# 良性网站测试用例 (反例)
BENIGN_TEST_CASES = [
    # 大型科技公司
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.linkedin.com",
    "https://www.github.com",

    # 中国知名网站
    "https://www.baidu.com",
    "https://www.alibaba.com",
    "https://www.tmall.com",
    "https://www.qq.com",
    "https://www.weibo.com",
    "https://www.zhihu.com",
    "https://www.douban.com",
    "https://www.jd.com",
    "https://www.bytedance.com",

    # 金融机构
    "https://www.bankofamerica.com",
    "https://www.chase.com",
    "https://www.wellsfargo.com",
    "https://www.citibank.com",
    "https://www.hsbc.com",
    "https://www.icbc.com.cn",
    "https://www.bankofchina.com",

    # 政府机构
    "https://www.gov.cn",
    "https://www.whitehouse.gov",
    "https://www.europa.eu",

    # 教育机构
    "https://www.harvard.edu",
    "https://www.mit.edu",
    "https://www.stanford.edu",
    "https://www.tsinghua.edu.cn",
    "https://www.pku.edu.cn",

    # 电商平台
    "https://www.ebay.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.costco.com",

    # 娱乐平台
    "https://www.netflix.com",
    "https://www.spotify.com",
    "https://www.youtube.com",
    "https://www.instagram.com",

    # 新闻媒体
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.reuters.com",
    "https://www.nytimes.com",
    "https://www.theguardian.com",
]

async def test_url_batch(session: aiohttp.ClientSession, urls: List[str], expected_label: int) -> List[Dict[str, Any]]:
    """批量测试URL"""
    results = []

    for url in urls:
        try:
            start_time = time.time()

            # 调用API进行测试
            async with session.post(
                "http://localhost:8001/predict",
                json={"url": url, "screenshot": False},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    prediction = data.get("prediction", {})
                    response_time = time.time() - start_time

                    result = {
                        "url": url,
                        "expected_label": expected_label,
                        "predicted_label": prediction.get("label", -1),
                        "final_prob": prediction.get("final_prob", 0),
                        "url_prob": prediction.get("url_prob", 0),
                        "fusion_prob": prediction.get("fusion_prob", 0),
                        "response_time": response_time,
                        "status_code": data.get("features", {}).get("status_code"),
                        "content_type": data.get("features", {}).get("content_type"),
                        "success": True
                    }
                else:
                    result = {
                        "url": url,
                        "expected_label": expected_label,
                        "predicted_label": -1,
                        "error": f"HTTP {response.status}",
                        "success": False
                    }

        except Exception as e:
            result = {
                "url": url,
                "expected_label": expected_label,
                "predicted_label": -1,
                "error": str(e),
                "success": False
            }

        results.append(result)

        # 避免请求过快
        await asyncio.sleep(0.5)

    return results

async def run_tests():
    """运行完整测试"""
    print("🧪 开始钓鱼网站检测测试")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # 测试钓鱼网站
        print("🎯 测试钓鱼网站 (正例)...")
        phishing_results = await test_url_batch(session, PHISHING_TEST_CASES, 1)

        print("\n✅ 测试良性网站 (反例)...")
        benign_results = await test_url_batch(session, BENIGN_TEST_CASES, 0)

        # 合并结果
        all_results = phishing_results + benign_results

        # 分析结果
        analyze_results(all_results, phishing_results, benign_results)

        # 保存结果
        save_results(all_results, phishing_results, benign_results)

def analyze_results(all_results: List[Dict], phishing_results: List[Dict], benign_results: List[Dict]):
    """分析测试结果"""
    print("\n📊 测试结果分析")
    print("=" * 60)

    # 总体统计
    total_tests = len(all_results)
    successful_tests = [r for r in all_results if r["success"]]

    print(f"📈 总体统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  成功测试: {len(successful_tests)}")
    print(f"  失败测试: {total_tests - len(successful_tests)}")

    if successful_tests:
        # 准确率计算
        correct_predictions = [r for r in successful_tests if r["predicted_label"] == r["expected_label"]]
        accuracy = len(correct_predictions) / len(successful_tests) * 100

        print(f"  准确率: {accuracy:.2f}% ({len(correct_predictions)}/{len(successful_tests)})")

        # 分类统计
        phishing_success = [r for r in phishing_results if r["success"]]
        benign_success = [r for r in benign_results if r["success"]]

        if phishing_success:
            phishing_correct = [r for r in phishing_success if r["predicted_label"] == 1]
            phishing_recall = len(phishing_correct) / len(phishing_success) * 100
            print(f"  钓鱼网站召回率: {phishing_recall:.2f}% ({len(phishing_correct)}/{len(phishing_success)})")

        if benign_success:
            benign_correct = [r for r in benign_success if r["predicted_label"] == 0]
            benign_accuracy = len(benign_correct) / len(benign_success) * 100
            print(f"  良性网站准确率: {benign_accuracy:.2f}% ({len(benign_correct)}/{len(benign_success)})")

        # 误报分析
        false_positives = [r for r in successful_tests if r["expected_label"] == 0 and r["predicted_label"] == 1]
        false_negatives = [r for r in successful_tests if r["expected_label"] == 1 and r["predicted_label"] == 0]

        print(f"\n❌ 误报分析:")
        print(f"  误报 (良性判为钓鱼): {len(false_positives)} 个")
        print(f"  漏报 (钓鱼判为良性): {len(false_negatives)} 个")

        # 显示误报详情
        if false_positives:
            print(f"\n🔍 误报详情:")
            for fp in false_positives:
                print(f"    - {fp['url']} (概率: {fp['final_prob']:.3f})")

        if false_negatives:
            print(f"\n🔍 漏报详情:")
            for fn in false_negatives:
                print(f"    - {fn['url']} (概率: {fn['final_prob']:.3f})")

        # 性能统计
        response_times = [r["response_time"] for r in successful_tests if "response_time" in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            print(f"\n⚡ 性能统计:")
            print(f"  平均响应时间: {avg_response_time:.2f}秒")
            print(f"  最大响应时间: {max_response_time:.2f}秒")

def save_results(all_results: List[Dict], phishing_results: List[Dict], benign_results: List[Dict]):
    """保存测试结果"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 创建结果目录
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    # 保存详细结果
    results_file = results_dir / f"test_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "all_results": all_results,
            "phishing_results": phishing_results,
            "benign_results": benign_results
        }, f, ensure_ascii=False, indent=2)

    # 生成报告
    report_file = results_dir / f"test_report_{timestamp}.md"
    generate_test_report(all_results, phishing_results, benign_results, report_file)

    print(f"\n💾 测试结果已保存到:")
    print(f"  详细结果: {results_file}")
    print(f"  测试报告: {report_file}")

def generate_test_report(all_results: List[Dict], phishing_results: List[Dict], benign_results: List[Dict], output_file: Path):
    """生成测试报告"""

    # 计算统计信息
    successful_tests = [r for r in all_results if r["success"]]
    correct_predictions = [r for r in successful_tests if r["predicted_label"] == r["expected_label"]]
    false_positives = [r for r in successful_tests if r["expected_label"] == 0 and r["predicted_label"] == 1]
    false_negatives = [r for r in successful_tests if r["expected_label"] == 1 and r["predicted_label"] == 0]

    accuracy = len(correct_predictions) / len(successful_tests) * 100 if successful_tests else 0

    phishing_success = [r for r in phishing_results if r["success"]]
    phishing_correct = [r for r in phishing_success if r["predicted_label"] == 1]
    phishing_recall = len(phishing_correct) / len(phishing_success) * 100 if phishing_success else 0

    benign_success = [r for r in benign_results if r["success"]]
    benign_correct = [r for r in benign_success if r["predicted_label"] == 0]
    benign_accuracy = len(benign_correct) / len(benign_success) * 100 if benign_success else 0

    # 生成报告
    report = f"""# 钓鱼网站检测测试报告

**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 📊 测试统计

### 总体表现
- **总测试数**: {len(all_results)}
- **成功测试**: {len(successful_tests)}
- **失败测试**: {len(all_results) - len(successful_tests)}
- **整体准确率**: {accuracy:.2f}%

### 分类表现
#### 钓鱼网站检测 (正例)
- **测试数量**: {len(phishing_results)}
- **成功检测**: {len(phishing_success)}
- **召回率**: {phishing_recall:.2f}%

#### 良性网站检测 (反例)
- **测试数量**: {len(benign_results)}
- **正确识别**: {len(benign_success)}
- **准确率**: {benign_accuracy:.2f}%

### 误报分析
- **误报 (良性→钓鱼)**: {len(false_positives)} 个
- **漏报 (钓鱼→良性)**: {len(false_negatives)} 个

## ❌ 误报详情

"""

    if false_positives:
        report += "### 良性网站被误判为钓鱼\n\n"
        for fp in false_positives:
            report += f"- {fp['url']} (预测概率: {fp['final_prob']:.3f})\n"
        report += "\n"

    if false_negatives:
        report += "### 钓鱼网站被误判为良性\n\n"
        for fn in false_negatives:
            report += f"- {fn['url']} (预测概率: {fn['final_prob']:.3f})\n"
        report += "\n"

    # 失败测试详情
    failed_tests = [r for r in all_results if not r["success"]]
    if failed_tests:
        report += "## 💥 失败测试\n\n"
        for ft in failed_tests:
            report += f"- {ft['url']}: {ft.get('error', '未知错误')}\n"
        report += "\n"

    # 建议
    report += f"""## 💡 建议

基于测试结果，建议：

1. **误报优化**: 当前有 {len(false_positives)} 个良性网站被误判，建议进一步优化特征工程
2. **漏报处理**: 当前有 {len(false_negatives)} 个钓鱼网站被漏判，建议加强钓鱼特征识别
3. **性能优化**: 建议优化网络请求处理，减少超时情况

---

*报告由 PhishGuard v1 自动生成*
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    asyncio.run(run_tests())