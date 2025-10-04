#!/usr/bin/env python3
"""
使用大量真实钓鱼网站全面测试新模型性能
"""

import asyncio
import httpx
from phishguard_v1.features.fetcher import fetch_one
from phishguard_v1.features.parser import extract_from_html
from phishguard_v1.models.inference import InferencePipeline

async def comprehensive_real_phishing_test():
    """全面的真实钓鱼网站测试"""
    print("🔍 全面真实钓鱼网站测试开始...")
    print("=" * 60)

    # 加载新模型
    try:
        pipeline = InferencePipeline(
            fusion_ckpt_path="artifacts/real_phishing_advanced_20251001_204447.pt",
            enable_fusion=True
        )
        print(f"✅ 新模型加载成功!")
        print(f"🧠 特征数量: {len(pipeline.fusion_feature_names)}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    # 合法网站
    legitimate_urls = [
        ("https://www.google.com", "合法", "搜索引擎"),
        ("https://www.github.com", "合法", "代码托管"),
        ("https://www.microsoft.com", "合法", "科技公司"),
        ("https://www.apple.com", "合法", "科技公司"),
        ("https://www.wikipedia.org", "合法", "百科全书"),
        ("https://www.stackoverflow.com", "合法", "技术问答"),
        ("https://www.paypal.com", "合法", "支付平台"),
        ("https://www.facebook.com", "合法", "社交媒体"),
        ("https://www.linkedin.com", "合法", "职业社交"),
        ("https://www.amazon.com", "合法", "电商平台"),
    ]

    # 钓鱼网站 - 从之前测试中找到的
    phishing_urls = [
        # 银行钓鱼
        ("http://wells-fargo-login.com", "钓鱼", "富国银行钓鱼"),
        ("http://citibank-online.com", "钓鱼", "花旗银行钓鱼"),
        ("http://wellsfargo-online.com", "钓鱼", "富国银行钓鱼变体"),
        ("http://www.wells-fargo-login.com", "钓鱼", "富国银行钓鱼(www)"),
        ("http://www.citibank-online.com", "钓鱼", "花旗银行钓鱼(www)"),

        # PayPal钓鱼
        ("http://paypal-verification.net", "钓鱼", "PayPal验证钓鱼"),
        ("http://www.paypal-verification.net", "钓鱼", "PayPal验证钓鱼(www)"),
        ("http://paypal-verification.org", "钓鱼", "PayPal验证钓鱼(.org)"),
        ("http://www.paypal-verification.org", "钓鱼", "PayPal验证钓鱼(.org,www)"),
        ("http://login.secure-paypal.com", "钓鱼", "PayPal子域名钓鱼"),

        # Apple钓鱼
        ("http://apple-id-verify.com", "钓鱼", "Apple ID钓鱼"),
        ("http://snapchat-security.com", "钓鱼", "Snapchat钓鱼"),

        # 科技公司钓鱼
        ("http://facebook.login-secure.net", "钓鱼", "Facebook钓鱼"),
        ("http://www.facebook.login-secure.net", "钓鱼", "Facebook钓鱼(www)"),
        ("http://gmail-security-alert.com", "钓鱼", "Gmail安全钓鱼"),
        ("http://gmail-security.online", "钓鱼", "Gmail安全钓鱼(online)"),
        ("http://www.snapchat-verify.com", "钓鱼", "Snapchat验证钓鱼(www)"),

        # 域名欺骗钓鱼
        ("http://paypa1.com", "钓鱼", "PayPal域名欺骗"),
        ("http://arnazon.com", "钓鱼", "Amazon域名欺骗"),
        ("http://faceb00k.com", "钓鱼", "Facebook域名欺骗"),
        ("http://amaz0n.com", "钓鱼", "Amazon域名欺骗(数字0)"),
        ("http://microsft.com", "钓鱼", "Microsoft域名欺骗"),
        ("http://linkdedin.com", "钓鱼", "LinkedIn域名欺骗"),

        # HTTPS钓鱼
        ("https://account-verification.net", "钓鱼", "HTTPS钓鱼验证"),
    ]

    print(f"\n📊 测试规模:")
    print(f"✅ 合法网站: {len(legitimate_urls)} 个")
    print(f"🎣 钓鱼网站: {len(phishing_urls)} 个")
    print(f"📈 总计: {len(legitimate_urls) + len(phishing_urls)} 个")

    print(f"\n🔍 开始测试...")
    print("-" * 60)

    all_test_cases = legitimate_urls + phishing_urls
    results = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for url, expected_type, description in all_test_cases:
            print(f"\n🔗 {url}")
            print(f"📝 期望: {expected_type} - {description}")

            try:
                # 获取网页数据
                item = await fetch_one(url.strip(), client)
                html_feats = extract_from_html(
                    item.get("html", ""),
                    item.get("final_url") or item.get("request_url")
                )
                item["html_feats"] = html_feats

                # 模型预测
                pred = pipeline.predict(item)

                actual_type = "钓鱼" if pred['label'] == 1 else "合法"
                is_correct = expected_type == actual_type
                phishing_prob = pred['final_prob']

                result = {
                    'url': url,
                    'expected': expected_type,
                    'actual': actual_type,
                    'correct': is_correct,
                    'prob': phishing_prob,
                    'decision': pred['details']['decision'],
                    'description': description,
                    'category': 'legitimate' if expected_type == '合法' else 'phishing'
                }
                results.append(result)

                # 显示结果
                if is_correct:
                    print(f"✅ 预测: {actual_type} ({phishing_prob:.1%}) - {pred['details']['decision']}")
                else:
                    print(f"❌ 预测: {actual_type} ({phishing_prob:.1%}) - {pred['details']['decision']}")
                    print(f"   ⚠️ 错误! 期望: {expected_type}")

            except Exception as e:
                print(f"❌ 测试失败: {e}")
                results.append({
                    'url': url,
                    'expected': expected_type,
                    'actual': '错误',
                    'correct': False,
                    'prob': 0,
                    'decision': 'error',
                    'description': description,
                    'category': 'legitimate' if expected_type == '合法' else 'phishing'
                })

    # 统计分析
    print(f"\n{'='*60}")
    print("📊 全面测试结果统计")
    print(f"{'='*60}")

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total

    print(f"🎯 总体准确率: {accuracy:.1%} ({correct}/{total})")

    # 按类别统计
    legit_results = [r for r in results if r['category'] == 'legitimate']
    phish_results = [r for r in results if r['category'] == 'phishing']

    legit_correct = sum(1 for r in legit_results if r['correct'])
    phish_correct = sum(1 for r in phish_results if r['correct'])

    legit_accuracy = legit_correct / len(legit_results) if legit_results else 0
    phish_accuracy = phish_correct / len(phish_results) if phish_results else 0

    print(f"✅ 合法网站准确率: {legit_accuracy:.1%} ({legit_correct}/{len(legit_results)})")
    print(f"🎣 钓鱼网站准确率: {phish_accuracy:.1%} ({phish_correct}/{len(phish_results)})")

    # 按钓鱼类型细分
    phish_categories = {}
    for r in phish_results:
        if '钓鱼' in r['description']:
            category = r['description'].split('钓鱼')[0].strip()
            if category not in phish_categories:
                phish_categories[category] = {'total': 0, 'correct': 0}
            phish_categories[category]['total'] += 1
            if r['correct']:
                phish_categories[category]['correct'] += 1

    print(f"\n🎣 钓鱼网站细分统计:")
    for category, stats in phish_categories.items():
        cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {category}: {cat_accuracy:.1%} ({stats['correct']}/{stats['total']})")

    # 错误案例分析
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\n❌ 错误案例分析 ({len(errors)}个):")

        # 按错误类型分类
        false_negatives = [r for r in errors if r['expected'] == '钓鱼' and r['actual'] == '合法']
        false_positives = [r for r in errors if r['expected'] == '合法' and r['actual'] == '钓鱼']

        print(f"\n🔴 漏报 (钓鱼→合法): {len(false_negatives)}个")
        for error in false_negatives:
            print(f"  • {error['url']} - 预测概率: {error['prob']:.1%}")

        if false_positives:
            print(f"\n🟡 误报 (合法→钓鱼): {len(false_positives)}个")
            for error in false_positives:
                print(f"  • {error['url']} - 预测概率: {error['prob']:.1%}")

    # 性能评估
    print(f"\n🎯 性能评估:")

    if accuracy >= 0.95:
        print("🏆 模型表现优秀 (≥95%)")
        grade = "A+"
    elif accuracy >= 0.90:
        print("🥇 模型表现很好 (≥90%)")
        grade = "A"
    elif accuracy >= 0.85:
        print("🥈 模型表现良好 (≥85%)")
        grade = "B+"
    elif accuracy >= 0.80:
        print("🥉 模型表现一般 (≥80%)")
        grade = "B"
    elif accuracy >= 0.70:
        print("⚠️ 模型表现及格 (≥70%)")
        grade = "C"
    else:
        print("❌ 模型表现不佳 (<70%)")
        grade = "D"

    # 特殊指标
    phishing_detection_rate = phish_accuracy if phish_results else 0
    false_negative_rate = len(false_negatives) / len(phish_results) if phish_results else 0

    print(f"\n📈 关键指标:")
    print(f"  钓鱼网站检测率: {phishing_detection_rate:.1%}")
    print(f"  漏报率: {false_negative_rate:.1%}")
    print(f"  误报数量: {len(false_positives)}")

    # 综合评级
    print(f"\n🎖️ 综合评级: {grade}")

    return {
        'total_accuracy': accuracy,
        'legitimate_accuracy': legit_accuracy,
        'phishing_accuracy': phish_accuracy,
        'grade': grade,
        'errors': errors,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'results': results
    }

if __name__ == "__main__":
    results = asyncio.run(comprehensive_real_phishing_test())

    if results['total_accuracy'] >= 0.80:
        print(f"\n✅ 新模型性能满足要求! 可以部署使用")
    else:
        print(f"\n⚠️ 新模型性能需要进一步改进")