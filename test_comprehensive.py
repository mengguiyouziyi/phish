#!/usr/bin/env python3
"""
综合测试优化后的系统效果
"""

import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline
from train_advanced_v3 import extract_enhanced_features

def test_comprehensive():
    """综合测试系统效果"""
    print("🔍 综合测试优化后的系统效果...")

    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_advanced_v3.pt", enable_fusion=True)

    # 测试用例
    test_cases = [
        # 良性网站
        ("https://www.baidu.com", "良性", "百度搜索引擎"),
        ("https://www.google.com", "良性", "Google搜索引擎"),
        ("https://github.com", "良性", "GitHub代码托管"),
        ("https://www.taobao.com", "良性", "淘宝购物网站"),
        ("https://www.qq.com", "良性", "腾讯QQ"),

        # 带路径的良性网站
        ("https://www.baidu.com/s?wd=test", "良性", "百度搜索结果页"),
        ("https://www.baidu.com/index.php", "良性", "百度首页变体"),
        ("https://github.com/user/repo", "良性", "GitHub仓库页面"),

        # 钓鱼网站
        ("http://secure-login.apple.com.verify-login.com", "钓鱼", "假冒Apple登录"),
        ("http://www.amazon.update.account.secure-login.net", "钓鱼", "假冒Amazon更新"),
        ("http://paypal.com.secure.transaction.update.com", "钓鱼", "假冒PayPal交易"),
        ("http://verify-paypal-account.com", "钓鱼", "假冒PayPal验证"),
        ("http://microsoft-login-alert.com", "钓鱼", "假冒Microsoft警告"),
    ]

    print(f"\n📊 测试结果汇总:")
    print("-" * 80)

    correct = 0
    total = len(test_cases)

    for i, (url, expected, description) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. 测试URL: {url}")
        print(f"    描述: {description}")
        print(f"    期望: {expected}")

        # 使用与训练相同的特征提取函数
        raw_features = extract_enhanced_features(url)

        # 转换为pipeline需要的格式
        host = url.split('//')[-1].split('/')[0] if '//' in url else url.split('/')[0]
        path = '/' + '/'.join(url.split('/')[3:]) if len(url.split('/')) > 3 else '/'
        query = url.split('?')[-1] if '?' in url else ''
        fragment = url.split('#')[-1] if '#' in url else ''

        features = {
            "request_url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "bytes": 1024,
            "url_feats": {
                "url_len": raw_features[0],
                "host_len": raw_features[1],
                "path_len": raw_features[2],
                "num_digits": raw_features[3],
                "num_letters": raw_features[4],
                "num_specials": raw_features[5],
                "num_dots": raw_features[6],
                "num_hyphen": raw_features[7],
                "num_slash": raw_features[8],
                "num_qm": raw_features[9],
                "num_at": raw_features[10],
                "num_pct": raw_features[11],
                "num_equal": raw_features[12],
                "num_amp": raw_features[13],
                "num_plus": raw_features[14],
                "num_hash": raw_features[15],
                "query_len": raw_features[16],
                "fragment_len": raw_features[17],
                "domain_len": raw_features[18],
                "digit_ratio": raw_features[19],
                "special_ratio": raw_features[20],
                "letter_ratio": raw_features[21],
                "path_depth": raw_features[22],
                "num_params": raw_features[23],
                "has_ip": raw_features[24],
                "tld_suspicious": raw_features[25],
                "has_punycode": raw_features[26],
                "scheme_https": raw_features[27],
                "has_params": raw_features[28],
                "has_file_ext": raw_features[29],
                "is_suspicious_file": raw_features[30],
                "has_www": raw_features[31],
                "is_long_domain": raw_features[32],
                "subdomain_depth": raw_features[33],
                "status_code": raw_features[34],  # 注意：这里是200
                "bytes": raw_features[35]  # 注意：这里是1024
            }
        }

        result = pipe.predict(features)
        actual = "良性" if result['label'] == 1 else "钓鱼"
        confidence = result['final_prob']

        print(f"    实际: {actual}")
        print(f"    置信度: {confidence:.4f}")
        print(f"    URL模型: {result['url_prob']:.4f}")
        print(f"    FusionDNN: {result['fusion_prob']:.4f}")

        if actual == expected:
            print(f"    ✅ 正确")
            correct += 1
        else:
            print(f"    ❌ 错误")

    print("\n" + "=" * 80)
    print(f"🎯 测试结果汇总:")
    print(f"    总数: {total}")
    print(f"    正确: {correct}")
    print(f"    准确率: {correct/total*100:.1f}%")

    # 分析错误
    if correct < total:
        print(f"\n❌ 错误分析:")
        print(f"    错误数: {total - correct}")
        print(f"    错误率: {(total-correct)/total*100:.1f}%")
    else:
        print(f"\n🎉 完美！所有测试用例都通过了！")

if __name__ == "__main__":
    test_comprehensive()