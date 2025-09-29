#!/usr/bin/env python3
"""
测试优化后的模型
"""

import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline

def test_optimized_model():
    """测试优化后的模型"""
    print("🔍 测试优化后的模型...")

    # 使用优化后的模型
    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_optimized.pt", enable_fusion=True)

    # 测试URLs
    test_urls = [
        "https://www.baidu.com",
        "https://www.baidu.com/index.php",
        "https://www.baidu.com/s?wd=test",
        "https://www.baidu.com/img/bd_logo1.png",
        "https://www.google.com/search?q=test",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.facebook.com/profile.php",
        "https://www.amazon.com/dp/B123456789",
        "https://www.taobao.com/item.htm?id=123456",
        "https://www.jd.com/product/123456.html",
        "https://www.qq.com/news/",
        "https://www.weibo.com/u/1234567890",
        "https://www.zhihu.com/question/123456",
        "https://www.douban.com/subject/123456/",
        "https://www.wikipedia.org/wiki/Python",
        "http://verify-paypal-account.com",
        "http://apple-account-verify.com",
        "http://amazon-security-check.com",
        "http://paypal-security-center.com/login.php",
        "http://apple-security.com/verify.php"
    ]

    print(f"{'URL':<50} {'URL模型':<10} {'FusionDNN':<10} {'最终':<10} {'结果':<8}")
    print("-" * 100)

    for url in test_urls:
        features = {
            "request_url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "bytes": 1024,
            "url_feats": {
                "url_len": len(url),
                "host_len": len(url.split('//')[-1].split('/')[0]),
                "path_len": len('/'.join(url.split('/')[3:])),
                "num_digits": sum(c.isdigit() for c in url),
                "num_letters": sum(c.isalpha() for c in url),
                "num_specials": sum(not c.isalnum() for c in url),
                "num_dots": url.count('.'),
                "num_hyphen": url.count('-'),
                "num_slash": url.count('/'),
                "num_qm": url.count('?'),
                "num_at": url.count('@'),
                "num_pct": url.count('%'),
                "has_ip": False,
                "subdomain_depth": url.split('//')[-1].split('/')[0].count('.'),
                "tld_suspicious": 0,
                "has_punycode": 0,
                "scheme_https": 1 if url.startswith('https') else 0,
                "query_len": len(url.split('?')[-1]) if '?' in url else 0,
                "fragment_len": len(url.split('#')[-1]) if '#' in url else 0
            }
        }

        result = pipe.predict(features)

        url_prob = result['url_prob']
        fusion_prob = result['fusion_prob']
        final_prob = result['final_prob']
        label = '良性' if result['label'] == 1 else '钓鱼'

        print(f"{url:<50} {url_prob:<10.4f} {fusion_prob:<10.4f} {final_prob:<10.4f} {label:<8}")

if __name__ == "__main__":
    test_optimized_model()