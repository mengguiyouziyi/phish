#!/usr/bin/env python3
"""
测试带路径的百度URL
"""

import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline

def test_baidu_with_path():
    """测试带路径的百度URL"""
    print("🔍 测试带路径的百度URL...")

    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_advanced_v2.pt", enable_fusion=True)

    # 测试带路径的百度URL
    test_urls = [
        "https://www.baidu.com",
        "https://www.baidu.com/index.php",
        "https://www.baidu.com/s?wd=test",
        "https://www.baidu.com/img/bd_logo1.png"
    ]

    for url in test_urls:
        print(f"\n📊 测试URL: {url}")

        # 计算高级特征
        host = url.split('//')[-1].split('/')[0]
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
                # 基础特征
                "url_len": len(url),
                "host_len": len(host),
                "path_len": len(path),
                "num_digits": sum(c.isdigit() for c in url),
                "num_letters": sum(c.isalpha() for c in url),
                "num_specials": sum(not c.isalnum() for c in url),

                # 特殊字符统计
                "num_dots": url.count('.'),
                "num_hyphen": url.count('-'),
                "num_slash": url.count('/'),
                "num_qm": url.count('?'),
                "num_at": url.count('@'),
                "num_pct": url.count('%'),
                "num_equal": url.count('='),
                "num_amp": url.count('&'),
                "num_plus": url.count('+'),
                "num_hash": url.count('#'),

                # 布尔特征
                "has_ip": any(part.isdigit() for part in host.split('.')),
                "subdomain_depth": host.count('.') if host != 'localhost' else 0,
                "tld_suspicious": 1 if any(tld in host.lower() for tld in ['.tk', '.ml', '.ga', '.cf', '.top', '.click']) else 0,
                "has_punycode": 1 if 'xn--' in host.lower() else 0,
                "scheme_https": 1 if url.startswith('https') else 0,

                # 查询和片段长度
                "query_len": len(query),
                "fragment_len": len(fragment),

                # 高级特征
                "domain_len": len(host.split('.')[0]) if '.' in host else len(host),
                "has_www": 1 if host.startswith('www.') else 0,
                "is_long_domain": 1 if len(host) > 30 else 0,
                "path_depth": len([p for p in path.split('/') if p]) if path != '/' else 0,
                "has_params": 1 if '?' in url else 0,
                "num_params": len([p for p in query.split('&') if p]) if query else 0,
                "has_file_ext": 1 if any(ext in path.lower() for ext in ['.php', '.html', '.htm', '.asp', '.aspx', '.jsp', '.cgi', '.pl']) else 0,
                "is_suspicious_file": 1 if any(ext in path.lower() for ext in ['.exe', '.bat', '.cmd', '.scr', '.pif']) else 0,

                # 字符比例特征
                "digit_ratio": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
                "special_ratio": sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0,
                "letter_ratio": sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0
            }
        }

        result = pipe.predict(features)
        print(f"  URL模型良性概率: {result['url_prob']:.4f}")
        print(f"  FusionDNN良性概率: {result['fusion_prob']:.4f}")
        print(f"  最终良性概率: {result['final_prob']:.4f}")
        print(f"  标签: {'良性' if result['label'] == 1 else '钓鱼'}")

if __name__ == "__main__":
    test_baidu_with_path()