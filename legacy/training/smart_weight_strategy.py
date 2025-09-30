#!/usr/bin/env python3
"""
智能权重策略，根据URL复杂度调整模型权重
"""

import re
import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline

def analyze_url_complexity(url):
    """分析URL复杂度"""
    complexity_score = 0

    # 长度得分
    if len(url) > 50:
        complexity_score += 1
    if len(url) > 100:
        complexity_score += 1

    # 路径深度
    path_depth = url.count('/') - 2  # 减去 http:// 的两个斜杠
    if path_depth > 2:
        complexity_score += 1
    if path_depth > 4:
        complexity_score += 1

    # 查询参数
    if '?' in url:
        complexity_score += 1
        query_params = url.split('?')[1]
        param_count = query_params.count('&') + 1
        if param_count > 2:
            complexity_score += 1

    # 特殊字符
    special_chars = sum(1 for c in url if c in ['%', '&', '=', '+', ';'])
    if special_chars > 2:
        complexity_score += 1

    # 文件扩展名
    if any(ext in url.lower() for ext in ['.php', '.html', '.htm', '.asp', '.aspx', '.jsp']):
        complexity_score += 0.5

    # ID-like patterns
    id_patterns = re.findall(r'/[a-zA-Z0-9]{8,}/', url)
    if id_patterns:
        complexity_score += 1

    return complexity_score

def get_dynamic_weights(url, url_prob, fusion_prob):
    """根据URL复杂度和模型置信度动态调整权重"""
    complexity = analyze_url_complexity(url)

    # 基础权重
    base_url_weight = 0.6
    base_fusion_weight = 0.4

    # 根据复杂度调整
    if complexity >= 3:
        # 高复杂度URL，FusionDNN权重更高
        url_weight = base_url_weight - 0.2
        fusion_weight = base_fusion_weight + 0.2
    elif complexity >= 2:
        # 中等复杂度，稍微调整
        url_weight = base_url_weight - 0.1
        fusion_weight = base_fusion_weight + 0.1
    else:
        # 低复杂度，使用基础权重
        url_weight = base_url_weight
        fusion_weight = base_fusion_weight

    # 根据模型置信度调整
    # 如果URL模型置信度很低，降低其权重
    if url_prob < 0.1:
        url_weight *= 0.7
        fusion_weight *= 1.3

    # 如果FusionDNN置信度很高，适当提高权重
    if fusion_prob > 0.95 or fusion_prob < 0.05:
        fusion_weight *= 1.1
        url_weight *= 0.9

    # 归一化权重
    total_weight = url_weight + fusion_weight
    url_weight /= total_weight
    fusion_weight /= total_weight

    return {"url": url_weight, "fusion": fusion_weight}

def test_smart_weight_strategy():
    """测试智能权重策略"""
    print("🔍 测试智能权重策略...")

    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_balanced_v2.pt", enable_fusion=True)

    # 测试URLs
    test_urls = [
        "https://www.baidu.com",
        "https://www.baidu.com/index.php",
        "https://www.baidu.com/s?wd=test",
        "https://www.baidu.com/img/bd_logo1.png",
        "https://www.google.com/search?q=test",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://verify-paypal-account.com",
        "http://apple-account-verify.com"
    ]

    print(f"{'URL':<50} {'复杂度':<6} {'URL权重':<8} {'Fusion权重':<10} {'URL模型':<10} {'FusionDNN':<10} {'最终':<10} {'结果':<8}")
    print("-" * 130)

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

        # 获取模型预测
        result = pipe.predict(features)
        url_prob = result['url_prob']
        fusion_prob = result['fusion_prob']

        # 计算动态权重
        weights = get_dynamic_weights(url, url_prob, fusion_prob)

        # 重新计算最终概率
        final_prob = weights["url"] * url_prob + weights["fusion"] * fusion_prob
        label = '良性' if final_prob >= 0.5 else '钓鱼'

        complexity = analyze_url_complexity(url)

        print(f"{url:<50} {complexity:<6} {weights['url']:<8.3f} {weights['fusion']:<10.3f} {url_prob:<10.4f} {fusion_prob:<10.4f} {final_prob:<10.4f} {label:<8}")

if __name__ == "__main__":
    test_smart_weight_strategy()