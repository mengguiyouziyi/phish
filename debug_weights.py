#!/usr/bin/env python3
"""
调试权重计算问题
"""

import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline

def debug_weight_calculation():
    """调试权重计算"""
    print("🔍 调试权重计算...")

    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_advanced_v3.pt", enable_fusion=True)

    test_url = "https://www.baidu.com/index.php"
    print(f"\n📊 测试URL: {test_url}")

    # 计算URL复杂度
    complexity = pipe._analyze_url_complexity(test_url)
    print(f"URL复杂度: {complexity}")

    # 获取模型预测
    from train_advanced_v3 import extract_enhanced_features

    host = test_url.split('//')[-1].split('/')[0] if '//' in test_url else test_url.split('/')[0]
    path = '/' + '/'.join(test_url.split('/')[3:]) if len(test_url.split('/')) > 3 else '/'
    query = test_url.split('?')[-1] if '?' in test_url else ''
    fragment = test_url.split('#')[-1] if '#' in test_url else ''

    raw_features = extract_enhanced_features(test_url)

    features = {
        "request_url": test_url,
        "final_url": test_url,
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
            "status_code": raw_features[34],
            "bytes": raw_features[35]
        }
    }

    # URL模型预测
    url_phishing_prob = pipe.url_model.score(test_url)
    url_prob = 1.0 - url_phishing_prob
    print(f"URL模型良性概率: {url_prob:.4f}")

    # FusionDNN预测
    fusion_features = extract_enhanced_features(test_url)
    import torch
    import numpy as np
    x_array = np.array(fusion_features).reshape(1, -1)
    x_array = (x_array - pipe.fusion_scaler_mean.numpy()) / pipe.fusion_scaler_scale.numpy()
    x = torch.tensor(x_array, dtype=torch.float32)

    from phishguard_v1.models.fusion_model import predict_proba
    fusion_prob = predict_proba(pipe.fusion, x)[0,0].item()
    print(f"FusionDNN良性概率: {fusion_prob:.4f}")

    # 权重计算
    weights = pipe._get_dynamic_weights(test_url, url_prob, fusion_prob)
    print(f"权重: URL={weights['url']:.4f}, FusionDNN={weights['fusion']:.4f}")

    # 最终概率
    final_prob = weights["url"] * url_prob + weights["fusion"] * fusion_prob
    print(f"最终良性概率: {final_prob:.4f}")
    print(f"最终钓鱼概率: {1-final_prob:.4f}")
    print(f"预测标签: {'良性' if final_prob >= 0.5 else '钓鱼'}")

    # 完整管道预测
    result = pipe.predict(features)
    print(f"\n完整管道结果:")
    print(f"  URL模型: {result['url_prob']:.4f}")
    print(f"  FusionDNN: {result['fusion_prob']:.4f}")
    print(f"  最终概率: {result['final_prob']:.4f}")
    print(f"  预测标签: {'良性' if result['label'] == 1 else '钓鱼'}")

if __name__ == "__main__":
    debug_weight_calculation()