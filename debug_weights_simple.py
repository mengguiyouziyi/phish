#!/usr/bin/env python3
"""
调试权重计算问题
"""

import sys
import os
sys.path.append('.')

# 先测试基本的feature extraction
from train_advanced_v3 import extract_enhanced_features

def debug_weight_calculation():
    """调试权重计算"""
    print("🔍 调试权重计算...")

    test_url = "https://www.baidu.com/index.php"
    print(f"\n📊 测试URL: {test_url}")

    # 直接测试特征提取
    try:
        features = extract_enhanced_features(test_url)
        print(f"✅ 特征提取成功，特征数量: {len(features)}")
        print(f"   前10个特征: {features[:10]}")

        # 检查关键特征
        print(f"   URL长度: {features[0]}")
        print(f"   主机长度: {features[1]}")
        print(f"   路径长度: {features[2]}")
        print(f"   是否HTTPS: {features[27]}")
        print(f"   子域名深度: {features[33]}")

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return

    # 测试模型加载
    try:
        import torch
        print(f"\n🧠 测试模型加载...")
        ckpt = torch.load("artifacts/fusion_advanced_v3.pt", map_location="cpu", weights_only=False)
        print(f"✅ 模型加载成功")
        print(f"   输入特征数: {ckpt.get('input_features', 'unknown')}")
        print(f"   模型类型: {'Advanced' if any(key.startswith('fc') for key in ckpt.get('model_state_dict', {}).keys()) else 'Basic'}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 测试推理管道
    try:
        from phishguard_v1.models.inference import InferencePipeline
        print(f"\n🔧 测试推理管道...")

        pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_advanced_v3.pt", enable_fusion=True)
        print(f"✅ 推理管道创建成功")

        # 创建测试特征
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
        print(f"   URL模型良性概率: {url_prob:.4f}")

        # FusionDNN预测
        result = pipe.predict(features)
        print(f"   FusionDNN良性概率: {result['fusion_prob']:.4f}")
        print(f"   最终良性概率: {result['final_prob']:.4f}")
        print(f"   预测标签: {'良性' if result['label'] == 1 else '钓鱼'}")

        # 检查权重计算
        complexity = pipe._analyze_url_complexity(test_url)
        print(f"   URL复杂度: {complexity}")

        weights = pipe._get_dynamic_weights(test_url, url_prob, result['fusion_prob'])
        print(f"   权重: URL={weights['url']:.4f}, FusionDNN={weights['fusion']:.4f}")

        # 手动计算验证
        manual_final = weights['url'] * url_prob + weights['fusion'] * result['fusion_prob']
        print(f"   手动计算最终概率: {manual_final:.4f}")
        print(f"   管道计算最终概率: {result['final_prob']:.4f}")
        print(f"   计算一致: {'✅' if abs(manual_final - result['final_prob']) < 0.001 else '❌'}")

    except Exception as e:
        print(f"❌ 推理管道测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_weight_calculation()