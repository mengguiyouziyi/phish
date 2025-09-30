#!/usr/bin/env python3
"""
调试推理管道中的概率解释问题
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline
from train_advanced_v3 import extract_enhanced_features

def debug_phishing_prediction():
    """调试钓鱼网站预测的详细过程"""
    print("🔍 调试钓鱼网站预测过程...")

    # 测试钓鱼网站
    test_url = "http://secure-login.apple.com.verify-login.com"
    print(f"\n📊 测试URL: {test_url}")

    # 创建推理管道
    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_advanced_v3.pt", enable_fusion=True)

    # 1. 提取原始特征
    print("\n1. 提取原始特征...")
    raw_features = extract_enhanced_features(test_url)
    print(f"   原始特征数量: {len(raw_features)}")
    print(f"   前10个特征: {raw_features[:10]}")

    # 2. 转换为管道格式
    print("\n2. 转换为管道格式...")
    host = test_url.split('//')[-1].split('/')[0] if '//' in test_url else test_url.split('/')[0]
    path = '/' + '/'.join(test_url.split('/')[3:]) if len(test_url.split('/')) > 3 else '/'
    query = test_url.split('?')[-1] if '?' in test_url else ''
    fragment = test_url.split('#')[-1] if '#' in test_url else ''

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

    # 3. URL模型预测
    print("\n3. URL模型预测...")
    url_phishing_prob = pipe.url_model.score(test_url)
    url_prob = 1.0 - url_phishing_prob  # 转换为良性概率
    print(f"   URL模型钓鱼概率: {url_phishing_prob:.4f}")
    print(f"   URL模型良性概率: {url_prob:.4f}")

    # 4. FusionDNN特征准备
    print("\n4. FusionDNN特征准备...")
    row = {}
    uf = features.get("url_feats", {})

    # URL numeric features
    for k in ["url_len","host_len","path_len","num_digits","num_letters","num_specials","num_dots","num_hyphen","num_slash","num_qm","num_at","num_pct","num_equal","num_amp","num_plus","num_hash","subdomain_depth","query_len","fragment_len","domain_len","digit_ratio","special_ratio","letter_ratio","path_depth","num_params"]:
        row[k] = float(uf.get(k, 0))
    for k in ["has_ip","tld_suspicious","has_punycode","scheme_https","has_params","has_file_ext","is_suspicious_file","has_www","is_long_domain"]:
        row[k] = float(1 if uf.get(k, 0) else 0)

    # HTTP response features
    row["status_code"] = float(features.get("status_code") or 200)
    row["bytes"] = 1024.0

    print(f"   特征数量: {len(row)}")
    print(f"   特征字典: {dict(list(row.items())[:5])}...")  # 只显示前5个

    # 5. 模型特征处理
    print("\n5. 模型特征处理...")
    fusion_features = []
    for feat_name in pipe.fusion_feature_names:
        if feat_name in row:
            fusion_features.append(row[feat_name])
        else:
            fusion_features.append(0.0)

    print(f"   模型特征数量: {len(fusion_features)}")
    print(f"   前10个模型特征: {fusion_features[:10]}")

    # 6. 标准化
    print("\n6. 标准化...")
    x_array = np.array(fusion_features).reshape(1, -1)
    x_array_original = x_array.copy()
    x_array = (x_array - pipe.fusion_scaler_mean.numpy()) / pipe.fusion_scaler_scale.numpy()
    x = torch.tensor(x_array, dtype=torch.float32)

    print(f"   标准化前: {x_array_original[0, :5]}")
    print(f"   标准化后: {x_array[0, :5]}")

    # 7. 直接模型预测
    print("\n7. 直接模型预测...")
    with torch.no_grad():
        outputs = pipe.fusion(x)
        probs = torch.softmax(outputs, dim=1)
        benign_prob_direct = probs[0, 0].item()
        phishing_prob_direct = probs[0, 1].item()

    print(f"   直接预测良性概率: {benign_prob_direct:.4f}")
    print(f"   直接预测钓鱼概率: {phishing_prob_direct:.4f}")

    # 8. 通过predict_proba函数
    print("\n8. 通过predict_proba函数...")
    from phishguard_v1.models.fusion_model import predict_proba
    fusion_prob_func = predict_proba(pipe.fusion, x)[0,0].item()

    print(f"   predict_proba良性概率: {fusion_prob_func:.4f}")

    # 9. 最终推理结果
    print("\n9. 最终推理结果...")
    result = pipe.predict(features)

    print(f"   URL模型良性概率: {result['url_prob']:.4f}")
    print(f"   FusionDNN良性概率: {result['fusion_prob']:.4f}")
    print(f"   最终良性概率: {result['final_prob']:.4f}")
    print(f"   预测标签: {'良性' if result['label'] == 1 else '钓鱼'}")

    # 10. 期望结果
    print("\n10. 期望结果...")
    print(f"   期望: 钓鱼")
    print(f"   实际: {'良性' if result['label'] == 1 else '钓鱼'}")
    print(f"   正确: {'✅' if result['label'] == 0 else '❌'}")

if __name__ == "__main__":
    debug_phishing_prediction()