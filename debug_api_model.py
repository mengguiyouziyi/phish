#!/usr/bin/env python3
"""
调试当前API的模型状态
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from phishguard_v1.models.inference import InferencePipeline

def debug_current_model():
    """调试当前模型的内部状态"""
    print("🔍 调试当前模型状态...")

    # 使用与API相同的配置创建管道
    pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_balanced_v2.pt", enable_fusion=True)

    print(f"✅ URL模型: {pipe.url_model.model_id}")
    print(f"✅ Fusion模型: {'已加载' if pipe.fusion is not None else '未加载'}")

    if pipe.fusion is not None and hasattr(pipe, 'fusion_feature_names'):
        print(f"✅ 特征数量: {len(pipe.fusion_feature_names)}")
        print(f"✅ 标准化器均值: {pipe.fusion_scaler_mean}")
        print(f"✅ 标准化器标准差: {pipe.fusion_scaler_scale}")

        # 模拟百度的特征
        features = {
            "request_url": "https://www.baidu.com",
            "final_url": "https://www.baidu.com",
            "status_code": 200,
            "content_type": "text/html",
            "bytes": 227,
            "url_feats": {
                "url_len": 21,
                "host_len": 13,
                "path_len": 1,
                "num_digits": 0,
                "num_letters": 16,
                "num_specials": 5,
                "num_dots": 2,
                "num_hyphen": 0,
                "num_slash": 2,
                "num_qm": 0,
                "num_at": 0,
                "num_pct": 0,
                "has_ip": False,
                "subdomain_depth": 1,
                "tld_suspicious": 0,
                "has_punycode": 0,
                "scheme_https": 1,
                "query_len": 0,
                "fragment_len": 0
            }
        }

        print(f"\n📊 测试特征:")
        print(f"  bytes: {features['bytes']}")

        # 测试预测
        result = pipe.predict(features)
        print(f"\n📊 预测结果:")
        print(f"  URL模型良性概率: {result['url_prob']:.4f}")
        print(f"  FusionDNN良性概率: {result['fusion_prob']:.4f}")
        print(f"  最终良性概率: {result['final_prob']:.4f}")
        print(f"  标签: {result['label']}")

        # 详细测试FusionDNN
        print(f"\n🧠 详细FusionDNN测试:")

        # 准备特征
        row = {}
        uf = features.get("url_feats", {})

        # URL numeric features
        for k in ["url_len","host_len","path_len","num_digits","num_letters","num_specials","num_dots","num_hyphen","num_slash","num_qm","num_at","num_pct","subdomain_depth","query_len","fragment_len"]:
            row[k] = float(uf.get(k, 0))
        for k in ["has_ip","tld_suspicious","has_punycode","scheme_https"]:
            row[k] = float(1 if uf.get(k, 0) else 0)

        # HTTP response features
        row["status_code"] = float(features.get("status_code") or 200)

        # 对于bytes特征，使用与训练时相同的处理逻辑
        # 模型训练时bytes特征统一设置为1024
        row["bytes"] = 1024.0

        print(f"  原始特征: {row}")

        # 使用模型定义的特征
        fusion_features = []
        for feat_name in pipe.fusion_feature_names:
            if feat_name in row:
                fusion_features.append(row[feat_name])
            else:
                fusion_features.append(0.0)

        print(f"  模型输入特征: {fusion_features}")

        # 标准化特征
        x_array = np.array(fusion_features).reshape(1, -1)
        print(f"  标准化前: {x_array}")

        x_array = (x_array - pipe.fusion_scaler_mean.numpy()) / pipe.fusion_scaler_scale.numpy()
        print(f"  标准化后: {x_array}")

        # 检查是否有极端值
        for i, val in enumerate(x_array[0]):
            if abs(val) > 10:
                print(f"  ⚠️  特征 {i} ({pipe.fusion_feature_names[i]}): 极端值 {val:.2f}")

        x = torch.tensor(x_array, dtype=torch.float32)

        # 预测
        with torch.no_grad():
            outputs = pipe.fusion(x)
            probs = torch.softmax(outputs, dim=1)
            print(f"  FusionDNN原始输出: {outputs}")
            print(f"  FusionDNN概率: {probs}")
            print(f"  FusionDNN良性概率: {probs[0, 0].item():.4f}")
            print(f"  FusionDNN钓鱼概率: {probs[0, 1].item():.4f}")
    else:
        print("❌ FusionDNN模型未正确加载")

if __name__ == "__main__":
    debug_current_model()