from __future__ import annotations

import re
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phishguard_v1.features.feature_engineering import compute_feature_dict, FEATURE_COLUMNS

from .fusion_model import FusionDNN, AdvancedFusionDNN, predict_proba
from .url_transformer import URLTransformer
from ..config import settings

class InferencePipeline:
    def __init__(
        self,
        fusion_ckpt_path: str = "artifacts/fusion_advanced.pt",
        enable_fusion: bool = False,
        url_threshold: float | None = None,
        url_min_threshold: float | None = None,
        fusion_threshold: float | None = None,
        final_threshold: float | None = None,
    ):
        self.url_model = URLTransformer()
        self.enable_fusion = enable_fusion
        self.url_threshold = url_threshold if url_threshold is not None else settings.url_phish_threshold
        self.url_min_threshold = url_min_threshold if url_min_threshold is not None else settings.url_phish_min_threshold
        self.fusion_threshold = fusion_threshold if fusion_threshold is not None else settings.fusion_phish_threshold
        self.final_threshold = final_threshold if final_threshold is not None else settings.final_phish_threshold
        self.fusion = None
        try:
            if enable_fusion:
                import torch.serialization

                torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                ckpt = torch.load(fusion_ckpt_path, map_location="cpu", weights_only=False)
                # 检查是否为高级架构模型
                state_dict = ckpt.get('model_state_dict', ckpt)
                is_advanced = any(key.startswith('fc') for key in state_dict.keys())
                has_backbone = any(key.startswith("backbone") for key in state_dict.keys())
                has_fc = any(key.startswith("fc") for key in state_dict.keys())

                print(f"🔍 模型架构分析:")
                print(f"  - 有fc层: {has_fc}")
                print(f"  - 有backbone层: {has_backbone}")
                print(f"  - 高级架构: {is_advanced}")

                # 获取特征数量
                if "input_features" in ckpt:
                    num_features = ckpt["input_features"]
                    print(f"🎯 使用input_features: {num_features}")
                elif "num_features" in ckpt:
                    num_features = ckpt["num_features"]
                    print(f"🔧 使用num_features: {num_features}")
                elif "model_config" in ckpt and "num_features" in ckpt["model_config"]:
                    num_features = ckpt["model_config"]["num_features"]
                    print(f"🏗️ 使用model_config中的num_features: {num_features}")
                else:
                    num_features = len(ckpt.get("feature_names", FEATURE_COLUMNS))
                    print(f"📊 使用feature_names长度: {num_features}")

                if is_advanced:
                    # 检查是否为真实钓鱼网站训练的新模型
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    has_backbone = any(key.startswith("backbone") for key in state_dict.keys())
                    has_fc = any(key.startswith("fc") for key in state_dict.keys())
                    has_fc7 = any(key.startswith("fc7") for key in state_dict.keys())
                    has_bn5 = any(key.startswith("bn5") for key in state_dict.keys())

                    print(f"🏗️ 架构分析: has_backbone={has_backbone}, has_fc={has_fc}, has_fc7={has_fc7}, has_bn5={has_bn5}")

                    # 检查是否为真实钓鱼网站训练的新模型 (EnhancedPhishingDetector)
                    if has_fc7 and has_bn5 and not has_backbone:
                        print("🎯 创建真实钓鱼网站检测模型 (EnhancedPhishingDetector)")

                        class EnhancedPhishingDetector(nn.Module):
                            def __init__(self, input_dim):
                                super().__init__()
                                # 输入层
                                self.fc1 = nn.Linear(input_dim, 1024)
                                self.bn1 = nn.BatchNorm1d(1024)
                                self.dropout1 = nn.Dropout(0.4)

                                # 隐藏层
                                self.fc2 = nn.Linear(1024, 512)
                                self.bn2 = nn.BatchNorm1d(512)
                                self.dropout2 = nn.Dropout(0.3)

                                self.fc3 = nn.Linear(512, 256)
                                self.bn3 = nn.BatchNorm1d(256)
                                self.dropout3 = nn.Dropout(0.3)

                                self.fc4 = nn.Linear(256, 128)
                                self.bn4 = nn.BatchNorm1d(128)
                                self.dropout4 = nn.Dropout(0.2)

                                self.fc5 = nn.Linear(128, 64)
                                self.bn5 = nn.BatchNorm1d(64)
                                self.dropout5 = nn.Dropout(0.2)

                                self.fc6 = nn.Linear(64, 32)
                                self.bn6 = nn.BatchNorm1d(32)
                                self.dropout6 = nn.Dropout(0.1)

                                # 输出层
                                self.fc7 = nn.Linear(32, 2)  # 2分类：合法vs钓鱼

                                self._advanced = True

                            def forward(self, x):
                                x = torch.relu(self.bn1(self.fc1(x)))
                                x = self.dropout1(x)

                                x = torch.relu(self.bn2(self.fc2(x)))
                                x = self.dropout2(x)

                                x = torch.relu(self.bn3(self.fc3(x)))
                                x = self.dropout3(x)

                                x = torch.relu(self.bn4(self.fc4(x)))
                                x = self.dropout4(x)

                                x = torch.relu(self.bn5(self.fc5(x)))
                                x = self.dropout5(x)

                                x = torch.relu(self.bn6(self.fc6(x)))
                                x = self.dropout6(x)

                                x = self.fc7(x)
                                return x

                        self.fusion = EnhancedPhishingDetector(num_features)

                    elif has_backbone and has_fc:
                        print("🎯 创建高级架构模型 (忽略backbone层)")
                        # 创建纯高级架构模型，忽略backbone层
                        class HybridFusionDNN(nn.Module):
                            def __init__(self, input_dim):
                                super().__init__()
                                # 高级架构层 - 匹配训练时的架构
                                self.fc1 = nn.Linear(input_dim, 512)
                                self.bn1 = nn.BatchNorm1d(512)
                                self.dropout1 = nn.Dropout(0.3)
                                self.fc2 = nn.Linear(512, 256)
                                self.bn2 = nn.BatchNorm1d(256)
                                self.dropout2 = nn.Dropout(0.3)
                                self.fc3 = nn.Linear(256, 128)
                                self.bn3 = nn.BatchNorm1d(128)
                                self.dropout3 = nn.Dropout(0.2)
                                self.fc4 = nn.Linear(128, 64)
                                self.bn4 = nn.BatchNorm1d(64)
                                self.dropout4 = nn.Dropout(0.2)
                                self.fc5 = nn.Linear(64, 2)
                                self._advanced = True

                            def forward(self, x):
                                # 使用高级架构
                                x = self.fc1(x)
                                x = self.bn1(x)
                                x = F.relu(x)
                                x = self.dropout1(x)
                                x = self.fc2(x)
                                x = self.bn2(x)
                                x = F.relu(x)
                                x = self.dropout2(x)
                                x = self.fc3(x)
                                x = self.bn3(x)
                                x = F.relu(x)
                                x = self.dropout3(x)
                                x = self.fc4(x)
                                x = self.bn4(x)
                                x = F.relu(x)
                                x = self.dropout4(x)
                                x = self.fc5(x)
                                return x

                        self.fusion = HybridFusionDNN(num_features)
                    else:
                        print("🎯 创建标准高级架构模型")
                        self.fusion = FusionDNN(num_features=num_features)
                        self.fusion._use_advanced_architecture = True
                        self.fusion._advanced = True
                        # 重新初始化高级架构层
                        self.fusion.fc1 = nn.Linear(num_features, 512)
                        self.fusion.bn1 = nn.BatchNorm1d(512)
                        self.fusion.dropout1 = nn.Dropout(0.3)
                        self.fusion.fc2 = nn.Linear(512, 256)
                        self.fusion.bn2 = nn.BatchNorm1d(256)
                        self.fusion.dropout2 = nn.Dropout(0.3)
                        self.fusion.fc3 = nn.Linear(256, 128)
                        self.fusion.bn3 = nn.BatchNorm1d(128)
                        self.fusion.dropout3 = nn.Dropout(0.2)
                        self.fusion.fc4 = nn.Linear(128, 64)
                        self.fusion.bn4 = nn.BatchNorm1d(64)
                        self.fusion.dropout4 = nn.Dropout(0.2)
                        self.fusion.fc5 = nn.Linear(64, 2)
                else:
                    print("🎯 创建标准架构模型")
                    self.fusion = FusionDNN(num_features=num_features)

                if "model_state_dict" in ckpt:
                    # 对于混合架构，使用strict=False忽略backbone层
                    use_strict = not (has_backbone and has_fc)
                    self.fusion.load_state_dict(ckpt["model_state_dict"], strict=use_strict)
                elif "state_dict" in ckpt:
                    self.fusion.load_state_dict(ckpt["state_dict"], strict=True)
                else:
                    self.fusion.load_state_dict(ckpt, strict=True)

                self.fusion.eval()
                self.fusion_feature_names = ckpt.get("feature_names", FEATURE_COLUMNS)
                self.fusion_scaler_mean = torch.tensor(ckpt.get("scaler_mean", [0.0] * len(self.fusion_feature_names)))
                self.fusion_scaler_scale = torch.tensor(ckpt.get("scaler_scale", [1.0] * len(self.fusion_feature_names)))
                print(f"✅ 融合模型加载成功，特征数: {len(self.fusion_feature_names)}")
        except Exception as e:
            print(f"❌ 加载融合模型失败: {e}")
            self.fusion = None
            self.enable_fusion = False

    def _analyze_url_complexity(self, url: str) -> float:
        complexity_score = 0
        if len(url) > 50:
            complexity_score += 1
        if len(url) > 100:
            complexity_score += 1
        path_depth = url.count('/') - 2
        if path_depth > 2:
            complexity_score += 1
        if path_depth > 4:
            complexity_score += 1
        if '?' in url:
            complexity_score += 1
            query_params = url.split('?')[1]
            param_count = query_params.count('&') + 1
            if param_count > 2:
                complexity_score += 1
        special_chars = sum(1 for c in url if c in ['%', '&', '=', '+', ';'])
        if special_chars > 2:
            complexity_score += 1
        if any(ext in url.lower() for ext in ['.php', '.html', '.htm', '.asp', '.aspx', '.jsp']):
            complexity_score += 0.5
        id_patterns = re.findall(r'/[a-zA-Z0-9]{8,}/', url)
        if id_patterns:
            complexity_score += 1
        return complexity_score

    def _get_dynamic_weights(self, url: str, url_phish_prob: float, fusion_phish_prob: float) -> Dict[str, float]:
        complexity = self._analyze_url_complexity(url)
        base_url_weight = 0.6
        base_fusion_weight = 0.4
        if complexity >= 3:
            url_weight = base_url_weight - 0.2
            fusion_weight = base_fusion_weight + 0.2
        elif complexity >= 2:
            url_weight = base_url_weight - 0.1
            fusion_weight = base_fusion_weight + 0.1
        else:
            url_weight = base_url_weight
            fusion_weight = base_fusion_weight
        if url_phish_prob < 0.2:
            url_weight *= 0.6
            fusion_weight *= 1.4
        elif url_phish_prob > 0.85:
            url_weight *= 1.1
            fusion_weight *= 0.9
        if fusion_phish_prob > 0.8:
            fusion_weight *= 1.2
            url_weight *= 0.8
        elif fusion_phish_prob < 0.2:
            fusion_weight *= 0.7
            url_weight *= 1.3
        total_weight = url_weight + fusion_weight
        if total_weight == 0:
            return {"url": 0.5, "fusion": 0.5}
        url_weight /= total_weight
        fusion_weight /= total_weight
        return {"url": url_weight, "fusion": fusion_weight}

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        url = features["request_url"]
        url_phish_prob = float(self.url_model.score(url))
        url_benign_prob = 1.0 - url_phish_prob
        feature_dict = compute_feature_dict(features, features.get("html_feats"))
        fusion_feature_vector = [feature_dict.get(name, 0.0) for name in getattr(self, "fusion_feature_names", FEATURE_COLUMNS)]

        fusion_phish_prob = None
        fusion_weights = {"url": 1.0, "fusion": 0.0}
        final_phish_prob = url_phish_prob
        decision_stage = "url-only"

        if self.fusion is not None and self.enable_fusion and hasattr(self, 'fusion_feature_names'):
            x_array = np.array(fusion_feature_vector, dtype=np.float32).reshape(1, -1)
            x_array = (x_array - self.fusion_scaler_mean.numpy()) / self.fusion_scaler_scale.numpy()
            x = torch.tensor(x_array, dtype=torch.float32)
            probs = predict_proba(self.fusion, x)[0].cpu().numpy()
            fusion_benign_prob = float(probs[0])
            fusion_phish_prob = float(probs[1])
            fusion_weights = self._get_dynamic_weights(url, url_phish_prob, fusion_phish_prob)
            combined = fusion_weights["url"] * url_phish_prob + fusion_weights["fusion"] * fusion_phish_prob
            final_phish_prob = float(combined)
            decision_stage = "weighted-fusion"
        else:
            fusion_benign_prob = None

        url_primary = url_phish_prob >= self.url_threshold
        url_support = url_phish_prob >= self.url_min_threshold
        fusion_support = fusion_phish_prob is not None and fusion_phish_prob >= self.fusion_threshold

        if fusion_phish_prob is None:
            final_phish_prob = url_phish_prob
            decision_stage = "url-only"
        else:
            if url_primary and fusion_support:
                final_phish_prob = max(final_phish_prob, (url_phish_prob + fusion_phish_prob) / 2)
                decision_stage = "both-strong"
            elif url_primary:
                final_phish_prob = max(final_phish_prob, url_phish_prob)
                decision_stage = "url-strong"
            elif fusion_support and url_support:
                final_phish_prob = max(final_phish_prob, 0.4 * url_phish_prob + 0.6 * fusion_phish_prob)
                decision_stage = "fusion-confirm"
            elif fusion_support:
                final_phish_prob = max(final_phish_prob, fusion_phish_prob * 0.9)
                decision_stage = "fusion-only"
            else:
                final_phish_prob = min(final_phish_prob, min(self.url_threshold, self.fusion_threshold) - 1e-3)
                decision_stage = "below-threshold"

        # 若仅由 URL 模型支撑，而融合分支明显偏良性，则降权避免权威站点误报
        if (
            fusion_phish_prob is not None
            and fusion_phish_prob < 0.30
            and decision_stage in {"url-strong", "url-only", "below-threshold"}
        ):
            final_phish_prob = min(final_phish_prob, self.final_threshold - 1e-3)
            decision_stage = "url-downgraded"

        final_phish_prob = min(max(final_phish_prob, 0.0), 1.0)
        label = int(final_phish_prob >= self.final_threshold)

        return {
            "url_prob": float(url_phish_prob),
            "fusion_prob": float(fusion_phish_prob) if fusion_phish_prob is not None else None,
            "final_prob": float(final_phish_prob),
            "label": label,
            "details": {
                "url_benign_prob": float(url_benign_prob),
                "fusion_benign_prob": float(fusion_benign_prob) if fusion_phish_prob is not None else None,
                "fusion_weights": fusion_weights,
                "decision": decision_stage,
                "thresholds": {
                    "url": self.url_threshold,
                    "url_min": self.url_min_threshold,
                "fusion": self.fusion_threshold,
                "final": self.final_threshold,
                },
                "feature_snapshot": {name: feature_dict.get(name, 0.0) for name in getattr(self, "fusion_feature_names", FEATURE_COLUMNS)},
            },
        }
