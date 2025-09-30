from __future__ import annotations

import re
from typing import Dict, Any

import numpy as np
import torch

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
                if "input_features" in ckpt:
                    if any(key.startswith('fc') for key in ckpt.get('model_state_dict', ckpt).keys()):
                        self.fusion = AdvancedFusionDNN(num_features=ckpt["input_features"])
                    else:
                        self.fusion = FusionDNN(num_features=ckpt["input_features"])
                elif "num_features" in ckpt:
                    self.fusion = FusionDNN(num_features=ckpt["num_features"])
                else:
                    self.fusion = FusionDNN(num_features=len(FEATURE_COLUMNS))

                if "model_state_dict" in ckpt:
                    self.fusion.load_state_dict(ckpt["model_state_dict"], strict=True)
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
