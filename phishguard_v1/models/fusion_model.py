from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class FusionDNN(nn.Module):
    def __init__(self, num_features: int, url_vec_dim: int = 0, vision_dim: int = 0):
        super().__init__()
        input_dim = num_features + url_vec_dim + vision_dim

        # 支持两种架构：新的高级架构和旧的backbone架构
        if hasattr(self, '_use_advanced_architecture'):
            # 高级架构 - 用于36特征模型
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.dropout1 = nn.Dropout(0.3)

            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.dropout2 = nn.Dropout(0.3)

            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.dropout3 = nn.Dropout(0.2)

            self.fc4 = nn.Linear(64, 32)
            self.bn4 = nn.BatchNorm1d(32)
            self.dropout4 = nn.Dropout(0.2)

            self.fc5 = nn.Linear(32, 2)

            self._advanced = True
        else:
            # 标准架构 - 用于兼容性
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2),
            )
            self._advanced = False

    def forward(self, x):
        if self._advanced:
            # 高级架构的前向传播
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
        else:
            # 标准架构的前向传播
            return self.backbone(x)

class AdvancedFusionDNN(nn.Module):
    """高级FusionDNN模型，支持36个特征"""
    def __init__(self, num_features: int):
        super(AdvancedFusionDNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
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

def predict_proba(model: FusionDNN, feats_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(feats_tensor)
        return F.softmax(logits, dim=-1)
