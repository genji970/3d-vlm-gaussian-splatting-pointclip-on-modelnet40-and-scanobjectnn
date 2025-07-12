import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassConditionalInterViewAdapter(nn.Module):
    def __init__(self, in_dim=512, bottleneck_dim=256, num_views=3, num_classes=40, class_embed_dim=128):
        super().__init__()
        self.num_views = num_views
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        self.global_fc = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, in_dim)
        )

        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, in_dim)
            ) for _ in range(num_views)
        ])

        # 각 view마다 클래스 조건부 weight 계산용 block
        self.view_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + class_embed_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, 1),
                nn.Sigmoid()  # 0~1 가중치
            ) for _ in range(num_views)
        ])

    def forward(self, view_features, class_labels):
        B = view_features.size(0)
        global_feature = view_features.mean(dim=1)  # (B, 512)
        global_transformed = self.global_fc(global_feature)  # (B, 512)

        class_embed = self.class_embed(class_labels)  # (B, class_embed_dim)

        adapted = []
        for v in range(self.num_views):
            feat = view_features[:, v]  # (B, 512)
            residual = self.residuals[v](feat)  # (B, 512)

            joint = torch.cat([feat, class_embed], dim=1)  # (B, 512 + class_embed_dim)
            weight = self.view_weights[v](joint)  # (B, 1), 값은 0~1

            adapted_feat = (feat + residual + global_transformed) * weight  # (B, 512)
            adapted.append(adapted_feat)
        return torch.stack(adapted, dim=1)  # (B, V, 512)