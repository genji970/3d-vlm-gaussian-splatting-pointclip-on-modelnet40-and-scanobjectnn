import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import naive_fps

class PointCloudToGaussian(nn.Module):
    def __init__(self, k, opacity_clip, scale_clip, hidden_dim):
        super().__init__()
        self.k = k
        self.opacity_clip = opacity_clip
        self.scale_clip_0 , self.scale_clip_1 =scale_clip

        self.rotation_predictor = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # axis-angle
        )

        self.scale_predictor = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softplus(),
            nn.Hardtanh(self.scale_clip_0, self.scale_clip_1)
        )  #    -> scale은 양수여야한다

    def compute_neighbors(self, points):
        B, N, _ = points.shape
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(self.k + 1, largest=False).indices[:, :, 1:]  # 자기 자신 제외
        return knn_idx, dists

    def compute_opacity(self, points, neighbor_idx):
        B, N, _ = points.shape
        k = neighbor_idx.shape[-1]
        batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, N, k)
        neighbors = points[batch_idx, neighbor_idx]  # (B, N, k, 3)
        center = points.unsqueeze(2)                 # (B, N, 1, 3)

        # ✅ 1. 선형성 기반: covariance + SVD
        centered = neighbors - center  # (B, N, k, 3)
        cov_matrix = torch.matmul(centered.transpose(2, 3), centered) / (k - 1 + 1e-6)
        _, S, _ = torch.svd(cov_matrix)  # (B, N, 3)
        λ1, λ2, λ3 = S[:, :, 0], S[:, :, 1], S[:, :, 2]
        linearness = λ1 / (λ2 + λ3 + 1e-6)  # (B, N)
        norm_linearness = (linearness - linearness.amin(dim=1, keepdim=True)) / (
            linearness.amax(dim=1, keepdim=True) - linearness.amin(dim=1, keepdim=True) + 1e-6
        )

        # ✅ 2. 밀도 기반: 반지름 내 점 개수
        dists_squared = ((neighbors - center) ** 2).sum(dim=3)  # (B, N, k)
        radius = self.density_radius if hasattr(self, "density_radius") else 0.1  # optional 하이퍼파라미터
        density = (dists_squared < radius ** 2).float().sum(dim=2) / k  # 비율 (B, N)
        norm_density = (density - density.amin(dim=1, keepdim=True)) / (
            density.amax(dim=1, keepdim=True) - density.amin(dim=1, keepdim=True) + 1e-6
        )

        # ✅ 3. 결합: 선형성 × 밀도
        combined = norm_linearness * (norm_density** 0.45 )  # (B, N)

        # ✅ 4. clamp 후 반환
        opacity = combined.clamp(*self.opacity_clip)  # (B, N)
        return opacity

    def axis_angle_to_rotation_matrix(self, axis_angle):  # -> Rodrigues 회전 공식
        B, N, _ = axis_angle.shape

        angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (B, N, 1)
        axis = axis_angle / (angle + 1e-6)  # (B, N, 3)

        x, y, z = axis.unbind(-1)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1.0 - ca

        # (B, N, 3) × (B, N, 1) → broadcasting ok
        row0 = torch.stack([
            ca[..., 0] + x * x * C[..., 0],
            x * y * C[..., 0] - z * sa[..., 0],
            x * z * C[..., 0] + y * sa[..., 0]
        ], dim=-1)

        row1 = torch.stack([
            y * x * C[..., 0] + z * sa[..., 0],
            ca[..., 0] + y * y * C[..., 0],
            y * z * C[..., 0] - x * sa[..., 0]
        ], dim=-1)

        row2 = torch.stack([
            z * x * C[..., 0] - y * sa[..., 0],
            z * y * C[..., 0] + x * sa[..., 0],
            ca[..., 0] + z * z * C[..., 0]
        ], dim=-1)

        R = torch.stack([row0, row1, row2], dim=-2)  # (B, N, 3, 3)
        return R

    def forward(self, pointclouds: torch.Tensor):
        pointclouds = pointclouds.float().contiguous()
        B, N, _ = pointclouds.shape

        # ✅ optional: downsample
        sample_ratio = 0.1
        sampled_N = int(N * sample_ratio)
        idx = naive_fps(pointclouds, sampled_N)  # (B, sampled_N)
        sampled = torch.gather(pointclouds, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, sampled_N, 3)

        # ✅ optional: jitter
        jitter_std = 0.01
        noise = torch.randn_like(sampled) * jitter_std
        pointclouds = torch.cat([sampled, sampled + noise], dim=1)  # (B, sampled_N * 2, 3)

        # ✅ 이 시점에서 N이 바뀌었으므로 다시 갱신
        B, N, _ = pointclouds.shape

        # ✅ 이웃 인덱스 및 거리 계산
        neighbor_idx, _ = self.compute_neighbors(pointclouds)  # (B, N, k)

        # ✅ 이웃 좌표 추출
        batch_idx = torch.arange(B, device=pointclouds.device).view(B, 1, 1).expand(B, N, self.k)
        neighbors = pointclouds[batch_idx, neighbor_idx]        # (B, N, k, 3)

        # ✅ 평균을 통한 weighted center 대체
        weighted_center = neighbors.mean(dim=2)  # (B, N, 3)

        # ✅ 회전 및 스케일 예측
        axis_angle = self.rotation_predictor(weighted_center)       # (B, N, 3)
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)        # (B, N, 1)

        default_axis = torch.tensor([1., 0., 0.], device=axis_angle.device).view(1, 1, 3)
        too_small = angle < 1e-4

        safe_angle = torch.where(too_small, torch.zeros_like(angle), angle.clamp(max=3.14))
        safe_axis = torch.where(
            too_small.expand(-1, -1, 3),
            default_axis.expand_as(axis_angle),
            axis_angle / (angle + 1e-6)
        )
        safe_axis_angle = safe_axis * safe_angle
        #print(f"[Rotation] safe_axis_angle: {safe_axis_angle.shape}")

        scale = self.scale_predictor(pointclouds)                    # (B, N, 3)

        S = torch.diag_embed(scale)   # (B, N, 3, 3)
        R = self.axis_angle_to_rotation_matrix(safe_axis_angle)      # (B, N, 3, 3)

        cov = R @ (S @ S) @ R.transpose(-1, -2)                       # (B, N, 3, 3)
        opacity = self.compute_opacity(pointclouds, neighbor_idx)    # (B, N)

        return pointclouds, cov, opacity, R, scale