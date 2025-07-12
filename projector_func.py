import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import fill_depth_map

class FullGaussianProjector(nn.Module):
    def __init__(self, hidden_dim, cov3d_hyperparameter, sigma, steps, distance_cutoff, importance_score,relative_normalized_weight, image_size=224, max_gaussians = 1000, device='cuda'):
        super().__init__()
        self.image_size = image_size
        self.device = torch.device(device)
        self.views = ['front', 'top', 'left']
        self.max_gaussians =max_gaussians
        self.cov3d_hyperparameter = cov3d_hyperparameter
        self.sigma = sigma
        self.steps = steps
        self.distance_cutoff = distance_cutoff
        self.importance_score = importance_score
        self.relative_normalized_weight = relative_normalized_weight

        Js = torch.stack([
            torch.tensor([[1, 0, 0], [0, 1, 0]]),  # front
            torch.tensor([[1, 0, 0], [0, 0, 1]]),  # top
            torch.tensor([[0, 0, 1], [0, 1, 0]]),  # left
        ], dim=0).float()
        self.register_buffer('Js', Js)

        self.importance = nn.Parameter(torch.ones(1, max_gaussians))

    def forward(self, position, cov3d, opacity):
        B, N, _ = position.shape
        #print(f"[forward] position: {position.shape}, cov3d: {cov3d.shape}, opacity: {opacity.shape}")
        H = W = self.image_size
        H_half, W_half = H // 2, W // 2
        eps = 1e-6
        device = position.device

        if N > self.max_gaussians:
            position = position[:, :self.max_gaussians]
            cov3d = cov3d[:, :self.max_gaussians]
            opacity = opacity[:, :self.max_gaussians]
            N = self.max_gaussians

        Js_all = self.Js.to(device)  # (3, 2, 3)
        JT_all = Js_all.transpose(1, 2)

        yy, xx = torch.meshgrid(
            torch.arange(H_half, device=device),
            torch.arange(W_half, device=device),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).float()[None, None]  # (1, 1, H_half, W_half, 2)

        importance_score = self.importance[:, :N].expand(B, -1)  # (B, N)
        importance_score = importance_score[:, :, None, None].clamp(self.importance_score[0], self.importance_score[1])  # (B, N, 1, 1)

        depth_maps = []

        for v in range(3):
            J = Js_all[v].unsqueeze(0).expand(B, -1, -1)   # (B, 2, 3)
            JT = JT_all[v].unsqueeze(0).expand(B, -1, -1)  # (B, 3, 2)

            cov3d = (cov3d + cov3d.transpose(-1, -2)) / 2
            cov3d = cov3d + eps * torch.eye(3, device=cov3d.device).unsqueeze(0).unsqueeze(0)
            cov3d = cov3d / (cov3d.norm(dim=(-1, -2), keepdim=True) + eps) * self.cov3d_hyperparameter  # 🔧 가우시안 크기 조정

            cov3d_inv = torch.linalg.inv(cov3d)  # (B, N, 3, 3)
            J_exp = J.unsqueeze(1).expand(-1, N, -1, -1)     # (B, N, 2, 3)
            JT_exp = JT.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, 3, 2)
            cov2d_inv = J_exp @ cov3d_inv @ JT_exp          # (B, N, 2, 2)
            cov2d_inv = cov2d_inv + torch.eye(2, device=cov2d_inv.device).unsqueeze(0).unsqueeze(0) * eps * 1e-4

            proj_xy = (J @ position.transpose(1, 2)).transpose(1, 2)  # (B, N, 2)

            min_xy = proj_xy.amin(dim=1, keepdim=True)  # (B, 1, 2)
            max_xy = proj_xy.amax(dim=1, keepdim=True)  # (B, 1, 2)
            range_xy = max_xy - min_xy + 1e-6
            margin_ratio = 0.5
            min_xy = min_xy - margin_ratio * range_xy
            max_xy = max_xy + margin_ratio * range_xy
            range_xy = max_xy - min_xy + 1e-6  # 다시 계산
            proj_xy_norm = (proj_xy - min_xy) / range_xy
            xy_pix = (proj_xy_norm * (W_half - 1)).clamp(0, W_half - 1)

            delta = grid - xy_pix[:, :, None, None, :]  # (B, N, H_half, W_half, 2)

            mask = (delta.norm(dim=-1) < self.distance_cutoff)  # (B, N, H_half, W_half)
            mask = mask.float()  # 먼저 float으로 바꾸고
            mask = F.interpolate(mask, size=(H, W), mode='nearest') > 0.5  # 다시 bool로

            tmp = torch.einsum('bnhwj,bnjk->bnhwk', delta, cov2d_inv)
            exponent_half = -torch.sum(tmp * delta, dim=-1)

            exponent = F.interpolate(exponent_half, size=(H, W), mode='bilinear', align_corners=False).clamp(min=-20.0, max=0)

            weight = opacity[:, :, None, None] * importance_score * torch.exp(exponent)
            weight = weight.masked_fill(~mask, 0.0)

            w_min = weight.amin(dim=1, keepdim=True)  # (B, 1, H, W)
            w_max = weight.amax(dim=1, keepdim=True)  # (B, 1, H, W)
            normalized_weight_1 = (weight - w_min) / (w_max - w_min + eps)
            normalized_weight_2 = weight / (weight.sum(dim=1, keepdim=True) + eps)

            normalized_weight = normalized_weight_1 * self.relative_normalized_weight + normalized_weight_2

            depth_map = torch.sum(normalized_weight, dim=1)  # (B, H, W)
            filled_map = fill_depth_map(depth_map, sigma=self.sigma, steps=self.steps)

            depth_maps.append(filled_map)

        depth_map_all = torch.stack(depth_maps, dim=1)  # (B, 3, H, W)

        d = F.interpolate(depth_map_all, size=(224, 224), mode='bilinear', align_corners=False)  # (B, 3, 224, 224)
        d_flat = d.view(B, 3, -1)
        d = (d - d.mean()) / (d.std() + eps)  # (B, 3, 224, 224)
        rgb_views=d
        return rgb_views