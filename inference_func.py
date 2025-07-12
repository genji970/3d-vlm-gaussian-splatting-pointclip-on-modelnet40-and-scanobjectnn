import torch
from torch.nn import functional as F
import clip
from gaussian_splatting_func import *
from projector_func import *
from adapter import *

@torch.no_grad()
def inference_from_checkpoint(checkpoint_path,loader, processor_gradient_flow, clip_model, classnames, device,hidden_dim = 512,distance_cutoff = 0.9,importance_score = (0.5,1.0),relative_normalized_weight = 0.1):
    gaussianizer = PointCloudToGaussian(k=4,
                                        opacity_clip=(0.4, 1.0),
                                        scale_clip=(0.05, 0.4),
                                        hidden_dim=hidden_dim).to(device)
    projector = FullGaussianProjector(hidden_dim=hidden_dim,
                                      cov3d_hyperparameter=2.8,
                                      sigma=2,
                                      steps=100,
                                      distance_cutoff=distance_cutoff,
                                      importance_score=importance_score,
                                      relative_normalized_weight=relative_normalized_weight).to(device)
    inter_view_adapter = ClassConditionalInterViewAdapter().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gaussianizer.load_state_dict(checkpoint['gaussianizer'])
    projector.load_state_dict(checkpoint['projector'])
    inter_view_adapter.load_state_dict(checkpoint['inter_view_adapter'])

    gaussianizer.eval()
    projector.eval()
    inter_view_adapter.eval()

    # ⬛ CLIP 텍스트 임베딩
    text_tokens = clip.tokenize(classnames).to(device)  # (40, token_len)
    text_features = clip_model.encode_text(text_tokens).float()  # (40, 512)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_preds = []

    for batch in loader:
        pointclouds, labels = batch
        labels = labels.to(device)
        pointclouds = pointclouds.to(device)
        B = pointclouds.size(0)

        # ⬛ 정규화
        mean = pointclouds.mean(dim=1, keepdim=True)
        std = pointclouds.std(dim=1, keepdim=True).clamp(min=1e-2)
        normed_pc = (pointclouds - mean) / std

        # ⬛ Gaussian → Depth
        pos, cov, opa, _, _ = gaussianizer(normed_pc)
        depth_per_view = projector(pos, cov, opa)  # (B, 3, H, W)
        depth_per_view = depth_per_view / (depth_per_view.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)

        # ⬛ Depth → RGB (grayscale 3채널 확장)
        rgb_views = depth_per_view.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # (B, 3, 3, H, W)

        # ⬛ (B, 3, 3, H, W) → (3B, 3, H, W)
        rgb_views = rgb_views.permute(1, 0, 2, 3, 4).reshape(3 * B, 3, *rgb_views.shape[-2:])  # (3B, 3, H, W)

        # ⬛ 전처리 → CLIP 이미지 인코딩
        processed_views = processor_gradient_flow(rgb_views)  # (3B, 3, 224, 224)
        clip_encoded = clip_model.encode_image(processed_views).float()  # (3B, 512)

        # ⬛ 다시 (B, 3, 512)로 복원
        clip_features = clip_encoded.view(3, B, 512).permute(1, 0, 2)  # (B, 3, 512)

        # ⬛ Adapter + 가중치 통합
        adapted = inter_view_adapter(clip_features, labels)  # (B, 3, 512)
        weights = torch.softmax(adapted.mean(dim=-1), dim=1)  # (B, 3)
        image_features = (adapted * weights.unsqueeze(-1)).sum(dim=1)  # (B, 512)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ⬛ cosine similarity 기반 클래스 예측
        logits = 100. * image_features @ text_features.T  # (B, 40)
        preds = logits.argmax(dim=1)  # (B,)

        all_preds.append(preds.cpu())

    return torch.cat(all_preds)  # (총 샘플 수,)