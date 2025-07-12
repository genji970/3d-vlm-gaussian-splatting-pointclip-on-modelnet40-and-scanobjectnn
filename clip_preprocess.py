import torch
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"

class ProcessorGradientFlow(torch.nn.Module):
    def __init__(self, device=device):
        super().__init__()
        self.device = device
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize(mean=self.image_mean, std=self.image_std)
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.transform(images.to(self.device))