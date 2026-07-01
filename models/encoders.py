import torch
import torch.nn as nn
import torchvision.models as models
from typing import List

class WSIEncoder(nn.Module):
    def __init__(self, encoder_name: str = "resnet50", pretrained: bool = True, out_dim: int = 1024):
        super().__init__()
        self.encoder = getattr(models, encoder_name)(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        feats = []
        for patch in x:
            feat = self.encoder(patch)
            feat = feat.flatten(1)
            feats.append(feat)
        bag_feat = torch.stack([torch.mean(f, dim=0) for f in feats])
        return self.proj(bag_feat)

class GenomicEncoder(nn.Module):
    def __init__(self, in_dim: int = 1, hidden_dim: int = 1024, out_dim: int = 1024):
        super().__init__()
        self.snn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.SELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        feat = self.snn(x)
        return feat.view(B, N, -1)