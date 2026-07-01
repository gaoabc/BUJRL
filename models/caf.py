import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAdaptiveFusion(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )

    def forward(self, feat_p: torch.Tensor, feat_g: torch.Tensor) -> torch.Tensor:
        pool_p = feat_p.mean(dim=1)
        pool_g = feat_g.mean(dim=1)
        concat = torch.cat([pool_p, pool_g], dim=-1)
        weights = F.softmax(self.weight_net(concat), dim=-1)
        w_p, w_g = weights[:, 0:1], weights[:, 1:2]
        fused = torch.cat([w_p.unsqueeze(-1) * feat_p, w_g.unsqueeze(-1) * feat_g], dim=-1)
        return fused