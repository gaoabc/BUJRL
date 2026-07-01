import torch
import torch.nn as nn
import torch.nn.functional as F

class SurvivalHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 1024, n_intervals: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_intervals),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pool = x.mean(dim=1)
        hazard = self.head(x_pool)
        survival = torch.cumprod(1 - hazard, dim=-1)
        return hazard, survival