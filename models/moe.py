import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim: int = 1024, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SoftRouter(nn.Module):
    def __init__(self, dim: int = 1024, num_experts: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pool = x.mean(dim=1)
        return F.softmax(self.net(x_pool), dim=-1)

class MixtureOfExperts(nn.Module):
    def __init__(self, dim: int = 1024, num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.router = SoftRouter(dim, num_experts)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        weights = self.router(feat)
        expert_outs = []
        for i, exp in enumerate(self.experts):
            expert_outs.append(exp(feat).unsqueeze(1))
        expert_outs = torch.cat(expert_outs, dim=1)
        fused = torch.einsum("bne,ben->ben", weights.unsqueeze(-1), expert_outs)
        return fused.sum(dim=1)