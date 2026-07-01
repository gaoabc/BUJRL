import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAdaptiveModalityReconstructor(nn.Module):
    def __init__(self, dim: int = 1024, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, src_feat: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = src_feat.shape
        query = torch.zeros_like(tgt_mask)
        for layer in self.cross_attn_layers:
            query, _ = layer(query, src_feat, src_feat)
        src_pool = src_feat.mean(dim=1)
        query_pool = query.mean(dim=1)
        gate_weight = self.gate(torch.cat([src_pool, query_pool], dim=-1)).unsqueeze(1)
        recon_feat = query * gate_weight
        return recon_feat