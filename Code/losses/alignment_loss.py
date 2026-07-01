import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fused: torch.Tensor, target: torch.Tensor) -> torch.T:
        fused_norm = F.log_softmax(fused, dim=-1)
        target_norm = F.softmax(target, dim=-1)
        loss = F.kl_div(fused_norm, target_norm, reduction="batchmean")
        return loss