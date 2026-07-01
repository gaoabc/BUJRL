import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.max() == 0:
            return 0.0
        loss = self.mse(recon[mask.bool()], target[mask.bool()])
        return loss