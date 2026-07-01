import torch
import torch.nn as nn

class SurvivalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hazard: torch.Tensor, survival: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        batch_size = time.size(0)
        loss = 0.0
        for i in range(batch_size):
            t = time[i]
            e = event[i]
            s = survival[i, :t]
            h = hazard[i, t]
            if e == 1:
                loss += -(torch.log(s).sum() + torch.log(h))
            else:
                loss += -(torch.log(s).sum())
        return loss / batch_size