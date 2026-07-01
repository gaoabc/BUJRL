import torch
import torch.nn as nn
from .reconstruction_loss import ReconstructionLoss
from .alignment_loss import AlignmentLoss
from .survival_loss import SurvivalLoss

class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_recon = cfg.train.lambda_recon
        self.lambda_align = cfg.train.lambda_align
        self.lambda_surv = cfg.train.lambda_surv
        self.recon_loss = ReconstructionLoss()
        self.align_loss = AlignmentLoss()
        self.surv_loss = SurvivalLoss()

    def forward(self, outputs, batch):
        recon_p_loss = self.recon_loss(outputs["recon_p"], outputs["feat_p"], outputs["mask_p"])
        recon_g_loss = self.recon_loss(outputs["recon_g"], outputs["feat_g"], outputs["mask_g"])
        recon_loss = recon_p_loss + recon_g_loss
        align_p_loss = self.align_loss(outputs["fused_p"], outputs["feat_p"])
        align_g_loss = self.align_loss(outputs["fused_g"], outputs["feat_g"])
        align_loss = align_p_loss + align_g_loss
        surv_loss = self.surv_loss(outputs["hazard"], outputs["survival"], batch["time"], batch["event"])
        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_align * align_loss
            + self.lambda_surv * surv_loss
        )
        return {
            "total": total_loss,
            "recon": recon_loss,
            "align": align_loss,
            "surv": surv_loss
        }