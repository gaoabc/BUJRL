import torch
import torch.nn as nn
from .encoders import WSIEncoder, GenomicEncoder
from .gamr import GatedAdaptiveModalityReconstructor
from .moe import MixtureOfExperts
from .caf import CrossModalAdaptiveFusion
from .survival_head import SurvivalHead

class BUJRL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wsi_encoder = WSIEncoder(
            encoder_name=cfg.model.encoder_name,
            pretrained=cfg.model.pretrained,
            out_dim=cfg.data.feature_dim
        )
        self.gen_encoder = GenomicEncoder(out_dim=cfg.data.feature_dim)
        self.gamr = GatedAdaptiveModalityReconstructor(
            dim=cfg.data.feature_dim,
            num_layers=cfg.model.gamma_layers
        )
        self.moe_path = MixtureOfExperts(dim=cfg.data.feature_dim)
        self.moe_gen = MixtureOfExperts(dim=cfg.data.feature_dim)
        self.caf = CrossModalAdaptiveFusion(dim=cfg.data.feature_dim)
        self.survival_head = SurvivalHead(in_dim=cfg.data.feature_dim * 2)

    def _reconstruct_missing(self, feat_p: torch.Tensor, feat_g: torch.Tensor):
        mask_p = (feat_p.sum(dim=-1) == 0).unsqueeze(-1).float()
        mask_g = (feat_g.sum(dim=-1) == 0).unsqueeze(-1).float()
        recon_p = self.gamr(feat_g, feat_p) if mask_p.max() > 0 else feat_p
        recon_g = self.gamr(feat_p, feat_g) if mask_g.max() > 0 else feat_g
        return recon_p, recon_g, mask_p, mask_g

    def forward(self, pathology, genomic):
        feat_p = self.wsi_encoder(pathology).unsqueeze(1)
        feat_g = self.gen_encoder(genomic)
        recon_p, recon_g, mask_p, mask_g = self._reconstruct_missing(feat_p, feat_g)
        fused_p = self.moe_path(recon_p)
        fused_g = self.moe_gen(recon_g)
        multimodal_feat = self.caf(fused_p, fused_g)
        hazard, survival = self.survival_head(multimodal_feat)
        return {
            "hazard": hazard,
            "survival": survival,
            "recon_p": recon_p,
            "recon_g": recon_g,
            "feat_p": feat_p,
            "feat_g": feat_g,
            "mask_p": mask_p,
            "mask_g": mask_g
        }