import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .preprocess import WSIPreprocessor, GenomicPreprocessor, survival_discretization

class TCGA_Survival_Dataset(Dataset):
    def __init__(self, cfg, mode: str = "train", missing_mode: Optional[str] = None):
        self.cfg = cfg.data
        self.mode = mode
        self.missing_mode = missing_mode
        self.wsi_prep = WSIPreprocessor(patch_size=cfg.data.input_size)
        self.gen_prep = GenomicPreprocessor(n_bins=cfg.data.n_genomic_bins)
        self._load_metadata()

    def _load_metadata(self):
        meta_path = os.path.join(self.cfg.data_root, self.cfg.dataset_name, "metadata.csv")
        self.meta = pd.read_csv(meta_path)
        self.samples = self.meta[self.meta["split"] == self.mode].reset_index(drop=True)

    def _load_wsi(self, slide_id: str) -> torch.Tensor:
        slide_path = os.path.join(self.cfg.data_root, self.cfg.dataset_name, "slides", f"{slide_id}.svs")
        patches = self.wsi_prep.extract_patches(slide_path)
        if len(patches) == 0:
            return torch.zeros(1, self.cfg.feature_dim)
        patches = np.stack(patches).astype(np.float32) / 255.0
        patches = torch.from_numpy(patches).permute(0, 3, 1, 2)
        return patches

    def _load_genomic(self, sample_id: str) -> torch.Tensor:
        gen_path = os.path.join(self.cfg.data_root, self.cfg.dataset_name, "genomic", f"{sample_id}.csv")
        gen_df = pd.read_csv(gen_path)
        gen_feat = self.gen_prep.bin_genomic_data(gen_df)
        return torch.from_numpy(gen_feat).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        wsi_feat = self._load_wsi(row["slide_id"])
        gen_feat = self._load_genomic(row["sample_id"])
        time = row["time"]
        event = row["event"]
        disc_time, disc_event = survival_discretization(np.array([time]), np.array([event]))
        if self.missing_mode == "missing_pathology":
            wsi_feat = torch.zeros_like(wsi_feat)
        if self.missing_mode == "missing_genomic":
            gen_feat = torch.zeros_like(gen_feat)
        return {
            "pathology": wsi_feat,
            "genomic": gen_feat,
            "time": torch.tensor(disc_time[0], dtype=torch.long),
            "event": torch.tensor(disc_event[0], dtype=torch.float32),
            "sample_id": row["sample_id"]
        }

def collate_fn(batch):
    pathology = [item["pathology"] for item in batch]
    genomic = torch.stack([item["genomic"] for item in batch])
    time = torch.stack([item["time"] for item in batch])
    event = torch.stack([item["event"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    return {
        "pathology": pathology,
        "genomic": genomic,
        "time": time,
        "event": event,
        "sample_ids": sample_ids
    }