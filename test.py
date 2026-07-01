import torch
import numpy as np
from torch.utils.data import DataLoader
from config.config import cfg
from data.dataset import TCGA_Survival_Dataset, collate_fn
from models.bujrl import BUJRL
from utils.metrics import compute_c_index
from utils.checkpoint import load_checkpoint

if __name__ == "__main__":
    test_dataset = TCGA_Survival_Dataset(cfg, mode="test", missing_mode=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn
    )
    model = BUJRL(cfg).to(cfg.train.device)
    load_checkpoint(model, cfg.ckpt_dir, device=cfg.train.device)
    model.eval()
    all_preds = []
    all_times = []
    all_events = []
    with torch.no_grad():
        for batch in test_loader:
            pathology = [p.to(cfg.train.device) for p in batch["pathology"]]
            genomic = batch["genomic"].to(cfg.train.device)
            outputs = model(pathology, genomic)
            risk = 1 - outputs["survival"].mean(dim=-1)
            all_preds.append(risk.cpu())
            all_times.append(batch["time"].cpu())
            all_events.append(batch["event"].cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()
    cindex = compute_c_index(all_times, all_events, all_preds)
    print(f"Test C-Index: {cindex:.4f}")