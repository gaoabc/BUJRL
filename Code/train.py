import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from config.config import cfg
from data.dataset import TCGA_Survival_Dataset, collate_fn
from models.bujrl import BUJRL
from losses.total_loss import TotalLoss
from engine.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    set_seed(cfg.train.seed)
    train_dataset = TCGA_Survival_Dataset(cfg, mode="train")
    val_dataset = TCGA_Survival_Dataset(cfg, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn
    )
    model = BUJRL(cfg).to(cfg.train.device)
    loss_fn = TotalLoss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    trainer = Trainer(model, loss_fn, train_loader, val_loader, optimizer, cfg)
    trainer.run()