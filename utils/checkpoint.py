import os
import torch
from typing import Optional, Dict, Any


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float,
        ckpt_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")

    state_dict: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric
    }

    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state_dict, checkpoint_path)


def load_checkpoint(
        model: torch.nn.Module,
        ckpt_dir: str,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> tuple[int, float]:
    checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_metric = checkpoint["best_metric"]

    return epoch, best_metric


def save_epoch_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        current_metric: float,
        ckpt_dir: str
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")

    state_dict: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "current_metric": current_metric
    }

    torch.save(state_dict, checkpoint_path)