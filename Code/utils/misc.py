import os
import random
import torch
import numpy as np
import torch.distributed as dist
from typing import Optional, Tuple, List

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_dirs(dirs: List[str]):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_device(data: dict, device: torch.device) -> dict:
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            data[k] = [x.to(device) for x in v]
    return data

def average_metrics_across_gpus(metrics: dict) -> dict:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    world_size = dist.get_world_size()
    for k, v in metrics.items():
        tensor = torch.tensor(v, device=get_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metrics[k] = (tensor / world_size).item()
    return metrics

def save_results(results: dict, save_path: str):
    with open(save_path, "w") as f:
        for k, v in results.items():
            if isinstance(v, (np.ndarray, list)):
                f.write(f"{k}: {len(v)}\n")
            else:
                f.write(f"{k}: {v}\n")

def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0