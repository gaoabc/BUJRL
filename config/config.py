from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class DataConfig:
    data_root: str = "./tcga_data"
    dataset_name: str = "BLCA"
    input_size: int = 256
    wsi_magnification: int = 20
    n_genomic_bins: int = 6
    feature_dim: int = 1024
    num_workers: int = 8

@dataclass
class ModelConfig:
    encoder_name: str = "resnet50"
    pretrained: bool = True
    gamma_layers: int = 3
    num_experts: int = 2
    hidden_dim: int = 1024
    dropout_rate: float = 0.1
    activation: str = "gelu"
    use_gate: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    gradient_accum_steps: int = 32
    lambda_recon: float = 1.0
    lambda_align: float = 0.5
    lambda_surv: float = 1.0
    patience: int = 10
    device: str = "cuda"
    seed: int = 42

@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    log_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"
    result_dir: str = "./results"

cfg = Config()