import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Any, Optional, Tuple


class BaseTransform:
    def __init__(self, mode: str = "train"):
        self.mode = mode

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


class WSITransform(BaseTransform):
    def __init__(
        self,
        input_size: int = 256,
        mode: str = "train",
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        super().__init__(mode)
        self.input_size = input_size
        self.mean = mean
        self.std = std

        self.train_transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.2)
        ])

        self.val_test_transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, patch: np.ndarray) -> torch.Tensor:
        pil_img = Image.fromarray(patch.astype(np.uint8))
        if self.mode == "train":
            return self.train_transform(pil_img)
        return self.val_test_transform(pil_img)


class GenomicTransform(BaseTransform):
    def __init__(
        self,
        mode: str = "train",
        noise_std: float = 0.01,
        clip_percentile: float = 99.9,
        mask_prob: float = 0.1
    ):
        super().__init__(mode)
        self.noise_std = noise_std
        self.clip_percentile = clip_percentile
        self.mask_prob = mask_prob

    def clip_outliers(self, data: np.ndarray) -> np.ndarray:
        upper = np.percentile(data, self.clip_percentile)
        lower = np.percentile(data, 100 - self.clip_percentile)
        return np.clip(data, lower, upper)

    def add_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, data.shape)
        return data + noise

    def random_masking(self, data: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 1 - self.mask_prob, data.shape)
        return data * mask

    def standardize(self, data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True) + 1e-8
        return (data - mean) / std

    def __call__(self, genomic_feat: np.ndarray) -> torch.Tensor:
        feat = self.clip_outliers(genomic_feat)
        feat = self.standardize(feat)

        if self.mode == "train":
            feat = self.add_gaussian_noise(feat)
            feat = self.random_masking(feat)

        return torch.from_numpy(feat.astype(np.float32))


class ModalityTransform(BaseTransform):
    def __init__(
        self,
        input_size: int = 256,
        mode: str = "train",
        missing_pathology_prob: float = 0.15,
        missing_genomic_prob: float = 0.15
    ):
        super().__init__(mode)
        self.wsi_transform = WSITransform(input_size, mode)
        self.genomic_transform = GenomicTransform(mode)
        self.missing_patho_p = missing_pathology_prob
        self.missing_gen_p = missing_genomic_prob

    def simulate_missing_pathology(self, feat: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.missing_patho_p and self.mode == "train":
            return torch.zeros_like(feat)
        return feat

    def simulate_missing_genomic(self, feat: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.missing_gen_p and self.mode == "train":
            return torch.zeros_like(feat)
        return feat

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        wsi_patches = [self.wsi_transform(p) for p in data["pathology"]]
        wsi_feat = torch.stack(wsi_patches) if len(wsi_patches) > 0 else torch.zeros(1, 3, 256, 256)

        gen_feat = self.genomic_transform(data["genomic"])

        wsi_feat = self.simulate_missing_pathology(wsi_feat)
        gen_feat = self.simulate_missing_genomic(gen_feat)

        return {
            "pathology": wsi_feat,
            "genomic": gen_feat,
            "time": data["time"],
            "event": data["event"],
            "sample_id": data["sample_id"]
        }


class InferenceTransform(BaseTransform):
    def __init__(self, input_size: int = 256):
        self.wsi_transform = WSITransform(input_size, mode="test")
        self.genomic_transform = GenomicTransform(mode="test")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        wsi_patches = [self.wsi_transform(p) for p in data["pathology"]]
        wsi_feat = torch.stack(wsi_patches) if len(wsi_patches) > 0 else torch.zeros(1, 3, 256, 256)
        gen_feat = self.genomic_transform(data["genomic"])

        return {
            "pathology": wsi_feat,
            "genomic": gen_feat,
            "time": data["time"],
            "event": data["event"],
            "sample_id": data["sample_id"]
        }


def get_transform(
    mode: str = "train",
    input_size: int = 256,
    missing_pathology: float = 0.15,
    missing_genomic: float = 0.15
) -> BaseTransform:
    if mode == "infer":
        return InferenceTransform(input_size)
    return ModalityTransform(
        input_size=input_size,
        mode=mode,
        missing_pathology_prob=missing_pathology,
        missing_genomic_prob=missing_genomic
    )