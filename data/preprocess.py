import os
import numpy as np
import pandas as pd
from PIL import Image
import openslide
from skimage.filters import threshold_otsu
from typing import Optional, Tuple, List

class WSIPreprocessor:
    def __init__(self, patch_size: int = 256, level: int = 0):
        self.patch_size = patch_size
        self.level = level

    def tissue_mask(self, slide: openslide.OpenSlide) -> np.ndarray:
        slide_low = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[slide.level_count - 1])
        slide_low = np.array(slide_low.convert("L"))
        thresh = threshold_otsu(slide_low)
        mask = slide_low < thresh
        return mask

    def extract_patches(self, slide_path: str) -> List[np.ndarray]:
        slide = openslide.OpenSlide(slide_path)
        mask = self.tissue_mask(slide)
        w, h = slide.level_dimensions[self.level]
        patches = []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                if mask[y // 4, x // 4]:
                    patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
                    patch = np.array(patch.convert("RGB"))
                    patches.append(patch)
        return patches

class GenomicPreprocessor:
    def __init__(self, n_bins: int = 6):
        self.n_bins = n_bins
        self.categories = [
            "tumor_suppression", "tumorigenesis", "protein_kinases",
            "cell_differentiation", "transcription", "cytokines_growth_factors"
        ]

    def bin_genomic_data(self, genomic_df: pd.DataFrame) -> np.ndarray:
        features = []
        for cat in self.categories:
            cat_features = genomic_df[genomic_df["category"] == cat].values[:, 1:].astype(np.float32)
            features.append(np.mean(cat_features, axis=0))
        return np.stack(features)

def survival_discretization(time: np.ndarray, event: np.ndarray, n_intervals: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    time_bins = np.linspace(time.min(), time.max(), n_intervals)
    disc_time = np.digitize(time, time_bins)
    disc_time = np.clip(disc_time, 0, n_intervals - 1)
    return disc_time, event