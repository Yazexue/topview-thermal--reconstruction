from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .clouds import synthesize_cloudy_patch
from .config import RepositoryConfig
from .io import load_patch, prepare_thermal_for_modeling
from .utils import stable_int_seed


@dataclass(slots=True)
class LoadedPatch:
    patch_id: str
    city: str
    acquisition_date: str
    reflectance: np.ndarray
    thermal: np.ndarray
    fmask: np.ndarray


class HlsSyntheticDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: RepositoryConfig, split: str) -> None:
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        self.cfg = cfg
        self.split = split

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _augment_spatial(arr: np.ndarray, rotations: int, flip_lr: bool, flip_ud: bool) -> np.ndarray:
        out = np.rot90(arr, k=rotations, axes=(-2, -1))
        if flip_lr:
            out = np.flip(out, axis=-1)
        if flip_ud:
            out = np.flip(out, axis=-2)
        return np.ascontiguousarray(out)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        row = self.frame.iloc[index]
        patch = load_patch(row["path"], self.cfg)
        thermal = prepare_thermal_for_modeling(patch["thermal"], self.cfg, row["path"], patch.get("metadata", {}))
        seed = stable_int_seed(self.cfg.training.seed, self.cfg.clouds.seed_offset, self.split, row["patch_id"])
        synth = synthesize_cloudy_patch(patch["reflectance"], thermal, seed, self.cfg.clouds)

        reflectance = patch["reflectance"].astype(np.float32, copy=False)
        thermal = thermal.astype(np.float32, copy=False)
        fmask = patch["fmask"].astype(np.float32, copy=False)
        cloudy_reflectance = synth.cloudy_reflectance.astype(np.float32, copy=False)
        cloudy_thermal = synth.cloudy_thermal.astype(np.float32, copy=False)
        soft_mask = synth.soft_mask.astype(np.float32, copy=False)
        binary_mask = synth.binary_mask.astype(np.float32, copy=False)

        if self.split == "train" and self.cfg.training.use_augmentation:
            rotations = int(np.random.randint(0, 4))
            flip_lr = bool(np.random.randint(0, 2))
            flip_ud = bool(np.random.randint(0, 2))
            reflectance = self._augment_spatial(reflectance, rotations, flip_lr, flip_ud)
            thermal = self._augment_spatial(thermal, rotations, flip_lr, flip_ud)
            fmask = self._augment_spatial(fmask, rotations, flip_lr, flip_ud)
            cloudy_reflectance = self._augment_spatial(cloudy_reflectance, rotations, flip_lr, flip_ud)
            cloudy_thermal = self._augment_spatial(cloudy_thermal, rotations, flip_lr, flip_ud)
            soft_mask = self._augment_spatial(soft_mask, rotations, flip_lr, flip_ud)
            binary_mask = self._augment_spatial(binary_mask, rotations, flip_lr, flip_ud)

        observed = np.concatenate(
            [
                cloudy_reflectance,
                cloudy_thermal,
                fmask[None, ...],
                soft_mask[None, ...],
            ],
            axis=0,
        )
        clear = np.concatenate(
            [
                reflectance,
                thermal,
            ],
            axis=0,
        )

        return {
            "patch_id": str(row["patch_id"]),
            "city": str(row["city"]),
            "split": self.split,
            "observed": torch.from_numpy(observed),
            "target_full": torch.from_numpy(clear),
            "target_thermal": torch.from_numpy(thermal),
            "cloud_mask": torch.from_numpy(binary_mask),
            "soft_mask": torch.from_numpy(soft_mask),
            "cloud_fraction": float(synth.cloud_fraction),
        }
