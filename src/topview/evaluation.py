from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass(slots=True)
class MetricBundle:
    rmse: float
    bias: float
    pearson_r: float
    thin_rmse: float
    thick_rmse: float
    global_mean_rmse: float
    psnr: float
    ssim: float


def _masked_values(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool)
    if pred.ndim == 3 and mask.ndim == 2:
        return pred[:, valid].reshape(-1), ref[:, valid].reshape(-1)
    return pred[valid], ref[valid]


def cloud_only_rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    pred_values, ref_values = _masked_values(pred, ref, mask)
    return float(np.sqrt(np.mean((pred_values - ref_values) ** 2))) if pred_values.size else math.nan


def mean_bias_error(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    pred_values, ref_values = _masked_values(pred, ref, mask)
    return float(np.mean(pred_values - ref_values)) if pred_values.size else math.nan


def patch_pearson_r(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    pred_values, ref_values = _masked_values(pred, ref, mask)
    if pred_values.size < 2:
        return math.nan
    return float(np.corrcoef(pred_values, ref_values)[0, 1])


def thin_thick_rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, soft_mask: np.ndarray) -> tuple[float, float]:
    thin = (mask > 0) & (soft_mask >= 0.20) & (soft_mask <= 0.65)
    thick = (mask > 0) & (soft_mask > 0.65)
    return cloud_only_rmse(pred, ref, thin), cloud_only_rmse(pred, ref, thick)


def global_mean_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_mean = np.mean(pred, axis=(-2, -1))
    ref_mean = np.mean(ref, axis=(-2, -1))
    return float(np.sqrt(np.mean((pred_mean - ref_mean) ** 2)))


def optical_psnr_ssim(pred_rgb: np.ndarray, ref_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    region = mask.astype(bool)
    if not np.any(region):
        return math.nan, math.nan

    pred = pred_rgb.copy()
    ref = ref_rgb.copy()
    pred[:, ~region] = ref[:, ~region]
    pred_hwc = np.moveaxis(pred, 0, -1)
    ref_hwc = np.moveaxis(ref, 0, -1)

    psnr = float(peak_signal_noise_ratio(ref_hwc, pred_hwc, data_range=1.0))
    ssim = float(structural_similarity(ref_hwc, pred_hwc, channel_axis=-1, data_range=1.0))
    return psnr, ssim


def evaluate_patch(
    pred_thermal: np.ndarray,
    ref_thermal: np.ndarray,
    pred_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    mask: np.ndarray,
    soft_mask: np.ndarray,
) -> MetricBundle:
    rmse = cloud_only_rmse(pred_thermal, ref_thermal, mask)
    bias = mean_bias_error(pred_thermal, ref_thermal, mask)
    r_value = patch_pearson_r(pred_thermal, ref_thermal, mask)
    thin_rmse, thick_rmse = thin_thick_rmse(pred_thermal, ref_thermal, mask, soft_mask)
    mean_rmse = global_mean_rmse(pred_thermal, ref_thermal)
    psnr, ssim = optical_psnr_ssim(pred_rgb, ref_rgb, mask)

    return MetricBundle(
        rmse=rmse,
        bias=bias,
        pearson_r=r_value,
        thin_rmse=thin_rmse,
        thick_rmse=thick_rmse,
        global_mean_rmse=mean_rmse,
        psnr=psnr,
        ssim=ssim,
    )
