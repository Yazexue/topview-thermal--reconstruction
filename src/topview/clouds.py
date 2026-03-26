from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import SyntheticCloudConfig


PLANCK_H = 6.62607015e-34
PLANCK_C = 2.99792458e8
PLANCK_K = 1.380649e-23


@dataclass(slots=True)
class SyntheticCloudResult:
    cloudy_reflectance: np.ndarray
    cloudy_thermal: np.ndarray
    binary_mask: np.ndarray
    soft_mask: np.ndarray
    cloud_fraction: float
    thermal_delta: np.ndarray


def _normalize_field(field: np.ndarray) -> np.ndarray:
    field = field.astype(np.float32)
    field -= field.min()
    denom = field.max() - field.min()
    return field / denom if denom > 0 else np.zeros_like(field)


def generate_cloud_opacity(shape: tuple[int, int], seed: int, cfg: SyntheticCloudConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    accum = np.zeros(shape, dtype=np.float32)

    for sigma in cfg.opacity_smoothing_sigmas:
        noise = rng.normal(size=shape).astype(np.float32)
        accum += gaussian_filter(noise, sigma=sigma)

    accum = _normalize_field(accum)
    target_fraction = float(rng.uniform(cfg.cloud_fraction_min, cfg.cloud_fraction_max))
    threshold = float(np.quantile(accum, 1.0 - target_fraction))
    soft = np.clip((accum - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    if cfg.boundary_smoothing_sigma > 0:
        soft = gaussian_filter(soft, sigma=cfg.boundary_smoothing_sigma)
        soft = np.clip(_normalize_field(soft), 0.0, 1.0)
    binary = (soft >= 0.5).astype(np.uint8)
    return binary, soft.astype(np.float32)


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.roll(mask, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return shifted


def synthesize_optical_clouds(
    reflectance: np.ndarray,
    soft_mask: np.ndarray,
    seed: int,
    cfg: SyntheticCloudConfig,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud_reflectance = float(rng.uniform(cfg.optical_cloud_reflectance_min, cfg.optical_cloud_reflectance_max))
    shadow_strength = float(rng.uniform(cfg.shadow_attenuation_min, cfg.shadow_attenuation_max))
    shadow_mask = _shift_mask(soft_mask, cfg.shadow_shift_x, cfg.shadow_shift_y)

    reflectance = reflectance.astype(np.float32)
    cloudy = reflectance * (1.0 - 0.25 * shadow_mask[None, ...] * shadow_strength)
    cloudy = cloudy * (1.0 - soft_mask[None, ...]) + cloud_reflectance * soft_mask[None, ...]
    return np.clip(cloudy, 0.0, 1.0)


def kelvin_to_radiance(temperature_kelvin: np.ndarray, wavelength_um: float) -> np.ndarray:
    wavelength_m = wavelength_um * 1e-6
    numerator = 2.0 * PLANCK_H * PLANCK_C**2
    denominator = wavelength_m**5
    exponent = (PLANCK_H * PLANCK_C) / (wavelength_m * PLANCK_K * np.clip(temperature_kelvin, 1e-6, None))
    return numerator / (denominator * (np.exp(exponent) - 1.0))


def radiance_to_kelvin(radiance: np.ndarray, wavelength_um: float) -> np.ndarray:
    wavelength_m = wavelength_um * 1e-6
    numerator = PLANCK_H * PLANCK_C
    denominator = wavelength_m * PLANCK_K
    inside = (2.0 * PLANCK_H * PLANCK_C**2) / (wavelength_m**5 * np.clip(radiance, 1e-12, None)) + 1.0
    return numerator / (denominator * np.log(inside))


def synthesize_thermal_clouds(
    thermal: np.ndarray,
    soft_mask: np.ndarray,
    seed: int,
    cfg: SyntheticCloudConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    thermal = thermal.astype(np.float32)

    cooling = rng.uniform(cfg.thermal_cooling_min_kelvin, cfg.thermal_cooling_max_kelvin, size=soft_mask.shape).astype(np.float32)
    cooling = gaussian_filter(cooling, sigma=12.0) * soft_mask
    proxy = np.clip(
        thermal - cooling[None, ...],
        cfg.thermal_clip_min_kelvin,
        cfg.thermal_clip_max_kelvin,
    )

    wavelengths = (cfg.band10_wavelength_um, cfg.band11_wavelength_um)
    mixed = np.empty_like(thermal)
    for band_index, wavelength in enumerate(wavelengths[: thermal.shape[0]]):
        clear_radiance = kelvin_to_radiance(thermal[band_index], wavelength)
        cloudy_radiance = kelvin_to_radiance(proxy[band_index], wavelength)
        weight = np.power(soft_mask, cfg.opacity_exponent)
        mixed_radiance = (1.0 - weight) * clear_radiance + weight * cloudy_radiance
        mixed[band_index] = radiance_to_kelvin(mixed_radiance, wavelength)

    return mixed.astype(np.float32), (mixed - thermal).astype(np.float32)


def synthesize_cloudy_patch(
    reflectance: np.ndarray,
    thermal: np.ndarray,
    seed: int,
    cfg: SyntheticCloudConfig,
) -> SyntheticCloudResult:
    binary_mask, soft_mask = generate_cloud_opacity(reflectance.shape[-2:], seed, cfg)
    cloudy_reflectance = synthesize_optical_clouds(reflectance, soft_mask, seed + 1, cfg)
    cloudy_thermal, thermal_delta = synthesize_thermal_clouds(thermal, soft_mask, seed + 2, cfg)

    return SyntheticCloudResult(
        cloudy_reflectance=cloudy_reflectance,
        cloudy_thermal=cloudy_thermal,
        binary_mask=binary_mask,
        soft_mask=soft_mask,
        cloud_fraction=float(binary_mask.mean()),
        thermal_delta=thermal_delta,
    )
