from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree

try:
    from pykrige.ok import OrdinaryKriging
except Exception:  # pragma: no cover
    OrdinaryKriging = None


def _coordinate_arrays(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing="ij")
    return yy, xx


def idw_inpaint(image: np.ndarray, cloud_mask: np.ndarray, power: float = 2.0, neighbors: int = 50) -> np.ndarray:
    reconstructed = image.copy().astype(np.float32)
    known = cloud_mask == 0
    missing = cloud_mask > 0
    if not np.any(missing):
        return reconstructed

    yy, xx = _coordinate_arrays(cloud_mask)
    known_points = np.column_stack([yy[known], xx[known]])
    missing_points = np.column_stack([yy[missing], xx[missing]])
    tree = cKDTree(known_points)
    distances, indices = tree.query(missing_points, k=min(neighbors, len(known_points)))
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    weights = 1.0 / np.power(np.clip(distances, 1e-6, None), power)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    for channel in range(reconstructed.shape[0]):
        known_values = reconstructed[channel][known]
        filled = np.sum(weights * known_values[indices], axis=1)
        channel_data = reconstructed[channel]
        channel_data[missing] = filled
        reconstructed[channel] = channel_data
    return reconstructed


def ordinary_kriging_inpaint(
    image: np.ndarray,
    cloud_mask: np.ndarray,
    variogram_model: str = "exponential",
    max_points: int = 3000,
) -> np.ndarray:
    if OrdinaryKriging is None:
        raise ImportError("pykrige is required for ordinary kriging baseline")

    reconstructed = image.copy().astype(np.float32)
    known = cloud_mask == 0
    missing = cloud_mask > 0
    if not np.any(missing):
        return reconstructed

    yy, xx = _coordinate_arrays(cloud_mask)
    known_y = yy[known].astype(float)
    known_x = xx[known].astype(float)
    missing_y = yy[missing].astype(float)
    missing_x = xx[missing].astype(float)

    if known_x.size > max_points:
        sample_idx = np.linspace(0, known_x.size - 1, max_points).astype(int)
        known_x = known_x[sample_idx]
        known_y = known_y[sample_idx]
    else:
        sample_idx = slice(None)

    for channel in range(reconstructed.shape[0]):
        values = reconstructed[channel][known].astype(float)[sample_idx]
        try:
            ok = OrdinaryKriging(
                known_x,
                known_y,
                values,
                variogram_model=variogram_model,
                enable_plotting=False,
                verbose=False,
            )
            pred, _ = ok.execute("points", missing_x, missing_y)
            channel_data = reconstructed[channel]
            channel_data[missing] = np.asarray(pred, dtype=np.float32)
            reconstructed[channel] = channel_data
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Kriging failed for channel {channel} with {exc}; falling back to IDW.")
            return idw_inpaint(image, cloud_mask)
    return reconstructed


def run_baseline(
    method: str,
    target_like_observed: np.ndarray,
    cloud_mask: np.ndarray,
    *,
    idw_power: float = 2.0,
    idw_neighbors: int = 50,
    ok_variogram_model: str = "exponential",
    ok_max_points: int = 3000,
) -> np.ndarray:
    if method == "idw":
        return idw_inpaint(
            target_like_observed,
            cloud_mask,
            power=idw_power,
            neighbors=idw_neighbors,
        )
    if method == "ok":
        return ordinary_kriging_inpaint(
            target_like_observed,
            cloud_mask,
            variogram_model=ok_variogram_model,
            max_points=ok_max_points,
        )
    raise ValueError(f"Unsupported baseline method: {method}")
