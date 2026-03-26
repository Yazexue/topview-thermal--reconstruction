from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RepositoryConfig
from .io import describe_thermal_preprocessing, estimate_cloud_fraction_from_fmask, load_patch


DATE_PATTERN = re.compile(r"(20\d{2}[-_]?\d{2}[-_]?\d{2}|20\d{6})")


@dataclass(slots=True)
class PatchMetadata:
    patch_id: str
    path: str
    city: str
    tile_id: str
    acquisition_date: str
    sensor: str
    height: int
    width: int
    reflectance_channels: int
    thermal_channels: int
    cloud_fraction: float
    cloud_pixels: int
    total_pixels: int
    invalid_pixels: int
    quality_layer_kind: str
    thermal_input_unit_guess: str
    thermal_unit_evidence: str
    thermal_preprocessing_action: str
    thermal_preprocessing_status: str
    thermal_input_min: float
    thermal_input_max: float
    thermal_input_mean: float
    thermal_input_finite_fraction: float
    thermal_prepared_min: float
    thermal_prepared_max: float
    thermal_prepared_mean: float
    thermal_prepared_finite_fraction: float


def _normalize_date(raw: str | None) -> str:
    if not raw:
        return "unknown"
    clean = raw.replace("_", "-")
    if len(clean) == 8 and clean.isdigit():
        return f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}"
    return clean


def _infer_patch_metadata(patch_path: Path, cfg: RepositoryConfig) -> PatchMetadata:
    sidecar = patch_path.with_suffix(".json")
    sidecar_data: dict = {}
    if sidecar.exists():
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))

    payload = load_patch(patch_path, cfg)
    reflectance = payload["reflectance"]
    thermal = payload["thermal"]
    fmask = payload["fmask"]
    embedded = payload.get("metadata", {})
    cloud_fraction, invalid_pixels, cloud_pixels = estimate_cloud_fraction_from_fmask(fmask, embedded)
    total_pixels = int(np.isfinite(fmask).sum())
    thermal_info = describe_thermal_preprocessing(thermal, cfg, patch_path, embedded)
    quality_layer_kind = str(embedded.get("_selected_quality_name", "unknown"))

    merged = {**embedded, **sidecar_data}
    stem = patch_path.stem
    tokens = stem.split("__")
    patch_id = str(merged.get("patch_id", stem))
    city = str(merged.get("city", tokens[0] if tokens else "unknown"))
    tile_id = str(merged.get("tile_id", merged.get("hls_tile", "unknown")))
    sensor = str(merged.get("sensor", "Landsat-8"))
    acquisition_date = _normalize_date(str(merged.get("acquisition_date", "")))

    if acquisition_date == "unknown":
        date_match = DATE_PATTERN.search(stem)
        acquisition_date = _normalize_date(date_match.group(1) if date_match else None)

    return PatchMetadata(
        patch_id=patch_id,
        path=str(patch_path),
        city=city,
        tile_id=tile_id,
        acquisition_date=acquisition_date,
        sensor=sensor,
        height=int(reflectance.shape[-2]),
        width=int(reflectance.shape[-1]),
        reflectance_channels=int(reflectance.shape[0]),
        thermal_channels=int(thermal.shape[0]),
        cloud_fraction=float(cloud_fraction),
        cloud_pixels=int(cloud_pixels),
        total_pixels=int(total_pixels),
        invalid_pixels=int(invalid_pixels),
        quality_layer_kind=quality_layer_kind,
        thermal_input_unit_guess=str(thermal_info["thermal_input_unit_guess"]),
        thermal_unit_evidence=str(thermal_info["thermal_unit_evidence"]),
        thermal_preprocessing_action=str(thermal_info["thermal_preprocessing_action"]),
        thermal_preprocessing_status=str(thermal_info["thermal_preprocessing_status"]),
        thermal_input_min=float(thermal_info["thermal_input_min"]),
        thermal_input_max=float(thermal_info["thermal_input_max"]),
        thermal_input_mean=float(thermal_info["thermal_input_mean"]),
        thermal_input_finite_fraction=float(thermal_info["thermal_input_finite_fraction"]),
        thermal_prepared_min=float(thermal_info["thermal_prepared_min"]),
        thermal_prepared_max=float(thermal_info["thermal_prepared_max"]),
        thermal_prepared_mean=float(thermal_info["thermal_prepared_mean"]),
        thermal_prepared_finite_fraction=float(thermal_info["thermal_prepared_finite_fraction"]),
    )


def build_manifest(cfg: RepositoryConfig, pattern: str = "**/*") -> pd.DataFrame:
    return build_manifest_frame(cfg, pattern=pattern, apply_selection=True)


def build_manifest_frame(cfg: RepositoryConfig, pattern: str = "**/*", apply_selection: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for patch_path in sorted(cfg.drive.raw_path.glob(pattern)):
        if patch_path.suffix.lower() not in {".npz", ".nc"}:
            continue
        if any(part in {"runs", "reports", "exports", "manifests", "splits"} for part in patch_path.parts):
            continue
        row = _infer_patch_metadata(patch_path, cfg)
        rows.append(asdict(row))

    if not rows:
        raise FileNotFoundError(f"No patch files found under {cfg.drive.raw_path} with pattern {pattern}")

    frame = pd.DataFrame(rows).sort_values(["city", "acquisition_date", "tile_id", "patch_id"]).reset_index(drop=True)
    if apply_selection:
        frame = frame[frame["cloud_fraction"] < cfg.selection.max_cloud_fraction].copy()
        if cfg.selection.max_invalid_fraction <= 0:
            frame = frame[frame["invalid_pixels"] == 0].copy()
    for column in (
        "thermal_input_min",
        "thermal_input_max",
        "thermal_input_mean",
        "thermal_prepared_min",
        "thermal_prepared_max",
        "thermal_prepared_mean",
    ):
        if column in frame.columns:
            frame[column] = frame[column].apply(lambda value: value if not isinstance(value, float) or not math.isnan(value) else np.nan)
    return frame


def save_manifest(cfg: RepositoryConfig, frame: pd.DataFrame, name: str = "patch_manifest.csv") -> Path:
    cfg.ensure_directories()
    path = cfg.drive.manifests_path / name
    frame.to_csv(path, index=False)
    return path
