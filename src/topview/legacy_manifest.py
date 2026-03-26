from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import RepositoryConfig


REQUIRED_COLUMNS = {
    "ts",
    "city",
    "path",
    "relpath",
    "product",
    "cloud_pixels",
    "total_pixels",
    "cloud_pct",
    "invalid_pixels",
    "error",
}


@dataclass(slots=True)
class ImportedManifestResult:
    frame: pd.DataFrame
    dropped_rows: int


def _parse_hls_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    parts = stem.split(".")
    tile_id = "unknown"
    acquisition_date = "unknown"
    sensor = "unknown"

    if len(parts) >= 4:
        sensor = parts[1]
        tile_id = parts[2]
        stamp = parts[3]
        try:
            year = int(stamp[:4])
            doy = int(stamp[4:7])
            acquisition_date = (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
        except Exception:
            acquisition_date = "unknown"

    return {
        "patch_id": stem,
        "tile_id": tile_id,
        "acquisition_date": acquisition_date,
        "sensor": sensor,
    }


def _infer_patch_shape(total_pixels: float | int) -> tuple[int, int]:
    side = int(round(math.sqrt(float(total_pixels))))
    if side * side == int(total_pixels):
        return side, side
    return -1, -1


def import_legacy_inventory(csv_path: str | Path, cfg: RepositoryConfig) -> ImportedManifestResult:
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Legacy CSV is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame = frame[frame["error"].fillna("").astype(str).str.strip() == ""].copy()
    dropped_rows = int(len(pd.read_csv(csv_path)) - len(frame))

    parsed = frame["relpath"].fillna(frame["path"]).astype(str).map(_parse_hls_filename)
    parsed_frame = pd.DataFrame(parsed.tolist())
    frame = pd.concat([frame.reset_index(drop=True), parsed_frame], axis=1)

    shapes = frame["total_pixels"].map(_infer_patch_shape)
    frame["height"] = shapes.map(lambda item: item[0])
    frame["width"] = shapes.map(lambda item: item[1])
    frame["reflectance_channels"] = 6
    frame["thermal_channels"] = 2

    frame["city"] = frame["city"].astype(str)
    frame["path"] = frame["path"].astype(str)
    frame["relpath"] = frame["relpath"].fillna("").astype(str)

    portable_paths = []
    for _, row in frame.iterrows():
        relpath = row["relpath"].strip()
        city = str(row["city"]).strip()
        if relpath:
            portable_paths.append(str(cfg.drive.raw_path / city / relpath))
        else:
            portable_paths.append(str(row["path"]))
    frame["path"] = portable_paths

    frame["cloud_pct"] = pd.to_numeric(frame["cloud_pct"], errors="coerce")
    frame["cloud_fraction"] = frame["cloud_pct"] / 100.0
    frame["invalid_pixels"] = pd.to_numeric(frame["invalid_pixels"], errors="coerce").fillna(0).astype(int)

    manifest = frame[
        [
            "patch_id",
            "path",
            "city",
            "tile_id",
            "acquisition_date",
            "sensor",
            "height",
            "width",
            "reflectance_channels",
            "thermal_channels",
            "product",
            "cloud_pixels",
            "total_pixels",
            "cloud_pct",
            "cloud_fraction",
            "invalid_pixels",
            "relpath",
            "ts",
        ]
    ].sort_values(["city", "acquisition_date", "tile_id", "patch_id"]).reset_index(drop=True)

    return ImportedManifestResult(frame=manifest, dropped_rows=dropped_rows)
