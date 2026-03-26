from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import RepositoryConfig


def _decode_metadata_blob(blob: Any) -> dict:
    if blob is None:
        return {}
    if hasattr(blob, "item"):
        try:
            blob = blob.item()
        except Exception:
            pass
    if isinstance(blob, str):
        return json.loads(blob)
    if isinstance(blob, dict):
        return blob
    return {}


def _find_variables(dataset: Any, candidates: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    vars_lower = {name.lower(): name for name in dataset.data_vars}
    for candidate in candidates:
        if candidate in dataset.data_vars:
            found.append(candidate)
        elif candidate.lower() in vars_lower:
            found.append(vars_lower[candidate.lower()])
    return found


def _extract_from_band_coordinate(dataset: Any, candidates: tuple[str, ...]) -> tuple[np.ndarray | None, str | None, tuple[str, ...]]:
    for name, variable in dataset.data_vars.items():
        for dim in variable.dims:
            if dim not in dataset.coords:
                continue
            labels = [str(value) for value in dataset.coords[dim].values.tolist()]
            labels_lower = {label.lower(): label for label in labels}
            selected = [labels_lower[candidate.lower()] for candidate in candidates if candidate.lower() in labels_lower]
            if not selected:
                continue
            data = variable.sel({dim: selected}).values.astype(np.float32)
            if data.ndim == 2:
                data = data[None, ...]
            return data, name, tuple(selected)
    return None, None, ()


def _merge_metadata_attrs(base: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in attrs.items():
        merged.setdefault(key, value)
    return merged


def _load_npz_patch(path: Path, cfg: RepositoryConfig) -> dict:
    with np.load(path, allow_pickle=True) as data:
        payload = {
            "reflectance": data[cfg.keys.reflectance].astype(np.float32),
            "thermal": data[cfg.keys.thermal].astype(np.float32),
            "fmask": data[cfg.keys.fmask].astype(np.int32),
        }
        if cfg.keys.soft_mask in data:
            payload["soft_mask"] = data[cfg.keys.soft_mask].astype(np.float32)
        payload["metadata"] = _decode_metadata_blob(data[cfg.keys.metadata_json]) if cfg.keys.metadata_json in data else {}
    return payload


def _load_netcdf_patch(path: Path, cfg: RepositoryConfig) -> dict:
    import xarray as xr

    try:
        dataset = xr.open_dataset(path, engine="netcdf4")
    except Exception:
        dataset = xr.open_dataset(path, engine="h5netcdf")

    reflectance_names = _find_variables(dataset, cfg.keys.reflectance_bands)
    thermal_names = _find_variables(dataset, cfg.keys.thermal_bands)
    fmask_names = _find_variables(dataset, cfg.keys.fmask_candidates)

    metadata = {key: value for key, value in dataset.attrs.items()}
    source_attr_names: list[str] = []

    if reflectance_names:
        reflectance = np.stack([dataset[name].values.astype(np.float32) for name in reflectance_names], axis=0)
        metadata["_selected_reflectance_names"] = list(reflectance_names)
        source_attr_names.append(reflectance_names[0])
    elif cfg.keys.reflectance in dataset.data_vars:
        reflectance = dataset[cfg.keys.reflectance].values.astype(np.float32)
        metadata["_selected_reflectance_names"] = [cfg.keys.reflectance]
        source_attr_names.append(cfg.keys.reflectance)
    else:
        reflectance, reflectance_source_name, reflectance_selected = _extract_from_band_coordinate(dataset, cfg.keys.reflectance_bands)
        if reflectance is None:
            raise KeyError(f"No reflectance bands found in {path}")
        metadata["_selected_reflectance_names"] = list(reflectance_selected)
        if reflectance_source_name is not None:
            source_attr_names.append(reflectance_source_name)

    if thermal_names:
        thermal = np.stack([dataset[name].values.astype(np.float32) for name in thermal_names], axis=0)
        metadata["_selected_thermal_names"] = list(thermal_names)
        source_attr_names.append(thermal_names[0])
    elif cfg.keys.thermal in dataset.data_vars:
        thermal = dataset[cfg.keys.thermal].values.astype(np.float32)
        metadata["_selected_thermal_names"] = [cfg.keys.thermal]
        source_attr_names.append(cfg.keys.thermal)
    else:
        thermal, thermal_source_name, thermal_selected = _extract_from_band_coordinate(dataset, cfg.keys.thermal_bands)
        if thermal is None:
            raise KeyError(f"No thermal bands found in {path}")
        metadata["_selected_thermal_names"] = list(thermal_selected)
        if thermal_source_name is not None:
            source_attr_names.append(thermal_source_name)

    if fmask_names:
        fmask = dataset[fmask_names[0]].values.astype(np.int32)
        metadata["_selected_quality_name"] = fmask_names[0]
        source_attr_names.append(fmask_names[0])
    elif cfg.keys.fmask in dataset.data_vars:
        fmask = dataset[cfg.keys.fmask].values.astype(np.int32)
        metadata["_selected_quality_name"] = cfg.keys.fmask
        source_attr_names.append(cfg.keys.fmask)
    else:
        fmask_from_band, quality_source_name, quality_selected = _extract_from_band_coordinate(dataset, cfg.keys.fmask_candidates)
        if fmask_from_band is None:
            raise KeyError(f"No Fmask/QA variable found in {path}")
        fmask = fmask_from_band[0].astype(np.int32)
        metadata["_selected_quality_name"] = quality_selected[0] if quality_selected else quality_source_name
        if quality_source_name is not None:
            source_attr_names.append(quality_source_name)

    for source_name in dict.fromkeys(source_attr_names):
        if source_name in dataset.data_vars:
            metadata = _merge_metadata_attrs(metadata, dict(dataset[source_name].attrs))

    dataset.close()
    return {
        "reflectance": reflectance,
        "thermal": thermal,
        "fmask": fmask,
        "metadata": metadata,
    }


def load_patch(path: str | Path, cfg: RepositoryConfig) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return _load_npz_patch(path, cfg)
    if path.suffix.lower() == ".nc":
        return _load_netcdf_patch(path, cfg)
    raise ValueError(f"Unsupported patch format: {path}")


def infer_quality_layer_kind(quality: np.ndarray, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    selected_name = str(metadata.get("_selected_quality_name", "")).lower()
    if "fmask" in selected_name:
        return "fmask_class"
    if selected_name == "qa" or selected_name.endswith("_qa"):
        return "qa_bits"

    values = np.asarray(quality)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "unknown"
    finite_int = finite.astype(np.int32, copy=False)
    if int(finite_int.max()) > 7:
        return "qa_bits"
    return "fmask_class"


def cloud_like_mask_from_quality_layer(quality: np.ndarray, metadata: dict[str, Any] | None = None) -> np.ndarray:
    quality = np.asarray(quality)
    valid = np.isfinite(quality)
    kind = infer_quality_layer_kind(quality, metadata)

    if kind == "qa_bits":
        qa = quality.astype(np.uint8, copy=False)
        cirrus = (qa & 1) > 0
        cloud = (qa & 2) > 0
        adjacent = (qa & 4) > 0
        shadow = (qa & 8) > 0
        snow_ice = (qa & 16) > 0
        return valid & (cirrus | cloud | adjacent | shadow | snow_ice)

    if kind == "fmask_class":
        classes = quality.astype(np.int32, copy=False)
        return valid & np.isin(classes, [2, 3, 4])

    raise ValueError("Unable to infer whether the quality layer is categorical Fmask or bit-packed QA.")


def estimate_cloud_fraction_from_fmask(
    fmask: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, int, int]:
    fmask = np.asarray(fmask)
    valid = np.isfinite(fmask)
    total_pixels = int(valid.sum())
    invalid_pixels = int((~valid).sum())
    if total_pixels == 0:
        return 1.0, invalid_pixels, 0

    cloud_like = cloud_like_mask_from_quality_layer(fmask, metadata)
    cloud_pixels = int((cloud_like & valid).sum())
    return cloud_pixels / total_pixels, invalid_pixels, cloud_pixels


def summarize_numeric_array(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    total = int(values.size)
    count = int(finite.size)
    if count == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "finite_fraction": 0.0,
        }
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "finite_fraction": count / total if total > 0 else 0.0,
    }


def _explicit_temperature_unit(metadata: dict[str, Any] | None = None) -> str | None:
    metadata = metadata or {}
    for key in ("thermal_units", "units", "unit", "brightness_temperature_unit"):
        value = metadata.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if "kelvin" in normalized or normalized == "k":
            return "kelvin"
        if "celsius" in normalized or "degc" in normalized or normalized in {"c", "°c"}:
            return "celsius"
    return None


def metadata_supports_hls_celsius(metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    if _explicit_temperature_unit(metadata) == "celsius":
        return True

    sensor = str(metadata.get("SENSOR", "")).upper()
    selected_thermal = tuple(str(name).upper() for name in metadata.get("_selected_thermal_names", []))
    has_landsat_keys = any(key in metadata for key in ("LANDSAT_PRODUCT_ID", "LANDSAT_SCENE_ID"))
    has_hls_keys = any(key in metadata for key in ("SENTINEL2_TILEID", "HLS_PROCESSING_TIME", "ACCODE"))
    thermal_is_b10_b11 = bool(selected_thermal) and all(name in {"B10", "B11"} for name in selected_thermal)
    return thermal_is_b10_b11 and (sensor == "OLI_TIRS" or has_landsat_keys or has_hls_keys)


def infer_thermal_units(thermal: np.ndarray, cfg: RepositoryConfig) -> str:
    finite = np.asarray(thermal, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "unknown"
    minimum = float(finite.min())
    maximum = float(finite.max())
    rules = cfg.thermal_preprocessing
    if rules.kelvin_min <= minimum <= rules.kelvin_max and rules.kelvin_min <= maximum <= rules.kelvin_max:
        return "kelvin"
    if rules.celsius_min <= minimum <= rules.celsius_max and rules.celsius_min <= maximum <= rules.celsius_max:
        return "celsius"
    return "unknown"


def prepare_thermal_for_modeling(
    thermal: np.ndarray,
    cfg: RepositoryConfig,
    path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    thermal = np.asarray(thermal, dtype=np.float32)
    requested = cfg.thermal_preprocessing.input_units.lower()
    detected = infer_thermal_units(thermal, cfg)
    explicit_units = _explicit_temperature_unit(metadata)

    if requested == "kelvin":
        prepared = thermal
    elif requested == "celsius":
        prepared = thermal + 273.15
    elif requested == "auto":
        if detected == "kelvin":
            prepared = thermal
        elif detected == "celsius":
            if explicit_units == "celsius" or metadata_supports_hls_celsius(metadata):
                prepared = thermal + 273.15
            else:
                location = f" in {Path(path)}" if path is not None else ""
                raise ValueError(
                    "Celsius-like thermal values are ambiguous without metadata evidence"
                    f"{location}. Set thermal units explicitly or provide HLS/Landsat metadata."
                )
        else:
            location = f" in {Path(path)}" if path is not None else ""
            raise ValueError(
                "Thermal bands do not fall into plausible Kelvin or Celsius ranges"
                f"{location}. The thesis workflow expects brightness temperature in Kelvin "
                "before synthetic cloud generation and model evaluation."
            )
    else:
        raise ValueError(f"Unsupported thermal input_units setting: {cfg.thermal_preprocessing.input_units}")

    return prepared.astype(np.float32, copy=False)


def describe_thermal_preprocessing(
    thermal: np.ndarray,
    cfg: RepositoryConfig,
    path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    raw_stats = summarize_numeric_array(thermal)
    unit_guess = infer_thermal_units(thermal, cfg)
    explicit_units = _explicit_temperature_unit(metadata)
    if explicit_units is not None:
        evidence = f"explicit_{explicit_units}"
    elif metadata_supports_hls_celsius(metadata):
        evidence = "hls_landsat_metadata"
    else:
        evidence = "none"

    try:
        prepared = prepare_thermal_for_modeling(thermal, cfg, path=path, metadata=metadata)
        prepared_stats = summarize_numeric_array(prepared)
        if unit_guess == "celsius":
            action = "celsius_to_kelvin"
        elif unit_guess == "kelvin":
            action = "as_is_kelvin"
        else:
            action = "configured_conversion"
        status = "ok"
    except Exception as exc:
        prepared_stats = {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "finite_fraction": 0.0,
        }
        action = "error"
        status = str(exc)

    return {
        "thermal_input_unit_guess": unit_guess,
        "thermal_unit_evidence": evidence,
        "thermal_preprocessing_action": action,
        "thermal_preprocessing_status": status,
        "thermal_input_min": raw_stats["min"],
        "thermal_input_max": raw_stats["max"],
        "thermal_input_mean": raw_stats["mean"],
        "thermal_input_finite_fraction": raw_stats["finite_fraction"],
        "thermal_prepared_min": prepared_stats["min"],
        "thermal_prepared_max": prepared_stats["max"],
        "thermal_prepared_mean": prepared_stats["mean"],
        "thermal_prepared_finite_fraction": prepared_stats["finite_fraction"],
    }
