from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_ROOT = "/content/drive/Shareddrives/HLS - TOPVIEW/patched"


@dataclass(slots=True)
class DriveLayout:
    root: str = DEFAULT_ROOT
    raw_subdir: str = "."
    manifests_subdir: str = "manifests"
    splits_subdir: str = "splits"
    runs_subdir: str = "runs"
    reports_subdir: str = "reports"
    exports_subdir: str = "exports"

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def raw_path(self) -> Path:
        return self.root_path / self.raw_subdir

    @property
    def manifests_path(self) -> Path:
        return self.root_path / self.manifests_subdir

    @property
    def splits_path(self) -> Path:
        return self.root_path / self.splits_subdir

    @property
    def runs_path(self) -> Path:
        return self.root_path / self.runs_subdir

    @property
    def reports_path(self) -> Path:
        return self.root_path / self.reports_subdir

    @property
    def exports_path(self) -> Path:
        return self.root_path / self.exports_subdir


@dataclass(slots=True)
class PatchKeys:
    reflectance: str = "reflectance"
    thermal: str = "thermal"
    fmask: str = "fmask"
    soft_mask: str = "soft_mask"
    metadata_json: str = "metadata_json"
    reflectance_bands: tuple[str, ...] = ("B02", "B03", "B04", "B05", "B06", "B07")
    thermal_bands: tuple[str, ...] = ("B10", "B11")
    fmask_candidates: tuple[str, ...] = ("Fmask", "fmask", "QA", "qa")


@dataclass(slots=True)
class SplitConfig:
    seed: int = 42
    test_fraction: float = 0.10
    val_fraction: float = 0.10
    strategy: str = "random_patch"
    group_columns: tuple[str, ...] = ("city", "tile_id", "acquisition_date")
    min_group_cardinality: int = 1


@dataclass(slots=True)
class DataSelectionConfig:
    max_cloud_fraction: float = 0.10
    max_invalid_fraction: float = 0.0


@dataclass(slots=True)
class SyntheticCloudConfig:
    seed_offset: int = 1000
    cloud_fraction_min: float = 0.15
    cloud_fraction_max: float = 0.25
    opacity_smoothing_sigmas: tuple[int, ...] = (8, 32, 96)
    boundary_smoothing_sigma: float = 5.0
    opacity_exponent: float = 1.5
    thermal_cooling_min_kelvin: float = 5.0
    thermal_cooling_max_kelvin: float = 15.0
    thermal_clip_min_kelvin: float = 180.0
    thermal_clip_max_kelvin: float = 330.0
    band10_wavelength_um: float = 10.9
    band11_wavelength_um: float = 12.0
    optical_cloud_reflectance_min: float = 0.75
    optical_cloud_reflectance_max: float = 0.98
    shadow_attenuation_min: float = 0.65
    shadow_attenuation_max: float = 0.88
    shadow_shift_x: int = 6
    shadow_shift_y: int = 10


@dataclass(slots=True)
class ThermalPreprocessingConfig:
    input_units: str = "auto"
    kelvin_min: float = 150.0
    kelvin_max: float = 400.0
    celsius_min: float = -120.0
    celsius_max: float = 120.0


@dataclass(slots=True)
class BaselineConfig:
    idw_power: float = 2.0
    idw_neighbors: int = 50
    ok_variogram_model: str = "exponential"
    ok_max_points: int = 3000


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 50
    max_iterations: int = 40000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    grad_clip_norm: float = 1.0
    use_augmentation: bool = True


@dataclass(slots=True)
class DiffusionConfig:
    image_size: int = 256
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    attention_heads: int = 4
    attention_resolutions: tuple[int, ...] = (32, 16, 8)
    timesteps: int = 1000
    sampling_steps: int = 250
    repaint_jump_length: int = 10
    repaint_resamples: int = 5
    ema_decay: float = 0.999
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass(slots=True)
class RepositoryConfig:
    drive: DriveLayout = field(default_factory=DriveLayout)
    keys: PatchKeys = field(default_factory=PatchKeys)
    selection: DataSelectionConfig = field(default_factory=DataSelectionConfig)
    thermal_preprocessing: ThermalPreprocessingConfig = field(default_factory=ThermalPreprocessingConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    clouds: SyntheticCloudConfig = field(default_factory=SyntheticCloudConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def ensure_directories(self) -> None:
        for path in (
            self.drive.manifests_path,
            self.drive.splits_path,
            self.drive.runs_path,
            self.drive.reports_path,
            self.drive.exports_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


def build_repository_config(root: str | None = None) -> RepositoryConfig:
    cfg = RepositoryConfig()
    if root is not None:
        cfg.drive.root = root
    return cfg
