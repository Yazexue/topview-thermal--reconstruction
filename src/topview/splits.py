from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import RepositoryConfig
from .utils import write_json


def _build_group_key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    safe_columns = [column for column in columns if column in frame.columns]
    if not safe_columns:
        return frame["patch_id"].astype(str)
    return frame[safe_columns].astype(str).agg("::".join, axis=1)


def create_grouped_splits(frame: pd.DataFrame, cfg: RepositoryConfig) -> pd.DataFrame:
    frame = frame.copy()
    groups = _build_group_key(frame, cfg.splits.group_columns)

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.splits.test_fraction,
        random_state=cfg.splits.seed,
    )
    train_val_idx, test_idx = next(outer.split(frame, groups=groups))
    train_val = frame.iloc[train_val_idx].copy()
    test = frame.iloc[test_idx].copy()

    inner_groups = _build_group_key(train_val, cfg.splits.group_columns)
    adjusted_val_fraction = cfg.splits.val_fraction / max(1.0 - cfg.splits.test_fraction, 1e-6)
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=adjusted_val_fraction,
        random_state=cfg.splits.seed + 1,
    )
    train_idx, val_idx = next(inner.split(train_val, groups=inner_groups))
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    out = pd.concat([train, val, test], ignore_index=True).sort_values(["split", "city", "acquisition_date", "patch_id"])
    return out.reset_index(drop=True)


def create_random_patch_splits(frame: pd.DataFrame, cfg: RepositoryConfig) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    train_val, test = train_test_split(
        frame,
        test_size=cfg.splits.test_fraction,
        random_state=cfg.splits.seed,
        shuffle=True,
    )
    adjusted_val_fraction = cfg.splits.val_fraction / max(1.0 - cfg.splits.test_fraction, 1e-6)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_fraction,
        random_state=cfg.splits.seed + 1,
        shuffle=True,
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    out = pd.concat([train, val, test], ignore_index=True).sort_values(["split", "city", "acquisition_date", "patch_id"])
    return out.reset_index(drop=True)


def create_splits(frame: pd.DataFrame, cfg: RepositoryConfig) -> pd.DataFrame:
    if cfg.splits.strategy == "grouped":
        return create_grouped_splits(frame, cfg)
    return create_random_patch_splits(frame, cfg)


def save_split_artifacts(cfg: RepositoryConfig, frame: pd.DataFrame, name: str = "patch_splits.csv") -> Path:
    cfg.ensure_directories()
    path = cfg.drive.splits_path / name
    frame.to_csv(path, index=False)

    summary = (
        frame.groupby("split")
        .agg(
            patches=("patch_id", "count"),
            cities=("city", lambda values: sorted(set(values))),
            acquisition_dates=("acquisition_date", lambda values: sorted(set(values))[:10]),
        )
        .to_dict(orient="index")
    )
    summary["split_strategy"] = cfg.splits.strategy
    write_json(cfg.drive.splits_path / "patch_splits_summary.json", summary)
    return path
