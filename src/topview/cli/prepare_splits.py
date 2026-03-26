from __future__ import annotations

import argparse

import pandas as pd

from topview.config import build_repository_config
from topview.splits import create_splits, save_split_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--strategy", choices=("random_patch", "grouped"), default=None)
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    if args.strategy is not None:
        cfg.splits.strategy = args.strategy
    manifest_path = args.manifest or str(cfg.drive.manifests_path / "patch_manifest.csv")
    frame = pd.read_csv(manifest_path)
    split_frame = create_splits(frame, cfg)
    path = save_split_artifacts(cfg, split_frame)
    print(f"Saved {cfg.splits.strategy} splits with {len(split_frame)} rows to {path}")


if __name__ == "__main__":
    main()
