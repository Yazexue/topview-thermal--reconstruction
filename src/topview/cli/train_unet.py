from __future__ import annotations

import argparse

import pandas as pd

from topview.config import build_repository_config
from topview.training import train_regression_unet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--splits", default=None)
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    split_path = args.splits or str(cfg.drive.splits_path / "patch_splits.csv")
    frame = pd.read_csv(split_path)
    run_dir = train_regression_unet(frame, cfg)
    print(f"U-Net training artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
