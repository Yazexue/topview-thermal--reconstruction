from __future__ import annotations

import argparse

from topview.config import build_repository_config
from topview.reporting import build_method_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--per-patch", default=None)
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    per_patch_csv = args.per_patch or str(cfg.drive.reports_path / "evaluation_per_patch.csv")
    out_dir = cfg.drive.reports_path / "tables"
    paths = build_method_tables(per_patch_csv, out_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
