from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--drive-root", required=True)
    args = parser.parse_args()

    local_root = Path(args.local_root)
    drive_root = Path(args.drive_root)
    for subdir in ("runs", "reports", "exports", "manifests", "splits"):
        _copy_tree(local_root / subdir, drive_root / subdir)

    print(f"Synchronized artifacts from {local_root} to {drive_root}")


if __name__ == "__main__":
    main()
