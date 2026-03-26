from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _relative_destination(source: Path, source_root: Path, dest_root: Path) -> Path:
    try:
        relative = source.relative_to(source_root)
    except ValueError:
        relative = Path(source.name)
    return dest_root / relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    split_csv = Path(args.splits) if args.splits else source_root / "splits" / "patch_splits.csv"
    frame = pd.read_csv(split_csv)
    if args.split != "all":
        frame = frame[frame["split"] == args.split].copy()

    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "splits").mkdir(parents=True, exist_ok=True)
    (dest_root / "manifests").mkdir(parents=True, exist_ok=True)
    copied = 0
    remapped_paths: list[str] = []
    for path_str in frame["path"].tolist():
        source = Path(path_str)
        destination = _relative_destination(source, source_root, dest_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            remapped_paths.append(str(destination))
            continue
        shutil.copy2(source, destination)
        remapped_paths.append(str(destination))

        sidecar = source.with_suffix(".json")
        if sidecar.exists():
            sidecar_dest = _relative_destination(sidecar, source_root, dest_root)
            sidecar_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, sidecar_dest)
        copied += 1

    frame = frame.copy()
    frame["path"] = remapped_paths
    local_split_csv = dest_root / "splits" / "patch_splits.csv"
    frame.to_csv(local_split_csv, index=False)

    manifest_columns = [column for column in frame.columns if column != "split"]
    frame[manifest_columns].to_csv(dest_root / "manifests" / "patch_manifest.csv", index=False)

    print(f"Copied {copied} patch files to {dest_root}")


if __name__ == "__main__":
    main()
