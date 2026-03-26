from __future__ import annotations

import argparse

from topview.config import build_repository_config
from topview.manifest import build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--pattern", default="**/*")
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    frame = build_manifest(cfg, pattern=args.pattern)
    path = save_manifest(cfg, frame)
    print(f"Saved manifest with {len(frame)} patches to {path}")


if __name__ == "__main__":
    main()
