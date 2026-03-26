from __future__ import annotations

import argparse

from topview.config import build_repository_config
from topview.legacy_manifest import import_legacy_inventory
from topview.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-name", default="patch_manifest.csv")
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    cfg.ensure_directories()

    result = import_legacy_inventory(args.csv, cfg)
    output_path = cfg.drive.manifests_path / args.output_name
    result.frame.to_csv(output_path, index=False)

    summary = {
        "source_csv": args.csv,
        "rows_written": int(len(result.frame)),
        "dropped_rows": int(result.dropped_rows),
        "cities": sorted(result.frame["city"].dropna().unique().tolist()),
        "cloud_pct_min": float(result.frame["cloud_pct"].min()),
        "cloud_pct_max": float(result.frame["cloud_pct"].max()),
    }
    write_json(cfg.drive.manifests_path / "import_legacy_inventory_summary.json", summary)
    print(f"Imported {len(result.frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
