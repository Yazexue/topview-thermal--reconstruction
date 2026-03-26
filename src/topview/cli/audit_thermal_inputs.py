from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from topview.config import build_repository_config
from topview.io import describe_thermal_preprocessing, load_patch
from topview.manifest import build_manifest_frame
from topview.utils import write_json


def _build_frame_from_manifest_or_root(cfg, manifest_path: str | None, pattern: str) -> pd.DataFrame:
    if manifest_path:
        return pd.read_csv(manifest_path)
    return build_manifest_frame(cfg, pattern=pattern, apply_selection=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pattern", default="**/*")
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    frame = _build_frame_from_manifest_or_root(cfg, args.manifest, args.pattern)

    rows: list[dict] = []
    for _, row in frame.iterrows():
        patch = load_patch(row["path"], cfg)
        audit = describe_thermal_preprocessing(patch["thermal"], cfg, row["path"], patch.get("metadata", {}))
        rows.append(
            {
                "patch_id": str(row.get("patch_id", Path(row["path"]).stem)),
                "path": str(row["path"]),
                **audit,
            }
        )

    audit_frame = pd.DataFrame(rows).sort_values(["thermal_preprocessing_action", "patch_id"]).reset_index(drop=True)
    out_dir = cfg.drive.reports_path / "thermal_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "thermal_audit.csv"
    audit_frame.to_csv(csv_path, index=False)

    summary = {
        "patch_count": int(len(audit_frame)),
        "input_unit_guess_counts": audit_frame["thermal_input_unit_guess"].value_counts(dropna=False).to_dict(),
        "preprocessing_action_counts": audit_frame["thermal_preprocessing_action"].value_counts(dropna=False).to_dict(),
        "status_counts": audit_frame["thermal_preprocessing_status"].value_counts(dropna=False).to_dict(),
        "prepared_min_range": [
            float(audit_frame["thermal_prepared_min"].min(skipna=True)) if len(audit_frame) else float("nan"),
            float(audit_frame["thermal_prepared_min"].max(skipna=True)) if len(audit_frame) else float("nan"),
        ],
        "prepared_max_range": [
            float(audit_frame["thermal_prepared_max"].min(skipna=True)) if len(audit_frame) else float("nan"),
            float(audit_frame["thermal_prepared_max"].max(skipna=True)) if len(audit_frame) else float("nan"),
        ],
        "error_examples": audit_frame[audit_frame["thermal_preprocessing_action"] == "error"]
        .head(20)
        .to_dict(orient="records"),
    }
    json_path = out_dir / "thermal_audit_summary.json"
    write_json(json_path, summary)

    print(f"Saved thermal audit CSV to {csv_path}")
    print(f"Saved thermal audit summary to {json_path}")


if __name__ == "__main__":
    main()
