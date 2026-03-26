from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_method_tables(per_patch_csv: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(per_patch_csv)

    table_4_1 = (
        frame.groupby("method", as_index=False)[["rmse", "bias", "pearson_r"]]
        .mean()
        .rename(columns={"rmse": "RMSE_K", "bias": "Bias_K", "pearson_r": "Pearson_r"})
    )
    table_4_2 = (
        frame.groupby("method", as_index=False)[["thin_rmse", "thick_rmse"]]
        .mean()
        .rename(columns={"thin_rmse": "RMSE_thin_K", "thick_rmse": "RMSE_thick_K"})
    )
    table_4_3 = (
        frame.groupby("method", as_index=False)[["global_mean_rmse", "psnr", "ssim"]]
        .mean()
        .rename(columns={"global_mean_rmse": "RMSE_mean_global_K", "psnr": "PSNR_dB", "ssim": "SSIM"})
    )

    paths = {
        "table_4_1": output_dir / "table_4_1_cloud_only_primary.csv",
        "table_4_2": output_dir / "table_4_2_thin_thick.csv",
        "table_4_3": output_dir / "table_4_3_global_optical.csv",
    }

    table_4_1.to_csv(paths["table_4_1"], index=False)
    table_4_2.to_csv(paths["table_4_2"], index=False)
    table_4_3.to_csv(paths["table_4_3"], index=False)

    if "diffusion" in set(frame["method"]):
        diffusion = frame[frame["method"] == "diffusion"].copy()
        table_4_4 = (
            diffusion.groupby("city", as_index=False)["rmse"]
            .mean()
            .rename(columns={"rmse": "RMSE_K"})
            .sort_values("city")
        )
        table_4_4.to_csv(output_dir / "table_4_4_citywise_diffusion.csv", index=False)
        paths["table_4_4"] = output_dir / "table_4_4_citywise_diffusion.csv"

    return paths
