from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from topview.config import build_repository_config
from topview.io import cloud_like_mask_from_quality_layer, load_patch, prepare_thermal_for_modeling
from topview.models import DiffusionUNet, GaussianDiffusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    frame = pd.read_csv(args.manifest).head(args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = DiffusionUNet(
        in_channels=18,
        out_channels=8,
        image_size=cfg.diffusion.image_size,
        base_channels=cfg.diffusion.base_channels,
        channel_mults=cfg.diffusion.channel_mults,
        attention_resolutions=cfg.diffusion.attention_resolutions,
        blocks_per_level=cfg.diffusion.num_res_blocks,
        num_heads=cfg.diffusion.attention_heads,
    ).to(device)
    diffusion = GaussianDiffusion(
        network,
        timesteps=cfg.diffusion.timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    diffusion.load_state_dict(state.get("ema_state", state["model_state"]))
    diffusion.eval()

    out_dir = cfg.drive.reports_path / "real_cloud_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in frame.iterrows():
        patch = load_patch(row["path"], cfg)
        reflectance = patch["reflectance"]
        thermal = prepare_thermal_for_modeling(patch["thermal"], cfg, row["path"], patch.get("metadata", {}))
        fmask = patch["fmask"]
        cloud_mask = cloud_like_mask_from_quality_layer(fmask, patch.get("metadata", {})).astype(np.float32)
        soft_mask = cloud_mask

        observed = np.concatenate(
            [reflectance, thermal, fmask.astype(np.float32, copy=False)[None, ...], soft_mask[None, ...]],
            axis=0,
        )
        observed_tensor = torch.from_numpy(observed[None, ...]).to(device)
        known_values = torch.from_numpy(np.concatenate([reflectance, thermal], axis=0)[None, ...]).to(device)
        cloud_mask_tensor = torch.from_numpy(cloud_mask[None, ...]).to(device)

        with torch.no_grad():
            pred = diffusion.repaint_inpaint(
                condition=observed_tensor,
                known_values=known_values,
                cloud_mask=cloud_mask_tensor,
                sampling_steps=cfg.diffusion.sampling_steps,
                jump_length=cfg.diffusion.repaint_jump_length,
                resamples=cfg.diffusion.repaint_resamples,
            ).cpu().numpy()[0]

        np.savez_compressed(
            out_dir / f"{row['patch_id']}_prediction.npz",
            prediction=pred,
            cloud_mask=cloud_mask,
            observed=observed,
        )

    print(f"Saved real-cloud inference outputs to {out_dir}")


if __name__ == "__main__":
    main()
