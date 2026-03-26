from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from topview.baselines import run_baseline
from topview.config import RepositoryConfig, build_repository_config
from topview.datasets import HlsSyntheticDataset
from topview.evaluation import evaluate_patch
from topview.models import DiffusionUNet, GaussianDiffusion, UNet2d
from topview.utils import write_json


def _load_unet(checkpoint: str, device: torch.device) -> UNet2d:
    model = UNet2d(in_channels=10, out_channels=8).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _load_diffusion(checkpoint: str, device: torch.device, cfg: RepositoryConfig) -> GaussianDiffusion:
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
    state = torch.load(checkpoint, map_location=device)
    diffusion.load_state_dict(state.get("ema_state", state["model_state"]))
    diffusion.eval()
    return diffusion


def _aggregate(metric_dicts: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_dicts[0]:
        values = [entry[key] for entry in metric_dicts if not np.isnan(entry[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--unet-checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = build_repository_config(args.root)
    split_path = args.splits or str(cfg.drive.splits_path / "patch_splits.csv")
    frame = pd.read_csv(split_path)
    dataset = HlsSyntheticDataset(frame, cfg, split="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods: dict[str, object] = {"idw": "idw", "ok": "ok"}
    if args.unet_checkpoint:
        methods["unet"] = _load_unet(args.unet_checkpoint, device)
    if args.diffusion_checkpoint:
        methods["diffusion"] = _load_diffusion(args.diffusion_checkpoint, device, cfg)

    all_metrics: dict[str, list[dict]] = {name: [] for name in methods}

    for batch in tqdm(loader, desc="Evaluating"):
        observed = batch["observed"]
        target_full = batch["target_full"]
        target_thermal = batch["target_thermal"]
        cloud_mask = batch["cloud_mask"]
        soft_mask = batch["soft_mask"]
        known_values = observed[:, :8]

        for method_name, method in methods.items():
            if isinstance(method, str):
                preds = []
                for index in range(observed.shape[0]):
                    preds.append(
                        run_baseline(
                            method,
                            known_values[index].numpy(),
                            cloud_mask[index].numpy(),
                            idw_power=cfg.baselines.idw_power,
                            idw_neighbors=cfg.baselines.idw_neighbors,
                            ok_variogram_model=cfg.baselines.ok_variogram_model,
                            ok_max_points=cfg.baselines.ok_max_points,
                        )
                    )
                pred_full = np.stack(preds, axis=0)
            elif isinstance(method, UNet2d):
                with torch.no_grad():
                    pred_full = method(observed.to(device)).cpu().numpy()
            else:
                assert isinstance(method, GaussianDiffusion)
                with torch.no_grad():
                    pred_full = method.repaint_inpaint(
                        condition=observed.to(device),
                        known_values=known_values.to(device),
                        cloud_mask=cloud_mask.to(device),
                        sampling_steps=cfg.diffusion.sampling_steps,
                        jump_length=cfg.diffusion.repaint_jump_length,
                        resamples=cfg.diffusion.repaint_resamples,
                    ).cpu().numpy()

            for index in range(pred_full.shape[0]):
                bundle = evaluate_patch(
                    pred_thermal=pred_full[index, 6:8],
                    ref_thermal=target_thermal[index].numpy(),
                    pred_rgb=np.clip(pred_full[index, 0:3], 0.0, 1.0),
                    ref_rgb=np.clip(target_full[index, 0:3].numpy(), 0.0, 1.0),
                    mask=cloud_mask[index].numpy(),
                    soft_mask=soft_mask[index].numpy(),
                )
                record = {
                    "method": method_name,
                    "patch_id": batch["patch_id"][index],
                    "city": batch["city"][index],
                    **asdict(bundle),
                }
                all_metrics[method_name].append(record)

    summary = {
        name: _aggregate(
            [
                {key: value for key, value in record.items() if key not in {"method", "patch_id", "city"}}
                for record in metric_dicts
            ]
        )
        for name, metric_dicts in all_metrics.items()
    }
    report_path = cfg.drive.reports_path / "evaluation_summary.json"
    write_json(report_path, summary)
    per_patch = pd.DataFrame([record for metric_dicts in all_metrics.values() for record in metric_dicts])
    per_patch_path = cfg.drive.reports_path / "evaluation_per_patch.csv"
    per_patch.to_csv(per_patch_path, index=False)
    print(f"Saved evaluation summary to {report_path}")
    print(f"Saved per-patch metrics to {per_patch_path}")


if __name__ == "__main__":
    main()
