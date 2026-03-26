from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RepositoryConfig
from .datasets import HlsSyntheticDataset
from .models import DiffusionUNet, GaussianDiffusion, UNet2d
from .utils import append_jsonl, seed_everything, utc_timestamp, write_json


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_dir(cfg: RepositoryConfig, prefix: str) -> Path:
    run_dir = cfg.drive.runs_path / f"{prefix}_{utc_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_loader(frame: pd.DataFrame, cfg: RepositoryConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = HlsSyntheticDataset(frame, cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_regression_unet(frame: pd.DataFrame, cfg: RepositoryConfig) -> Path:
    seed_everything(cfg.training.seed)
    device = _device()
    run_dir = _run_dir(cfg, "unet")
    write_json(run_dir / "config.json", cfg.to_dict())

    train_loader = _build_loader(frame, cfg, "train", shuffle=True)
    val_loader = _build_loader(frame, cfg, "val", shuffle=False)

    model = UNet2d(in_channels=10, out_channels=8).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    criterion = nn.MSELoss()
    best_val = float("inf")
    patience = 0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"U-Net train {epoch + 1}/{cfg.training.epochs}"):
            observed = batch["observed"].to(device)
            target = batch["target_full"].to(device)
            pred = model(observed)
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))
            global_step += 1
            if global_step >= cfg.training.max_iterations:
                break

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                observed = batch["observed"].to(device)
                target = batch["target_full"].to(device)
                pred = model(observed)
                val_losses.append(float(criterion(pred, target).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        append_jsonl(
            run_dir / "history.jsonl",
            {"epoch": epoch + 1, "step": global_step, "train_loss": train_loss, "val_loss": val_loss},
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": deepcopy(model.state_dict()), "val_loss": best_val}, run_dir / "best.pt")
            patience = 0
        else:
            patience += 1
            if patience >= cfg.training.early_stopping_patience:
                break
        if global_step >= cfg.training.max_iterations:
            break

    return run_dir


def train_diffusion_model(frame: pd.DataFrame, cfg: RepositoryConfig) -> Path:
    seed_everything(cfg.training.seed)
    device = _device()
    run_dir = _run_dir(cfg, "diffusion")
    write_json(run_dir / "config.json", cfg.to_dict())

    train_loader = _build_loader(frame, cfg, "train", shuffle=True)
    val_loader = _build_loader(frame, cfg, "val", shuffle=False)

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
        model=network,
        timesteps=cfg.diffusion.timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
    ).to(device)
    optimizer = AdamW(diffusion.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    ema_state = deepcopy(diffusion.state_dict())
    best_val = float("inf")
    patience = 0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        diffusion.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Diffusion train {epoch + 1}/{cfg.training.epochs}"):
            condition = batch["observed"].to(device)
            target = batch["target_full"].to(device)
            loss = diffusion.training_loss(target, condition)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                state = diffusion.state_dict()
                for key, value in state.items():
                    ema_state[key] = cfg.diffusion.ema_decay * ema_state[key] + (1.0 - cfg.diffusion.ema_decay) * value.detach()

            train_losses.append(float(loss.item()))
            global_step += 1
            if global_step >= cfg.training.max_iterations:
                break

        diffusion.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                condition = batch["observed"].to(device)
                target = batch["target_full"].to(device)
                val_losses.append(float(diffusion.training_loss(target, condition).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        append_jsonl(
            run_dir / "history.jsonl",
            {"epoch": epoch + 1, "step": global_step, "train_noise_mse": train_loss, "val_noise_mse": val_loss},
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": diffusion.state_dict(),
                    "ema_state": ema_state,
                    "val_noise_mse": best_val,
                },
                run_dir / "best.pt",
            )
            patience = 0
        else:
            patience += 1
            if patience >= cfg.training.early_stopping_patience:
                break
        if global_step >= cfg.training.max_iterations:
            break

    return run_dir
