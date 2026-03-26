from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        values = timesteps[:, None].float() * frequencies[None, :]
        embedding = torch.cat([torch.sin(values), torch.cos(values)], dim=1)
        if self.dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h + residual)
        return h


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.num_heads = max(1, min(num_heads, channels))
        if channels % self.num_heads != 0:
            self.num_heads = 1
        self.head_dim = channels // self.num_heads
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        normalized = self.norm(x).reshape(batch, channels, tokens)
        q, k, v = self.qkv(normalized).chunk(3, dim=1)
        q = q.reshape(batch, self.num_heads, self.head_dim, tokens).permute(0, 1, 3, 2)
        k = k.reshape(batch, self.num_heads, self.head_dim, tokens)
        v = v.reshape(batch, self.num_heads, self.head_dim, tokens).permute(0, 1, 3, 2)
        attn = torch.softmax(torch.matmul(q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(batch, channels, tokens)
        out = self.proj(out).reshape(batch, channels, height, width)
        return x + out


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int = 256,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8, 8),
        attention_resolutions: tuple[int, ...] = (32, 16, 8),
        blocks_per_level: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        widths = [base_channels * mult for mult in channel_mults]
        self.stem = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_width = widths[0]
        current_resolution = image_size
        for width in widths:
            blocks = nn.ModuleList([TimeBlock(in_width, width, time_dim)])
            for _ in range(max(blocks_per_level - 1, 0)):
                blocks.append(TimeBlock(width, width, time_dim))
            attn = SelfAttention2d(width, num_heads=num_heads) if current_resolution in attention_resolutions else nn.Identity()
            self.down_blocks.append(nn.ModuleDict({"blocks": blocks, "attn": attn}))
            self.downsamples.append(nn.Conv2d(width, width, kernel_size=4, stride=2, padding=1))
            in_width = width
            current_resolution //= 2

        self.mid1 = TimeBlock(widths[-1], widths[-1], time_dim)
        self.mid_attn = SelfAttention2d(widths[-1], num_heads=num_heads) if current_resolution in attention_resolutions else nn.Identity()
        self.mid2 = TimeBlock(widths[-1], widths[-1], time_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_resolution = max(current_resolution, 1)
        for width in reversed(widths):
            self.upsamples.append(nn.ConvTranspose2d(in_width, width, kernel_size=4, stride=2, padding=1))
            current_resolution *= 2
            blocks = nn.ModuleList([TimeBlock(width * 2, width, time_dim)])
            for _ in range(max(blocks_per_level - 1, 0)):
                blocks.append(TimeBlock(width, width, time_dim))
            attn = SelfAttention2d(width, num_heads=num_heads) if current_resolution in attention_resolutions else nn.Identity()
            self.up_blocks.append(nn.ModuleDict({"blocks": blocks, "attn": attn}))
            in_width = width

        self.out = nn.Sequential(
            nn.GroupNorm(8, widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(timesteps)
        x = self.stem(x)
        skips: list[torch.Tensor] = []

        for level, downsample in zip(self.down_blocks, self.downsamples):
            for block in level["blocks"]:
                x = block(x, time_emb)
            x = level["attn"](x)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, time_emb)

        for upsample, level in zip(self.upsamples, self.up_blocks):
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in level["blocks"]:
                x = block(x, time_emb)
            x = level["attn"](x)
        return self.out(x)


class GaussianDiffusion(nn.Module):
    def __init__(self, model: DiffusionUNet, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-8),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None, None]
        one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]
        return alpha * x_start + one_minus * noise

    def training_loss(self, x_start: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        pred_noise = self.model(torch.cat([x_t, condition], dim=1), timesteps)
        return nn.functional.mse_loss(pred_noise, noise)

    def p_sample(self, x_t: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        betas_t = self.betas[timesteps][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[timesteps][:, None, None, None]
        pred_noise = self.model(torch.cat([x_t, condition], dim=1), timesteps)
        model_mean = sqrt_recip_alpha * (x_t - betas_t * pred_noise / torch.clamp(sqrt_one_minus, min=1e-8))
        if torch.all(timesteps == 0):
            return model_mean
        variance = self.posterior_variance[timesteps][:, None, None, None]
        return model_mean + torch.sqrt(torch.clamp(variance, min=1e-12)) * torch.randn_like(x_t)

    @torch.no_grad()
    def repaint_inpaint(
        self,
        condition: torch.Tensor,
        known_values: torch.Tensor,
        cloud_mask: torch.Tensor,
        sampling_steps: int,
        jump_length: int,
        resamples: int,
    ) -> torch.Tensor:
        batch_size = condition.shape[0]
        x = torch.randn_like(known_values)
        known_mask = (1.0 - cloud_mask)[:, None, ...]
        unknown_mask = cloud_mask[:, None, ...]
        schedule = torch.linspace(self.timesteps - 1, 0, sampling_steps, device=condition.device).long()

        for idx, timestep in enumerate(schedule):
            t = torch.full((batch_size,), int(timestep.item()), device=condition.device, dtype=torch.long)
            for _ in range(max(resamples, 1)):
                x = self.p_sample(x, condition, t)
                if timestep.item() > 0:
                    x = x * unknown_mask + self.q_sample(known_values, t) * known_mask
                else:
                    x = x * unknown_mask + known_values * known_mask

            if jump_length > 0 and idx < len(schedule) - 1:
                jumped = min(int(timestep.item()) + jump_length, self.timesteps - 1)
                if jumped > int(timestep.item()):
                    jump_t = torch.full((batch_size,), jumped, device=condition.device, dtype=torch.long)
                    x = self.q_sample(x, jump_t)

        return x
