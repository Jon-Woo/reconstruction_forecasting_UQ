#!/usr/bin/env python3
"""
General conditional Gaussian-source stochastic-interpolant pipeline.

The model learns a conditional drift b_theta(s, x, y) from paired arrays
(Y, U), where Y is a conditioning vector and U is a target vector.  The
artificial interpolant is

    I_s = alpha(s) Z + beta(s) U + rho(s) B_s,

where Z ~ N(0, I_d), B_s ~ N(0, s I_d), and all three variables are mutually
independent conditional on (Y,U).  The regression target is

    R_s = alpha'(s) Z + beta'(s) U + rho'(s) B_s.

The SDE used for sampling is

    dG_s = b_theta(s, G_s, y) ds + rho(s) dW_s,
    G_0 ~ N(0, I_d).

This file contains no application-specific CFD code.
Dependencies: numpy, matplotlib, torch.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


Array = np.ndarray


@dataclass(frozen=True)
class DelayDatasetConfig:
    delay_steps: int
    embedding_dim: int
    gap_steps: int = 0
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    @property
    def window_span(self) -> int:
        return (self.embedding_dim - 1) * self.delay_steps

    @property
    def anchor_stride(self) -> int:
        # Consecutive retained windows share no raw sample indices.
        return self.window_span + self.gap_steps + 1


@dataclass(frozen=True)
class InterpolantConfig:
    eta: float = 0.25

    def coefficients(self, s: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Chen-style smooth schedule adapted to a Gaussian source:
          alpha=1-s, beta=s^2, rho=eta(1-s).
        It satisfies I_0=Z and I_1=U, and beta'(0)=0.
        """
        alpha = 1.0 - s
        beta = s.square()
        rho = self.eta * (1.0 - s)
        dalpha = -torch.ones_like(s)
        dbeta = 2.0 * s
        drho = -self.eta * torch.ones_like(s)
        return alpha, beta, rho, dalpha, dbeta, drho


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 256
    depth: int = 4
    time_features: int = 8
    batch_size: int = 256
    epochs: int = 2000
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0
    patience: int = 150
    seed: int = 17
    device: str = "auto"


@dataclass(frozen=True)
class Standardizer:
    mean: Array
    std: Array

    @classmethod
    def fit(cls, x: Array, eps: float = 1.0e-8) -> "Standardizer":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, x: Array) -> Array:
        return (np.asarray(x) - self.mean) / self.std

    def inverse_transform(self, x: Array) -> Array:
        return np.asarray(x) * self.std + self.mean


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_series(observations: Array, targets: Array) -> Tuple[Array, Array]:
    observations = np.asarray(observations, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if observations.ndim != 2 or targets.ndim != 2:
        raise ValueError("observations and targets must have shape (T,p) and (T,d).")
    if observations.shape[0] != targets.shape[0]:
        raise ValueError("observations and targets must have the same time length.")
    if observations.shape[0] < 3:
        raise ValueError("At least three time samples are required.")
    if not np.isfinite(observations).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite values found in observations or targets.")
    return observations, targets


def chronological_segments(
    n_times: int, train_fraction: float, val_fraction: float
) -> Dict[str, Tuple[int, int]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie in (0,1).")
    if not (0.0 <= val_fraction < 1.0 - train_fraction):
        raise ValueError("val_fraction must lie in [0,1-train_fraction).")
    n_train = int(math.floor(train_fraction * n_times))
    n_val = int(math.floor(val_fraction * n_times))
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_times),
    }


def build_nonoverlapping_delay_pairs(
    observations: Array,
    targets: Array,
    cfg: DelayDatasetConfig,
) -> Dict[str, Dict[str, Array]]:
    """
    Construct Y_j=[x_j,x_{j-l},...,x_{j-(m-1)l}] and U_j=target_j.

    Windows are generated separately inside chronological train/validation/test
    blocks.  Within each block, retained anchors differ by
    (m-1)*delay_steps + gap_steps + 1, so no two retained examples share a raw
    time index.  This removes literal overlap but does not prove statistical
    independence of samples from a dynamical trajectory.
    """
    observations, targets = _validate_series(observations, targets)
    if cfg.delay_steps < 1 or cfg.embedding_dim < 1 or cfg.gap_steps < 0:
        raise ValueError("delay_steps, embedding_dim must be positive; gap_steps nonnegative.")

    segments = chronological_segments(
        len(observations), cfg.train_fraction, cfg.val_fraction
    )
    result: Dict[str, Dict[str, Array]] = {}
    offsets = np.arange(cfg.embedding_dim, dtype=np.int64) * cfg.delay_steps

    for name, (start, stop) in segments.items():
        first_anchor = start + cfg.window_span
        anchors = np.arange(first_anchor, stop, cfg.anchor_stride, dtype=np.int64)
        if anchors.size == 0:
            raise ValueError(
                f"No {name} windows. Increase trajectory length or reduce tau/m/gap."
            )
        Y = np.stack(
            [observations[a - offsets].reshape(-1) for a in anchors], axis=0
        )
        U = targets[anchors]
        # Explicit exact check of disjoint raw index sets.
        raw_sets = [set((a - offsets).tolist()) for a in anchors]
        for left, right in zip(raw_sets[:-1], raw_sets[1:]):
            if left.intersection(right):
                raise AssertionError("Internal error: retained windows overlap.")
        result[name] = {"Y": Y, "U": U, "anchors": anchors}
    return result


class FourierTimeEmbedding(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive.")
        frequencies = (2.0 ** torch.arange(n_features, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.frequencies.numel()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        angles = s * self.frequencies[None, :]
        return torch.cat([s, torch.sin(angles), torch.cos(angles)], dim=-1)


class ConditionalDriftMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        time_features: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.time_embedding = FourierTimeEmbedding(time_features)
        input_dim = state_dim + condition_dim + self.time_embedding.output_dim
        layers = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        layers.append(nn.Linear(current, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if s.ndim == 1:
            s = s[:, None]
        return self.net(torch.cat([self.time_embedding(s), x, y], dim=-1))


def _device_from_config(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def simulation_free_batch(
    y: torch.Tensor,
    u: torch.Tensor,
    interpolant: InterpolantConfig,
    *,
    s_eps: float = 1.0e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw one unbiased Monte Carlo sample of the time-integrated square loss.

    B_s is sampled from its marginal law sqrt(s)*epsilon.  No Brownian path is
    needed during training because the objective at a sampled s depends only on
    that one-time marginal.
    """
    batch, d = u.shape
    s = torch.rand(batch, 1, device=u.device, dtype=u.dtype)
    s = s.clamp(min=s_eps, max=1.0 - s_eps)
    z = torch.randn_like(u)
    eps = torch.randn_like(u)
    b_s = torch.sqrt(s) * eps
    alpha, beta, rho, dalpha, dbeta, drho = interpolant.coefficients(s)
    i_s = alpha * z + beta * u + rho * b_s
    r_s = dalpha * z + dbeta * u + drho * b_s
    return s, i_s, r_s


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader: DataLoader,
    interpolant: InterpolantConfig,
    device: torch.device,
    mc_repeats: int = 4,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for y, u in loader:
        y, u = y.to(device), u.to(device)
        for _ in range(mc_repeats):
            s, i_s, r_s = simulation_free_batch(y, u, interpolant)
            err = model(s, i_s, y) - r_s
            total += float(torch.sum(err.square()).cpu())
            count += err.numel()
    return total / max(count, 1)


def train_conditional_drift(
    train_y: Array,
    train_u: Array,
    val_y: Array,
    val_u: Array,
    output_dir: str | Path,
    train_cfg: TrainConfig = TrainConfig(),
    interpolant_cfg: InterpolantConfig = InterpolantConfig(),
) -> Dict[str, object]:
    set_seed(train_cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_scaler = Standardizer.fit(train_y)
    u_scaler = Standardizer.fit(train_u)
    train_y_n = y_scaler.transform(train_y).astype(np.float32)
    train_u_n = u_scaler.transform(train_u).astype(np.float32)
    val_y_n = y_scaler.transform(val_y).astype(np.float32)
    val_u_n = u_scaler.transform(val_u).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(train_y_n), torch.from_numpy(train_u_n))
    val_ds = TensorDataset(torch.from_numpy(val_y_n), torch.from_numpy(val_u_n))
    generator = torch.Generator().manual_seed(train_cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=min(train_cfg.batch_size, len(train_ds)),
        shuffle=True, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(train_cfg.batch_size, len(val_ds)), shuffle=False
    )

    device = _device_from_config(train_cfg.device)
    model = ConditionalDriftMLP(
        state_dim=train_u.shape[1],
        condition_dim=train_y.shape[1],
        hidden_dim=train_cfg.hidden_dim,
        depth=train_cfg.depth,
        time_features=train_cfg.time_features,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    history = {"epoch": [], "train_mse": [], "val_mse": []}
    best_state = None
    best_val = math.inf
    stale = 0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for y, u in train_loader:
            y, u = y.to(device), u.to(device)
            s, i_s, r_s = simulation_free_batch(y, u, interpolant_cfg)
            pred = model(s, i_s, y)
            loss = torch.mean((pred - r_s).square())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            train_sum += float(torch.sum((pred.detach() - r_s).square()).cpu())
            train_count += pred.numel()

        train_mse = train_sum / max(train_count, 1)
        val_mse = estimate_loss(model, val_loader, interpolant_cfg, device)
        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:5d} train_mse={train_mse:.6e} val_mse={val_mse:.6e}")
        if stale >= train_cfg.patience:
            print(f"early stopping at epoch {epoch}; best val={best_val:.6e}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model.")
    model.load_state_dict(best_state)
    model.eval()

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": {
            "state_dim": train_u.shape[1],
            "condition_dim": train_y.shape[1],
            "hidden_dim": train_cfg.hidden_dim,
            "depth": train_cfg.depth,
            "time_features": train_cfg.time_features,
        },
        "train_config": asdict(train_cfg),
        "interpolant_config": asdict(interpolant_cfg),
        "y_mean": y_scaler.mean,
        "y_std": y_scaler.std,
        "u_mean": u_scaler.mean,
        "u_std": u_scaler.std,
        "history": history,
        "best_val_mse": best_val,
    }
    torch.save(checkpoint, output_dir / "conditional_drift.pt")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history["epoch"], history["train_mse"], label="training")
    ax.semilogy(history["epoch"], history["val_mse"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("drift regression MSE")
    ax.set_title("Stochastic-interpolant drift training")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_errors_log.png", dpi=180)
    plt.close(fig)

    return {
        "model": model,
        "device": device,
        "y_scaler": y_scaler,
        "u_scaler": u_scaler,
        "history": history,
        "checkpoint": checkpoint,
    }


def load_trained_pipeline(path: str | Path, device: str = "auto") -> Dict[str, object]:
    dev = _device_from_config(device)
    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    model = ConditionalDriftMLP(**checkpoint["model_config"]).to(dev)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "device": dev,
        "y_scaler": Standardizer(checkpoint["y_mean"], checkpoint["y_std"]),
        "u_scaler": Standardizer(checkpoint["u_mean"], checkpoint["u_std"]),
        "interpolant_cfg": InterpolantConfig(**checkpoint["interpolant_config"]),
        "checkpoint": checkpoint,
    }


@torch.no_grad()
def sample_conditional_sde(
    model: nn.Module,
    y: Array,
    y_scaler: Standardizer,
    u_scaler: Standardizer,
    interpolant_cfg: InterpolantConfig,
    *,
    n_samples: int = 200,
    n_steps: int = 300,
    device: Optional[torch.device] = None,
    seed: int = 123,
) -> Array:
    """
    Euler--Maruyama sampling from the learned conditional SDE.

    A fresh Gaussian initial state and fresh Brownian increments are used for
    every ensemble member.  The returned samples are in physical target units.
    """
    if n_samples < 1 or n_steps < 1:
        raise ValueError("n_samples and n_steps must be positive.")
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    y = np.asarray(y, dtype=np.float64).reshape(1, -1)
    y_n = torch.as_tensor(y_scaler.transform(y), dtype=torch.float32, device=device)
    y_batch = y_n.repeat(n_samples, 1)
    d = model.state_dim
    x = torch.randn((n_samples, d), generator=gen, device=device)
    ds = 1.0 / n_steps

    model.eval()
    for k in range(n_steps):
        s_value = k * ds
        s = torch.full((n_samples, 1), s_value, dtype=x.dtype, device=device)
        drift = model(s, x, y_batch)
        _, _, rho, _, _, _ = interpolant_cfg.coefficients(s)
        dW = math.sqrt(ds) * torch.randn(
            x.shape, generator=gen, device=device, dtype=x.dtype
        )
        x = x + drift * ds + rho * dW

    return u_scaler.inverse_transform(x.cpu().numpy())


def conditional_ensemble_metrics(samples: Array, truth: Array) -> Dict[str, float]:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    mean = samples.mean(axis=0, keepdims=True)
    rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
    spread = float(np.sqrt(np.mean(np.var(samples, axis=0, ddof=1))))
    return {"ensemble_mean_rmse": rmse, "ensemble_spread_rms": spread}
