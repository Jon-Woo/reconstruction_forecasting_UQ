""""
Core idea
---------
Given empirical pairs (Z, Y), where
    Z in R^{m d_p} is the delay embedding,
    Y in R^{d_y} is either x_p(t+tau), x_p(t+tau)-x_p(t), or x(t),
learn a conditional generative model for p(Y | Z=z).

The source and target are made dimension-compatible by working on the
augmented product space (Z, Y):
    source: (Z, Y_0),  Y_0 ~ q_0(. | Z)
    target: (Z, Y),    Y ~ p_data(. | Z)
The Z-coordinate is identical at the two endpoints and is kept fixed.  The
stochastic interpolant is only evolved in the Y-coordinate, while Z is passed
as context to the neural drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


Array = np.ndarray


class Standardizer:
    """Mean/std standardization for numpy arrays and torch tensors."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: Array) -> "Standardizer":
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.maximum(self.std, self.eps).astype(np.float32)
        return self

    def transform(self, x: Array) -> Array:
        self._check_fit()
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: Array) -> Array:
        self._check_fit()
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fit()
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def inverse_transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fit()
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean

    def _check_fit(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fit yet.")


@dataclass
class DelayPairData:
    """Container for delay-coordinate training pairs."""

    z: Array
    y: Array
    current_partial: Array
    time_indices: Array
    metadata: dict


def make_delay_pairs(
    x_full: Array,
    partial_indices: Optional[Sequence[int]] = None,
    delay_steps: int = 1,
    m: int = 3,
    horizon_steps: Optional[int] = None,
    task: str = "forecast",
    predict_increment: bool = False,
) -> DelayPairData:
    """Build supervised pairs (Z, Y) from a full trajectory.

    Parameters
    ----------
    x_full:
        Array of shape (T, d_x). Each row is the full state x(t_k).
    partial_indices:
        Components used for the observed partial state x_p. If None, use all
        state components as the observation.
    delay_steps:
        Integer delay tau measured in sample steps. For data sampled every dt,
        the physical delay is tau = delay_steps * dt.
    m:
        Number of delays in Z = [x_p(t), x_p(t-tau), ..., x_p(t-(m-1)tau)].
    horizon_steps:
        Forecast horizon measured in sample steps. If None, defaults to
        delay_steps so that the model predicts x_p(t+tau).
    task:
        "forecast" or "reconstruction".
        - forecast: Y = x_p(t + horizon)
        - reconstruction: Y = x_full(t)
    predict_increment:
        If True for forecast, use Y = x_p(t+horizon) - x_p(t). Sampling then
        returns increments; add x_p(t) to recover the forecasted partial state.

    Returns
    -------
    DelayPairData containing z with shape (N, m*d_p) and y with shape
    (N, d_y).
    """

    if delay_steps < 1:
        raise ValueError("delay_steps must be a positive integer.")
    if m < 1:
        raise ValueError("m must be a positive integer.")

    x_full = np.asarray(x_full, dtype=np.float32)
    if x_full.ndim != 2:
        raise ValueError("x_full must have shape (T, d_x).")

    T, d_x = x_full.shape
    if partial_indices is None:
        xp = x_full
        partial_indices = tuple(range(d_x))
    else:
        partial_indices = tuple(int(i) for i in partial_indices)
        xp = x_full[:, partial_indices]

    horizon_steps = delay_steps if horizon_steps is None else int(horizon_steps)
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be a positive integer.")

    task = task.lower()
    if task not in {"forecast", "reconstruction"}:
        raise ValueError("task must be 'forecast' or 'reconstruction'.")
    if predict_increment and task != "forecast":
        raise ValueError("predict_increment only applies to task='forecast'.")

    first_t = (m - 1) * delay_steps
    last_t_exclusive = T - horizon_steps if task == "forecast" else T
    if first_t >= last_t_exclusive:
        raise ValueError(
            "Not enough time samples for the requested delay embedding and horizon."
        )

    z_list, y_list, current_list, idx_list = [], [], [], []
    for k in range(first_t, last_t_exclusive):
        delays = [xp[k - j * delay_steps] for j in range(m)]
        z_k = np.concatenate(delays, axis=0)
        current = xp[k].copy()
        if task == "forecast":
            y_k = xp[k + horizon_steps].copy()
            if predict_increment:
                y_k = y_k - current
        else:
            y_k = x_full[k].copy()
        z_list.append(z_k)
        y_list.append(y_k)
        current_list.append(current)
        idx_list.append(k)

    z = np.stack(z_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    current_partial = np.stack(current_list).astype(np.float32)
    time_indices = np.asarray(idx_list, dtype=np.int64)
    metadata = {
        "task": task,
        "predict_increment": bool(predict_increment),
        "delay_steps": int(delay_steps),
        "m": int(m),
        "horizon_steps": int(horizon_steps),
        "partial_indices": tuple(partial_indices),
        "x_dim": int(d_x),
        "partial_dim": int(xp.shape[1]),
        "z_dim": int(z.shape[1]),
        "y_dim": int(y.shape[1]),
    }
    return DelayPairData(z=z, y=y, current_partial=current_partial, time_indices=time_indices, metadata=metadata)


@dataclass
class QuadraticBetaSchedule:
    """Interpolant coefficients.

    alpha(s) = 1 - s
    beta(s)  = s^2
    rho(s)   = eps * (1 - s)

    beta'(0)=0 is useful because the target contribution enters gently near the
    source endpoint, which stabilizes training in the point-source setting.
    """

    eps: float = 0.25

    def coefficients(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = 1.0 - s
        beta = s.square()
        rho = self.eps * (1.0 - s)
        return alpha, beta, rho

    def derivatives(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dalpha = -torch.ones_like(s)
        dbeta = 2.0 * s
        drho = -self.eps * torch.ones_like(s)
        return dalpha, dbeta, drho


class MLPDrift(nn.Module):
    """Drift b_theta(s, y, z) for the generative SDE in target space."""

    def __init__(
        self,
        z_dim: int,
        y_dim: int,
        hidden: Sequence[int] = (256, 256, 256),
        activation: str = "silu",
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.y_dim = int(y_dim)

        in_dim = 1 + self.y_dim + self.z_dim
        dims = [in_dim, *map(int, hidden), self.y_dim]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(a, b))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError("activation must be 'relu', 'gelu', or 'silu'.")
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if s.ndim == 1:
            s = s[:, None]
        if s.shape[0] != y.shape[0] or z.shape[0] != y.shape[0]:
            raise ValueError("s, y, and z must have the same batch dimension.")
        return self.net(torch.cat([s, y, z], dim=-1))


class ConditionalStochasticInterpolant(nn.Module):
    """Conditional stochastic interpolant model for p(Y | Z).

    The model evolves only in the Y-coordinate. Z is fixed and used as context.
    All inputs are assumed to already be standardized.
    """

    def __init__(
        self,
        z_dim: int,
        y_dim: int,
        hidden: Sequence[int] = (256, 256, 256),
        schedule: Optional[QuadraticBetaSchedule] = None,
        base_std: float = 1.0,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.y_dim = int(y_dim)
        self.schedule = schedule if schedule is not None else QuadraticBetaSchedule()
        self.base_std = float(base_std)
        self.drift = MLPDrift(z_dim=z_dim, y_dim=y_dim, hidden=hidden)

    def sample_base(self, z: torch.Tensor) -> torch.Tensor:
        """Draw Y_0 ~ q_0 in standardized target coordinates."""
        return self.base_std * torch.randn(z.shape[0], self.y_dim, device=z.device, dtype=z.dtype)

    def interpolant_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simulation-free square loss for the conditional drift.

        For a minibatch of data pairs (z, y), draw y0 ~ q0, s ~ U(0,1),
        epsilon ~ N(0,I), form
            I_s = alpha_s y0 + beta_s y + rho_s sqrt(s) epsilon,
            R_s = alpha'_s y0 + beta'_s y + rho'_s sqrt(s) epsilon,
        and regress b_theta(s, I_s, z) onto R_s.
        """
        batch = y.shape[0]
        s = torch.rand(batch, 1, device=y.device, dtype=y.dtype)
        sqrt_s = torch.sqrt(torch.clamp(s, min=1e-6))
        eps = torch.randn_like(y)
        y0 = self.sample_base(z)

        alpha, beta, rho = self.schedule.coefficients(s)
        dalpha, dbeta, drho = self.schedule.derivatives(s)

        I_s = alpha * y0 + beta * y + rho * sqrt_s * eps
        R_s = dalpha * y0 + dbeta * y + drho * sqrt_s * eps
        pred = self.drift(s, I_s, z)
        return F.mse_loss(pred, R_s)

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        n_samples: int = 64,
        n_steps: int = 100,
        diffusion_scale: float = 1.0,
        keep_trajectory: bool = False,
    ) -> torch.Tensor:
        """Generate samples from the learned approximation to p(Y | Z=z).

        Parameters
        ----------
        z:
            Standardized conditioning embeddings with shape (B, z_dim).
        n_samples:
            Number of ensemble members per condition.
        n_steps:
            Euler-Maruyama steps in interpolant time s in [0,1].
        diffusion_scale:
            Multiplier on rho(s). Values >1 produce more diverse but rougher
            samples; values <1 approach a probability-flow-like sampler.
        keep_trajectory:
            If True, return all intermediate states with shape
            (n_steps+1, B, n_samples, y_dim). Otherwise return final samples
            with shape (B, n_samples, y_dim).
        """
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        self.eval()
        B = z.shape[0]
        z_rep = z[:, None, :].expand(B, n_samples, self.z_dim).reshape(B * n_samples, self.z_dim)
        y = self.sample_base(z_rep)
        ds = 1.0 / float(n_steps)
        traj = []
        if keep_trajectory:
            traj.append(y.reshape(B, n_samples, self.y_dim).clone())

        for i in range(n_steps):
            # Left endpoint Euler-Maruyama. Avoid exactly s=1 inside the loop.
            s_value = i * ds
            s = torch.full((B * n_samples, 1), s_value, device=y.device, dtype=y.dtype)
            _, _, rho = self.schedule.coefficients(s)
            b = self.drift(s, y, z_rep)
            noise = torch.randn_like(y)
            y = y + b * ds + diffusion_scale * rho * math.sqrt(ds) * noise
            if keep_trajectory:
                traj.append(y.reshape(B, n_samples, self.y_dim).clone())

        if keep_trajectory:
            return torch.stack(traj, dim=0)
        return y.reshape(B, n_samples, self.y_dim)


@dataclass
class TrainedSIModel:
    """A trained conditional stochastic interpolant plus standardizers."""

    model: ConditionalStochasticInterpolant
    z_scaler: Standardizer
    y_scaler: Standardizer
    metadata: dict
    train_losses: list

    @torch.no_grad()
    def sample_numpy(
        self,
        z: Array,
        n_samples: int = 64,
        n_steps: int = 100,
        diffusion_scale: float = 1.0,
        device: Optional[str] = None,
    ) -> Array:
        """Sample in original, unstandardized target coordinates."""
        device = device or next(self.model.parameters()).device.type
        z_std = self.z_scaler.transform(z)
        z_tensor = torch.as_tensor(z_std, dtype=torch.float32, device=device)
        y_std = self.model.sample(
            z_tensor,
            n_samples=n_samples,
            n_steps=n_steps,
            diffusion_scale=diffusion_scale,
        )
        B, S, dy = y_std.shape
        y_flat = y_std.reshape(B * S, dy).cpu().numpy()
        y = self.y_scaler.inverse_transform(y_flat).reshape(B, S, dy)
        return y


def train_interpolant(
    z: Array,
    y: Array,
    *,
    hidden: Sequence[int] = (256, 256, 256),
    eps: float = 0.25,
    base_std: float = 1.0,
    batch_size: int = 512,
    n_steps: int = 20_000,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: Optional[str] = None,
    print_every: int = 1000,
    metadata: Optional[dict] = None,
) -> TrainedSIModel:
    """Fit a conditional stochastic interpolant from arrays of pairs (z,y)."""
    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if z.ndim != 2 or y.ndim != 2:
        raise ValueError("z and y must both be rank-2 arrays.")
    if z.shape[0] != y.shape[0]:
        raise ValueError("z and y must contain the same number of samples.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    z_scaler = Standardizer().fit(z)
    y_scaler = Standardizer().fit(y)
    z_std = z_scaler.transform(z)
    y_std = y_scaler.transform(y)

    dataset = TensorDataset(
        torch.as_tensor(z_std, dtype=torch.float32),
        torch.as_tensor(y_std, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:
        raise ValueError("batch_size is larger than the dataset; reduce batch_size or add data.")
    loader_iter = iter(loader)

    model = ConditionalStochasticInterpolant(
        z_dim=z.shape[1],
        y_dim=y.shape[1],
        hidden=hidden,
        schedule=QuadraticBetaSchedule(eps=eps),
        base_std=base_std,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    for step in range(1, n_steps + 1):
        try:
            z_b, y_b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            z_b, y_b = next(loader_iter)

        z_b = z_b.to(device)
        y_b = y_b.to(device)
        loss = model.interpolant_loss(z_b, y_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        losses.append(float(loss.detach().cpu()))
        if print_every and step % print_every == 0:
            recent = np.mean(losses[-min(print_every, len(losses)):])
            print(f"step {step:>7d} | loss {recent:.6f}")

    return TrainedSIModel(
        model=model,
        z_scaler=z_scaler,
        y_scaler=y_scaler,
        metadata=dict(metadata or {}),
        train_losses=losses,
    )


def ensemble_summary(samples: Array) -> Tuple[Array, Array, Array, Array]:
    """Return mean, std, 5% quantile, and 95% quantile over ensemble axis."""
    samples = np.asarray(samples)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    return mean, std, q05, q95
