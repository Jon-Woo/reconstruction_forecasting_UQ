#!/usr/bin/env python3
"""
Generate conditional samples with Euler--Maruyama using a trained drift model.

This file is compatible with checkpoints produced by
`train_drift_with_visualization.py`.

The training checkpoint is expected to contain:

    model_state
    model_config
    interpolant_config

and, for sampling in physical coordinates:

    normalization_stats = {
        "mu_Y": ...,
        "sigma_Y": ...,
        "mu_U": ...,
        "sigma_U": ...,
    }

The learned conditional SDE is

    dG_s = b_theta(s, G_s, y_tilde) ds + rho(s) dW_s,
    G_0 ~ N(0, I_d),

where y_tilde is the normalized conditioning input. Euler--Maruyama uses

    G_{n+1}
        = G_n
          + b_theta(s_n, G_n, y_tilde) Delta_s
          + rho(s_n) sqrt(Delta_s) xi_n,

with independent xi_n ~ N(0, I_d).

The same diffusion schedule stored in the training checkpoint is used during
sampling. The terminal normalized samples are transformed back to physical
target coordinates and saved together with the original input y.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from train_drift_with_visualization import (
    ConditionalDriftMLP,
    InterpolantConfig,
    select_device,
    set_seed,
)


Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingConfig:
    """Euler--Maruyama and ensemble-generation settings."""

    num_samples: int = 1000
    num_steps: int = 1000
    sample_batch_size: int = 1024
    seed: int = 17
    device: str = "auto"

    def validate(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive.")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if self.sample_batch_size < 1:
            raise ValueError("sample_batch_size must be positive.")


@dataclass(frozen=True)
class NormalizationStats:
    """Componentwise normalization statistics used during training."""

    mu_Y: Array
    sigma_Y: Array
    mu_U: Array
    sigma_U: Array

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        condition_dim: int,
        target_dim: int,
    ) -> "NormalizationStats":
        required = ("mu_Y", "sigma_Y", "mu_U", "sigma_U")
        missing = [key for key in required if key not in values]

        if missing:
            raise KeyError(
                "Normalization statistics are missing: "
                + ", ".join(missing)
            )

        mu_Y = _as_vector(values["mu_Y"], "mu_Y", condition_dim)
        sigma_Y = _as_vector(
            values["sigma_Y"],
            "sigma_Y",
            condition_dim,
        )
        mu_U = _as_vector(values["mu_U"], "mu_U", target_dim)
        sigma_U = _as_vector(
            values["sigma_U"],
            "sigma_U",
            target_dim,
        )

        if np.any(sigma_Y <= 0.0):
            raise ValueError("Every entry of sigma_Y must be positive.")

        if np.any(sigma_U <= 0.0):
            raise ValueError("Every entry of sigma_U must be positive.")

        return cls(
            mu_Y=mu_Y,
            sigma_Y=sigma_Y,
            mu_U=mu_U,
            sigma_U=sigma_U,
        )

    def as_dict(self) -> Dict[str, Array]:
        return {
            "mu_Y": self.mu_Y,
            "sigma_Y": self.sigma_Y,
            "mu_U": self.mu_U,
            "sigma_U": self.sigma_U,
        }


# ---------------------------------------------------------------------------
# Array validation
# ---------------------------------------------------------------------------

def _as_vector(
    value: Any,
    name: str,
    expected_dimension: Optional[int] = None,
) -> Array:
    """Convert an input to a finite one-dimensional float32 array."""
    result = np.asarray(value, dtype=np.float32).reshape(-1)

    if result.size < 1:
        raise ValueError(f"{name} must be nonempty.")

    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    if (
        expected_dimension is not None
        and result.size != expected_dimension
    ):
        raise ValueError(
            f"{name} must have dimension {expected_dimension}; "
            f"received {result.size}."
        )

    return result


def normalize_condition(
    y: Array,
    stats: NormalizationStats,
) -> Array:
    """Normalize a physical conditioning vector with training statistics."""
    y = _as_vector(y, "y", stats.mu_Y.size)
    return (y - stats.mu_Y) / stats.sigma_Y


def denormalize_targets(
    normalized_samples: Array,
    stats: NormalizationStats,
) -> Array:
    """Map normalized target samples back to physical coordinates."""
    normalized_samples = np.asarray(
        normalized_samples,
        dtype=np.float32,
    )

    if normalized_samples.ndim != 2:
        raise ValueError(
            "normalized_samples must have shape (num_samples, target_dim)."
        )

    if normalized_samples.shape[1] != stats.mu_U.size:
        raise ValueError(
            "The normalized sample dimension does not match mu_U."
        )

    return stats.mu_U[None, :] + (
        stats.sigma_U[None, :] * normalized_samples
    )


# ---------------------------------------------------------------------------
# Checkpoint and normalization loading
# ---------------------------------------------------------------------------

def _torch_load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a complete training checkpoint.

    weights_only=False is required because the checkpoint contains model
    metadata, histories, and possibly NumPy normalization arrays in addition
    to the state dictionary.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose the
        # weights_only keyword.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must contain a dictionary.")

    return checkpoint


def load_normalization_stats_from_npz(
    path: str | Path,
    *,
    condition_dim: int,
    target_dim: int,
) -> NormalizationStats:
    """Load mu_Y, sigma_Y, mu_U, and sigma_U from an NPZ file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        values = {
            key: np.asarray(data[key])
            for key in ("mu_Y", "sigma_Y", "mu_U", "sigma_U")
            if key in data
        }

    return NormalizationStats.from_mapping(
        values,
        condition_dim=condition_dim,
        target_dim=target_dim,
    )


def load_trained_drift(
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
    normalization_file: Optional[str | Path] = None,
) -> Tuple[
    ConditionalDriftMLP,
    InterpolantConfig,
    NormalizationStats,
    Dict[str, Any],
    torch.device,
]:
    """
    Reconstruct the trained drift and associated sampling information.

    Normalization statistics are first read from the checkpoint. If they are
    absent, `normalization_file` must point to an NPZ file containing
    mu_Y, sigma_Y, mu_U, and sigma_U.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = select_device(device_name)
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)

    for key in ("model_state", "model_config", "interpolant_config"):
        if key not in checkpoint:
            raise KeyError(
                f"The training checkpoint does not contain '{key}'."
            )

    model_config = checkpoint["model_config"]

    target_dim = int(model_config["target_dim"])
    condition_dim = int(model_config["condition_dim"])
    hidden_dims = tuple(
        int(width)
        for width in model_config["hidden_dims"]
    )

    model = ConditionalDriftMLP(
        target_dim=target_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    interpolant_values = checkpoint["interpolant_config"]
    interpolant_config = InterpolantConfig(
        sigma_I=float(interpolant_values["sigma_I"])
    )
    interpolant_config.validate()

    if "normalization_stats" in checkpoint:
        normalization_stats = NormalizationStats.from_mapping(
            checkpoint["normalization_stats"],
            condition_dim=condition_dim,
            target_dim=target_dim,
        )
    elif normalization_file is not None:
        normalization_stats = load_normalization_stats_from_npz(
            normalization_file,
            condition_dim=condition_dim,
            target_dim=target_dim,
        )
    else:
        raise KeyError(
            "The checkpoint does not contain normalization_stats. "
            "Supply --normalization-file with an NPZ file containing "
            "mu_Y, sigma_Y, mu_U, and sigma_U."
        )

    return (
        model,
        interpolant_config,
        normalization_stats,
        checkpoint,
        device,
    )


# ---------------------------------------------------------------------------
# Euler--Maruyama sampling
# ---------------------------------------------------------------------------

def diffusion_coefficient(
    s: torch.Tensor,
    interpolant_config: InterpolantConfig,
) -> torch.Tensor:
    """
    Evaluate the same rho(s) used by train_drift_with_visualization.py.

    The training schedule is rho(s) = sigma_I (1 - s).
    """
    return interpolant_config.sigma_I * (1.0 - s)


@torch.inference_mode()
def sample_normalized_conditional_em(
    model: ConditionalDriftMLP,
    normalized_y: Array,
    interpolant_config: InterpolantConfig,
    sampling_config: SamplingConfig,
    device: torch.device,
) -> Array:
    """
    Generate normalized terminal samples using batched Euler--Maruyama.

    Returns
    -------
    samples:
        Array with shape (num_samples, target_dim).
    """
    sampling_config.validate()

    normalized_y = _as_vector(
        normalized_y,
        "normalized_y",
        model.condition_dim,
    )

    target_dim = model.target_dim
    num_samples = sampling_config.num_samples
    num_steps = sampling_config.num_steps
    batch_size = min(
        sampling_config.sample_batch_size,
        num_samples,
    )

    time_grid = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        dtype=torch.float32,
        device=device,
    )

    condition = torch.as_tensor(
        normalized_y,
        dtype=torch.float32,
        device=device,
    )

    output = np.empty(
        (num_samples, target_dim),
        dtype=np.float32,
    )

    model.eval()

    start = 0
    while start < num_samples:
        stop = min(start + batch_size, num_samples)
        current_batch_size = stop - start

        repeated_condition = condition.unsqueeze(0).expand(
            current_batch_size,
            -1,
        )

        # Independent Gaussian source for every ensemble member.
        G = torch.randn(
            current_batch_size,
            target_dim,
            dtype=torch.float32,
            device=device,
        )

        for step in range(num_steps):
            s_n = time_grid[step]
            delta_s = time_grid[step + 1] - s_n

            s_batch = torch.full(
                (current_batch_size, 1),
                float(s_n),
                dtype=torch.float32,
                device=device,
            )

            drift = model(
                s_batch,
                G,
                repeated_condition,
            )

            # Independent Euler--Maruyama Gaussian increment for each
            # ensemble member and target component.
            xi = torch.randn_like(G)

            rho_n = diffusion_coefficient(
                s_n,
                interpolant_config,
            )

            G = (
                G
                + drift * delta_s
                + rho_n * torch.sqrt(delta_s) * xi
            )

            if not torch.isfinite(G).all():
                raise FloatingPointError(
                    "Euler--Maruyama produced a non-finite state at "
                    f"step {step + 1} of {num_steps}."
                )

        output[start:stop] = G.cpu().numpy()
        start = stop

    return output


def make_uniform_time_grid(num_steps: int) -> Array:
    """Return the Euler--Maruyama grid 0=s_0<...<s_N=1."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")

    return np.linspace(
        0.0,
        1.0,
        num_steps + 1,
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_conditional_samples(
    output_path: str | Path,
    *,
    input_y: Array,
    normalized_input_y: Array,
    samples: Array,
    normalized_samples: Array,
    time_grid: Array,
    checkpoint_path: str | Path,
    normalization_stats: NormalizationStats,
    interpolant_config: InterpolantConfig,
    sampling_config: SamplingConfig,
    model_config: Mapping[str, Any],
) -> Path:
    """Save the condition, terminal samples, and sampling metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(Path(checkpoint_path)),
        "num_samples": int(sampling_config.num_samples),
        "num_steps": int(sampling_config.num_steps),
        "sample_batch_size": int(sampling_config.sample_batch_size),
        "seed": int(sampling_config.seed),
        "device_request": sampling_config.device,
        "diffusion_schedule": "rho(s) = sigma_I * (1 - s)",
        "sigma_I": float(interpolant_config.sigma_I),
        "condition_dim": int(model_config["condition_dim"]),
        "target_dim": int(model_config["target_dim"]),
        "hidden_dims": [
            int(width)
            for width in model_config["hidden_dims"]
        ],
        "activation": model_config.get("activation", "SiLU"),
        "time_embedding": model_config.get("time_embedding"),
    }

    np.savez_compressed(
        output_path,
        input_y=np.asarray(input_y, dtype=np.float32),
        normalized_input_y=np.asarray(
            normalized_input_y,
            dtype=np.float32,
        ),
        samples=np.asarray(samples, dtype=np.float32),
        normalized_samples=np.asarray(
            normalized_samples,
            dtype=np.float32,
        ),
        artificial_time_grid=np.asarray(
            time_grid,
            dtype=np.float64,
        ),
        mu_Y=normalization_stats.mu_Y,
        sigma_Y=normalization_stats.sigma_Y,
        mu_U=normalization_stats.mu_U,
        sigma_U=normalization_stats.sigma_U,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True),
        ),
    )

    return output_path


# ---------------------------------------------------------------------------
# Complete sampling pipeline
# ---------------------------------------------------------------------------

def sample_and_save(
    checkpoint_path: str | Path,
    input_y: Array,
    output_path: str | Path,
    *,
    sampling_config: SamplingConfig = SamplingConfig(),
    normalization_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load the trained drift, generate a conditional ensemble, and save it.

    The saved `samples` array is in physical target coordinates. The saved
    `normalized_samples` array contains the terminal Euler--Maruyama states.
    """
    sampling_config.validate()
    set_seed(sampling_config.seed)

    (
        model,
        interpolant_config,
        normalization_stats,
        checkpoint,
        device,
    ) = load_trained_drift(
        checkpoint_path,
        device_name=sampling_config.device,
        normalization_file=normalization_file,
    )

    input_y = _as_vector(
        input_y,
        "input_y",
        model.condition_dim,
    )

    normalized_input_y = normalize_condition(
        input_y,
        normalization_stats,
    )

    normalized_samples = sample_normalized_conditional_em(
        model,
        normalized_input_y,
        interpolant_config,
        sampling_config,
        device,
    )

    samples = denormalize_targets(
        normalized_samples,
        normalization_stats,
    )

    time_grid = make_uniform_time_grid(
        sampling_config.num_steps
    )

    saved_path = save_conditional_samples(
        output_path,
        input_y=input_y,
        normalized_input_y=normalized_input_y,
        samples=samples,
        normalized_samples=normalized_samples,
        time_grid=time_grid,
        checkpoint_path=checkpoint_path,
        normalization_stats=normalization_stats,
        interpolant_config=interpolant_config,
        sampling_config=sampling_config,
        model_config=checkpoint["model_config"],
    )

    return {
        "input_y": input_y,
        "normalized_input_y": normalized_input_y,
        "samples": samples,
        "normalized_samples": normalized_samples,
        "time_grid": time_grid,
        "output_path": saved_path,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Command-line input handling
# ---------------------------------------------------------------------------

def parse_comma_separated_vector(value: str) -> Array:
    """Parse a comma-separated numerical vector."""
    try:
        vector = np.asarray(
            [
                float(part.strip())
                for part in value.split(",")
                if part.strip()
            ],
            dtype=np.float32,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--y must be a comma-separated list of numbers."
        ) from error

    if vector.size < 1:
        raise argparse.ArgumentTypeError(
            "--y must contain at least one number."
        )

    return vector


def load_condition_file(
    path: str | Path,
    *,
    key: str = "y",
) -> Array:
    """
    Load one conditioning vector from NPY or NPZ.

    For NPZ files, `key` selects the array. A stored row with shape (1, q)
    is accepted and flattened to shape (q,).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Condition file not found: {path}"
        )

    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(
                    f"The NPZ condition file does not contain key '{key}'."
                )
            value = np.asarray(data[key])
    else:
        raise ValueError(
            "Condition files must use the .npy or .npz extension."
        )

    return _as_vector(value, "condition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conditional samples with Euler--Maruyama using a "
            "checkpoint from train_drift_with_visualization.py."
        )
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to conditional_drift_checkpoint.pt.",
    )

    condition_group = parser.add_mutually_exclusive_group(
        required=True
    )

    condition_group.add_argument(
        "--y",
        type=parse_comma_separated_vector,
        help="Physical conditioning vector as comma-separated values.",
    )

    condition_group.add_argument(
        "--condition-file",
        type=Path,
        help="NPY or NPZ file containing one physical conditioning vector.",
    )

    parser.add_argument(
        "--condition-key",
        type=str,
        default="y",
        help="Array key used when --condition-file is an NPZ file.",
    )

    parser.add_argument(
        "--normalization-file",
        type=Path,
        default=None,
        help=(
            "Optional NPZ file containing mu_Y, sigma_Y, mu_U, sigma_U. "
            "Needed only when the checkpoint lacks normalization_stats."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conditional_samples.npz"),
        help="Output NPZ path.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.y is not None:
        input_y = args.y
    else:
        input_y = load_condition_file(
            args.condition_file,
            key=args.condition_key,
        )

    sampling_config = SamplingConfig(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        sample_batch_size=args.sample_batch_size,
        seed=args.seed,
        device=args.device,
    )

    result = sample_and_save(
        args.checkpoint,
        input_y,
        args.output,
        sampling_config=sampling_config,
        normalization_file=args.normalization_file,
    )

    summary = {
        "output_path": str(result["output_path"]),
        "num_samples": int(result["samples"].shape[0]),
        "target_dim": int(result["samples"].shape[1]),
        "condition_dim": int(result["input_y"].size),
        "device": str(result["device"]),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
