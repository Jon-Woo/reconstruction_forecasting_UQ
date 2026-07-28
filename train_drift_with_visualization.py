#!/usr/bin/env python3
"""
Train a Gaussian-source conditional stochastic-interpolant drift MLP.

Assumptions
-----------
The training, validation, and test pairs have already been split and normalized:

    Y_train: shape (N_train, q)
    U_train: shape (N_train, d)

    Y_val:   shape (N_val, q)
    U_val:   shape (N_val, d)

    Y_test:  shape (N_test, q)
    U_test:  shape (N_test, d)

The drift network represents

    b_theta(s, x, y): [0, 1] x R^d x R^q -> R^d

and receives the direct concatenation [s, x, y]. No artificial-time embedding
or Fourier features are used.

Training uses the simulation-free stochastic-interpolant loss. For each
normalized target U and independently sampled

    S ~ Uniform(0, 1),
    Z, Xi ~ N(0, I_d),

the fixed-time Brownian marginal is represented as

    B_S = sqrt(S) Xi.

The default interpolant schedule is

    alpha(s) = 1 - s,
    beta(s)  = s,
    rho(s)   = sigma_I (1 - s),

with derivatives

    alpha_dot(s) = -1,
    beta_dot(s)  = 1,
    rho_dot(s)   = -sigma_I.

The simulation-free regression variables are

    I_S = alpha(S) Z + beta(S) U + rho(S) sqrt(S) Xi,

    R_S = alpha_dot(S) Z
          + beta_dot(S) U
          + rho_dot(S) sqrt(S) Xi.

The MLP minimizes

    E ||b_theta(S, I_S, Y) - R_S||_2^2

using minibatches and AdamW. Training stops by validation-loss early stopping.
The validation-selected checkpoint is restored before a single final test-loss
evaluation. After training, the implementation saves a logarithmic
training/validation loss plot and a separate MLP architecture diagram.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterpolantConfig:
    """Default Gaussian-source stochastic-interpolant schedule."""

    sigma_I: float = 0.25

    def validate(self) -> None:
        if self.sigma_I < 0.0:
            raise ValueError("sigma_I must be nonnegative.")

    def coefficients(
        self,
        s: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Return alpha, beta, rho and their derivatives at artificial time s.
        """
        alpha = 1.0 - s
        beta = s
        rho = self.sigma_I * (1.0 - s)

        alpha_dot = -torch.ones_like(s)
        beta_dot = torch.ones_like(s)
        rho_dot = -self.sigma_I * torch.ones_like(s)

        return alpha, beta, rho, alpha_dot, beta_dot, rho_dot


@dataclass(frozen=True)
class TrainConfig:
    """Default MLP architecture, optimization, and stopping parameters."""

    hidden_dims: Tuple[int, ...] = (256, 256, 256, 256)
    batch_size: int = 256
    validation_batch_size: int = 512

    max_epochs: int = 2000
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-6

    patience: int = 150
    min_delta: float = 1.0e-6

    validation_mc_repeats: int = 4
    test_mc_repeats: int = 16

    seed: int = 17
    device: str = "auto"
    print_every: int = 25

    def validate(self) -> None:
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer width.")
        if any(width < 1 for width in self.hidden_dims):
            raise ValueError("Every hidden-layer width must be positive.")
        if self.batch_size < 1 or self.validation_batch_size < 1:
            raise ValueError("Batch sizes must be positive.")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be nonnegative.")
        if self.validation_mc_repeats < 1 or self.test_mc_repeats < 1:
            raise ValueError("Monte Carlo repeat counts must be positive.")
        if self.print_every < 1:
            raise ValueError("print_every must be positive.")


# ---------------------------------------------------------------------------
# Reproducibility and device selection
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Resolve 'auto' to CUDA, MPS, or CPU, in that order."""
    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data validation and loading
# ---------------------------------------------------------------------------

def _as_float_matrix(array: Array, name: str) -> Array:
    """
    Convert an array to a finite float32 matrix.

    A one-dimensional array of shape (N,) is interpreted as a scalar-valued
    variable and reshaped to (N, 1).
    """
    result = np.asarray(array)

    if result.ndim == 1:
        result = result[:, None]

    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (N, features).")

    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must be nonempty.")

    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numerical.")

    result = np.asarray(result, dtype=np.float32)

    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return result


def validate_normalized_splits(
    Y_train: Array,
    U_train: Array,
    Y_val: Array,
    U_val: Array,
    Y_test: Array,
    U_test: Array,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Validate normalized train/validation/test arrays and their dimensions.
    """
    Y_train = _as_float_matrix(Y_train, "Y_train")
    U_train = _as_float_matrix(U_train, "U_train")
    Y_val = _as_float_matrix(Y_val, "Y_val")
    U_val = _as_float_matrix(U_val, "U_val")
    Y_test = _as_float_matrix(Y_test, "Y_test")
    U_test = _as_float_matrix(U_test, "U_test")

    if Y_train.shape[0] != U_train.shape[0]:
        raise ValueError("Y_train and U_train must have the same row count.")
    if Y_val.shape[0] != U_val.shape[0]:
        raise ValueError("Y_val and U_val must have the same row count.")
    if Y_test.shape[0] != U_test.shape[0]:
        raise ValueError("Y_test and U_test must have the same row count.")

    condition_dim = Y_train.shape[1]
    target_dim = U_train.shape[1]

    if Y_val.shape[1] != condition_dim or Y_test.shape[1] != condition_dim:
        raise ValueError("All Y splits must have the same feature dimension.")

    if U_val.shape[1] != target_dim or U_test.shape[1] != target_dim:
        raise ValueError("All U splits must have the same feature dimension.")

    return Y_train, U_train, Y_val, U_val, Y_test, U_test


def make_loader(
    Y: Array,
    U: Array,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a PyTorch loader for a normalized paired split."""
    dataset = TensorDataset(
        torch.from_numpy(Y),
        torch.from_numpy(U),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Drift MLP
# ---------------------------------------------------------------------------

class ConditionalDriftMLP(nn.Module):
    """
    MLP approximation of b_theta(s, x, y).

    The direct input is the concatenation [s, x, y]. Artificial time is passed
    as one scalar. Hidden layers use SiLU. The output layer is linear.
    """

    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256, 256),
    ) -> None:
        super().__init__()

        if target_dim < 1:
            raise ValueError("target_dim must be positive.")
        if condition_dim < 1:
            raise ValueError("condition_dim must be positive.")
        if len(hidden_dims) < 1 or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")

        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)

        input_dim = 1 + self.target_dim + self.condition_dim

        layers = []
        previous_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.SiLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, self.target_dim))
        self.network = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Apply Xavier--Glorot uniform initialization to every weight matrix and
        initialize every bias vector to zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        s: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate b_theta(s, x, y).

        Expected shapes:
            s: (batch, 1) or (batch,)
            x: (batch, target_dim)
            y: (batch, condition_dim)
        """
        if s.ndim == 1:
            s = s[:, None]

        if s.ndim != 2 or s.shape[1] != 1:
            raise ValueError("s must have shape (batch, 1) or (batch,).")

        if x.ndim != 2 or x.shape[1] != self.target_dim:
            raise ValueError(
                f"x must have shape (batch, {self.target_dim})."
            )

        if y.ndim != 2 or y.shape[1] != self.condition_dim:
            raise ValueError(
                f"y must have shape (batch, {self.condition_dim})."
            )

        if not (s.shape[0] == x.shape[0] == y.shape[0]):
            raise ValueError("s, x, and y must have the same batch size.")

        inputs = torch.cat([s, x, y], dim=-1)
        return self.network(inputs)


# ---------------------------------------------------------------------------
# Simulation-free stochastic-interpolant construction
# ---------------------------------------------------------------------------

def construct_simulation_free_batch(
    Y: torch.Tensor,
    U: torch.Tensor,
    interpolant_config: InterpolantConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct one simulation-free sample of (S, I_S, R_S) per data pair.

    No Brownian path and no SDE trajectory are simulated.
    """
    if Y.ndim != 2 or U.ndim != 2:
        raise ValueError("Y and U must be batched matrices.")

    if Y.shape[0] != U.shape[0]:
        raise ValueError("Y and U must have the same batch size.")

    batch_size = U.shape[0]

    S = torch.rand(
        (batch_size, 1),
        device=U.device,
        dtype=U.dtype,
    )

    Z = torch.randn_like(U)
    Xi = torch.randn_like(U)
    B_S = torch.sqrt(S) * Xi

    (
        alpha,
        beta,
        rho,
        alpha_dot,
        beta_dot,
        rho_dot,
    ) = interpolant_config.coefficients(S)

    I_S = alpha * Z + beta * U + rho * B_S

    R_S = (
        alpha_dot * Z
        + beta_dot * U
        + rho_dot * B_S
    )

    return S, I_S, R_S


def vector_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Return the minibatch mean of squared Euclidean errors.

    This implements
        (1 / M) sum_i ||prediction_i - target_i||_2^2.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes.")

    return (prediction - target).square().sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Validation and test evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_drift_loss(
    model: ConditionalDriftMLP,
    loader: DataLoader,
    interpolant_config: InterpolantConfig,
    device: torch.device,
    *,
    mc_repeats: int,
) -> float:
    """
    Estimate the simulation-free drift loss with fresh random variables.

    Each repeat redraws S, Z, and Xi.
    """
    if mc_repeats < 1:
        raise ValueError("mc_repeats must be positive.")

    model.eval()

    total_squared_error = 0.0
    total_sample_count = 0

    for Y, U in loader:
        Y = Y.to(device)
        U = U.to(device)

        for _ in range(mc_repeats):
            S, I_S, R_S = construct_simulation_free_batch(
                Y,
                U,
                interpolant_config,
            )

            prediction = model(S, I_S, Y)
            batch_squared_error = (
                prediction - R_S
            ).square().sum()

            total_squared_error += float(batch_squared_error.cpu())
            total_sample_count += U.shape[0]

    if total_sample_count == 0:
        raise RuntimeError("Cannot estimate loss on an empty data loader.")

    return total_squared_error / total_sample_count


# ---------------------------------------------------------------------------
# Training visualizations
# ---------------------------------------------------------------------------

def format_architecture_description(
    model_config: Mapping[str, Any],
) -> str:
    """Return a compact human-readable description of the drift MLP."""
    condition_dim = int(model_config["condition_dim"])
    target_dim = int(model_config["target_dim"])
    hidden_dims = [int(width) for width in model_config["hidden_dims"]]
    input_dim = 1 + target_dim + condition_dim

    hidden_text = " -> ".join(str(width) for width in hidden_dims)

    return (
        f"Input {input_dim} = [s: 1, x: {target_dim}, y: {condition_dim}]"
        f" | Hidden {hidden_text} with SiLU"
        f" | Output {target_dim} linear"
        f" | Xavier-Glorot uniform weights, zero biases"
    )


def plot_training_and_validation_losses(
    history: Mapping[str, Sequence[float]],
    model_config: Mapping[str, Any],
    best_epoch: int,
    output_path: str | Path,
) -> Path:
    """
    Save training and validation losses against epoch on a logarithmic scale.

    The network architecture is printed directly beneath the figure title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.asarray(history["epoch"], dtype=np.int64)
    train_losses = np.asarray(history["train_loss"], dtype=np.float64)
    validation_losses = np.asarray(
        history["validation_loss"],
        dtype=np.float64,
    )

    if not (
        len(epochs)
        == len(train_losses)
        == len(validation_losses)
        and len(epochs) > 0
    ):
        raise ValueError("Training history arrays must be nonempty and aligned.")

    # A log axis requires strictly positive values. Exact zeros, if produced by
    # finite precision, are displayed at the smallest positive float.
    positive_floor = np.finfo(np.float64).tiny
    train_losses = np.maximum(train_losses, positive_floor)
    validation_losses = np.maximum(validation_losses, positive_floor)

    architecture = format_architecture_description(model_config)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.semilogy(
        epochs,
        train_losses,
        label="Training drift loss",
    )
    axis.semilogy(
        epochs,
        validation_losses,
        label="Validation drift loss",
    )
    axis.axvline(
        best_epoch,
        linestyle="--",
        label=f"Selected epoch: {best_epoch}",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared drift loss")
    axis.set_title("Conditional drift training history")
    axis.grid(True, which="both", linestyle=":")
    axis.legend()

    figure.text(
        0.5,
        0.93,
        architecture,
        ha="center",
        va="top",
        wrap=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_network_architecture(
    model_config: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save a schematic of the direct-input conditional drift MLP architecture.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    condition_dim = int(model_config["condition_dim"])
    target_dim = int(model_config["target_dim"])
    hidden_dims = [int(width) for width in model_config["hidden_dims"]]
    input_dim = 1 + target_dim + condition_dim

    layer_labels = [
        (
            "Input\n"
            f"{input_dim}\n"
            f"[s: 1, x: {target_dim}, y: {condition_dim}]"
        )
    ]
    layer_labels.extend(
        f"Hidden {index}\n{width}\nSiLU"
        for index, width in enumerate(hidden_dims, start=1)
    )
    layer_labels.append(f"Output\n{target_dim}\nLinear")

    x_positions = np.arange(len(layer_labels), dtype=np.float64)

    figure_width = max(10.0, 1.9 * len(layer_labels))
    figure, axis = plt.subplots(figsize=(figure_width, 4.5))

    for index, (x_position, label) in enumerate(
        zip(x_positions, layer_labels)
    ):
        axis.text(
            x_position,
            0.0,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.6"},
        )

        if index < len(layer_labels) - 1:
            axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.35, 0.0),
                xytext=(x_position + 0.35, 0.0),
                arrowprops={"arrowstyle": "->"},
            )

    axis.set_xlim(-0.8, len(layer_labels) - 0.2)
    axis.set_ylim(-1.0, 1.0)
    axis.axis("off")
    axis.set_title(
        "Conditional drift MLP architecture\n"
        "Direct scalar time input; Xavier-Glorot uniform weights; "
        "zero biases"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def create_training_visualizations(
    history: Mapping[str, Sequence[float]],
    model_config: Mapping[str, Any],
    best_epoch: int,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Create and save all post-training visualization files."""
    output_dir = Path(output_dir)

    loss_plot = plot_training_and_validation_losses(
        history,
        model_config,
        best_epoch,
        output_dir / "training_validation_loss_log.png",
    )

    architecture_plot = plot_network_architecture(
        model_config,
        output_dir / "network_architecture.png",
    )

    return {
        "training_validation_loss_log": str(loss_plot),
        "network_architecture": str(architecture_plot),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_conditional_drift(
    Y_train: Array,
    U_train: Array,
    Y_val: Array,
    U_val: Array,
    Y_test: Array,
    U_test: Array,
    output_dir: str | Path,
    *,
    train_config: TrainConfig = TrainConfig(),
    interpolant_config: InterpolantConfig = InterpolantConfig(),
    normalization_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train, validation-select, test-evaluate, and save the conditional drift.

    The test set is used only after the validation-selected checkpoint has been
    restored. No parameter update uses the test set.
    """
    train_config.validate()
    interpolant_config.validate()

    (
        Y_train,
        U_train,
        Y_val,
        U_val,
        Y_test,
        U_test,
    ) = validate_normalized_splits(
        Y_train,
        U_train,
        Y_val,
        U_val,
        Y_test,
        U_test,
    )

    set_seed(train_config.seed)
    device = select_device(train_config.device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        Y_train,
        U_train,
        train_config.batch_size,
        shuffle=True,
        seed=train_config.seed,
    )

    val_loader = make_loader(
        Y_val,
        U_val,
        train_config.validation_batch_size,
        shuffle=False,
        seed=train_config.seed,
    )

    test_loader = make_loader(
        Y_test,
        U_test,
        train_config.validation_batch_size,
        shuffle=False,
        seed=train_config.seed,
    )

    model = ConditionalDriftMLP(
        target_dim=U_train.shape[1],
        condition_dim=Y_train.shape[1],
        hidden_dims=train_config.hidden_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, train_config.max_epochs + 1):
        model.train()

        epoch_squared_error = 0.0
        epoch_sample_count = 0

        for Y_batch, U_batch in train_loader:
            Y_batch = Y_batch.to(device)
            U_batch = U_batch.to(device)

            S, I_S, R_S = construct_simulation_free_batch(
                Y_batch,
                U_batch,
                interpolant_config,
            )

            prediction = model(S, I_S, Y_batch)
            loss = vector_mse(prediction, R_S)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Encountered a non-finite loss at epoch {epoch}."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_squared_error += float(
                    (prediction - R_S).square().sum().cpu()
                )
                epoch_sample_count += U_batch.shape[0]

        train_loss = epoch_squared_error / epoch_sample_count

        validation_loss = estimate_drift_loss(
            model,
            val_loader,
            interpolant_config,
            device,
            mc_repeats=train_config.validation_mc_repeats,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        improved = (
            validation_loss
            < best_validation_loss - train_config.min_delta
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % train_config.print_every == 0:
            print(
                f"epoch={epoch:5d} "
                f"train_loss={train_loss:.6e} "
                f"validation_loss={validation_loss:.6e}"
            )

        if epochs_without_improvement >= train_config.patience:
            print(
                "Early stopping: "
                f"no sufficient validation improvement for "
                f"{train_config.patience} consecutive epochs."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    test_loss = estimate_drift_loss(
        model,
        test_loader,
        interpolant_config,
        device,
        mc_repeats=train_config.test_mc_repeats,
    )

    model_config = {
        "target_dim": int(U_train.shape[1]),
        "condition_dim": int(Y_train.shape[1]),
        "hidden_dims": list(train_config.hidden_dims),
        "activation": "SiLU",
        "output_activation": None,
        "weight_initialization": "Xavier-Glorot uniform",
        "bias_initialization": "zeros",
        "direct_time_input": True,
        "time_embedding": None,
    }

    visualization_paths = create_training_visualizations(
        history,
        model_config,
        best_epoch,
        output_dir,
    )

    checkpoint: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "train_config": asdict(train_config),
        "interpolant_config": asdict(interpolant_config),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
        "visualization_paths": visualization_paths,
    }

    if normalization_stats is not None:
        checkpoint["normalization_stats"] = dict(normalization_stats)

    checkpoint_path = output_dir / "conditional_drift_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "n_train": int(Y_train.shape[0]),
        "n_validation": int(Y_val.shape[0]),
        "n_test": int(Y_test.shape[0]),
        "condition_dim": int(Y_train.shape[1]),
        "target_dim": int(U_train.shape[1]),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "test_loss": float(test_loss),
        "checkpoint": str(checkpoint_path),
        "training_validation_loss_log": visualization_paths[
            "training_validation_loss_log"
        ],
        "network_architecture_plot": visualization_paths[
            "network_architecture"
        ],
    }

    with (output_dir / "training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

    return {
        "model": model,
        "device": device,
        "history": history,
        "checkpoint": checkpoint,
        "summary": summary,
        "visualization_paths": visualization_paths,
    }


# ---------------------------------------------------------------------------
# NPZ interface
# ---------------------------------------------------------------------------

_REQUIRED_NPZ_KEYS = (
    "Y_train",
    "U_train",
    "Y_val",
    "U_val",
    "Y_test",
    "U_test",
)

_OPTIONAL_NORMALIZATION_KEYS = (
    "mu_Y",
    "sigma_Y",
    "mu_U",
    "sigma_U",
)


def load_normalized_npz(
    path: str | Path,
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Optional[Dict[str, Array]],
]:
    """
    Load normalized train/validation/test arrays from an NPZ file.

    Required keys:
        Y_train, U_train, Y_val, U_val, Y_test, U_test

    Optional normalization keys:
        mu_Y, sigma_Y, mu_U, sigma_U
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_NPZ_KEYS if key not in data]
        if missing:
            raise KeyError(
                "The NPZ file is missing required arrays: "
                + ", ".join(missing)
            )

        arrays = tuple(np.asarray(data[key]) for key in _REQUIRED_NPZ_KEYS)

        normalization_stats: Optional[Dict[str, Array]]
        if all(key in data for key in _OPTIONAL_NORMALIZATION_KEYS):
            normalization_stats = {
                key: np.asarray(data[key])
                for key in _OPTIONAL_NORMALIZATION_KEYS
            }
        else:
            normalization_stats = None

    return (*arrays, normalization_stats)


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated hidden-width specification."""
    try:
        widths = tuple(
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "hidden dimensions must be comma-separated integers"
        ) from error

    if len(widths) < 1 or any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError(
            "hidden dimensions must be positive integers"
        )

    return widths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simulation-free Gaussian-source conditional drift MLP "
            "from normalized train/validation/test arrays."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help=(
            "NPZ file containing Y_train, U_train, Y_val, U_val, "
            "Y_test, and U_test."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modularized_implementation/conditional_drift_output"),
    )

    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(256, 256, 256, 256),
        help="Comma-separated hidden-layer widths.",
    )

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument("--validation-mc-repeats", type=int, default=4)
    parser.add_argument("--test-mc-repeats", type=int, default=16)
    parser.add_argument("--sigma-I", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print-every", type=int, default=25)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    (
        Y_train,
        U_train,
        Y_val,
        U_val,
        Y_test,
        U_test,
        normalization_stats,
    ) = load_normalized_npz(args.dataset)

    train_config = TrainConfig(
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        validation_mc_repeats=args.validation_mc_repeats,
        test_mc_repeats=args.test_mc_repeats,
        seed=args.seed,
        device=args.device,
        print_every=args.print_every,
    )

    interpolant_config = InterpolantConfig(
        sigma_I=args.sigma_I,
    )

    result = train_conditional_drift(
        Y_train,
        U_train,
        Y_val,
        U_val,
        Y_test,
        U_test,
        args.output_dir,
        train_config=train_config,
        interpolant_config=interpolant_config,
        normalization_stats=normalization_stats,
    )

    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
