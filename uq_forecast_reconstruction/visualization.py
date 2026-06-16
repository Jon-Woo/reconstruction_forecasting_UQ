"""Visualization utilities for conditional stochastic interpolant examples.

    y_true:  (N, y_dim)
    samples: (N, ensemble_size, y_dim)

For a single conditioning point, pass

    y_true:  (y_dim,)
    samples: (ensemble_size, y_dim)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# Safe default for scripts running on clusters/servers without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from delay_si import ensemble_summary


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Convert a vector to shape (N, 1), leaving matrices unchanged."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def plot_training_loss(losses: Sequence[float], out_path: str | Path, title: str = "Training loss") -> Path:
    """Save a line plot of the stochastic-interpolant training loss."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(losses), lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Interpolant regression loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_predictive_timeseries(
    y_true: np.ndarray,
    samples: np.ndarray,
    out_path: str | Path,
    title: str,
    component_names: Sequence[str] | None = None,
) -> Path:
    """Plot true target vs ensemble mean and central 90% interval.

    Parameters
    ----------
    y_true:
        True targets with shape (N, y_dim).
    samples:
        Ensemble samples with shape (N, S, y_dim).
    out_path:
        File path for the saved PNG/PDF/etc.
    title:
        Figure title.
    component_names:
        Optional y-axis labels for each target component.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = ensure_2d(y_true)
    samples = np.asarray(samples)
    mean, _, q05, q95 = ensemble_summary(samples)
    n_points, y_dim = y_true.shape
    xs = np.arange(n_points)
    component_names = list(component_names or [f"component {j}" for j in range(y_dim)])

    fig, axes = plt.subplots(y_dim, 1, figsize=(10, 2.8 * y_dim), sharex=True)
    if y_dim == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(xs, y_true[:, j], label="true", lw=1.8)
        ax.plot(xs, mean[:, j], label="ensemble mean", lw=1.6)
        ax.fill_between(xs, q05[:, j], q95[:, j], alpha=0.25, label="90% interval")
        ax.set_ylabel(component_names[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Test sample index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_single_case_histograms(
    y_true: np.ndarray,
    samples: np.ndarray,
    out_path: str | Path,
    title: str,
    component_names: Sequence[str] | None = None,
) -> Path:
    """Plot histograms of the predictive ensemble for one conditioning point."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).reshape(-1)
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    y_dim = samples.shape[1]
    component_names = list(component_names or [f"component {j}" for j in range(y_dim)])
    mean = samples.mean(axis=0)

    fig, axes = plt.subplots(1, y_dim, figsize=(4.5 * y_dim, 4), squeeze=False)
    axes = axes[0]
    for j, ax in enumerate(axes):
        ax.hist(samples[:, j], bins=30, density=True, alpha=0.75)
        ax.axvline(y_true[j], lw=2.0, linestyle="--", label="true")
        ax.axvline(mean[j], lw=2.0, linestyle=":", label="ensemble mean")
        ax.set_title(component_names[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_phase_portrait(
    y_true: np.ndarray,
    samples: np.ndarray,
    out_path: str | Path,
    title: str,
    components: tuple[int, int] = (0, 1),
    axis_labels: tuple[str, str] = ("x", "y"),
) -> Path:
    """Plot true state and reconstructed ensemble mean in a 2D projection."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = ensure_2d(y_true)
    samples = np.asarray(samples)
    mean, _, _, _ = ensemble_summary(samples)
    i, j = components
    if y_true.shape[1] <= max(i, j):
        raise ValueError("Requested phase-portrait components exceed target dimension.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(y_true[:, i], y_true[:, j], label="true", lw=1.5)
    ax.plot(mean[:, i], mean[:, j], label="ensemble mean", lw=1.5)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def maybe_show(show: bool) -> None:
    """Display figures when using an interactive backend."""
    if show:
        plt.show()
