#!/usr/bin/env python3
"""
Local conditional-coverage experiment for partial-observation forecasting
in the additive-noise stochastic Lorenz system.

The observed variable is the first coordinate z_t = X_t^(1).  The default
delay embedding is

    Y_t = [z_t, z_{t-0.1}, z_{t-0.2}] in R^3,

so m=3 and tau=0.1 in physical time.  The scalar forecast target is

    U_t = z_{t+h},

where h is configurable.

The script:
  1. generates independent training/validation Lorenz trajectories;
  2. constructs non-overlapping delay-to-future pairs;
  3. trains the generic Gaussian-source conditional stochastic interpolant
     from gaussian_source_si.py;
  4. selects one held-out delay embedding after a long burn-in;
  5. branches many independent physical futures from its latent state;
  6. generates many SI samples conditioned on the fixed delay vector;
  7. compares means, central 90% prediction intervals, widths, and coverage;
  8. creates the requested visualizations.

Important modeling assumption
-----------------------------
This benchmark assumes that the chosen delay vector determines the current
latent state on the relevant set. Under that assumption and the Markov property,
Law(U | Y=y*) = Law(U | X=x*), so physical branching from x* is a valid
reference conditional ensemble.

Dependencies: numpy, matplotlib, torch, gaussian_source_si.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gaussian_source_si import (
    InterpolantConfig,
    TrainConfig,
    sample_conditional_sde,
    train_conditional_drift,
)


Array = np.ndarray


@dataclass(frozen=True)
class LorenzConfig:
    sigma_lorenz: float = 10.0
    rho_lorenz: float = 28.0
    beta_lorenz: float = 8.0 / 3.0
    noise_amplitude: float = 0.05
    initial_condition: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    dt: float = 0.002
    observation_dt: float = 0.01

    @property
    def observation_stride(self) -> int:
        ratio = self.observation_dt / self.dt
        rounded = int(round(ratio))
        if rounded < 1 or not np.isclose(ratio, rounded, rtol=0.0, atol=1e-12):
            raise ValueError("observation_dt must be an integer multiple of dt.")
        return rounded


def lorenz_drift(x: Array, cfg: LorenzConfig) -> Array:
    """Evaluate the Lorenz drift for x with final dimension 3."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError("Lorenz states must have final dimension 3.")

    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    return np.stack(
        [
            cfg.sigma_lorenz * (x2 - x1),
            x1 * (cfg.rho_lorenz - x3) - x2,
            x1 * x2 - cfg.beta_lorenz * x3,
        ],
        axis=-1,
    )


def simulate_lorenz_path(
    cfg: LorenzConfig,
    total_time: float,
    seed: int,
    initial_state: Array | None = None,
) -> Tuple[Array, Array]:
    """
    Simulate one path by Euler--Maruyama and return saved times/states.

    Returned states have shape (n_saved, 3), with adjacent saved states
    separated by cfg.observation_dt.
    """
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")

    n_steps_float = total_time / cfg.dt
    n_steps = int(round(n_steps_float))
    if not np.isclose(n_steps_float, n_steps, rtol=0.0, atol=1e-10):
        raise ValueError("total_time must be an integer multiple of dt.")

    stride = cfg.observation_stride
    if n_steps % stride != 0:
        raise ValueError("total_time must contain an integer number of saved steps.")

    rng = np.random.default_rng(seed)
    state = np.asarray(
        cfg.initial_condition if initial_state is None else initial_state,
        dtype=np.float64,
    ).copy()
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("initial_state must be a finite vector of shape (3,).")

    n_saved = n_steps // stride + 1
    states = np.empty((n_saved, 3), dtype=np.float64)
    times = np.arange(n_saved, dtype=np.float64) * cfg.observation_dt
    states[0] = state

    noise_scale = cfg.noise_amplitude * math.sqrt(cfg.dt)
    saved_index = 1

    for step in range(1, n_steps + 1):
        state = (
            state
            + lorenz_drift(state, cfg) * cfg.dt
            + noise_scale * rng.standard_normal(3)
        )
        if not np.isfinite(state).all():
            raise FloatingPointError(
                "Non-finite Lorenz state encountered. Reduce dt or use a stabilized method."
            )
        if step % stride == 0:
            states[saved_index] = state
            saved_index += 1

    return times, states


def simulate_lorenz_ensemble_from_state(
    cfg: LorenzConfig,
    initial_state: Array,
    forecast_time: float,
    n_ensemble: int,
    seed: int,
    save_paths: bool = True,
) -> Tuple[Array, Array]:
    """
    Vectorized independent branches from one fixed latent state.

    Returns
    -------
    times:
        Shape (n_saved,).
    paths:
        Shape (n_ensemble, n_saved, 3) when save_paths=True; otherwise only
        endpoint states are returned with shape (n_ensemble, 1, 3).
    """
    if n_ensemble < 1:
        raise ValueError("n_ensemble must be positive.")
    n_steps_float = forecast_time / cfg.dt
    n_steps = int(round(n_steps_float))
    if n_steps < 1 or not np.isclose(n_steps_float, n_steps, atol=1e-10):
        raise ValueError("forecast_time must be a positive integer multiple of dt.")

    stride = cfg.observation_stride
    if n_steps % stride != 0:
        raise ValueError("forecast_time must be an integer multiple of observation_dt.")

    initial_state = np.asarray(initial_state, dtype=np.float64)
    if initial_state.shape != (3,):
        raise ValueError("initial_state must have shape (3,).")

    rng = np.random.default_rng(seed)
    states = np.repeat(initial_state[None, :], n_ensemble, axis=0)
    n_saved = n_steps // stride + 1
    times = np.arange(n_saved, dtype=np.float64) * cfg.observation_dt

    if save_paths:
        paths = np.empty((n_ensemble, n_saved, 3), dtype=np.float64)
        paths[:, 0, :] = states
    else:
        paths = np.empty((n_ensemble, 1, 3), dtype=np.float64)

    noise_scale = cfg.noise_amplitude * math.sqrt(cfg.dt)
    saved_index = 1
    for step in range(1, n_steps + 1):
        states = (
            states
            + lorenz_drift(states, cfg) * cfg.dt
            + noise_scale * rng.standard_normal(states.shape)
        )
        if not np.isfinite(states).all():
            raise FloatingPointError(
                "Non-finite reference branch encountered. Reduce dt."
            )
        if save_paths and step % stride == 0:
            paths[:, saved_index, :] = states
            saved_index += 1

    if not save_paths:
        paths[:, 0, :] = states
    return times, paths


def physical_steps(value: float, observation_dt: float, name: str) -> int:
    ratio = value / observation_dt
    steps = int(round(ratio))
    if steps < 1 or not np.isclose(ratio, steps, atol=1e-10):
        raise ValueError(
            f"{name}={value} must be a positive integer multiple of "
            f"observation_dt={observation_dt}."
        )
    return steps


def build_forecast_pairs_from_path(
    states: Array,
    delay_steps: int,
    embedding_dim: int,
    horizon_steps: int,
    gap_steps: int,
) -> Tuple[Array, Array, Array]:
    """
    Build non-overlapping Y_t -> X_1(t+h) pairs from one saved path.

    One retained example occupies the closed index interval
    [t-(m-1)delay_steps, t+horizon_steps].
    """
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (T,3).")
    if delay_steps < 1 or embedding_dim < 1 or horizon_steps < 1:
        raise ValueError("delay_steps, embedding_dim, horizon_steps must be positive.")
    if gap_steps < 0:
        raise ValueError("gap_steps must be nonnegative.")

    observed = states[:, 0]
    history_span = (embedding_dim - 1) * delay_steps
    occupied_span = history_span + horizon_steps
    anchor_stride = occupied_span + gap_steps + 1

    first_anchor = history_span
    last_anchor_exclusive = len(states) - horizon_steps
    anchors = np.arange(
        first_anchor, last_anchor_exclusive, anchor_stride, dtype=np.int64
    )
    if anchors.size == 0:
        raise ValueError("Path is too short for the requested delay and horizon.")

    offsets = np.arange(embedding_dim, dtype=np.int64) * delay_steps
    Y = np.stack([observed[a - offsets] for a in anchors], axis=0)
    U = observed[anchors + horizon_steps, None]

    # Exact no-reuse check over the full history-to-target occupied interval.
    intervals = [
        set(range(a - history_span, a + horizon_steps + 1)) for a in anchors
    ]
    for left, right in zip(intervals[:-1], intervals[1:]):
        if left.intersection(right):
            raise AssertionError("Internal error: retained forecast windows overlap.")

    return Y, U, anchors


def generate_training_dataset(
    cfg: LorenzConfig,
    n_train_paths: int,
    n_val_paths: int,
    burn_in: float,
    data_time: float,
    delay_steps: int,
    embedding_dim: int,
    horizon_steps: int,
    gap_steps: int,
    seed: int,
) -> Dict[str, Array]:
    """Generate independent realization-split training and validation arrays."""
    if n_train_paths < 1 or n_val_paths < 1:
        raise ValueError("At least one training and one validation path are required.")

    total_time = burn_in + data_time
    burn_index = int(round(burn_in / cfg.observation_dt))
    train_y: List[Array] = []
    train_u: List[Array] = []
    val_y: List[Array] = []
    val_u: List[Array] = []

    for path_id in range(n_train_paths + n_val_paths):
        _, states = simulate_lorenz_path(
            cfg, total_time=total_time, seed=seed + path_id
        )
        post_burn = states[burn_index:]
        Y, U, _ = build_forecast_pairs_from_path(
            post_burn,
            delay_steps=delay_steps,
            embedding_dim=embedding_dim,
            horizon_steps=horizon_steps,
            gap_steps=gap_steps,
        )
        if path_id < n_train_paths:
            train_y.append(Y)
            train_u.append(U)
        else:
            val_y.append(Y)
            val_u.append(U)

    return {
        "train_Y": np.concatenate(train_y, axis=0),
        "train_U": np.concatenate(train_u, axis=0),
        "val_Y": np.concatenate(val_y, axis=0),
        "val_U": np.concatenate(val_u, axis=0),
    }


def select_fixed_embedding(
    cfg: LorenzConfig,
    burn_in: float,
    pre_anchor_time: float,
    delay_steps: int,
    embedding_dim: int,
    seed: int,
) -> Dict[str, Array | float | int]:
    """
    Simulate a held-out path and select one anchor after burn-in.

    The path includes enough post-burn time to display the trajectory before
    the selected anchor. The anchor is exactly burn_in + pre_anchor_time.
    """
    if pre_anchor_time <= (embedding_dim - 1) * delay_steps * cfg.observation_dt:
        raise ValueError("pre_anchor_time must exceed the delay-history span.")

    total_time = burn_in + pre_anchor_time
    times, states = simulate_lorenz_path(cfg, total_time=total_time, seed=seed)
    anchor = len(times) - 1
    offsets = np.arange(embedding_dim, dtype=np.int64) * delay_steps
    indices = anchor - offsets
    if indices[-1] < 0:
        raise ValueError("Held-out path is too short for the delay embedding.")

    y_star = states[indices, 0].copy()
    x_star = states[anchor].copy()

    return {
        "times": times,
        "states": states,
        "anchor": anchor,
        "delay_indices": indices,
        "y_star": y_star,
        "x_star": x_star,
        "anchor_time": float(times[anchor]),
    }


def central_interval(samples: Array, level: float = 0.90) -> Tuple[float, float]:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if samples.size < 2 or not np.isfinite(samples).all():
        raise ValueError("samples must contain at least two finite values.")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0,1).")
    alpha = 1.0 - level
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def compare_ensembles(
    reference: Array,
    estimated: Array,
    level: float = 0.90,
) -> Dict[str, float]:
    """Compute mean, interval, width, and reference coverage diagnostics."""
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimated = np.asarray(estimated, dtype=np.float64).reshape(-1)

    ref_low, ref_high = central_interval(reference, level)
    est_low, est_high = central_interval(estimated, level)
    ref_mean = float(np.mean(reference))
    est_mean = float(np.mean(estimated))
    coverage = float(np.mean((reference >= est_low) & (reference <= est_high)))
    se = float(math.sqrt(coverage * (1.0 - coverage) / len(reference)))

    return {
        "nominal_level": float(level),
        "reference_mean": ref_mean,
        "estimated_mean": est_mean,
        "absolute_mean_error": abs(est_mean - ref_mean),
        "reference_lower": ref_low,
        "reference_upper": ref_high,
        "estimated_lower": est_low,
        "estimated_upper": est_high,
        "reference_width": ref_high - ref_low,
        "estimated_width": est_high - est_low,
        "absolute_width_error": abs((est_high - est_low) - (ref_high - ref_low)),
        "reference_coverage_of_estimated_interval": coverage,
        "coverage_error": abs(coverage - level),
        "coverage_monte_carlo_standard_error": se,
    }


def empirical_cdf(samples: Array) -> Tuple[Array, Array]:
    x = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def plot_selected_embedding(
    heldout: Dict[str, Array | float | int],
    forecast_time: float,
    output_path: Path,
) -> None:
    """Show the held-out trajectory and the three selected delay observations."""
    times = np.asarray(heldout["times"])
    states = np.asarray(heldout["states"])
    delay_indices = np.asarray(heldout["delay_indices"], dtype=np.int64)
    anchor = int(heldout["anchor"])
    anchor_time = float(heldout["anchor_time"])

    left_time = max(0.0, anchor_time - 4.0)
    mask = times >= left_time

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    coordinate_names = [r"$X^{(1)}$", r"$X^{(2)}$", r"$X^{(3)}$"]

    for coordinate, ax in enumerate(axes):
        ax.plot(times[mask], states[mask, coordinate], linewidth=1.1)
        ax.axvline(anchor_time, linestyle="--", linewidth=1.3, label="anchor")
        ax.axvline(
            anchor_time + forecast_time,
            linestyle=":",
            linewidth=1.3,
            label="forecast target time",
        )
        if coordinate == 0:
            ax.scatter(
                times[delay_indices],
                states[delay_indices, 0],
                s=55,
                zorder=5,
                label=r"$Y_{t_*}=[X_1(t_*),X_1(t_*-0.1),X_1(t_*-0.2)]$",
            )
            for rank, idx in enumerate(delay_indices):
                ax.annotate(
                    f"delay {rank}",
                    (times[idx], states[idx, 0]),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_ylabel(coordinate_names[coordinate])
        ax.grid(True, alpha=0.3)
        if coordinate == 0:
            ax.legend(fontsize=8, ncol=3)

    axes[-1].set_xlabel("physical time")
    fig.suptitle(
        "Held-out stochastic Lorenz trajectory and selected delay embedding"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reference_paths(
    branch_times: Array,
    branch_paths: Array,
    n_plot: int,
    output_dir: Path,
) -> None:
    """Create one reference-ensemble trajectory figure per Lorenz coordinate."""
    n_plot = min(n_plot, branch_paths.shape[0])
    for coordinate in range(3):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for realization in range(n_plot):
            ax.plot(
                branch_times,
                branch_paths[realization, :, coordinate],
                linewidth=0.9,
                alpha=0.75,
            )
        ax.set_xlabel("time after branch")
        ax.set_ylabel(fr"$X^{{({coordinate + 1})}}$")
        ax.set_title(
            f"Reference conditional ensemble: coordinate {coordinate + 1}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"reference_ensemble_coordinate_{coordinate + 1}.png",
            dpi=180,
        )
        plt.close(fig)


def plot_reference_vs_estimated(
    reference: Array,
    estimated: Array,
    diagnostics: Dict[str, float],
    output_path: Path,
) -> None:
    """Compare empirical distributions and their means/90% intervals."""
    ref_x, ref_f = empirical_cdf(reference)
    est_x, est_f = empirical_cdf(estimated)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bins = max(20, min(80, int(math.sqrt(min(len(reference), len(estimated))))))
    axes[0].hist(
        reference,
        bins=bins,
        density=True,
        alpha=0.45,
        label="reference branches",
    )
    axes[0].hist(
        estimated,
        bins=bins,
        density=True,
        alpha=0.45,
        label="SI samples",
    )
    axes[0].axvline(
        diagnostics["reference_mean"],
        linestyle="-",
        linewidth=1.8,
        label="reference mean",
    )
    axes[0].axvline(
        diagnostics["estimated_mean"],
        linestyle="--",
        linewidth=1.8,
        label="SI mean",
    )
    axes[0].set_xlabel(r"forecast target $X^{(1)}(t_*+h)$")
    axes[0].set_ylabel("empirical density")
    axes[0].set_title("Reference and estimated conditional ensembles")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].step(ref_x, ref_f, where="post", label="reference ECDF")
    axes[1].step(est_x, est_f, where="post", label="SI ECDF")
    axes[1].hlines(
        [0.05, 0.95],
        xmin=min(ref_x[0], est_x[0]),
        xmax=max(ref_x[-1], est_x[-1]),
        linestyles=":",
        linewidth=1.0,
    )
    axes[1].set_xlabel(r"forecast target $X^{(1)}(t_*+h)$")
    axes[1].set_ylabel("empirical CDF")
    axes[1].set_title("Empirical CDF comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    coverage = diagnostics["reference_coverage_of_estimated_interval"]
    se = diagnostics["coverage_monte_carlo_standard_error"]
    fig.suptitle(
        "Local conditional coverage: "
        f"nominal 90%, observed {100*coverage:.2f}% "
        f"(reference Monte Carlo SE {100*se:.2f} percentage points)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    # Separate compact interval comparison.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rows = [1.0, 0.0]
    labels = ["Reference", "Stochastic interpolant"]
    means = [diagnostics["reference_mean"], diagnostics["estimated_mean"]]
    lows = [diagnostics["reference_lower"], diagnostics["estimated_lower"]]
    highs = [diagnostics["reference_upper"], diagnostics["estimated_upper"]]

    for y, mean, low, high in zip(rows, means, lows, highs):
        ax.hlines(y, low, high, linewidth=5)
        ax.plot(mean, y, "o", markersize=8)
        ax.plot([low, high], [y, y], "|", markersize=15)

    ax.set_yticks(rows, labels)
    ax.set_xlabel(r"$X^{(1)}(t_*+h)$")
    ax.set_title(
        "Conditional means and central 90% prediction intervals\n"
        f"SI interval coverage under reference law: {100*coverage:.2f}%"
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_path.with_name("reference_vs_estimated_intervals.png"),
        dpi=180,
    )
    plt.close(fig)


def parse_initial_condition(text: str) -> Tuple[float, float, float]:
    parts = [float(v.strip()) for v in text.split(",")]
    if len(parts) != 3 or not np.isfinite(parts).all():
        raise argparse.ArgumentTypeError(
            "initial-condition must contain three finite comma-separated values."
        )
    return tuple(parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local conditional coverage benchmark for stochastic Lorenz forecasting."
    )
    p.add_argument("--output-dir", type=Path, default=Path("lorenz_local_coverage_output"))

    p.add_argument("--lorenz-sigma", type=float, default=10.0)
    p.add_argument("--lorenz-rho", type=float, default=28.0)
    p.add_argument("--lorenz-beta", type=float, default=8.0 / 3.0)
    p.add_argument("--noise-amplitude", type=float, default=0.05)
    p.add_argument(
        "--initial-condition",
        type=parse_initial_condition,
        default=(1.0, 1.0, 1.0),
    )
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--observation-dt", type=float, default=0.01)

    p.add_argument("--embedding-dim", type=int, default=3)
    p.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Physical delay tau. Default gives [t,t-0.1,t-0.2].",
    )
    p.add_argument("--forecast-time", type=float, default=0.5)
    p.add_argument("--gap-steps", type=int, default=0)

    p.add_argument("--burn-in", type=float, default=30.0)
    p.add_argument("--training-data-time", type=float, default=30.0)
    p.add_argument("--anchor-post-burn-time", type=float, default=5.0)
    p.add_argument("--n-train-paths", type=int, default=80)
    p.add_argument("--n-val-paths", type=int, default=20)
    p.add_argument("--training-seed", type=int, default=100)
    p.add_argument("--anchor-seed", type=int, default=9001)

    p.add_argument("--reference-size", type=int, default=5000)
    p.add_argument("--estimated-size", type=int, default=5000)
    p.add_argument("--reference-seed", type=int, default=20001)
    p.add_argument("--estimated-seed", type=int, default=30001)
    p.add_argument("--n-reference-paths-to-plot", type=int, default=30)

    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--eta", type=float, default=0.35)
    p.add_argument("--sde-steps", type=int, default=300)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate data and trajectory/reference plots, but skip SI training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = LorenzConfig(
        sigma_lorenz=args.lorenz_sigma,
        rho_lorenz=args.lorenz_rho,
        beta_lorenz=args.lorenz_beta,
        noise_amplitude=args.noise_amplitude,
        initial_condition=args.initial_condition,
        dt=args.dt,
        observation_dt=args.observation_dt,
    )
    _ = cfg.observation_stride

    if args.embedding_dim != 3:
        print(
            "Warning: the requested benchmark specifies m=3; "
            f"running with user value m={args.embedding_dim}."
        )

    delay_steps = physical_steps(args.delay, cfg.observation_dt, "delay")
    horizon_steps = physical_steps(
        args.forecast_time, cfg.observation_dt, "forecast_time"
    )

    dataset = generate_training_dataset(
        cfg=cfg,
        n_train_paths=args.n_train_paths,
        n_val_paths=args.n_val_paths,
        burn_in=args.burn_in,
        data_time=args.training_data_time,
        delay_steps=delay_steps,
        embedding_dim=args.embedding_dim,
        horizon_steps=horizon_steps,
        gap_steps=args.gap_steps,
        seed=args.training_seed,
    )

    heldout = select_fixed_embedding(
        cfg=cfg,
        burn_in=args.burn_in,
        pre_anchor_time=args.anchor_post_burn_time,
        delay_steps=delay_steps,
        embedding_dim=args.embedding_dim,
        seed=args.anchor_seed,
    )

    branch_times, reference_paths = simulate_lorenz_ensemble_from_state(
        cfg=cfg,
        initial_state=np.asarray(heldout["x_star"]),
        forecast_time=args.forecast_time,
        n_ensemble=args.reference_size,
        seed=args.reference_seed,
        save_paths=True,
    )
    reference_targets = reference_paths[:, -1, 0]

    plot_selected_embedding(
        heldout,
        forecast_time=args.forecast_time,
        output_path=args.output_dir / "selected_delay_embedding_vs_trajectory.png",
    )
    plot_reference_paths(
        branch_times,
        reference_paths,
        n_plot=args.n_reference_paths_to_plot,
        output_dir=args.output_dir,
    )

    metadata = {
        "lorenz_config": asdict(cfg),
        "embedding_dimension": int(args.embedding_dim),
        "delay_physical": float(args.delay),
        "delay_steps": int(delay_steps),
        "forecast_time": float(args.forecast_time),
        "forecast_steps_saved": int(horizon_steps),
        "burn_in": float(args.burn_in),
        "anchor_time": float(heldout["anchor_time"]),
        "fixed_delay_embedding": np.asarray(heldout["y_star"]).tolist(),
        "fixed_latent_state": np.asarray(heldout["x_star"]).tolist(),
        "train_pair_count": int(len(dataset["train_Y"])),
        "val_pair_count": int(len(dataset["val_Y"])),
        "reference_size": int(args.reference_size),
        "state_determined_by_delay_assumption": True,
    }

    np.savez_compressed(
        args.output_dir / "prepared_lorenz_coverage_data.npz",
        train_Y=dataset["train_Y"],
        train_U=dataset["train_U"],
        val_Y=dataset["val_Y"],
        val_U=dataset["val_U"],
        heldout_times=np.asarray(heldout["times"]),
        heldout_states=np.asarray(heldout["states"]),
        delay_indices=np.asarray(heldout["delay_indices"]),
        y_star=np.asarray(heldout["y_star"]),
        x_star=np.asarray(heldout["x_star"]),
        branch_times=branch_times,
        reference_targets=reference_targets,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with open(
        args.output_dir / "experiment_metadata.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))

    if args.prepare_only:
        print("Preparation complete; SI training and estimated ensemble were skipped.")
        return

    train_cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
    )
    interpolant_cfg = InterpolantConfig(eta=args.eta)

    fit = train_conditional_drift(
        dataset["train_Y"],
        dataset["train_U"],
        dataset["val_Y"],
        dataset["val_U"],
        args.output_dir,
        train_cfg=train_cfg,
        interpolant_cfg=interpolant_cfg,
    )

    estimated_samples = sample_conditional_sde(
        fit["model"],
        np.asarray(heldout["y_star"]),
        fit["y_scaler"],
        fit["u_scaler"],
        interpolant_cfg,
        n_samples=args.estimated_size,
        n_steps=args.sde_steps,
        device=fit["device"],
        seed=args.estimated_seed,
    )[:, 0]

    diagnostics = compare_ensembles(
        reference_targets,
        estimated_samples,
        level=0.90,
    )

    plot_reference_vs_estimated(
        reference_targets,
        estimated_samples,
        diagnostics,
        args.output_dir / "reference_vs_estimated_ensemble.png",
    )

    np.savez_compressed(
        args.output_dir / "local_coverage_results.npz",
        reference_targets=reference_targets,
        estimated_targets=estimated_samples,
        y_star=np.asarray(heldout["y_star"]),
        x_star=np.asarray(heldout["x_star"]),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    with open(
        args.output_dir / "coverage_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(diagnostics, file, indent=2)

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
