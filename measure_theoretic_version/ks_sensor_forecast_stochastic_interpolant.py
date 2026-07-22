#!/usr/bin/env python3
"""
Probabilistic sparse-sensor forecasting for stochastic 1D Kuramoto--Sivashinsky data.

Input
-----
``test.npy`` must contain a numerical array with shape (N,T,D):

    N : number of independent stochastic realizations,
    T : number of stored time samples,
    D : number of periodic spatial grid points on [0,L).

For each realization n and anchor index t, four uniformly spaced sensors define

    z[n,t] in R^4.

The conditioning vector is the delay history

    Y[n,t] = [z[n,t], z[n,t-tau], ..., z[n,t-(m-1)tau]],

and the forecast target is

    U[n,t] = z[n,t+h],

where h is ``--forecast-horizon`` in stored time steps.

Entire realizations are split into train/validation/test groups. Within each
realization, selected examples have disjoint closed time intervals from the
oldest input sample through the forecast target. This prevents exact time-index
reuse across retained examples. It does not, by itself, prove probabilistic
independence for a temporally correlated stochastic process.

The conditional stochastic-interpolant implementation is imported from
``gaussian_source_si.py`` and must be in the same directory or on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gaussian_source_si import (
    InterpolantConfig,
    TrainConfig,
    conditional_ensemble_metrics,
    sample_conditional_sde,
    train_conditional_drift,
)


Array = np.ndarray


@dataclass(frozen=True)
class RealizationSplit:
    train: Array
    val: Array
    test: Array


def load_trajectory_array(path: str | Path) -> Array:
    """Load and rigorously validate an array with shape (N,T,D)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input trajectory file not found: {path}")

    trajectories = np.load(path, allow_pickle=False)
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected an array with shape (N,T,D), received {trajectories.shape}."
        )
    if min(trajectories.shape) < 1:
        raise ValueError("N, T, and D must all be positive.")
    if not np.issubdtype(trajectories.dtype, np.number):
        raise TypeError("The trajectory array must be numerical.")

    trajectories = np.asarray(trajectories, dtype=np.float64)
    if not np.isfinite(trajectories).all():
        raise ValueError("Trajectory data contain NaN or infinite values.")

    return trajectories


def select_realizations(
    trajectories: Array,
    n_realizations: int | None,
    selection: str,
    seed: int,
) -> Tuple[Array, Array]:
    """Select all, the first n, or n random realizations without replacement."""
    n_available = trajectories.shape[0]

    if n_realizations is None:
        indices = np.arange(n_available, dtype=np.int64)
    else:
        if n_realizations < 3:
            raise ValueError(
                "--n-realizations must be at least 3 for nonempty "
                "train/validation/test realization splits."
            )
        if n_realizations > n_available:
            raise ValueError(
                f"Requested {n_realizations} realizations, but only "
                f"{n_available} are available."
            )

        if selection == "first":
            indices = np.arange(n_realizations, dtype=np.int64)
        elif selection == "random":
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(n_available, size=n_realizations, replace=False)
            ).astype(np.int64)
        else:
            raise ValueError("selection must be 'first' or 'random'.")

    if indices.size < 3:
        raise ValueError("At least three realizations are required.")

    return trajectories[indices], indices


def split_realizations(
    n_realizations: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> RealizationSplit:
    """Split whole realization IDs into nonempty train/validation/test sets."""
    if n_realizations < 3:
        raise ValueError("At least three realizations are required.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0,1).")
    if not 0.0 < val_fraction < 1.0 - train_fraction:
        raise ValueError("val_fraction must lie in (0,1-train_fraction).")

    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_realizations)

    n_train = max(1, int(math.floor(train_fraction * n_realizations)))
    n_val = max(1, int(math.floor(val_fraction * n_realizations)))

    if n_train + n_val >= n_realizations:
        overflow = n_train + n_val - (n_realizations - 1)
        if n_train >= n_val and n_train - overflow >= 1:
            n_train -= overflow
        elif n_val - overflow >= 1:
            n_val -= overflow
        else:
            raise ValueError("Fractions do not permit three nonempty splits.")

    return RealizationSplit(
        train=np.sort(ids[:n_train]),
        val=np.sort(ids[n_train:n_train + n_val]),
        test=np.sort(ids[n_train + n_val:]),
    )


def uniform_periodic_sensor_indices(
    n_space: int,
    n_sensors: int = 4,
) -> Array:
    """
    Select uniformly spaced sensor indices on x_j=jL/D, j=0,...,D-1.
    """
    if n_sensors < 1:
        raise ValueError("n_sensors must be positive.")
    if n_space < n_sensors:
        raise ValueError("D must be at least the number of sensors.")

    indices = np.floor(
        np.arange(n_sensors, dtype=np.float64) * n_space / n_sensors
    ).astype(np.int64)

    if np.unique(indices).size != n_sensors:
        raise RuntimeError("Uniform sensor construction produced duplicates.")

    return indices


def sensor_observations(
    trajectories: Array,
    sensor_indices: Array,
) -> Array:
    """Extract point observations, producing shape (N,T,n_sensors)."""
    return trajectories[:, :, sensor_indices]


def example_raw_indices(
    anchor: int,
    tau: int,
    embedding_dim: int,
    forecast_horizon: int,
) -> set[int]:
    """Indices used by one input window together with its future target."""
    input_indices = anchor - np.arange(embedding_dim, dtype=np.int64) * tau
    return set(input_indices.tolist() + [anchor + forecast_horizon])


def build_nonoverlapping_forecast_pairs(
    observations: Array,
    realization_ids: Iterable[int],
    tau: int,
    embedding_dim: int,
    forecast_horizon: int,
    gap_steps: int,
) -> Dict[str, Array]:
    r"""
    Construct non-overlapping sensor-history-to-future-sensor pairs.

    For realization n and anchor t,

        Y[n,t] = [z[n,t], z[n,t-tau], ..., z[n,t-(m-1)tau]],
        U[n,t] = z[n,t+h].

    A retained example occupies the closed index interval

        [t-(m-1)tau, t+h].

    The anchor stride is

        (m-1)tau + h + gap_steps + 1,

    so occupied intervals from consecutive retained examples are disjoint and
    separated by ``gap_steps`` unused indices.
    """
    if observations.ndim != 3:
        raise ValueError("observations must have shape (N,T,n_sensors).")
    if tau < 1:
        raise ValueError("tau must be positive.")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive.")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be positive.")
    if gap_steps < 0:
        raise ValueError("gap_steps must be nonnegative.")

    _, n_times, _ = observations.shape
    history_span = (embedding_dim - 1) * tau
    occupied_span = history_span + forecast_horizon
    anchor_stride = occupied_span + gap_steps + 1

    first_anchor = history_span
    last_anchor_exclusive = n_times - forecast_horizon
    if first_anchor >= last_anchor_exclusive:
        raise ValueError(
            "No valid forecast pairs: T is too short for the requested "
            "tau, embedding dimension, and forecast horizon."
        )

    offsets = np.arange(embedding_dim, dtype=np.int64) * tau
    y_list: List[Array] = []
    u_list: List[Array] = []
    realization_list: List[int] = []
    anchor_list: List[int] = []
    target_time_list: List[int] = []

    for realization in np.asarray(list(realization_ids), dtype=np.int64):
        anchors = np.arange(
            first_anchor,
            last_anchor_exclusive,
            anchor_stride,
            dtype=np.int64,
        )

        previous_indices: set[int] | None = None
        for anchor in anchors:
            current_indices = example_raw_indices(
                int(anchor), tau, embedding_dim, forecast_horizon
            )
            if previous_indices is not None and previous_indices.intersection(
                current_indices
            ):
                raise AssertionError("Internal error: retained examples overlap.")
            previous_indices = current_indices

            y_list.append(
                observations[realization, anchor - offsets].reshape(-1)
            )
            u_list.append(
                observations[realization, anchor + forecast_horizon]
            )
            realization_list.append(int(realization))
            anchor_list.append(int(anchor))
            target_time_list.append(int(anchor + forecast_horizon))

    if not y_list:
        raise ValueError("No forecast pairs were constructed.")

    return {
        "Y": np.asarray(y_list, dtype=np.float64),
        "U": np.asarray(u_list, dtype=np.float64),
        "realization": np.asarray(realization_list, dtype=np.int64),
        "anchor": np.asarray(anchor_list, dtype=np.int64),
        "target_time": np.asarray(target_time_list, dtype=np.int64),
        "history_span": np.asarray(history_span, dtype=np.int64),
        "occupied_span": np.asarray(occupied_span, dtype=np.int64),
        "anchor_stride": np.asarray(anchor_stride, dtype=np.int64),
    }


def plot_original_trajectories_and_sensors(
    trajectories: Array,
    observations: Array,
    sensor_indices: Array,
    split: RealizationSplit,
    domain_length: float,
    dt: float,
    output_path: Path,
) -> None:
    """Visualize representative trajectories and the four partial observations."""
    _, n_times, n_space = trajectories.shape
    x = domain_length * np.arange(n_space) / n_space
    t = dt * np.arange(n_times)

    representative = int(split.train[0])
    snapshot_index = n_times // 2

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    image = axes[0, 0].imshow(
        trajectories[representative],
        origin="lower",
        aspect="auto",
        extent=[0.0, domain_length, t[0], t[-1]],
        interpolation="nearest",
    )
    for sensor_number, index in enumerate(sensor_indices):
        axes[0, 0].axvline(
            x[index],
            linestyle="--",
            linewidth=1.2,
            label="sensor locations" if sensor_number == 0 else None,
        )
    axes[0, 0].set_title(
        f"Space-time field, training realization {representative}"
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("time")
    axes[0, 0].legend()
    fig.colorbar(image, ax=axes[0, 0], label="u(x,t)")

    axes[0, 1].plot(
        x,
        trajectories[representative, snapshot_index],
        label="full field",
    )
    axes[0, 1].scatter(
        x[sensor_indices],
        trajectories[representative, snapshot_index, sensor_indices],
        marker="o",
        zorder=5,
        label="sensor measurements",
    )
    axes[0, 1].set_title(f"Spatial field at t={t[snapshot_index]:.4g}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    for sensor_number, index in enumerate(sensor_indices):
        axes[1, 0].plot(
            t,
            observations[representative, :, sensor_number],
            label=f"x={x[index]:.4g}",
        )
    axes[1, 0].set_title("Four sensor time series")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("sensor value")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(ncol=2)

    chosen_realizations = [
        int(split.train[0]),
        int(split.val[0]),
        int(split.test[0]),
    ]
    labels = ["train", "validation", "test"]
    sensor_number = 0
    for realization, label in zip(chosen_realizations, labels):
        axes[1, 1].plot(
            t,
            observations[realization, :, sensor_number],
            label=f"{label} realization {realization}",
        )
    axes[1, 1].set_title(
        f"Sensor 1 across train/validation/test realizations"
    )
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("sensor value")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(
        "Stochastic Kuramoto--Sivashinsky trajectories and sparse sensors"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_ordered_test_subset(
    test_pairs: Dict[str, Array],
    maximum_points: int,
) -> Array:
    """
    Select time-ordered forecast pairs from one held-out realization.

    Samples from distinct stochastic realizations are not connected into a
    single temporal curve.
    """
    ids, counts = np.unique(test_pairs["realization"], return_counts=True)
    realization = int(ids[np.argmax(counts)])
    indices = np.where(test_pairs["realization"] == realization)[0]
    indices = indices[np.argsort(test_pairs["target_time"][indices])]
    return indices[:maximum_points]


def plot_sensor_forecasts(
    test_pairs: Dict[str, Array],
    selected_indices: Array,
    truth: Array,
    ensemble_mean: Array,
    lower: Array,
    upper: Array,
    sensor_locations: Array,
    dt: float,
    confidence_level: float,
    output_path: Path,
) -> None:
    """Compare true and probabilistic forecasts for each sensor."""
    n_sensors = truth.shape[1]
    target_times = dt * test_pairs["target_time"][selected_indices]
    realization = int(test_pairs["realization"][selected_indices[0]])

    fig, axes = plt.subplots(
        n_sensors,
        1,
        figsize=(11, 2.5 * n_sensors),
        sharex=True,
        squeeze=False,
    )

    for sensor in range(n_sensors):
        ax = axes[sensor, 0]
        ax.fill_between(
            target_times,
            lower[:, sensor],
            upper[:, sensor],
            alpha=0.25,
            label=f"{100.0 * confidence_level:.1f}% empirical prediction interval",
            color="blue"
        )
        ax.plot(
            target_times,
            truth[:, sensor],
            "o-",
            markersize=3,
            linewidth=1.0,
            label="true future measurement",
            color="black"
        )
        ax.plot(
            target_times,
            ensemble_mean[:, sensor],
            "s--",
            markersize=3,
            linewidth=1.0,
            label="SI ensemble mean",
            color="blue"
        )
        ax.set_ylabel(fr"$z_{{{sensor + 1}}}$")
        ax.set_title(f"Sensor at x={sensor_locations[sensor]:.4g}")
        ax.grid(True, alpha=0.3)
        if sensor == 0:
            ax.legend(ncol=3, fontsize=8)

    axes[-1, 0].set_xlabel("forecast target time")
    fig.suptitle(
        f"Held-out realization {realization}: probabilistic sensor forecasts"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def empirical_interval(
    samples: Array,
    confidence_level: float,
) -> Tuple[Array, Array]:
    """
    Return equal-tailed empirical prediction bounds from ensemble samples.

    For nominal level c, alpha=1-c and the bounds are the componentwise
    empirical alpha/2 and 1-alpha/2 quantiles.
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (ensemble_size,target_dim).")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1).")

    alpha = 1.0 - confidence_level
    return (
        np.quantile(samples, alpha / 2.0, axis=0),
        np.quantile(samples, 1.0 - alpha / 2.0, axis=0),
    )


def empirical_marginal_coverage(
    truth: Array,
    lower: Array,
    upper: Array,
) -> Array:
    """Compute componentwise empirical coverage over evaluated test pairs."""
    if truth.shape != lower.shape or truth.shape != upper.shape:
        raise ValueError("truth, lower, and upper must have identical shapes.")
    return np.mean((truth >= lower) & (truth <= upper), axis=0)


def save_prepared_dataset(
    output_path: Path,
    trajectories_shape: Tuple[int, int, int],
    selected_original_indices: Array,
    split: RealizationSplit,
    sensor_indices: Array,
    sensor_locations: Array,
    pairs: Dict[str, Dict[str, Array]],
    metadata: dict,
) -> None:
    """Save all arrays required to train/evaluate the forecasting model."""
    payload = {
        "trajectory_shape": np.asarray(trajectories_shape, dtype=np.int64),
        "selected_original_realization_indices": np.asarray(
            selected_original_indices, dtype=np.int64
        ),
        "train_realizations": split.train,
        "val_realizations": split.val,
        "test_realizations": split.test,
        "sensor_indices": sensor_indices,
        "sensor_locations": sensor_locations,
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    for split_name, split_pairs in pairs.items():
        for key, value in split_pairs.items():
            payload[f"{split_name}_{key}"] = value

    np.savez_compressed(output_path, **payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare and train a conditional stochastic interpolant for "
            "forecasting four KS sensor measurements."
        )
    )
    p.add_argument("--input", type=Path, default=Path("test.npy"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ks_sensor_forecast_si_output"),
    )
    p.add_argument("--domain-length", type=float, default=22.0)
    p.add_argument("--noise-level", type=float, default=0.05)
    p.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Physical time between adjacent stored samples.",
    )
    p.add_argument(
        "--tau",
        type=int,
        default=5,
        help="Delay in stored time-index steps.",
    )
    p.add_argument("--embedding-dim", type=int, default=4)
    p.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="Forecast lead time in stored time-index steps.",
    )
    p.add_argument(
        "--gap-steps",
        type=int,
        default=0,
        help="Unused indices placed between consecutive retained examples.",
    )
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--n-realizations", type=int, default=None)
    p.add_argument(
        "--realization-selection",
        choices=("first", "random"),
        default="first",
    )
    p.add_argument("--selection-seed", type=int, default=17)

    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save the forecast dataset and overview plot but skip SI training.",
    )
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--eta", type=float, default=0.35)
    p.add_argument("--ensemble-size", type=int, default=500)
    p.add_argument("--sde-steps", type=int, default=300)
    p.add_argument("--max-forecast-points", type=int, default=40)
    p.add_argument(
        "--confidence-level",
        type=float,
        default=0.90,
        help="Nominal equal-tailed empirical prediction-interval level.",
    )
    p.add_argument("--evaluation-seed", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.domain_length <= 0.0:
        raise ValueError("domain-length must be positive.")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if args.max_forecast_points < 1:
        raise ValueError("max-forecast-points must be positive.")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence-level must lie in (0,1).")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_trajectories = load_trajectory_array(args.input)
    trajectories, selected_original_indices = select_realizations(
        full_trajectories,
        args.n_realizations,
        args.realization_selection,
        args.selection_seed,
    )
    n_realizations, n_times, n_space = trajectories.shape

    split = split_realizations(
        n_realizations,
        args.train_fraction,
        args.val_fraction,
        args.split_seed,
    )

    sensor_indices = uniform_periodic_sensor_indices(n_space, n_sensors=4)
    sensor_locations = args.domain_length * sensor_indices / n_space
    observations = sensor_observations(trajectories, sensor_indices)

    pairs = {
        name: build_nonoverlapping_forecast_pairs(
            observations,
            realization_ids=getattr(split, name),
            tau=args.tau,
            embedding_dim=args.embedding_dim,
            forecast_horizon=args.forecast_horizon,
            gap_steps=args.gap_steps,
        )
        for name in ("train", "val", "test")
    }

    metadata = {
        "equation": "stochastic Kuramoto-Sivashinsky; user-supplied trajectories",
        "domain_length": float(args.domain_length),
        "noise_level": float(args.noise_level),
        "dt_between_stored_samples": float(args.dt),
        "tau_in_stored_steps": int(args.tau),
        "tau_physical": float(args.tau * args.dt),
        "embedding_dimension": int(args.embedding_dim),
        "forecast_horizon_in_stored_steps": int(args.forecast_horizon),
        "forecast_horizon_physical": float(args.forecast_horizon * args.dt),
        "gap_steps": int(args.gap_steps),
        "n_sensors": 4,
        "condition_dimension": int(4 * args.embedding_dim),
        "target_dimension": 4,
        "split_by_realization": True,
        "available_realizations_in_input": int(full_trajectories.shape[0]),
        "selected_realization_count": int(n_realizations),
        "realization_selection": args.realization_selection,
        "selection_seed": int(args.selection_seed),
        "selected_original_realization_indices": (
            selected_original_indices.tolist()
        ),
        "train_pair_count": int(len(pairs["train"]["Y"])),
        "val_pair_count": int(len(pairs["val"]["Y"])),
        "test_pair_count": int(len(pairs["test"]["Y"])),
        "history_span": int(pairs["train"]["history_span"]),
        "occupied_span": int(pairs["train"]["occupied_span"]),
        "anchor_stride": int(pairs["train"]["anchor_stride"]),
    }

    save_prepared_dataset(
        output_dir / "ks_sensor_forecast_dataset.npz",
        trajectories.shape,
        selected_original_indices,
        split,
        sensor_indices,
        sensor_locations,
        pairs,
        metadata,
    )

    plot_original_trajectories_and_sensors(
        trajectories,
        observations,
        sensor_indices,
        split,
        args.domain_length,
        args.dt,
        output_dir / "original_trajectories_and_sensors.png",
    )

    with open(
        output_dir / "dataset_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                **metadata,
                "input_shape_full": list(full_trajectories.shape),
                "input_shape_selected": list(trajectories.shape),
                "train_realizations": split.train.tolist(),
                "val_realizations": split.val.tolist(),
                "test_realizations": split.test.tolist(),
                "sensor_indices": sensor_indices.tolist(),
                "sensor_locations": sensor_locations.tolist(),
            },
            file,
            indent=2,
        )

    print(json.dumps(metadata, indent=2))
    print(
        "Saved prepared forecast dataset to "
        f"{output_dir / 'ks_sensor_forecast_dataset.npz'}"
    )

    if args.prepare_only:
        print("Preparation complete; SI training was skipped.")
        return

    train_cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    interpolant_cfg = InterpolantConfig(eta=args.eta)

    fit = train_conditional_drift(
        pairs["train"]["Y"],
        pairs["train"]["U"],
        pairs["val"]["Y"],
        pairs["val"]["U"],
        output_dir,
        train_cfg=train_cfg,
        interpolant_cfg=interpolant_cfg,
    )

    selected = select_ordered_test_subset(
        pairs["test"], args.max_forecast_points
    )
    truth = pairs["test"]["U"][selected]

    means: List[Array] = []
    lowers: List[Array] = []
    uppers: List[Array] = []
    rmses: List[float] = []
    spreads: List[float] = []

    for local_index, pair_index in enumerate(selected):
        samples = sample_conditional_sde(
            fit["model"],
            pairs["test"]["Y"][pair_index],
            fit["y_scaler"],
            fit["u_scaler"],
            interpolant_cfg,
            n_samples=args.ensemble_size,
            n_steps=args.sde_steps,
            device=fit["device"],
            seed=args.evaluation_seed + local_index,
        )
        lower, upper = empirical_interval(
            samples, args.confidence_level
        )

        means.append(np.mean(samples, axis=0))
        lowers.append(lower)
        uppers.append(upper)

        metrics = conditional_ensemble_metrics(
            samples,
            pairs["test"]["U"][pair_index],
        )
        rmses.append(float(metrics["ensemble_mean_rmse"]))
        spreads.append(float(metrics["ensemble_spread_rms"]))

    means_array = np.asarray(means)
    lower_array = np.asarray(lowers)
    upper_array = np.asarray(uppers)

    plot_sensor_forecasts(
        pairs["test"],
        selected,
        truth,
        means_array,
        lower_array,
        upper_array,
        sensor_locations,
        args.dt,
        args.confidence_level,
        output_dir / "sensor_forecasts_vs_truth.png",
    )

    marginal_coverage = empirical_marginal_coverage(
        truth,
        lower_array,
        upper_array,
    )
    mean_width = np.mean(upper_array - lower_array, axis=0)

    np.savez_compressed(
        output_dir / "test_forecast_results.npz",
        selected_pair_indices=selected,
        realization=pairs["test"]["realization"][selected],
        anchor=pairs["test"]["anchor"][selected],
        target_time=pairs["test"]["target_time"][selected],
        truth=truth,
        ensemble_mean=means_array,
        interval_lower=lower_array,
        interval_upper=upper_array,
        confidence_level=np.asarray(args.confidence_level),
        marginal_coverage=marginal_coverage,
        mean_interval_width=mean_width,
        ensemble_mean_rmse=np.asarray(rmses),
        ensemble_spread_rms=np.asarray(spreads),
    )

    evaluation_summary = {
        "number_evaluated": int(len(selected)),
        "evaluated_realization": int(
            pairs["test"]["realization"][selected[0]]
        ),
        "nominal_interval_level": float(args.confidence_level),
        "marginal_sensor_coverage": marginal_coverage.tolist(),
        "mean_interval_width_by_sensor": mean_width.tolist(),
        "mean_ensemble_mean_rmse": float(np.mean(rmses)),
        "mean_ensemble_spread_rms": float(np.mean(spreads)),
    }
    with open(
        output_dir / "evaluation_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(evaluation_summary, file, indent=2)

    print(json.dumps(evaluation_summary, indent=2))


if __name__ == "__main__":
    main()
