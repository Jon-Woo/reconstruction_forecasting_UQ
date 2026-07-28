#!/usr/bin/env python3
"""
Create normalized delay-coordinate/POD-coefficient datasets from stochastic
Kuramoto--Sivashinsky trajectories.

The input is a NumPy array with shape (N, T, D):

    N = number of trajectories,
    T = number of stored time indices,
    D = number of spatial grid points.

The periodic spatial domain has length L=22.

For each trajectory, the time axis is split chronologically:

    training:   first 70 percent,
    validation: next 15 percent,
    test:       final 15 percent.

Four uniformly spaced spatial sensors define the partial observation map P.
For embedding dimension m and integer time delay tau, the conditioning vector
at anchor time t is

    Y_t = [
        P(X_t),
        P(X_{t-tau}),
        ...,
        P(X_{t-(m-1)tau})
    ] in R^(4m).

The target is the vector of leading POD coefficients of X_t. The POD mean and
basis are fitted using only snapshots from the training portions of the
trajectories. The smallest number of modes whose cumulative singular-value
energy is at least the requested threshold is retained; the default threshold
is 0.99.

Delay-window spans are non-overlapping. If one span ends at time e, the next
span starts no earlier than e + gap + 1, leaving exactly `gap` unused time
indices between consecutive selected windows.

The output NPZ contains the normalized arrays expected by
train_drift_with_visualization.py:

    Y_train, U_train,
    Y_val,   U_val,
    Y_test,  U_test,
    mu_Y, sigma_Y,
    mu_U, sigma_U.

Additional POD, sensor, split, and sample-provenance arrays are also saved.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
NUM_SENSORS = 4


@dataclass(frozen=True)
class PairConfig:
    """Configuration for delay windows, POD truncation, and normalization."""

    m: int
    tau: int
    gap: int
    energy_threshold: float = 0.99
    normalization_epsilon: float = 1.0e-8
    domain_length: float = 22.0

    def validate(self) -> None:
        if self.m < 1:
            raise ValueError("m must be a positive integer.")
        if self.tau < 1:
            raise ValueError("tau must be a positive integer.")
        if self.gap < 0:
            raise ValueError("gap must be nonnegative.")
        if not 0.0 < self.energy_threshold <= 1.0:
            raise ValueError(
                "energy_threshold must lie in the interval (0,1]."
            )
        if self.normalization_epsilon <= 0.0:
            raise ValueError(
                "normalization_epsilon must be positive."
            )
        if self.domain_length <= 0.0:
            raise ValueError("domain_length must be positive.")

    @property
    def embedding_span(self) -> int:
        """Distance from the oldest observation to the anchor time."""
        return (self.m - 1) * self.tau

    @property
    def window_span_length(self) -> int:
        """Number of time indices in the inclusive delay-window span."""
        return self.embedding_span + 1

    @property
    def anchor_stride(self) -> int:
        """
        Distance between consecutive anchor indices.

        This makes the full spans [t-(m-1)tau, t] non-overlapping and leaves
        exactly `gap` unused time indices between consecutive spans.
        """
        return self.window_span_length + self.gap


@dataclass(frozen=True)
class SplitBounds:
    """Half-open chronological split intervals."""

    train: Tuple[int, int]
    validation: Tuple[int, int]
    test: Tuple[int, int]

    def as_dict(self) -> Dict[str, Tuple[int, int]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class Standardizer:
    """Componentwise affine standardizer fitted on training pairs only."""

    mean: Array
    scale: Array
    regularized_mask: Array

    def transform(self, values: Array) -> Array:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class PODModel:
    """Training-only POD model."""

    mean_field: Array
    modes: Array
    singular_values: Array
    energy_fraction: Array
    cumulative_energy: Array
    num_modes: int

    def coefficients(self, snapshots: Array) -> Array:
        centered = snapshots - self.mean_field
        return centered @ self.modes.T


@dataclass
class PairBuffer:
    """Raw delay/POD pairs and sample provenance."""

    Y: List[Array]
    U: List[Array]
    trajectory_index: List[int]
    anchor_index: List[int]
    window_start: List[int]
    window_end: List[int]

    @classmethod
    def empty(cls) -> "PairBuffer":
        return cls(
            Y=[],
            U=[],
            trajectory_index=[],
            anchor_index=[],
            window_start=[],
            window_end=[],
        )

    def append(
        self,
        *,
        y: Array,
        u: Array,
        trajectory_index: int,
        anchor_index: int,
        window_start: int,
        window_end: int,
    ) -> None:
        self.Y.append(y)
        self.U.append(u)
        self.trajectory_index.append(trajectory_index)
        self.anchor_index.append(anchor_index)
        self.window_start.append(window_start)
        self.window_end.append(window_end)

    def finalize(
        self,
        *,
        split_name: str,
        condition_dim: int,
        target_dim: int,
    ) -> Dict[str, Array]:
        if not self.Y:
            raise ValueError(
                f"No valid pairs were created for the {split_name} split. "
                "Use longer trajectories or reduce m, tau, or gap."
            )

        Y = np.stack(self.Y, axis=0).astype(np.float32, copy=False)
        U = np.stack(self.U, axis=0).astype(np.float32, copy=False)

        if Y.shape[1] != condition_dim:
            raise RuntimeError(
                f"Internal Y dimension mismatch in the {split_name} split."
            )
        if U.shape[1] != target_dim:
            raise RuntimeError(
                f"Internal U dimension mismatch in the {split_name} split."
            )

        return {
            "Y": Y,
            "U": U,
            "trajectory_index": np.asarray(
                self.trajectory_index,
                dtype=np.int64,
            ),
            "anchor_index": np.asarray(
                self.anchor_index,
                dtype=np.int64,
            ),
            "window_start": np.asarray(
                self.window_start,
                dtype=np.int64,
            ),
            "window_end": np.asarray(
                self.window_end,
                dtype=np.int64,
            ),
        }


def load_trajectories(path: str | Path) -> Array:
    """Load and validate a finite numerical array with shape (N,T,D)."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    trajectories = np.load(path, allow_pickle=False)
    trajectories = np.asarray(trajectories)

    if trajectories.ndim != 3:
        raise ValueError(
            "Expected an array with shape (N,T,D), "
            f"but received shape {trajectories.shape}."
        )

    n_trajectories, n_times, n_grid = trajectories.shape

    if n_trajectories < 1:
        raise ValueError("N must be positive.")
    if n_times < 3:
        raise ValueError("T must be at least 3.")
    if n_grid < NUM_SENSORS:
        raise ValueError(
            f"D must be at least {NUM_SENSORS} to use four sensors."
        )
    if not np.issubdtype(trajectories.dtype, np.number):
        raise TypeError("The trajectory array must be numerical.")
    if not np.isfinite(trajectories).all():
        raise ValueError(
            "The trajectory array contains NaN or infinite values."
        )

    return np.asarray(trajectories, dtype=np.float32)


def compute_split_bounds(n_times: int) -> SplitBounds:
    """
    Compute the per-trajectory 70%/15%/15% chronological split.

    The intervals are half-open and cover every time index exactly once.
    """
    train_end = int(np.floor(0.70 * n_times))
    validation_end = int(np.floor(0.85 * n_times))

    bounds = SplitBounds(
        train=(0, train_end),
        validation=(train_end, validation_end),
        test=(validation_end, n_times),
    )

    for name, (start, end) in bounds.as_dict().items():
        if end <= start:
            raise ValueError(
                f"The {name} split is empty for T={n_times}."
            )

    return bounds


def choose_uniform_periodic_sensors(
    n_grid: int,
    *,
    domain_length: float,
) -> Tuple[Array, Array]:
    """
    Select four uniformly spaced sensors on a periodic spatial grid.

    The indices are floor(kD/4), k=0,1,2,3. This avoids treating the final
    spatial point as a duplicated periodic endpoint.
    """
    sensor_indices = np.floor(
        np.arange(NUM_SENSORS, dtype=np.float64)
        * n_grid
        / NUM_SENSORS
    ).astype(np.int64)

    if np.unique(sensor_indices).size != NUM_SENSORS:
        raise RuntimeError("Uniform sensor selection produced duplicates.")

    sensor_positions = (
        domain_length * sensor_indices.astype(np.float64) / n_grid
    )

    return sensor_indices, sensor_positions


def fit_training_pod(
    trajectories: Array,
    train_bounds: Tuple[int, int],
    energy_threshold: float,
) -> PODModel:
    """
    Fit the POD mean and basis using all training-portion snapshots only.
    """
    train_start, train_end = train_bounds

    training_snapshots = trajectories[:, train_start:train_end, :].reshape(
        -1,
        trajectories.shape[2],
    )
    training_snapshots = np.asarray(training_snapshots, dtype=np.float64)

    if training_snapshots.shape[0] < 2:
        raise ValueError(
            "At least two training snapshots are required to fit POD."
        )

    mean_field = training_snapshots.mean(axis=0)
    centered = training_snapshots - mean_field

    _, singular_values, right_singular_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    squared_singular_values = singular_values**2
    total_energy = squared_singular_values.sum()

    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError(
            "The centered training snapshots have zero or invalid POD energy."
        )

    energy_fraction = squared_singular_values / total_energy
    cumulative_energy = np.cumsum(energy_fraction)

    num_modes = int(
        np.searchsorted(
            cumulative_energy,
            energy_threshold,
            side="left",
        )
        + 1
    )

    modes = right_singular_vectors[:num_modes, :]

    return PODModel(
        mean_field=mean_field.astype(np.float32),
        modes=modes.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        energy_fraction=energy_fraction.astype(np.float64),
        cumulative_energy=cumulative_energy.astype(np.float64),
        num_modes=num_modes,
    )


def save_pod_energy_plot(
    pod: PODModel,
    *,
    threshold: float,
    output_path: str | Path,
) -> Path:
    """Save a cumulative POD-energy plot identifying the selected rank."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode_numbers = np.arange(
        1,
        pod.cumulative_energy.size + 1,
        dtype=np.int64,
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        mode_numbers,
        pod.cumulative_energy,
        label="Cumulative POD energy",
    )
    axis.axhline(
        threshold,
        label=f"Energy threshold = {threshold:.2%}",
    )
    axis.axvline(
        pod.num_modes,
        label=f"Selected modes = {pod.num_modes}",
    )
    axis.scatter(
        [pod.num_modes],
        [pod.cumulative_energy[pod.num_modes - 1]],
        label=(
            "Retained energy = "
            f"{pod.cumulative_energy[pod.num_modes - 1]:.4%}"
        ),
    )

    axis.set_xlabel("Number of retained POD modes")
    axis.set_ylabel("Cumulative captured energy")
    axis.set_title(
        "Training-only POD cumulative energy\n"
        f"Retaining r={pod.num_modes} modes"
    )
    axis.set_ylim(0.0, 1.01)
    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def anchor_indices_for_split(
    split_start: int,
    split_end: int,
    config: PairConfig,
) -> Array:
    """
    Construct anchor times whose complete delay spans lie in one split.

    The first window begins at split_start. Consecutive window spans are
    separated by exactly `gap` unused time indices.
    """
    first_anchor = split_start + config.embedding_span
    last_anchor = split_end - 1

    if first_anchor > last_anchor:
        return np.empty(0, dtype=np.int64)

    return np.arange(
        first_anchor,
        last_anchor + 1,
        config.anchor_stride,
        dtype=np.int64,
    )


def verify_window_selection(
    anchors: Array,
    *,
    split_start: int,
    split_end: int,
    config: PairConfig,
) -> None:
    """Verify split containment, non-overlap, and the requested gap."""
    if anchors.size == 0:
        return

    starts = anchors - config.embedding_span
    ends = anchors

    if starts[0] < split_start or ends[-1] >= split_end:
        raise RuntimeError("A selected delay window crosses a split boundary.")

    if anchors.size > 1:
        unused_counts = starts[1:] - ends[:-1] - 1
        if np.any(unused_counts < config.gap):
            raise RuntimeError(
                "Selected delay-window spans overlap or violate the gap."
            )


def build_split_pairs(
    trajectories: Array,
    *,
    split_name: str,
    split_bounds: Tuple[int, int],
    sensor_indices: Array,
    pod: PODModel,
    config: PairConfig,
) -> Dict[str, Array]:
    """
    Build raw partial-delay/POD-coefficient pairs for one split.
    """
    split_start, split_end = split_bounds
    anchors = anchor_indices_for_split(
        split_start,
        split_end,
        config,
    )
    verify_window_selection(
        anchors,
        split_start=split_start,
        split_end=split_end,
        config=config,
    )

    condition_dim = NUM_SENSORS * config.m
    target_dim = pod.num_modes
    pairs = PairBuffer.empty()

    delay_offsets = (
        np.arange(config.m, dtype=np.int64) * config.tau
    )

    for trajectory_index in range(trajectories.shape[0]):
        for anchor_index in anchors:
            delay_indices = anchor_index - delay_offsets

            # Shape (m, 4), ordered from current time to the oldest delay.
            sensor_history = trajectories[
                trajectory_index,
                delay_indices[:, None],
                sensor_indices[None, :],
            ]

            y = sensor_history.reshape(condition_dim)

            snapshot = trajectories[
                trajectory_index,
                anchor_index,
                :,
            ]
            u = pod.coefficients(snapshot[None, :])[0]

            pairs.append(
                y=y,
                u=u,
                trajectory_index=trajectory_index,
                anchor_index=int(anchor_index),
                window_start=int(delay_indices[-1]),
                window_end=int(anchor_index),
            )

    return pairs.finalize(
        split_name=split_name,
        condition_dim=condition_dim,
        target_dim=target_dim,
    )


def fit_standardizer(
    values: Array,
    *,
    epsilon: float,
) -> Standardizer:
    """Fit componentwise mean/std using one training array."""
    values64 = np.asarray(values, dtype=np.float64)

    mean = values64.mean(axis=0)
    raw_scale = values64.std(axis=0, ddof=0)
    regularized_mask = raw_scale < epsilon
    scale = np.where(regularized_mask, 1.0, raw_scale)

    return Standardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        regularized_mask=regularized_mask.astype(bool),
    )


def prepare_dataset(
    trajectories: Array,
    *,
    config: PairConfig,
) -> Tuple[Dict[str, Array], Dict[str, object]]:
    """
    Fit training-only POD, create split pairs, and normalize all arrays.
    """
    config.validate()

    n_trajectories, n_times, n_grid = trajectories.shape
    split_bounds = compute_split_bounds(n_times)
    sensor_indices, sensor_positions = choose_uniform_periodic_sensors(
        n_grid,
        domain_length=config.domain_length,
    )

    pod = fit_training_pod(
        trajectories,
        split_bounds.train,
        config.energy_threshold,
    )

    raw_splits = {
        split_name: build_split_pairs(
            trajectories,
            split_name=split_name,
            split_bounds=bounds,
            sensor_indices=sensor_indices,
            pod=pod,
            config=config,
        )
        for split_name, bounds in split_bounds.as_dict().items()
    }

    y_standardizer = fit_standardizer(
        raw_splits["train"]["Y"],
        epsilon=config.normalization_epsilon,
    )
    u_standardizer = fit_standardizer(
        raw_splits["train"]["U"],
        epsilon=config.normalization_epsilon,
    )

    output: Dict[str, Array] = {
        "Y_train": y_standardizer.transform(
            raw_splits["train"]["Y"]
        ).astype(np.float32),
        "U_train": u_standardizer.transform(
            raw_splits["train"]["U"]
        ).astype(np.float32),
        "Y_val": y_standardizer.transform(
            raw_splits["validation"]["Y"]
        ).astype(np.float32),
        "U_val": u_standardizer.transform(
            raw_splits["validation"]["U"]
        ).astype(np.float32),
        "Y_test": y_standardizer.transform(
            raw_splits["test"]["Y"]
        ).astype(np.float32),
        "U_test": u_standardizer.transform(
            raw_splits["test"]["U"]
        ).astype(np.float32),
        "mu_Y": y_standardizer.mean,
        "sigma_Y": y_standardizer.scale,
        "mu_U": u_standardizer.mean,
        "sigma_U": u_standardizer.scale,
        "regularized_Y_channels": y_standardizer.regularized_mask,
        "regularized_U_channels": u_standardizer.regularized_mask,
        "sensor_indices": sensor_indices,
        "sensor_positions": sensor_positions.astype(np.float64),
        "pod_mean_field": pod.mean_field,
        "pod_modes": pod.modes,
        "pod_singular_values": pod.singular_values,
        "pod_energy_fraction": pod.energy_fraction,
        "pod_cumulative_energy": pod.cumulative_energy,
        "pod_num_modes": np.asarray(pod.num_modes, dtype=np.int64),
        "m": np.asarray(config.m, dtype=np.int64),
        "tau": np.asarray(config.tau, dtype=np.int64),
        "gap": np.asarray(config.gap, dtype=np.int64),
        "energy_threshold": np.asarray(
            config.energy_threshold,
            dtype=np.float64,
        ),
        "domain_length": np.asarray(
            config.domain_length,
            dtype=np.float64,
        ),
        "train_time_bounds": np.asarray(
            split_bounds.train,
            dtype=np.int64,
        ),
        "validation_time_bounds": np.asarray(
            split_bounds.validation,
            dtype=np.int64,
        ),
        "test_time_bounds": np.asarray(
            split_bounds.test,
            dtype=np.int64,
        ),
    }

    provenance_prefixes = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for split_name, output_prefix in provenance_prefixes.items():
        split = raw_splits[split_name]
        output[f"trajectory_index_{output_prefix}"] = split[
            "trajectory_index"
        ]
        output[f"anchor_index_{output_prefix}"] = split[
            "anchor_index"
        ]
        output[f"window_start_{output_prefix}"] = split[
            "window_start"
        ]
        output[f"window_end_{output_prefix}"] = split[
            "window_end"
        ]

    retained_energy = float(
        pod.cumulative_energy[pod.num_modes - 1]
    )

    metadata: Dict[str, object] = {
        "input_shape": [
            int(n_trajectories),
            int(n_times),
            int(n_grid),
        ],
        "domain_length": float(config.domain_length),
        "num_sensors": NUM_SENSORS,
        "sensor_indices": sensor_indices.tolist(),
        "sensor_positions": sensor_positions.tolist(),
        "embedding_dimension_m": int(config.m),
        "time_delay_tau_indices": int(config.tau),
        "gap_indices": int(config.gap),
        "delay_window_span_indices": int(config.embedding_span),
        "anchor_stride_indices": int(config.anchor_stride),
        "condition_dimension": int(NUM_SENSORS * config.m),
        "target": "leading POD coefficients at the anchor time",
        "target_dimension": int(pod.num_modes),
        "pod_energy_threshold": float(config.energy_threshold),
        "pod_retained_energy": retained_energy,
        "pod_fit_data": (
            "all snapshots from the first 70 percent of every trajectory"
        ),
        "split_rule": (
            "first 70 percent train, next 15 percent validation, "
            "final 15 percent test, independently within every trajectory"
        ),
        "normalization": (
            "componentwise statistics fitted from training pairs only"
        ),
        "num_train_pairs": int(output["Y_train"].shape[0]),
        "num_validation_pairs": int(output["Y_val"].shape[0]),
        "num_test_pairs": int(output["Y_test"].shape[0]),
    }

    output["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )

    return output, metadata


def save_prepared_dataset(
    output_path: str | Path,
    arrays: Mapping[str, Array],
) -> Path:
    """Save all required arrays and metadata in a compressed NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create normalized four-sensor delay/POD datasets from an "
            "(N,T,D) stochastic KS trajectory array."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("test.npy"),
        help="Input trajectory file with shape (N,T,D).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ks_pod_delay_data.npz"),
        help="Output NPZ file for train_drift_with_visualization.py.",
    )
    parser.add_argument(
        "--energy-plot",
        type=Path,
        default=None,
        help=(
            "POD cumulative-energy PNG path. By default, a PNG is created "
            "beside the output NPZ."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Embedding dimension: number of sensor observations per window.",
    )
    parser.add_argument(
        "--tau",
        type=int,
        required=True,
        help="Time delay in stored time-index units.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        required=True,
        help=(
            "Number of unused time indices between consecutive delay-window "
            "spans."
        ),
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.99,
        help="Cumulative POD energy threshold.",
    )
    parser.add_argument(
        "--normalization-epsilon",
        type=float,
        default=1.0e-8,
        help="Standard deviations below this value are replaced by 1.",
    )
    parser.add_argument(
        "--domain-length",
        type=float,
        default=22.0,
        help="Periodic KS domain length.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PairConfig(
        m=args.m,
        tau=args.tau,
        gap=args.gap,
        energy_threshold=args.energy_threshold,
        normalization_epsilon=args.normalization_epsilon,
        domain_length=args.domain_length,
    )

    trajectories = load_trajectories(args.input)
    arrays, metadata = prepare_dataset(
        trajectories,
        config=config,
    )

    output_path = save_prepared_dataset(
        args.output,
        arrays,
    )

    energy_plot_path = (
        args.energy_plot
        if args.energy_plot is not None
        else output_path.with_name(
            f"{output_path.stem}_pod_energy.png"
        )
    )

    pod = PODModel(
        mean_field=arrays["pod_mean_field"],
        modes=arrays["pod_modes"],
        singular_values=arrays["pod_singular_values"],
        energy_fraction=arrays["pod_energy_fraction"],
        cumulative_energy=arrays["pod_cumulative_energy"],
        num_modes=int(arrays["pod_num_modes"]),
    )

    energy_plot_path = save_pod_energy_plot(
        pod,
        threshold=config.energy_threshold,
        output_path=energy_plot_path,
    )

    summary = {
        "output_npz": str(output_path),
        "pod_energy_plot": str(energy_plot_path),
        **metadata,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
