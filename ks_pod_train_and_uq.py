#!/usr/bin/env python3
"""
End-to-end stochastic KS pipeline:

1. Load trajectory data from test.npy with shape (N,T,D).
2. Split every trajectory chronologically into 70% train, 15% validation,
   and 15% test portions.
3. Fit a POD basis using training snapshots only.
4. Retain the minimum number of POD modes that capture a specified fraction
   of training energy; the default is 99%.
5. Form non-overlapping delay-coordinate windows from four uniformly spaced
   spatial sensors.
6. Use the POD coefficients at each window's anchor time as the target.
7. Normalize conditions and targets using training-pair statistics only.
8. Save an NPZ file compatible with train_drift_with_visualization.py.
9. Train the conditional stochastic-interpolant drift by calling
   train_conditional_drift from train_drift_with_visualization.py.
10. Generate conditional POD-coefficient ensembles with the Euler--Maruyama
    implementation in sampling.py.
11. Save the test inputs, true POD coefficients, and generated ensembles.
12. Create uncertainty-quantification plots for POD coefficient reconstruction.

Required neighboring files
--------------------------
Place this file in the same directory as

    train_drift_with_visualization.py
    sampling.py

Input
-----
The input array must have shape

    (N,T,D),

where N is the number of trajectories, T is the number of stored time indices,
and D is the number of spatial grid points. The periodic spatial domain length
defaults to L=22.

Delay condition
---------------
Let P extract four uniformly spaced sensors. For embedding dimension m and
integer delay tau, the physical conditioning vector at anchor time t is

    Y_t = [
        P(X_t),
        P(X_{t-tau}),
        ...,
        P(X_{t-(m-1)tau})
    ] in R^(4m).

The target is the vector of retained POD coefficients at the anchor time:

    U_t = a(t) in R^r.

The delay blocks are flattened in current-to-past order.

Non-overlapping windows
-----------------------
The temporal span of one delay window is

    [t-(m-1)tau, t].

Within each trajectory and each chronological split, selected spans do not
overlap. If one span ends at index e, the next span begins at index

    e + gap + 1,

so exactly ``gap`` unused time indices separate consecutive spans.

Outputs
-------
The output directory contains, among other files:

    ks_pod_delay_data.npz
    pod_cumulative_energy.png
    training/conditional_drift_checkpoint.pt
    training/training_validation_loss_log.png
    training/network_architecture.png
    uq/conditional_pod_samples.npz
    uq/pod_coefficients_example_*.png
    uq/coefficient_coverage_<level>.png
    uq/coefficient_interval_width_<level>.png
    uq/calibration_curve.png
    pipeline_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampling import (
    SamplingConfig,
    denormalize_targets,
    load_trained_drift,
    sample_normalized_conditional_em,
)
from train_drift_with_visualization import (
    InterpolantConfig,
    TrainConfig,
    set_seed,
    train_conditional_drift,
)


Array = np.ndarray
NUM_SENSORS = 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Data construction, POD, and normalization configuration."""

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
        """Number of time steps from the oldest delay to the anchor."""
        return (self.m - 1) * self.tau

    @property
    def window_span_length(self) -> int:
        """Number of indices in the inclusive delay-window span."""
        return self.embedding_span + 1

    @property
    def anchor_stride(self) -> int:
        """
        Distance between consecutive selected anchor indices.

        This gives non-overlapping full spans and exactly ``gap`` unused
        indices between consecutive spans.
        """
        return self.window_span_length + self.gap


@dataclass(frozen=True)
class UQConfig:
    """Conditional-ensemble and UQ-plot configuration."""

    num_test_conditions: int = 12
    ensemble_size: int = 500
    em_steps: int = 500
    sample_batch_size: int = 512
    interval_level: float = 0.90
    seed: int = 101

    def validate(self) -> None:
        if self.num_test_conditions < 1:
            raise ValueError("num_test_conditions must be positive.")
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2.")
        if self.em_steps < 1:
            raise ValueError("em_steps must be positive.")
        if self.sample_batch_size < 1:
            raise ValueError("sample_batch_size must be positive.")
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError("interval_level must lie in (0,1).")


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
    """Componentwise affine standardizer."""

    mean: Array
    scale: Array
    regularized_mask: Array

    def transform(self, values: Array) -> Array:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: Array) -> Array:
        return self.mean + self.scale * values


@dataclass(frozen=True)
class PODModel:
    """POD model fitted from training snapshots only."""

    mean_field: Array
    modes: Array
    singular_values: Array
    energy_fraction: Array
    cumulative_energy: Array
    num_modes: int

    def coefficients(self, snapshots: Array) -> Array:
        snapshots = np.asarray(snapshots, dtype=np.float32)
        centered = snapshots - self.mean_field
        return centered @ self.modes.T

    def reconstruct(self, coefficients: Array) -> Array:
        coefficients = np.asarray(coefficients, dtype=np.float32)
        return self.mean_field + coefficients @ self.modes


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


# ---------------------------------------------------------------------------
# Data loading and chronological splitting
# ---------------------------------------------------------------------------

def load_trajectories(path: str | Path) -> Array:
    """Load and validate a finite numerical array with shape (N,T,D)."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input trajectory file not found: {path}")

    trajectories = np.asarray(
        np.load(path, allow_pickle=False)
    )

    if trajectories.ndim != 3:
        raise ValueError(
            "The input must have shape (N,T,D); "
            f"received {trajectories.shape}."
        )

    n_trajectories, n_times, n_grid = trajectories.shape

    if n_trajectories < 1:
        raise ValueError("N must be positive.")
    if n_times < 3:
        raise ValueError("T must be at least 3.")
    if n_grid < NUM_SENSORS:
        raise ValueError(
            f"D must be at least {NUM_SENSORS}."
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
    Compute the per-trajectory 70%/15%/15% chronological partition.
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
    Choose four uniformly spaced sensors on a periodic grid.

    The selected indices are floor(kD/4), k=0,1,2,3. The corresponding
    physical positions are L*j/D.
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


# ---------------------------------------------------------------------------
# Training-only POD
# ---------------------------------------------------------------------------

def fit_training_pod(
    trajectories: Array,
    train_bounds: Tuple[int, int],
    *,
    energy_threshold: float,
) -> PODModel:
    """
    Fit the POD basis using every snapshot in the training portions only.
    """
    train_start, train_end = train_bounds

    snapshots = trajectories[:, train_start:train_end, :].reshape(
        -1,
        trajectories.shape[2],
    )
    snapshots64 = np.asarray(snapshots, dtype=np.float64)

    if snapshots64.shape[0] < 2:
        raise ValueError("At least two training snapshots are required.")

    mean_field = snapshots64.mean(axis=0)
    centered = snapshots64 - mean_field

    _, singular_values, right_singular_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    squared_values = singular_values**2
    total_energy = squared_values.sum()

    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError(
            "The centered training snapshots have zero or invalid energy."
        )

    energy_fraction = squared_values / total_energy
    cumulative_energy = np.cumsum(energy_fraction)

    num_modes = int(
        np.searchsorted(
            cumulative_energy,
            energy_threshold,
            side="left",
        )
        + 1
    )

    return PODModel(
        mean_field=mean_field.astype(np.float32),
        modes=right_singular_vectors[:num_modes].astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        energy_fraction=energy_fraction.astype(np.float64),
        cumulative_energy=cumulative_energy.astype(np.float64),
        num_modes=num_modes,
    )


def plot_pod_energy(
    pod: PODModel,
    *,
    threshold: float,
    output_path: str | Path,
) -> Path:
    """Save a cumulative POD-energy verification plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modes = np.arange(
        1,
        pod.cumulative_energy.size + 1,
        dtype=np.int64,
    )
    retained_energy = pod.cumulative_energy[pod.num_modes - 1]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        modes,
        pod.cumulative_energy,
        label="Cumulative POD energy",
    )
    axis.axhline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.1%}",
    )
    axis.axvline(
        pod.num_modes,
        linestyle="--",
        label=f"Selected rank r = {pod.num_modes}",
    )
    axis.scatter(
        [pod.num_modes],
        [retained_energy],
        label=f"Retained energy = {retained_energy:.3%}",
    )
    axis.set_xlabel("Number of retained POD modes")
    axis.set_ylabel("Cumulative energy fraction")
    axis.set_title(
        "Training-only POD energy selection\n"
        f"r={pod.num_modes} modes retain {retained_energy:.3%}"
    )
    axis.set_ylim(0.0, 1.01)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------------
# Delay-window construction
# ---------------------------------------------------------------------------

def anchor_indices_for_split(
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> Array:
    """
    Return anchors whose complete delay spans lie inside one split.

    The first selected window starts at the split boundary. The anchor stride
    makes complete temporal spans non-overlapping with the requested gap.
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


def verify_delay_windows(
    anchors: Array,
    *,
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> None:
    """Verify split containment and non-overlap of selected delay spans."""
    if anchors.size == 0:
        return

    starts = anchors - config.embedding_span
    ends = anchors

    if starts[0] < split_start or ends[-1] >= split_end:
        raise RuntimeError("A delay window crosses a split boundary.")

    if anchors.size > 1:
        unused_indices = starts[1:] - ends[:-1] - 1
        if np.any(unused_indices != config.gap):
            raise RuntimeError(
                "Delay windows do not have the requested exact gap."
            )


def build_split_pairs(
    trajectories: Array,
    *,
    split_name: str,
    bounds: Tuple[int, int],
    sensor_indices: Array,
    pod: PODModel,
    config: DataConfig,
) -> Dict[str, Array]:
    """
    Construct raw four-sensor delay conditions and POD targets.
    """
    split_start, split_end = bounds

    anchors = anchor_indices_for_split(
        split_start,
        split_end,
        config,
    )
    verify_delay_windows(
        anchors,
        split_start=split_start,
        split_end=split_end,
        config=config,
    )

    condition_dim = NUM_SENSORS * config.m
    delay_offsets = (
        np.arange(config.m, dtype=np.int64) * config.tau
    )

    buffer = PairBuffer.empty()

    for trajectory_index in range(trajectories.shape[0]):
        for anchor_index in anchors:
            delay_indices = anchor_index - delay_offsets

            # np.ix_ avoids advanced-index broadcasting ambiguity.
            sensor_history = trajectories[
                trajectory_index
            ][np.ix_(delay_indices, sensor_indices)]

            y = sensor_history.reshape(condition_dim)

            snapshot = trajectories[
                trajectory_index,
                anchor_index,
                :,
            ]
            u = pod.coefficients(snapshot[None, :])[0]

            buffer.append(
                y=y,
                u=u,
                trajectory_index=trajectory_index,
                anchor_index=int(anchor_index),
                window_start=int(delay_indices[-1]),
                window_end=int(anchor_index),
            )

    return buffer.finalize(
        split_name=split_name,
        condition_dim=condition_dim,
        target_dim=pod.num_modes,
    )


# ---------------------------------------------------------------------------
# Normalization and NPZ preparation
# ---------------------------------------------------------------------------

def fit_standardizer(
    values: Array,
    *,
    epsilon: float,
) -> Standardizer:
    """Fit componentwise training mean and population standard deviation."""
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


def prepare_ks_pod_data(
    trajectories: Array,
    *,
    config: DataConfig,
) -> Dict[str, Any]:
    """
    Fit training-only POD, construct raw split pairs, and normalize them.
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
        energy_threshold=config.energy_threshold,
    )

    raw_splits = {
        name: build_split_pairs(
            trajectories,
            split_name=name,
            bounds=bounds,
            sensor_indices=sensor_indices,
            pod=pod,
            config=config,
        )
        for name, bounds in split_bounds.as_dict().items()
    }

    y_standardizer = fit_standardizer(
        raw_splits["train"]["Y"],
        epsilon=config.normalization_epsilon,
    )
    u_standardizer = fit_standardizer(
        raw_splits["train"]["U"],
        epsilon=config.normalization_epsilon,
    )

    normalized_splits = {
        name: {
            "Y": y_standardizer.transform(split["Y"]).astype(np.float32),
            "U": u_standardizer.transform(split["U"]).astype(np.float32),
        }
        for name, split in raw_splits.items()
    }

    return {
        "split_bounds": split_bounds,
        "sensor_indices": sensor_indices,
        "sensor_positions": sensor_positions,
        "pod": pod,
        "raw_splits": raw_splits,
        "normalized_splits": normalized_splits,
        "y_standardizer": y_standardizer,
        "u_standardizer": u_standardizer,
        "input_shape": (
            n_trajectories,
            n_times,
            n_grid,
        ),
    }


def dataset_arrays_for_npz(
    prepared: Mapping[str, Any],
    *,
    config: DataConfig,
) -> Dict[str, Array]:
    """Assemble training-compatible arrays and additional metadata."""
    normalized = prepared["normalized_splits"]
    raw = prepared["raw_splits"]
    split_bounds: SplitBounds = prepared["split_bounds"]
    pod: PODModel = prepared["pod"]
    y_standardizer: Standardizer = prepared["y_standardizer"]
    u_standardizer: Standardizer = prepared["u_standardizer"]

    arrays: Dict[str, Array] = {
        "Y_train": normalized["train"]["Y"],
        "U_train": normalized["train"]["U"],
        "Y_val": normalized["validation"]["Y"],
        "U_val": normalized["validation"]["U"],
        "Y_test": normalized["test"]["Y"],
        "U_test": normalized["test"]["U"],
        "mu_Y": y_standardizer.mean,
        "sigma_Y": y_standardizer.scale,
        "mu_U": u_standardizer.mean,
        "sigma_U": u_standardizer.scale,
        "Y_train_physical": raw["train"]["Y"],
        "U_train_physical": raw["train"]["U"],
        "Y_val_physical": raw["validation"]["Y"],
        "U_val_physical": raw["validation"]["U"],
        "Y_test_physical": raw["test"]["Y"],
        "U_test_physical": raw["test"]["U"],
        "regularized_Y_channels": y_standardizer.regularized_mask,
        "regularized_U_channels": u_standardizer.regularized_mask,
        "sensor_indices": prepared["sensor_indices"],
        "sensor_positions": prepared["sensor_positions"].astype(np.float64),
        "pod_mean_field": pod.mean_field,
        "pod_modes": pod.modes,
        "pod_singular_values": pod.singular_values,
        "pod_energy_fraction": pod.energy_fraction,
        "pod_cumulative_energy": pod.cumulative_energy,
        "pod_num_modes": np.asarray(pod.num_modes, dtype=np.int64),
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
        "m": np.asarray(config.m, dtype=np.int64),
        "tau": np.asarray(config.tau, dtype=np.int64),
        "gap": np.asarray(config.gap, dtype=np.int64),
        "domain_length": np.asarray(
            config.domain_length,
            dtype=np.float64,
        ),
        "energy_threshold": np.asarray(
            config.energy_threshold,
            dtype=np.float64,
        ),
    }

    split_prefixes = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for name, prefix in split_prefixes.items():
        arrays[f"trajectory_index_{prefix}"] = raw[name][
            "trajectory_index"
        ]
        arrays[f"anchor_index_{prefix}"] = raw[name][
            "anchor_index"
        ]
        arrays[f"window_start_{prefix}"] = raw[name][
            "window_start"
        ]
        arrays[f"window_end_{prefix}"] = raw[name][
            "window_end"
        ]

    metadata = {
        "input_shape": [
            int(value)
            for value in prepared["input_shape"]
        ],
        "domain_length": float(config.domain_length),
        "num_sensors": NUM_SENSORS,
        "sensor_indices": prepared["sensor_indices"].tolist(),
        "sensor_positions": prepared["sensor_positions"].tolist(),
        "embedding_dimension_m": int(config.m),
        "time_delay_tau_indices": int(config.tau),
        "gap_indices": int(config.gap),
        "condition_dimension": int(NUM_SENSORS * config.m),
        "target": "leading training-POD coefficients at anchor time",
        "target_dimension": int(pod.num_modes),
        "pod_energy_threshold": float(config.energy_threshold),
        "pod_retained_energy": float(
            pod.cumulative_energy[pod.num_modes - 1]
        ),
        "pod_fit_data": (
            "all snapshots in the first 70 percent of each trajectory"
        ),
        "split_rule": (
            "first 70 percent train, next 15 percent validation, "
            "final 15 percent test within every trajectory"
        ),
        "normalization_rule": (
            "componentwise statistics fitted from raw training pairs only"
        ),
        "num_train_pairs": int(arrays["Y_train"].shape[0]),
        "num_validation_pairs": int(arrays["Y_val"].shape[0]),
        "num_test_pairs": int(arrays["Y_test"].shape[0]),
    }

    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )

    return arrays


def save_dataset_npz(
    output_path: str | Path,
    arrays: Mapping[str, Array],
) -> Path:
    """Save the prepared arrays in a compressed NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_from_prepared_data(
    prepared: Mapping[str, Any],
    *,
    output_dir: str | Path,
    train_config: TrainConfig,
    interpolant_config: InterpolantConfig,
) -> Dict[str, Any]:
    """Call train_drift_with_visualization.py on the prepared arrays."""
    normalized = prepared["normalized_splits"]
    y_standardizer: Standardizer = prepared["y_standardizer"]
    u_standardizer: Standardizer = prepared["u_standardizer"]

    normalization_stats = {
        "mu_Y": y_standardizer.mean,
        "sigma_Y": y_standardizer.scale,
        "mu_U": u_standardizer.mean,
        "sigma_U": u_standardizer.scale,
    }

    return train_conditional_drift(
        normalized["train"]["Y"],
        normalized["train"]["U"],
        normalized["validation"]["Y"],
        normalized["validation"]["U"],
        normalized["test"]["Y"],
        normalized["test"]["U"],
        output_dir,
        train_config=train_config,
        interpolant_config=interpolant_config,
        normalization_stats=normalization_stats,
    )


# ---------------------------------------------------------------------------
# Conditional sampling and UQ diagnostics
# ---------------------------------------------------------------------------

def select_evenly_spaced_indices(
    population_size: int,
    requested_count: int,
) -> Array:
    """Select deterministic indices spanning the full test-pair ordering."""
    if population_size < 1:
        raise ValueError("population_size must be positive.")

    count = min(population_size, requested_count)

    if count == population_size:
        return np.arange(population_size, dtype=np.int64)

    indices = np.linspace(
        0,
        population_size - 1,
        count,
        dtype=np.int64,
    )
    return np.unique(indices)


def generate_test_ensembles(
    prepared: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    uq_config: UQConfig,
    device_name: str,
) -> Dict[str, Any]:
    """
    Generate POD-coefficient ensembles for selected test conditions.

    This function uses the model-loading and Euler--Maruyama sampling
    implementation from sampling.py.
    """
    uq_config.validate()

    (
        model,
        interpolant_config,
        normalization_stats,
        checkpoint,
        device,
    ) = load_trained_drift(
        checkpoint_path,
        device_name=device_name,
    )

    normalized_test = prepared["normalized_splits"]["test"]
    raw_test = prepared["raw_splits"]["test"]

    selected_indices = select_evenly_spaced_indices(
        normalized_test["Y"].shape[0],
        uq_config.num_test_conditions,
    )

    normalized_ensembles: List[Array] = []
    physical_ensembles: List[Array] = []

    for order, test_index in enumerate(selected_indices):
        # Use a different reproducible random stream for each condition.
        set_seed(uq_config.seed + order)

        sampling_config = SamplingConfig(
            num_samples=uq_config.ensemble_size,
            num_steps=uq_config.em_steps,
            sample_batch_size=uq_config.sample_batch_size,
            seed=uq_config.seed + order,
            device=device_name,
        )

        normalized_samples = sample_normalized_conditional_em(
            model,
            normalized_test["Y"][test_index],
            interpolant_config,
            sampling_config,
            device,
        )
        physical_samples = denormalize_targets(
            normalized_samples,
            normalization_stats,
        )

        normalized_ensembles.append(normalized_samples)
        physical_ensembles.append(physical_samples)

    return {
        "selected_test_indices": selected_indices,
        "input_y": raw_test["Y"][selected_indices],
        "normalized_input_y": normalized_test["Y"][selected_indices],
        "true_coefficients": raw_test["U"][selected_indices],
        "normalized_true_coefficients": normalized_test["U"][
            selected_indices
        ],
        "samples": np.stack(physical_ensembles, axis=0),
        "normalized_samples": np.stack(
            normalized_ensembles,
            axis=0,
        ),
        "trajectory_index": raw_test["trajectory_index"][
            selected_indices
        ],
        "anchor_index": raw_test["anchor_index"][selected_indices],
        "window_start": raw_test["window_start"][selected_indices],
        "window_end": raw_test["window_end"][selected_indices],
        "checkpoint": checkpoint,
        "interpolant_config": interpolant_config,
        "device": device,
    }


def central_interval(
    samples: Array,
    level: float,
) -> Tuple[Array, Array]:
    """Compute a componentwise central empirical interval."""
    alpha = 1.0 - level
    lower = np.quantile(samples, alpha / 2.0, axis=-2)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=-2)
    return lower, upper


def save_uq_samples(
    output_path: str | Path,
    ensembles: Mapping[str, Any],
    *,
    uq_config: UQConfig,
) -> Path:
    """Save physical inputs, truths, ensembles, and provenance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interpolant_config: InterpolantConfig = ensembles[
        "interpolant_config"
    ]

    metadata = {
        "num_test_conditions": int(
            ensembles["selected_test_indices"].size
        ),
        "ensemble_size": int(uq_config.ensemble_size),
        "euler_maruyama_steps": int(uq_config.em_steps),
        "interval_level": float(uq_config.interval_level),
        "seed": int(uq_config.seed),
        "diffusion_schedule": "rho(s) = sigma_I * (1 - s)",
        "sigma_I": float(interpolant_config.sigma_I),
    }

    np.savez_compressed(
        output_path,
        selected_test_indices=ensembles["selected_test_indices"],
        input_y=ensembles["input_y"],
        normalized_input_y=ensembles["normalized_input_y"],
        true_coefficients=ensembles["true_coefficients"],
        normalized_true_coefficients=ensembles[
            "normalized_true_coefficients"
        ],
        samples=ensembles["samples"],
        normalized_samples=ensembles["normalized_samples"],
        trajectory_index=ensembles["trajectory_index"],
        anchor_index=ensembles["anchor_index"],
        window_start=ensembles["window_start"],
        window_end=ensembles["window_end"],
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )

    return output_path


def plot_example_coefficient_intervals(
    ensembles: Mapping[str, Any],
    *,
    interval_level: float,
    output_dir: str | Path,
) -> List[Path]:
    """
    Plot truth, ensemble mean, and a central interval for each test condition.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = ensembles["samples"]
    truths = ensembles["true_coefficients"]
    lower, upper = central_interval(samples, interval_level)
    means = samples.mean(axis=1)

    paths: List[Path] = []

    for example_index in range(samples.shape[0]):
        coefficient_index = np.arange(
            1,
            samples.shape[2] + 1,
            dtype=np.int64,
        )

        figure, axis = plt.subplots(figsize=(10, 5.8))
        axis.fill_between(
            coefficient_index,
            lower[example_index],
            upper[example_index],
            alpha=0.25,
            label=f"{interval_level:.0%} empirical interval",
        )
        axis.plot(
            coefficient_index,
            means[example_index],
            marker="o",
            label="Ensemble mean",
        )
        axis.plot(
            coefficient_index,
            truths[example_index],
            marker="x",
            linestyle="--",
            label="True POD coefficients",
        )
        axis.set_xlabel("POD coefficient index")
        axis.set_ylabel("POD coefficient value")
        axis.set_title(
            "Conditional POD-coefficient reconstruction\n"
            f"trajectory={int(ensembles['trajectory_index'][example_index])}, "
            f"anchor={int(ensembles['anchor_index'][example_index])}"
        )
        axis.grid(True)
        axis.legend()
        figure.tight_layout()

        path = output_dir / (
            f"pod_coefficients_example_{example_index:03d}.png"
        )
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    return paths


def plot_coefficient_coverage(
    ensembles: Mapping[str, Any],
    *,
    interval_level: float,
    output_path: str | Path,
) -> Tuple[Path, Array]:
    """Plot empirical central-interval coverage for each POD coefficient."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = ensembles["samples"]
    truths = ensembles["true_coefficients"]
    lower, upper = central_interval(samples, interval_level)

    covered = (truths >= lower) & (truths <= upper)
    coverage = covered.mean(axis=0)

    coefficient_index = np.arange(
        1,
        coverage.size + 1,
        dtype=np.int64,
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        coefficient_index,
        coverage,
        marker="o",
        label="Empirical coefficient-wise coverage",
    )
    axis.axhline(
        interval_level,
        linestyle="--",
        label=f"Nominal coverage = {interval_level:.0%}",
    )
    axis.set_xlabel("POD coefficient index")
    axis.set_ylabel("Empirical coverage")
    axis.set_title(
        f"Coefficient-wise {interval_level:.0%} interval coverage\n"
        f"evaluated on {samples.shape[0]} test conditions"
    )
    axis.set_ylim(0.0, 1.02)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path, coverage


def plot_coefficient_interval_width(
    ensembles: Mapping[str, Any],
    *,
    interval_level: float,
    output_path: str | Path,
) -> Tuple[Path, Array]:
    """Plot average central-interval width for each POD coefficient."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lower, upper = central_interval(
        ensembles["samples"],
        interval_level,
    )
    mean_width = (upper - lower).mean(axis=0)

    coefficient_index = np.arange(
        1,
        mean_width.size + 1,
        dtype=np.int64,
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        coefficient_index,
        mean_width,
        marker="o",
        label=f"Mean {interval_level:.0%} interval width",
    )
    axis.set_xlabel("POD coefficient index")
    axis.set_ylabel("Average interval width")
    axis.set_title(
        "Conditional-ensemble sharpness by POD coefficient"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path, mean_width


def compute_calibration_curve(
    samples: Array,
    truths: Array,
    nominal_levels: Array,
) -> Array:
    """Compute overall empirical coverage for several central intervals."""
    empirical = np.empty_like(nominal_levels, dtype=np.float64)

    for index, level in enumerate(nominal_levels):
        lower, upper = central_interval(samples, float(level))
        empirical[index] = np.mean(
            (truths >= lower) & (truths <= upper)
        )

    return empirical


def plot_calibration_curve(
    ensembles: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> Tuple[Path, Array, Array]:
    """Plot nominal central-interval coverage against empirical coverage."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nominal_levels = np.asarray(
        [0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        dtype=np.float64,
    )
    empirical_levels = compute_calibration_curve(
        ensembles["samples"],
        ensembles["true_coefficients"],
        nominal_levels,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        nominal_levels,
        empirical_levels,
        marker="o",
        label="Empirical coverage",
    )
    axis.plot(
        nominal_levels,
        nominal_levels,
        linestyle="--",
        label="Ideal calibration",
    )
    axis.set_xlabel("Nominal central-interval coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_title(
        "POD-coefficient conditional-ensemble calibration"
    )
    axis.set_xlim(0.45, 1.0)
    axis.set_ylim(0.45, 1.0)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path, nominal_levels, empirical_levels


def create_uq_visualizations(
    ensembles: Mapping[str, Any],
    *,
    uq_config: UQConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Create POD-coefficient reconstruction and calibration figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example_paths = plot_example_coefficient_intervals(
        ensembles,
        interval_level=uq_config.interval_level,
        output_dir=output_dir / "examples",
    )

    level_label = int(round(100.0 * uq_config.interval_level))

    coverage_path, coefficient_coverage = plot_coefficient_coverage(
        ensembles,
        interval_level=uq_config.interval_level,
        output_path=(
            output_dir
            / f"coefficient_coverage_{level_label}.png"
        ),
    )

    width_path, coefficient_width = plot_coefficient_interval_width(
        ensembles,
        interval_level=uq_config.interval_level,
        output_path=(
            output_dir
            / f"coefficient_interval_width_{level_label}.png"
        ),
    )

    (
        calibration_path,
        nominal_levels,
        empirical_levels,
    ) = plot_calibration_curve(
        ensembles,
        output_path=output_dir / "calibration_curve.png",
    )

    metrics_path = output_dir / "uq_metrics.npz"
    np.savez_compressed(
        metrics_path,
        coefficient_coverage=coefficient_coverage,
        coefficient_interval_width=coefficient_width,
        nominal_levels=nominal_levels,
        empirical_levels=empirical_levels,
    )

    return {
        "example_plots": [str(path) for path in example_paths],
        "coverage_plot": str(coverage_path),
        "interval_width_plot": str(width_path),
        "calibration_plot": str(calibration_path),
        "metrics_file": str(metrics_path),
        "mean_coverage": float(coefficient_coverage.mean()),
        "mean_interval_width": float(coefficient_width.mean()),
    }


# ---------------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
    interpolant_config: InterpolantConfig,
    uq_config: UQConfig,
) -> Dict[str, Any]:
    """Run preparation, training, conditional sampling, and UQ plotting."""
    data_config.validate()
    train_config.validate()
    interpolant_config.validate()
    uq_config.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(input_path)

    prepared = prepare_ks_pod_data(
        trajectories,
        config=data_config,
    )

    dataset_arrays = dataset_arrays_for_npz(
        prepared,
        config=data_config,
    )

    dataset_path = save_dataset_npz(
        output_dir / "ks_pod_delay_data.npz",
        dataset_arrays,
    )

    pod: PODModel = prepared["pod"]
    pod_plot_path = plot_pod_energy(
        pod,
        threshold=data_config.energy_threshold,
        output_path=output_dir / "pod_cumulative_energy.png",
    )

    training_dir = output_dir / "training"
    training_result = train_from_prepared_data(
        prepared,
        output_dir=training_dir,
        train_config=train_config,
        interpolant_config=interpolant_config,
    )

    checkpoint_path = Path(
        training_result["summary"]["checkpoint"]
    )

    ensembles = generate_test_ensembles(
        prepared,
        checkpoint_path=checkpoint_path,
        uq_config=uq_config,
        device_name=train_config.device,
    )

    uq_dir = output_dir / "uq"
    uq_samples_path = save_uq_samples(
        uq_dir / "conditional_pod_samples.npz",
        ensembles,
        uq_config=uq_config,
    )

    uq_visualizations = create_uq_visualizations(
        ensembles,
        uq_config=uq_config,
        output_dir=uq_dir,
    )

    summary = {
        "input_file": str(Path(input_path)),
        "output_directory": str(output_dir),
        "dataset_file": str(dataset_path),
        "pod_energy_plot": str(pod_plot_path),
        "checkpoint": str(checkpoint_path),
        "training_summary": training_result["summary"],
        "uq_samples_file": str(uq_samples_path),
        "uq_visualizations": uq_visualizations,
        "data_config": asdict(data_config),
        "train_config": asdict(train_config),
        "interpolant_config": asdict(interpolant_config),
        "uq_config": asdict(uq_config),
        "input_shape": [
            int(value)
            for value in prepared["input_shape"]
        ],
        "condition_dimension": int(NUM_SENSORS * data_config.m),
        "pod_target_dimension": int(pod.num_modes),
        "pod_retained_energy": float(
            pod.cumulative_energy[pod.num_modes - 1]
        ),
        "num_train_pairs": int(
            prepared["normalized_splits"]["train"]["Y"].shape[0]
        ),
        "num_validation_pairs": int(
            prepared["normalized_splits"]["validation"]["Y"].shape[0]
        ),
        "num_test_pairs": int(
            prepared["normalized_splits"]["test"]["Y"].shape[0]
        ),
    }

    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    summary["pipeline_summary_file"] = str(summary_path)
    return summary


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    """Parse comma-separated hidden-layer widths."""
    try:
        dimensions = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Hidden dimensions must be comma-separated integers."
        ) from error

    if not dimensions or any(width < 1 for width in dimensions):
        raise argparse.ArgumentTypeError(
            "Every hidden-layer width must be positive."
        )

    return dimensions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare stochastic KS delay/POD data, train the conditional "
            "drift, sample test-condition ensembles, and create UQ plots."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("test.npy"),
        help="Input trajectory array with shape (N,T,D).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modularized_implementation/ks_pod_train_uq_output"),
    )

    # Data construction.
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--tau", type=int, required=True)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--normalization-epsilon",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--domain-length",
        type=float,
        default=22.0,
    )

    # Drift network and optimization.
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(256, 256, 256, 256),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.0e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument(
        "--validation-mc-repeats",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--test-mc-repeats",
        type=int,
        default=16,
    )
    parser.add_argument("--sigma-I", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print-every", type=int, default=25)

    # Conditional sampling and UQ.
    parser.add_argument(
        "--num-uq-conditions",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=500,
    )
    parser.add_argument("--em-steps", type=int, default=500)
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--uq-seed",
        type=int,
        default=101,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(
        m=args.m,
        tau=args.tau,
        gap=args.gap,
        energy_threshold=args.energy_threshold,
        normalization_epsilon=args.normalization_epsilon,
        domain_length=args.domain_length,
    )

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

    uq_config = UQConfig(
        num_test_conditions=args.num_uq_conditions,
        ensemble_size=args.ensemble_size,
        em_steps=args.em_steps,
        sample_batch_size=args.sample_batch_size,
        interval_level=args.interval_level,
        seed=args.uq_seed,
    )

    summary = run_pipeline(
        args.input,
        args.output_dir,
        data_config=data_config,
        train_config=train_config,
        interpolant_config=interpolant_config,
        uq_config=uq_config,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
