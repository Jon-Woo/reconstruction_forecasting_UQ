#!/usr/bin/env python3
"""
Stochastic 2D Kolmogorov-flow POD reconstruction from sparse delayed sensors.

Input
-----
``test.npy`` must contain a finite numerical array with shape

    (N, T, X, Y),

where N is the number of trajectories, T is the number of stored time indices,
and X and Y are the two spatial grid dimensions.

Condition and target
--------------------
Let P extract K = sensors_x * sensors_y sparse sensors placed at the centers of
uniform spatial bins. At anchor index t,

    Y_t = [
        P(X_t),
        P(X_{t-tau}),
        ...,
        P(X_{t-(m-1)tau})
    ] in R^(K*m).

A mean-centered POD basis is fitted using training snapshots only. The target is

    U_t = a(t) in R^r,

where a(t) contains the retained POD coefficients at the anchor time. The rank
r is the smallest rank whose cumulative squared-singular-value energy reaches
the requested threshold, which defaults to 99%.

Chronological splitting
-----------------------
Every trajectory is split independently:

    train:      [0, floor(0.70*T))
    validation: [floor(0.70*T), floor(0.85*T))
    test:       [floor(0.85*T), T).

A delay pair belongs to a split only when its complete history span

    [t-(m-1)tau, t]

lies inside that split.

Non-overlap
-----------
The inclusive delay-window span has length

    (m-1)*tau + 1.

Consecutive selected spans within one trajectory and split are non-overlapping
and have exactly ``gap`` unused time indices between them.

Execution modes
---------------
train:
    Fit the training-only POD basis, prepare and normalize the pairs, train the
    conditional drift, generate conditional coefficient ensembles, and create
    the UQ figure.

inference:
    Reuse a saved prepared NPZ dataset and trained checkpoint. This generates
    new Euler--Maruyama ensembles and new UQ figures without refitting POD,
    rebuilding pairs, or retraining.

plot:
    Reuse an already saved ensemble NPZ and redraw the coefficient-subplot
    figure without retraining or rerunning Euler--Maruyama. This is the fastest
    mode for changing plot appearance.

Required neighboring modules
----------------------------
The canonical filenames are

    train_drift_with_visualization.py
    sampling.py

For convenience, sibling files named

    train_drift_with_visualization(1).py
    sampling(1).py

are also recognized.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------------
# Dependency loading
# ---------------------------------------------------------------------------

def _load_sibling_module(
    module_name: str,
    fallback_filename: str,
):
    """Import a canonical module, falling back to a sibling filename."""
    try:
        return __import__(module_name)
    except ModuleNotFoundError as original_error:
        fallback_path = Path(__file__).resolve().parent / fallback_filename
        if not fallback_path.exists():
            raise ModuleNotFoundError(
                f"Could not import '{module_name}' and fallback file "
                f"'{fallback_path}' does not exist."
            ) from original_error

        spec = importlib.util.spec_from_file_location(
            module_name,
            fallback_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not create an import specification for {fallback_path}."
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


_train_module = _load_sibling_module(
    "train_drift_with_visualization",
    "train_drift_with_visualization(1).py",
)
_sampling_module = _load_sibling_module(
    "sampling",
    "sampling(1).py",
)

InterpolantConfig = _train_module.InterpolantConfig
TrainConfig = _train_module.TrainConfig
set_seed = _train_module.set_seed
train_conditional_drift = _train_module.train_conditional_drift

SamplingConfig = _sampling_module.SamplingConfig
denormalize_targets = _sampling_module.denormalize_targets
load_trained_drift = _sampling_module.load_trained_drift
sample_normalized_conditional_em = (
    _sampling_module.sample_normalized_conditional_em
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Sensor, delay-window, POD, and normalization settings."""

    m: int
    tau: int
    gap: int
    sensors_x: int = 3
    sensors_y: int = 3
    energy_threshold: float = 0.99
    normalization_epsilon: float = 1.0e-8
    domain_length_x: float = 1.0
    domain_length_y: float = 1.0

    def validate(self) -> None:
        if self.m < 1:
            raise ValueError("m must be a positive integer.")
        if self.tau < 1:
            raise ValueError("tau must be a positive integer.")
        if self.gap < 0:
            raise ValueError("gap must be nonnegative.")
        if self.sensors_x < 1 or self.sensors_y < 1:
            raise ValueError(
                "sensors_x and sensors_y must be positive integers."
            )
        if not 0.0 < self.energy_threshold <= 1.0:
            raise ValueError(
                "energy_threshold must lie in (0,1]."
            )
        if self.normalization_epsilon <= 0.0:
            raise ValueError(
                "normalization_epsilon must be positive."
            )
        if self.domain_length_x <= 0.0:
            raise ValueError("domain_length_x must be positive.")
        if self.domain_length_y <= 0.0:
            raise ValueError("domain_length_y must be positive.")

    @property
    def num_sensors(self) -> int:
        return self.sensors_x * self.sensors_y

    @property
    def embedding_span(self) -> int:
        """Distance from the oldest delay to the anchor."""
        return (self.m - 1) * self.tau

    @property
    def window_span_length(self) -> int:
        """Number of stored indices in [t-(m-1)tau,t]."""
        return self.embedding_span + 1

    @property
    def anchor_stride(self) -> int:
        """Stride giving non-overlap and exactly ``gap`` unused indices."""
        return self.window_span_length + self.gap


@dataclass(frozen=True)
class UQConfig:
    """Conditional ensemble and UQ evaluation settings."""

    num_test_conditions: int = 20
    ensemble_size: int = 500
    em_steps: int = 500
    sample_batch_size: int = 512
    interval_level: float = 0.90
    trajectory_index: int = 0
    seed: int = 101

    def validate(self) -> None:
        if self.num_test_conditions < 1:
            raise ValueError(
                "num_test_conditions must be positive."
            )
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2.")
        if self.em_steps < 1:
            raise ValueError("em_steps must be positive.")
        if self.sample_batch_size < 1:
            raise ValueError(
                "sample_batch_size must be positive."
            )
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError(
                "interval_level must lie strictly between 0 and 1."
            )
        if self.trajectory_index < 0:
            raise ValueError(
                "trajectory_index must be nonnegative."
            )


@dataclass(frozen=True)
class PlotConfig:
    """Appearance of the one-subplot-per-mode coefficient figure."""

    subplot_columns: int = 3
    subplot_width: float = 5.0
    subplot_height: float = 3.5
    interval_alpha: float = 0.25
    line_width: float = 1.8
    marker_size: float = 4.0
    legend_columns: int = 1
    dpi: int = 200
    title: str = (
        "2D Kolmogorov-flow POD coefficient reconstruction with uncertainty"
    )

    def validate(self) -> None:
        if self.subplot_columns < 1:
            raise ValueError(
                "subplot_columns must be positive."
            )
        if self.subplot_width <= 0.0:
            raise ValueError("subplot_width must be positive.")
        if self.subplot_height <= 0.0:
            raise ValueError("subplot_height must be positive.")
        if not 0.0 <= self.interval_alpha <= 1.0:
            raise ValueError(
                "interval_alpha must lie in [0,1]."
            )
        if self.line_width <= 0.0:
            raise ValueError("line_width must be positive.")
        if self.marker_size < 0.0:
            raise ValueError("marker_size must be nonnegative.")
        if self.legend_columns < 1:
            raise ValueError(
                "legend_columns must be positive."
            )
        if self.dpi < 1:
            raise ValueError("dpi must be positive.")


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
    """Componentwise affine standardization."""

    mean: Array
    scale: Array
    regularized_mask: Array

    def transform(self, values: Array) -> Array:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: Array) -> Array:
        return self.mean + self.scale * values


@dataclass(frozen=True)
class SensorLayout:
    """Cartesian product of centered one-dimensional sensor grids."""

    x_indices: Array
    y_indices: Array
    x_positions: Array
    y_positions: Array

    @property
    def num_sensors(self) -> int:
        return int(self.x_indices.size)


@dataclass(frozen=True)
class PODModel:
    """Mean-centered POD fitted from training snapshots only."""

    mean_field_flat: Array
    modes_flat: Array
    singular_values: Array
    energy_fraction: Array
    cumulative_energy: Array
    num_modes: int
    spatial_shape: Tuple[int, int]

    def coefficients(self, snapshots: Array) -> Array:
        """
        Project snapshots onto retained POD modes.

        Accepted shapes are (X,Y), (M,X,Y), (X*Y,), or (M,X*Y).
        """
        snapshots = np.asarray(snapshots, dtype=np.float32)

        if snapshots.shape == self.spatial_shape:
            flattened = snapshots.reshape(1, -1)
            single = True
        elif snapshots.ndim == 3 and snapshots.shape[1:] == self.spatial_shape:
            flattened = snapshots.reshape(snapshots.shape[0], -1)
            single = False
        elif snapshots.ndim == 1:
            flattened = snapshots.reshape(1, -1)
            single = True
        elif snapshots.ndim == 2 and snapshots.shape[1] == (
            self.spatial_shape[0] * self.spatial_shape[1]
        ):
            flattened = snapshots
            single = False
        else:
            raise ValueError(
                "Snapshots have a shape incompatible with the POD spatial grid."
            )

        centered = flattened - self.mean_field_flat[None, :]
        coefficients = centered @ self.modes_flat.T
        return coefficients[0] if single else coefficients

    def reconstruct(self, coefficients: Array) -> Array:
        """Reconstruct flattened or batched fields from retained coefficients."""
        coefficients = np.asarray(coefficients, dtype=np.float32)
        single = coefficients.ndim == 1
        coefficients_2d = (
            coefficients.reshape(1, -1)
            if single
            else coefficients
        )
        flattened = (
            self.mean_field_flat[None, :]
            + coefficients_2d @ self.modes_flat
        )
        fields = flattened.reshape(
            coefficients_2d.shape[0],
            *self.spatial_shape,
        )
        return fields[0] if single else fields


@dataclass
class PairBuffer:
    """Physical delay/POD pairs and temporal provenance."""

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
                f"No valid pairs were created for the {split_name} split."
            )

        Y = np.stack(self.Y, axis=0).astype(
            np.float32,
            copy=False,
        )
        U = np.stack(self.U, axis=0).astype(
            np.float32,
            copy=False,
        )

        if Y.shape[1] != condition_dim:
            raise RuntimeError(
                f"Internal condition dimension mismatch in {split_name}."
            )
        if U.shape[1] != target_dim:
            raise RuntimeError(
                f"Internal target dimension mismatch in {split_name}."
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
# Loading, chronological splitting, and sparse sensors
# ---------------------------------------------------------------------------

def load_trajectories(path: str | Path) -> Array:
    """Load and validate a finite array with shape (N,T,X,Y)."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {path}"
        )

    trajectories = np.asarray(
        np.load(path, allow_pickle=False)
    )

    if trajectories.ndim != 4:
        raise ValueError(
            "Expected an array with shape (N,T,X,Y); "
            f"received {trajectories.shape}."
        )
    if not np.issubdtype(trajectories.dtype, np.number):
        raise TypeError("Trajectory array must be numerical.")
    if not np.isfinite(trajectories).all():
        raise ValueError(
            "Trajectory array contains NaN or infinite values."
        )

    n_trajectories, n_times, n_x, n_y = trajectories.shape
    if n_trajectories < 1:
        raise ValueError("N must be positive.")
    if n_times < 3:
        raise ValueError("T must be at least 3.")
    if n_x < 1 or n_y < 1:
        raise ValueError(
            "Both spatial dimensions must be positive."
        )

    return np.asarray(trajectories, dtype=np.float32)


def compute_split_bounds(n_times: int) -> SplitBounds:
    """Compute the per-trajectory 70%/15%/15% chronological split."""
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


def choose_centered_uniform_sensors(
    n_x: int,
    n_y: int,
    *,
    sensors_x: int,
    sensors_y: int,
    domain_length_x: float,
    domain_length_y: float,
) -> SensorLayout:
    """
    Place one sensor at the center of every equal spatial bin.

    The index rules are

        floor((k+1/2)*n_x/sensors_x),
        floor((l+1/2)*n_y/sensors_y).

    Their Cartesian product defines the complete sparse-sensor layout.
    """
    if sensors_x > n_x:
        raise ValueError(
            f"sensors_x={sensors_x} exceeds X={n_x}."
        )
    if sensors_y > n_y:
        raise ValueError(
            f"sensors_y={sensors_y} exceeds Y={n_y}."
        )

    unique_x = np.floor(
        (
            np.arange(sensors_x, dtype=np.float64)
            + 0.5
        )
        * n_x
        / sensors_x
    ).astype(np.int64)
    unique_y = np.floor(
        (
            np.arange(sensors_y, dtype=np.float64)
            + 0.5
        )
        * n_y
        / sensors_y
    ).astype(np.int64)

    if np.unique(unique_x).size != sensors_x:
        raise RuntimeError(
            "Centered x sensor placement produced duplicates."
        )
    if np.unique(unique_y).size != sensors_y:
        raise RuntimeError(
            "Centered y sensor placement produced duplicates."
        )

    grid_x, grid_y = np.meshgrid(
        unique_x,
        unique_y,
        indexing="ij",
    )
    x_indices = grid_x.reshape(-1)
    y_indices = grid_y.reshape(-1)

    return SensorLayout(
        x_indices=x_indices,
        y_indices=y_indices,
        x_positions=(
            domain_length_x
            * x_indices.astype(np.float64)
            / n_x
        ),
        y_positions=(
            domain_length_y
            * y_indices.astype(np.float64)
            / n_y
        ),
    )


def observe_sensors(
    trajectory: Array,
    time_indices: Array,
    sensors: SensorLayout,
) -> Array:
    """Extract all sparse sensors at each requested time."""
    snapshots = trajectory[time_indices]
    return snapshots[
        :,
        sensors.x_indices,
        sensors.y_indices,
    ]


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
    Fit an exact mean-centered snapshot POD using training portions only.

    Energy is defined by squared singular values of the centered training
    snapshot matrix. The retained rank is the smallest r whose cumulative
    energy is at least ``energy_threshold``.
    """
    train_start, train_end = train_bounds
    n_x = trajectories.shape[2]
    n_y = trajectories.shape[3]
    spatial_dimension = n_x * n_y

    snapshots = trajectories[
        :,
        train_start:train_end,
        :,
        :,
    ].reshape(-1, spatial_dimension)

    if snapshots.shape[0] < 2:
        raise ValueError(
            "At least two training snapshots are required for POD."
        )

    snapshots64 = np.asarray(snapshots, dtype=np.float64)
    mean_field = snapshots64.mean(axis=0)
    centered = snapshots64 - mean_field[None, :]

    _, singular_values, right_singular_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    squared_values = singular_values**2
    total_energy = float(squared_values.sum())

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
        mean_field_flat=mean_field.astype(np.float32),
        modes_flat=right_singular_vectors[
            :num_modes
        ].astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        energy_fraction=energy_fraction.astype(np.float64),
        cumulative_energy=cumulative_energy.astype(np.float64),
        num_modes=num_modes,
        spatial_shape=(n_x, n_y),
    )


def plot_pod_energy(
    pod: PODModel,
    *,
    threshold: float,
    output_path: str | Path,
    dpi: int = 200,
) -> Path:
    """Save cumulative training-POD energy and selected rank."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode_numbers = np.arange(
        1,
        pod.cumulative_energy.size + 1,
        dtype=np.int64,
    )
    retained_energy = float(
        pod.cumulative_energy[pod.num_modes - 1]
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        mode_numbers,
        pod.cumulative_energy,
        label="Cumulative fluctuation energy",
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
        "Training-only mean-centered POD energy selection"
    )
    axis.set_ylim(0.0, 1.01)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_path


# ---------------------------------------------------------------------------
# Non-overlapping delay windows and targets
# ---------------------------------------------------------------------------

def anchor_indices_for_split(
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> Array:
    """Return anchors whose complete delay spans lie inside one split."""
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
    """Verify containment, non-overlap, and exact gap."""
    if anchors.size == 0:
        return

    starts = anchors - config.embedding_span
    ends = anchors

    if starts[0] < split_start or ends[-1] >= split_end:
        raise RuntimeError(
            "A delay window crosses a split boundary."
        )

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
    sensors: SensorLayout,
    pod: PODModel,
    config: DataConfig,
) -> Dict[str, Array]:
    """Construct physical sparse-delay conditions and POD targets."""
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

    if anchors.size == 0:
        split_length = split_end - split_start
        raise ValueError(
            f"No valid pairs can be created for the {split_name} split. "
            f"The split contains {split_length} time indices, while one "
            f"delay span requires at least {config.window_span_length}. "
            "Increase T or reduce m or tau."
        )

    condition_dim = sensors.num_sensors * config.m
    delay_offsets = (
        np.arange(config.m, dtype=np.int64) * config.tau
    )
    buffer = PairBuffer.empty()

    for trajectory_index, trajectory in enumerate(trajectories):
        anchor_snapshots = trajectory[anchors]
        anchor_coefficients = pod.coefficients(
            anchor_snapshots
        )

        for local_index, anchor_index in enumerate(anchors):
            delay_indices = anchor_index - delay_offsets
            sensor_history = observe_sensors(
                trajectory,
                delay_indices,
                sensors,
            )
            condition = sensor_history.reshape(condition_dim)

            buffer.append(
                y=condition,
                u=anchor_coefficients[local_index],
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


def prepare_kolmogorov_pod_data(
    trajectories: Array,
    *,
    config: DataConfig,
) -> Dict[str, Any]:
    """Fit training POD, construct split pairs, and normalize them."""
    config.validate()

    n_trajectories, n_times, n_x, n_y = trajectories.shape
    split_bounds = compute_split_bounds(n_times)

    sensors = choose_centered_uniform_sensors(
        n_x,
        n_y,
        sensors_x=config.sensors_x,
        sensors_y=config.sensors_y,
        domain_length_x=config.domain_length_x,
        domain_length_y=config.domain_length_y,
    )

    pod = fit_training_pod(
        trajectories,
        split_bounds.train,
        energy_threshold=config.energy_threshold,
    )

    raw_splits = {
        split_name: build_split_pairs(
            trajectories,
            split_name=split_name,
            bounds=bounds,
            sensors=sensors,
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

    normalized_splits = {
        split_name: {
            "Y": y_standardizer.transform(
                split["Y"]
            ).astype(np.float32),
            "U": u_standardizer.transform(
                split["U"]
            ).astype(np.float32),
        }
        for split_name, split in raw_splits.items()
    }

    return {
        "split_bounds": split_bounds,
        "sensors": sensors,
        "pod": pod,
        "raw_splits": raw_splits,
        "normalized_splits": normalized_splits,
        "y_standardizer": y_standardizer,
        "u_standardizer": u_standardizer,
        "input_shape": (
            n_trajectories,
            n_times,
            n_x,
            n_y,
        ),
    }


def dataset_arrays_for_npz(
    prepared: Mapping[str, Any],
    *,
    config: DataConfig,
) -> Dict[str, Array]:
    """Assemble required training arrays and preprocessing metadata."""
    normalized = prepared["normalized_splits"]
    raw = prepared["raw_splits"]
    split_bounds: SplitBounds = prepared["split_bounds"]
    sensors: SensorLayout = prepared["sensors"]
    pod: PODModel = prepared["pod"]
    y_standardizer: Standardizer = prepared["y_standardizer"]
    u_standardizer: Standardizer = prepared["u_standardizer"]

    arrays: Dict[str, Array] = {
        # Required by train_drift_with_visualization.py.
        "Y_train": normalized["train"]["Y"],
        "U_train": normalized["train"]["U"],
        "Y_val": normalized["validation"]["Y"],
        "U_val": normalized["validation"]["U"],
        "Y_test": normalized["test"]["Y"],
        "U_test": normalized["test"]["U"],

        # Training normalization, also required for physical sampling.
        "mu_Y": y_standardizer.mean,
        "sigma_Y": y_standardizer.scale,
        "mu_U": u_standardizer.mean,
        "sigma_U": u_standardizer.scale,

        # Physical-coordinate copies.
        "Y_train_physical": raw["train"]["Y"],
        "U_train_physical": raw["train"]["U"],
        "Y_val_physical": raw["validation"]["Y"],
        "U_val_physical": raw["validation"]["U"],
        "Y_test_physical": raw["test"]["Y"],
        "U_test_physical": raw["test"]["U"],

        # Normalization diagnostics.
        "regularized_Y_channels": (
            y_standardizer.regularized_mask
        ),
        "regularized_U_channels": (
            u_standardizer.regularized_mask
        ),

        # Sensor layout.
        "sensor_x_indices": sensors.x_indices,
        "sensor_y_indices": sensors.y_indices,
        "sensor_x_positions": sensors.x_positions,
        "sensor_y_positions": sensors.y_positions,

        # POD model.
        "pod_mean_field_flat": pod.mean_field_flat,
        "pod_modes_flat": pod.modes_flat,
        "pod_singular_values": pod.singular_values,
        "pod_energy_fraction": pod.energy_fraction,
        "pod_cumulative_energy": pod.cumulative_energy,
        "pod_num_modes": np.asarray(
            pod.num_modes,
            dtype=np.int64,
        ),
        "pod_spatial_shape": np.asarray(
            pod.spatial_shape,
            dtype=np.int64,
        ),

        # Split and pair metadata.
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
        "sensors_x": np.asarray(
            config.sensors_x,
            dtype=np.int64,
        ),
        "sensors_y": np.asarray(
            config.sensors_y,
            dtype=np.int64,
        ),
        "energy_threshold": np.asarray(
            config.energy_threshold,
            dtype=np.float64,
        ),
        "domain_length_x": np.asarray(
            config.domain_length_x,
            dtype=np.float64,
        ),
        "domain_length_y": np.asarray(
            config.domain_length_y,
            dtype=np.float64,
        ),
    }

    split_prefixes = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    for split_name, prefix in split_prefixes.items():
        arrays[f"trajectory_index_{prefix}"] = raw[split_name][
            "trajectory_index"
        ]
        arrays[f"anchor_index_{prefix}"] = raw[split_name][
            "anchor_index"
        ]
        arrays[f"window_start_{prefix}"] = raw[split_name][
            "window_start"
        ]
        arrays[f"window_end_{prefix}"] = raw[split_name][
            "window_end"
        ]

    metadata = {
        "input_shape": [
            int(value)
            for value in prepared["input_shape"]
        ],
        "condition_definition": (
            "[P(X_t), P(X_{t-tau}), ..., "
            "P(X_{t-(m-1)tau})]"
        ),
        "target_definition": (
            "retained mean-centered training-POD coefficients at anchor t"
        ),
        "split_rule": (
            "first 70 percent train, next 15 percent validation, "
            "final 15 percent test within every trajectory"
        ),
        "pod_fit_data": (
            "all snapshots in the training portion of every trajectory"
        ),
        "pod_energy_definition": (
            "squared singular values of mean-centered training snapshots"
        ),
        "pod_energy_threshold": float(
            config.energy_threshold
        ),
        "pod_num_modes": int(pod.num_modes),
        "pod_retained_energy": float(
            pod.cumulative_energy[pod.num_modes - 1]
        ),
        "embedding_dimension_m": int(config.m),
        "time_delay_tau_indices": int(config.tau),
        "gap_indices": int(config.gap),
        "window_span_length_indices": int(
            config.window_span_length
        ),
        "anchor_stride_indices": int(
            config.anchor_stride
        ),
        "sensors_x": int(config.sensors_x),
        "sensors_y": int(config.sensors_y),
        "num_sensors": int(config.num_sensors),
        "condition_dimension": int(
            config.num_sensors * config.m
        ),
        "target_dimension": int(pod.num_modes),
        "sensor_x_indices": sensors.x_indices.tolist(),
        "sensor_y_indices": sensors.y_indices.tolist(),
        "sensor_x_positions": sensors.x_positions.tolist(),
        "sensor_y_positions": sensors.y_positions.tolist(),
        "normalization_rule": (
            "componentwise statistics fitted from physical training pairs only"
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
    """Save the prepared training-compatible dataset."""
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
    train_config: Any,
    interpolant_config: Any,
) -> Dict[str, Any]:
    """Call the supplied conditional-drift training implementation."""
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
# Loading a prepared dataset for checkpoint reuse
# ---------------------------------------------------------------------------

def load_prepared_test_data_from_npz(
    dataset_path: str | Path,
) -> Tuple[
    Dict[str, Any],
    DataConfig,
    SensorLayout,
    Dict[str, Any],
]:
    """Load the held-out split and metadata needed for inference-only mode."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_path}"
        )

    required = {
        "Y_test",
        "U_test",
        "Y_test_physical",
        "U_test_physical",
        "trajectory_index_test",
        "anchor_index_test",
        "window_start_test",
        "window_end_test",
        "sensor_x_indices",
        "sensor_y_indices",
        "sensor_x_positions",
        "sensor_y_positions",
        "mu_Y",
        "sigma_Y",
        "mu_U",
        "sigma_U",
        "m",
        "tau",
        "gap",
        "sensors_x",
        "sensors_y",
        "energy_threshold",
        "domain_length_x",
        "domain_length_y",
        "pod_num_modes",
        "pod_cumulative_energy",
    }

    with np.load(dataset_path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                "Prepared dataset is missing: "
                + ", ".join(missing)
            )

        normalized_y = np.asarray(
            data["Y_test"],
            dtype=np.float32,
        )
        normalized_u = np.asarray(
            data["U_test"],
            dtype=np.float32,
        )
        physical_y = np.asarray(
            data["Y_test_physical"],
            dtype=np.float32,
        )
        physical_u = np.asarray(
            data["U_test_physical"],
            dtype=np.float32,
        )

        raw_test = {
            "Y": physical_y,
            "U": physical_u,
            "trajectory_index": np.asarray(
                data["trajectory_index_test"],
                dtype=np.int64,
            ),
            "anchor_index": np.asarray(
                data["anchor_index_test"],
                dtype=np.int64,
            ),
            "window_start": np.asarray(
                data["window_start_test"],
                dtype=np.int64,
            ),
            "window_end": np.asarray(
                data["window_end_test"],
                dtype=np.int64,
            ),
        }

        config = DataConfig(
            m=int(np.asarray(data["m"]).item()),
            tau=int(np.asarray(data["tau"]).item()),
            gap=int(np.asarray(data["gap"]).item()),
            sensors_x=int(
                np.asarray(data["sensors_x"]).item()
            ),
            sensors_y=int(
                np.asarray(data["sensors_y"]).item()
            ),
            energy_threshold=float(
                np.asarray(data["energy_threshold"]).item()
            ),
            domain_length_x=float(
                np.asarray(data["domain_length_x"]).item()
            ),
            domain_length_y=float(
                np.asarray(data["domain_length_y"]).item()
            ),
        )
        config.validate()

        sensors = SensorLayout(
            x_indices=np.asarray(
                data["sensor_x_indices"],
                dtype=np.int64,
            ).reshape(-1),
            y_indices=np.asarray(
                data["sensor_y_indices"],
                dtype=np.int64,
            ).reshape(-1),
            x_positions=np.asarray(
                data["sensor_x_positions"],
                dtype=np.float64,
            ).reshape(-1),
            y_positions=np.asarray(
                data["sensor_y_positions"],
                dtype=np.float64,
            ).reshape(-1),
        )

        normalization_stats = {
            "mu_Y": np.asarray(
                data["mu_Y"],
                dtype=np.float32,
            ).reshape(-1),
            "sigma_Y": np.asarray(
                data["sigma_Y"],
                dtype=np.float32,
            ).reshape(-1),
            "mu_U": np.asarray(
                data["mu_U"],
                dtype=np.float32,
            ).reshape(-1),
            "sigma_U": np.asarray(
                data["sigma_U"],
                dtype=np.float32,
            ).reshape(-1),
        }

        pod_num_modes = int(
            np.asarray(data["pod_num_modes"]).item()
        )
        pod_cumulative_energy = np.asarray(
            data["pod_cumulative_energy"],
            dtype=np.float64,
        )

        saved_metadata = (
            json.loads(str(np.asarray(data["metadata_json"]).item()))
            if "metadata_json" in data.files
            else {}
        )

    if normalized_y.ndim != 2 or normalized_u.ndim != 2:
        raise ValueError(
            "Y_test and U_test must both be matrices."
        )
    if physical_y.shape != normalized_y.shape:
        raise ValueError(
            "Y_test_physical and Y_test shapes differ."
        )
    if physical_u.shape != normalized_u.shape:
        raise ValueError(
            "U_test_physical and U_test shapes differ."
        )

    num_pairs = normalized_y.shape[0]
    for name, values in raw_test.items():
        if values.shape[0] != num_pairs:
            raise ValueError(
                f"{name} has an inconsistent pair count."
            )

    expected_condition_dim = config.num_sensors * config.m
    if normalized_y.shape[1] != expected_condition_dim:
        raise ValueError(
            "Condition dimension is inconsistent with the sensor layout and m."
        )
    if normalized_u.shape[1] != pod_num_modes:
        raise ValueError(
            "Target dimension is inconsistent with pod_num_modes."
        )
    if sensors.num_sensors != config.num_sensors:
        raise ValueError(
            "Saved sensor arrays are inconsistent with sensors_x*sensors_y."
        )

    prepared = {
        "normalized_splits": {
            "test": {
                "Y": normalized_y,
                "U": normalized_u,
            }
        },
        "raw_splits": {
            "test": raw_test,
        },
    }

    metadata = {
        "dataset_path": str(dataset_path),
        "condition_dimension": int(normalized_y.shape[1]),
        "target_dimension": int(normalized_u.shape[1]),
        "num_test_pairs": int(num_pairs),
        "normalization_stats": normalization_stats,
        "pod_num_modes": pod_num_modes,
        "pod_retained_energy": float(
            pod_cumulative_energy[pod_num_modes - 1]
        ),
        "saved_metadata": saved_metadata,
    }

    return prepared, config, sensors, metadata


# ---------------------------------------------------------------------------
# Conditional coefficient ensembles
# ---------------------------------------------------------------------------

def select_evenly_spaced_indices(
    population_size: int,
    requested_count: int,
) -> Array:
    """Select deterministic positions spanning the available ordering."""
    if population_size < 1:
        raise ValueError("population_size must be positive.")

    count = min(population_size, requested_count)
    if count == population_size:
        return np.arange(population_size, dtype=np.int64)

    return np.unique(
        np.linspace(
            0,
            population_size - 1,
            count,
            dtype=np.int64,
        )
    )


def generate_test_ensembles(
    prepared: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    uq_config: UQConfig,
    device_name: str,
    loaded_bundle: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate coefficient ensembles for selected anchors in one trajectory."""
    uq_config.validate()

    if loaded_bundle is None:
        loaded_bundle = load_trained_drift(
            checkpoint_path,
            device_name=device_name,
        )

    (
        model,
        interpolant_config,
        normalization_stats,
        checkpoint,
        device,
    ) = loaded_bundle

    normalized_test = prepared["normalized_splits"]["test"]
    raw_test = prepared["raw_splits"]["test"]

    candidates = np.flatnonzero(
        raw_test["trajectory_index"]
        == uq_config.trajectory_index
    )
    if candidates.size == 0:
        available = np.unique(
            raw_test["trajectory_index"]
        ).astype(int).tolist()
        raise ValueError(
            f"No test conditions exist for trajectory "
            f"{uq_config.trajectory_index}. Available trajectories: "
            f"{available}."
        )

    chronological_order = np.argsort(
        raw_test["anchor_index"][candidates]
    )
    candidates = candidates[chronological_order]

    selected_positions = select_evenly_spaced_indices(
        candidates.size,
        uq_config.num_test_conditions,
    )
    selected_indices = candidates[selected_positions]

    normalized_ensembles: List[Array] = []
    physical_ensembles: List[Array] = []

    for local_index, test_index in enumerate(selected_indices):
        sample_seed = uq_config.seed + local_index
        set_seed(sample_seed)

        sampling_config = SamplingConfig(
            num_samples=uq_config.ensemble_size,
            num_steps=uq_config.em_steps,
            sample_batch_size=uq_config.sample_batch_size,
            seed=sample_seed,
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
        "normalized_input_y": normalized_test["Y"][
            selected_indices
        ],
        "true_coefficients": raw_test["U"][
            selected_indices
        ],
        "normalized_true_coefficients": normalized_test["U"][
            selected_indices
        ],
        "samples": np.stack(
            physical_ensembles,
            axis=0,
        ),
        "normalized_samples": np.stack(
            normalized_ensembles,
            axis=0,
        ),
        "trajectory_index": raw_test["trajectory_index"][
            selected_indices
        ],
        "anchor_index": raw_test["anchor_index"][
            selected_indices
        ],
        "window_start": raw_test["window_start"][
            selected_indices
        ],
        "window_end": raw_test["window_end"][
            selected_indices
        ],
        "checkpoint": checkpoint,
        "interpolant_config": interpolant_config,
        "device": device,
    }


def validate_checkpoint_dataset_compatibility(
    loaded_bundle: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Reject checkpoint/data dimension or normalization mismatches."""
    (
        _model,
        _interpolant_config,
        checkpoint_stats,
        checkpoint,
        _device,
    ) = loaded_bundle

    model_config = checkpoint["model_config"]
    if (
        int(model_config["condition_dim"])
        != int(metadata["condition_dimension"])
    ):
        raise ValueError(
            "Checkpoint and dataset condition dimensions do not match."
        )
    if (
        int(model_config["target_dim"])
        != int(metadata["target_dimension"])
    ):
        raise ValueError(
            "Checkpoint and dataset target dimensions do not match."
        )

    checkpoint_mapping = checkpoint_stats.as_dict()
    dataset_mapping = metadata["normalization_stats"]

    for key in ("mu_Y", "sigma_Y", "mu_U", "sigma_U"):
        left = np.asarray(
            checkpoint_mapping[key],
            dtype=np.float32,
        ).reshape(-1)
        right = np.asarray(
            dataset_mapping[key],
            dtype=np.float32,
        ).reshape(-1)

        if left.shape != right.shape or not np.allclose(
            left,
            right,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"Checkpoint and dataset use different {key}. "
                "Use files from the same training run."
            )


# ---------------------------------------------------------------------------
# Saving and loading UQ ensembles
# ---------------------------------------------------------------------------

def save_uq_samples(
    output_path: str | Path,
    ensembles: Mapping[str, Any],
    *,
    data_config: DataConfig,
    uq_config: UQConfig,
    pod_num_modes: int,
    pod_retained_energy: float,
) -> Path:
    """Save coefficient ensembles, truths, provenance, and plot metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interpolant_config = ensembles["interpolant_config"]
    metadata = {
        "target": "mean-centered training-POD coefficients at anchor t",
        "pod_num_modes": int(pod_num_modes),
        "pod_energy_threshold": float(
            data_config.energy_threshold
        ),
        "pod_retained_energy": float(pod_retained_energy),
        "embedding_dimension_m": int(data_config.m),
        "time_delay_tau_indices": int(data_config.tau),
        "gap_indices": int(data_config.gap),
        "trajectory_index": int(uq_config.trajectory_index),
        "num_test_conditions": int(
            ensembles["selected_test_indices"].size
        ),
        "ensemble_size": int(uq_config.ensemble_size),
        "em_steps": int(uq_config.em_steps),
        "interval_level": float(uq_config.interval_level),
        "seed": int(uq_config.seed),
        "sigma_I": float(interpolant_config.sigma_I),
    }

    np.savez_compressed(
        output_path,
        selected_test_indices=ensembles[
            "selected_test_indices"
        ],
        input_y=ensembles["input_y"],
        normalized_input_y=ensembles[
            "normalized_input_y"
        ],
        true_coefficients=ensembles[
            "true_coefficients"
        ],
        normalized_true_coefficients=ensembles[
            "normalized_true_coefficients"
        ],
        samples=ensembles["samples"],
        normalized_samples=ensembles["normalized_samples"],
        trajectory_index=ensembles["trajectory_index"],
        anchor_index=ensembles["anchor_index"],
        window_start=ensembles["window_start"],
        window_end=ensembles["window_end"],
        m=np.asarray(data_config.m, dtype=np.int64),
        tau=np.asarray(data_config.tau, dtype=np.int64),
        gap=np.asarray(data_config.gap, dtype=np.int64),
        energy_threshold=np.asarray(
            data_config.energy_threshold,
            dtype=np.float64,
        ),
        pod_num_modes=np.asarray(
            pod_num_modes,
            dtype=np.int64,
        ),
        pod_retained_energy=np.asarray(
            pod_retained_energy,
            dtype=np.float64,
        ),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )
    return output_path


def load_saved_uq_samples(
    samples_path: str | Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load ensemble data needed for plot-only mode."""
    samples_path = Path(samples_path)
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Saved ensemble file not found: {samples_path}"
        )

    required = {
        "samples",
        "true_coefficients",
        "trajectory_index",
        "anchor_index",
        "pod_num_modes",
        "pod_retained_energy",
    }

    with np.load(samples_path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                "Saved ensemble file is missing: "
                + ", ".join(missing)
            )

        ensembles = {
            "samples": np.asarray(
                data["samples"],
                dtype=np.float32,
            ),
            "true_coefficients": np.asarray(
                data["true_coefficients"],
                dtype=np.float32,
            ),
            "trajectory_index": np.asarray(
                data["trajectory_index"],
                dtype=np.int64,
            ),
            "anchor_index": np.asarray(
                data["anchor_index"],
                dtype=np.int64,
            ),
        }
        metadata = {
            "pod_num_modes": int(
                np.asarray(data["pod_num_modes"]).item()
            ),
            "pod_retained_energy": float(
                np.asarray(data["pod_retained_energy"]).item()
            ),
        }

    return ensembles, metadata


# ---------------------------------------------------------------------------
# Modal-coefficient UQ subplot figure
# ---------------------------------------------------------------------------

def central_interval(
    samples: Array,
    level: float,
) -> Tuple[Array, Array]:
    """Compute componentwise central empirical intervals over ensembles."""
    alpha = 1.0 - level
    lower = np.quantile(
        samples,
        alpha / 2.0,
        axis=1,
    )
    upper = np.quantile(
        samples,
        1.0 - alpha / 2.0,
        axis=1,
    )
    return lower, upper


def plot_modal_coefficient_subplots(
    ensembles: Mapping[str, Any],
    *,
    interval_level: float,
    pod_retained_energy: float,
    plot_config: PlotConfig,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Put every retained POD coefficient in its own subplot."""
    plot_config.validate()
    if not 0.0 < interval_level < 1.0:
        raise ValueError(
            "interval_level must lie strictly between 0 and 1."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(
        ensembles["samples"],
        dtype=np.float64,
    )
    truths = np.asarray(
        ensembles["true_coefficients"],
        dtype=np.float64,
    )
    anchors = np.asarray(
        ensembles["anchor_index"],
        dtype=np.int64,
    )
    trajectories = np.asarray(
        ensembles["trajectory_index"],
        dtype=np.int64,
    )

    if samples.ndim != 3:
        raise ValueError(
            "samples must have shape "
            "(conditions, ensemble_size, modes)."
        )
    if truths.shape != (
        samples.shape[0],
        samples.shape[2],
    ):
        raise ValueError(
            "true_coefficients has an incompatible shape."
        )
    if anchors.shape != (samples.shape[0],):
        raise ValueError(
            "anchor_index must have one entry per condition."
        )
    if trajectories.shape != (samples.shape[0],):
        raise ValueError(
            "trajectory_index must have one entry per condition."
        )
    if np.unique(trajectories).size != 1:
        raise ValueError(
            "The anchor-time subplot figure requires one trajectory."
        )

    order = np.argsort(anchors)
    anchors = anchors[order]
    samples = samples[order]
    truths = truths[order]

    lower, upper = central_interval(
        samples,
        interval_level,
    )
    means = samples.mean(axis=1)

    covered = (
        (truths >= lower)
        & (truths <= upper)
    )
    coefficient_coverage = covered.mean(axis=0)
    coefficient_rmse = np.sqrt(
        np.mean(
            (means - truths) ** 2,
            axis=0,
        )
    )
    coefficient_width = (
        upper - lower
    ).mean(axis=0)

    num_modes = samples.shape[2]
    columns = min(
        plot_config.subplot_columns,
        num_modes,
    )
    rows = int(np.ceil(num_modes / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            plot_config.subplot_width * columns,
            plot_config.subplot_height * rows,
        ),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)

    for mode_index, axis in enumerate(flat_axes):
        if mode_index >= num_modes:
            axis.set_visible(False)
            continue

        mode_number = mode_index + 1
        mean_line, = axis.plot(
            anchors,
            means[:, mode_index],
            marker="o",
            markersize=plot_config.marker_size,
            linewidth=plot_config.line_width,
            label="Ensemble mean",
            zorder=3,
        )
        interval_color = mean_line.get_color()

        axis.fill_between(
            anchors,
            lower[:, mode_index],
            upper[:, mode_index],
            color=interval_color,
            alpha=plot_config.interval_alpha,
            linewidth=0.0,
            label=f"{interval_level:.0%} interval",
            zorder=1,
        )
        axis.plot(
            anchors,
            truths[:, mode_index],
            color=interval_color,
            marker="x",
            markersize=plot_config.marker_size,
            linewidth=plot_config.line_width,
            linestyle="--",
            label="True coefficient",
            zorder=4,
        )

        axis.set_title(
            f"Mode {mode_number}: "
            f"coverage={coefficient_coverage[mode_index]:.1%}, "
            f"RMSE={coefficient_rmse[mode_index]:.3e}"
        )
        axis.set_ylabel(
            rf"$a_{{{mode_number}}}(t)$"
        )
        axis.grid(True)
        axis.legend(
            ncol=plot_config.legend_columns,
            fontsize="small",
            loc="best",
        )

    trajectory_index = int(trajectories[0])
    figure.supxlabel("Anchor time index t")
    figure.supylabel("POD coefficient")
    figure.suptitle(
        f"{plot_config.title}\n"
        f"trajectory={trajectory_index}, "
        f"retained modes={num_modes}, "
        f"retained energy={pod_retained_energy:.3%}, "
        f"central interval={interval_level:.0%}",
        y=0.998,
    )
    figure.tight_layout(
        rect=(0.02, 0.02, 1.0, 0.965)
    )
    figure.savefig(
        output_path,
        dpi=plot_config.dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    return {
        "plot_path": output_path,
        "anchor_indices": anchors,
        "forecast_mean": means,
        "lower": lower,
        "upper": upper,
        "truth": truths,
        "coefficient_coverage": coefficient_coverage,
        "coefficient_rmse": coefficient_rmse,
        "coefficient_mean_interval_width": coefficient_width,
        "trajectory_index": trajectory_index,
        "subplot_rows": rows,
        "subplot_columns": columns,
    }


def create_uq_visualization(
    ensembles: Mapping[str, Any],
    *,
    interval_level: float,
    pod_retained_energy: float,
    plot_config: PlotConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Create the modal subplot figure and save numerical UQ metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = plot_modal_coefficient_subplots(
        ensembles,
        interval_level=interval_level,
        pod_retained_energy=pod_retained_energy,
        plot_config=plot_config,
        output_path=(
            output_dir
            / "pod_coefficient_uq_subplots.png"
        ),
    )

    metrics_path = output_dir / "pod_coefficient_uq_metrics.npz"
    np.savez_compressed(
        metrics_path,
        anchor_indices=result["anchor_indices"],
        forecast_mean=result["forecast_mean"],
        lower=result["lower"],
        upper=result["upper"],
        truth=result["truth"],
        coefficient_coverage=result[
            "coefficient_coverage"
        ],
        coefficient_rmse=result[
            "coefficient_rmse"
        ],
        coefficient_mean_interval_width=result[
            "coefficient_mean_interval_width"
        ],
        interval_level=np.asarray(
            interval_level,
            dtype=np.float64,
        ),
        pod_retained_energy=np.asarray(
            pod_retained_energy,
            dtype=np.float64,
        ),
    )

    return {
        "coefficient_subplot_figure": str(
            result["plot_path"]
        ),
        "metrics_file": str(metrics_path),
        "trajectory_index": int(
            result["trajectory_index"]
        ),
        "mean_coverage": float(
            np.mean(result["coefficient_coverage"])
        ),
        "mean_rmse": float(
            np.mean(result["coefficient_rmse"])
        ),
        "mean_interval_width": float(
            np.mean(
                result["coefficient_mean_interval_width"]
            )
        ),
        "subplot_rows": int(result["subplot_rows"]),
        "subplot_columns": int(
            result["subplot_columns"]
        ),
    }


# ---------------------------------------------------------------------------
# Complete train, inference, and plot pipelines
# ---------------------------------------------------------------------------

def run_training_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    data_config: DataConfig,
    train_config: Any,
    interpolant_config: Any,
    uq_config: UQConfig,
    plot_config: PlotConfig,
) -> Dict[str, Any]:
    """Prepare data, train, sample, and create UQ figures."""
    data_config.validate()
    train_config.validate()
    interpolant_config.validate()
    uq_config.validate()
    plot_config.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(input_path)
    prepared = prepare_kolmogorov_pod_data(
        trajectories,
        config=data_config,
    )

    dataset_path = save_dataset_npz(
        output_dir / "kolmogorov_pod_data.npz",
        dataset_arrays_for_npz(
            prepared,
            config=data_config,
        ),
    )

    pod: PODModel = prepared["pod"]
    retained_energy = float(
        pod.cumulative_energy[pod.num_modes - 1]
    )
    pod_energy_path = plot_pod_energy(
        pod,
        threshold=data_config.energy_threshold,
        output_path=(
            output_dir / "pod_cumulative_energy.png"
        ),
        dpi=plot_config.dpi,
    )

    training_result = train_from_prepared_data(
        prepared,
        output_dir=output_dir / "training",
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
    samples_path = save_uq_samples(
        uq_dir / "conditional_pod_samples.npz",
        ensembles,
        data_config=data_config,
        uq_config=uq_config,
        pod_num_modes=pod.num_modes,
        pod_retained_energy=retained_energy,
    )
    visualization = create_uq_visualization(
        ensembles,
        interval_level=uq_config.interval_level,
        pod_retained_energy=retained_energy,
        plot_config=plot_config,
        output_dir=uq_dir,
    )

    summary = {
        "mode": "train",
        "input_file": str(Path(input_path)),
        "output_directory": str(output_dir),
        "dataset_file": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "pod_energy_plot": str(pod_energy_path),
        "training_summary": training_result["summary"],
        "uq_samples_file": str(samples_path),
        "uq_visualization": visualization,
        "data_config": asdict(data_config),
        "train_config": asdict(train_config),
        "interpolant_config": asdict(interpolant_config),
        "uq_config": asdict(uq_config),
        "plot_config": asdict(plot_config),
        "input_shape": [
            int(value)
            for value in prepared["input_shape"]
        ],
        "num_sensors": int(data_config.num_sensors),
        "condition_dimension": int(
            data_config.num_sensors * data_config.m
        ),
        "pod_target_dimension": int(pod.num_modes),
        "pod_retained_energy": retained_energy,
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


def run_inference_only(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    uq_config: UQConfig,
    plot_config: PlotConfig,
    device_name: str,
) -> Dict[str, Any]:
    """Generate new coefficient ensembles and plots without retraining."""
    uq_config.validate()
    plot_config.validate()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    (
        prepared,
        data_config,
        _sensors,
        metadata,
    ) = load_prepared_test_data_from_npz(dataset_path)

    loaded_bundle = load_trained_drift(
        checkpoint_path,
        device_name=device_name,
    )
    validate_checkpoint_dataset_compatibility(
        loaded_bundle,
        metadata,
    )

    ensembles = generate_test_ensembles(
        prepared,
        checkpoint_path=checkpoint_path,
        uq_config=uq_config,
        device_name=device_name,
        loaded_bundle=loaded_bundle,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = save_uq_samples(
        output_dir / "conditional_pod_samples.npz",
        ensembles,
        data_config=data_config,
        uq_config=uq_config,
        pod_num_modes=metadata["pod_num_modes"],
        pod_retained_energy=metadata[
            "pod_retained_energy"
        ],
    )
    visualization = create_uq_visualization(
        ensembles,
        interval_level=uq_config.interval_level,
        pod_retained_energy=metadata[
            "pod_retained_energy"
        ],
        plot_config=plot_config,
        output_dir=output_dir,
    )

    summary = {
        "mode": "inference",
        "dataset_file": str(Path(dataset_path)),
        "checkpoint": str(checkpoint_path),
        "output_directory": str(output_dir),
        "uq_samples_file": str(samples_path),
        "uq_visualization": visualization,
        "data_config": asdict(data_config),
        "uq_config": asdict(uq_config),
        "plot_config": asdict(plot_config),
        "device": str(ensembles["device"]),
        "pod_target_dimension": int(
            metadata["pod_num_modes"]
        ),
        "pod_retained_energy": float(
            metadata["pod_retained_energy"]
        ),
        "num_available_test_pairs": int(
            metadata["num_test_pairs"]
        ),
        "num_sampled_conditions": int(
            ensembles["selected_test_indices"].size
        ),
    }

    summary_path = output_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    summary["inference_summary_file"] = str(summary_path)
    return summary


def run_plot_only(
    samples_path: str | Path,
    output_dir: str | Path,
    *,
    interval_level: float,
    plot_config: PlotConfig,
) -> Dict[str, Any]:
    """Redraw coefficient subplots from saved ensembles only."""
    ensembles, metadata = load_saved_uq_samples(
        samples_path
    )

    visualization = create_uq_visualization(
        ensembles,
        interval_level=interval_level,
        pod_retained_energy=metadata[
            "pod_retained_energy"
        ],
        plot_config=plot_config,
        output_dir=output_dir,
    )

    return {
        "mode": "plot",
        "samples_file": str(Path(samples_path)),
        "output_directory": str(Path(output_dir)),
        "interval_level": float(interval_level),
        "pod_num_modes": int(
            metadata["pod_num_modes"]
        ),
        "pod_retained_energy": float(
            metadata["pod_retained_energy"]
        ),
        "plot_config": asdict(plot_config),
        "uq_visualization": visualization,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    try:
        widths = tuple(
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Hidden dimensions must be comma-separated integers."
        ) from error

    if not widths or any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError(
            "Hidden dimensions must be positive integers."
        )
    return widths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, reuse, or replot a stochastic 2D Kolmogorov-flow "
            "sparse-sensor-to-POD reconstruction model."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("test.npy"),
        help="Trajectory array with shape (N,T,X,Y); used in train mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "inference", "plot"),
        default="train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kolmogorov_pod_output"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Prepared NPZ for inference mode. Defaults to "
            "OUTPUT_DIR/kolmogorov_pod_data.npz."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint for inference mode. Defaults to "
            "OUTPUT_DIR/training/conditional_drift_checkpoint.pt."
        ),
    )
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=None,
        help=(
            "Saved ensemble NPZ for plot mode. If omitted, checks "
            "OUTPUT_DIR/inference_uq and OUTPUT_DIR/uq."
        ),
    )
    parser.add_argument(
        "--inference-output-dir",
        type=Path,
        default=None,
        help=(
            "Inference output directory. Defaults to OUTPUT_DIR/inference_uq."
        ),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=None,
        help=(
            "Plot-only output directory. Defaults to OUTPUT_DIR/replotted_uq."
        ),
    )

    # Data construction, required only in train mode.
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--gap", type=int, default=None)
    parser.add_argument(
        "--sensors-x",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--sensors-y",
        type=int,
        default=3,
    )
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
        "--domain-length-x",
        type=float,
        default=1.0,
        help=(
            "Physical x length used only for saved sensor coordinates."
        ),
    )
    parser.add_argument(
        "--domain-length-y",
        type=float,
        default=1.0,
        help=(
            "Physical y length used only for saved sensor coordinates."
        ),
    )

    # Training.
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

    # Sampling/UQ.
    parser.add_argument(
        "--num-uq-conditions",
        type=int,
        default=20,
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
        "--uq-trajectory-index",
        type=int,
        default=0,
    )
    parser.add_argument("--uq-seed", type=int, default=101)

    # Plot appearance.
    parser.add_argument(
        "--subplot-columns",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--subplot-width",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--subplot-height",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--interval-alpha",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=1,
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--plot-title",
        type=str,
        default=(
            "2D Kolmogorov-flow POD coefficient reconstruction "
            "with uncertainty"
        ),
    )

    return parser.parse_args()


def resolve_plot_samples_path(
    requested_path: Optional[Path],
    output_dir: Path,
) -> Path:
    if requested_path is not None:
        return requested_path

    candidates = [
        output_dir
        / "inference_uq"
        / "conditional_pod_samples.npz",
        output_dir
        / "uq"
        / "conditional_pod_samples.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Plot mode could not find saved ensembles. "
        "Supply --samples-path explicitly."
    )


def main() -> None:
    args = parse_args()

    plot_config = PlotConfig(
        subplot_columns=args.subplot_columns,
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
        interval_alpha=args.interval_alpha,
        line_width=args.line_width,
        marker_size=args.marker_size,
        legend_columns=args.legend_columns,
        dpi=args.dpi,
        title=args.plot_title,
    )

    if args.mode == "plot":
        samples_path = resolve_plot_samples_path(
            args.samples_path,
            args.output_dir,
        )
        plot_output_dir = (
            args.plot_output_dir
            if args.plot_output_dir is not None
            else args.output_dir / "replotted_uq"
        )
        summary = run_plot_only(
            samples_path,
            plot_output_dir,
            interval_level=args.interval_level,
            plot_config=plot_config,
        )
        print(json.dumps(summary, indent=2))
        return

    uq_config = UQConfig(
        num_test_conditions=args.num_uq_conditions,
        ensemble_size=args.ensemble_size,
        em_steps=args.em_steps,
        sample_batch_size=args.sample_batch_size,
        interval_level=args.interval_level,
        trajectory_index=args.uq_trajectory_index,
        seed=args.uq_seed,
    )

    if args.mode == "inference":
        dataset_path = (
            args.dataset_path
            if args.dataset_path is not None
            else (
                args.output_dir
                / "kolmogorov_pod_data.npz"
            )
        )
        checkpoint_path = (
            args.checkpoint_path
            if args.checkpoint_path is not None
            else (
                args.output_dir
                / "training"
                / "conditional_drift_checkpoint.pt"
            )
        )
        inference_output_dir = (
            args.inference_output_dir
            if args.inference_output_dir is not None
            else args.output_dir / "inference_uq"
        )

        summary = run_inference_only(
            dataset_path,
            checkpoint_path,
            inference_output_dir,
            uq_config=uq_config,
            plot_config=plot_config,
            device_name=args.device,
        )
        print(json.dumps(summary, indent=2))
        return

    missing = [
        name
        for name, value in (
            ("--m", args.m),
            ("--tau", args.tau),
            ("--gap", args.gap),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(
            "Train mode requires: " + ", ".join(missing)
        )

    data_config = DataConfig(
        m=args.m,
        tau=args.tau,
        gap=args.gap,
        sensors_x=args.sensors_x,
        sensors_y=args.sensors_y,
        energy_threshold=args.energy_threshold,
        normalization_epsilon=args.normalization_epsilon,
        domain_length_x=args.domain_length_x,
        domain_length_y=args.domain_length_y,
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

    summary = run_training_pipeline(
        args.input,
        args.output_dir,
        data_config=data_config,
        train_config=train_config,
        interpolant_config=interpolant_config,
        uq_config=uq_config,
        plot_config=plot_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
