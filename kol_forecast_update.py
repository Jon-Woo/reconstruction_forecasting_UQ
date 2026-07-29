#!/usr/bin/env python3
"""
Stochastic 2D Kolmogorov-flow sparse-sensor forecasting with UQ.

Input
-----
test.npy must contain a finite numerical array with shape

    (N, T, X, Y),

where N is the number of trajectories, T is the number of stored time indices,
and X and Y are the two spatial grid dimensions.

Condition and target
--------------------
Let P extract K = sensors_x * sensors_y uniformly placed sensors from the
periodic spatial grid. At anchor index t,

    condition:
        Y_t = [
            P(X_t),
            P(X_{t-tau}),
            ...,
            P(X_{t-(m-1)tau})
        ] in R^(K*m),

    target:
        U_t = P(X_{t+tau}) in R^K.

The history is flattened in current-to-past order. Sensor order is the
Cartesian-product order of the selected x and y grid indices, with y varying
fastest.

Chronological splitting
-----------------------
Every trajectory is split independently:

    train:      [0, floor(0.70*T))
    validation: [floor(0.70*T), floor(0.85*T))
    test:       [floor(0.85*T), T).

A pair is admitted only when its entire input-target footprint

    [t-(m-1)tau, t+tau]

lies inside one split.

Non-overlap
-----------
The inclusive footprint length is

    m*tau + 1.

Consecutive selected footprints within one trajectory and split are
non-overlapping and have exactly ``gap`` unused time indices between them.

Execution modes
---------------
train:
    Prepare and normalize the dataset, train the drift, generate conditional
    forecast ensembles, and create the UQ plot.

inference:
    Reuse a saved NPZ dataset and trained checkpoint to generate new ensembles
    and UQ plots without retraining.

plot:
    Reuse an already saved ensemble NPZ and redraw the UQ subplot figure without either
    retraining or rerunning Euler--Maruyama. This is the fastest mode for
    changing figure size, transparency, line width, legend layout, or DPI.

Required neighboring modules
----------------------------
The canonical filenames are

    train_drift_with_visualization.py
    sampling.py

For convenience, this script also recognizes sibling files named

    train_drift_with_visualization(1).py
    sampling(1).py.
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
    """Load a canonical module, falling back to a sibling file."""
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
    """Sparse sensors, delay embedding, split, and normalization settings."""

    m: int
    tau: int
    gap: int
    sensors_x: int = 3
    sensors_y: int = 3
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
        return (self.m - 1) * self.tau

    @property
    def forecast_horizon(self) -> int:
        return self.tau

    @property
    def footprint_length(self) -> int:
        return (
            self.embedding_span
            + self.forecast_horizon
            + 1
        )

    @property
    def anchor_stride(self) -> int:
        return self.footprint_length + self.gap


@dataclass(frozen=True)
class UQConfig:
    """Conditional ensemble settings."""

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
    """Appearance settings for the one-subplot-per-sensor UQ figure."""

    figure_width: float = 15.0
    figure_height: float = 12.0
    interval_alpha: float = 0.18
    line_width: float = 1.8
    marker_size: float = 4.0
    legend_columns: int = 2
    dpi: int = 200
    title: str = (
        "2D Kolmogorov-flow sparse-sensor forecast with uncertainty"
    )

    def validate(self) -> None:
        if self.figure_width <= 0.0 or self.figure_height <= 0.0:
            raise ValueError(
                "Figure width and height must be positive."
            )
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
    """Componentwise affine standardization fitted from training pairs."""

    mean: Array
    scale: Array
    regularized_mask: Array

    def transform(self, values: Array) -> Array:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: Array) -> Array:
        return self.mean + self.scale * values


@dataclass(frozen=True)
class SensorLayout:
    """Cartesian-product sparse-sensor layout."""

    x_indices: Array
    y_indices: Array
    x_positions: Array
    y_positions: Array

    @property
    def num_sensors(self) -> int:
        return int(self.x_indices.size)


@dataclass
class PairBuffer:
    """Physical delay/forecast pairs and temporal provenance."""

    Y: List[Array]
    U: List[Array]
    trajectory_index: List[int]
    anchor_index: List[int]
    target_index: List[int]
    footprint_start: List[int]
    footprint_end: List[int]

    @classmethod
    def empty(cls) -> "PairBuffer":
        return cls(
            Y=[],
            U=[],
            trajectory_index=[],
            anchor_index=[],
            target_index=[],
            footprint_start=[],
            footprint_end=[],
        )

    def append(
        self,
        *,
        y: Array,
        u: Array,
        trajectory_index: int,
        anchor_index: int,
        target_index: int,
        footprint_start: int,
        footprint_end: int,
    ) -> None:
        self.Y.append(y)
        self.U.append(u)
        self.trajectory_index.append(trajectory_index)
        self.anchor_index.append(anchor_index)
        self.target_index.append(target_index)
        self.footprint_start.append(footprint_start)
        self.footprint_end.append(footprint_end)

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
            "target_index": np.asarray(
                self.target_index,
                dtype=np.int64,
            ),
            "footprint_start": np.asarray(
                self.footprint_start,
                dtype=np.int64,
            ),
            "footprint_end": np.asarray(
                self.footprint_end,
                dtype=np.int64,
            ),
        }


# ---------------------------------------------------------------------------
# Loading, chronological splitting, and sensor placement
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
            "Expected test.npy to have shape (N,T,X,Y); "
            f"received {trajectories.shape}."
        )
    if not np.issubdtype(trajectories.dtype, np.number):
        raise TypeError("Trajectory data must be numerical.")
    if not np.isfinite(trajectories).all():
        raise ValueError(
            "Trajectory data contain NaN or infinite values."
        )

    n_trajectories, n_times, n_x, n_y = trajectories.shape
    if n_trajectories < 1:
        raise ValueError("N must be positive.")
    if n_times < 3:
        raise ValueError("T must be at least 3.")
    if n_x < 1 or n_y < 1:
        raise ValueError(
            "Both spatial grid dimensions must be positive."
        )

    return np.asarray(trajectories, dtype=np.float32)


def compute_split_bounds(n_times: int) -> SplitBounds:
    """Return per-trajectory 70%/15%/15% chronological bounds."""
    train_end = int(np.floor(0.70 * n_times))
    validation_end = int(np.floor(0.85 * n_times))

    bounds = SplitBounds(
        train=(0, train_end),
        validation=(train_end, validation_end),
        test=(validation_end, n_times),
    )

    for split_name, (start, end) in bounds.as_dict().items():
        if end <= start:
            raise ValueError(
                f"The {split_name} split is empty for T={n_times}."
            )

    return bounds


def choose_uniform_sensor_layout(
    n_x: int,
    n_y: int,
    *,
    sensors_x: int,
    sensors_y: int,
    domain_length_x: float,
    domain_length_y: float,
) -> SensorLayout:
    """
    Place sensors at the centers of uniform bins on a periodic grid.

    The one-dimensional index sets are

        floor((k + 0.5)*n_x/sensors_x), k=0,...,sensors_x-1,
        floor((l + 0.5)*n_y/sensors_y), l=0,...,sensors_y-1.

    Their Cartesian product defines the complete sensor set.
    """
    if sensors_x > n_x:
        raise ValueError(
            f"sensors_x={sensors_x} exceeds X={n_x}."
        )
    if sensors_y > n_y:
        raise ValueError(
            f"sensors_y={sensors_y} exceeds Y={n_y}."
        )

    # Put one sensor at the center of each equal spatial bin.
    # The +0.5 shift moves the layout away from the boundary index 0.
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
            "Uniform x sensor placement produced duplicates."
        )
    if np.unique(unique_y).size != sensors_y:
        raise RuntimeError(
            "Uniform y sensor placement produced duplicates."
        )

    grid_x, grid_y = np.meshgrid(
        unique_x,
        unique_y,
        indexing="ij",
    )
    x_indices = grid_x.reshape(-1)
    y_indices = grid_y.reshape(-1)

    x_positions = (
        domain_length_x * x_indices.astype(np.float64) / n_x
    )
    y_positions = (
        domain_length_y * y_indices.astype(np.float64) / n_y
    )

    return SensorLayout(
        x_indices=x_indices,
        y_indices=y_indices,
        x_positions=x_positions,
        y_positions=y_positions,
    )


# ---------------------------------------------------------------------------
# Non-overlapping delay/forecast pairs
# ---------------------------------------------------------------------------

def anchor_indices_for_split(
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> Array:
    """
    Select anchors whose complete history-and-target footprint lies in a split.
    """
    first_anchor = split_start + config.embedding_span
    last_anchor = (
        split_end
        - 1
        - config.forecast_horizon
    )

    if first_anchor > last_anchor:
        return np.empty(0, dtype=np.int64)

    return np.arange(
        first_anchor,
        last_anchor + 1,
        config.anchor_stride,
        dtype=np.int64,
    )


def verify_footprints(
    anchors: Array,
    *,
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> None:
    """Verify split containment, non-overlap, and the exact requested gap."""
    if anchors.size == 0:
        return

    starts = anchors - config.embedding_span
    ends = anchors + config.forecast_horizon

    if starts[0] < split_start or ends[-1] >= split_end:
        raise RuntimeError(
            "A delay/forecast footprint crosses a split boundary."
        )

    if anchors.size > 1:
        unused_indices = starts[1:] - ends[:-1] - 1
        if np.any(unused_indices != config.gap):
            raise RuntimeError(
                "Selected footprints do not have the requested gap."
            )


def observe_sensors(
    trajectory: Array,
    time_indices: Array,
    sensors: SensorLayout,
) -> Array:
    """
    Extract all sparse sensors at each requested time.

    Returns shape (len(time_indices), num_sensors).
    """
    snapshots = trajectory[time_indices]
    return snapshots[
        :,
        sensors.x_indices,
        sensors.y_indices,
    ]


def build_split_pairs(
    trajectories: Array,
    *,
    split_name: str,
    bounds: Tuple[int, int],
    sensors: SensorLayout,
    config: DataConfig,
) -> Dict[str, Array]:
    """
    Construct physical delay inputs and future sparse-sensor targets.
    """
    split_start, split_end = bounds
    anchors = anchor_indices_for_split(
        split_start,
        split_end,
        config,
    )
    verify_footprints(
        anchors,
        split_start=split_start,
        split_end=split_end,
        config=config,
    )

    condition_dim = sensors.num_sensors * config.m
    target_dim = sensors.num_sensors
    delay_offsets = (
        np.arange(config.m, dtype=np.int64) * config.tau
    )

    buffer = PairBuffer.empty()

    for trajectory_index, trajectory in enumerate(trajectories):
        for anchor_index in anchors:
            delay_indices = anchor_index - delay_offsets
            target_index = (
                anchor_index + config.forecast_horizon
            )

            sensor_history = observe_sensors(
                trajectory,
                delay_indices,
                sensors,
            )
            condition = sensor_history.reshape(condition_dim)

            target = trajectory[
                target_index,
                sensors.x_indices,
                sensors.y_indices,
            ]

            buffer.append(
                y=condition,
                u=target,
                trajectory_index=trajectory_index,
                anchor_index=int(anchor_index),
                target_index=int(target_index),
                footprint_start=int(delay_indices[-1]),
                footprint_end=int(target_index),
            )

    return buffer.finalize(
        split_name=split_name,
        condition_dim=condition_dim,
        target_dim=target_dim,
    )


# ---------------------------------------------------------------------------
# Training-only normalization and NPZ construction
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


def prepare_kolmogorov_forecast_data(
    trajectories: Array,
    *,
    config: DataConfig,
) -> Dict[str, Any]:
    """Construct splits and standardize them using training pairs only."""
    config.validate()

    n_trajectories, n_times, n_x, n_y = trajectories.shape
    split_bounds = compute_split_bounds(n_times)

    sensors = choose_uniform_sensor_layout(
        n_x,
        n_y,
        sensors_x=config.sensors_x,
        sensors_y=config.sensors_y,
        domain_length_x=config.domain_length_x,
        domain_length_y=config.domain_length_y,
    )

    raw_splits = {
        split_name: build_split_pairs(
            trajectories,
            split_name=split_name,
            bounds=bounds,
            sensors=sensors,
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
    """Build the training-compatible NPZ dictionary and provenance arrays."""
    normalized = prepared["normalized_splits"]
    raw = prepared["raw_splits"]
    split_bounds: SplitBounds = prepared["split_bounds"]
    sensors: SensorLayout = prepared["sensors"]
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

        # Needed for physical-coordinate inference.
        "mu_Y": y_standardizer.mean,
        "sigma_Y": y_standardizer.scale,
        "mu_U": u_standardizer.mean,
        "sigma_U": u_standardizer.scale,

        # Physical-coordinate copies for interpretation and plotting.
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

        # Split and construction metadata.
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
        arrays[f"target_index_{prefix}"] = raw[split_name][
            "target_index"
        ]
        arrays[f"footprint_start_{prefix}"] = raw[split_name][
            "footprint_start"
        ]
        arrays[f"footprint_end_{prefix}"] = raw[split_name][
            "footprint_end"
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
        "target_definition": "P(X_{t+tau})",
        "embedding_dimension_m": int(config.m),
        "time_delay_tau_indices": int(config.tau),
        "forecast_horizon_indices": int(config.tau),
        "gap_indices": int(config.gap),
        "footprint_length_indices": int(
            config.footprint_length
        ),
        "anchor_stride_indices": int(config.anchor_stride),
        "sensors_x": int(config.sensors_x),
        "sensors_y": int(config.sensors_y),
        "num_sensors": int(config.num_sensors),
        "condition_dimension": int(
            config.num_sensors * config.m
        ),
        "target_dimension": int(config.num_sensors),
        "sensor_x_indices": sensors.x_indices.tolist(),
        "sensor_y_indices": sensors.y_indices.tolist(),
        "sensor_x_positions": sensors.x_positions.tolist(),
        "sensor_y_positions": sensors.y_positions.tolist(),
        "domain_length_x": float(config.domain_length_x),
        "domain_length_y": float(config.domain_length_y),
        "split_rule": (
            "first 70 percent train, next 15 percent validation, "
            "final 15 percent test within every trajectory"
        ),
        "normalization_rule": (
            "componentwise means and standard deviations fitted "
            "from physical training pairs only"
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
    """Call the provided drift-training implementation."""
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
# Loading saved data for inference-only mode
# ---------------------------------------------------------------------------

def load_prepared_test_data_from_npz(
    dataset_path: str | Path,
) -> Tuple[
    Dict[str, Any],
    DataConfig,
    SensorLayout,
    Dict[str, Any],
]:
    """Load the held-out split and metadata required for checkpoint reuse."""
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
        "target_index_test",
        "footprint_start_test",
        "footprint_end_test",
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
        "domain_length_x",
        "domain_length_y",
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
            "target_index": np.asarray(
                data["target_index_test"],
                dtype=np.int64,
            ),
            "footprint_start": np.asarray(
                data["footprint_start_test"],
                dtype=np.int64,
            ),
            "footprint_end": np.asarray(
                data["footprint_end_test"],
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

        saved_metadata = (
            json.loads(str(np.asarray(data["metadata_json"]).item()))
            if "metadata_json" in data.files
            else {}
        )

    if normalized_y.ndim != 2 or normalized_u.ndim != 2:
        raise ValueError(
            "Y_test and U_test must be matrices."
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
            "Saved condition dimension is inconsistent with m "
            "and the sensor layout."
        )
    if normalized_u.shape[1] != config.num_sensors:
        raise ValueError(
            "Saved target dimension is inconsistent with the "
            "sensor layout."
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
        "saved_metadata": saved_metadata,
    }

    return prepared, config, sensors, metadata


# ---------------------------------------------------------------------------
# Conditional ensemble generation
# ---------------------------------------------------------------------------

def select_evenly_spaced_indices(
    population_size: int,
    requested_count: int,
) -> Array:
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
    """Generate future sparse-sensor ensembles at selected test anchors."""
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

    trajectory_candidates = np.flatnonzero(
        raw_test["trajectory_index"]
        == uq_config.trajectory_index
    )

    if trajectory_candidates.size == 0:
        available = np.unique(
            raw_test["trajectory_index"]
        ).astype(int).tolist()
        raise ValueError(
            f"No test conditions exist for trajectory "
            f"{uq_config.trajectory_index}. Available trajectories: "
            f"{available}."
        )

    order = np.argsort(
        raw_test["anchor_index"][trajectory_candidates]
    )
    trajectory_candidates = trajectory_candidates[order]

    selected_positions = select_evenly_spaced_indices(
        trajectory_candidates.size,
        uq_config.num_test_conditions,
    )
    selected_indices = trajectory_candidates[selected_positions]

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
        "true_future_observation": raw_test["U"][
            selected_indices
        ],
        "normalized_true_future_observation": (
            normalized_test["U"][selected_indices]
        ),
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
        "target_index": raw_test["target_index"][
            selected_indices
        ],
        "footprint_start": raw_test["footprint_start"][
            selected_indices
        ],
        "footprint_end": raw_test["footprint_end"][
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
    """Reject mismatched model dimensions or normalization statistics."""
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

    saved_stats = metadata["normalization_stats"]
    checkpoint_mapping = checkpoint_stats.as_dict()

    for key in ("mu_Y", "sigma_Y", "mu_U", "sigma_U"):
        left = np.asarray(
            checkpoint_mapping[key],
            dtype=np.float32,
        ).reshape(-1)
        right = np.asarray(
            saved_stats[key],
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
                "Reuse files from the same training run."
            )


# ---------------------------------------------------------------------------
# Saving and loading ensembles
# ---------------------------------------------------------------------------

def save_uq_samples(
    output_path: str | Path,
    ensembles: Mapping[str, Any],
    *,
    sensors: SensorLayout,
    data_config: DataConfig,
    uq_config: UQConfig,
) -> Path:
    """Save generated ensembles, truths, anchors, sensors, and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interpolant_config = ensembles["interpolant_config"]

    metadata = {
        "target": "P(X_{t+tau})",
        "embedding_dimension_m": int(data_config.m),
        "time_delay_tau_indices": int(data_config.tau),
        "forecast_horizon_indices": int(data_config.tau),
        "gap_indices": int(data_config.gap),
        "sensors_x": int(data_config.sensors_x),
        "sensors_y": int(data_config.sensors_y),
        "num_sensors": int(data_config.num_sensors),
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
        true_future_observation=ensembles[
            "true_future_observation"
        ],
        normalized_true_future_observation=ensembles[
            "normalized_true_future_observation"
        ],
        samples=ensembles["samples"],
        normalized_samples=ensembles["normalized_samples"],
        trajectory_index=ensembles["trajectory_index"],
        anchor_index=ensembles["anchor_index"],
        target_index=ensembles["target_index"],
        footprint_start=ensembles["footprint_start"],
        footprint_end=ensembles["footprint_end"],
        sensor_x_indices=sensors.x_indices,
        sensor_y_indices=sensors.y_indices,
        sensor_x_positions=sensors.x_positions,
        sensor_y_positions=sensors.y_positions,
        m=np.asarray(data_config.m, dtype=np.int64),
        tau=np.asarray(data_config.tau, dtype=np.int64),
        gap=np.asarray(data_config.gap, dtype=np.int64),
        sensors_x=np.asarray(
            data_config.sensors_x,
            dtype=np.int64,
        ),
        sensors_y=np.asarray(
            data_config.sensors_y,
            dtype=np.int64,
        ),
        domain_length_x=np.asarray(
            data_config.domain_length_x,
            dtype=np.float64,
        ),
        domain_length_y=np.asarray(
            data_config.domain_length_y,
            dtype=np.float64,
        ),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )

    return output_path


def load_saved_uq_samples(
    samples_path: str | Path,
) -> Tuple[Dict[str, Any], SensorLayout, DataConfig]:
    """Load a saved ensemble file for plot-only mode."""
    samples_path = Path(samples_path)
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Saved ensemble file not found: {samples_path}"
        )

    required = {
        "samples",
        "true_future_observation",
        "trajectory_index",
        "anchor_index",
        "target_index",
        "sensor_x_indices",
        "sensor_y_indices",
        "sensor_x_positions",
        "sensor_y_positions",
        "m",
        "tau",
        "gap",
        "sensors_x",
        "sensors_y",
        "domain_length_x",
        "domain_length_y",
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
            "true_future_observation": np.asarray(
                data["true_future_observation"],
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
            "target_index": np.asarray(
                data["target_index"],
                dtype=np.int64,
            ),
        }

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
            domain_length_x=float(
                np.asarray(data["domain_length_x"]).item()
            ),
            domain_length_y=float(
                np.asarray(data["domain_length_y"]).item()
            ),
        )

    return ensembles, sensors, config


# ---------------------------------------------------------------------------
# One-subplot-per-sensor UQ figure
# ---------------------------------------------------------------------------

def central_interval(
    samples: Array,
    level: float,
) -> Tuple[Array, Array]:
    """Compute componentwise central empirical intervals."""
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


def plot_all_sensor_forecasts(
    ensembles: Mapping[str, Any],
    *,
    sensors: SensorLayout,
    data_config: DataConfig,
    interval_level: float,
    plot_config: PlotConfig,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Create one figure with a separate subplot for every sparse sensor.

    Each subplot shows that sensor's conditional ensemble mean, central
    empirical interval, and true future value across the selected anchor times.
    """
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
        ensembles["true_future_observation"],
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
            "(conditions, ensemble_size, sensors)."
        )
    if samples.shape[2] != sensors.num_sensors:
        raise ValueError(
            "The ensemble target dimension does not match the sensor layout."
        )
    if truths.shape != (
        samples.shape[0],
        sensors.num_sensors,
    ):
        raise ValueError(
            "Truth array shape is inconsistent with samples."
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
            "The anchor-time subplot figure requires conditions from "
            "one trajectory."
        )

    order = np.argsort(anchors)
    anchors = anchors[order]
    samples = samples[order]
    truths = truths[order]

    lower, upper = central_interval(
        samples,
        interval_level,
    )
    forecast_mean = samples.mean(axis=1)

    coverage = (
        (truths >= lower)
        & (truths <= upper)
    ).mean(axis=0)
    rmse = np.sqrt(
        np.mean(
            (forecast_mean - truths) ** 2,
            axis=0,
        )
    )
    mean_width = (upper - lower).mean(axis=0)

    num_sensors = sensors.num_sensors

    # Use an approximately square subplot grid. For the default 3x3 sensor
    # layout this produces a 3x3 figure.
    subplot_columns = int(np.ceil(np.sqrt(num_sensors)))
    subplot_rows = int(
        np.ceil(num_sensors / subplot_columns)
    )

    figure, axes = plt.subplots(
        subplot_rows,
        subplot_columns,
        figsize=(
            plot_config.figure_width,
            plot_config.figure_height,
        ),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)

    for sensor_index, axis in enumerate(flat_axes):
        if sensor_index >= num_sensors:
            axis.set_visible(False)
            continue

        sensor_label = (
            f"Sensor {sensor_index + 1}: "
            f"(x={sensors.x_positions[sensor_index]:.3g}, "
            f"y={sensors.y_positions[sensor_index]:.3g})"
        )

        mean_line, = axis.plot(
            anchors,
            forecast_mean[:, sensor_index],
            marker="o",
            markersize=plot_config.marker_size,
            linewidth=plot_config.line_width,
            label="Forecast mean",
            zorder=3,
        )
        sensor_color = mean_line.get_color()

        axis.fill_between(
            anchors,
            lower[:, sensor_index],
            upper[:, sensor_index],
            color=sensor_color,
            alpha=plot_config.interval_alpha,
            linewidth=0.0,
            label=f"{interval_level:.0%} interval",
            zorder=1,
        )

        axis.plot(
            anchors,
            truths[:, sensor_index],
            color=sensor_color,
            marker="x",
            markersize=plot_config.marker_size,
            linewidth=plot_config.line_width,
            linestyle="--",
            label="True value",
            zorder=4,
        )

        axis.set_title(
            f"{sensor_label}\n"
            f"coverage={coverage[sensor_index]:.1%}, "
            f"RMSE={rmse[sensor_index]:.3e}"
        )
        axis.grid(True)
        axis.legend(
            ncol=plot_config.legend_columns,
            fontsize="small",
            loc="best",
        )

    trajectory_index = int(trajectories[0])

    figure.supxlabel("Anchor time index t")
    figure.supylabel("Forecasted sensor value")
    figure.suptitle(
        f"{plot_config.title}\n"
        f"trajectory={trajectory_index}, "
        f"target=P(X_(t+tau)), tau={data_config.tau}, "
        f"central interval={interval_level:.0%}",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    figure.savefig(
        output_path,
        dpi=plot_config.dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    return {
        "plot_path": output_path,
        "anchor_indices": anchors,
        "forecast_mean": forecast_mean,
        "lower": lower,
        "upper": upper,
        "truth": truths,
        "sensor_coverage": coverage,
        "sensor_rmse": rmse,
        "sensor_mean_interval_width": mean_width,
        "trajectory_index": trajectory_index,
        "subplot_rows": subplot_rows,
        "subplot_columns": subplot_columns,
    }


def create_uq_visualization(
    ensembles: Mapping[str, Any],
    *,
    sensors: SensorLayout,
    data_config: DataConfig,
    uq_config: UQConfig,
    plot_config: PlotConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Create the sensor-subplot figure and save its numerical plotting data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = plot_all_sensor_forecasts(
        ensembles,
        sensors=sensors,
        data_config=data_config,
        interval_level=uq_config.interval_level,
        plot_config=plot_config,
        output_path=(
            output_dir
            / "sensor_forecast_uq_subplots.png"
        ),
    )

    metrics_path = output_dir / "forecast_uq_metrics.npz"
    np.savez_compressed(
        metrics_path,
        anchor_indices=result["anchor_indices"],
        forecast_mean=result["forecast_mean"],
        lower=result["lower"],
        upper=result["upper"],
        truth=result["truth"],
        sensor_coverage=result["sensor_coverage"],
        sensor_rmse=result["sensor_rmse"],
        sensor_mean_interval_width=result[
            "sensor_mean_interval_width"
        ],
        sensor_x_indices=sensors.x_indices,
        sensor_y_indices=sensors.y_indices,
        sensor_x_positions=sensors.x_positions,
        sensor_y_positions=sensors.y_positions,
        interval_level=np.asarray(
            uq_config.interval_level,
            dtype=np.float64,
        ),
    )

    return {
        "forecast_uq_plot": str(result["plot_path"]),
        "metrics_file": str(metrics_path),
        "trajectory_index": int(result["trajectory_index"]),
        "mean_coverage": float(
            np.mean(result["sensor_coverage"])
        ),
        "mean_rmse": float(
            np.mean(result["sensor_rmse"])
        ),
        "mean_interval_width": float(
            np.mean(result["sensor_mean_interval_width"])
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
    """Prepare data, train, sample, and plot."""
    data_config.validate()
    train_config.validate()
    interpolant_config.validate()
    uq_config.validate()
    plot_config.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(input_path)
    prepared = prepare_kolmogorov_forecast_data(
        trajectories,
        config=data_config,
    )

    dataset_path = save_dataset_npz(
        output_dir / "kolmogorov_forecast_data.npz",
        dataset_arrays_for_npz(
            prepared,
            config=data_config,
        ),
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
    sensors: SensorLayout = prepared["sensors"]

    samples_path = save_uq_samples(
        uq_dir / "conditional_forecast_samples.npz",
        ensembles,
        sensors=sensors,
        data_config=data_config,
        uq_config=uq_config,
    )
    visualization = create_uq_visualization(
        ensembles,
        sensors=sensors,
        data_config=data_config,
        uq_config=uq_config,
        plot_config=plot_config,
        output_dir=uq_dir,
    )

    summary = {
        "mode": "train",
        "input_file": str(Path(input_path)),
        "output_directory": str(output_dir),
        "dataset_file": str(dataset_path),
        "checkpoint": str(checkpoint_path),
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
        "target_dimension": int(data_config.num_sensors),
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
    """Generate new ensembles and plots from a checkpoint without training."""
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
        sensors,
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
        output_dir / "conditional_forecast_samples.npz",
        ensembles,
        sensors=sensors,
        data_config=data_config,
        uq_config=uq_config,
    )
    visualization = create_uq_visualization(
        ensembles,
        sensors=sensors,
        data_config=data_config,
        uq_config=uq_config,
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
    """Redraw the UQ figure from saved ensembles without sampling or training."""
    ensembles, sensors, data_config = load_saved_uq_samples(
        samples_path
    )

    uq_config = UQConfig(
        num_test_conditions=int(
            ensembles["samples"].shape[0]
        ),
        ensemble_size=int(
            ensembles["samples"].shape[1]
        ),
        interval_level=interval_level,
        trajectory_index=int(
            ensembles["trajectory_index"][0]
        ),
    )

    visualization = create_uq_visualization(
        ensembles,
        sensors=sensors,
        data_config=data_config,
        uq_config=uq_config,
        plot_config=plot_config,
        output_dir=output_dir,
    )

    return {
        "mode": "plot",
        "samples_file": str(Path(samples_path)),
        "output_directory": str(Path(output_dir)),
        "interval_level": float(interval_level),
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
            "sparse-sensor forecast model."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("test.npy"),
        help="Trajectory file with shape (N,T,X,Y); used in train mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "inference", "plot"),
        default="train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kolmogorov_forecast_output"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Prepared dataset for inference mode. Defaults to "
            "OUTPUT_DIR/kolmogorov_forecast_data.npz."
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
            "Saved ensemble NPZ for plot mode. If omitted, the script "
            "checks OUTPUT_DIR/inference_uq and then OUTPUT_DIR/uq."
        ),
    )
    parser.add_argument(
        "--inference-output-dir",
        type=Path,
        default=None,
        help=(
            "Inference-only output directory. Defaults to "
            "OUTPUT_DIR/inference_uq."
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

    # Data construction: required only in train mode.
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--gap", type=int, default=None)
    parser.add_argument(
        "--sensors-x",
        type=int,
        default=3,
        help="Number of uniformly placed sensor coordinates in x.",
    )
    parser.add_argument(
        "--sensors-y",
        type=int,
        default=3,
        help="Number of uniformly placed sensor coordinates in y.",
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
            "Physical x-domain length used only for sensor-position labels."
        ),
    )
    parser.add_argument(
        "--domain-length-y",
        type=float,
        default=1.0,
        help=(
            "Physical y-domain length used only for sensor-position labels."
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
        "--figure-width",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--interval-alpha",
        type=float,
        default=0.18,
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
        default=2,
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--plot-title",
        type=str,
        default=(
            "2D Kolmogorov-flow sparse-sensor forecast "
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
        / "conditional_forecast_samples.npz",
        output_dir
        / "uq"
        / "conditional_forecast_samples.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Plot mode could not find a saved ensemble file. "
        "Supply --samples-path explicitly."
    )


def main() -> None:
    args = parse_args()

    plot_config = PlotConfig(
        figure_width=args.figure_width,
        figure_height=args.figure_height,
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
                / "kolmogorov_forecast_data.npz"
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
