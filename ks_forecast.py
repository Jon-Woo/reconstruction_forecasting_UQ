#!/usr/bin/env python3
"""
End-to-end stochastic KS partial-observation forecasting pipeline.

This script is analogous to the POD-reconstruction pipeline, except that the
target is the future partial observation at time t + tau:

    U_t = P(X_{t+tau}) in R^4.

The conditioning variable is a delay vector built from the same four uniformly
spaced sensors:

    Y_t = [
        P(X_t),
        P(X_{t-tau}),
        ...,
        P(X_{t-(m-1)tau})
    ] in R^(4m).

The script:

1. Loads stochastic Kuramoto--Sivashinsky trajectories from test.npy with
   shape (N,T,D).
2. Splits every trajectory chronologically into 70% training, 15% validation,
   and 15% test portions.
3. Chooses four uniformly spaced sensors on the periodic spatial grid.
4. Constructs non-overlapping input-target footprints with a specified gap.
5. Uses P(X_{t+tau}) as the target.
6. Fits condition and target normalization using training pairs only.
7. Saves an NPZ file compatible with train_drift_with_visualization.py.
8. Calls train_conditional_drift from train_drift_with_visualization.py.
9. Uses the Euler--Maruyama sampler implemented in sampling.py.
10. Creates one uncertainty-quantification figure showing the forecast mean, a shaded 90% interval, and the truth across anchor times.
11. Optionally reuses a saved dataset and trained checkpoint in inference-only
    mode, generating new Euler--Maruyama ensembles and UQ plots without
    retraining.

Required neighboring files
--------------------------
Place this script in the same directory as

    train_drift_with_visualization.py
    sampling.py

Execution modes
---------------
Training mode prepares the forecast dataset, trains the conditional drift,
generates conditional ensembles, and creates the UQ figure.

Inference-only mode loads

    ks_partial_forecast_data.npz
    training/conditional_drift_checkpoint.pt

from a previous training run. It generates new stochastic ensembles and the UQ
figure without loading test.npy, rebuilding the data pairs, or calling
train_conditional_drift.

Input
-----
The input file must contain an array with shape

    (N,T,D),

where

    N = number of trajectories,
    T = number of stored time indices,
    D = number of spatial grid points.

The periodic domain length defaults to L=22.

Chronological split
-------------------
For every trajectory independently:

    training:   [0, floor(0.70 T))
    validation: [floor(0.70 T), floor(0.85 T))
    test:       [floor(0.85 T), T).

A pair belongs to a split only when its entire temporal footprint

    [t-(m-1)tau, t+tau]

lies inside that split. Thus, neither the delay history nor the forecast target
crosses a split boundary.

Non-overlapping footprints
--------------------------
The full input-target footprint has inclusive length

    m*tau + 1.

Within each trajectory and split, consecutive selected footprints are
non-overlapping. If one footprint ends at index e, the next footprint starts at

    e + gap + 1,

so exactly ``gap`` unused time indices separate consecutive footprints.

Output
------
The output directory contains:

    ks_partial_forecast_data.npz
    training/conditional_drift_checkpoint.pt
    training/training_validation_loss_log.png
    training/network_architecture.png
    uq/conditional_partial_forecast_samples.npz
    uq/forecast_uq_over_anchor_times.png
    uq/forecast_uq_metrics.npz
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
    """Delay-pair construction and normalization configuration."""

    m: int
    tau: int
    gap: int
    normalization_epsilon: float = 1.0e-8
    domain_length: float = 22.0

    def validate(self) -> None:
        if self.m < 1:
            raise ValueError("m must be a positive integer.")
        if self.tau < 1:
            raise ValueError("tau must be a positive integer.")
        if self.gap < 0:
            raise ValueError("gap must be nonnegative.")
        if self.normalization_epsilon <= 0.0:
            raise ValueError(
                "normalization_epsilon must be positive."
            )
        if self.domain_length <= 0.0:
            raise ValueError("domain_length must be positive.")

    @property
    def embedding_span(self) -> int:
        """Distance from the oldest delay observation to the anchor time."""
        return (self.m - 1) * self.tau

    @property
    def forecast_horizon(self) -> int:
        """Forecast horizon in stored time-index units."""
        return self.tau

    @property
    def footprint_length(self) -> int:
        """
        Number of indices in [t-(m-1)tau, t+tau], including endpoints.
        """
        return (
            self.embedding_span
            + self.forecast_horizon
            + 1
        )

    @property
    def anchor_stride(self) -> int:
        """
        Distance between consecutive selected anchors.

        This makes complete input-target footprints non-overlapping and leaves
        exactly ``gap`` unused indices between them.
        """
        return self.footprint_length + self.gap


@dataclass(frozen=True)
class UQConfig:
    """Conditional ensemble and UQ-plot configuration."""

    num_test_conditions: int = 12
    ensemble_size: int = 500
    em_steps: int = 500
    sample_batch_size: int = 512
    interval_level: float = 0.90
    trajectory_index: int = 0
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
        if self.trajectory_index < 0:
            raise ValueError("trajectory_index must be nonnegative.")


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


@dataclass
class PairBuffer:
    """Raw delay/forecast pairs and their temporal provenance."""

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
# Trajectory loading, splitting, and sensors
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
    Choose four uniformly spaced sensors on a periodic spatial grid.

    The selected indices are floor(kD/4), k=0,1,2,3. This avoids treating
    the last grid point as a repeated periodic endpoint.
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
# Forecast-pair construction
# ---------------------------------------------------------------------------

def anchor_indices_for_split(
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> Array:
    """
    Return anchors whose complete input-target footprints lie in one split.

    For an anchor t, the complete footprint is

        [t-(m-1)tau, t+tau].

    The first footprint begins at split_start. The final target lies no later
    than split_end-1.
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


def verify_forecast_footprints(
    anchors: Array,
    *,
    split_start: int,
    split_end: int,
    config: DataConfig,
) -> None:
    """Verify split containment and the exact requested gap."""
    if anchors.size == 0:
        return

    starts = anchors - config.embedding_span
    ends = anchors + config.forecast_horizon

    if starts[0] < split_start or ends[-1] >= split_end:
        raise RuntimeError(
            "A selected input-target footprint crosses a split boundary."
        )

    if anchors.size > 1:
        unused_indices = starts[1:] - ends[:-1] - 1
        if np.any(unused_indices != config.gap):
            raise RuntimeError(
                "Forecast footprints do not have the requested exact gap."
            )


def build_split_pairs(
    trajectories: Array,
    *,
    split_name: str,
    bounds: Tuple[int, int],
    sensor_indices: Array,
    config: DataConfig,
) -> Dict[str, Array]:
    """
    Construct raw sensor-delay inputs and future sensor targets.

    For every selected anchor t:

        Y_t = [P(X_t), ..., P(X_{t-(m-1)tau})],
        U_t = P(X_{t+tau}).
    """
    split_start, split_end = bounds

    anchors = anchor_indices_for_split(
        split_start,
        split_end,
        config,
    )
    verify_forecast_footprints(
        anchors,
        split_start=split_start,
        split_end=split_end,
        config=config,
    )

    condition_dim = NUM_SENSORS * config.m
    target_dim = NUM_SENSORS
    delay_offsets = (
        np.arange(config.m, dtype=np.int64) * config.tau
    )

    buffer = PairBuffer.empty()

    for trajectory_index in range(trajectories.shape[0]):
        trajectory = trajectories[trajectory_index]

        for anchor_index in anchors:
            delay_indices = anchor_index - delay_offsets
            target_index = (
                anchor_index + config.forecast_horizon
            )

            # Shape (m,4), ordered current-to-past.
            sensor_history = trajectory[
                np.ix_(delay_indices, sensor_indices)
            ]
            y = sensor_history.reshape(condition_dim)

            # Future partial observation P(X_{t+tau}) in R^4.
            u = trajectory[target_index, sensor_indices]

            buffer.append(
                y=y,
                u=u,
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
# Normalization and training-compatible NPZ
# ---------------------------------------------------------------------------

def fit_standardizer(
    values: Array,
    *,
    epsilon: float,
) -> Standardizer:
    """Fit componentwise mean and population standard deviation."""
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


def prepare_partial_forecast_data(
    trajectories: Array,
    *,
    config: DataConfig,
) -> Dict[str, Any]:
    """
    Construct raw split pairs and normalize them using training pairs only.
    """
    config.validate()

    n_trajectories, n_times, n_grid = trajectories.shape
    split_bounds = compute_split_bounds(n_times)

    sensor_indices, sensor_positions = choose_uniform_periodic_sensors(
        n_grid,
        domain_length=config.domain_length,
    )

    raw_splits = {
        name: build_split_pairs(
            trajectories,
            split_name=name,
            bounds=bounds,
            sensor_indices=sensor_indices,
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
            "Y": y_standardizer.transform(split["Y"]).astype(
                np.float32
            ),
            "U": u_standardizer.transform(split["U"]).astype(
                np.float32
            ),
        }
        for name, split in raw_splits.items()
    }

    return {
        "split_bounds": split_bounds,
        "sensor_indices": sensor_indices,
        "sensor_positions": sensor_positions,
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
    """Assemble required training arrays and additional provenance."""
    normalized = prepared["normalized_splits"]
    raw = prepared["raw_splits"]
    split_bounds: SplitBounds = prepared["split_bounds"]
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
        "mu_Y": y_standardizer.mean,
        "sigma_Y": y_standardizer.scale,
        "mu_U": u_standardizer.mean,
        "sigma_U": u_standardizer.scale,

        # Physical-coordinate copies for interpretation and UQ.
        "Y_train_physical": raw["train"]["Y"],
        "U_train_physical": raw["train"]["U"],
        "Y_val_physical": raw["validation"]["Y"],
        "U_val_physical": raw["validation"]["U"],
        "Y_test_physical": raw["test"]["Y"],
        "U_test_physical": raw["test"]["U"],

        # Sensor and preprocessing metadata.
        "regularized_Y_channels": y_standardizer.regularized_mask,
        "regularized_U_channels": u_standardizer.regularized_mask,
        "sensor_indices": prepared["sensor_indices"],
        "sensor_positions": prepared["sensor_positions"].astype(
            np.float64
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
        "m": np.asarray(config.m, dtype=np.int64),
        "tau": np.asarray(config.tau, dtype=np.int64),
        "forecast_horizon": np.asarray(
            config.forecast_horizon,
            dtype=np.int64,
        ),
        "gap": np.asarray(config.gap, dtype=np.int64),
        "domain_length": np.asarray(
            config.domain_length,
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
        arrays[f"target_index_{prefix}"] = raw[name][
            "target_index"
        ]
        arrays[f"footprint_start_{prefix}"] = raw[name][
            "footprint_start"
        ]
        arrays[f"footprint_end_{prefix}"] = raw[name][
            "footprint_end"
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
        "forecast_horizon_indices": int(config.forecast_horizon),
        "gap_indices": int(config.gap),
        "footprint_length_indices": int(config.footprint_length),
        "anchor_stride_indices": int(config.anchor_stride),
        "condition_dimension": int(NUM_SENSORS * config.m),
        "target": "four sensor values P(X_{t+tau})",
        "target_dimension": NUM_SENSORS,
        "split_rule": (
            "first 70 percent train, next 15 percent validation, "
            "final 15 percent test within every trajectory"
        ),
        "pair_containment": (
            "the complete interval [t-(m-1)tau, t+tau] lies within "
            "one split"
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
    """Save the training-compatible dataset in compressed NPZ format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_prepared_forecast_data_from_npz(
    dataset_path: str | Path,
) -> Tuple[
    Dict[str, Any],
    DataConfig,
    Array,
    Array,
    Dict[str, Any],
]:
    """
    Load the saved test split and metadata needed for inference-only sampling.

    The NPZ file must have been produced by this forecasting pipeline. Only the
    held-out test conditions and targets are loaded because inference mode does
    not refit normalization, rebuild pairs, or train the drift.

    Returns
    -------
    prepared:
        Minimal data structure accepted by generate_test_ensembles.
    data_config:
        The original m, tau, gap, and domain length saved with the dataset.
    sensor_indices, sensor_positions:
        The four sensor locations used during training.
    metadata:
        Dimensions, normalization statistics, and saved JSON metadata used for
        checkpoint compatibility checks and reporting.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared forecast dataset not found: {dataset_path}"
        )

    required_keys = {
        "Y_test",
        "U_test",
        "Y_test_physical",
        "U_test_physical",
        "trajectory_index_test",
        "anchor_index_test",
        "target_index_test",
        "footprint_start_test",
        "footprint_end_test",
        "sensor_indices",
        "sensor_positions",
        "mu_Y",
        "sigma_Y",
        "mu_U",
        "sigma_U",
        "m",
        "tau",
        "gap",
        "domain_length",
    }

    with np.load(dataset_path, allow_pickle=False) as data:
        missing = sorted(required_keys.difference(data.files))
        if missing:
            raise KeyError(
                "The prepared forecast dataset is missing required arrays: "
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

        trajectory_index = np.asarray(
            data["trajectory_index_test"],
            dtype=np.int64,
        )
        anchor_index = np.asarray(
            data["anchor_index_test"],
            dtype=np.int64,
        )
        target_index = np.asarray(
            data["target_index_test"],
            dtype=np.int64,
        )
        footprint_start = np.asarray(
            data["footprint_start_test"],
            dtype=np.int64,
        )
        footprint_end = np.asarray(
            data["footprint_end_test"],
            dtype=np.int64,
        )

        sensor_indices = np.asarray(
            data["sensor_indices"],
            dtype=np.int64,
        ).reshape(-1)
        sensor_positions = np.asarray(
            data["sensor_positions"],
            dtype=np.float64,
        ).reshape(-1)

        m = int(np.asarray(data["m"]).item())
        tau = int(np.asarray(data["tau"]).item())
        gap = int(np.asarray(data["gap"]).item())
        domain_length = float(
            np.asarray(data["domain_length"]).item()
        )

        data_config = DataConfig(
            m=m,
            tau=tau,
            gap=gap,
            domain_length=domain_length,
        )
        data_config.validate()

        saved_metadata = (
            json.loads(str(np.asarray(data["metadata_json"]).item()))
            if "metadata_json" in data.files
            else {}
        )

        normalization_stats = {
            "mu_Y": np.asarray(data["mu_Y"], dtype=np.float32),
            "sigma_Y": np.asarray(
                data["sigma_Y"],
                dtype=np.float32,
            ),
            "mu_U": np.asarray(data["mu_U"], dtype=np.float32),
            "sigma_U": np.asarray(
                data["sigma_U"],
                dtype=np.float32,
            ),
        }

    if normalized_y.ndim != 2 or normalized_u.ndim != 2:
        raise ValueError(
            "Y_test and U_test must both be two-dimensional."
        )

    num_pairs = normalized_y.shape[0]
    pair_arrays = {
        "U_test": normalized_u,
        "Y_test_physical": physical_y,
        "U_test_physical": physical_u,
        "trajectory_index_test": trajectory_index,
        "anchor_index_test": anchor_index,
        "target_index_test": target_index,
        "footprint_start_test": footprint_start,
        "footprint_end_test": footprint_end,
    }

    for name, values in pair_arrays.items():
        if values.shape[0] != num_pairs:
            raise ValueError(
                f"{name} has {values.shape[0]} pairs, while Y_test has "
                f"{num_pairs}."
            )

    if physical_y.shape != normalized_y.shape:
        raise ValueError(
            "Y_test_physical must have the same shape as Y_test."
        )
    if physical_u.shape != normalized_u.shape:
        raise ValueError(
            "U_test_physical must have the same shape as U_test."
        )

    expected_condition_dim = NUM_SENSORS * data_config.m
    if normalized_y.shape[1] != expected_condition_dim:
        raise ValueError(
            "Saved condition dimension is inconsistent with m and the four "
            f"sensors: expected {expected_condition_dim}, received "
            f"{normalized_y.shape[1]}."
        )
    if normalized_u.shape[1] != NUM_SENSORS:
        raise ValueError(
            f"The forecast target must have dimension {NUM_SENSORS}; "
            f"received {normalized_u.shape[1]}."
        )

    if sensor_indices.shape != (NUM_SENSORS,):
        raise ValueError(
            f"sensor_indices must have shape ({NUM_SENSORS},)."
        )
    if sensor_positions.shape != (NUM_SENSORS,):
        raise ValueError(
            f"sensor_positions must have shape ({NUM_SENSORS},)."
        )

    for key, expected_size in (
        ("mu_Y", expected_condition_dim),
        ("sigma_Y", expected_condition_dim),
        ("mu_U", NUM_SENSORS),
        ("sigma_U", NUM_SENSORS),
    ):
        values = normalization_stats[key].reshape(-1)
        if values.size != expected_size:
            raise ValueError(
                f"{key} must have dimension {expected_size}; "
                f"received {values.size}."
            )
        normalization_stats[key] = values

    prepared = {
        "normalized_splits": {
            "test": {
                "Y": normalized_y,
                "U": normalized_u,
            }
        },
        "raw_splits": {
            "test": {
                "Y": physical_y,
                "U": physical_u,
                "trajectory_index": trajectory_index,
                "anchor_index": anchor_index,
                "target_index": target_index,
                "footprint_start": footprint_start,
                "footprint_end": footprint_end,
            }
        },
    }

    metadata = {
        "dataset_path": str(dataset_path),
        "num_test_pairs": int(num_pairs),
        "condition_dimension": int(normalized_y.shape[1]),
        "target_dimension": int(normalized_u.shape[1]),
        "normalization_stats": normalization_stats,
        "saved_metadata": saved_metadata,
    }

    return (
        prepared,
        data_config,
        sensor_indices,
        sensor_positions,
        metadata,
    )


# ---------------------------------------------------------------------------
# Drift training
# ---------------------------------------------------------------------------

def train_from_prepared_data(
    prepared: Mapping[str, Any],
    *,
    output_dir: str | Path,
    train_config: TrainConfig,
    interpolant_config: InterpolantConfig,
) -> Dict[str, Any]:
    """Train the conditional drift using the existing training module."""
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
# Conditional sampling for selected test inputs
# ---------------------------------------------------------------------------

def select_evenly_spaced_indices(
    population_size: int,
    requested_count: int,
) -> Array:
    """Select deterministic test-pair indices spanning the full ordering."""
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
    loaded_bundle: Any = None,
) -> Dict[str, Any]:
    """
    Generate future-sensor ensembles for selected held-out conditions.
    """
    uq_config.validate()

    if loaded_bundle is None:
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
    else:
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
        raw_test["trajectory_index"] == uq_config.trajectory_index
    )

    if trajectory_candidates.size == 0:
        available = np.unique(
            raw_test["trajectory_index"]
        ).astype(int).tolist()
        raise ValueError(
            "No test pairs are available for trajectory_index="
            f"{uq_config.trajectory_index}. Available trajectory indices: "
            f"{available}."
        )

    # Sort chronologically before selecting anchor times for the UQ figure.
    candidate_order = np.argsort(
        raw_test["anchor_index"][trajectory_candidates]
    )
    trajectory_candidates = trajectory_candidates[candidate_order]

    positions = select_evenly_spaced_indices(
        trajectory_candidates.size,
        uq_config.num_test_conditions,
    )
    selected_indices = trajectory_candidates[positions]

    normalized_ensembles: List[Array] = []
    physical_ensembles: List[Array] = []

    for order, test_index in enumerate(selected_indices):
        # Distinct reproducible random stream for each test condition.
        sample_seed = uq_config.seed + order
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
        "normalized_input_y": normalized_test["Y"][selected_indices],
        "true_future_observation": raw_test["U"][selected_indices],
        "normalized_true_future_observation": normalized_test["U"][
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
        "target_index": raw_test["target_index"][selected_indices],
        "footprint_start": raw_test["footprint_start"][
            selected_indices
        ],
        "footprint_end": raw_test["footprint_end"][selected_indices],
        "checkpoint": checkpoint,
        "interpolant_config": interpolant_config,
        "device": device,
    }


def save_uq_samples(
    output_path: str | Path,
    ensembles: Mapping[str, Any],
    *,
    sensor_indices: Array,
    sensor_positions: Array,
    uq_config: UQConfig,
    data_config: DataConfig,
) -> Path:
    """Save test inputs, future truths, forecast ensembles, and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interpolant_config: InterpolantConfig = ensembles[
        "interpolant_config"
    ]

    metadata = {
        "target": "P(X_{t+tau})",
        "num_sensors": NUM_SENSORS,
        "num_test_conditions": int(
            ensembles["selected_test_indices"].size
        ),
        "trajectory_index": int(uq_config.trajectory_index),
        "ensemble_size": int(uq_config.ensemble_size),
        "euler_maruyama_steps": int(uq_config.em_steps),
        "interval_level": float(uq_config.interval_level),
        "embedding_dimension_m": int(data_config.m),
        "time_delay_tau_indices": int(data_config.tau),
        "forecast_horizon_indices": int(data_config.forecast_horizon),
        "seed": int(uq_config.seed),
        "diffusion_schedule": "rho(s) = sigma_I * (1 - s)",
        "sigma_I": float(interpolant_config.sigma_I),
    }

    np.savez_compressed(
        output_path,
        selected_test_indices=ensembles["selected_test_indices"],
        input_y=ensembles["input_y"],
        normalized_input_y=ensembles["normalized_input_y"],
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
        sensor_indices=np.asarray(sensor_indices, dtype=np.int64),
        sensor_positions=np.asarray(
            sensor_positions,
            dtype=np.float64,
        ),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )

    return output_path


# ---------------------------------------------------------------------------
# UQ calculations and visualizations
# ---------------------------------------------------------------------------

def central_interval(
    samples: Array,
    level: float,
) -> Tuple[Array, Array]:
    """
    Compute componentwise central intervals.

    samples has shape (K,L,4), where K is the number of conditions and L is
    the ensemble size. Quantiles are taken over the ensemble axis.
    """
    alpha = 1.0 - level
    lower = np.quantile(samples, alpha / 2.0, axis=1)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
    return lower, upper


def plot_forecast_uq_over_anchor_times(
    ensembles: Mapping[str, Any],
    *,
    sensor_positions: Array,
    data_config: DataConfig,
    interval_level: float,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Create one UQ figure across selected anchor times.

    The figure contains one vertically stacked panel for each of the four
    sensors. Every panel shows:

        - the conditional ensemble-mean forecast of P_j(X_{t+tau});
        - a shaded central empirical interval; and
        - the true future sensor value.

    All selected conditions come from one test trajectory and are ordered by
    anchor time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(ensembles["samples"], dtype=np.float64)
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

    if samples.ndim != 3 or samples.shape[2] != NUM_SENSORS:
        raise ValueError(
            "samples must have shape "
            "(num_conditions, ensemble_size, 4)."
        )
    if truths.shape != (samples.shape[0], NUM_SENSORS):
        raise ValueError(
            "true_future_observation must have shape "
            "(num_conditions, 4)."
        )
    if anchors.shape != (samples.shape[0],):
        raise ValueError(
            "anchor_index must contain one value per condition."
        )
    if trajectories.shape != (samples.shape[0],):
        raise ValueError(
            "trajectory_index must contain one value per condition."
        )
    if np.unique(trajectories).size != 1:
        raise ValueError(
            "The single anchor-time UQ figure requires all selected "
            "conditions to come from one trajectory."
        )

    order = np.argsort(anchors)
    anchors = anchors[order]
    samples = samples[order]
    truths = truths[order]

    lower, upper = central_interval(samples, interval_level)
    forecast_mean = samples.mean(axis=1)

    covered = (truths >= lower) & (truths <= upper)
    sensor_coverage = covered.mean(axis=0)
    sensor_rmse = np.sqrt(
        np.mean((forecast_mean - truths) ** 2, axis=0)
    )
    sensor_mean_width = (upper - lower).mean(axis=0)

    figure, axes = plt.subplots(
        NUM_SENSORS,
        1,
        figsize=(12, 12),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for sensor_index, axis in enumerate(axes):
        axis.fill_between(
            anchors,
            lower[:, sensor_index],
            upper[:, sensor_index],
            alpha=0.25,
            label=f"{interval_level:.0%} confidence interval",
        )
        axis.plot(
            anchors,
            forecast_mean[:, sensor_index],
            marker="o",
            label="Forecast ensemble mean",
        )
        axis.plot(
            anchors,
            truths[:, sensor_index],
            marker="x",
            linestyle="--",
            label="True value at t + tau",
        )

        axis.set_ylabel(
            "KS value\n"
            f"x={float(sensor_positions[sensor_index]):.3f}"
        )
        axis.set_title(
            f"Sensor {sensor_index + 1}: "
            f"coverage={sensor_coverage[sensor_index]:.1%}, "
            f"RMSE={sensor_rmse[sensor_index]:.3e}"
        )
        axis.grid(True)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Anchor time index t")

    trajectory_index = int(trajectories[0])
    figure.suptitle(
        "Stochastic KS partial-observation forecast with UQ\n"
        f"trajectory={trajectory_index}, target=P(X_(t+tau)), "
        f"tau={data_config.tau}, interval={interval_level:.0%}",
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {
        "plot_path": output_path,
        "anchor_indices": anchors,
        "forecast_mean": forecast_mean,
        "lower": lower,
        "upper": upper,
        "truth": truths,
        "sensor_coverage": sensor_coverage,
        "sensor_rmse": sensor_rmse,
        "sensor_mean_interval_width": sensor_mean_width,
        "trajectory_index": trajectory_index,
    }


def create_uq_visualizations(
    ensembles: Mapping[str, Any],
    *,
    sensor_positions: Array,
    data_config: DataConfig,
    uq_config: UQConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Create one shaded-interval forecast figure and save its numerical metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = plot_forecast_uq_over_anchor_times(
        ensembles,
        sensor_positions=sensor_positions,
        data_config=data_config,
        interval_level=uq_config.interval_level,
        output_path=(
            output_dir / "forecast_uq_over_anchor_times.png"
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
        trajectory_index=np.asarray(
            result["trajectory_index"],
            dtype=np.int64,
        ),
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
        "mean_rmse": float(np.mean(result["sensor_rmse"])),
        "mean_interval_width": float(
            np.mean(result["sensor_mean_interval_width"])
        ),
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
    """Run data preparation, drift training, sampling, and UQ plotting."""
    data_config.validate()
    train_config.validate()
    interpolant_config.validate()
    uq_config.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(input_path)

    prepared = prepare_partial_forecast_data(
        trajectories,
        config=data_config,
    )

    dataset_arrays = dataset_arrays_for_npz(
        prepared,
        config=data_config,
    )

    dataset_path = save_dataset_npz(
        output_dir / "ks_partial_forecast_data.npz",
        dataset_arrays,
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
        uq_dir / "conditional_partial_forecast_samples.npz",
        ensembles,
        sensor_indices=prepared["sensor_indices"],
        sensor_positions=prepared["sensor_positions"],
        uq_config=uq_config,
        data_config=data_config,
    )

    uq_visualizations = create_uq_visualizations(
        ensembles,
        sensor_positions=prepared["sensor_positions"],
        data_config=data_config,
        uq_config=uq_config,
        output_dir=uq_dir,
    )

    summary = {
        "input_file": str(Path(input_path)),
        "output_directory": str(output_dir),
        "dataset_file": str(dataset_path),
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
        "target_dimension": NUM_SENSORS,
        "target": "P(X_{t+tau})",
        "sensor_indices": prepared["sensor_indices"].tolist(),
        "sensor_positions": prepared["sensor_positions"].tolist(),
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


def validate_checkpoint_for_forecast_dataset(
    loaded_bundle: Any,
    dataset_metadata: Mapping[str, Any],
) -> None:
    """
    Verify that the checkpoint and prepared dataset describe the same problem.

    In particular, the condition/target dimensions and the normalization
    statistics must agree. This prevents silently applying a checkpoint to
    test conditions normalized with different training statistics.
    """
    (
        _model,
        _interpolant_config,
        checkpoint_normalization,
        checkpoint,
        _device,
    ) = loaded_bundle

    model_config = checkpoint["model_config"]
    checkpoint_condition_dim = int(model_config["condition_dim"])
    checkpoint_target_dim = int(model_config["target_dim"])

    dataset_condition_dim = int(
        dataset_metadata["condition_dimension"]
    )
    dataset_target_dim = int(
        dataset_metadata["target_dimension"]
    )

    if checkpoint_condition_dim != dataset_condition_dim:
        raise ValueError(
            "Checkpoint/data condition-dimension mismatch: checkpoint "
            f"expects {checkpoint_condition_dim}, while the dataset has "
            f"{dataset_condition_dim}."
        )

    if checkpoint_target_dim != dataset_target_dim:
        raise ValueError(
            "Checkpoint/data target-dimension mismatch: checkpoint expects "
            f"{checkpoint_target_dim}, while the dataset has "
            f"{dataset_target_dim}."
        )

    checkpoint_stats = checkpoint_normalization.as_dict()
    dataset_stats = dataset_metadata["normalization_stats"]

    for key in ("mu_Y", "sigma_Y", "mu_U", "sigma_U"):
        checkpoint_values = np.asarray(
            checkpoint_stats[key],
            dtype=np.float32,
        ).reshape(-1)
        dataset_values = np.asarray(
            dataset_stats[key],
            dtype=np.float32,
        ).reshape(-1)

        if checkpoint_values.shape != dataset_values.shape:
            raise ValueError(
                f"Checkpoint/data normalization shape mismatch for {key}: "
                f"{checkpoint_values.shape} versus {dataset_values.shape}."
            )

        if not np.allclose(
            checkpoint_values,
            dataset_values,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"Checkpoint and dataset use different {key} values. "
                "Use the NPZ file produced by the same training run as the "
                "checkpoint."
            )


def run_inference_only(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    uq_config: UQConfig,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Generate new forecast ensembles and UQ plots without retraining.

    This mode reuses the normalized held-out forecast conditions saved in the
    NPZ dataset and the trained drift checkpoint. It does not load test.npy,
    rebuild temporal footprints, refit normalization, or call
    train_conditional_drift.
    """
    uq_config.validate()

    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    (
        prepared,
        data_config,
        sensor_indices,
        sensor_positions,
        dataset_metadata,
    ) = load_prepared_forecast_data_from_npz(dataset_path)

    loaded_bundle = load_trained_drift(
        checkpoint_path,
        device_name=device_name,
    )
    validate_checkpoint_for_forecast_dataset(
        loaded_bundle,
        dataset_metadata,
    )

    ensembles = generate_test_ensembles(
        prepared,
        checkpoint_path=checkpoint_path,
        uq_config=uq_config,
        device_name=device_name,
        loaded_bundle=loaded_bundle,
    )

    samples_path = save_uq_samples(
        output_dir / "conditional_partial_forecast_samples.npz",
        ensembles,
        sensor_indices=sensor_indices,
        sensor_positions=sensor_positions,
        uq_config=uq_config,
        data_config=data_config,
    )

    visualizations = create_uq_visualizations(
        ensembles,
        sensor_positions=sensor_positions,
        data_config=data_config,
        uq_config=uq_config,
        output_dir=output_dir,
    )

    summary = {
        "mode": "inference",
        "dataset_file": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "output_directory": str(output_dir),
        "uq_samples_file": str(samples_path),
        "uq_visualizations": visualizations,
        "data_config": asdict(data_config),
        "uq_config": asdict(uq_config),
        "device": str(ensembles["device"]),
        "condition_dimension": int(
            dataset_metadata["condition_dimension"]
        ),
        "target_dimension": int(
            dataset_metadata["target_dimension"]
        ),
        "target": "P(X_{t+tau})",
        "sensor_indices": sensor_indices.tolist(),
        "sensor_positions": sensor_positions.tolist(),
        "num_available_test_pairs": int(
            dataset_metadata["num_test_pairs"]
        ),
        "num_sampled_test_conditions": int(
            ensembles["selected_test_indices"].size
        ),
        "saved_dataset_metadata": dataset_metadata[
            "saved_metadata"
        ],
    }

    summary_path = output_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    summary["inference_summary_file"] = str(summary_path)
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
            "Train a stochastic-KS partial-observation forecasting model, "
            "or reuse a prepared dataset and checkpoint in inference-only "
            "mode to generate UQ plots without retraining."
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
        default=Path("ks_partial_forecast_train_uq_output"),
        help=(
            "Training output directory. In inference mode it is also used "
            "to locate the default saved dataset and checkpoint."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "inference"),
        default="train",
        help=(
            "'train' runs the complete pipeline. 'inference' reuses a "
            "prepared NPZ dataset and trained checkpoint without retraining."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Prepared ks_partial_forecast_data.npz for inference mode. "
            "Defaults to OUTPUT_DIR/ks_partial_forecast_data.npz."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Trained checkpoint for inference mode. Defaults to "
            "OUTPUT_DIR/training/conditional_drift_checkpoint.pt."
        ),
    )
    parser.add_argument(
        "--inference-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for newly generated inference samples and plots. "
            "Defaults to OUTPUT_DIR/inference_uq."
        ),
    )

    # Data construction.
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Embedding dimension; required in training mode.",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=None,
        help="Time delay and forecast horizon; required in training mode.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help="Gap between full input-target footprints; required in training mode.",
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
        help=(
            "Central empirical interval level used in the shaded UQ region. "
            "The default is 0.90."
        ),
    )
    parser.add_argument(
        "--uq-trajectory-index",
        type=int,
        default=0,
        help=(
            "Trajectory whose held-out anchor times are shown in the single "
            "forecast UQ figure."
        ),
    )
    parser.add_argument(
        "--uq-seed",
        type=int,
        default=101,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
            else args.output_dir / "ks_partial_forecast_data.npz"
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
            device_name=args.device,
        )

        print(json.dumps(summary, indent=2))
        return

    missing_training_arguments = [
        name
        for name, value in (
            ("--m", args.m),
            ("--tau", args.tau),
            ("--gap", args.gap),
        )
        if value is None
    ]
    if missing_training_arguments:
        raise SystemExit(
            "Training mode requires: "
            + ", ".join(missing_training_arguments)
        )

    data_config = DataConfig(
        m=args.m,
        tau=args.tau,
        gap=args.gap,
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
