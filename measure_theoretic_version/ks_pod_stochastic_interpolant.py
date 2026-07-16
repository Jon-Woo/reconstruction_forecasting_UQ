#!/usr/bin/env python3
"""
Stochastic Kuramoto--Sivashinsky POD modal reconstruction from sparse sensors.

Expected input
--------------
A NumPy file ``test.npy`` containing an array U with shape (N, T, D):

    N : number of independent solution realizations,
    T : number of stored time indices per realization,
    D : number of periodic spatial grid points on [0, L).

The script:
1. splits entire realizations into train/validation/test sets;
2. fits the POD mean and basis only on training realizations;
3. places four uniformly spaced point sensors on the periodic domain;
4. constructs non-overlapping delay-coordinate windows within each realization;
5. saves a dataset suitable for the conditional stochastic-interpolant pipeline;
6. optionally trains the Gaussian-source stochastic interpolant;
7. visualizes trajectories, sensors, POD spectra, coefficient reconstruction,
   and field reconstruction on held-out test realizations.

The stochastic-interpolant implementation is imported from gaussian_source_si.py,
which must be in the same directory or on PYTHONPATH.

The split-by-realization design is deliberate: no time sample from a held-out
realization enters the POD basis, drift training, or normalization statistics.
Within each realization, retained delay windows use disjoint raw time indices.
Literal non-overlap does not by itself prove probabilistic independence, but
independent realizations plus disjoint windows is substantially stronger than
randomly splitting overlapping windows from one trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class PODModel:
    mean: Array                 # (D,)
    modes: Array                # (r,D), orthonormal in the discrete Euclidean product
    eigenvalues: Array          # all nonzero empirical covariance eigenvalues
    singular_values: Array
    explained_energy: Array
    retained_energy: float


def load_trajectory_array(path: str | Path) -> Array:
    """Load and validate an array with shape (N,T,D)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input trajectory file not found: {path}")
    u = np.load(path, allow_pickle=False)
    if u.ndim != 3:
        raise ValueError(f"Expected shape (N,T,D), received {u.shape}.")
    if min(u.shape) < 1:
        raise ValueError("All array dimensions must be positive.")
    if not np.issubdtype(u.dtype, np.number):
        raise TypeError("The trajectory array must be numeric.")
    u = np.asarray(u, dtype=np.float64)
    if not np.isfinite(u).all():
        raise ValueError("Trajectory data contain NaN or infinite values.")
    return u


def split_realizations(
    n_realizations: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> RealizationSplit:
    """
    Randomly split realization IDs, with at least one realization in each split.

    This requires N >= 3. Splitting entire realizations prevents the same
    stochastic path from contributing to both training and testing.
    """
    if n_realizations < 3:
        raise ValueError(
            "At least three realizations are required for disjoint "
            "train/validation/test splits by realization."
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0,1).")
    if not 0.0 < val_fraction < 1.0 - train_fraction:
        raise ValueError("val_fraction must lie in (0,1-train_fraction).")

    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_realizations)

    n_train = max(1, int(math.floor(train_fraction * n_realizations)))
    n_val = max(1, int(math.floor(val_fraction * n_realizations)))
    if n_train + n_val >= n_realizations:
        # Preserve a nonempty test split.
        overflow = n_train + n_val - (n_realizations - 1)
        if n_train >= n_val and n_train - overflow >= 1:
            n_train -= overflow
        elif n_val - overflow >= 1:
            n_val -= overflow
        else:
            raise ValueError("Fractions cannot produce three nonempty splits.")

    return RealizationSplit(
        train=np.sort(ids[:n_train]),
        val=np.sort(ids[n_train:n_train + n_val]),
        test=np.sort(ids[n_train + n_val:]),
    )


def fit_training_pod(
    trajectories: Array,
    training_realizations: Array,
    n_modes: int,
) -> Tuple[PODModel, Array]:
    r"""
    Fit snapshot POD using only training realizations and project all snapshots.

    Let q_{n,t} in R^D denote one spatial snapshot. With M=N_train*T training
    snapshots, define

        q_bar = M^{-1} sum q_j,
        X = M^{-1/2} [q_1-q_bar, ..., q_M-q_bar].

    If X = Phi Sigma V^T, the columns of Phi are POD modes. Numerically we store
    snapshots as rows, so np.linalg.svd(centered/sqrt(M)) returns V^T whose rows
    are the spatial POD modes.

    The projection coefficients are
        a_k(n,t) = <q_{n,t}-q_bar, phi_k>_2.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be positive.")

    train_snapshots = trajectories[training_realizations].reshape(
        -1, trajectories.shape[-1]
    )
    m = train_snapshots.shape[0]
    if m < 2:
        raise ValueError("At least two training snapshots are required.")

    mean = np.mean(train_snapshots, axis=0)
    centered_train = train_snapshots - mean
    scaled = centered_train / np.sqrt(m)

    _, singular_values, vt = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(singular_values > np.finfo(float).eps * singular_values[0]))
    r = min(n_modes, rank)
    if r < 1:
        raise RuntimeError("Training data have zero numerical POD rank.")

    eigenvalues = singular_values**2
    total_energy = float(np.sum(eigenvalues))
    explained = eigenvalues / total_energy if total_energy > 0 else np.zeros_like(eigenvalues)
    modes = vt[:r].copy()

    # Numerical orthonormality check.
    gram_error = np.linalg.norm(modes @ modes.T - np.eye(r), ord=2)
    if gram_error > 1.0e-10:
        raise RuntimeError(f"POD modes failed orthonormality check: {gram_error:.3e}")

    centered_all = trajectories - mean[None, None, :]
    coefficients = np.einsum("ntd,rd->ntr", centered_all, modes)

    pod = PODModel(
        mean=mean,
        modes=modes,
        eigenvalues=eigenvalues,
        singular_values=singular_values,
        explained_energy=explained,
        retained_energy=float(np.sum(explained[:r])),
    )
    return pod, coefficients


def uniform_periodic_sensor_indices(n_space: int, n_sensors: int = 4) -> Array:
    """
    Select uniformly spaced grid indices on a periodic grid x_j=jL/D.

    For D not divisible by n_sensors, rounding is used and uniqueness is checked.
    """
    if n_space < n_sensors:
        raise ValueError("The spatial grid must contain at least n_sensors points.")
    indices = np.floor(
        np.arange(n_sensors, dtype=np.float64) * n_space / n_sensors
    ).astype(np.int64)
    if np.unique(indices).size != n_sensors:
        raise RuntimeError("Uniform sensor construction produced duplicate indices.")
    return indices


def sensor_observations(trajectories: Array, sensor_indices: Array) -> Array:
    """Return point observations with shape (N,T,n_sensors)."""
    return trajectories[:, :, sensor_indices]


def _window_index_sets(anchor: int, tau: int, m: int) -> set[int]:
    return set((anchor - np.arange(m, dtype=np.int64) * tau).tolist())


def build_nonoverlapping_multirealization_pairs(
    observations: Array,
    coefficients: Array,
    realization_ids: Iterable[int],
    tau: int,
    embedding_dim: int,
    gap_steps: int,
) -> Dict[str, Array]:
    r"""
    Construct non-overlapping delay windows for a set of realizations.

    For realization n and anchor t,

        Y_{n,t} = [z_{n,t}, z_{n,t-tau}, ..., z_{n,t-(m-1)tau}],
        A_{n,t} = [a_1(n,t), ..., a_r(n,t)].

    Consecutive retained anchors differ by

        h = (m-1)tau + gap_steps + 1,

    hence their raw time-index sets are disjoint. Windows from different
    realizations are also distinct by construction.
    """
    if observations.ndim != 3 or coefficients.ndim != 3:
        raise ValueError("observations and coefficients must both be rank-three.")
    if observations.shape[:2] != coefficients.shape[:2]:
        raise ValueError("Observation and coefficient leading dimensions must match.")
    if tau < 1 or embedding_dim < 1 or gap_steps < 0:
        raise ValueError("tau,m must be positive and gap_steps nonnegative.")

    _, n_times, _ = observations.shape
    span = (embedding_dim - 1) * tau
    anchor_stride = span + gap_steps + 1
    offsets = np.arange(embedding_dim, dtype=np.int64) * tau

    y_list: List[Array] = []
    u_list: List[Array] = []
    realization_list: List[int] = []
    anchor_list: List[int] = []

    for realization in np.asarray(list(realization_ids), dtype=np.int64):
        anchors = np.arange(span, n_times, anchor_stride, dtype=np.int64)
        if anchors.size == 0:
            raise ValueError(
                "No valid windows. Increase T or reduce tau/embedding_dim/gap."
            )

        previous: set[int] | None = None
        for anchor in anchors:
            current = _window_index_sets(int(anchor), tau, embedding_dim)
            if previous is not None and previous.intersection(current):
                raise AssertionError("Internal error: delay windows overlap.")
            previous = current

            # Ordering: current four sensors, then sensors tau steps back, etc.
            y_list.append(observations[realization, anchor - offsets].reshape(-1))
            u_list.append(coefficients[realization, anchor])
            realization_list.append(int(realization))
            anchor_list.append(int(anchor))

    if not y_list:
        raise ValueError("No delay-coordinate examples were constructed.")

    return {
        "Y": np.asarray(y_list, dtype=np.float64),
        "U": np.asarray(u_list, dtype=np.float64),
        "realization": np.asarray(realization_list, dtype=np.int64),
        "anchor": np.asarray(anchor_list, dtype=np.int64),
        "window_span": np.asarray(span, dtype=np.int64),
        "anchor_stride": np.asarray(anchor_stride, dtype=np.int64),
    }


def reconstruct_field(pod: PODModel, coefficients: Array) -> Array:
    """Compute q_r = mean + sum_k a_k phi_k."""
    a = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if len(a) > pod.modes.shape[0]:
        raise ValueError("Coefficient vector is longer than the retained POD basis.")
    return pod.mean + a @ pod.modes[: len(a)]


def plot_dataset_overview(
    trajectories: Array,
    coefficients: Array,
    sensor_indices: Array,
    split: RealizationSplit,
    domain_length: float,
    dt: float,
    output_path: Path,
) -> None:
    """Visualize representative trajectories, sensor placement, and POD coefficients."""
    n_realizations, n_times, n_space = trajectories.shape
    x = domain_length * np.arange(n_space) / n_space
    t = dt * np.arange(n_times)

    chosen = [
        int(split.train[0]),
        int(split.val[0]),
        int(split.test[0]),
    ]
    labels = ["training realization", "validation realization", "test realization"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    image = axes[0, 0].imshow(
        trajectories[chosen[0]],
        origin="lower",
        aspect="auto",
        extent=[0.0, domain_length, t[0], t[-1]],
        interpolation="nearest",
    )
    for j, index in enumerate(sensor_indices):
        axes[0, 0].axvline(
            x[index], linestyle="--", linewidth=1.2,
            label="sensor locations" if j == 0 else None,
        )
    axes[0, 0].set_title(f"Space-time field: {labels[0]} {chosen[0]}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("time")
    axes[0, 0].legend()
    fig.colorbar(image, ax=axes[0, 0], label="u(x,t)")

    snapshot_time = n_times // 2
    for rid, label in zip(chosen, labels):
        axes[0, 1].plot(x, trajectories[rid, snapshot_time], label=f"{label} {rid}")
    axes[0, 1].scatter(
        x[sensor_indices],
        trajectories[chosen[0], snapshot_time, sensor_indices],
        marker="o",
        zorder=5,
        label="four sensor samples (training curve)",
    )
    axes[0, 1].set_title(f"Spatial snapshots at t={t[snapshot_time]:.3g}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    for j, index in enumerate(sensor_indices):
        axes[1, 0].plot(
            t, trajectories[chosen[0], :, index],
            label=f"x={x[index]:.3g}",
        )
    axes[1, 0].set_title("Four partial-observation time series")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("sensor value")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(ncol=2)

    for k in range(min(4, coefficients.shape[-1])):
        axes[1, 1].plot(
            t, coefficients[chosen[0], :, k],
            label=fr"$a_{k+1}(t)$",
        )
    axes[1, 1].set_title("POD coefficients for one training realization")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("coefficient")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("Stochastic Kuramoto--Sivashinsky dataset and four sensors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pod_diagnostics(pod: PODModel, domain_length: float, output_path: Path) -> None:
    """Plot retained POD modes and cumulative energy."""
    n_space = pod.mean.size
    x = domain_length * np.arange(n_space) / n_space
    n_show = min(6, pod.modes.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in range(n_show):
        axes[0].plot(x, pod.modes[k], label=fr"$\phi_{k+1}$")
    axes[0].set_title("Training-only POD modes")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("mode amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    cumulative = np.cumsum(pod.explained_energy)
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative)
    axes[1].axvline(pod.modes.shape[0], linestyle="--", label="retained rank")
    axes[1].set_xlim(1, max(2, min(len(cumulative), 50)))
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("Cumulative training fluctuation energy")
    axes[1].set_xlabel("number of modes")
    axes[1].set_ylabel("cumulative energy fraction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_ordered_test_subset(
    pairs: Dict[str, Array],
    max_points: int,
) -> Array:
    """
    Select an ordered sequence from one held-out realization for line plotting.

    Connecting samples from different stochastic realizations would create a
    meaningless temporal curve, so the realization with the most retained test
    windows is chosen.
    """
    realization_ids, counts = np.unique(pairs["realization"], return_counts=True)
    chosen_realization = int(realization_ids[np.argmax(counts)])
    indices = np.where(pairs["realization"] == chosen_realization)[0]
    indices = indices[np.argsort(pairs["anchor"][indices])]
    return indices[:max_points]


def plot_coefficient_reconstruction(
    pairs: Dict[str, Array],
    selected_indices: Array,
    true_coefficients: Array,
    means: Array,
    lower: Array,
    upper: Array,
    dt: float,
    output_path: Path,
) -> None:
    """Plot true and reconstructed coefficients on one held-out realization."""
    n_show = min(6, true_coefficients.shape[1])
    anchors = pairs["anchor"][selected_indices]
    times = dt * anchors
    realization = int(pairs["realization"][selected_indices[0]])

    fig, axes = plt.subplots(
        n_show, 1, figsize=(11, 2.3 * n_show), sharex=True, squeeze=False
    )
    for k in range(n_show):
        ax = axes[k, 0]
        ax.fill_between(times, lower[:, k], upper[:, k], alpha=0.25,
                        label="90% SI sample interval")
        ax.plot(times, true_coefficients[:, k], "o-", markersize=3,
                linewidth=1.0, label="true coefficient")
        ax.plot(times, means[:, k], "s--", markersize=3,
                linewidth=1.0, label="ensemble mean")
        ax.set_ylabel(fr"$a_{k+1}$")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(ncol=3)

    axes[-1, 0].set_xlabel("physical time")
    fig.suptitle(
        f"Held-out realization {realization}: POD coefficient reconstruction"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_field_reconstruction(
    x: Array,
    truth: Array,
    true_pod: Array,
    predicted: Array,
    sensor_indices: Array,
    realization: int,
    anchor: int,
    dt: float,
    output_path: Path,
) -> None:
    """Compare full test field, true POD projection, SI reconstruction, and error."""
    error = np.abs(predicted - truth)
    denominator = max(float(np.linalg.norm(truth)), 1.0e-14)
    si_error = float(np.linalg.norm(predicted - truth) / denominator)
    truncation_error = float(np.linalg.norm(true_pod - truth) / denominator)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(x, truth)
    axes[0, 0].scatter(x[sensor_indices], truth[sensor_indices], marker="o",
                       label="partial observations")
    axes[0, 0].set_title("Held-out true KS field and four sensors")
    axes[0, 0].legend()

    axes[0, 1].plot(x, truth, label="true")
    axes[0, 1].plot(x, true_pod, linestyle="--", label="true rank-r POD projection")
    axes[0, 1].set_title("POD truncation comparison")
    axes[0, 1].legend()

    axes[1, 0].plot(x, truth, label="true")
    axes[1, 0].plot(x, predicted, linestyle="--",
                    label="SI ensemble-mean reconstruction")
    axes[1, 0].set_title("Stochastic-interpolant reconstruction")
    axes[1, 0].legend()

    axes[1, 1].plot(x, error)
    axes[1, 1].set_title("Pointwise absolute field error")
    axes[1, 1].set_ylabel(r"$|\widehat u_r-u|$")

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Test realization {realization}, t={anchor * dt:.4g}; "
        f"SI relative L2 error={si_error:.3e}, "
        f"POD truncation error={truncation_error:.3e}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_dataset(
    path: Path,
    trajectories_shape: Tuple[int, int, int],
    split: RealizationSplit,
    sensor_indices: Array,
    sensor_locations: Array,
    pod: PODModel,
    pairs: Dict[str, Dict[str, Array]],
    metadata: dict,
) -> None:
    payload = {
        "trajectory_shape": np.asarray(trajectories_shape, dtype=np.int64),
        "train_realizations": split.train,
        "val_realizations": split.val,
        "test_realizations": split.test,
        "sensor_indices": sensor_indices,
        "sensor_locations": sensor_locations,
        "pod_mean": pod.mean,
        "pod_modes": pod.modes,
        "pod_eigenvalues": pod.eigenvalues,
        "pod_explained_energy": pod.explained_energy,
        "pod_retained_energy": np.asarray(pod.retained_energy),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    for split_name, split_pairs in pairs.items():
        for key, value in split_pairs.items():
            payload[f"{split_name}_{key}"] = value
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare and train a POD stochastic-interpolant model for "
                    "stochastic 1D Kuramoto--Sivashinsky trajectories."
    )
    p.add_argument("--input", type=Path, default=Path("test.npy"))
    p.add_argument("--output-dir", type=Path, default=Path("ks_pod_si_output"))
    p.add_argument("--domain-length", type=float, default=22.0)
    p.add_argument("--noise-level", type=float, default=0.05,
                   help="Metadata describing the trajectory-generating SPDE.")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Time between stored samples; used for plot labels.")
    p.add_argument("--tau", type=int, default=5,
                   help="Delay in stored time-index steps.")
    p.add_argument("--embedding-dim", type=int, default=4)
    p.add_argument("--gap-steps", type=int, default=0)
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--prepare-only", action="store_true",
                   help="Prepare/save the dataset and plots but skip SI training.")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--eta", type=float, default=0.35)
    p.add_argument("--ensemble-size", type=int, default=300)
    p.add_argument("--sde-steps", type=int, default=300)
    p.add_argument("--max-reconstruction-points", type=int, default=40)
    p.add_argument("--evaluation-seed", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.domain_length <= 0 or args.dt <= 0:
        raise ValueError("domain-length and dt must be positive.")
    if args.max_reconstruction_points < 1:
        raise ValueError("max-reconstruction-points must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectory_array(args.input)
    n_realizations, n_times, n_space = trajectories.shape

    split = split_realizations(
        n_realizations,
        args.train_fraction,
        args.val_fraction,
        args.split_seed,
    )
    pod, coefficients = fit_training_pod(
        trajectories, split.train, args.n_modes
    )

    sensor_indices = uniform_periodic_sensor_indices(n_space, n_sensors=4)
    sensor_locations = args.domain_length * sensor_indices / n_space
    observations = sensor_observations(trajectories, sensor_indices)

    pairs = {
        name: build_nonoverlapping_multirealization_pairs(
            observations,
            coefficients,
            realization_ids=getattr(split, name),
            tau=args.tau,
            embedding_dim=args.embedding_dim,
            gap_steps=args.gap_steps,
        )
        for name in ("train", "val", "test")
    }

    metadata = {
        "equation": "stochastic Kuramoto-Sivashinsky, user-supplied trajectories",
        "domain_length": args.domain_length,
        "noise_level": args.noise_level,
        "dt_between_stored_samples": args.dt,
        "tau_in_stored_steps": args.tau,
        "tau_physical": args.tau * args.dt,
        "embedding_dimension": args.embedding_dim,
        "gap_steps": args.gap_steps,
        "n_sensors": 4,
        "n_modes": int(pod.modes.shape[0]),
        "retained_energy": pod.retained_energy,
        "split_by_realization": True,
        "train_pair_count": len(pairs["train"]["Y"]),
        "val_pair_count": len(pairs["val"]["Y"]),
        "test_pair_count": len(pairs["test"]["Y"]),
    }

    save_dataset(
        output_dir / "ks_pod_si_dataset.npz",
        trajectories.shape,
        split,
        sensor_indices,
        sensor_locations,
        pod,
        pairs,
        metadata,
    )

    plot_dataset_overview(
        trajectories,
        coefficients,
        sensor_indices,
        split,
        args.domain_length,
        args.dt,
        output_dir / "original_trajectories_and_sensors.png",
    )
    plot_pod_diagnostics(
        pod,
        args.domain_length,
        output_dir / "training_pod_modes_and_energy.png",
    )

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "input_shape": list(trajectories.shape),
                "train_realizations": split.train.tolist(),
                "val_realizations": split.val.tolist(),
                "test_realizations": split.test.tolist(),
                "sensor_indices": sensor_indices.tolist(),
                "sensor_locations": sensor_locations.tolist(),
                "condition_dimension": int(pairs["train"]["Y"].shape[1]),
                "target_dimension": int(pairs["train"]["U"].shape[1]),
                "window_span": int(pairs["train"]["window_span"]),
                "anchor_stride": int(pairs["train"]["anchor_stride"]),
            },
            f,
            indent=2,
        )

    print(json.dumps(metadata, indent=2))
    print(f"Saved prepared dataset to {output_dir / 'ks_pod_si_dataset.npz'}")

    if args.prepare_only:
        print("Preparation complete; stochastic-interpolant training was skipped.")
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
        pairs["test"], args.max_reconstruction_points
    )
    truth = pairs["test"]["U"][selected]
    means, lower, upper = [], [], []
    rmses, spreads = [], []

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
        means.append(np.mean(samples, axis=0))
        lower.append(np.quantile(samples, 0.05, axis=0))
        upper.append(np.quantile(samples, 0.95, axis=0))
        metrics = conditional_ensemble_metrics(
            samples, pairs["test"]["U"][pair_index]
        )
        rmses.append(metrics["ensemble_mean_rmse"])
        spreads.append(metrics["ensemble_spread_rms"])

    means = np.asarray(means)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    plot_coefficient_reconstruction(
        pairs["test"],
        selected,
        truth,
        means,
        lower,
        upper,
        args.dt,
        output_dir / "pod_coefficient_reconstruction_vs_truth.png",
    )

    first = int(selected[0])
    realization = int(pairs["test"]["realization"][first])
    anchor = int(pairs["test"]["anchor"][first])
    true_field = trajectories[realization, anchor]
    true_pod_field = reconstruct_field(pod, truth[0])
    predicted_field = reconstruct_field(pod, means[0])
    x = args.domain_length * np.arange(n_space) / n_space

    plot_field_reconstruction(
        x,
        true_field,
        true_pod_field,
        predicted_field,
        sensor_indices,
        realization,
        anchor,
        args.dt,
        output_dir / "field_reconstruction_vs_truth.png",
    )

    np.savez_compressed(
        output_dir / "test_reconstruction_results.npz",
        selected_pair_indices=selected,
        realization=pairs["test"]["realization"][selected],
        anchor=pairs["test"]["anchor"][selected],
        truth=truth,
        ensemble_mean=means,
        interval_05=lower,
        interval_95=upper,
        ensemble_mean_rmse=np.asarray(rmses),
        ensemble_spread_rms=np.asarray(spreads),
    )

    evaluation_summary = {
        "number_evaluated": int(len(selected)),
        "mean_ensemble_mean_rmse": float(np.mean(rmses)),
        "mean_ensemble_spread_rms": float(np.mean(spreads)),
        "evaluated_realization": int(
            pairs["test"]["realization"][selected[0]]
        ),
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(json.dumps(evaluation_summary, indent=2))


if __name__ == "__main__":
    main()
