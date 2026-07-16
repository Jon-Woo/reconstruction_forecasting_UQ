#!/usr/bin/env python3
"""
Cylinder-flow POD modal reconstruction with a conditional Gaussian-source
stochastic interpolant.

This script:
1. runs or loads a long D2Q9 cylinder-wake trajectory;
2. fits POD only on the chronological training portion (prevents basis leakage);
3. uses sparse velocity sensors as partial observations;
4. creates non-overlapping delay windows;
5. trains b_theta(s,x,y);
6. samples POD coefficient ensembles and plots diagnostics.

Dependencies: numpy, matplotlib, torch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from cylinder_flow_pod import Config as FlowConfig
from cylinder_flow_pod import compute_pod, run_lbm
from gaussian_source_si import (
    DelayDatasetConfig,
    InterpolantConfig,
    TrainConfig,
    build_nonoverlapping_delay_pairs,
    conditional_ensemble_metrics,
    sample_conditional_sde,
    train_conditional_drift,
)


def select_wake_sensors(
    solid: np.ndarray,
    cylinder_x: int,
    cylinder_radius: int,
    n_sensors: int,
) -> np.ndarray:
    """Deterministically select approximately space-filling downstream fluid nodes."""
    ny, nx = solid.shape
    x_min = min(nx - 2, cylinder_x + 2 * cylinder_radius)
    xs = np.linspace(x_min, nx - 3, max(2, int(np.ceil(np.sqrt(n_sensors))))).round().astype(int)
    ys = np.linspace(2, ny - 3, max(2, int(np.ceil(n_sensors / len(xs))))).round().astype(int)
    candidates = [(int(y), int(x)) for x in xs for y in ys if not solid[y, x]]
    if len(candidates) < n_sensors:
        fluid_y, fluid_x = np.where(~solid)
        order = np.argsort(fluid_x)
        for idx in order[::-1]:
            point = (int(fluid_y[idx]), int(fluid_x[idx]))
            if point not in candidates:
                candidates.append(point)
            if len(candidates) >= n_sensors:
                break
    return np.asarray(candidates[:n_sensors], dtype=np.int64)


def sensor_observations(snapshots: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Return [u_x,u_y] at each sensor, flattened to shape (T,2*n_sensors)."""
    values = []
    for y, x in sensors:
        values.append(snapshots[:, 0, y, x])
        values.append(snapshots[:, 1, y, x])
    return np.stack(values, axis=1)


def fit_training_pod_and_project(
    snapshots: np.ndarray,
    solid: np.ndarray,
    train_stop: int,
    n_modes: int,
) -> Tuple[dict, np.ndarray]:
    """
    Fit the POD basis on training snapshots only, then project every snapshot
    using the fixed training mean and modes.
    """
    if train_stop < 2:
        raise ValueError("Need at least two training snapshots for POD.")
    pod = compute_pod(snapshots[:train_stop], solid)
    r = min(n_modes, pod["U"].shape[1])
    fluid = pod["fluid_mask"]
    n_fluid = int(fluid.sum())
    q_all = np.empty((2 * n_fluid, len(snapshots)), dtype=np.float64)
    for j in range(len(snapshots)):
        q_all[:n_fluid, j] = snapshots[j, 0, fluid]
        q_all[n_fluid:, j] = snapshots[j, 1, fluid]
    centered = q_all - pod["mean_vector"]
    coefficients = (pod["U"][:, :r].T @ centered).T
    return pod, coefficients


def load_or_generate_flow(args: argparse.Namespace, output_dir: Path):
    data_path = output_dir / "flow_data.npz"
    if args.reuse_flow and data_path.exists():
        data = np.load(data_path)
        return (
            data["snapshots"], data["snapshot_times"], data["solid_mask"],
            data["lbm_params"]
        )

    cfg = FlowConfig(
        nx=args.nx,
        ny=args.ny,
        steps=args.steps,
        burn_in=args.burn_in,
        snapshot_stride=args.snapshot_stride,
        n_modes_plot=args.n_modes,
        inlet_velocity=args.inlet_velocity,
        reynolds=args.reynolds,
        cylinder_radius=args.cylinder_radius,
        cylinder_x=args.cylinder_x,
        cylinder_y=args.ny // 2,
        perturbation=args.perturbation,
        seed=args.flow_seed,
        output_dir=str(output_dir / "flow_diagnostics"),
    )
    snapshots, times, solid, lbm_params = run_lbm(cfg)
    np.savez_compressed(
        data_path,
        snapshots=snapshots,
        snapshot_times=times,
        solid_mask=solid,
        lbm_params=lbm_params,
    )
    return snapshots, times, solid, lbm_params


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="cylinder_si_output")
    p.add_argument("--reuse-flow", action="store_true")

    # Long-trajectory defaults: 750 post-transient snapshots.
    p.add_argument("--nx", type=int, default=240)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--steps", type=int, default=21000)
    p.add_argument("--burn-in", type=int, default=6000)
    p.add_argument("--snapshot-stride", type=int, default=20)
    p.add_argument("--inlet-velocity", type=float, default=0.06)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--cylinder-radius", type=int, default=10)
    p.add_argument("--cylinder-x", type=int, default=48)
    p.add_argument("--perturbation", type=float, default=1e-3)
    p.add_argument("--flow-seed", type=int, default=7)

    p.add_argument("--n-sensors", type=int, default=12)
    p.add_argument("--n-modes", type=int, default=6)
    p.add_argument(
        "--tau-lattice", type=int, default=100,
        help="Physical delay in LBM time steps; must be divisible by snapshot_stride."
    )
    p.add_argument("--embedding-dim", type=int, default=4)
    p.add_argument("--gap-snapshots", type=int, default=2)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)

    p.add_argument("--eta", type=float, default=0.25)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--train-seed", type=int, default=17)
    p.add_argument("--sde-steps", type=int, default=400)
    p.add_argument("--ensemble-size", type=int, default=500)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.tau_lattice % args.snapshot_stride != 0:
        raise ValueError("tau_lattice must be divisible by snapshot_stride.")
    delay_steps = args.tau_lattice // args.snapshot_stride

    snapshots, times, solid, lbm_params = load_or_generate_flow(args, out)
    n_times = len(snapshots)
    train_stop = int(np.floor(args.train_fraction * n_times))
    val_stop = train_stop + int(np.floor(args.val_fraction * n_times))

    pod, coefficients = fit_training_pod_and_project(
        snapshots, solid, train_stop, args.n_modes
    )
    sensors = select_wake_sensors(
        solid, args.cylinder_x, args.cylinder_radius, args.n_sensors
    )
    observations = sensor_observations(snapshots, sensors)

    delay_cfg = DelayDatasetConfig(
        delay_steps=delay_steps,
        embedding_dim=args.embedding_dim,
        gap_steps=args.gap_snapshots,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    pairs = build_nonoverlapping_delay_pairs(observations, coefficients, delay_cfg)

    interp_cfg = InterpolantConfig(eta=args.eta)
    train_cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.train_seed,
    )
    fit = train_conditional_drift(
        pairs["train"]["Y"], pairs["train"]["U"],
        pairs["val"]["Y"], pairs["val"]["U"],
        output_dir=out / "model",
        train_cfg=train_cfg,
        interpolant_cfg=interp_cfg,
    )

    # Evaluate a set of test conditions with independent ensemble draws.
    test_y, test_u = pairs["test"]["Y"], pairs["test"]["U"]
    n_eval = min(20, len(test_y))
    rmses, spreads = [], []
    first_samples = None
    for i in range(n_eval):
        samples = sample_conditional_sde(
            fit["model"], test_y[i], fit["y_scaler"], fit["u_scaler"],
            interp_cfg, n_samples=args.ensemble_size, n_steps=args.sde_steps,
            device=fit["device"], seed=1000 + i,
        )
        metrics = conditional_ensemble_metrics(samples, test_u[i])
        rmses.append(metrics["ensemble_mean_rmse"])
        spreads.append(metrics["ensemble_spread_rms"])
        if i == 0:
            first_samples = samples

    if first_samples is not None:
        r = first_samples.shape[1]
        fig, ax = plt.subplots(figsize=(9, 5))
        positions = np.arange(1, r + 1)
        q05, q50, q95 = np.quantile(first_samples, [0.05, 0.50, 0.95], axis=0)
        ax.errorbar(
            positions, q50, yerr=np.vstack([q50 - q05, q95 - q50]),
            fmt="o", capsize=4, label="90% sample interval"
        )
        ax.scatter(positions, test_u[0], marker="x", s=70, label="true coefficients")
        ax.set_xlabel("POD mode")
        ax.set_ylabel("coefficient")
        ax.set_title("Conditional POD-coefficient ensemble for one test delay vector")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "test_conditional_ensemble.png", dpi=180)
        plt.close(fig)

    # Dataset and POD summaries.
    np.savez_compressed(
        out / "modal_reconstruction_dataset.npz",
        observations=observations,
        pod_coefficients=coefficients,
        snapshot_times=times,
        sensors_yx=sensors,
        train_Y=pairs["train"]["Y"],
        train_U=pairs["train"]["U"],
        train_anchors=pairs["train"]["anchors"],
        val_Y=pairs["val"]["Y"],
        val_U=pairs["val"]["U"],
        val_anchors=pairs["val"]["anchors"],
        test_Y=pairs["test"]["Y"],
        test_U=pairs["test"]["U"],
        test_anchors=pairs["test"]["anchors"],
        pod_mean_field=pod["mean_field"],
        pod_modes=pod["modes"][:args.n_modes],
        pod_eigenvalues=pod["eigenvalues"],
        pod_energy_fraction=pod["energy_fraction"],
        solid_mask=solid,
    )

    summary = {
        "n_snapshots": n_times,
        "snapshot_time_start": int(times[0]),
        "snapshot_time_end": int(times[-1]),
        "post_transient_duration_lattice_steps": int(times[-1] - times[0]),
        "train_raw_range": [0, train_stop],
        "val_raw_range": [train_stop, val_stop],
        "test_raw_range": [val_stop, n_times],
        "delay_steps_in_snapshot_indices": delay_steps,
        "tau_lattice": args.tau_lattice,
        "embedding_dim": args.embedding_dim,
        "window_span_snapshot_indices": delay_cfg.window_span,
        "anchor_stride_snapshot_indices": delay_cfg.anchor_stride,
        "n_sensors": len(sensors),
        "observation_dim": observations.shape[1],
        "target_dim": coefficients.shape[1],
        "n_train_pairs": len(pairs["train"]["Y"]),
        "n_val_pairs": len(pairs["val"]["Y"]),
        "n_test_pairs": len(pairs["test"]["Y"]),
        "mean_test_ensemble_mean_rmse": float(np.mean(rmses)),
        "mean_test_ensemble_spread_rms": float(np.mean(spreads)),
        "lbm_viscosity": float(lbm_params[0]),
        "lbm_relaxation_time": float(lbm_params[1]),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
