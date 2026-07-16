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
6. visualizes the raw flow/sensor dataset;
7. samples POD coefficient ensembles;
8. compares reconstructed coefficients and velocity fields with held-out truth.

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


def discrete_vorticity(snapshot: np.ndarray) -> np.ndarray:
    """Compute lattice-grid vorticity omega = d(u_y)/dx - d(u_x)/dy."""
    ux, uy = snapshot
    return np.gradient(uy, axis=1) - np.gradient(ux, axis=0)


def plot_original_dataset(
    snapshots: np.ndarray,
    snapshot_times: np.ndarray,
    solid: np.ndarray,
    sensors: np.ndarray,
    observations: np.ndarray,
    coefficients: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the flow field, sensors, sensor traces, and POD coefficient data."""
    snapshot = snapshots[-1]
    speed = np.sqrt(snapshot[0] ** 2 + snapshot[1] ** 2)
    vorticity = discrete_vorticity(snapshot)
    speed_masked = np.ma.masked_where(solid, speed)
    vorticity_masked = np.ma.masked_where(solid, vorticity)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))

    image = axes[0, 0].imshow(
        speed_masked, origin="lower", aspect="equal", interpolation="nearest"
    )
    axes[0, 0].scatter(
        sensors[:, 1],
        sensors[:, 0],
        s=52,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
        label="velocity sensors",
    )
    axes[0, 0].contour(
        solid.astype(float),
        levels=[0.5],
        origin="lower",
        linewidths=1.0,
        colors="black",
    )
    axes[0, 0].set_title(
        f"Instantaneous speed and sensors, lattice time {snapshot_times[-1]}"
    )
    axes[0, 0].set_xlabel("lattice x-index")
    axes[0, 0].set_ylabel("lattice y-index")
    axes[0, 0].legend(loc="upper right")
    fig.colorbar(image, ax=axes[0, 0], label="speed")

    fluid_vorticity = np.abs(vorticity[~solid])
    vmax = max(float(np.percentile(fluid_vorticity, 99.0)), 1.0e-12)
    image = axes[0, 1].imshow(
        vorticity_masked,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm",
    )
    axes[0, 1].scatter(
        sensors[:, 1],
        sensors[:, 0],
        s=35,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
    )
    axes[0, 1].contour(
        solid.astype(float),
        levels=[0.5],
        origin="lower",
        linewidths=1.0,
        colors="black",
    )
    axes[0, 1].set_title("Instantaneous vorticity and sensors")
    axes[0, 1].set_xlabel("lattice x-index")
    axes[0, 1].set_ylabel("lattice y-index")
    fig.colorbar(image, ax=axes[0, 1], label=r"$\omega$")

    for j in range(min(6, observations.shape[1])):
        sensor_number = j // 2 + 1
        component = "u_x" if j % 2 == 0 else "u_y"
        axes[1, 0].plot(
            snapshot_times,
            observations[:, j],
            linewidth=1.0,
            label=f"sensor {sensor_number} {component}",
        )
    axes[1, 0].set_title("Representative sparse-observation time series")
    axes[1, 0].set_xlabel("lattice time")
    axes[1, 0].set_ylabel("velocity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(ncol=2, fontsize=8)

    for k in range(min(4, coefficients.shape[1])):
        axes[1, 1].plot(
            snapshot_times,
            coefficients[:, k],
            linewidth=1.0,
            label=fr"$a_{k+1}(t)$",
        )
    axes[1, 1].set_title("POD coefficient trajectories")
    axes[1, 1].set_xlabel("lattice time")
    axes[1, 1].set_ylabel("coefficient")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("Cylinder-wake modal-reconstruction dataset", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def reconstruct_velocity_from_coefficients(
    pod: dict, coefficients: np.ndarray
) -> np.ndarray:
    """Reconstruct a velocity field from the fixed training POD basis."""
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    modes = np.asarray(pod["modes"][: len(coefficients)], dtype=np.float64)
    return np.asarray(pod["mean_field"], dtype=np.float64) + np.tensordot(
        coefficients, modes, axes=(0, 0)
    )


def plot_modal_coefficient_reconstruction(
    snapshot_times: np.ndarray,
    test_anchors: np.ndarray,
    truth: np.ndarray,
    ensemble_means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: Path,
) -> None:
    """Compare true and reconstructed POD coefficients on held-out data."""
    n_show = min(6, truth.shape[1])
    fig, axes = plt.subplots(
        n_show, 1, figsize=(11, 2.35 * n_show), sharex=True, squeeze=False
    )
    times = snapshot_times[test_anchors]

    for k in range(n_show):
        ax = axes[k, 0]
        ax.fill_between(
            times, lower[:, k], upper[:, k], alpha=0.25, label="90% SI interval"
        )
        ax.plot(times, truth[:, k], "o-", markersize=3, linewidth=1.0, label="true")
        ax.plot(
            times,
            ensemble_means[:, k],
            "s--",
            markersize=3,
            linewidth=1.0,
            label="ensemble mean",
        )
        ax.set_ylabel(fr"$a_{k+1}$")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(ncol=3)

    axes[-1, 0].set_xlabel("lattice time")
    fig.suptitle("Held-out POD coefficients: stochastic reconstruction versus truth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_velocity_field_reconstruction(
    true_snapshot: np.ndarray,
    true_pod_field: np.ndarray,
    predicted_field: np.ndarray,
    solid: np.ndarray,
    output_path: Path,
    *,
    lattice_time: int,
) -> None:
    """Compare true flow, true rank-r POD projection, and SI reconstruction."""
    true_speed = np.sqrt(np.sum(true_snapshot ** 2, axis=0))
    pod_speed = np.sqrt(np.sum(true_pod_field ** 2, axis=0))
    predicted_speed = np.sqrt(np.sum(predicted_field ** 2, axis=0))
    error = np.sqrt(np.sum((predicted_field - true_snapshot) ** 2, axis=0))

    fields = [true_speed, pod_speed, predicted_speed, error]
    titles = [
        "Held-out CFD speed",
        "Rank-r POD projection using true coefficients",
        "SI ensemble-mean POD reconstruction",
        "Pointwise velocity-vector error",
    ]
    masked_fields = [np.ma.masked_where(solid, field) for field in fields]
    shared_max = max(
        float(np.percentile(true_speed[~solid], 99.5)),
        float(np.percentile(pod_speed[~solid], 99.5)),
        float(np.percentile(predicted_speed[~solid], 99.5)),
    )
    error_max = max(float(np.percentile(error[~solid], 99.5)), 1.0e-12)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5))
    for index, ax in enumerate(axes.flat):
        vmax = shared_max if index < 3 else error_max
        image = ax.imshow(
            masked_fields[index],
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        ax.contour(
            solid.astype(float),
            levels=[0.5],
            origin="lower",
            linewidths=1.0,
            colors="black",
        )
        ax.set_title(titles[index])
        ax.set_xlabel("lattice x-index")
        ax.set_ylabel("lattice y-index")
        fig.colorbar(
            image, ax=ax, label="speed" if index < 3 else "absolute error"
        )

    fluid = ~solid
    denominator = max(np.linalg.norm(true_snapshot[:, fluid]), 1.0e-14)
    reconstruction_error = (
        np.linalg.norm((predicted_field - true_snapshot)[:, fluid]) / denominator
    )
    truncation_error = (
        np.linalg.norm((true_pod_field - true_snapshot)[:, fluid]) / denominator
    )
    fig.suptitle(
        f"Velocity reconstruction at lattice time {lattice_time}; "
        f"SI relative L2 error={reconstruction_error:.3e}, "
        f"POD truncation error={truncation_error:.3e}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--burn-in", type=int, default=6000)
    p.add_argument("--snapshot-stride", type=int, default=20)
    p.add_argument("--inlet-velocity", type=float, default=0.06)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--cylinder-radius", type=int, default=10)
    p.add_argument("--cylinder-x", type=int, default=48)
    p.add_argument("--perturbation", type=float, default=1e-3)
    p.add_argument("--flow-seed", type=int, default=7)

    p.add_argument("--n-sensors", type=int, default=4)
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
    p.add_argument(
        "--n-reconstruction-plots",
        type=int,
        default=40,
        help="Number of held-out conditions used in coefficient comparison plots.",
    )
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
    plot_original_dataset(
        snapshots=snapshots,
        snapshot_times=times,
        solid=solid,
        sensors=sensors,
        observations=observations,
        coefficients=coefficients,
        output_path=out / "original_dataset_and_sensors.png",
    )

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

    # Evaluate held-out conditions with independent ensemble draws.
    test_y, test_u = pairs["test"]["Y"], pairs["test"]["U"]
    n_eval = min(args.n_reconstruction_plots, len(test_y))
    rmses, spreads = [], []
    first_samples = None
    ensemble_means, lower_intervals, upper_intervals = [], [], []

    for i in range(n_eval):
        samples = sample_conditional_sde(
            fit["model"], test_y[i], fit["y_scaler"], fit["u_scaler"],
            interp_cfg, n_samples=args.ensemble_size, n_steps=args.sde_steps,
            device=fit["device"], seed=1000 + i,
        )
        metrics = conditional_ensemble_metrics(samples, test_u[i])
        rmses.append(metrics["ensemble_mean_rmse"])
        spreads.append(metrics["ensemble_spread_rms"])
        ensemble_means.append(np.mean(samples, axis=0))
        lower_intervals.append(np.quantile(samples, 0.05, axis=0))
        upper_intervals.append(np.quantile(samples, 0.95, axis=0))
        if i == 0:
            first_samples = samples

    ensemble_means = np.asarray(ensemble_means)
    lower_intervals = np.asarray(lower_intervals)
    upper_intervals = np.asarray(upper_intervals)

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

        evaluated_anchors = pairs["test"]["anchors"][:n_eval]
        plot_modal_coefficient_reconstruction(
            snapshot_times=times,
            test_anchors=evaluated_anchors,
            truth=test_u[:n_eval],
            ensemble_means=ensemble_means,
            lower=lower_intervals,
            upper=upper_intervals,
            output_path=out / "pod_coefficients_reconstruction_vs_truth.png",
        )

        first_anchor = int(evaluated_anchors[0])
        true_pod_field = reconstruct_velocity_from_coefficients(pod, test_u[0])
        predicted_field = reconstruct_velocity_from_coefficients(
            pod, ensemble_means[0]
        )
        plot_velocity_field_reconstruction(
            true_snapshot=snapshots[first_anchor],
            true_pod_field=true_pod_field,
            predicted_field=predicted_field,
            solid=solid,
            output_path=out / "velocity_field_reconstruction_vs_truth.png",
            lattice_time=int(times[first_anchor]),
        )

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
        evaluated_test_anchors=pairs["test"]["anchors"][:n_eval],
        evaluated_true_coefficients=test_u[:n_eval],
        evaluated_ensemble_means=ensemble_means,
        evaluated_interval_05=lower_intervals,
        evaluated_interval_95=upper_intervals,
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
