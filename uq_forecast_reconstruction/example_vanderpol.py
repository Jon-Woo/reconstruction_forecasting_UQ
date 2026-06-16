"""
Example: probabilistic delay-coordinate forecast and reconstruction for a
stochastically forced Van der Pol oscillator.

The full state is x(t) = [position, velocity].  The partial observation is only
position.  We train

    forecast:       p(position(t + horizon) | delay embedding)
    reconstruction: p([position(t), velocity(t)] | delay embedding)

Run:
    python example_vanderpol.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from delay_si import make_delay_pairs, train_interpolant, ensemble_summary
from visualization import (
    maybe_show,
    plot_phase_portrait,
    plot_predictive_timeseries,
    plot_single_case_histograms,
    plot_training_loss,
)

VDP_COMPONENTS = ("position", "velocity")


def vanderpol_drift(state: np.ndarray, mu: float) -> np.ndarray:
    """Drift for dx/dt = v, dv/dt = mu(1 - x^2)v - x."""
    x, v = state
    return np.array([v, mu * (1.0 - x * x) * v - x], dtype=np.float32)


def rk4_drift_step(state: np.ndarray, dt: float, mu: float) -> np.ndarray:
    """Fourth-order Runge-Kutta step for the deterministic drift."""
    k1 = vanderpol_drift(state, mu)
    k2 = vanderpol_drift(state + 0.5 * dt * k1, mu)
    k3 = vanderpol_drift(state + 0.5 * dt * k2, mu)
    k4 = vanderpol_drift(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_stochastic_vanderpol(
    n_steps: int = 60_000,
    dt: float = 0.01,
    mu: float = 2.0,
    noise_level: float = 0.15,
    burn_in: int = 2_000,
    seed: int = 1,
) -> np.ndarray:
    """Additive-noise Van der Pol simulation.

    The deterministic drift is advanced by RK4 and additive noise is added as
    sigma * sqrt(dt) * normal.  This is a simple demonstration integrator, not a
    high-order SDE method.
    """
    rng = np.random.default_rng(seed)
    x = np.empty((n_steps, 2), dtype=np.float32)
    x[0] = np.array([2.0, 0.0], dtype=np.float32)
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps - 1):
        deterministic_next = rk4_drift_step(x[k], dt, mu)
        x[k + 1] = deterministic_next + noise_level * sqrt_dt * rng.standard_normal(2)
    return x[burn_in:]


def split_train_test(z: np.ndarray, y: np.ndarray, frac_train: float = 0.8):
    n = z.shape[0]
    n_train = int(frac_train * n)
    return z[:n_train], y[:n_train], z[n_train:], y[n_train:]


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short smoke-test training.")
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--noise_level", type=float, default=0.15)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--delay_steps", type=int, default=8)
    parser.add_argument("--horizon_steps", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ensemble", type=int, default=256)
    parser.add_argument("--plot_dir", type=str, default="plots_vanderpol/full")
    parser.add_argument("--n_plot_points", type=int, default=200)
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_steps = 10_000
        train_steps = 120
        hidden = (64, 64)
        n_plot_points = min(args.n_plot_points, 80)
    else:
        n_steps = 60_000
        train_steps = args.train_steps
        hidden = (256, 256, 256)
        n_plot_points = args.n_plot_points

    print("Generating stochastic Van der Pol data...")
    x = simulate_stochastic_vanderpol(
        n_steps=n_steps,
        dt=args.dt,
        mu=args.mu,
        noise_level=args.noise_level,
    )

    # Forecast: observe only position x(t), forecast position x(t+horizon).
    forecast_data = make_delay_pairs(
        x,
        partial_indices=[0],
        delay_steps=args.delay_steps,
        m=args.m,
        horizon_steps=args.horizon_steps,
        task="forecast",
        predict_increment=False,
    )
    ztr, ytr, zte, yte = split_train_test(forecast_data.z, forecast_data.y)

    print("Training probabilistic forecast model p(position(t+horizon) | delay embedding)...")
    forecast_model = train_interpolant(
        ztr,
        ytr,
        hidden=hidden,
        eps=0.25,
        batch_size=min(args.batch_size, ztr.shape[0]),
        n_steps=train_steps,
        lr=1e-3,
        print_every=max(100, train_steps // 5),
        metadata=forecast_data.metadata,
    )

    # Reconstruction: recover both position and velocity from delayed positions.
    recon_data = make_delay_pairs(
        x,
        partial_indices=[0],
        delay_steps=args.delay_steps,
        m=args.m,
        horizon_steps=args.horizon_steps,
        task="reconstruction",
    )
    ztr_r, ytr_r, zte_r, yte_r = split_train_test(recon_data.z, recon_data.y)

    print("Training probabilistic reconstruction model p([position, velocity] | delay embedding)...")
    recon_model = train_interpolant(
        ztr_r,
        ytr_r,
        hidden=hidden,
        eps=0.25,
        batch_size=min(args.batch_size, ztr_r.shape[0]),
        n_steps=train_steps,
        lr=1e-3,
        print_every=max(100, train_steps // 5),
        metadata=recon_data.metadata,
    )

    idx = 0
    samples = forecast_model.sample_numpy(zte[idx : idx + 1], n_samples=args.ensemble, n_steps=100)
    mean, std, q05, q95 = ensemble_summary(samples)
    print("\nForecast ensemble for one test point")
    print(f"true position(t+horizon): {yte[idx]}")
    print(f"ensemble mean            : {mean[0]}")
    print(f"ensemble std             : {std[0]}")
    print(f"central 90% interval     : [{q05[0]}, {q95[0]}]")

    samples_r = recon_model.sample_numpy(zte_r[idx : idx + 1], n_samples=args.ensemble, n_steps=100)
    mean_r, std_r, q05_r, q95_r = ensemble_summary(samples_r)
    print("\nReconstruction ensemble for one test point")
    print(f"true [position, velocity]: {yte_r[idx]}")
    print(f"ensemble mean            : {mean_r[0]}")
    print(f"ensemble std             : {std_r[0]}")
    print(f"central 90% interval     : [{q05_r[0]}, {q95_r[0]}]")

    print("\nGenerating visualization figures...")
    saved_paths = []
    saved_paths.append(plot_training_loss(forecast_model.train_losses, plot_dir / "forecast_training_loss.png"))
    saved_paths.append(plot_training_loss(recon_model.train_losses, plot_dir / "reconstruction_training_loss.png"))

    n_plot_forecast = min(n_plot_points, len(zte))
    forecast_block = forecast_model.sample_numpy(zte[:n_plot_forecast], n_samples=args.ensemble, n_steps=100)
    saved_paths.append(plot_predictive_timeseries(
        yte[:n_plot_forecast],
        forecast_block,
        plot_dir / "forecast_timeseries.png",
        title="Van der Pol: probabilistic one-step forecast from delay embeddings",
        component_names=["position at t + horizon"],
    ))
    saved_paths.append(plot_single_case_histograms(
        yte[idx],
        samples[0],
        plot_dir / "forecast_single_case_hist.png",
        title="Van der Pol: forecast ensemble for one test embedding",
        component_names=["position at t + horizon"],
    ))

    n_plot_recon = min(n_plot_points, len(zte_r))
    recon_block = recon_model.sample_numpy(zte_r[:n_plot_recon], n_samples=args.ensemble, n_steps=100)
    saved_paths.append(plot_predictive_timeseries(
        yte_r[:n_plot_recon],
        recon_block,
        plot_dir / "reconstruction_timeseries.png",
        title="Van der Pol: probabilistic reconstruction of the full state",
        component_names=list(VDP_COMPONENTS),
    ))
    saved_paths.append(plot_phase_portrait(
        yte_r[:n_plot_recon],
        recon_block,
        plot_dir / "reconstruction_phase_portrait.png",
        title="Van der Pol reconstruction phase portrait",
        components=(0, 1),
        axis_labels=("position", "velocity"),
    ))
    saved_paths.append(plot_single_case_histograms(
        yte_r[idx],
        samples_r[0],
        plot_dir / "reconstruction_single_case_hist.png",
        title="Van der Pol: reconstruction ensemble for one test embedding",
        component_names=list(VDP_COMPONENTS),
    ))

    print("Saved figures:")
    for path in saved_paths:
        print(f"  - {path}")
    maybe_show(args.show)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
