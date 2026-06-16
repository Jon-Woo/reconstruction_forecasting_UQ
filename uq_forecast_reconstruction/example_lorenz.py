"""
Example: probabilistic delay-coordinate forecast and reconstruction for a
stochastically forced Lorenz system.

This script demonstrates how to:
  1. Generate stochastic trajectory data.
  2. Build time-delay embedding pairs.
  3. Train one conditional stochastic interpolant for forecasting.
  4. Train a second conditional stochastic interpolant for reconstruction.
  5. Draw ensembles and call visualization.py to save plots.

Run:
    python example_lorenz.py --quick
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

LORENZ_COMPONENTS = ("x", "y", "z")


def simulate_stochastic_lorenz(
    n_steps: int = 80_000,
    dt: float = 0.005,
    noise_level: float = 1.0,
    burn_in: int = 5_000,
    seed: int = 0,
) -> np.ndarray:
    """Euler-Maruyama simulation of additive-noise Lorenz-63."""
    rng = np.random.default_rng(seed)
    lorenz_sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    x = np.empty((n_steps, 3), dtype=np.float32)
    x[0] = np.array([1.0, 1.0, 20.0], dtype=np.float32)
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps - 1):
        a, b, c = x[k]
        drift = np.array([
            lorenz_sigma * (b - a),
            a * (rho - c) - b,
            a * b - beta * c,
        ], dtype=np.float32)
        x[k + 1] = x[k] + drift * dt + noise_level * sqrt_dt * rng.standard_normal(3)
    return x[burn_in:]


def split_train_test(z: np.ndarray, y: np.ndarray, frac_train: float = 0.8):
    n = z.shape[0]
    n_train = int(frac_train * n)
    return z[:n_train], y[:n_train], z[n_train:], y[n_train:]


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short smoke-test training.")
    parser.add_argument("--noise_level", type=float, default=2.0)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--delay_steps", type=int, default=10)
    parser.add_argument("--horizon_steps", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ensemble", type=int, default=256)
    parser.add_argument("--plot_dir", type=str, default="plots_lorenz/full")
    parser.add_argument("--n_plot_points", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_steps = 8_000
        train_steps = 100
        hidden = (64, 64)
        n_plot_points = min(args.n_plot_points, 60)
    else:
        n_steps = 80_000
        train_steps = args.train_steps
        hidden = (256, 256, 256)
        n_plot_points = args.n_plot_points

    print("Generating stochastic Lorenz data...")
    x = simulate_stochastic_lorenz(n_steps=n_steps, noise_level=args.noise_level)

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

    print("Training probabilistic forecast model p(x_p(t+horizon) | delay embedding)...")
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

    recon_data = make_delay_pairs(
        x,
        partial_indices=[0],
        delay_steps=args.delay_steps,
        m=args.m,
        horizon_steps=args.horizon_steps,
        task="reconstruction",
    )
    ztr_r, ytr_r, zte_r, yte_r = split_train_test(recon_data.z, recon_data.y)

    print("Training probabilistic reconstruction model p(x(t) | delay embedding)...")
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
    print(f"true y                 : {yte[idx]}")
    print(f"ensemble mean          : {mean[0]}")
    print(f"ensemble std           : {std[0]}")
    print(f"central 90% interval   : [{q05[0]}, {q95[0]}]")

    samples_r = recon_model.sample_numpy(zte_r[idx : idx + 1], n_samples=args.ensemble, n_steps=100)
    mean_r, std_r, q05_r, q95_r = ensemble_summary(samples_r)
    print("\nReconstruction ensemble for one test point")
    print(f"true x(t)              : {yte_r[idx]}")
    print(f"ensemble mean          : {mean_r[0]}")
    print(f"ensemble std           : {std_r[0]}")
    print(f"central 90% interval   : [{q05_r[0]}, {q95_r[0]}]")

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
        title="Lorenz: probabilistic one-step forecast from delay embeddings",
        component_names=["observed component at t + horizon"],
    ))
    saved_paths.append(plot_single_case_histograms(
        yte[idx],
        samples[0],
        plot_dir / "forecast_single_case_hist.png",
        title="Lorenz: forecast ensemble for one test embedding",
        component_names=["observed component at t + horizon"],
    ))

    n_plot_recon = min(n_plot_points, len(zte_r))
    recon_block = recon_model.sample_numpy(zte_r[:n_plot_recon], n_samples=args.ensemble, n_steps=100)
    saved_paths.append(plot_predictive_timeseries(
        yte_r[:n_plot_recon],
        recon_block,
        plot_dir / "reconstruction_timeseries.png",
        title="Lorenz: probabilistic reconstruction of the full state",
        component_names=list(LORENZ_COMPONENTS),
    ))
    saved_paths.append(plot_phase_portrait(
        yte_r[:n_plot_recon],
        recon_block,
        plot_dir / "reconstruction_phase_portrait.png",
        title="Lorenz reconstruction phase portrait",
        components=(0, 2),
        axis_labels=("x", "z"),
    ))
    saved_paths.append(plot_single_case_histograms(
        yte_r[idx],
        samples_r[0],
        plot_dir / "reconstruction_single_case_hist.png",
        title="Lorenz: reconstruction ensemble for one test embedding",
        component_names=list(LORENZ_COMPONENTS),
    ))

    print("Saved figures:")
    for path in saved_paths:
        print(f"  - {path}")
    maybe_show(args.show)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
