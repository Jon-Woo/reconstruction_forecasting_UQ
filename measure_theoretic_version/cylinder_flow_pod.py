#!/usr/bin/env python3
"""
Cylinder-wake flow and proper orthogonal decomposition (POD).

This script:
1. approximates two-dimensional incompressible channel flow past a circular
   cylinder using a D2Q9 BGK lattice-Boltzmann method (LBM);
2. records velocity snapshots after a transient;
3. computes velocity POD modes in the discrete kinetic-energy inner product;
4. verifies orthonormality, the POD/SVD energy identity, and reconstruction error;
5. writes figures and numerical data.

The LBM is intended as a compact, reproducible demonstration, not as a
benchmark-grade CFD solver. Grid refinement and benchmark validation are needed
for quantitative drag, lift, pressure-drop, or Strouhal-number claims.

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# D2Q9 velocities ordered as:
# rest, east, north, west, south, northeast, northwest, southwest, southeast
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
     [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
             dtype=np.float64)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class Config:
    nx: int = 240
    ny: int = 96
    steps: int = 8000
    burn_in: int = 3500
    snapshot_stride: int = 30
    n_modes_plot: int = 6
    inlet_velocity: float = 0.06
    reynolds: float = 100.0
    cylinder_radius: int = 10
    cylinder_x: int = 48
    cylinder_y: int = 48
    perturbation: float = 1.0e-3
    seed: int = 7
    output_dir: str = "cylinder_pod_output"


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Return D2Q9 equilibrium populations with shape (9, ny, nx)."""
    cu = C[:, 0, None, None] * ux[None, :, :] + C[:, 1, None, None] * uy[None, :, :]
    u2 = ux * ux + uy * uy
    return W[:, None, None] * rho[None, :, :] * (
        1.0 + cu / CS2 + 0.5 * (cu * cu) / (CS2 * CS2)
        - 0.5 * u2[None, :, :] / CS2
    )


def macroscopic(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute density and velocity from D2Q9 populations."""
    rho = np.sum(f, axis=0)
    rho_safe = np.maximum(rho, 1.0e-14)
    ux = np.sum(f * C[:, 0, None, None], axis=0) / rho_safe
    uy = np.sum(f * C[:, 1, None, None], axis=0) / rho_safe
    return rho, ux, uy


def make_solid_mask(cfg: Config) -> np.ndarray:
    """No-slip top/bottom walls plus a circular cylinder."""
    yy, xx = np.mgrid[0:cfg.ny, 0:cfg.nx]
    cylinder = ((xx - cfg.cylinder_x) ** 2 + (yy - cfg.cylinder_y) ** 2
                <= cfg.cylinder_radius ** 2)
    walls = (yy == 0) | (yy == cfg.ny - 1)
    return cylinder | walls


def validate_config(cfg: Config) -> None:
    if cfg.nx < 40 or cfg.ny < 24:
        raise ValueError("Use nx >= 40 and ny >= 24.")
    if not (0 < cfg.inlet_velocity < 0.15):
        raise ValueError("For this weakly compressible LBM, use 0 < inlet_velocity < 0.15.")
    if cfg.reynolds <= 0:
        raise ValueError("Reynolds number must be positive.")
    if cfg.cylinder_radius < 3:
        raise ValueError("Cylinder radius must be at least 3 lattice nodes.")
    if not (cfg.cylinder_radius + 2 <= cfg.cylinder_x < cfg.nx - cfg.cylinder_radius - 2):
        raise ValueError("Cylinder does not fit in the x direction.")
    if not (cfg.cylinder_radius + 2 <= cfg.cylinder_y < cfg.ny - cfg.cylinder_radius - 2):
        raise ValueError("Cylinder does not fit in the y direction.")
    if cfg.steps <= cfg.burn_in:
        raise ValueError("steps must exceed burn_in.")
    if cfg.snapshot_stride <= 0:
        raise ValueError("snapshot_stride must be positive.")


def inlet_profile(cfg: Config) -> np.ndarray:
    """
    Parabolic channel profile with mean approximately cfg.inlet_velocity.

    For y in [0,H], u(y) = 6 U_mean (y/H)(1-y/H), whose continuous
    cross-sectional mean is U_mean.
    """
    y = np.arange(cfg.ny, dtype=np.float64)
    H = float(cfg.ny - 1)
    profile = 6.0 * cfg.inlet_velocity * (y / H) * (1.0 - y / H)
    profile[0] = profile[-1] = 0.0
    return profile


def run_lbm(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a compact D2Q9 BGK simulation and return snapshot arrays.

    Boundary treatment:
    - halfway/full-way-style local bounce-back on solid nodes;
    - equilibrium velocity/density prescription at inlet;
    - first-order zero-gradient copying at outlet.

    This is deliberately simple and suitable for qualitative POD demonstrations.
    """
    validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    solid = make_solid_mask(cfg)
    fluid = ~solid

    diameter = 2.0 * cfg.cylinder_radius
    nu = cfg.inlet_velocity * diameter / cfg.reynolds
    tau = 0.5 + nu / CS2
    omega = 1.0 / tau
    if tau <= 0.5:
        raise ValueError("LBM relaxation time must be > 0.5.")
    if tau < 0.53:
        print(f"Warning: tau={tau:.5f} is close to 0.5; stability may be poor.")

    u_in = inlet_profile(cfg)
    rho = np.ones((cfg.ny, cfg.nx), dtype=np.float64)
    ux = np.tile(u_in[:, None], (1, cfg.nx))
    uy = cfg.perturbation * rng.standard_normal((cfg.ny, cfg.nx))
    # Localize perturbations downstream and impose zero velocity on solids.
    x = np.arange(cfg.nx)[None, :]
    uy *= np.exp(-((x - 2.2 * cfg.cylinder_x) / (0.35 * cfg.nx)) ** 2)
    ux[solid] = 0.0
    uy[solid] = 0.0
    f = equilibrium(rho, ux, uy)

    snapshots = []
    snapshot_times = []

    for step in range(cfg.steps):
        rho, ux, uy = macroscopic(f)
        ux[solid] = 0.0
        uy[solid] = 0.0

        # BGK collision on all nodes; solid values will be handled by bounce-back.
        feq = equilibrium(rho, ux, uy)
        f_post = f - omega * (f - feq)

        # Streaming: periodic roll is subsequently overwritten at inlet/outlet.
        f_stream = np.empty_like(f_post)
        for i, (cx, cy) in enumerate(C):
            f_stream[i] = np.roll(np.roll(f_post[i], cy, axis=0), cx, axis=1)

        # Local bounce-back at solid nodes.
        bounced = f_stream[:, solid].copy()
        f_stream[:, solid] = bounced[OPPOSITE, :]

        # Inlet: prescribe parabolic velocity and rho=1 using equilibrium.
        rho_left = np.ones((cfg.ny, 1), dtype=np.float64)
        ux_left = u_in[:, None]
        uy_left = np.zeros((cfg.ny, 1), dtype=np.float64)
        f_left = equilibrium(rho_left, ux_left, uy_left)
        f_stream[:, :, 0] = f_left[:, :, 0]

        # Outlet: simple zero-normal-gradient extrapolation.
        f_stream[:, :, -1] = f_stream[:, :, -2]

        # Re-enforce wall populations by bounce-back consistency.
        f = f_stream

        if step >= cfg.burn_in and (step - cfg.burn_in) % cfg.snapshot_stride == 0:
            _, ux_s, uy_s = macroscopic(f)
            ux_s[solid] = 0.0
            uy_s[solid] = 0.0
            snapshots.append(np.stack((ux_s.copy(), uy_s.copy()), axis=0))
            snapshot_times.append(step)

        if step % max(1, cfg.steps // 10) == 0:
            speed_max = np.sqrt(ux * ux + uy * uy)[fluid].max()
            print(
                f"step {step:6d}/{cfg.steps}, tau={tau:.5f}, "
                f"max|u|={speed_max:.4f}, snapshots={len(snapshots)}"
            )
            if not np.isfinite(f).all():
                raise FloatingPointError("Non-finite LBM populations encountered.")

    if len(snapshots) < 2:
        raise RuntimeError("At least two snapshots are required for POD.")
    return np.asarray(snapshots), np.asarray(snapshot_times), solid, np.array([nu, tau, omega])


def compute_pod(
    snapshots: np.ndarray,
    solid: np.ndarray,
) -> dict:
    """
    Compute snapshot POD for a two-component velocity field.

    Let q_j be the vector containing u and v at every fluid node, let qbar be
    the sample mean, and let X=[q_1-qbar,...,q_m-qbar]/sqrt(m). Since all
    lattice cells have equal area, the constant quadrature weight cancels in
    normalized modes. The thin SVD X=U S V^T gives POD modes U and eigenvalues
    lambda_k=S_k^2.

    The returned spatial modes have shape (r, 2, ny, nx), with zeros on solids.
    """
    if snapshots.ndim != 4 or snapshots.shape[1] != 2:
        raise ValueError("snapshots must have shape (m, 2, ny, nx).")
    m, _, ny, nx = snapshots.shape
    fluid = ~solid
    n_fluid = int(fluid.sum())

    # State matrix Q has columns q_j and Euclidean inner product corresponding
    # to equal-area kinetic-energy quadrature on fluid nodes.
    Q = np.empty((2 * n_fluid, m), dtype=np.float64)
    for j in range(m):
        Q[:n_fluid, j] = snapshots[j, 0, fluid]
        Q[n_fluid:, j] = snapshots[j, 1, fluid]

    mean = np.mean(Q, axis=1, keepdims=True)
    X = (Q - mean) / np.sqrt(m)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = s * s
    total = float(np.sum(eigenvalues))
    energy_fraction = eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    cumulative_energy = np.cumsum(energy_fraction)

    # Coefficients of the unscaled centered snapshots:
    # Q-mean = sqrt(m) U diag(s) V^T, so A=U^T(Q-mean)=sqrt(m)diag(s)V^T.
    coefficients = np.sqrt(m) * s[:, None] * Vt

    modes = np.zeros((U.shape[1], 2, ny, nx), dtype=np.float64)
    for k in range(U.shape[1]):
        modes[k, 0, fluid] = U[:n_fluid, k]
        modes[k, 1, fluid] = U[n_fluid:, k]

    mean_field = np.zeros((2, ny, nx), dtype=np.float64)
    mean_field[0, fluid] = mean[:n_fluid, 0]
    mean_field[1, fluid] = mean[n_fluid:, 0]

    gram = U.T @ U
    orthogonality_error = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord=2))

    # Exact rank-r SVD errors and a direct check for a representative r.
    centered_norm_sq = float(np.linalg.norm(Q - mean, ord="fro") ** 2)
    svd_energy_identity_error = abs(
        centered_norm_sq - m * float(np.sum(eigenvalues))
    ) / max(centered_norm_sq, 1.0e-30)

    return {
        "Q": Q,
        "mean_vector": mean,
        "mean_field": mean_field,
        "U": U,
        "singular_values": s,
        "Vt": Vt,
        "eigenvalues": eigenvalues,
        "energy_fraction": energy_fraction,
        "cumulative_energy": cumulative_energy,
        "coefficients": coefficients,
        "modes": modes,
        "orthogonality_error": orthogonality_error,
        "svd_energy_identity_error": svd_energy_identity_error,
        "fluid_mask": fluid,
    }


def reconstruct_snapshot(pod: dict, snapshot_index: int, rank: int) -> np.ndarray:
    """Return the rank-r POD reconstruction of one velocity snapshot."""
    Q = pod["Q"]
    mean = pod["mean_vector"]
    U = pod["U"]
    A = pod["coefficients"]
    fluid = pod["fluid_mask"]
    ny, nx = fluid.shape
    n_fluid = int(fluid.sum())

    rank = min(max(0, rank), U.shape[1])
    q_r = mean[:, 0] + U[:, :rank] @ A[:rank, snapshot_index]
    field = np.zeros((2, ny, nx), dtype=np.float64)
    field[0, fluid] = q_r[:n_fluid]
    field[1, fluid] = q_r[n_fluid:]
    return field


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Second-order interior/one-sided-edge finite-difference vorticity dv/dx-du/dy."""
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    return dv_dx - du_dy


def add_cylinder(ax: plt.Axes, cfg: Config) -> None:
    ax.add_patch(Circle((cfg.cylinder_x, cfg.cylinder_y), cfg.cylinder_radius,
                        facecolor="white", edgecolor="black", linewidth=1.0))


def plot_results(cfg: Config, snapshots: np.ndarray, times: np.ndarray, pod: dict) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    extent = [0, cfg.nx - 1, 0, cfg.ny - 1]
    last = snapshots[-1]
    speed = np.sqrt(last[0] ** 2 + last[1] ** 2)
    vort = vorticity(last[0], last[1])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(speed, origin="lower", extent=extent, aspect="auto")
    add_cylinder(ax, cfg)
    ax.set_title(f"Instantaneous speed at lattice step {times[-1]}")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, label="speed")
    fig.tight_layout()
    fig.savefig(out / "instantaneous_speed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    vmax = np.nanpercentile(np.abs(vort[~make_solid_mask(cfg)]), 99)
    im = ax.imshow(vort, origin="lower", extent=extent, aspect="auto",
                   vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    add_cylinder(ax, cfg)
    ax.set_title(f"Instantaneous vorticity at lattice step {times[-1]}")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, label=r"$\omega_z$")
    fig.tight_layout()
    fig.savefig(out / "instantaneous_vorticity.png", dpi=180)
    plt.close(fig)

    mean = pod["mean_field"]
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(np.sqrt(mean[0] ** 2 + mean[1] ** 2), origin="lower",
                   extent=extent, aspect="auto")
    add_cylinder(ax, cfg)
    ax.set_title("Temporal-mean speed")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, label="mean speed")
    fig.tight_layout()
    fig.savefig(out / "mean_speed.png", dpi=180)
    plt.close(fig)

    nplot = min(cfg.n_modes_plot, pod["modes"].shape[0])
    for k in range(nplot):
        mode = pod["modes"][k]
        vort_mode = vorticity(mode[0], mode[1])
        vmax = np.nanpercentile(np.abs(vort_mode[~make_solid_mask(cfg)]), 99)
        vmax = max(vmax, 1e-14)
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(vort_mode, origin="lower", extent=extent, aspect="auto",
                       vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        add_cylinder(ax, cfg)
        ax.set_title(
            f"POD mode {k+1}: vorticity; energy fraction "
            f"{100*pod['energy_fraction'][k]:.2f}%"
        )
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
        fig.colorbar(im, ax=ax, label=r"$\nabla\times\phi_k$")
        fig.tight_layout()
        fig.savefig(out / f"pod_mode_{k+1:02d}_vorticity.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(1, len(pod["energy_fraction"]) + 1)
    ax.semilogy(indices, np.maximum(pod["energy_fraction"], 1e-18), "o-")
    ax.set_xlabel("POD mode index")
    ax.set_ylabel("fraction of fluctuation kinetic energy")
    ax.set_title("POD eigenvalue spectrum")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pod_energy_spectrum.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(indices, pod["cumulative_energy"], "o-")
    ax.set_xlabel("number of retained POD modes")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Cumulative POD energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pod_cumulative_energy.png", dpi=180)
    plt.close(fig)

    ncoeff = min(4, pod["coefficients"].shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(ncoeff):
        ax.plot(times, pod["coefficients"][k], label=fr"$a_{k+1}$")
    ax.set_xlabel("lattice time step")
    ax.set_ylabel("POD coefficient")
    ax.set_title("Leading POD temporal coefficients")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pod_coefficients.png", dpi=180)
    plt.close(fig)

    # Rank-r reconstruction comparison.
    r = min(6, pod["U"].shape[1])
    original = snapshots[-1]
    reconstruction = reconstruct_snapshot(pod, len(snapshots) - 1, r)
    err = original - reconstruction
    fields = [
        ("original", vorticity(original[0], original[1])),
        (f"rank-{r} POD reconstruction", vorticity(reconstruction[0], reconstruction[1])),
        ("reconstruction error", vorticity(err[0], err[1])),
    ]
    vmax = max(np.nanpercentile(np.abs(z[~make_solid_mask(cfg)]), 99)
               for _, z in fields[:2])
    for name, z in fields:
        fig, ax = plt.subplots(figsize=(12, 4))
        local_vmax = vmax if name != "reconstruction error" else max(
            np.nanpercentile(np.abs(z[~make_solid_mask(cfg)]), 99), 1e-14
        )
        im = ax.imshow(z, origin="lower", extent=extent, aspect="auto",
                       vmin=-local_vmax, vmax=local_vmax, cmap="RdBu_r")
        add_cylinder(ax, cfg)
        ax.set_title(name.capitalize() + " vorticity")
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        safe_name = name.replace(" ", "_").replace("-", "_")
        fig.savefig(out / f"{safe_name}_vorticity.png", dpi=180)
        plt.close(fig)


def save_results(
    cfg: Config,
    snapshots: np.ndarray,
    times: np.ndarray,
    solid: np.ndarray,
    lbm_params: np.ndarray,
    pod: dict,
) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "cylinder_pod_data.npz",
        snapshots=snapshots,
        snapshot_times=times,
        solid_mask=solid,
        viscosity=lbm_params[0],
        relaxation_time=lbm_params[1],
        collision_frequency=lbm_params[2],
        mean_field=pod["mean_field"],
        modes=pod["modes"],
        eigenvalues=pod["eigenvalues"],
        energy_fraction=pod["energy_fraction"],
        cumulative_energy=pod["cumulative_energy"],
        coefficients=pod["coefficients"],
    )

    r90 = int(np.searchsorted(pod["cumulative_energy"], 0.90) + 1)
    r99 = int(np.searchsorted(pod["cumulative_energy"], 0.99) + 1)
    with (out / "diagnostics.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"number of snapshots: {snapshots.shape[0]}\n")
        fh.write(f"fluid nodes: {(~solid).sum()}\n")
        fh.write(f"kinematic viscosity (lattice units): {lbm_params[0]:.12g}\n")
        fh.write(f"relaxation time tau: {lbm_params[1]:.12g}\n")
        fh.write(f"collision frequency omega: {lbm_params[2]:.12g}\n")
        fh.write(f"POD orthogonality spectral-norm error: "
                 f"{pod['orthogonality_error']:.6e}\n")
        fh.write(f"SVD energy identity relative error: "
                 f"{pod['svd_energy_identity_error']:.6e}\n")
        fh.write(f"modes for >=90% fluctuation energy: {r90}\n")
        fh.write(f"modes for >=99% fluctuation energy: {r99}\n")
        fh.write("\nmode, eigenvalue, energy_fraction, cumulative_energy\n")
        for k, (lam, ef, ce) in enumerate(zip(
            pod["eigenvalues"], pod["energy_fraction"], pod["cumulative_energy"]
        ), start=1):
            fh.write(f"{k}, {lam:.12e}, {ef:.12e}, {ce:.12e}\n")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=Config.nx)
    parser.add_argument("--ny", type=int, default=Config.ny)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--burn-in", type=int, default=Config.burn_in)
    parser.add_argument("--snapshot-stride", type=int, default=Config.snapshot_stride)
    parser.add_argument("--n-modes-plot", type=int, default=Config.n_modes_plot)
    parser.add_argument("--inlet-velocity", type=float, default=Config.inlet_velocity)
    parser.add_argument("--reynolds", type=float, default=Config.reynolds)
    parser.add_argument("--cylinder-radius", type=int, default=Config.cylinder_radius)
    parser.add_argument("--cylinder-x", type=int, default=Config.cylinder_x)
    parser.add_argument("--cylinder-y", type=int, default=None)
    parser.add_argument("--perturbation", type=float, default=Config.perturbation)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    args = parser.parse_args()
    cy = args.cylinder_y if args.cylinder_y is not None else args.ny // 2
    return Config(
        nx=args.nx,
        ny=args.ny,
        steps=args.steps,
        burn_in=args.burn_in,
        snapshot_stride=args.snapshot_stride,
        n_modes_plot=args.n_modes_plot,
        inlet_velocity=args.inlet_velocity,
        reynolds=args.reynolds,
        cylinder_radius=args.cylinder_radius,
        cylinder_x=args.cylinder_x,
        cylinder_y=cy,
        perturbation=args.perturbation,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    print(cfg)
    snapshots, times, solid, lbm_params = run_lbm(cfg)
    pod = compute_pod(snapshots, solid)
    save_results(cfg, snapshots, times, solid, lbm_params, pod)
    plot_results(cfg, snapshots, times, pod)

    r90 = int(np.searchsorted(pod["cumulative_energy"], 0.90) + 1)
    r99 = int(np.searchsorted(pod["cumulative_energy"], 0.99) + 1)
    print("\nPOD diagnostics")
    print(f"  snapshots: {snapshots.shape[0]}")
    print(f"  orthogonality error: {pod['orthogonality_error']:.3e}")
    print(f"  energy identity relative error: {pod['svd_energy_identity_error']:.3e}")
    print(f"  modes for 90% energy: {r90}")
    print(f"  modes for 99% energy: {r99}")
    print(f"  results written to: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()
