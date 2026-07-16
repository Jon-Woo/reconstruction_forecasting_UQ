# Gaussian-Source Stochastic Interpolants for POD Modal Reconstruction from Sparse Cylinder-Wake Sensors

## 1. Purpose and files

This implementation learns the regular conditional probability kernel

\[
y\longmapsto \kappa(y,\cdot)
=\mathcal L(A_t\mid Y_t=y),
\]

where \(A_t=(a_1(t),\ldots,a_r(t))\) contains POD coefficients and \(Y_t\) is a delay vector built from sparse velocity sensors.

Files:

- `cylinder_flow_pod.py`: the supplied D2Q9 BGK cylinder-flow and POD code, renamed for import.
- `gaussian_source_si.py`: application-independent conditional stochastic-interpolant implementation.
- `cylinder_pod_si_example.py`: cylinder-flow data generation, training-pair construction, training, SDE sampling, and plots.

The CFD model is a compact lattice-Boltzmann demonstration, not a benchmark-grade direct numerical simulation. POD/SVD identities are exact for the resulting finite snapshot matrix, but the snapshots retain discretization and boundary-condition error. The supplied cylinder documentation carefully distinguishes these two levels. fileciteturn4file12

---

## 2. Cylinder-flow trajectory and POD target

The continuum reference problem is the incompressible Navier--Stokes system

\[
\partial_t u+(u\cdot\nabla)u+\nabla p-\frac1{\mathrm{Re}}\Delta u=0,
\qquad \nabla\cdot u=0,
\]

in a channel with a circular obstacle and no-slip cylinder/wall conditions. The code approximates this system with a D2Q9 BGK lattice-Boltzmann method. Its viscosity relation and boundary approximations are documented in the supplied explanation. fileciteturn4file18

The default example integrates for 21,000 lattice steps, discards the first 6,000 as burn-in, and records every 20 steps. This gives 750 post-transient snapshots spanning 14,980 lattice steps. “Sufficiently long” is not a theorem: adequacy must be checked empirically by inspecting stationarity, multiple shedding cycles, POD-spectrum stability, and sensitivity to a longer burn-in or observation horizon.

### Leakage-safe POD

The temporal trajectory is split chronologically first. The mean field and POD basis are fitted **only on the raw training block**. Every training, validation, and test snapshot is then projected onto that fixed training basis:

\[
a_k(t_j)=\left\langle q_j-\bar q_{\mathrm{train}},
\phi_k^{\mathrm{train}}\right\rangle.
\]

This avoids leaking validation/test flow information into the target representation. For equal-area lattice cells, the Euclidean inner product differs from the discrete kinetic-energy inner product only by a positive constant. The snapshot SVD yields the finite-sample optimal rank-\(r\) subspace and exact discarded-energy identity described in the supplied POD derivation. fileciteturn4file13

---

## 3. Partial observations

Let the deterministic sensor locations be

\[
\mathcal S=\{(x_\ell,y_\ell)\}_{\ell=1}^{N_s}
\]

in the fluid wake. At snapshot time \(t_j\),

\[
x_p(t_j)=
\bigl(
u_x(x_1,y_1,t_j),u_y(x_1,y_1,t_j),\ldots,
u_x(x_{N_s},y_{N_s},t_j),u_y(x_{N_s},y_{N_s},t_j)
\bigr)\in\mathbb R^{2N_s}.
\]

The code selects approximately space-filling downstream fluid nodes unless the selection routine is replaced by user-defined coordinates.

The target is

\[
U_j=A_{t_j}=(a_1(t_j),\ldots,a_r(t_j))\in\mathbb R^r.
\]

Thus the learned model performs **contemporaneous modal reconstruction**, not future forecasting.

---

## 4. Delay coordinates and non-overlapping pairs

Let snapshots be separated by \(\Delta t_{\mathrm{snap}}\) lattice steps. The requested physical delay \(\tau\) must satisfy

\[
\tau=\ell\,\Delta t_{\mathrm{snap}}
\]

for an integer `delay_steps` \(\ell\). For embedding dimension \(m\),

\[
Y_j=
\left[
x_p(t_j),x_p(t_{j-\ell}),\ldots,
x_p(t_{j-(m-1)\ell})
\right]\in\mathbb R^{2N_s m}.
\]

The corresponding target is \(U_j=A_{t_j}\).

The set of raw indices used by window \(j\) is

\[
\mathcal I_j=\{j,j-\ell,\ldots,j-(m-1)\ell\}.
\]

Within each chronological train/validation/test block, anchors are spaced by

\[
h=(m-1)\ell+g+1,
\]

where \(g\ge 0\) is a user-specified guard gap in snapshot indices. Consequently, retained windows have disjoint raw index sets. The code checks this exactly.

**Important qualification.** Disjoint windows do not imply probabilistic independence for a dynamical trajectory. They eliminate shared measurements and reduce short-range dependence. Approximate independence additionally requires a mixing assumption and a gap large relative to the correlation time. The effective sample size should therefore be diagnosed from coefficient/sensor autocorrelations or block-bootstrap methods.

Young and Graham use delay vectors of the same form for reconstruction from partial observations and emphasize that \(m\) and \(\tau\) require empirical selection. fileciteturn4file14 Classical Takens conclusions do not automatically apply to a general stochastic or numerically forced system; here the target is correctly treated as a conditional law rather than an assumed deterministic inverse.

---

## 5. Gaussian-source stochastic interpolant

After standardizing \(Y\) and \(U\) using training statistics, let

\[
Z\sim\mathcal N(0,I_r),\qquad
B_s\sim\mathcal N(0,sI_r),
\]

with \(Z\), the Brownian motion \(B\), and \((Y,U)\) mutually independent. Define

\[
I_s=\alpha_s Z+\beta_sU+\rho_sB_s,\qquad s\in[0,1],
\]

using

\[
\alpha_s=1-s,\qquad
\beta_s=s^2,\qquad
\rho_s=\eta(1-s).
\]

Then

\[
I_0=Z,\qquad I_1=U.
\]

The use of \(\beta_s=s^2\) gives \(\dot\beta_0=0\), mirroring the endpoint-regularity choice reported by Chen et al. fileciteturn4file2 The broader stochastic-interpolant framework explicitly permits a standard Gaussian one-sided source. fileciteturn4file6

Define

\[
R_s=\dot\alpha_sZ+\dot\beta_sU+\dot\rho_sB_s.
\]

The population regression problem is

\[
\mathcal J(b)=
\int_0^1
\mathbb E\!\left[
\|b(s,I_s,Y)-R_s\|^2
\right]\,ds.
\]

Its unique \(L^2\)-minimizer is

\[
b^\star(s,I_s,Y)
=
\mathbb E[R_s\mid I_s,Y].
\]

The conditional-expectation projection identity gives, for every square-integrable \(b\),

\[
\mathcal J(b)=\mathcal J(b^\star)
+
\int_0^1
\mathbb E\!\left[
\|b(s,I_s,Y)-b^\star(s,I_s,Y)\|^2
\right]ds.
\]

Chen et al. use precisely this simulation-free conditional square-loss principle, sampling artificial time uniformly and replacing \(B_s\) by its marginal representation \(\sqrt{s}\,\varepsilon\). fileciteturn4file1

---

## 6. Numerical loss and training

For a minibatch \(\{(Y_i,U_i)\}_{i=1}^M\), independently draw

\[
S_i\sim\mathrm{Unif}(0,1),\quad
Z_i,\varepsilon_i\sim\mathcal N(0,I_r),
\quad B_{S_i}=\sqrt{S_i}\,\varepsilon_i.
\]

Construct

\[
I_i=\alpha_{S_i}Z_i+\beta_{S_i}U_i+\rho_{S_i}B_{S_i},
\]

\[
R_i=\dot\alpha_{S_i}Z_i+\dot\beta_{S_i}U_i
+\dot\rho_{S_i}B_{S_i}.
\]

The code minimizes

\[
\widehat{\mathcal J}_M(\theta)
=
\frac1M\sum_{i=1}^M
\|b_\theta(S_i,I_i,Y_i)-R_i\|_2^2
\]

using AdamW, gradient clipping, validation-based early stopping, and a multilayer perceptron with Fourier features in \(s\). This is an unbiased Monte Carlo estimator of the time-integrated empirical objective when minibatches and artificial variables are sampled uniformly/independently.

Only training statistics are used for normalization. The saved checkpoint contains all normalization constants, architecture choices, interpolant parameters, and training history. `training_errors_log.png` plots training and validation drift MSE on a logarithmic vertical scale.

---

## 7. Conditional SDE sampling

For a fixed standardized delay vector \(y\), the learned SDE is

\[
dG_s=b_\theta(s,G_s,y)\,ds+\rho_s\,dW_s,
\qquad
G_0\sim\mathcal N(0,I_r).
\]

Euler--Maruyama with \(s_k=k\Delta s\), \(\Delta s=1/N\), is

\[
G_{k+1}
=
G_k+b_\theta(s_k,G_k,y)\Delta s
+\rho_{s_k}\sqrt{\Delta s}\,\xi_k,
\qquad
\xi_k\overset{\mathrm{iid}}{\sim}\mathcal N(0,I_r).
\]

Each ensemble member uses an independent Gaussian source and independent Brownian increments. Chen et al. use the same Euler--Maruyama structure for their conditional forecasting SDE. fileciteturn4file2

Under the exact-population theorem and well-posedness/uniqueness assumptions for the associated Fokker--Planck equation,

\[
\mathcal L(G_1\mid Y=y)=\mathcal L(U\mid Y=y)
\]

for \(\mathcal L(Y)\)-almost every \(y\). In computation, neural approximation, finite data, optimization error, and SDE discretization produce an approximate kernel.

A numerical convergence study should compare output statistics for, e.g., \(N=100,200,400,800\) SDE steps. Euler--Maruyama has strong order \(1/2\) and weak order \(1\) under standard global regularity assumptions; those assumptions are not guaranteed for an unconstrained neural drift, so the step study is essential.

---

## 8. Dataset contents

`modal_reconstruction_dataset.npz` stores:

- `observations`: shape \((T,2N_s)\);
- `pod_coefficients`: shape \((T,r)\), projected using the training POD basis;
- `snapshot_times`: LBM times;
- `sensors_yx`: integer sensor coordinates;
- `train_Y`, `val_Y`, `test_Y`: delay vectors of dimension \(2N_sm\);
- `train_U`, `val_U`, `test_U`: corresponding coefficient targets;
- anchor indices for all three splits;
- training POD mean, modes, eigenvalues, and energy fractions;
- the solid mask.

The chronological blocks are formed before windows are constructed. No delay vector crosses a train/validation/test boundary, and no two retained windows within one split share a raw sample.

---

## 9. Running the example

Install:

```bash
python -m pip install numpy matplotlib torch
```

A full default run:

```bash
python cylinder_pod_si_example.py \
  --output-dir cylinder_si_output \
  --tau-lattice 100 \
  --embedding-dim 4 \
  --n-sensors 12 \
  --n-modes 6
```

Reuse an already generated trajectory:

```bash
python cylinder_pod_si_example.py \
  --output-dir cylinder_si_output \
  --reuse-flow
```

A quick smoke test:

```bash
python cylinder_pod_si_example.py \
  --output-dir quick_si \
  --nx 120 --ny 48 \
  --cylinder-radius 5 --cylinder-x 24 \
  --steps 2500 --burn-in 1000 \
  --snapshot-stride 10 \
  --tau-lattice 20 \
  --embedding-dim 3 \
  --n-sensors 6 --n-modes 4 \
  --epochs 20 --patience 10 \
  --ensemble-size 20 --sde-steps 20
```

The quick run is only a software test; it is not a scientifically adequate wake dataset.

---

## 10. Required validation before scientific interpretation

1. **Flow stationarity:** compare coefficient means/variances across post-burn-in subblocks.
2. **Wake resolution:** verify several shedding periods and estimate the dominant frequency without aliasing.
3. **POD stability:** compare modes/eigenvalues under longer trajectories and different training blocks.
4. **Dependence:** estimate integrated autocorrelation times and increase `gap_snapshots` accordingly.
5. **Sensor observability:** vary sensor layouts and quantify conditional uncertainty.
6. **SDE convergence:** repeat inference at increasing artificial-time resolution.
7. **Calibration:** evaluate empirical coverage of coefficient intervals over many test conditions.
8. **Distributional quality:** compare marginal/joint coefficient histograms, energy \( \sum_k a_k^2 \), and multivariate proper scores.
9. **CFD validation:** perform mesh/domain/boundary refinement before making quantitative physical claims.

The distinction between exact finite-dimensional POD algebra and approximate CFD dynamics is central. fileciteturn4file17

---

## 11. References

1. Y. Chen, M. Goldstein, M. Hua, M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, *Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes*, ICML/PMLR 235, 2024.
2. M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*, JMLR 26, 2025.
3. C. D. Young and M. D. Graham, *Deep learning delay coordinate dynamics for chaotic attractors from partial observable data*, Physical Review E 107, 034215, 2023.
4. L. Sirovich, *Turbulence and the Dynamics of Coherent Structures. Part I*, Quarterly of Applied Mathematics 45, 1987.
5. P. E. Kloeden and E. Platen, *Numerical Solution of Stochastic Differential Equations*, Springer, 1992.
6. T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer, 2017.
