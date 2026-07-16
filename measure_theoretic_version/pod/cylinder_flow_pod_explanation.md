# Flow Past a Circular Cylinder and Proper Orthogonal Decomposition

## 1. Scope and mathematical status of the example

This document formulates the two-dimensional incompressible flow past a circular cylinder, derives the proper orthogonal decomposition (POD) of velocity snapshots, and explains the accompanying Python program.

There are two distinct mathematical objects:

1. **Continuum model:** the incompressible Navier–Stokes initial-boundary-value problem on a channel with a circular obstacle.
2. **Computed dataset:** a finite-dimensional D2Q9 BGK lattice-Boltzmann approximation of that continuum problem.

The code is a compact, reproducible demonstration of vortex shedding and snapshot POD. It is **not** claimed to be a benchmark-grade direct numerical simulation. Quantitative claims about drag, lift, pressure drop, or Strouhal number require mesh/time-step refinement, a more carefully controlled open-boundary treatment, and comparison with a standard cylinder benchmark such as Schäfer and Turek [1].

---

## 2. Geometry and governing differential equations

### 2.1 Fluid domain

Let

\[
\Omega_{\mathrm{ch}}=(0,L)\times(0,H)\subset\mathbb{R}^2
\]

be a channel and let

\[
B=\{x\in\mathbb{R}^2:\lvert x-x_c\rvert\le R\}
\]

be a closed disk representing the cylinder. Assume \(\overline B\subset\Omega_{\mathrm{ch}}\). The fluid domain is

\[
\Omega=\Omega_{\mathrm{ch}}\setminus B.
\]

Its boundary is decomposed into inlet, outlet, channel walls, and cylinder:

\[
\partial\Omega
 =\Gamma_{\mathrm{in}}\cup\Gamma_{\mathrm{out}}
  \cup\Gamma_{\mathrm{wall}}\cup\Gamma_{\mathrm{cyl}}.
\]

### 2.2 Dimensional incompressible Navier–Stokes equations

Let \(u:\Omega\times(0,T]\to\mathbb{R}^2\) be velocity and
\(p:\Omega\times(0,T]\to\mathbb{R}\) pressure. For a Newtonian fluid of constant density \(\rho>0\) and dynamic viscosity \(\mu>0\),

\[
\rho\left(\partial_tu+(u\cdot\nabla)u\right)
  =-\nabla p+\mu\Delta u+f
  \qquad\text{in }\Omega\times(0,T],
\]

\[
\nabla\cdot u=0
  \qquad\text{in }\Omega\times(0,T].
\]

Equivalently, with kinematic viscosity \(\nu=\mu/\rho\) and kinematic pressure
\(\pi=p/\rho\),

\[
\boxed{
\partial_tu+(u\cdot\nabla)u+\nabla\pi-\nu\Delta u=f/\rho,
\qquad \nabla\cdot u=0.
}
\]

A typical set of boundary and initial conditions is

\[
u=u_{\mathrm{in}}
 \quad\text{on }\Gamma_{\mathrm{in}},
\]

\[
u=0
 \quad\text{on }\Gamma_{\mathrm{wall}}\cup\Gamma_{\mathrm{cyl}},
\]

\[
\left(-\pi I+\nu(\nabla u+\nabla u^\top)\right)n=0
 \quad\text{on }\Gamma_{\mathrm{out}},
\]

\[
u(\cdot,0)=u_0,\qquad \nabla\cdot u_0=0.
\]

The no-slip condition on \(\Gamma_{\mathrm{cyl}}\) is what makes the circular body an obstacle. In the script, the inlet is a parabolic channel profile and the outlet is approximated by a simple zero-normal-gradient extrapolation rather than the exact traction condition above.

### 2.3 Nondimensional form

Choose a reference speed \(U\) and cylinder diameter \(D=2R\). Define

\[
x^*=\frac{x}{D},\qquad
t^*=\frac{Ut}{D},\qquad
u^*=\frac{u}{U},\qquad
p^*=\frac{p}{\rho U^2}.
\]

Dropping stars gives

\[
\boxed{
\partial_tu+(u\cdot\nabla)u+\nabla p
-\frac{1}{\operatorname{Re}}\Delta u=0,
\qquad
\nabla\cdot u=0,
}
\]

where

\[
\operatorname{Re}=\frac{UD}{\nu}.
\]

The cylinder wake is a canonical setting for modal analysis and reduced-order modeling; see Noack et al. [2] and the reviews by Taira et al. [3,4].

---

## 3. Functional setting

Define

\[
H=\overline{
\left\{
v\in C^\infty(\overline\Omega;\mathbb{R}^2):
\nabla\cdot v=0,\;
v=0\text{ on }\Gamma_{\mathrm{wall}}\cup\Gamma_{\mathrm{cyl}}
\right\}
}^{\,L^2(\Omega)^2},
\]

with the treatment of nonhomogeneous inlet data understood through a lifting. The kinetic-energy inner product is

\[
\langle v,w\rangle_H
 =\int_\Omega v(x)\cdot w(x)\,dx,
\qquad
\|v\|_H^2=\int_\Omega |v(x)|^2\,dx.
\]

Multiplication by \(\rho/2\) converts \(\|v\|_H^2\) into physical kinetic energy. This constant does not change normalized POD modes.

For finite snapshots \(u^1,\ldots,u^m\in H\), define the sample mean

\[
\overline u=\frac1m\sum_{j=1}^m u^j
\]

and fluctuations

\[
u^{\prime j}=u^j-\overline u.
\]

Mean subtraction is important: the POD below optimally represents **fluctuation energy**. Without centering, the leading mode often largely represents the mean flow.

---

## 4. POD as an optimization problem

### 4.1 First POD mode

The first POD mode is a solution of

\[
\max_{\phi\in H,\ \|\phi\|_H=1}
\frac1m\sum_{j=1}^m
|\langle u^{\prime j},\phi\rangle_H|^2.
\]

Define the empirical correlation operator \(C_m:H\to H\) by

\[
C_m\phi
 =\frac1m\sum_{j=1}^m
 \langle u^{\prime j},\phi\rangle_H\,u^{\prime j}.
\]

Then

\[
\langle C_m\phi,\phi\rangle_H
 =\frac1m\sum_{j=1}^m
 |\langle u^{\prime j},\phi\rangle_H|^2.
\]

The operator \(C_m\) is linear, self-adjoint, positive semidefinite, and has rank at most \(m\). Therefore its nonzero spectrum consists of finitely many nonnegative eigenvalues

\[
\lambda_1\ge\lambda_2\ge\cdots\ge 0.
\]

By the Rayleigh–Ritz principle, any unit eigenvector associated with \(\lambda_1\) solves the first-mode maximization problem.

### 4.2 Higher POD modes

Recursively, \(\phi_k\) maximizes captured variance subject to

\[
\|\phi_k\|_H=1,
\qquad
\langle\phi_k,\phi_\ell\rangle_H=0
\quad(1\le\ell<k).
\]

Thus the POD modes may be chosen as orthonormal eigenfunctions satisfying

\[
C_m\phi_k=\lambda_k\phi_k.
\]

The coefficient of snapshot \(j\) in mode \(k\) is

\[
a_k^j=\langle u^{\prime j},\phi_k\rangle_H.
\]

The rank-\(r\) POD approximation is

\[
u_r^j
 =\overline u+\sum_{k=1}^r a_k^j\phi_k.
\]

### 4.3 Exact energy identities

Because the modes are orthonormal,

\[
\frac1m\sum_{j=1}^m\|u^{\prime j}\|_H^2
 =\operatorname{tr}(C_m)
 =\sum_{k=1}^{\operatorname{rank}(C_m)}\lambda_k.
\]

Moreover,

\[
\frac1m\sum_{j=1}^m
\left\|
u^{\prime j}-\sum_{k=1}^r a_k^j\phi_k
\right\|_H^2
=\sum_{k>r}\lambda_k.
\]

Hence the fraction of sample fluctuation energy captured by the first \(r\) modes is

\[
E_r
 =\frac{\sum_{k=1}^r\lambda_k}
        {\sum_{k}\lambda_k}.
\]

### 4.4 Optimality theorem

Let \(S_r\subset H\) range over all subspaces of dimension at most \(r\), and let
\(P_{S_r}\) be the orthogonal projector onto \(S_r\). Then

\[
\boxed{
\frac1m\sum_{j=1}^m
\|u^{\prime j}-P_{S_r}u^{\prime j}\|_H^2
\ge
\sum_{k>r}\lambda_k.
}
\]

Equality holds for

\[
S_r=\operatorname{span}\{\phi_1,\ldots,\phi_r\}.
\]

This is the finite-sample POD optimality property. In matrix form it is the Eckart–Young–Mirsky theorem applied in the norm induced by the chosen spatial inner product.

#### Proof

Choose any orthonormal basis \(\psi_1,\ldots,\psi_r\) of \(S_r\). By the Pythagorean theorem,

\[
\|u^{\prime j}-P_{S_r}u^{\prime j}\|_H^2
=\|u^{\prime j}\|_H^2
-\sum_{\ell=1}^r|\langle u^{\prime j},\psi_\ell\rangle_H|^2.
\]

Averaging gives

\[
\frac1m\sum_j\|u^{\prime j}-P_{S_r}u^{\prime j}\|_H^2
=\operatorname{tr}(C_m)
-\sum_{\ell=1}^r\langle C_m\psi_\ell,\psi_\ell\rangle_H.
\]

The Ky Fan maximum principle gives

\[
\sum_{\ell=1}^r\langle C_m\psi_\ell,\psi_\ell\rangle_H
\le\sum_{k=1}^r\lambda_k.
\]

Therefore

\[
\frac1m\sum_j\|u^{\prime j}-P_{S_r}u^{\prime j}\|_H^2
\ge\sum_k\lambda_k-\sum_{k=1}^r\lambda_k
=\sum_{k>r}\lambda_k.
\]

Taking \(\psi_\ell=\phi_\ell\) yields equality. \(\square\)

This energetic optimality, along with the method of snapshots, originates in the classical POD literature; Sirovich [5] introduced the snapshot formulation for large spatial state dimension.

---

## 5. Discrete POD used by the code

### 5.1 State vectors

Suppose the velocity is sampled at \(N_f\) fluid grid nodes. For snapshot \(j\), form

\[
q_j=
\begin{bmatrix}
u_x^j(x_1)\\
\vdots\\
u_x^j(x_{N_f})\\
u_y^j(x_1)\\
\vdots\\
u_y^j(x_{N_f})
\end{bmatrix}
\in\mathbb{R}^{2N_f}.
\]

For equal-area cells with area \(\Delta A\),

\[
\langle q,z\rangle_M=q^\top Mz,
\qquad
M=\Delta A\,I.
\]

Since \(M\) is a positive scalar multiple of the identity, its factor cancels when the modes are normalized. The code can therefore use the Euclidean inner product without changing the mode shapes. For nonuniform finite elements or finite volumes, one should instead use a nontrivial mass matrix \(M\).

### 5.2 Snapshot matrix and SVD

Let

\[
\bar q=\frac1m\sum_{j=1}^m q_j,
\qquad
X=\frac1{\sqrt m}
\begin{bmatrix}
q_1-\bar q & \cdots & q_m-\bar q
\end{bmatrix}.
\]

Compute the thin singular value decomposition

\[
X=U\Sigma V^\top.
\]

Then

\[
XX^\top=U\Sigma^2U^\top,
\]

so the columns \(U_k\) are the discrete POD modes and

\[
\lambda_k=\sigma_k^2.
\]

For the original, unscaled centered snapshot matrix,

\[
[q_1-\bar q,\ldots,q_m-\bar q]
=\sqrt m\,U\Sigma V^\top.
\]

Therefore the coefficient matrix is

\[
A=U^\top[q_1-\bar q,\ldots,q_m-\bar q]
=\sqrt m\,\Sigma V^\top.
\]

The code uses precisely these formulas.

### 5.3 Method of snapshots

When \(2N_f\gg m\), one may diagonalize the smaller matrix

\[
X^\top X=V\Sigma^2V^\top
\]

and recover

\[
U_k=\frac{X V_k}{\sigma_k}
\quad\text{for }\sigma_k>0.
\]

Calling `numpy.linalg.svd(X, full_matrices=False)` is algebraically equivalent and lets the numerical linear algebra library select an efficient implementation.

### 5.4 Numerical checks performed by the code

The script checks

\[
\|U^\top U-I\|_2,
\]

and the energy identity

\[
\|Q-\bar q\mathbf{1}^\top\|_F^2
=m\sum_k\lambda_k.
\]

It also plots the eigenvalue spectrum, cumulative energy, leading coefficients, and a rank-\(r\) reconstruction.

---

## 6. D2Q9 lattice-Boltzmann discretization

The code uses the nine lattice velocities

\[
c_0=(0,0),\quad
c_{1,\ldots,4}=(\pm1,0),(0,\pm1),
\]

\[
c_{5,\ldots,8}=(\pm1,\pm1),
\]

with weights

\[
w_0=\frac49,\qquad
w_{1,\ldots,4}=\frac19,\qquad
w_{5,\ldots,8}=\frac1{36}.
\]

For populations \(f_i(x,t)\), the BGK update is

\[
\boxed{
f_i(x+c_i,t+1)
=
f_i(x,t)-\omega
\left(f_i(x,t)-f_i^{\mathrm{eq}}(x,t)\right),
}
\]

where

\[
f_i^{\mathrm{eq}}
=w_i\rho
\left[
1+\frac{c_i\cdot u}{c_s^2}
+\frac{(c_i\cdot u)^2}{2c_s^4}
-\frac{|u|^2}{2c_s^2}
\right],
\qquad c_s^2=\frac13.
\]

The macroscopic variables are

\[
\rho=\sum_i f_i,
\qquad
\rho u=\sum_i c_i f_i.
\]

For the D2Q9 BGK scheme in lattice units, the kinematic viscosity is

\[
\nu=c_s^2\left(\tau-\frac12\right),
\qquad
\tau=\omega^{-1}.
\]

The code chooses

\[
\nu=\frac{U_{\mathrm{mean}}D}{\operatorname{Re}},
\qquad
\tau=\frac12+\frac{\nu}{c_s^2}.
\]

The LBM recovers the weakly compressible Navier–Stokes equations in the low-Mach, hydrodynamic limit under the usual Chapman–Enskog scaling; a standard reference is Krüger et al. [6]. The code keeps the inlet velocity below \(0.15\) lattice units and defaults to \(0.06\), but this alone does not constitute a convergence proof for the implemented open-boundary scheme.

### Boundary approximation in the script

- Cylinder and horizontal walls: local bounce-back, approximating no slip.
- Inlet: prescribed parabolic velocity through equilibrium populations.
- Outlet: first-order copying of populations from the adjacent interior column.

The outlet treatment is intentionally minimal. Reflections or mass drift can occur. A research CFD study should replace it with a validated pressure/convective boundary condition and perform a refinement study.

---

## 7. Interpretation of cylinder-wake POD modes

For a statistically periodic von Kármán wake, leading fluctuation POD modes often appear in nearly degenerate pairs. A pair represents approximately the same oscillatory spatial structure in quadrature:

\[
u'(x,t)\approx
a_1(t)\phi_1(x)+a_2(t)\phi_2(x),
\]

with

\[
a_1(t)\approx A\cos(\omega t),
\qquad
a_2(t)\approx A\sin(\omega t).
\]

A POD mode itself is a spatial vector field, not a physical flow state. It may contain positive and negative velocity or vorticity values. The dimensional flow reconstruction is the mean plus a linear combination of modes with time-dependent coefficients.

Because POD maximizes variance rather than spectral purity, a POD mode need not have a single temporal frequency. This distinction from dynamic mode decomposition is emphasized in modal-analysis reviews [3,4].

---

## 8. Running the code

Install dependencies:

```bash
python -m pip install numpy matplotlib
```

Run the default example:

```bash
python cylinder_flow_pod.py
```

A faster, lower-resolution test is:

```bash
python cylinder_flow_pod.py \
  --nx 120 --ny 48 \
  --cylinder-radius 5 --cylinder-x 24 \
  --steps 1500 --burn-in 700 \
  --snapshot-stride 20 \
  --output-dir quick_test
```

The default run writes:

- instantaneous speed and vorticity;
- temporal-mean speed;
- leading POD-mode vorticity fields;
- POD energy spectrum and cumulative energy;
- leading temporal coefficients;
- a rank-six reconstruction and its error;
- `cylinder_pod_data.npz`;
- `diagnostics.txt`.

A longer burn-in and more snapshots generally produce a better-resolved empirical correlation operator. The snapshot interval should be small enough to sample vortex shedding without temporal aliasing.

---

## 9. What is rigorous, and what must still be validated

### Rigorous algebraic statements

For the finite snapshot matrix constructed by the code:

1. the SVD exists;
2. its left singular vectors are orthonormal POD modes for the stated discrete inner product;
3. \(\lambda_k=\sigma_k^2\);
4. the rank-\(r\) reconstruction minimizes mean squared snapshot error;
5. the discarded mean squared fluctuation energy equals \(\sum_{k>r}\lambda_k\), up to floating-point roundoff.

These are exact finite-dimensional results.

### Numerical-model limitations

The generated snapshots approximate the Navier–Stokes cylinder flow, but the script does not establish:

1. convergence under grid refinement;
2. benchmark agreement for forces or pressure;
3. a rigorous error bound between the LBM solution and a strong or weak Navier–Stokes solution;
4. independence of POD modes from numerical domain size, boundary placement, snapshot window, or sampling rate.

Those require a dedicated numerical-analysis and validation study. The distinction is essential: POD optimality is exact **for the supplied snapshots**, even when the snapshots themselves contain discretization error.

---

## 10. References

[1] M. Schäfer and S. Turek, “Benchmark Computations of Laminar Flow Around a Cylinder,” in *Flow Simulation with High-Performance Computers II*, Notes on Numerical Fluid Mechanics, vol. 52, Vieweg, 1996, pp. 547–566. DOI: [10.1007/978-3-322-89849-4_39](https://doi.org/10.1007/978-3-322-89849-4_39).

[2] B. R. Noack, K. Afanasiev, M. Morzyński, G. Tadmor, and F. Thiele, “A hierarchy of low-dimensional models for the transient and post-transient cylinder wake,” *Journal of Fluid Mechanics*, vol. 497, pp. 335–363, 2003. DOI: [10.1017/S0022112003006694](https://doi.org/10.1017/S0022112003006694).

[3] K. Taira et al., “Modal Analysis of Fluid Flows: An Overview,” *AIAA Journal*, vol. 55, no. 12, pp. 4013–4041, 2017. DOI: [10.2514/1.J056060](https://doi.org/10.2514/1.J056060). Open manuscript: [arXiv:1702.01453](https://arxiv.org/abs/1702.01453).

[4] K. Taira et al., “Modal Analysis of Fluid Flows: Applications and Outlook,” *AIAA Journal*, vol. 58, no. 3, pp. 998–1022, 2020. DOI: [10.2514/1.J058462](https://doi.org/10.2514/1.J058462). Open manuscript: [arXiv:1903.05750](https://arxiv.org/abs/1903.05750).

[5] L. Sirovich, “Turbulence and the Dynamics of Coherent Structures. Part I: Coherent Structures,” *Quarterly of Applied Mathematics*, vol. 45, no. 3, pp. 561–571, 1987. DOI: [10.1090/qam/910462](https://doi.org/10.1090/qam/910462).

[6] T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, and E. M. Viggen, *The Lattice Boltzmann Method: Principles and Practice*, Springer, 2017. DOI: [10.1007/978-3-319-44649-3](https://doi.org/10.1007/978-3-319-44649-3).

[7] P. Holmes, J. L. Lumley, G. Berkooz, and C. W. Rowley, *Turbulence, Coherent Structures, Dynamical Systems and Symmetry*, 2nd ed., Cambridge University Press, 2012. DOI: [10.1017/CBO9780511919701](https://doi.org/10.1017/CBO9780511919701).
