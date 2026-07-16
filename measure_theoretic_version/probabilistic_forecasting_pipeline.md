# Practical pipeline for conditional sampling from delay coordinates with a Gaussian-source stochastic interpolant

## 1. Goal and relation to the formal theorem

Let

\[
Y=Y_t^{\tau,m}\in\mathbb R^q,\qquad q=mp,
\]

be the delay-coordinate conditioning variable and let

\[
U\in\mathbb R^d
\]

be one of the following targets:

\[
U=P(X_{t+\tau}),\qquad U=X_t,\qquad
U=(a_1(t),\ldots,a_r(t)).
\]

The statistical target is the regular conditional probability kernel

\[
\kappa(y,A)=\mathbb P(U\in A\mid Y=y),
\qquad A\in\mathcal B(\mathbb R^d).
\]

The purpose of this document is to turn the population theorem in
`time_delay_stochastic_interpolant_framework.md` into an implementable procedure. The pipeline uses a **Gaussian source in the target space** and a conditional drift network. This is preferable to interpolating directly from the delay vector because generally

\[
Y\in\mathbb R^{mp},\qquad U\in\mathbb R^d,
\]

with \(mp\ne d\).

Chen et al. learn the drift of a conditional stochastic interpolant by a simulation-free square loss, approximate the integral over artificial time by random time samples, and solve the learned SDE repeatedly to obtain a forecast ensemble. Their empirical loss and training/sampling algorithms appear in equations (14)--(16) and Algorithms 1--2 of their paper [1]. The construction below retains those ingredients while replacing the point source by a standard Gaussian source in \(\mathbb R^d\).

---

## 2. Data assumptions and construction of supervised pairs

### 2.1 Observed trajectories

Suppose trajectory \(\ell\) is sampled at physical times

\[
t_j=j\Delta t,\qquad j=0,\ldots,T_\ell.
\]

Write

\[
X_j^{(\ell)}=X_{t_j}^{(\ell)},\qquad
x_{p,j}^{(\ell)}=P(X_j^{(\ell)}).
\]

Choose an integer delay stride \(h\ge 1\) and set

\[
\tau=h\Delta t.
\]

For every admissible index \(j\ge (m-1)h\), define

\[
Y_j^{(\ell)}=
\big(x_{p,j}^{(\ell)},x_{p,j-h}^{(\ell)},\ldots,
 x_{p,j-(m-1)h}^{(\ell)}\big).
\]

The paired target depends on the task.

#### One-step partial forecasting

For \(j+h\le T_\ell\), set

\[
U_j^{(\ell)}=x_{p,j+h}^{(\ell)}.
\]

#### Full-state reconstruction

Set

\[
U_j^{(\ell)}=X_j^{(\ell)}.
\]

This requires paired full-state data during training.

#### Modal reconstruction

First estimate the training-set mean \(\bar X\) and modes
\(\phi_1,\ldots,\phi_r\) using **training trajectories only**, freeze them, and set

\[
U_j^{(\ell)}=
\left(
\langle X_j^{(\ell)}-\bar X,\phi_1\rangle,
\ldots,
\langle X_j^{(\ell)}-\bar X,\phi_r\rangle
\right).
\]

The resulting dataset is

\[
\mathcal D=\{(Y_i,U_i)\}_{i=1}^N.
\]

### 2.2 Stationarity and time dependence

Pooling all time indices estimates one common law \(\mathcal L(Y,U)\). This is justified when the process is stationary, or approximately stationary after burn-in. For nonstationary systems, augment the conditioning variable with physical time, parameters, forcing, or regime information:

\[
C_i=(Y_i,t_i,\lambda_i,\text{forcing}_i,\ldots),
\]

and replace \(Y_i\) by \(C_i\) everywhere below.

### 2.3 Dependence and leakage

Pairs from one trajectory overlap strongly. They are not i.i.d. This does not invalidate stochastic-gradient training, but it invalidates naive uncertainty estimates based on an i.i.d. sample assumption. The train/validation/test split should therefore be done by entire trajectories or by long contiguous time blocks separated by a gap of at least \((m-1)\tau\) plus the forecast horizon. Randomly splitting overlapping windows leaks nearly identical histories across splits.

### 2.4 Delay choice

For deterministic low-dimensional systems, mutual information and false-nearest-neighbor methods are common empirical choices for \(\tau\) and \(m\), as used by Young and Graham [3]. For stochastic systems these deterministic embedding diagnostics do not prove sufficiency. Here \((\tau,m)\) should be selected by held-out conditional forecast/reconstruction performance and calibration, not by invoking Takens' theorem as an exact guarantee.

---

## 3. Preprocessing

Compute target and conditioning normalization statistics from the training set only:

\[
\widetilde Y=D_Y^{-1}(Y-\mu_Y),\qquad
\widetilde U=D_U^{-1}(U-\mu_U),
\]

where \(D_Y,D_U\) are diagonal standard-deviation matrices or whitening factors. The generative model is trained in normalized target coordinates. A generated normalized sample \(\widetilde U\) is mapped back by

\[
U=\mu_U+D_U\widetilde U.
\]

For fields, channel-wise scaling is usually preferable to one global scalar. Near-zero-variance channels must be removed or regularized.

---

## 4. Gaussian-source Brownian stochastic interpolant

Let

\[
Z\sim N(0,I_d)
\]

be independent of \((Y,U)\), and let \(B=(B_s)_{s\in[0,1]}\) be a \(d\)-dimensional Wiener process independent of \((Y,U,Z)\). Choose

\[
\alpha,\beta,\rho\in C^1([0,1])
\]

satisfying

\[
\alpha_0=1,\quad \beta_0=0,\quad
\alpha_1=0,\quad \beta_1=1,\quad \rho_1=0.
\]

Define

\[
\boxed{
I_s=\alpha_s Z+\beta_s U+\rho_s B_s.
}
\]

Then

\[
I_0=Z\sim N(0,I_d),\qquad I_1=U.
\]

The conditioning variable \(Y\) is not interpolated; it is supplied to the drift network as a fixed covariate. Applying Itô's product rule,

\[
\boxed{
dI_s=R_s\,ds+\rho_s\,dB_s,
\qquad
R_s=\dot\alpha_s Z+\dot\beta_s U+\dot\rho_s B_s.
}
\]

For each fixed \(s\),

\[
B_s\overset d=\sqrt{s}\,\Xi,
\qquad \Xi\sim N(0,I_d),
\]

so samples of \((I_s,R_s)\) can be generated without simulating an entire Brownian path:

\[
\boxed{
\begin{aligned}
I_s&=\alpha_s Z+\beta_s U+\rho_s\sqrt{s}\,\Xi,\\
R_s&=\dot\alpha_s Z+\dot\beta_s U+\dot\rho_s\sqrt{s}\,\Xi.
\end{aligned}}
\]

Here \(Z\) and \(\Xi\) are independent standard Gaussian vectors. The identity is only a fixed-time marginal identity; \(\sqrt{s}\Xi\) must not be reused as a Brownian path during SDE integration.

### 4.1 A concrete regular schedule

A simple choice is

\[
\alpha_s=1-s,\qquad \beta_s=s,
\qquad \rho_s=\sigma_I(1-s),
\]

where \(\sigma_I\ge0\). Then

\[
I_s=(1-s)Z+sU+\sigma_I(1-s)B_s,
\]

and

\[
R_s=-Z+U-\sigma_I B_s.
\]

This schedule is mathematically valid, but \(\rho_0=\sigma_I\) means the generative SDE is diffusive immediately after \(s=0\). Since \(B_0=0\), the source remains exactly Gaussian. A bridge-shaped alternative is

\[
\rho_s=\sigma_I s(1-s),
\]

which vanishes at both endpoints. Its derivative is

\[
\dot\rho_s=\sigma_I(1-2s).
\]

Both schedules agree with the theorem. The best choice is an empirical modeling decision. The schedule should be fixed before training because it changes the regression targets.

---

## 5. Population drift objective

For measurable

\[
b:[0,1]\times\mathbb R^d\times\mathbb R^q\to\mathbb R^d,
\]

define

\[
\boxed{
\mathcal J(b)=
\int_0^1
\mathbb E\left[
\|b_s(I_s,Y)-R_s\|^2
\right]ds.
}
\]

Under the square-integrability assumptions in the companion theorem, the unique minimizer in

\[
L^2\big(ds\otimes\mathcal L(I_s,Y)\big)
\]

is

\[
\boxed{
b_s^*(I_s,Y)=\mathbb E[R_s\mid I_s,Y].
}
\]

The conditional SDE

\[
\boxed{
dG_s=b_s^*(G_s,y)\,ds+\rho_s\,d\widetilde B_s,
\qquad G_0\sim N(0,I_d),
}
\]

has, under the stated well-posedness/uniqueness assumptions,

\[
\mathcal L(G_s\mid Y=y)=\mathcal L(I_s\mid Y=y)
\]

and therefore

\[
\boxed{
\mathcal L(G_1\mid Y=y)=\kappa(y,\cdot)
}
\]

for \(\mathcal L(Y)\)-almost every \(y\).

This is the direct Gaussian-source counterpart of the conditional drift-regression construction in Chen et al. [1] and of the general stochastic-interpolant quadratic objectives in Albergo, Boffi, and Vanden-Eijnden [2].

---

## 6. Monte Carlo approximation of the drift objective

Let \(b_\theta\) be a neural network with inputs

\[
(s,x,y)\in[0,1]\times\mathbb R^d\times\mathbb R^q
\]

and output in \(\mathbb R^d\). For a minibatch \(\mathcal B\) of size \(M\), independently draw

\[
S_i\sim\operatorname{Unif}(0,1),\qquad
Z_i,\Xi_i\overset{\mathrm{iid}}\sim N(0,I_d).
\]

Construct

\[
\begin{aligned}
I_i
&=\alpha_{S_i}Z_i+\beta_{S_i}U_i
 +\rho_{S_i}\sqrt{S_i}\,\Xi_i,\\
R_i
&=\dot\alpha_{S_i}Z_i+\dot\beta_{S_i}U_i
 +\dot\rho_{S_i}\sqrt{S_i}\,\Xi_i.
\end{aligned}
\]

The unbiased one-sample Monte Carlo estimator of the time integral is

\[
\boxed{
\widehat{\mathcal J}_{\mathcal B}(\theta)
=\frac1M\sum_{i\in\mathcal B}
\left\|
 b_\theta(S_i,I_i,Y_i)-R_i
\right\|^2.
}
\]

Indeed, because \(S_i\sim\operatorname{Unif}(0,1)\),

\[
\mathbb E_S[f(S)]=\int_0^1 f(s)\,ds.
\]

This is the same random-time approximation used by Chen et al. for their empirical loss [1].

### 6.1 Importance sampling in artificial time

If \(S\) is sampled from a density \(w(s)>0\) rather than uniformly, use

\[
\widehat{\mathcal J}_{\mathcal B}^{(w)}(\theta)
=\frac1M\sum_i
\frac{
\|b_\theta(S_i,I_i,Y_i)-R_i\|^2
}{w(S_i)}.
\]

Without the factor \(1/w(S_i)\), the optimized population objective changes. Endpoint-focused time sampling can reduce error where integration is numerically difficult, but the weights should be clipped only with the understanding that clipping introduces bias.

### 6.2 Equivalent half-square objective

One may multiply the loss by \(1/2\) without changing its minimizer:

\[
\frac{1}{2M}\sum_i\|b_\theta-R_i\|^2.
\]

### 6.3 Antithetic Gaussian sampling

For each \((S_i,Y_i,U_i)\), one may evaluate the loss at

\[
(Z_i,\Xi_i),\quad (-Z_i,-\Xi_i)
\]

and average the two losses. This preserves the expectation and can reduce Monte Carlo variance. Antithetic sampling is discussed as a practical stabilization device in the general stochastic-interpolant paper [2].

---

## 7. Drift-network parameterization

The network must represent

\[
b_\theta(s,x,y).
\]

A minimal implementation concatenates embeddings of \(s\), \(x\), and \(y\), then uses an MLP. For spatial fields, a U-Net or operator network can process \(x\), while an encoder processes the delay history \(y\); cross-attention, channel concatenation, or feature-wise affine modulation can provide conditioning. Chen et al. use U-Net-based models and condition on previous observations by concatenation in their high-dimensional experiments [1].

The network should be constrained or regularized enough that the numerical SDE remains stable. The exact theorem assumes a well-posed drift. A generic unconstrained neural network does not automatically satisfy global Lipschitz and linear-growth conditions. Practical options include spectral normalization, bounded residual blocks, gradient penalties, state clipping as a diagnostic rather than a theoretical fix, and monitoring drift norms on validation trajectories.

---

## 8. Algorithm 1: create the paired dataset

```text
Algorithm: BUILD-PAIRS
Input:
    trajectories {X_j^(ell)} or synchronized full/partial data
    observation map P
    delay stride h, embedding dimension m
    task in {one-step forecast, full reconstruction, modal reconstruction}
    fixed training-only POD basis if modal reconstruction is used

Output:
    paired dataset D = {(Y_i, U_i)}

1. For every trajectory ell:
2.     Compute x_p,j^(ell) = P(X_j^(ell)).
3.     For j = (m-1)h, ..., T_ell:
4.         Form Y_j^(ell) = [x_p,j, x_p,j-h, ..., x_p,j-(m-1)h].
5.         If one-step forecasting and j+h <= T_ell:
6.             Set U_j^(ell) = x_p,j+h.
7.         Else if full reconstruction:
8.             Set U_j^(ell) = X_j^(ell).
9.         Else if modal reconstruction:
10.            Set U_j^(ell) = modal coefficients of X_j^(ell).
11.        Store (Y_j^(ell), U_j^(ell)).
12. Split by trajectory or separated contiguous blocks.
13. Fit all normalization statistics using the training split only.
14. Normalize Y and U.
15. Return training, validation, and test pairs.
```

---

## 9. Algorithm 2: train the conditional drift

```text
Algorithm: TRAIN-GAUSSIAN-SOURCE-DRIFT
Input:
    training pairs D_train = {(Y_i, U_i)}
    schedules alpha(s), beta(s), rho(s) and their derivatives
    drift network b_theta(s, x, y)
    batch size M
    optimizer
    number of gradient steps N_g

Output:
    trained drift b_theta

1. Initialize theta.
2. For k = 1, ..., N_g:
3.     Draw a minibatch {(Y_i, U_i)}_{i=1}^M.
4.     Draw S_i iid ~ Uniform(0,1).
5.     Draw Z_i iid ~ N(0, I_d).
6.     Draw Xi_i iid ~ N(0, I_d), independently of Z_i.
7.     For each i, compute
           I_i = alpha(S_i) Z_i + beta(S_i) U_i
                 + rho(S_i) sqrt(S_i) Xi_i,
           R_i = alpha_dot(S_i) Z_i + beta_dot(S_i) U_i
                 + rho_dot(S_i) sqrt(S_i) Xi_i.
8.     Compute
           L = (1/M) sum_i ||b_theta(S_i, I_i, Y_i) - R_i||^2.
9.     Differentiate L with respect to theta.
10.    Apply the optimizer update.
11.    Optionally update an exponential moving average of theta.
12.    Periodically evaluate validation loss and sample-based calibration metrics.
13. Stop according to a prespecified rule or validation criterion.
14. Return the selected parameters, preferably the EMA parameters if validated.
```

The loss is simulation-free: no SDE is solved during training. This is one of the main computational advantages emphasized by Chen et al. [1] and by the stochastic-interpolant framework [2].

---

## 10. Algorithm 3: Euler--Maruyama conditional sampler

Given a new physical delay vector \(y\), normalize it with training statistics. Choose a grid

\[
0=s_0<s_1<\cdots<s_N=1,
\qquad \Delta s_n=s_{n+1}-s_n.
\]

Draw a Gaussian source

\[
G_0\sim N(0,I_d).
\]

Then apply Euler--Maruyama:

\[
\boxed{
G_{n+1}=G_n+b_\theta(s_n,G_n,y)\Delta s_n
+\rho_{s_n}\sqrt{\Delta s_n}\,\xi_n,
\qquad
\xi_n\overset{\mathrm{iid}}\sim N(0,I_d).
}
\]

```text
Algorithm: SAMPLE-CONDITIONAL-EM
Input:
    normalized condition y
    trained drift b_theta
    diffusion schedule rho(s)
    grid 0 = s_0 < ... < s_N = 1
    number of ensemble members L

Output:
    samples {U_hat^(ell)}_{ell=1}^L

1. For ell = 1, ..., L:
2.     Draw G_0^(ell) ~ N(0, I_d).
3.     For n = 0, ..., N-1:
4.         Draw xi_n^(ell) ~ N(0, I_d).
5.         Set Delta_s = s_{n+1} - s_n.
6.         Set
               G_{n+1}^(ell)
               = G_n^(ell)
               + b_theta(s_n, G_n^(ell), y) Delta_s
               + rho(s_n) sqrt(Delta_s) xi_n^(ell).
7.     Set U_tilde_hat^(ell) = G_N^(ell).
8.     Undo target normalization:
           U_hat^(ell) = mu_U + D_U U_tilde_hat^(ell).
9. Return the ensemble.
```

For fixed \(y\), the Gaussian source and Brownian increments must be independently redrawn for each ensemble member. Reusing the same source and increments would not produce an independent conditional ensemble.

Euler--Maruyama is the direct analogue of the sampling algorithm used by Chen et al. [1]. For globally Lipschitz drift and diffusion, it has strong order \(1/2\) and weak order \(1\) under standard regularity assumptions [4,5]. The learned neural drift may not satisfy these assumptions globally, so step-size convergence must be checked empirically.

---

## 11. Algorithm 4: stochastic Heun sampler

The general stochastic-interpolant paper uses Heun-type integration for SDE sampling in its numerical work [2]. For additive, time-dependent noise, a predictor-corrector form is

\[
\begin{aligned}
\widetilde G_{n+1}
&=G_n+b_\theta(s_n,G_n,y)\Delta s_n
+\rho_{s_n}\Delta W_n,\\
G_{n+1}
&=G_n+rac12\left[
 b_\theta(s_n,G_n,y)
+b_\theta(s_{n+1},\widetilde G_{n+1},y)
\right]\Delta s_n
+\rho_{s_n}\Delta W_n,
\end{aligned}
\]

where

\[
\Delta W_n=\sqrt{\Delta s_n}\,\xi_n.
\]

Because \(\rho\) is time-dependent, a trapezoidal treatment of the stochastic coefficient may also be used,

\[
\frac12(\rho_{s_n}+\rho_{s_{n+1}})\Delta W_n,
\]

but that is a distinct discretization and should be convergence-tested. The same Brownian increment must be used in predictor and corrector.

---

## 12. Numerical endpoint handling

The Gaussian-source construction avoids Chen et al.'s point-mass initialization and therefore does not require their special first Euler step designed to bypass a singular score correction at \(s=0\). Nevertheless:

1. The schedule and its derivatives must remain finite on \([0,1]\).
2. The drift network should be trained on times arbitrarily near both endpoints.
3. If numerical instability appears at \(s=1\), integrate only to \(1-\varepsilon\) and assess the resulting endpoint bias; do not silently treat this as exact sampling.
4. A nonuniform grid with smaller steps near difficult endpoints may reduce discretization error.
5. Always perform a step-halving study: compare samples using \(N,2N,4N\) steps through moments, energy spectra, rank histograms, or distributional distances.

---

## 13. Optional post-training diffusion adjustment

Chen et al. show that, in their point-source interpolant, the diffusion coefficient can be changed after training if the drift is corrected using the score of the intermediate conditional density [1]. In general, if \(p_s(x\mid y)\) denotes the density of \(I_s\mid Y=y\), then replacing \(\rho_s\) by another scalar diffusion \(g_s\) while preserving the same marginals requires

\[
\boxed{
b_s^{g}(x,y)
=b_s(x,y)+\frac12\big(g_s^2-\rho_s^2\big)
\nabla_x\log p_s(x\mid y).
}
\]

This follows by matching the two Fokker--Planck equations. It is valid only where the conditional density and score exist with sufficient regularity.

For the present Gaussian-source pipeline, the safest theorem-consistent default is

\[
g_s=\rho_s,
\]

which requires only the drift network. A tunable-diffusion implementation requires either:

- a separately learned conditional score network; or
- a rigorously derived algebraic score identity for the chosen interpolant.

One must not change the diffusion coefficient while keeping the same drift: that generally changes the intermediate and terminal laws.

---

## 14. Conditional ensemble generation for each task

### 14.1 One-step partial forecast

Given observed history

\[
y=[x_p(t),x_p(t-\tau),\ldots,x_p(t-(m-1)\tau)],
\]

run Algorithm 3 or 4 repeatedly. The outputs approximate draws from

\[
\mathcal L(x_p(t+\tau)\mid Y_t=y).
\]

### 14.2 Full-state reconstruction

The same sampler outputs approximations to

\[
\mathcal L(X_t\mid Y_t=y).
\]

Any hard physical constraints should be built into the representation or architecture when possible. Post hoc projection onto a constraint set changes the generated distribution unless the target law is already supported there and the projection is identity on that support.

### 14.3 Modal reconstruction

The terminal samples are coefficient vectors

\[
\widehat a^{(\ell)}=(\widehat a_1^{(\ell)},\ldots,\widehat a_r^{(\ell)}).
\]

A corresponding truncated field is

\[
\widehat X_r^{(\ell)}
=\bar X+\sum_{j=1}^r\widehat a_j^{(\ell)}\phi_j.
\]

This samples the conditional law of the retained coefficients, not the unresolved POD residual. Full-field uncertainty requires either generating the residual as part of the target or explicitly modeling its conditional distribution.

---

## 15. Autoregressive multi-step forecasting

For one-step partial forecasting, define the delay-shift operator

\[
\mathsf S(y,u^+)
=
[u^+,x_p(t),x_p(t-\tau),\ldots,x_p(t-(m-2)\tau)].
\]

Starting from \(Y^{(0)}=y\), generate

\[
\widehat U_{k+1}\sim \widehat\kappa_\theta(Y^{(k)},\cdot),
\qquad
Y^{(k+1)}=\mathsf S(Y^{(k)},\widehat U_{k+1}).
\]

Chen et al. use this autoregressive reuse of the same one-step conditional model [1]. It requires no retraining. However, exact one-step conditional sampling does **not** by itself imply the correct joint path law unless the delay state is sufficient for the transition. For coherent path uncertainty, a path-valued target

\[
U_t=(x_p(t+\tau),\ldots,x_p(t+H\tau))
\]

is often preferable.

---

## 16. Validation and uncertainty-quantification diagnostics

Training loss alone does not establish calibrated uncertainty. On held-out trajectories, evaluate at least the following.

### 16.1 Marginal and conditional calibration

For scalar components or scalar observables \(f(U)\):

- empirical coverage of nominal intervals;
- probability integral transform histograms when a smooth CDF estimate is available;
- rank histograms from ensembles;
- continuous ranked probability score;
- energy score for multivariate targets.

### 16.2 Sharpness

Report interval width or ensemble spread together with coverage. Wide intervals can be calibrated but uninformative.

### 16.3 Distributional structure

Compare moments, correlations, spectra, conserved quantities, modal-energy distributions, and physically meaningful summary statistics. Chen et al. evaluate conditional distributions and field statistics rather than relying only on pointwise error [1].

### 16.4 Numerical convergence

Repeat sampling with smaller SDE steps. Separate:

- statistical error from finite ensemble size;
- model error from drift approximation;
- numerical error from SDE discretization.

### 16.5 Baselines

Compare against deterministic conditional-mean regression, Gaussian heteroscedastic regression, conditional diffusion/flow baselines, persistence for forecasting, and classical reconstruction methods. The stochastic model should be judged on proper scoring rules and calibration, not only RMSE.

---

## 17. Complete end-to-end pipeline

```text
1. Generate or collect synchronized trajectories.
2. Remove physical transients or include regime/time as conditioning.
3. Choose P, tau, m, and the target U.
4. For POD targets, fit the basis on training trajectories only.
5. Construct delay-target pairs with exact temporal alignment.
6. Split by trajectory or separated blocks to prevent leakage.
7. Fit and freeze training-only normalization.
8. Choose alpha, beta, rho and record their derivatives.
9. Train b_theta with random artificial times and Gaussian latent variables.
10. Select the checkpoint using validation loss plus calibration diagnostics.
11. For a new delay vector y, normalize y.
12. Draw G_0 ~ N(0,I_d) and solve the learned SDE.
13. Repeat with independent source/noise to form an ensemble.
14. Undo target normalization.
15. For modal targets, reconstruct fields using the frozen basis.
16. Verify step-size convergence and ensemble-size convergence.
17. Report calibration, sharpness, proper scores, and physical statistics.
18. For multi-step forecasts, either roll out autoregressively with caveats
    or train a path-valued conditional target.
```

---

## 18. Agreement with the formal theorem

The implementation matches the formal theorem after replacing the deterministic anchor source by a random Gaussian source:

- The source \(Z\) and target \(U\) lie in the same space \(\mathbb R^d\).
- The conditioning variable \(Y\) enters only through the conditional drift.
- The training target \(R_s\) is exactly the finite-variation coefficient in the Itô decomposition of \(I_s\).
- The empirical loss is an unbiased Monte Carlo approximation of the population objective.
- The sampler uses the same diffusion coefficient \(\rho_s\) as the interpolant.
- At the population minimizer and under Fokker--Planck/martingale-problem uniqueness, the terminal conditional law is \(\kappa(y,\cdot)\).

The exact theorem does not cover finite network capacity, finite data, optimizer error, distribution shift, or time discretization. Those are approximation layers and must be assessed separately.

---

## 19. References

[1] Y. Chen, M. Goldstein, M. Hua, M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, “Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes,” *Proceedings of Machine Learning Research* **235**, 6728–6756 (2024). In particular, see their conditional drift objective, empirical loss, and Algorithms 1--2.

[2] M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions,” *Journal of Machine Learning Research* **26**(209), 1–80 (2025). See the quadratic objectives, learning algorithms, antithetic sampling discussion, and SDE samplers.

[3] C. D. Young and M. D. Graham, “Deep learning delay coordinate dynamics for chaotic attractors from partial observable data,” *Physical Review E* **107**, 034215 (2023). See their construction of multivariate delay vectors and empirical selection of delay parameters.

[4] P. E. Kloeden and E. Platen, *Numerical Solution of Stochastic Differential Equations*, Springer, 1992.

[5] D. J. Higham, “An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations,” *SIAM Review* **43**(3), 525–546 (2001).
