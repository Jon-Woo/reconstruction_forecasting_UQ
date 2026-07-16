# Probabilistic Modal Reconstruction and Forecasting from Sparse Sensors

## 1. Purpose and scope

This document gives a measure-theoretic formulation of the following problem. A partially observed stochastic dynamical system produces a delay vector

\[
Y_t^{\tau,m}
=
\big(P(X_t),P(X_{t-\tau}),\ldots,P(X_{t-(m-1)\tau})\big),
\]

and one wishes to generate samples from the conditional law of a target random variable

\[
U_t\in\mathbb R^d
\quad\text{given}\quad
Y_t^{\tau,m}=y.
\]

The target may be a future partial observation, a current full state, or a vector of modal coefficients. The aim is **not** to learn a deterministic inverse or forecast map. It is to learn a probability kernel

\[
y\longmapsto \kappa(y,\cdot)=\mathcal L(U_t\mid Y_t^{\tau,m}=y)
\]

and then sample from that kernel by solving an artificial generative SDE.

The construction below is an adaptation of the point-mass conditional interpolant used by Chen et al. to the case in which the conditioning variable and target may have different dimensions. Chen et al. construct an interpolant from a conditioning state to a future state and characterize the drift by a quadratic regression objective; the general stochastic-interpolant framework similarly characterizes transport drifts through conditional expectations and quadratic objectives [1,2].

---

## 2. Correct mathematical model for the physical system

Let

\[
(\Omega,\mathcal F,(\mathcal F_t)_{t\ge 0},\mathbb P)
\]

be a filtered probability space satisfying the usual conditions. Let

\[
W=(W_t)_{t\ge 0}
\]

be a \(k\)-dimensional standard Wiener process. The physical state is an \(\mathbb R^n\)-valued process satisfying

\[
\boxed{
 dX_t=F(X_t,t)\,dt+\sigma(X_t,t)\,dW_t,
}
\]

where

\[
F:\mathbb R^n\times[0,\infty)\to\mathbb R^n,
\qquad
\sigma:\mathbb R^n\times[0,\infty)\to\mathbb R^{n\times k}.
\]

This corrects the expression “\(\sigma_t(X_t,t)W_t\)”: a Wiener-driven Itô SDE contains the stochastic increment \(dW_t\), not the value \(W_t\), and its diffusion coefficient is generally matrix-valued.

For the theorem below, it is enough that the relevant random variables exist and are square-integrable. For example, standard global Lipschitz and linear-growth assumptions on \(F\) and \(\Sigma\) imply existence and pathwise uniqueness of a strong solution with finite second moments on finite time intervals [3,4]. The generative theorem does not otherwise depend on the precise physical SDE.

Let

\[
P:\mathbb R^n\to\mathbb R^p
\]

be Borel measurable, and define the partial observation

\[
x_p(t)=P(X_t).
\]

Fix \(\tau>0\) and \(m\in\mathbb N\). At times \(t\ge (m-1)\tau\), define

\[
\boxed{
Y_t=Y_t^{\tau,m}
=
\big(x_p(t),x_p(t-\tau),\ldots,x_p(t-(m-1)\tau)\big)
\in\mathbb R^q,
\qquad q=mp.
}
\]

No embedding theorem is needed merely to define \(Y_t\). Delay-embedding theorems become relevant only if one claims that \(Y_t\) determines the hidden state. Such a claim is generally false for a stochastic system unless substantially stronger hypotheses are imposed.

---

## 3. Target random variables

Let \(d\) denote the dimension of the chosen target.

### 3.1 One-step partial forecast

\[
U_t=P(X_{t+\tau})\in\mathbb R^p.
\]

Then \(d=p\), and the desired law is

\[
\mathcal L(P(X_{t+\tau})\mid Y_t=y).
\]

### 3.2 Full-state reconstruction

\[
U_t=X_t\in\mathbb R^n.
\]

Then \(d=n\), and the desired law is

\[
\mathcal L(X_t\mid Y_t=y).
\]

This law need not be a point mass. Non-degeneracy is expected whenever the observation history fails to identify the hidden state or the physical system contains unresolved noise.

### 3.3 Modal reconstruction

Fix deterministic modes \(\phi_1,\ldots,\phi_r\) and a deterministic centering state \(\bar X\). In a finite-dimensional discretization, let

\[
a_j(t)=\langle X_t-\bar X,\phi_j\rangle,
\qquad
U_t=(a_1(t),\ldots,a_r(t))\in\mathbb R^r.
\]

More generally, one may define a Borel map \(A:\mathbb R^n\to\mathbb R^r\) and set \(U_t=A(X_t)\). The POD basis should be treated as fixed when defining the probability law. If the basis is estimated from data, it should be fitted on training data only, or included explicitly as a random object in the statistical model.

---

## 4. Conditional law as a probability kernel

Let

\[
Y:\Omega\to\mathbb R^q,
\qquad
U:\Omega\to\mathbb R^d
\]

stand for a generic training pair \((Y_t,U_t)\). Since Euclidean spaces with their Borel \(\sigma\)-algebras are standard Borel spaces, there exists a regular conditional distribution of \(U\) given \(Y\): a Markov kernel

\[
\kappa:\mathbb R^q\times\mathcal B(\mathbb R^d)\to[0,1]
\]

such that

1. for each \(y\), \(A\mapsto\kappa(y,A)\) is a probability measure;
2. for each Borel \(A\subseteq\mathbb R^d\), \(y\mapsto\kappa(y,A)\) is Borel measurable;
3. for Borel \(A\subseteq\mathbb R^d\) and \(C\subseteq\mathbb R^q\),

\[
\mathbb P(U\in A,Y\in C)
=
\int_C \kappa(y,A)\,\mu_Y(dy),
\qquad
\mu_Y=\mathcal L(Y).
\]

We write

\[
\kappa(y,\cdot)=\mathcal L(U\mid Y=y).
\]

This notation is intrinsically meaningful only for \(\mu_Y\)-almost every \(y\), because two versions of the conditional kernel may differ on a \(\mu_Y\)-null set [5,6]. A density \(p(u\mid y)\) is not required.

---

## 5. Conditional stochastic interpolant

The physical time variable is denoted by \(t\). The artificial generative time is denoted by

\[
s\in[0,1].
\]

Let

\[
a:\mathbb R^q\to\mathbb R^d
\]

be a Borel anchor map. Typical choices are \(a(y)=0\), a linear reconstruction, or a deterministic neural prediction. The anchor is required because the delay vector has dimension \(q=mp\), while the target has dimension \(d\), so the expression \((1-s)y+sU\) is generally not defined.

Let \(B=(B_s)_{0\le s\le 1}\) be a \(d\)-dimensional Wiener process independent of \((Y,U)\). Let

\[
\theta,\beta,\rho\in C^1([0,1])
\]

satisfy

\[
\theta(0)=1,
\quad
\beta(0)=0,
\quad
\theta(1)=0,
\quad
\beta(1)=1,
\quad
\rho(1)=0.
\]

Since \(B_0=0\), no condition on \(\rho(0)\) is needed for the initial endpoint, although \(\rho(0)=0\) is often convenient. Define

\[
\boxed{
I_s=\theta_s a(Y)+\beta_s U+\rho_s B_s.
}
\]

Then

\[
I_0=a(Y),
\qquad
I_1=U.
\]

For \(s\in[0,1]\), define the finite-variation coefficient

\[
\boxed{
R_s=\dot\theta_s a(Y)+\dot\beta_s U+\dot\rho_s B_s.
}
\]

Itô's product rule gives

\[
\boxed{
 dI_s=R_s\,ds+\rho_s\,dB_s.
}
\]

For training at a single time \(s\), one may replace \(B_s\) in distribution by \(\sqrt{s}\,Z\), where \(Z\sim N(0,I_d)\) is independent of \((Y,U)\). This is a marginal-distribution identity only; \((\sqrt{s}Z)_{s\ge0}\) is not a Wiener process when the same \(Z\) is reused at every \(s\).

---

## 6. Population drift objective

For a measurable vector field

\[
h:[0,1]\times\mathbb R^d\times\mathbb R^q\to\mathbb R^d,
\]

write \(h_s(x,y)=h(s,x,y)\). Define the population risk

\[
\boxed{
\mathcal J(h)
=
\int_0^1
\mathbb E\!
\left[
\left\|h_s(I_s,Y)-R_s\right\|^2
\right]ds,
}
\]

on the Hilbert space of equivalence classes satisfying

\[
\int_0^1\mathbb E\|h_s(I_s,Y)\|^2ds<\infty.
\]

The Bayes drift is any jointly measurable version of

\[
\boxed{
b_s(I_s,Y)=\mathbb E[R_s\mid I_s,Y].
}
\]

Equivalently,

\[
b_s(x,y)=\mathbb E[R_s\mid I_s=x,Y=y]
\]

for \(ds\otimes\mathcal L(I_s,Y)\)-almost every \((s,x,y)\).

The equality is an equality of \(L^2\)-equivalence classes; pointwise uniqueness is neither true nor needed.

---

## 7. Main theorem

### Theorem 1 (conditional delay-coordinate stochastic-interpolant theorem)

Let \(Y:\Omega\to\mathbb R^q\) and \(U:\Omega\to\mathbb R^d\) satisfy

\[
\mathbb E\|a(Y)\|^2+\mathbb E\|U\|^2<\infty.
\]

Let \(B\) be a \(d\)-dimensional Wiener process independent of \((Y,U)\), and define \(I_s\) and \(R_s\) as above. Assume

\[
\int_0^1\mathbb E\|R_s\|^2ds<\infty.
\]

Let \(b\) be a jointly measurable version of

\[
b_s(I_s,Y)=\mathbb E[R_s\mid I_s,Y].
\]

Then:

#### (a) Unique minimizer of the regression objective

For every admissible \(h\),

\[
\boxed{
\mathcal J(h)
=
\mathcal J(b)
+
\int_0^1
\mathbb E
\left[
\|h_s(I_s,Y)-b_s(I_s,Y)\|^2
\right]ds.
}
\]

Consequently, \(b\) is the unique minimizer of \(\mathcal J\) in

\[
L^2\big(ds\otimes\mathcal L(I_s,Y);\mathbb R^d\big).
\]

#### (b) Conditional weak Fokker--Planck equation

There exists a Borel set \(N\subseteq\mathbb R^q\) with \(\mu_Y(N)=0\) such that, for every \(y\notin N\), the conditional laws

\[
\mu_s^y=\mathcal L(I_s\mid Y=y)
\]

satisfy, for all \(\varphi\in C_c^2(\mathbb R^d)\),

\[
\boxed{
\begin{aligned}
\int \varphi(x)\,\mu_s^y(dx)
&=\varphi(a(y))\\
&\quad+\int_0^s\int
\left[
 b_r(x,y)\cdot\nabla\varphi(x)
 +\frac12\rho_r^2\Delta\varphi(x)
\right]
\mu_r^y(dx)\,dr.
\end{aligned}
}
\]

Thus \((\mu_s^y)_{s\in[0,1]}\) is a distributional solution of

\[
\partial_s\mu_s^y
=-\nabla\cdot(b_s(\cdot,y)\mu_s^y)
+\frac12\rho_s^2\Delta\mu_s^y,
\qquad
\mu_0^y=\delta_{a(y)}.
\]

#### (c) Exact conditional generation under a well-posedness assumption

Assume, in addition, that for \(\mu_Y\)-almost every \(y\), the SDE

\[
\boxed{
 dG_s^y=b_s(G_s^y,y)\,ds+\rho_s\,d\widetilde B_s,
 \qquad
 G_0^y=a(y),
}
\]

has a unique solution in law and that its one-time marginals are uniquely determined by the weak Fokker--Planck equation in part (b). A sufficient condition is that \(x\mapsto b_s(x,y)\) be globally Lipschitz uniformly in \(s\), with at most linear growth, and that \(s\mapsto\rho_s\) be continuous [3,4,7]. Then

\[
\boxed{
\mathcal L(G_s^y)=\mathcal L(I_s\mid Y=y)
\quad\text{for every }s\in[0,1]
}
\]

for \(\mu_Y\)-almost every \(y\). In particular,

\[
\boxed{
\mathcal L(G_1^y)
=
\mathcal L(U\mid Y=y)
=
\kappa(y,\cdot).
}
\]

Hence solving the learned generative SDE at conditioning value \(y\) produces exact samples from the desired conditional law at the population optimum.

---

## 8. Proof of Theorem 1

### Proof of part (a)

For fixed \(s\), define

\[
\mathcal G_s=\sigma(I_s,Y).
\]

By definition,

\[
b_s(I_s,Y)=\mathbb E[R_s\mid\mathcal G_s].
\]

For any admissible \(h\), write

\[
h_s(I_s,Y)-R_s
=
\big(h_s(I_s,Y)-b_s(I_s,Y)\big)
+
\big(b_s(I_s,Y)-R_s\big).
\]

After squaring,

\[
\begin{aligned}
\|h_s(I_s,Y)-R_s\|^2
&=
\|h_s(I_s,Y)-b_s(I_s,Y)\|^2\\
&\quad+
\|b_s(I_s,Y)-R_s\|^2\\
&\quad+2\big(h_s(I_s,Y)-b_s(I_s,Y)\big)
\cdot\big(b_s(I_s,Y)-R_s\big).
\end{aligned}
\]

The cross term has expectation zero. Indeed,
\(h_s(I_s,Y)-b_s(I_s,Y)\) is \(\mathcal G_s\)-measurable, so

\[
\begin{aligned}
&\mathbb E\left[
\big(h_s(I_s,Y)-b_s(I_s,Y)\big)
\cdot\big(b_s(I_s,Y)-R_s\big)
\right]\\
&=\mathbb E\left[
\big(h_s(I_s,Y)-b_s(I_s,Y)\big)
\cdot
\mathbb E[b_s(I_s,Y)-R_s\mid\mathcal G_s]
\right]\\
&=0.
\end{aligned}
\]

Therefore

\[
\mathbb E\|h_s(I_s,Y)-R_s\|^2
=
\mathbb E\|b_s(I_s,Y)-R_s\|^2
+
\mathbb E\|h_s(I_s,Y)-b_s(I_s,Y)\|^2.
\]

Integrating over \(s\) and using Tonelli's theorem gives the stated identity. The second term is nonnegative, so \(b\) minimizes \(\mathcal J\). Equality holds precisely when

\[
h_s(I_s,Y)=b_s(I_s,Y)
\]

for \(ds\otimes\mathbb P\)-almost every \((s,\omega)\), equivalently when \(h=b\) in

\[
L^2(ds\otimes\mathcal L(I_s,Y)).
\]

This proves uniqueness in the correct Hilbert-space sense. \(\square\)

### Proof of part (b)

Apply Itô's formula to \(\varphi(I_s)\), where \(\varphi\in C_c^2(\mathbb R^d)\):

\[
\begin{aligned}
d\varphi(I_s)
&=
\nabla\varphi(I_s)\cdot dI_s
+\frac12\,d\langle I\rangle_s:D^2\varphi(I_s)\\
&=
\left[
R_s\cdot\nabla\varphi(I_s)
+\frac12\rho_s^2\Delta\varphi(I_s)
\right]ds
+\rho_s\nabla\varphi(I_s)\cdot dB_s.
\end{aligned}
\]

Because \(\varphi\) has compact support and \(R\) is square-integrable, the stochastic integral is a true martingale after the usual localization argument. Integrating from \(0\) to \(s\) and conditioning on \(Y\) yields

\[
\begin{aligned}
\mathbb E[\varphi(I_s)\mid Y]
&=
\varphi(a(Y))\\
&\quad+
\int_0^s
\mathbb E\left[
R_r\cdot\nabla\varphi(I_r)
+\frac12\rho_r^2\Delta\varphi(I_r)
\mid Y
\right]dr.
\end{aligned}
\]

Use the tower property and the definition of \(b\):

\[
\begin{aligned}
\mathbb E[R_r\cdot\nabla\varphi(I_r)\mid Y]
&=
\mathbb E\left[
\mathbb E[R_r\mid I_r,Y]\cdot\nabla\varphi(I_r)
\mid Y
\right]\\
&=
\mathbb E[b_r(I_r,Y)\cdot\nabla\varphi(I_r)\mid Y].
\end{aligned}
\]

Disintegrating with respect to \(Y=y\) gives, for \(\mu_Y\)-almost every \(y\),

\[
\begin{aligned}
\int\varphi(x)\mu_s^y(dx)
&=\varphi(a(y))\\
&\quad+
\int_0^s\int
\left[
 b_r(x,y)\cdot\nabla\varphi(x)
 +\frac12\rho_r^2\Delta\varphi(x)
\right]
\mu_r^y(dx)dr.
\end{aligned}
\]

A single exceptional null set can be chosen for all \(s\) and all test functions by first taking rational \(s\), a countable dense subset of \(C_c^2\), and then using continuity and approximation. This is the desired weak equation. \(\square\)

### Proof of part (c)

Fix \(y\) outside the exceptional set in part (b) and outside the null set on which the SDE well-posedness assumption fails. Let

\[
\nu_s^y=\mathcal L(G_s^y).
\]

Applying Itô's formula to \(G^y\) shows that \((\nu_s^y)\) satisfies exactly the same weak Fokker--Planck equation as \((\mu_s^y)\), with the same initial condition \(\delta_{a(y)}\). By the assumed uniqueness of the marginal solution,

\[
\nu_s^y=\mu_s^y
\]

for all \(s\in[0,1]\). At \(s=1\),

\[
I_1=\theta_1a(Y)+\beta_1U+\rho_1B_1=U,
\]

so

\[
\mu_1^y=\mathcal L(U\mid Y=y).
\]

Therefore

\[
\mathcal L(G_1^y)=\mathcal L(U\mid Y=y).
\]

This proves exact conditional generation. \(\square\)

---

## 9. A Gaussian-base alternative

The point-anchor construction begins at the Dirac mass \(\delta_{a(y)}\). A simpler and often more regular alternative is to introduce

\[
Z\sim N(0,I_d),
\qquad Z\perp(Y,U),
\]

and define

\[
I_s=\alpha_s Z+\beta_sU+\gamma_s\varepsilon,
\]

where \(\varepsilon\sim N(0,I_d)\) is another independent latent variable, or use a Brownian-based interpolant. With

\[
\alpha_0=1,
\quad\beta_0=0,
\quad
\alpha_1=0,
\quad\beta_1=1,
\quad
\gamma_0=\gamma_1=0,
\]

one obtains

\[
\mathcal L(I_0\mid Y=y)=N(0,I_d),
\qquad
\mathcal L(I_1\mid Y=y)=\mathcal L(U\mid Y=y).
\]

This avoids a singular initial distribution and removes the need for an anchor map. It is the natural choice when the conditioning and target spaces have different dimensions. The regression theorem is the same: the optimal drift or velocity is the conditional expectation of the pathwise interpolant velocity given \((I_s,Y)\). The point-anchor version was retained in Theorem 1 because it is closest to Chen et al.'s Theorem 3.1.

---

## 10. Statistical learning formulation

From trajectory data, build training pairs

\[
(Y_i,U_i),\qquad i=1,\ldots,N.
\]

Sample independently

\[
S_i\sim\mathrm{Unif}(0,1),
\qquad
Z_i\sim N(0,I_d),
\]

and use

\[
B_{S_i}\overset d=\sqrt{S_i}Z_i.
\]

Define

\[
\begin{aligned}
I_i
&=
\theta_{S_i}a(Y_i)+\beta_{S_i}U_i
+\rho_{S_i}\sqrt{S_i}Z_i,\\
R_i
&=
\dot\theta_{S_i}a(Y_i)+\dot\beta_{S_i}U_i
+\dot\rho_{S_i}\sqrt{S_i}Z_i.
\end{aligned}
\]

For a parameterized drift \(b_\vartheta\), minimize

\[
\widehat{\mathcal J}_N(\vartheta)
=
\frac1N\sum_{i=1}^N
\left\|
 b_\vartheta(S_i,I_i,Y_i)-R_i
\right\|^2.
\]

At inference for an observed delay vector \(y\), solve

\[
dG_s=b_\vartheta(s,G_s,y)ds+\rho_s d\widetilde B_s,
\qquad G_0=a(y),
\]

independently many times. The terminal ensemble approximates \(\mathcal L(U\mid Y=y)\).

The theorem proves exactness only for the population drift. Finite data, restricted model classes, optimization error, and time-discretization error all introduce approximation error.

---

## 11. Interpretation for the three targets

The mathematical theorem is target-agnostic. Only the definition and dimension of \(U\) change.

| Task | Conditioning variable | Target | Terminal law |
|---|---|---|---|
| One-step forecast | \(Y_t^{\tau,m}\) | \(P(X_{t+\tau})\) | \(\mathcal L(P(X_{t+\tau})\mid Y_t=y)\) |
| Full reconstruction | \(Y_t^{\tau,m}\) | \(X_t\) | \(\mathcal L(X_t\mid Y_t=y)\) |
| Modal reconstruction | \(Y_t^{\tau,m}\) | \(A(X_t)\) | \(\mathcal L(A(X_t)\mid Y_t=y)\) |

The network input dimension is \(1+d+q\): generative time, current generative state, and conditioning delay vector. The network output dimension is \(d\).

---

## 12. What this theorem does and does not establish

It establishes that, under explicit regularity and well-posedness assumptions, the population minimizer of a simulation-free square loss defines an artificial SDE whose conditional terminal law is exactly the desired conditional law.

It does **not** establish that:

- the finite delay vector is a sufficient statistic for the entire observed past;
- a stochastic Takens theorem applies;
- a neural network finds the population minimizer;
- the learned SDE is calibrated out of distribution;
- recursively sampled one-step forecasts have the correct multi-time joint law;
- a finite ensemble resolves all tail probabilities.

Those are separate statistical, dynamical-systems, and numerical questions.

---

## 13. References

[1] Y. Chen, M. Goldstein, M. Hua, M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, “Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes,” *Proceedings of Machine Learning Research* **235**, 6728–6756 (2024). Official PMLR page: <https://proceedings.mlr.press/v235/chen24n.html>.

[2] M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions,” *Journal of Machine Learning Research* **26**(209), 1–80 (2025). Official JMLR page: <https://www.jmlr.org/papers/v26/23-1605.html>.

[3] B. Øksendal, *Stochastic Differential Equations: An Introduction with Applications*, 6th ed., Springer, 2003. See the standard existence-and-uniqueness theorem for globally Lipschitz coefficients with linear growth.

[4] I. Karatzas and S. E. Shreve, *Brownian Motion and Stochastic Calculus*, 2nd ed., Springer, 1991.

[5] O. Kallenberg, *Foundations of Modern Probability*, 3rd ed., Springer, 2021. See the chapters on conditional distributions and disintegration on Borel spaces.

[6] A. Klenke, *Probability Theory: A Comprehensive Course*, 3rd ed., Springer, 2020. See the theorem on existence of regular conditional distributions on Borel spaces.

[7] D. Trevisan, “Well-posedness of multidimensional diffusion processes with weakly differentiable coefficients,” *Electronic Journal of Probability* **21** (2016), paper 22. This and the superposition-principle literature explain the relationship between Fokker--Planck equations and martingale problems.

[8] C. D. Young and M. D. Graham, “Deep learning delay coordinate dynamics for chaotic attractors from partial observable data,” *Physical Review E* **107**, 034215 (2023). Official DOI page: <https://doi.org/10.1103/PhysRevE.107.034215>.

[9] F. Takens, “Detecting strange attractors in turbulence,” in *Dynamical Systems and Turbulence, Warwick 1980*, Lecture Notes in Mathematics 898, Springer, 1981, pp. 366–381.

