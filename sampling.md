# Pseudocode for Conditional Sampling with the Trained Drift

## 1. Setting

Assume that training has produced a selected conditional drift

\[
b_{\theta^\star}:
[0,1]\times\mathbb R^d\times\mathbb R^q
\longrightarrow
\mathbb R^d.
\]

The drift was trained in normalized coordinates. Therefore, sampling must use:

\[
\mu_Y,\qquad D_Y,
\]

for normalization of the conditioning variable, and

\[
\mu_U,\qquad D_U,
\]

for undoing the target normalization.

For a physical conditioning input

\[
y\in\mathbb R^q,
\]

define its normalized version by

\[
\widetilde y
=
D_Y^{-1}(y-\mu_Y).
\]

The trained drift is evaluated as

\[
b_{\theta^\star}(s,x,\widetilde y).
\]

The terminal SDE state is a normalized target sample. It is mapped back to physical target coordinates by

\[
\widehat U
=
\mu_U+D_U\widetilde U.
\]

---

## 2. Learned conditional SDE

Use the same diffusion schedule \(\rho(s)\) that was fixed during drift training.

For the normalized condition \(\widetilde y\), solve

\[
dG_s
=
b_{\theta^\star}(s,G_s,\widetilde y)\,ds
+
\rho(s)\,dW_s,
\qquad
G_0\sim N(0,I_d).
\]

At artificial time \(s=1\), the terminal state \(G_1\) is treated as a normalized conditional target sample:

\[
\widetilde U^{(\ell)}_{\mathrm{sample}}
=
G_1^{(\ell)}.
\]

Repeated independent solutions produce an ensemble approximating the conditional target law given \(Y=y\).

The diffusion coefficient must not be changed after training while keeping the same drift. The sampling SDE uses the same \(\rho(s)\) as the stochastic interpolant used to define the drift objective.

---

## 3. Euler--Maruyama grid

Choose an increasing artificial-time grid

\[
0=s_0<s_1<\cdots<s_N=1.
\]

For each step, define

\[
\Delta s_n=s_{n+1}-s_n.
\]

For a uniform grid,

\[
s_n=\frac{n}{N},
\qquad
\Delta s_n=\frac{1}{N}.
\]

The Euler--Maruyama update is

\[
G_{n+1}^{(\ell)}
=
G_n^{(\ell)}
+
b_{\theta^\star}
\left(
s_n,G_n^{(\ell)},\widetilde y
\right)\Delta s_n
+
\rho(s_n)\sqrt{\Delta s_n}\,
\xi_n^{(\ell)},
\]

where

\[
\xi_n^{(\ell)}
\overset{\mathrm{iid}}{\sim}
N(0,I_d).
\]

The Gaussian source \(G_0^{(\ell)}\) and all Brownian increments must be independently redrawn for every ensemble member \(\ell\).

---

# Algorithm 1: Validate sampling inputs

```text
ALGORITHM VALIDATE-SAMPLING-INPUTS

INPUT:
    physical condition y
    trained drift b_theta_star
    condition normalization statistics mu_Y and D_Y
    target normalization statistics mu_U and D_U
    target dimension d
    artificial-time grid {s_n}_{n=0}^N
    number of ensemble members L
    diffusion schedule rho

OUTPUT:
    validated sampling inputs

STEPS:

1. Verify that y is a finite vector in R^q.

2. Verify that mu_Y has dimension q.

3. Verify that D_Y is compatible with dimension q and is invertible
   on all retained conditioning components.

4. Verify that mu_U has dimension d.

5. Verify that D_U is compatible with dimension d.

6. Verify that L is a positive integer.

7. Verify that the artificial-time grid satisfies

       s_0 = 0,
       s_N = 1,
       s_{n+1} > s_n

   for every n = 0, ..., N-1.

8. Verify that rho(s_n) is finite at every grid point used by the solver.

9. Verify that b_theta_star accepts inputs

       scalar artificial time s,
       state x in R^d,
       condition y_tilde in R^q,

   and returns a vector in R^d.

10. Return the validated inputs.
```

---

# Algorithm 2: Normalize the conditioning input

```text
ALGORITHM NORMALIZE-CONDITION

INPUT:
    physical condition y
    training-set condition mean mu_Y
    training-set condition scale D_Y

OUTPUT:
    normalized condition y_tilde

STEPS:

1. Compute

       y_tilde
       = inverse(D_Y) (y - mu_Y).

2. Return y_tilde.
```

The normalization statistics are exactly the frozen training-set statistics used when the drift was trained.

---

# Algorithm 3: Generate one Euler--Maruyama sample

```text
ALGORITHM SAMPLE-ONE-CONDITIONAL-EM

INPUT:
    normalized condition y_tilde
    trained drift b_theta_star
    diffusion schedule rho
    artificial-time grid
        0 = s_0 < s_1 < ... < s_N = 1
    target dimension d

OUTPUT:
    normalized terminal sample U_tilde_sample

STEPS:

1. Draw the Gaussian source

       G_0 ~ N(0,I_d).

2. For n = 0, ..., N-1:

       2.1 Compute

               Delta_s_n
               = s_{n+1} - s_n.

       2.2 Draw

               xi_n ~ N(0,I_d).

       2.3 Evaluate the trained drift

               drift_n
               = b_theta_star(
                     s_n,
                     G_n,
                     y_tilde
                 ).

       2.4 Apply the Euler--Maruyama update

               G_{n+1}
               = G_n
                 + drift_n Delta_s_n
                 + rho(s_n) sqrt(Delta_s_n) xi_n.

3. Set

       U_tilde_sample = G_N.

4. Return U_tilde_sample.
```

Only the terminal state \(G_N\) is required for the final conditional sample. The intermediate Euler--Maruyama states do not need to be saved unless required for a separate numerical diagnostic.

---

# Algorithm 4: Generate a conditional ensemble

```text
ALGORITHM SAMPLE-CONDITIONAL-ENSEMBLE-EM

INPUT:
    physical condition y
    trained drift b_theta_star
    diffusion schedule rho
    condition normalization statistics mu_Y and D_Y
    target normalization statistics mu_U and D_U
    artificial-time grid
        0 = s_0 < s_1 < ... < s_N = 1
    number of ensemble members L
    target dimension d

OUTPUT:
    normalized terminal samples
        U_tilde_samples[1:L]

    physical terminal samples
        U_samples[1:L]

STEPS:

1. Validate all inputs using VALIDATE-SAMPLING-INPUTS.

2. Normalize the physical condition:

       y_tilde
       = NORMALIZE-CONDITION(y, mu_Y, D_Y).

3. Allocate

       U_tilde_samples as an L-by-d array,
       U_samples       as an L-by-d array.

4. For ell = 1, ..., L:

       4.1 Independently draw a new Gaussian source and new Brownian
           increments by calling

               U_tilde_samples[ell]
               = SAMPLE-ONE-CONDITIONAL-EM(
                     y_tilde,
                     b_theta_star,
                     rho,
                     artificial-time grid,
                     d
                 ).

       4.2 Undo target normalization:

               U_samples[ell]
               = mu_U
                 + D_U U_tilde_samples[ell].

5. Return U_tilde_samples and U_samples.
```

The random variables used for different values of \(\ell\) are mutually independent. In particular, neither the Gaussian source nor the Euler--Maruyama Gaussian increments are reused across ensemble members.

---

# Algorithm 5: Save the condition and final samples

Save the physical input \(y\) together with the generated terminal samples. Also save the normalized quantities and numerical metadata needed to interpret the result.

```text
ALGORITHM SAVE-CONDITIONAL-SAMPLES

INPUT:
    output file path
    physical condition y
    normalized condition y_tilde
    physical samples U_samples[1:L]
    normalized samples U_tilde_samples[1:L]
    artificial-time grid {s_n}_{n=0}^N
    diffusion-schedule specification
    checkpoint identifier for b_theta_star
    condition normalization statistics mu_Y and D_Y
    target normalization statistics mu_U and D_U
    random seed or random-generator metadata

OUTPUT:
    saved sample file

STEPS:

1. Construct a saved record containing

       input_y                 = y,
       normalized_input_y      = y_tilde,
       samples                 = U_samples,
       normalized_samples      = U_tilde_samples,
       artificial_time_grid    = {s_n}_{n=0}^N,
       ensemble_size           = L,
       target_dimension        = d,
       diffusion_schedule      = rho specification,
       drift_checkpoint        = checkpoint identifier,
       condition_mean          = mu_Y,
       condition_scale         = D_Y,
       target_mean             = mu_U,
       target_scale            = D_U,
       random_metadata         = supplied random metadata.

2. Save the record to the requested output path.

3. Return the output path.
```

A direct array-oriented format is

```text
conditional_samples.npz
```

with entries

```text
input_y
normalized_input_y
samples
normalized_samples
artificial_time_grid
mu_Y
D_Y
mu_U
D_U
```

and accompanying scalar or string metadata for the ensemble size, diffusion schedule, checkpoint, and random seed.

---

# Algorithm 6: Complete conditional sampling pipeline

```text
ALGORITHM RUN-CONDITIONAL-SAMPLING

INPUT:
    physical condition y
    trained drift b_theta_star
    diffusion schedule rho used during training
    condition normalization statistics mu_Y and D_Y
    target normalization statistics mu_U and D_U
    number of Euler--Maruyama steps N
    number of ensemble members L
    target dimension d
    output file path
    checkpoint identifier
    random seed or random-generator metadata

OUTPUT:
    saved physical conditional samples paired with y

STEPS:

1. Construct the artificial-time grid

       s_n = n / N,
       n = 0, ..., N.

2. Validate the condition, dimensions, normalization statistics,
   grid, diffusion schedule, and trained drift.

3. Compute

       y_tilde
       = inverse(D_Y) (y - mu_Y).

4. Generate the conditional ensemble using

       SAMPLE-CONDITIONAL-ENSEMBLE-EM.

5. Obtain

       U_tilde_samples[1:L]
       and
       U_samples[1:L].

6. Save the input condition and terminal samples using

       SAVE-CONDITIONAL-SAMPLES.

7. Return

       input y,
       normalized input y_tilde,
       physical samples U_samples,
       normalized samples U_tilde_samples,
       output file path.
```

---

## 4. Batched implementation of the ensemble

The ensemble loop can be evaluated in parallel without changing the algorithm.

```text
ALGORITHM SAMPLE-CONDITIONAL-ENSEMBLE-EM-BATCHED

INPUT:
    normalized condition y_tilde
    trained drift b_theta_star
    diffusion schedule rho
    grid {s_n}_{n=0}^N
    ensemble size L
    target dimension d

OUTPUT:
    normalized terminal samples G in R^{L x d}

STEPS:

1. Repeat y_tilde along the ensemble dimension:

       Y_condition
       = repeat(y_tilde, L times)
       in R^{L x q}.

2. Draw independent source rows:

       G
       = [G_0^(1); ...; G_0^(L)]
       in R^{L x d},

   where every row is independently distributed as N(0,I_d).

3. For n = 0, ..., N-1:

       3.1 Set
               Delta_s_n = s_{n+1} - s_n.

       3.2 Form an L-by-1 time array whose entries equal s_n.

       3.3 Draw an L-by-d Gaussian array Xi with independent
           N(0,1) entries.

       3.4 Evaluate all ensemble drifts:

               B
               = b_theta_star(
                     s_n repeated L times,
                     G,
                     Y_condition
                 ).

       3.5 Update all ensemble members:

               G
               = G
                 + B Delta_s_n
                 + rho(s_n) sqrt(Delta_s_n) Xi.

4. Return G as the normalized terminal sample array.
```

This batched form preserves the requirement that each ensemble member uses an independent Gaussian source and independent Gaussian increments.

---

## 5. Agreement with `probabilistic_forecasting_pipeline.md`

The pseudocode follows the source pipeline in the following ways:

1. A new physical condition \(y\) is normalized using frozen training statistics.
2. Sampling begins from an independent Gaussian source
   \(G_0\sim N(0,I_d)\).
3. The trained drift is evaluated as
   \(b_{\theta^\star}(s_n,G_n,\widetilde y)\).
4. Euler--Maruyama uses
   \(\rho(s_n)\sqrt{\Delta s_n}\xi_n\).
5. The diffusion schedule is the same schedule used to define and train the drift.
6. Independent source variables and independent Brownian increments are drawn for every ensemble member.
7. The terminal SDE state is interpreted in normalized target coordinates.
8. Every terminal state is mapped back using
   \(\widehat U=\mu_U+D_U\widetilde U\).
9. Repeated SDE solutions form a conditional ensemble.
10. Saving the original input \(y\) beside the terminal ensemble is a storage step performed after the theorem-consistent sampler and does not alter the generated samples.
