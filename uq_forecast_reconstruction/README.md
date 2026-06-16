# Conditional stochastic interpolants for delay-coordinate UQ

This code implements uncertainty quantification for time-delay embedding models when the conditioning variable and the prediction target do **not** live in the same vector space.

You have a dynamical system

```math
dx = \phi(x)\,dt + \sigma\,dW_t
```

and only observe a partial state `x_p(t)`. For a delay spacing `tau` and number of delays `m`, define

```math
Z_t = [x_p(t), x_p(t-\tau), \ldots, x_p(t-(m-1)\tau)] \in \mathbb R^{m d_p}.
```

The two targets are:

1. **Forecasting**: `Y_t = x_p(t + tau)` or the residual `Y_t = x_p(t+tau)-x_p(t)`.
2. **Reconstruction**: `Y_t = x(t)`.

The goal is not just to learn a point estimate, but to sample from

```math
p(Y_t \mid Z_t=z).
```

This is the conditional distribution that captures uncertainty from stochastic forcing, unresolved variables, measurement noise, and non-invertibility caused by partial observations.

---

## Why the dimension mismatch is not a problem

A direct stochastic interpolant between `Z` and `Y` is not well-defined when

```math
\dim Z = m d_p \neq d_y = \dim Y.
```

The mathematically legitimate fix used here is to work on the **augmented product space**

```math
(Z,Y) \in \mathbb R^{m d_p} \times \mathbb R^{d_y}.
```

For each data pair `(z,y)`, draw a base sample

```math
Y_0 \sim q_0(\cdot), \qquad Y_0 \in \mathbb R^{d_y},
```

usually a standard Gaussian after target standardization. Then define the source and target joint distributions

```math
\mu_0(dz,dy_0) = p_Z(dz) q_0(dy_0),
\qquad
\mu_1(dz,dy) = p_Z(dz) p(dy \mid z).
```

Both distributions live in the same augmented space. The `Z` coordinate is identical at the two endpoints and is kept fixed. Only the `Y` coordinate is transported.

So the stochastic interpolant is

```math
I_s = \alpha_s Y_0 + \beta_s Y + \rho_s W_s,
\qquad s \in [0,1],
```

with `Z` passed as a conditioning variable to the neural drift. This defines a conditional generative model for `p(Y|Z=z)`.

The default schedule in the code is

```math
\alpha_s = 1-s,
\qquad
\beta_s = s^2,
\qquad
\rho_s = \varepsilon(1-s).
```

The choice `beta'(0)=0` is useful for stability near the source endpoint.

---

## Training objective

For a minibatch of empirical pairs `(z,y)`, sample

```math
Y_0 \sim q_0, \qquad s \sim \mathrm{Unif}(0,1), \qquad \xi \sim N(0,I).
```

Since `W_s` has law `sqrt(s) xi`, form

```math
I_s = \alpha_s Y_0 + \beta_s Y + \rho_s \sqrt{s}\,\xi,
```

and the regression target

```math
R_s = \dot\alpha_s Y_0 + \dot\beta_s Y + \dot\rho_s \sqrt{s}\,\xi.
```

The neural network `b_theta(s, I_s, z)` is trained by square loss

```math
\mathbb E\,\|b_\theta(s,I_s,Z)-R_s\|^2.
```

With an expressive enough model and enough data, the minimizer is the conditional expectation

```math
b_s(y,z) = \mathbb E[R_s \mid I_s=y, Z=z].
```

This is the drift whose SDE transports the base law in target space to the conditional target law.

---

## Sampling

Given a new delay embedding `z`, sample `Y_0 ~ q_0` and integrate the learned SDE

```math
dY_s = b_\theta(s,Y_s,z)\,ds + \rho_s\,dB_s,
\qquad Y_{s=0}=Y_0.
```

The output `Y_1` is one probabilistic forecast or reconstruction sample. Repeating this many times gives an ensemble, from which you can estimate a mean, standard deviation, credible intervals, calibration metrics, etc.

---

## Files

- `delay_si.py`
  - `make_delay_pairs`: creates `(Z,Y)` pairs for forecasting or reconstruction.
  - `ConditionalStochasticInterpolant`: PyTorch model for the conditional stochastic interpolant.
  - `train_interpolant`: training loop with standardization, AdamW, and gradient clipping.
  - `sample_numpy`: ensemble sampler in original physical coordinates.
  - `ensemble_summary`: mean, standard deviation, and 90% interval.

- `visualization.py`
  - Reusable plotting utilities for training losses, forecast/reconstruction time series, single-case ensemble histograms, and phase portraits.

- `example_lorenz.py`
  - Generates a noisy Lorenz trajectory using Euler-Maruyama.
  - Uses the first Lorenz coordinate as the partial observation.
  - Trains one model for `p(x_p(t+tau) | Z_t)`.
  - Trains another model for `p(x(t) | Z_t)`.
  - Prints ensemble summaries and saves plots by calling `visualization.py`.

- `example_vanderpol.py`
  - Generates a noisy Van der Pol trajectory.
  - Observes only position and reconstructs both position and velocity.
  - Trains forecast and reconstruction models and saves the same standard plots.

- `requirements.txt`
  - Minimal dependencies.

---

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run quick smoke tests:

```bash
python example_lorenz.py --quick
python example_vanderpol.py --quick
```

The examples save figures to `plots_lorenz/` and `plots_vanderpol/` by default. You can change the directory with `--plot_dir`.

Run a longer Lorenz training job:

```bash
python example_lorenz.py --train_steps 10000 --noise_level 2.0 --m 3 --delay_steps 10 --horizon_steps 10
```

Here `delay_steps` and `horizon_steps` are measured in data sample steps, not physical time. If your data sampling interval is `dt`, then the physical delay is `tau = delay_steps * dt`.

---

## How to use with your own trajectories

Suppose `x_full` has shape `(T, d_x)` and the first component is observed. Then:

```python
from delay_si import make_delay_pairs, train_interpolant

# Forecast target: x_p(t+tau)
forecast_data = make_delay_pairs(
    x_full,
    partial_indices=[0],
    delay_steps=10,
    m=3,
    horizon_steps=10,
    task="forecast",
    predict_increment=False,
)
forecast_model = train_interpolant(forecast_data.z, forecast_data.y)

# Reconstruction target: full x(t)
recon_data = make_delay_pairs(
    x_full,
    partial_indices=[0],
    delay_steps=10,
    m=3,
    task="reconstruction",
)
recon_model = train_interpolant(recon_data.z, recon_data.y)

# Ensemble samples for new delay embeddings z_new, shape (B, m*d_p)
y_samples = forecast_model.sample_numpy(z_new, n_samples=256, n_steps=100)
x_samples = recon_model.sample_numpy(z_new, n_samples=256, n_steps=100)
```

If you train the forecast model on increments using `predict_increment=True`, the samples are increments. Add the current partial observation back to obtain the forecasted partial state:

```python
increment_samples = forecast_model.sample_numpy(z_new, n_samples=256)
forecast_samples = current_xp[:, None, :] + increment_samples
```

---

## Recommended workflow

1. Generate or collect noisy trajectories.
2. Choose candidate `(m, tau)` values using false nearest neighbors, mutual information, validation likelihood-style metrics, or ensemble forecast calibration.
3. Build `(Z,Y)` pairs for each candidate.
4. Train separate stochastic interpolant models for forecasting and reconstruction.
5. Evaluate:
   - deterministic error of ensemble mean,
   - ensemble spread versus empirical error,
   - interval coverage,
   - long-time rollout statistics,
   - calibration as the dynamical noise level changes.
6. Use the deterministic models as baselines. At high noise, the deterministic model is expected to learn something like a conditional mean, while the stochastic interpolant should learn a conditional distribution.

---

## Mathematical checks

- The interpolant is **not** defined between `Z` and `Y` directly. That would be illegal when their dimensions differ.
- The source and target are both measures on `(Z,Y)` space.
- The source and target share the same `Z` marginal.
- The model does not evolve `Z`; it evolves only `Y` conditioned on fixed `Z`.
- Forecasting and reconstruction can have different target dimensions, so they should usually use separate models.
- The diffusion in the sampling SDE acts in the target space only, which is exactly where uncertainty is being represented.

---

## Extensions

- Use a deterministic forecast/reconstruction network as a warm-start base:

  ```math
  Y_0 = h_\theta(Z) + \sigma_0 \eta.
  ```

  This can make the generative task easier because the stochastic interpolant learns residual uncertainty around a good deterministic estimate.

- Replace the MLP with a CNN/UNet/operator network when `Y` is a spatial field.
- Add conditioning on the physical forecast horizon if you want one model for several horizons.
- Add likelihood-free calibration diagnostics: rank histograms, coverage curves, CRPS, and energy distance.


---

## Visualization outputs

Both example scripts call `visualization.py` and save:

- `forecast_training_loss.png`
- `reconstruction_training_loss.png`
- `forecast_timeseries.png`
- `forecast_single_case_hist.png`
- `reconstruction_timeseries.png`
- `reconstruction_phase_portrait.png`
- `reconstruction_single_case_hist.png`

For example:

```bash
python example_vanderpol.py --quick --plot_dir plots_vanderpol
```
