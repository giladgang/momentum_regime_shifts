"""
hmm_model.py
============
2-state Bayesian Hidden Markov Model estimated via Gibbs sampling (MCMC).

Model
-----
Observation equation:
    z_t | s_t = k  ~  N(μ_k, Σ_k)

Transition equation:
    P(s_t = j | s_{t-1} = i) = P[i, j]

where z_t = [DD_z, VOL_z, CS_z] (standardized stress variables, pre-computed
in import_wrds.py). The regime s_t ∈ {0, 1} is latent.

Gibbs sampler blocks (per iteration)
--------------------------------------
1. FFBS  : sample full hidden state path s_{1:T} via Forward-Filtering
           Backward-Sampling, given current (μ, Σ, P).
2. NIW   : sample (μ_k, Σ_k) per regime from Normal-Inverse-Wishart
           conjugate posterior, given current state assignment.
3. Dir   : sample each row of transition matrix P from Dirichlet
           conjugate posterior, given current state path.

Outputs
-------
pi_smooth : P(s_t = panic | z_{1:T})  — uses full sample, for interpretation
pi_filter : P(s_t = panic | z_{1:t}) — uses data up to t only, for trading

The filtered probability pi_filter is what drives the momentum switching rule:
    w_fast_{t+1}  = pi_filter_t
    w_slow_{t+1}  = 1 - pi_filter_t
    r_switch_{t+1} = w_fast * r_fast_{t+1} + w_slow * r_slow_{t+1}
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# ── Section 1: Load data ──────────────────────────────────────────────────────

panel = pd.read_parquet('data/panel.parquet')

# All 5 standardized features.
# LVIX is NaN before 1990 and GDP_g is NaN for the first quarter.
# We restrict to rows where all features are non-null (~1990 onward).
features_z = ['DD_z', 'VOL_z', 'CS_z', 'LVIX_z', 'GDP_g_z']
panel = panel.dropna(subset=features_z).reset_index(drop=True)

Z = panel[features_z].values.astype(float)  # shape (T, D)
T, D = Z.shape

print(f"Loaded panel: {T} months, {D} features (restricted to non-null rows)")
print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")

# ── Section 2: Priors ─────────────────────────────────────────────────────────

K = 2  # number of regimes

# Normal-Inverse-Wishart (NIW) prior for (μ_k, Σ_k)
# After standardization, we expect regime means near 0 and covariances near I.
m_0  = np.zeros(D)                       # prior mean: centered
κ_0  = 0.01                              # weak prior on mean (almost no pull)
ν_0  = D + 2                             # minimum valid df for IW + small buffer
Ψ_0  = np.eye(D) * (ν_0 - D - 1)        # prior scale → E[Σ] = I

# Dirichlet prior for each row of transition matrix P
# (9, 1) encodes a prior belief of ~90% regime persistence per row
α_dir = np.array([9.0, 1.0])

# ── Section 3: Initialization ─────────────────────────────────────────────────

np.random.seed(42)

# Initialize state assignment: state 1 = above-median VOL_z (likely panic)
vol_idx = features_z.index('VOL_z')  # used to identify panic regime
states  = (Z[:, vol_idx] > np.median(Z[:, vol_idx])).astype(int)

# Initialize regime parameters from initial assignment
mu    = np.zeros((K, D))
Sigma = np.array([np.eye(D)] * K)
for k in range(K):
    idx = states == k
    if idx.sum() > D + 1:
        mu[k]    = Z[idx].mean(axis=0)
        Sigma[k] = np.cov(Z[idx].T) + 1e-6 * np.eye(D)

# Initialize transition matrix with strong persistence
P = np.array([[0.95, 0.05],
              [0.10, 0.90]])

# ── Section 4: Gibbs sampler helper functions ─────────────────────────────────

def log_emission(Z, mu, Sigma):
    """Compute log N(z_t; μ_k, Σ_k) for all t and k. Returns (T, K)."""
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])


def ffbs(Z, mu, Sigma, P):
    """
    Forward-Filtering Backward-Sampling.
    Returns a sampled state path s_{1:T} of shape (T,).

    Forward pass: α_t(k) = P(s_t=k, z_{1:t})
        α_t(k) ∝ emission(z_t|k) * Σ_j P[j,k] * α_{t-1}(j)
    Backward sampling: s_T ~ α_T, then s_t ~ α_t * P[·, s_{t+1}]
    """
    log_emit  = log_emission(Z, mu, Sigma)  # (T, K)
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]  # uniform initial distribution

    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k])
            )

    # Normalize each row (prevents underflow; doesn't affect sampling)
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)

    # Backward sampling
    s = np.zeros(T, dtype=int)
    s[T-1] = np.random.choice(K, p=alpha[T-1])
    for t in range(T - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]
        probs /= probs.sum()
        s[t] = np.random.choice(K, p=probs)

    return s


def sample_niw(Z, states, k):
    """
    Sample (μ_k, Σ_k) from Normal-Inverse-Wishart posterior.

    Given observations Z_k = {z_t : s_t = k}:
        κ_n = κ_0 + n_k
        m_n = (κ_0 * m_0 + n_k * x̄) / κ_n
        ν_n = ν_0 + n_k
        Ψ_n = Ψ_0 + S_k + (κ_0 n_k / κ_n) * (x̄ - m_0)(x̄ - m_0)^T
        Σ_k ~ IW(ν_n, Ψ_n)
        μ_k | Σ_k ~ N(m_n, Σ_k / κ_n)
    """
    idx = states == k
    Z_k = Z[idx]
    n_k = len(Z_k)

    if n_k < D + 2:
        return mu[k], Sigma[k]  # too few observations — keep current

    x_bar = Z_k.mean(axis=0)
    S_k   = (Z_k - x_bar).T @ (Z_k - x_bar)

    κ_n = κ_0 + n_k
    m_n = (κ_0 * m_0 + n_k * x_bar) / κ_n
    ν_n = ν_0 + n_k
    Ψ_n = (Ψ_0 + S_k +
           (κ_0 * n_k / κ_n) * np.outer(x_bar - m_0, x_bar - m_0))

    Sigma_k = invwishart.rvs(df=ν_n, scale=Ψ_n)
    mu_k    = np.random.multivariate_normal(m_n, Sigma_k / κ_n)

    return mu_k, Sigma_k


def sample_P(states):
    """
    Sample transition matrix from Dirichlet posterior.

    Count n_ij = # transitions i→j in current state path.
    Row i ~ Dirichlet(α_dir + n_i).
    """
    P_new = np.zeros((K, K))
    for i in range(K):
        counts = np.array([
            np.sum((states[:-1] == i) & (states[1:] == j))
            for j in range(K)
        ], dtype=float)
        P_new[i] = np.random.dirichlet(α_dir + counts)
    return P_new


def forward_filter(Z, mu, Sigma, P):
    """
    Compute filtered probabilities P(s_t = k | z_{1:t}) using fixed parameters.
    Returns array of shape (T, K).
    Used to compute pi_filter for trading — no future data used.
    """
    log_emit  = log_emission(Z, mu, Sigma)
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]

    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k])
            )

    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)

# ── Section 5: Gibbs sampler ──────────────────────────────────────────────────

n_iter   = 2000
n_burnin = 500
n_keep   = n_iter - n_burnin

# Storage for post-burnin draws
state_draws = np.zeros((n_keep, T), dtype=int)
mu_draws    = np.zeros((n_keep, K, D))
Sigma_draws = np.zeros((n_keep, K, D, D))
P_draws     = np.zeros((n_keep, K, K))

print(f"\nRunning Gibbs sampler: {n_iter} iterations ({n_burnin} burn-in) ...")
for m in range(n_iter):
    if m % 500 == 0:
        print(f"  Iteration {m}/{n_iter}")

    # Block 1: sample hidden state path via FFBS
    states = ffbs(Z, mu, Sigma, P)

    # Block 2: sample (μ_k, Σ_k) for each regime from NIW posterior
    for k in range(K):
        mu[k], Sigma[k] = sample_niw(Z, states, k)

    # Block 3: sample transition matrix from Dirichlet posterior
    P = sample_P(states)

    if m >= n_burnin:
        idx = m - n_burnin
        state_draws[idx] = states
        mu_draws[idx]    = mu
        Sigma_draws[idx] = Sigma
        P_draws[idx]     = P

print("Gibbs sampler complete.")

# ── Section 6: Identify panic regime ─────────────────────────────────────────
# Panic = regime with higher average VOL_z.
# This resolves label-switching: state numbering is arbitrary, but VOL_z
# ordering is not — panic periods always have higher realized volatility.

smoothed_assignment = state_draws.mean(axis=0)  # fraction of draws in state 1
vol_state1 = Z[smoothed_assignment >= 0.5, vol_idx].mean()
vol_state0 = Z[smoothed_assignment <  0.5, vol_idx].mean()
panic_state = 1 if vol_state1 > vol_state0 else 0
calm_state  = 1 - panic_state
print(f"\nPanic regime identified as state {panic_state} (higher VOL_z)")

# ── Section 7: Smoothed panic probability ────────────────────────────────────
# pi_smooth_t = fraction of post-burnin draws where s_t == panic_state
# Uses full sample (past and future) — for interpretation and plots only.

pi_smooth = (state_draws == panic_state).mean(axis=0)

# ── Section 8: Filtered panic probability ────────────────────────────────────
# pi_filter_t = P(s_t = panic | z_{1:t})
# Uses posterior mean parameters from MCMC — no future data.
# This is the trading signal: use pi_filter_t to weight month t+1.

mu_post    = mu_draws.mean(axis=0)
Sigma_post = Sigma_draws.mean(axis=0)
P_post     = P_draws.mean(axis=0)

filtered  = forward_filter(Z, mu_post, Sigma_post, P_post)
pi_filter = filtered[:, panic_state]

# ── Section 9: Attach to panel and save ──────────────────────────────────────

panel = panel.copy()
panel['pi_smooth'] = pi_smooth
panel['pi_filter'] = pi_filter

panel.to_parquet('data/panel_with_regimes.parquet', index=False)
print("Saved: data/panel_with_regimes.parquet")

# ── Section 10: Posterior summary ────────────────────────────────────────────

# Check 1: Regime means table — all 5 variables should point in opposite
# directions for calm vs panic. If they don't, the model isn't finding
# a genuine stress regime.
print("\n--- Check 1: Posterior regime means (standardized) ---")
print(f"{'Feature':<12} {'Calm mean':>12} {'Panic mean':>12}  {'Separation':>12}")
print("-" * 52)
for j, feat in enumerate(features_z):
    calm_mean  = mu_draws[:, calm_state,  j].mean()
    panic_mean = mu_draws[:, panic_state, j].mean()
    sep = panic_mean - calm_mean
    flag = " ✓" if abs(sep) > 0.3 else " ← weak"
    print(f"{feat:<12} {calm_mean:>12.3f} {panic_mean:>12.3f}  {sep:>+12.3f}{flag}")
print("\nExpected: DD negative, VOL/CS/LVIX positive in panic vs calm.")
print("Weak separation on any variable means it adds little to the model.")

print("\n--- Posterior transition matrix ---")
for i in range(K):
    label = 'Calm' if i == calm_state else 'Panic'
    stay  = P_draws[:, i, i].mean()
    print(f"  P(stay in {label}) = {stay:.3f}  "
          f"[expected duration: {1/(1-stay):.1f} months]")

# Check 2: Crisis alignment — does pi_smooth spike in known stress periods?
# The model has no knowledge of calendar dates, so alignment is pure validation.
print("\n--- Check 2: Average panic probability in known crisis periods ---")
crisis_check = [
    ('Dot-com',      '2000-03-01', '2002-10-01'),
    ('GFC',          '2007-10-01', '2009-06-01'),
    ('COVID',        '2020-02-01', '2020-05-01'),
    ('Non-crisis',   '2003-01-01', '2006-12-01'),
]
for label, start, end in crisis_check:
    mask = (panel['date'] >= start) & (panel['date'] <= end)
    if mask.any():
        avg = pi_smooth[mask].mean()
        flag = " ✓" if (label != 'Non-crisis' and avg > 0.5) or \
                       (label == 'Non-crisis' and avg < 0.3) else " ← check"
        print(f"  {label:<15} {start} → {end}:  avg π = {avg:.3f}{flag}")
print("\nExpected: crisis periods > 0.5, non-crisis < 0.3.")

# ── Section 11: Diagnostic plots ─────────────────────────────────────────────

crises = [
    ('1973-10-01', '1974-12-01', 'Oil shock'),
    ('1987-10-01', '1987-12-01', 'Black Monday'),
    ('2000-03-01', '2002-10-01', 'Dot-com'),
    ('2007-10-01', '2009-06-01', 'GFC'),
    ('2020-02-01', '2020-05-01', 'COVID'),
]

def shade_crises(ax, label=False):
    for start, end, lbl in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color='grey')
        if label:
            ax.text(pd.Timestamp(start), 0.97, lbl, fontsize=7, color='grey',
                    va='top', transform=ax.get_xaxis_transform())

dates = panel['date']

# Plot A: smoothed vs filtered panic probability
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].fill_between(dates, pi_smooth, alpha=0.6, color='crimson')
axes[0].set_ylabel('Smoothed π_panic', fontsize=9)
axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color='black', linewidth=0.5, linestyle='--')
shade_crises(axes[0], label=True)

axes[1].fill_between(dates, pi_filter, alpha=0.6, color='steelblue')
axes[1].set_ylabel('Filtered π_panic', fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].axhline(0.5, color='black', linewidth=0.5, linestyle='--')
shade_crises(axes[1])

axes[1].set_xlabel('Date')
fig.suptitle('Posterior Panic Probability (grey = known crisis)', fontsize=11)
plt.tight_layout()
fig.savefig('regime_probabilities.png', dpi=150)
plt.close(fig)

# Plot B: raw features coloured by regime (smoothed majority vote)
regime_label = (pi_smooth > 0.5).astype(int)  # 1 = panic, 0 = calm
colors = np.where(regime_label == 1, 'crimson', 'steelblue')

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, feat_z, feat_raw in zip(axes, features_z, ['DD', 'VOL', 'CS']):
    ax.scatter(dates, panel[feat_raw], c=colors, s=4, alpha=0.7)
    ax.set_ylabel(feat_z, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    shade_crises(ax)
axes[-1].set_xlabel('Date')
fig.suptitle('Features coloured by regime (red = panic, blue = calm)', fontsize=11)
plt.tight_layout()
fig.savefig('regime_features.png', dpi=150)
plt.close(fig)

print("\nPlots saved: regime_probabilities.png, regime_features.png")
