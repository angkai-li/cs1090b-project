"""
Synthetic AR(1) data generators for Routes 2/3 training (Sec 9.5/Sec 9.6).

Routes 2/3 train on a mix of synthetic + real weak labels:
  - 50K synthetic windows (50/50 no-change vs mean-shift), AR(1) noise phi=0.3
  - all per-product PELT weak labels (~2K monthly, ~13K weekly)

Synthetic windows have known ground-truth boundary labels -> fast classifier training
without depending on noisy real-data weak labels alone.
"""

import numpy as np


def simulate_ar1(n, phi=0.3, sigma=1.0, rng=None):
    """Generate an AR(1) noise sequence of length n with autocorrelation phi.

    x[0] = 0, x[t] = phi * x[t-1] + eps[t]   where eps[t] ~ N(0, sigma^2).
    """
    if rng is None:
        rng = np.random
    eps = rng.normal(0, sigma, n)
    x = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def generate_window(window_len, phi=0.3, sigma=1.0, rng=None):
    """Generate one labeled window for Route 2/3 training.

    50/50 split:
      - label=0: stationary AR(1), no change point
      - label=1: mean-shift at cp in [L/4, 3L/4]; magnitude uniform on [0.5, 2.0]
        with random sign.

    Returns (window: np.ndarray of shape (window_len,), label: 0 or 1).
    """
    if rng is None:
        rng = np.random
    if rng.rand() < 0.5:
        return simulate_ar1(window_len, phi=phi, sigma=sigma, rng=rng), 0

    cp = rng.randint(window_len // 4, 3 * window_len // 4)
    delta = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    pre  = simulate_ar1(cp, phi=phi, sigma=sigma, rng=rng)
    post = simulate_ar1(window_len - cp, phi=phi, sigma=sigma, rng=rng) + delta
    return np.concatenate([pre, post]).astype(np.float32), 1


def empirical_sigma_norm(records):
    """Compute empirical std of `series_norm` concatenated across all products.

    Used to calibrate the AR(1) noise level so synthetic training data matches
    real-data statistics. Routes 2/3 use this as their `sigma` parameter.
    """
    return float(np.nanstd(np.concatenate([r['series_norm'] for r in records])))
