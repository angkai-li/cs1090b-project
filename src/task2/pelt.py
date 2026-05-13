"""
Task 2 Sec 9.4 Route 1 PELT: L2 cost + CROPS-elbow penalty selection.

Algorithm:
  - L2 cost model (Killick 2012: L2 is for mean-shift detection)
  - Penalty grid scaled by sigma_hat^2: c * sigma_hat^2 * log(n)
    Standard BIC = 2 * sigma^2 * log(n) corresponds to multiplier c=2.
  - CROPS (Haynes et al. 2017): scan penalty grid, pick elbow via Kneedle.
  - Kneedle (Satopaa et al. 2011): max signed deviation from anti-diagonal.
  - Walk-back: if elbow lands on zero-cps plateau, step back to last non-zero.
  - Robust sigma: Donoho-Johnstone 1994 MAD of lag-1 differences.

Why series_raw not series_norm: per-product z-scoring removes the absolute-level
information CPD relies on. arXiv 2510.04667 (2025) shows RevIN-style
normalization can degrade time-series methods by ~683% on regime-shifted data.
"""

import json
import pickle
import time

import numpy as np
import pandas as pd

from ..config.hyperparams import R1_MIN_SERIES_MONTHS, R1_MIN_SIZE, RANDOM_STATE
from ..config.paths import (
    T2_R1_CP_PATH, T2_R1_DIR, T2_R1_LOOKUP_PATH, T2_R1_META_PATH,
    T2_RECORDS_META_PATH,
)
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv, atomic_to_pickle, atomic_write_text, read_json_or_none


T2_R1_VERSION = 't2_r1'


def find_elbow_index_kneedle(x_values, y_values):
    """Elbow on a monotonically non-increasing (pen, n_cps) curve.

    Normalizes both axes to [0, 1], measures vertical deviation from the
    anti-diagonal chord (0,1)->(1,0), returns argmax(|deviation|).
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    n = len(x)
    if n < 3:
        return n - 1
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range < 1e-12 or y_range < 1e-12:
        return n - 1
    x_norm = (x - x.min()) / x_range
    y_norm = (y - y.min()) / y_range
    deviations = y_norm - (1.0 - x_norm)
    return int(np.argmax(np.abs(deviations)))


def robust_sigma_squared(series):
    """Donoho-Johnstone (1994) MAD-of-lag-1-differences robust sigma^2 estimator.

    For x_t = mu_t + eps_t with eps_t ~ N(0, sigma^2) iid within segments:
        diff_t = x_t - x_{t-1} = (mu_t - mu_{t-1}) + (eps_t - eps_{t-1})
    Within-segment: diff ~ N(0, 2 sigma^2). MAD of diffs is robust to the
    sparse large diffs caused by actual change points.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 3:
        return 1.0
    diffs = np.diff(series)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    sigma_mad = mad * 1.4826 / np.sqrt(2.0)
    if sigma_mad < 1e-4:
        sigma_eps = np.std(diffs) / np.sqrt(2.0)
    else:
        sigma_eps = sigma_mad
    return max(sigma_eps ** 2, 1e-6)


def crops_elbow_pelt(series, model='l2', min_size=4, pen_grid_multipliers=None):
    """Run PELT at multiple sigma^2-scaled penalties, return cps at the elbow.

    Penalty grid: [m * sigma_hat^2 * log(n) for m in multipliers].
    Standard Killick 2012 BIC corresponds to m = 2.
    Falls back to BIC (m=2) when penalty grid yields degenerate curve.
    """
    import ruptures as rpt

    n = len(series)
    if n < 4:
        return 0.0, []

    log_n = np.log(max(n, 2))
    sigma2 = robust_sigma_squared(series)
    scale = sigma2 * log_n

    if pen_grid_multipliers is None:
        pen_grid_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    pen_grid = [m * scale for m in pen_grid_multipliers]

    algo = rpt.Pelt(model=model, min_size=min_size).fit(np.asarray(series, dtype=float))
    n_cps_grid, cps_grid = [], []
    for pen in pen_grid:
        try:
            cps = algo.predict(pen=float(pen))[:-1]
        except Exception:
            cps = []
        n_cps_grid.append(len(cps))
        cps_grid.append([int(c) for c in cps])

    if max(n_cps_grid) == 0:
        return 2.0 * scale, []
    if len(set(n_cps_grid)) == 1:
        return 2.0 * scale, cps_grid[3]  # index 3 = BIC multiplier (2.0)

    elbow_idx = find_elbow_index_kneedle(pen_grid, n_cps_grid)
    if n_cps_grid[elbow_idx] == 0 and elbow_idx > 0:
        for i in range(elbow_idx - 1, -1, -1):
            if n_cps_grid[i] > 0:
                elbow_idx = i
                break
    return float(pen_grid[elbow_idx]), cps_grid[elbow_idx]


# ====================================================================
# Synthetic calibration (validates CROPS-elbow on data-stats-matched signals)
# ====================================================================
def simulate_planted_cps(L, n_true_cps, mean, sigma, phi=0.3,
                         min_size=R1_MIN_SIZE, rng=None):
    """Simulate a series of length L with n_true_cps planted change points.

    Each segment has a uniform mean shift in {+/-[0.15, 0.5]}. Noise is AR(1) with
    autocorrelation phi. Output clipped to [-1, +1]. Returns (series, true_cps_list).
    """
    if rng is None:
        rng = np.random
    if n_true_cps == 0 or L < (n_true_cps + 1) * min_size:
        true_cps = []
    else:
        positions = np.arange(min_size, L - min_size)
        if len(positions) >= n_true_cps:
            true_cps = sorted(rng.choice(positions, size=n_true_cps, replace=False).tolist())
        else:
            true_cps = []
    segments = [0] + list(true_cps) + [L]
    x = np.zeros(L, dtype=np.float32)
    for i in range(len(segments) - 1):
        s, e = segments[i], segments[i + 1]
        seg_mean = mean + rng.uniform(0.15, 0.5) * rng.choice([-1, 1])
        eps = rng.normal(0, sigma, e - s).astype(np.float32)
        ar = np.zeros(e - s, dtype=np.float32)
        for t in range(1, e - s):
            ar[t] = phi * ar[t - 1] + eps[t]
        x[s:e] = ar + seg_mean
    return np.clip(x, -1.0, 1.0), true_cps


def run_synthetic_calibration(records, n_trials=1000, margin=2, verbose=True):
    """Validate CROPS-elbow PELT on synthetic AR(1) signals matched to real data stats."""
    np.random.seed(RANDOM_STATE)
    real_lens = [len(r['series_raw']) for r in records]
    real_concat = np.concatenate([r['series_raw'] for r in records])
    real_mean = float(np.nanmean(real_concat))
    per_prod_sigmas = [
        np.sqrt(robust_sigma_squared(r['series_raw']))
        for r in records if len(r['series_raw']) > 2
    ]
    real_sigma = float(np.median(per_prod_sigmas))
    if verbose:
        print(f"  Within-product median sigma: {real_sigma:.4f}  "
              f"(cross-product std would be {float(np.nanstd(real_concat)):.4f})")

    match_pred = match_true = total_pred = total_true = 0
    n_detected, n_true_list, pens = [], [], []
    for _ in range(n_trials):
        L = int(np.random.choice(real_lens))
        n_true = int(np.random.choice([0, 1, 2, 3], p=[0.25, 0.40, 0.25, 0.10]))
        x, true_cps = simulate_planted_cps(L, n_true, real_mean, real_sigma)
        try:
            best_pen, detected = crops_elbow_pelt(x, model='l2', min_size=R1_MIN_SIZE)
        except Exception:
            best_pen, detected = (2 * np.log(max(L, 2)), [])
        match_pred += sum(1 for d in detected if any(abs(d - t) <= margin for t in true_cps))
        match_true += sum(1 for t in true_cps if any(abs(d - t) <= margin for d in detected))
        total_pred += len(detected)
        total_true += len(true_cps)
        n_detected.append(len(detected))
        n_true_list.append(len(true_cps))
        pens.append(best_pen)

    p = match_pred / total_pred if total_pred > 0 else 0.0
    r = match_true / total_true if total_true > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    avg_len = float(np.mean(real_lens))
    if verbose:
        print(f"  N={n_trials} synthetic trials, margin=+/-{margin} (matches Sec 9.9 eval)")
        print(f"  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
        print(f"  Mean detected/series: {np.mean(n_detected):.2f}   "
              f"Mean true/series: {np.mean(n_true_list):.2f}")
        print(f"  Mean penalty chosen: {np.mean(pens):.2f}  "
              f"(BIC=2*log(n)~{2*np.log(avg_len):.2f}, "
              f"MBIC=3*log(n)~{3*np.log(avg_len):.2f})")

    return {
        'n_trials': n_trials,
        'precision': float(p), 'recall': float(r), 'f1': float(f1),
        'mean_detected_per_series': float(np.mean(n_detected)),
        'mean_true_per_series': float(np.mean(n_true_list)),
        'mean_penalty_chosen': float(np.mean(pens)),
    }


# ====================================================================
# Per-product worker (joblib-pickleable) + main driver
# ====================================================================
def _process_one(r, min_series_len=R1_MIN_SERIES_MONTHS, has_claspy=False):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    series = r['series_raw']
    if len(series) < min_series_len:
        return None

    best_pen, pelt_cps = crops_elbow_pelt(series, model='l2', min_size=R1_MIN_SIZE)

    clasp_cps = []
    if has_claspy:
        try:
            from claspy.segmentation import BinaryClaSPSegmentation
            clasp = BinaryClaSPSegmentation()
            clasp.fit(np.asarray(series, dtype=float))
            clasp_cps = list(map(int, clasp.change_points))
        except Exception:
            clasp_cps = []

    return {
        'asin': r['asin'],
        'asin_id': r['asin_id'],
        'n_obs': len(series),
        'pelt_cps':      json.dumps(list(map(int, pelt_cps))),
        'clasp_cps':     json.dumps(clasp_cps),
        'pelt_best_pen': float(best_pen),
    }


def task2_r1_cache_is_valid():
    return T2_R1_CP_PATH.exists() and T2_R1_LOOKUP_PATH.exists()


def run_route1(records, n_jobs=8, force_rerun=False, run_calibration=True,
               run_penalty_sensitivity=True, min_series_len=R1_MIN_SERIES_MONTHS,
               cp_path=None, lookup_path=None, meta_path=None,
               out_dir=None, verbose=True):
    """Run PELT (CROPS-elbow L2) + optional ClaSP across all products.

    Args (paths default to monthly):
      cp_path:     T2_R1_CP_PATH    (or T2_R1_WEEKLY_CP_PATH for weekly)
      lookup_path: T2_R1_LOOKUP_PATH
      meta_path:   T2_R1_META_PATH
      out_dir:     T2_R1_DIR        (for the penalty_sensitivity.csv)

    Returns (classical_df, pelt_lookup, calibration_result).
    """
    from ..config.runtime import HAS_CLASPY

    cp_path = cp_path or T2_R1_CP_PATH
    lookup_path = lookup_path or T2_R1_LOOKUP_PATH
    meta_path = meta_path or T2_R1_META_PATH
    out_dir = out_dir or T2_R1_DIR

    if not force_rerun and cp_path.exists() and lookup_path.exists():
        classical_df = pd.read_csv(cp_path)
        with open(lookup_path, 'rb') as f:
            pelt_lookup = pickle.load(f)
        if verbose:
            print(f"Loaded Route 1 from cache: {len(classical_df)} products")
            print(f"Average PELT(l2) boundaries per product: "
                  f"{np.mean([len(v) for v in pelt_lookup.values()]):.2f}")
        return classical_df, pelt_lookup, None

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_result = None
    if run_calibration:
        print("\n--- CROPS-elbow PELT calibration on synthetic AR(1) with planted CPs ---")
        calibration_result = run_synthetic_calibration(records, n_trials=1000, margin=2, verbose=verbose)

    from joblib import Parallel, delayed
    print(f"\nRoute 1: parallel processing {len(records)} products with {n_jobs} workers...")
    raw_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_process_one)(r, min_series_len, HAS_CLASPY) for r in records
    )
    results = [x for x in raw_results if x is not None]

    classical_df = pd.DataFrame(results)
    atomic_to_csv(classical_df, cp_path, index=False)
    pelt_lookup = {
        row['asin']: json.loads(row['pelt_cps'])
        for _, row in classical_df.iterrows()
    }
    atomic_to_pickle(pelt_lookup, lookup_path)

    if verbose:
        avg_bkps = np.mean([len(v) for v in pelt_lookup.values()])
        print(f"Saved {len(classical_df)} products' classical boundaries to {cp_path}")
        print(f"Average PELT(l2) boundaries per product: {avg_bkps:.2f}")
        print(f"Average chosen penalty: {classical_df['pelt_best_pen'].mean():.2f}  "
              f"(min={classical_df['pelt_best_pen'].min():.2f}, "
              f"max={classical_df['pelt_best_pen'].max():.2f})")

    config = {
        'version': T2_R1_VERSION,
        'pelt_model': 'l2', 'pelt_min_size': R1_MIN_SIZE,
        'pen_selection': 'CROPS_elbow_Kneedle_with_walk_back',
        'pen_grid_formula': '[0.5, 1.0, 1.5, 2.0, 3.0, 5.0] * sigma_hat_squared * log(n)',
        'pen_fallback': 'BIC = 2 * sigma_hat_squared * log(n) (Killick 2012)',
        'sigma_estimator': 'MAD_of_lag1_diffs (Donoho-Johnstone 1994)',
        'input_signal': 'series_raw',
        't2_dataprep_meta': read_json_or_none(T2_RECORDS_META_PATH),
    }
    meta = {'t2_r1_config': config}
    if calibration_result is not None:
        meta['calibration'] = calibration_result
    atomic_write_text(meta_path, json.dumps(meta, indent=2))
    if verbose:
        print(f"Route 1 elapsed: {format_elapsed(time.time() - t0)}")

    if run_penalty_sensitivity:
        run_penalty_sensitivity_table(records, n_jobs=n_jobs,
                                       min_series_len=min_series_len,
                                       out_dir=out_dir, verbose=verbose)

    return classical_df, pelt_lookup, calibration_result


def _sensitivity_one(r, min_series_len=R1_MIN_SERIES_MONTHS):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    s = r['series_raw']
    if len(s) < min_series_len:
        return None

    import ruptures as rpt
    n = len(s)
    sigma2 = robust_sigma_squared(s)
    log_n = np.log(n)
    scale = sigma2 * log_n
    algo = rpt.Pelt(model='l2', min_size=R1_MIN_SIZE).fit(np.asarray(s, dtype=float))

    rows = []
    for label, mult in [
        ('0.5*sigma2*log n',           0.5),
        ('sigma2*log n',               1.0),
        ('1.5*sigma2*log n',           1.5),
        ('2*sigma2*log n (BIC)',       2.0),
        ('3*sigma2*log n (MBIC)',      3.0),
    ]:
        pen = mult * scale
        try:
            cps = algo.predict(pen=float(pen))[:-1]
        except Exception:
            cps = []
        rows.append({'asin': r['asin'], 'penalty': label,
                     'pen_value': float(pen), 'sigma2': float(sigma2),
                     'n_cps': len(cps)})
    elbow_pen, elbow_cps = crops_elbow_pelt(s, model='l2', min_size=R1_MIN_SIZE)
    rows.append({'asin': r['asin'], 'penalty': 'CROPS-elbow',
                 'pen_value': float(elbow_pen), 'sigma2': float(sigma2),
                 'n_cps': len(elbow_cps)})
    return rows


def run_penalty_sensitivity_table(records, n_jobs=8, min_series_len=R1_MIN_SERIES_MONTHS,
                                  out_dir=None, verbose=True):
    """Compare CROPS-elbow penalty vs AIC/BIC/MBIC on real products."""
    from joblib import Parallel, delayed
    out_dir = out_dir or T2_R1_DIR
    if verbose:
        print("\n--- Penalty sensitivity table on real products ---")
    t0 = time.time()
    sens_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_sensitivity_one)(r, min_series_len) for r in records
    )
    sens_rows = [row for sub in sens_results if sub for row in sub]
    sens_df = pd.DataFrame(sens_rows)
    pen_order = ['0.5*sigma2*log n', 'sigma2*log n', '1.5*sigma2*log n',
                 '2*sigma2*log n (BIC)', '3*sigma2*log n (MBIC)', 'CROPS-elbow']
    sens_summary = (
        sens_df.groupby('penalty')
        .agg(mean_pen=('pen_value', 'mean'),
             mean_cps=('n_cps', 'mean'),
             median_cps=('n_cps', 'median'),
             pct_products_with_cps=('n_cps', lambda x: (x > 0).mean() * 100))
        .reindex(pen_order)
        .round(3)
    )
    if verbose:
        print(sens_summary)
    sens_csv = out_dir / 'penalty_sensitivity.csv'
    atomic_to_csv(sens_df, sens_csv, index=False)
    if verbose:
        print(f"Saved penalty sensitivity table to {sens_csv}")
        print(f"Penalty sensitivity elapsed: {format_elapsed(time.time() - t0)}")
    return sens_summary
