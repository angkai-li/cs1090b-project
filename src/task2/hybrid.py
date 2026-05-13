"""
Task 2 Sec 9.8 Hybrid PELT-style decoding: PELT candidates + neural-prior filter.

Two-stage decoding:
  1. PELT(L2 + CROPS-elbow) proposes candidate boundaries from `series_raw`.
  2. Filter candidates by `p_neural(t) >= exp(-1/gamma)`:
     - gamma=0.5 -> threshold ~ 0.135 -> loose (neural weakly vetoes)
     - gamma=1.0 -> threshold ~ 0.368 -> moderate
     - gamma=2.0 -> threshold ~ 0.607 -> strict (only strong neural endorsement survives)
"""

import json
import time

import numpy as np
import pandas as pd

from ..config.hyperparams import R1_MIN_SERIES_MONTHS, R1_MIN_SIZE
from ..config.paths import (
    T2_HYBRID_MONTHLY_CP_PATH, T2_HYBRID_MONTHLY_META_PATH,
    T2_R1_META_PATH, T2_R2_META_PATH, T2_R3_META_PATH, T2_R4_META_PATH,
)
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none
from .pelt import crops_elbow_pelt


T2_HYBRID_VERSION = 't2_hybrid'
GAMMA_SWEEP = [0.5, 1.0, 2.0]


def hybrid_decoding(series, neural_probs, gamma=1.0):
    """Single-product hybrid decoding at a given gamma.

    Returns sorted list of (t, score, p) tuples, where:
      t = boundary position
      score = -gamma * log(p)   (lower = more confidently endorsed)
      p = neural probability at position t
    """
    _, candidates = crops_elbow_pelt(series, model='l2', min_size=R1_MIN_SIZE)
    threshold = float(np.exp(-1.0 / max(gamma, 1e-9)))
    selected = []
    for t in candidates:
        if 0 <= t < len(neural_probs):
            p = max(float(neural_probs[t]), 1e-9)
        else:
            p = 0.5
        if p >= threshold:
            score = -gamma * np.log(p)
            selected.append((int(t), float(score), float(p)))
    return sorted(selected, key=lambda x: x[1])


def task2_hybrid_cache_is_valid(cp_path=None):
    cp_path = cp_path or T2_HYBRID_MONTHLY_CP_PATH
    return cp_path.exists()


def _hybrid_one(r, probs, min_series_len=R1_MIN_SERIES_MONTHS, gammas=GAMMA_SWEEP):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    series = r['series_raw']
    if len(series) < min_series_len:
        return []
    rows = []
    for gamma in gammas:
        scored = hybrid_decoding(series, probs, gamma=gamma)
        bkps = [t for t, _, _ in scored]
        rows.append({
            'asin': r['asin'], 'gamma': gamma,
            'hybrid_cps': json.dumps(bkps),
            'n_cps': len(bkps),
        })
    return rows


def run_hybrid_decoding(records, neural_probs_dict, n_jobs=8,
                        cp_path=None, meta_path=None,
                        min_series_len=R1_MIN_SERIES_MONTHS,
                        gammas=GAMMA_SWEEP, force_rerun=False, verbose=True):
    """Run hybrid gamma-sweep decoding across all products."""
    from joblib import Parallel, delayed

    cp_path = cp_path or T2_HYBRID_MONTHLY_CP_PATH
    meta_path = meta_path or T2_HYBRID_MONTHLY_META_PATH

    if not force_rerun and task2_hybrid_cache_is_valid(cp_path):
        hybrid_df = pd.read_csv(cp_path)
        if verbose:
            print(f"Loaded hybrid decoding from cache: {hybrid_df.shape}")
        return hybrid_df

    t0 = time.time()
    if verbose:
        print(f"Hybrid: parallel decoding {len(records)} products * {len(gammas)} gammas "
              f"with {n_jobs} workers...")
    raw_lists = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_hybrid_one)(r,
            neural_probs_dict.get(r['asin'], np.full(len(r['series_raw']), 0.5)),
            min_series_len, gammas)
        for r in records
    )
    results = [row for sublist in raw_lists for row in sublist]
    hybrid_df = pd.DataFrame(results)
    atomic_to_csv(hybrid_df, cp_path, index=False)

    config = {
        'version': T2_HYBRID_VERSION,
        'gamma_sweep': list(gammas),
        'pen_selection': 'CROPS_elbow_Kneedle_with_walk_back',
        'pen_fallback': 'BIC = 2 * sigma^2 * log(n) (Killick 2012)',
        'pelt_model': 'l2', 'pelt_min_size': R1_MIN_SIZE,
        'neural_filter': 'p_neural >= exp(-1/gamma)',
        'sigma_estimator': 'MAD_of_lag1_diffs (Donoho-Johnstone 1994)',
        'input_signal': 'series_raw',
        't2_r1_meta': read_json_or_none(T2_R1_META_PATH),
        't2_r2_meta': read_json_or_none(T2_R2_META_PATH),
        't2_r3_meta': read_json_or_none(T2_R3_META_PATH),
        't2_r4_meta': read_json_or_none(T2_R4_META_PATH),
    }
    atomic_write_text(meta_path, json.dumps({'t2_hybrid_config': config}, indent=2))
    if verbose:
        print(f"Hybrid decoding done. {len(hybrid_df)} rows.")
        print(f"Hybrid decoding elapsed: {format_elapsed(time.time() - t0)}")
    return hybrid_df
