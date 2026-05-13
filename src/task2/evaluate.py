"""
Task 2 Sec 9.9 evaluation + phase profiling.

Three parts:
  1. Re-extract neural cps from saved probs via Top-K = PELT count (threshold=0.5
     is too strict given real-data class imbalance; Top-K restores coverage).
  2. Method-consistency F1: Routes 2/3/4 vs PELT pseudo-GT, tolerance swept over
     {+/-2, +/-3, +/-4} with sentinel-augmented `ruptures.metrics.precision_recall`.
  3. Phase profiling: 5-D per-(product, phase) profile + KMeans(n_clusters=4)
     into business-level cluster labels.
"""

import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..config.hyperparams import R1_MIN_SERIES_MONTHS, R1_MIN_SIZE, RANDOM_STATE
from ..config.paths import (
    T2_EVAL_MONTHLY_DIR, T2_EVAL_MONTHLY_META_PATH,
    T2_HYBRID_META_PATH,
    T2_PHASE_CLUSTERED_MONTHLY_PATH, T2_PHASE_PROFILES_MONTHLY_PATH,
    T2_R1_META_PATH, T2_R2_CP_PATH, T2_R2_META_PATH, T2_R2_PROBS_PATH,
    T2_R3_CP_PATH, T2_R3_META_PATH, T2_R3_PROBS_PATH,
    T2_R4_CP_PATH, T2_R4_META_PATH, T2_R4_PROBS_PATH,
    T2_ROUTE_CONSISTENCY_MONTHLY_PATH,
)
from ..config.runtime import HAS_MOMENT
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none


T2_EVAL_VERSION = 't2_eval'


# ====================================================================
# Top-K = PELT count re-extraction (replaces threshold=0.5)
# ====================================================================
def topk_cps_from_probs(probs_aligned, k, min_gap):
    """Select top-K positions from aligned probability array with a min-gap constraint.

    Returns sorted list of selected indices.
    """
    if k <= 0 or len(probs_aligned) == 0:
        return []
    order = np.argsort(probs_aligned)[::-1]
    selected = []
    for idx in order:
        if len(selected) >= k:
            break
        if probs_aligned[idx] <= 0:
            break
        if any(abs(int(idx) - s) < min_gap for s in selected):
            continue
        selected.append(int(idx))
    return sorted(selected)


def reextract_neural_cps(probs_dict, cps_path, cps_col, route_name,
                         pelt_lookup, k_cap=4, min_gap=R1_MIN_SIZE,
                         records_for_rebuild=None, verbose=True):
    """Apply Top-K=PELT-count extraction to neural probs and write back to cps CSV.

    If `cps_path` exists, updates `cps_col` column in place. If missing and
    `records_for_rebuild` is given (weekly case), builds the CSV from scratch.
    """
    if not probs_dict:
        if verbose:
            print(f"    {route_name}: probs dict empty, skipping")
        return None
    if cps_path.exists():
        df = pd.read_csv(cps_path)
        mode = "updated existing csv"
    elif records_for_rebuild is not None:
        asin_to_record = {r['asin']: r for r in records_for_rebuild}
        rows = []
        base_col = cps_col.replace('_cps', '')
        for asin in sorted(probs_dict.keys()):
            r = asin_to_record.get(asin, None)
            probs = probs_dict[asin]
            rows.append({
                'asin': asin,
                'asin_id': int(r['asin_id']) if r is not None else -1,
                'n_obs': int(len(probs)),
                base_col + '_probs_max':  float(np.max(probs))  if len(probs) > 0 else 0.0,
                base_col + '_probs_mean': float(np.mean(probs)) if len(probs) > 0 else 0.0,
            })
        df = pd.DataFrame(rows)
        mode = "built fresh from probs.pkl"
    else:
        if verbose:
            print(f"    {route_name}: change_points.csv missing and no rebuild source, skipping")
        return None

    n_old = int((df[cps_col].astype(str) != '[]').sum()) if cps_col in df.columns else 0
    new_cps_list = []
    for asin in df['asin']:
        probs = probs_dict.get(asin, np.array([]))
        n_pelt = len(pelt_lookup.get(asin, []))
        k = max(1, min(n_pelt, k_cap)) if n_pelt > 0 else 0
        cps = topk_cps_from_probs(probs, k, min_gap=min_gap)
        new_cps_list.append(json.dumps(cps))
    df[cps_col] = new_cps_list
    n_new = sum(1 for c in new_cps_list if c != '[]')
    atomic_to_csv(df, cps_path, index=False)
    if verbose:
        print(f"    {route_name} ({mode}): products with cps  {n_old} -> {n_new}  (out of {len(df)})")
    return df


# ====================================================================
# F1 evaluation vs PELT pseudo-GT
# ====================================================================
def compute_f1_vs_pelt(records, all_routes, margins=(2, 3, 4),
                       min_series_len=R1_MIN_SERIES_MONTHS):
    """Per-(route, asin, margin) F1 vs PELT pseudo-GT.

    Args:
      records: list of records
      all_routes: dict route_name -> {asin: cps_list}; must include 'pelt_l2'

    Returns DataFrame with columns route, asin, tolerance_periods, n_true_cps,
    n_pred_cps, precision, recall, f1.
    """
    from ruptures.metrics import precision_recall

    eval_rows = []
    for route_name, route_cps in all_routes.items():
        if route_name == 'pelt_l2':
            continue
        for r in records:
            true_cps_raw = list(all_routes['pelt_l2'].get(r['asin'], []))
            pred_cps_raw = list(route_cps.get(r['asin'], []))
            n_obs = len(r['series_norm'])
            if not true_cps_raw or not pred_cps_raw or n_obs < min_series_len:
                continue
            # Sentinel-augment: both lists must end with the same boundary
            # so precision_recall doesn't raise BadPartitions.
            true_cps = sorted(set(true_cps_raw) | {n_obs})
            pred_cps = sorted(set(pred_cps_raw) | {n_obs})
            for margin in margins:
                try:
                    p, rec = precision_recall(true_cps, pred_cps, margin=margin)
                    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
                except Exception:
                    p, rec, f1 = np.nan, np.nan, np.nan
                eval_rows.append({
                    'route': route_name, 'asin': r['asin'],
                    'tolerance_periods': margin,
                    'n_true_cps': len(true_cps_raw), 'n_pred_cps': len(pred_cps_raw),
                    'precision': p, 'recall': rec, 'f1': f1,
                })
    return pd.DataFrame(eval_rows)


def display_f1_summary(eval_df, granularity_label='Monthly'):
    """Print F1 / precision / recall / cps-count summary tables."""
    print(f"\n=== F1 vs PELT (mean per route) ===")
    print(f"\n{granularity_label} F1 by route x tolerance window (sensitivity analysis):")
    print(eval_df.groupby(['route', 'tolerance_periods'])['f1']
          .agg(['mean', 'median', 'count']).round(3))
    print(f"\n{granularity_label} Precision / Recall by route x tolerance (over-detect vs under-detect):")
    print(eval_df.groupby(['route', 'tolerance_periods'])[['precision', 'recall']]
          .mean().round(3))
    print("\nMean cps count by route (predicted vs PELT pseudo-GT):")
    print(eval_df.groupby('route')[['n_true_cps', 'n_pred_cps']].mean().round(2))


# ====================================================================
# Phase profiling + KMeans
# ====================================================================
def _phase_one(r, bkps):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    series = r['series_raw']
    bkps_full = [0] + sorted(list(bkps)) + [len(series)]
    rows = []
    for i in range(len(bkps_full) - 1):
        s, e = bkps_full[i], bkps_full[i+1]
        if e <= s:
            continue
        seg_idx = slice(s, e)
        rows.append({
            'asin': r['asin'], 'phase_id': i,
            'duration_months': e - s,
            'mean_deberta_sentiment_weighted': float(np.mean(series[seg_idx])),
            'mean_deberta_sentiment_norm':     float(np.mean(r['series_norm'][seg_idx])),
            'mean_rating':       float(np.mean(r['rating_mean'][seg_idx])),
            'total_reviews':     int(np.sum(r['num_reviews'][seg_idx])),
            'verified_ratio':    float(np.mean(r['verified_ratio'][seg_idx])),
            'helpful_vote_mean': float(np.mean(r['helpful_mean'][seg_idx])),
        })
    return rows


def build_phase_profiles(records, pelt_lookup, n_jobs=8, verbose=True):
    """Parallel phase profiling using PELT(L2) as final boundary set."""
    from joblib import Parallel, delayed
    if verbose:
        print(f"Phase profiling: parallel for {len(records)} products with {n_jobs} workers...")
    raw_lists = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_phase_one)(r, pelt_lookup.get(r['asin'], [])) for r in records
    )
    rows = [row for sublist in raw_lists for row in sublist]
    return pd.DataFrame(rows)


def cluster_phases(phase_df, n_clusters=4, random_state=RANDOM_STATE, verbose=True):
    """KMeans cluster the 5-D phase profiles into business-level labels.

    Returns (phase_df_with_cluster_col, centroids_df).
    """
    features = [
        'mean_deberta_sentiment_weighted', 'mean_rating',
        'total_reviews', 'verified_ratio', 'helpful_vote_mean',
    ]
    X = phase_df[features].fillna(0.0).values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X_scaled)
    phase_df = phase_df.copy()
    phase_df['cluster'] = km.labels_
    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features)
    if verbose:
        print("\n=== Phase cluster centroids (original scale) ===")
        print(centroids.round(3))
    return phase_df, centroids


# ====================================================================
# Top-level evaluation orchestrator
# ====================================================================
def task2_eval_cache_is_valid(consistency_path=None, clustered_path=None):
    consistency_path = consistency_path or T2_ROUTE_CONSISTENCY_MONTHLY_PATH
    clustered_path = clustered_path or T2_PHASE_CLUSTERED_MONTHLY_PATH
    return consistency_path.exists() and clustered_path.exists()


def run_evaluation(records, classical_df, pelt_lookup,
                   r2_probs_path=None, r3_probs_path=None, r4_probs_path=None,
                   r2_cp_path=None, r3_cp_path=None, r4_cp_path=None,
                   consistency_path=None, phase_profiles_path=None,
                   phase_clustered_path=None, meta_path=None,
                   margins=(2, 3, 4), min_series_len=R1_MIN_SERIES_MONTHS,
                   granularity_label='Monthly', force_rerun=False, n_jobs=8,
                   verbose=True):
    """Full Sec 9.9 evaluation pipeline: Top-K re-extract + F1 + phase profile + cluster.

    Args (defaults to monthly paths; override for weekly).
    Returns (eval_df, phase_df_with_cluster).
    """
    r2_probs_path = r2_probs_path or T2_R2_PROBS_PATH
    r3_probs_path = r3_probs_path or T2_R3_PROBS_PATH
    r4_probs_path = r4_probs_path or T2_R4_PROBS_PATH
    r2_cp_path = r2_cp_path or T2_R2_CP_PATH
    r3_cp_path = r3_cp_path or T2_R3_CP_PATH
    r4_cp_path = r4_cp_path or T2_R4_CP_PATH
    consistency_path = consistency_path or T2_ROUTE_CONSISTENCY_MONTHLY_PATH
    phase_profiles_path = phase_profiles_path or T2_PHASE_PROFILES_MONTHLY_PATH
    phase_clustered_path = phase_clustered_path or T2_PHASE_CLUSTERED_MONTHLY_PATH
    meta_path = meta_path or T2_EVAL_MONTHLY_META_PATH

    out_dir = consistency_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force_rerun and task2_eval_cache_is_valid(consistency_path, phase_clustered_path):
        eval_df = pd.read_csv(consistency_path)
        phase_df = pd.read_csv(phase_clustered_path)
        if verbose:
            print(f"Loaded evaluation from cache:  F1={eval_df.shape}  phases={phase_df.shape}")
            display_f1_summary(eval_df, granularity_label)
            print(f"\n=== Task 2 {granularity_label.lower()} pipeline complete. ===")
        return eval_df, phase_df

    t0 = time.time()

    if verbose:
        print("--- Re-extracting neural cps from saved probs (Top-K = PELT count) ---")
    t_re = time.time()
    with open(r2_probs_path, 'rb') as f:
        r2_probs = pickle.load(f)
    with open(r3_probs_path, 'rb') as f:
        r3_probs = pickle.load(f)
    r4_probs = {}
    if r4_probs_path.exists():
        with open(r4_probs_path, 'rb') as f:
            r4_probs = pickle.load(f)

    autocpd_df = reextract_neural_cps(r2_probs, r2_cp_path, 'autocpd_cps', 'R2 autocpd', pelt_lookup,
                                       records_for_rebuild=records, verbose=verbose)
    tst_df     = reextract_neural_cps(r3_probs, r3_cp_path, 'tst_cps',     'R3 tst',     pelt_lookup,
                                       records_for_rebuild=records, verbose=verbose)
    moment_df = None
    if r4_probs:
        moment_df = reextract_neural_cps(r4_probs, r4_cp_path, 'moment_cps', 'R4 moment', pelt_lookup,
                                          records_for_rebuild=records, verbose=verbose)
    if verbose:
        print(f"    Re-extraction elapsed: {format_elapsed(time.time() - t_re)}")

    # Build all_routes dict
    all_routes = {
        'pelt_l2': dict(zip(classical_df['asin'], classical_df['pelt_cps'].map(json.loads))),
        'clasp':   dict(zip(classical_df['asin'], classical_df['clasp_cps'].map(json.loads))),
        'autocpd': dict(zip(autocpd_df['asin'],   autocpd_df['autocpd_cps'].map(json.loads))),
        'tst':     dict(zip(tst_df['asin'],       tst_df['tst_cps'].map(json.loads))),
    }
    if moment_df is not None:
        all_routes['moment'] = dict(zip(moment_df['asin'], moment_df['moment_cps'].map(json.loads)))

    eval_df = compute_f1_vs_pelt(records, all_routes, margins=margins, min_series_len=min_series_len)
    atomic_to_csv(eval_df, consistency_path, index=False)
    if verbose:
        display_f1_summary(eval_df, granularity_label)

    # Phase profiling + KMeans
    phase_df = build_phase_profiles(records, all_routes['pelt_l2'], n_jobs=n_jobs, verbose=verbose)
    atomic_to_csv(phase_df, phase_profiles_path, index=False)
    if verbose:
        print(f"\nPhase profiles: {phase_df.shape}")
        print(phase_df.head())

    try:
        phase_df, _ = cluster_phases(phase_df, verbose=verbose)
        atomic_to_csv(phase_df, phase_clustered_path, index=False)
    except Exception as e:
        print(f"K-means clustering skipped: {e}")

    config = {
        'version': T2_EVAL_VERSION,
        'precision_recall_margins': list(margins),
        'kmeans_n_clusters': 4,
        'kmeans_random_state': RANDOM_STATE,
        'cps_extraction': 'TopK_eq_PELT_count_min_gap_R1_MIN_SIZE',
        't2_r1_meta':     read_json_or_none(T2_R1_META_PATH),
        't2_r2_meta':     read_json_or_none(T2_R2_META_PATH),
        't2_r3_meta':     read_json_or_none(T2_R3_META_PATH),
        't2_r4_meta':     read_json_or_none(T2_R4_META_PATH),
        't2_hybrid_meta': read_json_or_none(T2_HYBRID_META_PATH),
    }
    atomic_write_text(meta_path, json.dumps({'t2_eval_config': config}, indent=2))
    if verbose:
        print(f"Evaluation elapsed: {format_elapsed(time.time() - t0)}")
        print(f"\n=== Task 2 {granularity_label.lower()} pipeline complete. ===")
    return eval_df, phase_df
