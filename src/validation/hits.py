"""
Sec 10.1 - single-anchor hit-rate + exact Poisson-Binomial p-value.

N=5 trials is too small for normal approximation; we compute the exact PMF by
convolving non-identical Bernoullis (O(N^2) = 25 ops, trivially exact).
"""

import numpy as np
import pandas as pd

from ..utils.io import atomic_to_csv
from .events import (
    EVENT_ANCHORS, T2_EVENT_VALIDATION_DIR, TOLERANCES, TOLERANCE_PRIMARY,
    ix_to_date, months_apart, random_hit_prob,
)


T2_EVENT_HITS_PATH = T2_EVENT_VALIDATION_DIR / "event_hits_long.csv"
T2_EVENT_SUMMARY_PATH = T2_EVENT_VALIDATION_DIR / "event_hits_summary.csv"


def compute_hit_table(route_cps_dict, records_dict, anchors=EVENT_ANCHORS, tolerances=TOLERANCES):
    """One row per (asin, route, tolerance). Returns DataFrame."""
    rows = []
    for asin, info in anchors.items():
        if asin not in records_dict:
            for route_name in route_cps_dict:
                for tol in tolerances:
                    rows.append({
                        'asin': asin, 'phone': info['title'], 'route': route_name,
                        'tolerance_months': tol, 'hit': False, 'min_dist_months': None,
                        'n_cps_route': 0, 'n_obs': 0, 'reason': 'record_missing',
                    })
            continue
        event_date = info['event_date']
        n_obs = len(records_dict[asin]['series_norm'])
        for route_name, cps_per_asin in route_cps_dict.items():
            cps = cps_per_asin.get(asin, [])
            cp_dates = [ix_to_date(asin, ix, records_dict) for ix in cps]
            cp_dates = [d for d in cp_dates if d is not None]
            n_cps_route = len(cp_dates)
            if not cp_dates:
                for tol in tolerances:
                    rows.append({
                        'asin': asin, 'phone': info['title'], 'route': route_name,
                        'tolerance_months': tol, 'hit': False, 'min_dist_months': None,
                        'n_cps_route': 0, 'n_obs': n_obs, 'reason': 'no_cps',
                    })
                continue
            min_dist = min(months_apart(d, event_date) for d in cp_dates)
            for tol in tolerances:
                rows.append({
                    'asin': asin, 'phone': info['title'], 'route': route_name,
                    'tolerance_months': tol, 'hit': bool(min_dist <= tol),
                    'min_dist_months': int(min_dist), 'n_cps_route': n_cps_route,
                    'n_obs': n_obs, 'reason': 'evaluated',
                })
    return pd.DataFrame(rows)


def compute_summary(hits_df, route_names, tolerances=TOLERANCES):
    """Per-(route, tolerance) hits + expected random + exact Poisson-Binomial p."""
    rows = []
    for route_name in route_names:
        for tol in tolerances:
            sub = hits_df[
                (hits_df['route'] == route_name)
                & (hits_df['tolerance_months'] == tol)
                & (hits_df['reason'] == 'evaluated')
            ]
            n_trials = len(sub)
            n_hits = int(sub['hit'].sum())
            p_per_trial = [random_hit_prob(r['n_obs'], tol, r['n_cps_route']) for _, r in sub.iterrows()]
            exp_hits = sum(p_per_trial)
            if n_trials > 0:
                pmf = np.array([1.0])
                for p in p_per_trial:
                    pmf = np.convolve(pmf, [1.0 - p, p])
                p_value = float(pmf[n_hits:].sum()) if n_hits <= len(pmf) - 1 else 0.0
            else:
                p_value = float('nan')
            rows.append({
                'route': route_name, 'tolerance_months': tol,
                'n_hits': n_hits, 'n_trials': n_trials,
                'hit_rate': round(n_hits / n_trials, 3) if n_trials else float('nan'),
                'expected_random_hits': round(exp_hits, 2),
                'p_value_one_sided': round(p_value, 4),
            })
    return pd.DataFrame(rows)


def display_distance_pivot(hits_df, tolerance=TOLERANCE_PRIMARY):
    """Pivot of phone x route -> closest cp distance (months)."""
    dist_pivot = hits_df[
        (hits_df['tolerance_months'] == tolerance) & (hits_df['reason'] == 'evaluated')
    ].pivot_table(index='phone', columns='route', values='min_dist_months', aggfunc='first')
    print(dist_pivot)
    return dist_pivot


def display_verdict(summary_df, tolerance=TOLERANCE_PRIMARY):
    print(f"\n=== Verdict at primary tolerance +/-{tolerance} months ===")
    for _, row in summary_df[summary_df['tolerance_months'] == tolerance].iterrows():
        rate = row['hit_rate']
        if rate < 0.40:
            verdict = "NOT detected" if rate < 0.30 else "weak detection (p ~ 0.15)"
        elif rate < 0.60:
            verdict = "moderate evidence (p ~ 0.03)"
        else:
            verdict = "STRONG evidence (p <= 0.005)"
        print(f"  {row['route']:14s} {int(row['n_hits'])}/{int(row['n_trials'])} hits  "
              f"(expected random {row['expected_random_hits']:.2f}, "
              f"p = {row['p_value_one_sided']:.3f}) -> {verdict}")


def run_single_anchor_evaluation(route_cps_dict, records_dict, verbose=True):
    """Run the full Sec 10.1 single-anchor evaluation. Returns (hits_df, summary_df)."""
    T2_EVENT_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    hits_df = compute_hit_table(route_cps_dict, records_dict)
    atomic_to_csv(hits_df, T2_EVENT_HITS_PATH, index=False)
    summary_df = compute_summary(hits_df, list(route_cps_dict.keys()))
    atomic_to_csv(summary_df, T2_EVENT_SUMMARY_PATH, index=False)
    if verbose:
        print(f"Saved long hit table -> {T2_EVENT_HITS_PATH}")
        print(f"Saved per-route x tol summary -> {T2_EVENT_SUMMARY_PATH}")
        print("\n=== Closest predicted-cp distance to event (months) ===")
        display_distance_pivot(hits_df)
        print("\n=== Hit rate per method x tolerance (random baseline + p-value) ===")
        print(summary_df.pivot_table(
            index='route', columns='tolerance_months',
            values=['n_hits', 'hit_rate', 'p_value_one_sided'],
            aggfunc='first',
        ).round(3))
        display_verdict(summary_df)
    return hits_df, summary_df
