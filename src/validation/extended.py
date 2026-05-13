"""
Sec 10.4 - extensions: Moto G2 diagnostic + extended tolerance + edge filter +
N=5 bootstrap empirical p-values.
"""

import pickle
import time

import numpy as np
import pandas as pd

from ..config.hyperparams import RANDOM_STATE
from ..config.paths import (
    T2_R2_PROBS_PATH, T2_R3_PROBS_PATH, T2_R4_PROBS_PATH,
)
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv
from .events import (
    EVENT_ANCHORS, T2_EVENT_VALIDATION_DIR, TOLERANCES_EXTENDED,
    ix_to_date, months_apart,
)


T2_EVENT_HITS_EXTENDED_PATH = T2_EVENT_VALIDATION_DIR / "event_hits_extended.csv"
T2_EVENT_SUMMARY_EXTENDED_PATH = T2_EVENT_VALIDATION_DIR / "event_hits_summary_extended.csv"
T2_EVENT_BOOTSTRAP_PATH = T2_EVENT_VALIDATION_DIR / "event_bootstrap_null.csv"
T2_EVENT_EMPIRICAL_P_PATH = T2_EVENT_VALIDATION_DIR / "event_empirical_p.csv"

MOTO_ASIN = 'B00MWI4IN8'
EDGE_THRESHOLD = 0.25
N_BOOTSTRAP_DEFAULT = 10000
N_SAMPLE_PER_ROUND = 5
FLAT_PROBS_THRESHOLD = 0.01


def diagnose_moto_g2_and_recover(route_cps_dict, verbose=True):
    """Sec 10.4 PART A+B: Moto G2 NaN diagnostic + Top-K recovery with min_gap=2.

    Returns (probs_dicts, recovered_cps_dict, flat_probs_routes).
    Side effect: updates route_cps_dict with recovered Moto G2 cps.
    """
    if verbose:
        print("="*78)
        print("PART A: Moto G2 NaN diagnostic")
        print("="*78)

    probs_paths = {
        'R2 (AutoCPD)': T2_R2_PROBS_PATH,
        'R3 (TST)':     T2_R3_PROBS_PATH,
        'R4 (MOMENT)':  T2_R4_PROBS_PATH,
    }
    probs_dicts = {}
    for name, path in probs_paths.items():
        if not path.exists():
            if verbose:
                print(f"  {name}: probs.pkl missing at {path}")
            continue
        with open(path, 'rb') as f:
            probs_dicts[name] = pickle.load(f)
        d = probs_dicts[name]
        if MOTO_ASIN not in d:
            if verbose:
                print(f"  {name}: MISSING KEY - asin {MOTO_ASIN} not in probs.pkl (n_keys={len(d)})")
            continue
        arr = d[MOTO_ASIN]
        if verbose:
            print(f"  {name}: key present, len={len(arr)}, mean={arr.mean():.4f}, "
                  f"max={arr.max():.4f}, count>0.5={int((arr>0.5).sum())}, "
                  f"count>0.3={int((arr>0.3).sum())}, count>0.1={int((arr>0.1).sum())}")

    if verbose:
        print("\nPART B: Recovery attempt - re-run Top-K with min_gap=2 for Moto G2\n")
    pelt_cps_moto = route_cps_dict.get('PELT (R1)', {}).get(MOTO_ASIN, [])
    K_target = max(1, len(pelt_cps_moto))
    if verbose:
        print(f"  PELT cps for Moto G2: {pelt_cps_moto}  -> K_target = {K_target}")

    def _topk(probs, K, min_gap):
        order = np.argsort(-probs)
        chosen = []
        for pos in order:
            if all(abs(pos - c) >= min_gap for c in chosen):
                chosen.append(int(pos))
            if len(chosen) >= K:
                break
        return sorted(chosen)

    recovered = {}
    flat_routes = []
    short_map = {'R2 (AutoCPD)': 'AutoCPD (R2)', 'R3 (TST)': 'TST (R3)',
                 'R4 (MOMENT)': 'MOMENT (R4)'}
    for name in ['R2 (AutoCPD)', 'R3 (TST)', 'R4 (MOMENT)']:
        d = probs_dicts.get(name, {})
        if MOTO_ASIN not in d:
            if verbose:
                print(f"  {name}: cannot recover - key missing in probs.pkl")
            continue
        arr = d[MOTO_ASIN]
        if float(arr.max()) < FLAT_PROBS_THRESHOLD:
            if verbose:
                print(f"  {name}: FLAT PROBS (max={float(arr.max()):.4f} < {FLAT_PROBS_THRESHOLD}) "
                      f"-> STRUCTURAL NO-DETECT. Setting cps=[].")
            flat_routes.append(name)
            recovered[short_map[name]] = []
            continue
        relaxed = _topk(arr, K=K_target, min_gap=2)
        strict = _topk(arr, K=K_target, min_gap=4)
        if verbose:
            print(f"  {name}: recovered cps (min_gap=2): {relaxed}  (original min_gap=4: {strict})")
        recovered[short_map[name]] = relaxed

    if verbose and flat_routes:
        print(f"\n  FLAT-PROBS REPORT: {len(flat_routes)} of 3 neural routes produced near-zero")
        print(f"     probability outputs for Moto G2 (structural failure). Routes: {flat_routes}")

    if recovered:
        if verbose:
            print("\nUpdating route_cps_dict with recovered Moto G2 cps...")
        for route_name, cps in recovered.items():
            if route_name not in route_cps_dict:
                route_cps_dict[route_name] = {}
            route_cps_dict[route_name][MOTO_ASIN] = cps
        if verbose:
            print(f"  Recovered for {len(recovered)} routes.")
    return probs_dicts, recovered, flat_routes


def compute_extended_hits(route_cps_dict, records_dict, anchors=EVENT_ANCHORS,
                          tolerances=TOLERANCES_EXTENDED):
    """Sec 10.4 PART C: extended tolerance sweep +/-2..+/-7. Returns (ext_df, ext_agg_df)."""
    rows = []
    for asin, info in anchors.items():
        if asin not in records_dict:
            for route_name in route_cps_dict:
                for tol in tolerances:
                    rows.append({'asin': asin, 'phone': info['title'], 'route': route_name,
                                 'tolerance_months': tol, 'hit': False,
                                 'min_dist_months': None, 'reason': 'record_missing'})
            continue
        event_date = info['event_date']
        for route_name, cps_per_asin in route_cps_dict.items():
            cps = cps_per_asin.get(asin, [])
            cp_dates = [ix_to_date(asin, ix, records_dict) for ix in cps]
            cp_dates = [d for d in cp_dates if d is not None]
            if not cp_dates:
                for tol in tolerances:
                    rows.append({'asin': asin, 'phone': info['title'], 'route': route_name,
                                 'tolerance_months': tol, 'hit': False,
                                 'min_dist_months': None, 'reason': 'no_cps'})
                continue
            min_dist = min(months_apart(d, event_date) for d in cp_dates)
            for tol in tolerances:
                rows.append({'asin': asin, 'phone': info['title'], 'route': route_name,
                             'tolerance_months': tol, 'hit': bool(min_dist <= tol),
                             'min_dist_months': int(min_dist), 'reason': 'evaluated'})

    ext_df = pd.DataFrame(rows)
    atomic_to_csv(ext_df, T2_EVENT_HITS_EXTENDED_PATH, index=False)

    agg = []
    for route_name in route_cps_dict:
        for tol in tolerances:
            sub = ext_df[(ext_df['route'] == route_name)
                         & (ext_df['tolerance_months'] == tol)
                         & (ext_df['reason'] == 'evaluated')]
            n_trials = len(sub)
            n_hits = int(sub['hit'].sum())
            agg.append({'route': route_name, 'tolerance_months': tol,
                        'n_hits': n_hits, 'n_trials': n_trials,
                        'hit_rate': round(n_hits / n_trials, 3) if n_trials else float('nan')})
    ext_agg_df = pd.DataFrame(agg)
    atomic_to_csv(ext_agg_df, T2_EVENT_SUMMARY_EXTENDED_PATH, index=False)
    return ext_df, ext_agg_df


def compute_edge_effect_flags(records_dict, anchors=EVENT_ANCHORS, threshold=EDGE_THRESHOLD):
    """Sec 10.4 PART E: flag events in series first/last 25%. Returns dict asin -> bool."""
    flags = {}
    positions = {}
    for asin, info in anchors.items():
        if asin not in records_dict:
            continue
        dates = pd.to_datetime(records_dict[asin]['dates'])
        n_obs = len(dates)
        event_date = info['event_date']
        deltas = abs(dates - event_date)
        closest_idx = int(deltas.argmin())
        position = closest_idx / max(n_obs - 1, 1)
        edge_first = position < threshold
        edge_last = position > (1 - threshold)
        flags[asin] = edge_first or edge_last
        positions[asin] = position
    return flags, positions


def run_bootstrap_n5(route_cps_dict, records_dict, ext_agg_df,
                     n_bootstrap=N_BOOTSTRAP_DEFAULT, tolerances=TOLERANCES_EXTENDED,
                     force_rerun=False, verbose=True):
    """Sec 10.4 PART F: 10K-round random-accessory bootstrap for empirical p.

    Cache-aware: if both output CSVs exist and force_rerun=False, load + return early
    (saves ~5-10 min on re-run).

    Returns (bootstrap_df, emp_p_df).
    """
    if (not force_rerun
            and T2_EVENT_BOOTSTRAP_PATH.exists()
            and T2_EVENT_EMPIRICAL_P_PATH.exists()):
        bootstrap_df = pd.read_csv(T2_EVENT_BOOTSTRAP_PATH)
        emp_p_df = pd.read_csv(T2_EVENT_EMPIRICAL_P_PATH)
        if verbose:
            print(f"Loaded N=5 bootstrap from cache: {len(bootstrap_df):,} null trials, "
                  f"{len(emp_p_df)} routextol empirical-p rows")
        return bootstrap_df, emp_p_df

    rng = np.random.default_rng(seed=RANDOM_STATE)
    phone_asins = set(EVENT_ANCHORS.keys())
    accessory_asins = sorted(set(records_dict.keys()) - phone_asins)
    if verbose:
        print(f"Pool size for bootstrap: {len(accessory_asins)} accessory ASINs")

    asin_date_ranges = {}
    for asin in accessory_asins:
        dates = pd.to_datetime(records_dict[asin]['dates'])
        asin_date_ranges[asin] = (dates.min(), dates.max())

    asin_cp_dates_per_route = {}
    for route_name in route_cps_dict:
        asin_cp_dates_per_route[route_name] = {}
        for asin in accessory_asins:
            cps = route_cps_dict[route_name].get(asin, [])
            dates = pd.to_datetime(records_dict[asin]['dates'])
            cp_dates = [dates[ix] for ix in cps if 0 <= ix < len(dates)]
            asin_cp_dates_per_route[route_name][asin] = cp_dates

    if verbose:
        print(f"Running {n_bootstrap} bootstrap rounds, {N_SAMPLE_PER_ROUND}/round, "
              f"{len(route_cps_dict)} routes x {len(tolerances)} tolerances...")
    t0 = time.time()

    bootstrap_rows = []
    for round_idx in range(n_bootstrap):
        sampled_asins = rng.choice(accessory_asins, size=N_SAMPLE_PER_ROUND, replace=False)
        sampled_dates = []
        for a in sampled_asins:
            lo, hi = asin_date_ranges[a]
            span_days = (hi - lo).days
            offset = int(rng.integers(0, max(span_days, 1) + 1))
            sampled_dates.append(lo + pd.Timedelta(days=offset))
        for route_name, asin_cp_dates in asin_cp_dates_per_route.items():
            for tol in tolerances:
                n_hits = 0
                n_evaluated = 0
                for asin, anchor_date in zip(sampled_asins, sampled_dates):
                    cp_dates = asin_cp_dates[asin]
                    if not cp_dates:
                        continue
                    n_evaluated += 1
                    min_d = min(months_apart(d, anchor_date) for d in cp_dates)
                    if min_d <= tol:
                        n_hits += 1
                bootstrap_rows.append({
                    'route': route_name, 'tolerance_months': tol, 'round': round_idx,
                    'null_n_hits': n_hits, 'null_n_trials': n_evaluated,
                    'null_hit_rate': n_hits / n_evaluated if n_evaluated else 0.0,
                })
    if verbose:
        print(f"Bootstrap elapsed: {format_elapsed(time.time() - t0)}")
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    atomic_to_csv(bootstrap_df, T2_EVENT_BOOTSTRAP_PATH, index=False)

    # Empirical p
    emp_rows = []
    for route_name in route_cps_dict:
        for tol in tolerances:
            obs_row = ext_agg_df[(ext_agg_df['route'] == route_name)
                                  & (ext_agg_df['tolerance_months'] == tol)]
            if obs_row.empty:
                continue
            obs_rate = float(obs_row.iloc[0]['hit_rate'])
            obs_hits = int(obs_row.iloc[0]['n_hits'])
            obs_trials = int(obs_row.iloc[0]['n_trials'])
            null_sub = bootstrap_df[(bootstrap_df['route'] == route_name)
                                     & (bootstrap_df['tolerance_months'] == tol)]
            null_rates = null_sub['null_hit_rate'].values
            if len(null_rates) == 0:
                continue
            empirical_p = float((null_rates >= obs_rate).mean())
            emp_rows.append({
                'route': route_name, 'tolerance_months': tol,
                'observed_hits': obs_hits, 'observed_trials': obs_trials,
                'observed_hit_rate': round(obs_rate, 3),
                'null_mean_hit_rate': round(float(null_rates.mean()), 3),
                'null_p95_hit_rate': round(float(np.percentile(null_rates, 95)), 3),
                'empirical_p_one_sided': round(empirical_p, 3),
            })
    emp_p_df = pd.DataFrame(emp_rows)
    atomic_to_csv(emp_p_df, T2_EVENT_EMPIRICAL_P_PATH, index=False)
    return bootstrap_df, emp_p_df
