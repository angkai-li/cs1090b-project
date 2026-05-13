"""
Sec 10.5 PART H + Sec 10.6 PART K - Multi-anchor (N=14) hit rate + 10K bootstrap.

Filter the 21 KNOWN_EVENTS to interior (25-75% of series) -> N=14 anchors.
Compute hit rate per route x tolerance, then bootstrap a null distribution by
sampling N=14 random (asin, date) pairs from the 269 accessory products.
"""

import time

import numpy as np
import pandas as pd

from ..config.hyperparams import RANDOM_STATE
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv
from .events import (
    EVENT_ANCHORS, KNOWN_EVENTS, T2_EVENT_VALIDATION_DIR, TOLERANCES_EXTENDED,
    ix_to_date, months_apart,
)


T2_EVENT_MULTI_HITS_PATH = T2_EVENT_VALIDATION_DIR / "event_multi_anchor_hits.csv"
T2_EVENT_MULTI_SUMMARY_PATH = T2_EVENT_VALIDATION_DIR / "event_multi_anchor_summary.csv"
T2_EVENT_MULTI_BOOTSTRAP_PATH = T2_EVENT_VALIDATION_DIR / "event_multi_anchor_bootstrap.csv"

INTERIOR_LOW = 0.25
INTERIOR_HIGH = 0.75


def build_multi_anchors(records_dict, known_events=KNOWN_EVENTS,
                       interior_low=INTERIOR_LOW, interior_high=INTERIOR_HIGH, verbose=True):
    """Filter all KNOWN_EVENTS to those falling in interior (25-75%) of their series."""
    multi_anchors = []
    for asin, info in known_events.items():
        if asin not in records_dict:
            continue
        dates = pd.to_datetime(records_dict[asin]['dates'])
        n_obs = len(dates)
        for event_date, label, strength in info['events']:
            if event_date < dates.min() or event_date > dates.max():
                continue
            deltas = abs(dates - event_date)
            closest_idx = int(deltas.argmin())
            pos = closest_idx / max(n_obs - 1, 1)
            if interior_low <= pos <= interior_high:
                multi_anchors.append({
                    'asin': asin, 'phone': info['title'],
                    'event_date': event_date, 'event_label': label,
                    'strength': strength, 'position_pct': pos,
                })
    if verbose:
        print(f"Total interior anchors across all 5 phones: {len(multi_anchors)}")
        for a in multi_anchors:
            print(f"  {a['phone']:25s} {a['event_date'].strftime('%Y-%m-%d')}  "
                  f"{a['event_label'][:50]:52s} ({a['strength']}, pos={a['position_pct']:.0%})")
    return multi_anchors


def compute_multi_anchor_hits(multi_anchors, route_cps_dict, records_dict,
                               tolerances=TOLERANCES_EXTENDED, verbose=True):
    """Compute hits per (anchor, route, tolerance) + summary."""
    rows = []
    for anchor in multi_anchors:
        asin = anchor['asin']
        event_date = anchor['event_date']
        for route_name, cps_per_asin in route_cps_dict.items():
            cps = cps_per_asin.get(asin, [])
            cp_dates = [ix_to_date(asin, ix, records_dict) for ix in cps]
            cp_dates = [d for d in cp_dates if d is not None]
            if not cp_dates:
                for tol in tolerances:
                    rows.append({
                        'asin': asin, 'phone': anchor['phone'],
                        'event_label': anchor['event_label'],
                        'event_date': event_date.strftime('%Y-%m-%d'),
                        'route': route_name, 'tolerance_months': tol,
                        'hit': False, 'min_dist_months': None,
                        'reason': 'no_cps',
                    })
                continue
            min_dist = min(months_apart(d, event_date) for d in cp_dates)
            for tol in tolerances:
                rows.append({
                    'asin': asin, 'phone': anchor['phone'],
                    'event_label': anchor['event_label'],
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'route': route_name, 'tolerance_months': tol,
                    'hit': bool(min_dist <= tol),
                    'min_dist_months': int(min_dist),
                    'reason': 'evaluated',
                })
    hits_df = pd.DataFrame(rows)
    atomic_to_csv(hits_df, T2_EVENT_MULTI_HITS_PATH, index=False)

    agg = []
    for route_name in route_cps_dict:
        for tol in tolerances:
            sub = hits_df[(hits_df['route'] == route_name)
                          & (hits_df['tolerance_months'] == tol)
                          & (hits_df['reason'] == 'evaluated')]
            n_trials = len(sub)
            n_hits = int(sub['hit'].sum())
            agg.append({
                'route': route_name, 'tolerance_months': tol,
                'n_hits': n_hits, 'n_trials': n_trials,
                'hit_rate': round(n_hits / n_trials, 3) if n_trials else float('nan'),
            })
    summary_df = pd.DataFrame(agg)
    atomic_to_csv(summary_df, T2_EVENT_MULTI_SUMMARY_PATH, index=False)

    if verbose:
        print(f"\n=== Multi-anchor hit rates (interior-only, N up to {len(multi_anchors)}) ===")
        pivot = summary_df.pivot_table(index='route', columns='tolerance_months',
                                        values='hit_rate', aggfunc='first').round(3)
        print(pivot)
    return hits_df, summary_df


def run_multi_anchor_bootstrap(route_cps_dict, records_dict, multi_summary_df,
                                anchors_n=14, n_bootstrap=10000,
                                tolerances=TOLERANCES_EXTENDED, force_rerun=False, verbose=True):
    """Sec 10.6 PART K: 10K-round bootstrap with N random (asin, date) anchors.

    Per round: sample anchors_n random (accessory_asin, date in series window),
    compute hit rate vs each route at each tolerance.
    Returns DataFrame with empirical p per (route, tolerance).
    """
    if not force_rerun and T2_EVENT_MULTI_BOOTSTRAP_PATH.exists():
        df = pd.read_csv(T2_EVENT_MULTI_BOOTSTRAP_PATH)
        if verbose:
            print(f"Loaded multi-anchor bootstrap from cache: {len(df)} routextol rows")
        return df

    rng = np.random.default_rng(seed=RANDOM_STATE)
    phone_asins = set(EVENT_ANCHORS.keys())
    accessory_asins = sorted(set(records_dict.keys()) - phone_asins)

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
        print(f"Running {n_bootstrap}-round multi-anchor null bootstrap "
              f"(N={anchors_n} random anchors/round)...")
    t0 = time.time()

    null_rates = {(rn, tol): [] for rn in route_cps_dict for tol in tolerances}

    for _ in range(n_bootstrap):
        # Sample N random accessory anchors (with replacement) + random dates
        sampled_asins = rng.choice(accessory_asins, size=anchors_n, replace=True)
        sampled_dates = []
        for a in sampled_asins:
            lo, hi = asin_date_ranges[a]
            span_days = (hi - lo).days
            offset = int(rng.integers(0, max(span_days, 1) + 1))
            sampled_dates.append(lo + pd.Timedelta(days=offset))
        for route_name, asin_cp_dates in asin_cp_dates_per_route.items():
            for tol in tolerances:
                n_hits = 0
                n_eval = 0
                for asin, anchor_date in zip(sampled_asins, sampled_dates):
                    cp_dates = asin_cp_dates[asin]
                    if not cp_dates:
                        continue
                    n_eval += 1
                    min_d = min(months_apart(d, anchor_date) for d in cp_dates)
                    if min_d <= tol:
                        n_hits += 1
                rate = n_hits / n_eval if n_eval else 0.0
                null_rates[(route_name, tol)].append(rate)

    if verbose:
        print(f"Multi-anchor bootstrap elapsed: {format_elapsed(time.time() - t0)}")

    rows = []
    for (route_name, tol), null_list in null_rates.items():
        obs_row = multi_summary_df[(multi_summary_df['route'] == route_name)
                                    & (multi_summary_df['tolerance_months'] == tol)]
        if obs_row.empty:
            continue
        obs_rate = float(obs_row.iloc[0]['hit_rate'])
        obs_hits = int(obs_row.iloc[0]['n_hits'])
        obs_trials = int(obs_row.iloc[0]['n_trials'])
        if not null_list:
            continue
        null_arr = np.array(null_list)
        empirical_p = float((null_arr >= obs_rate).mean())
        rows.append({
            'route': route_name, 'tolerance_months': tol,
            'observed_hits': obs_hits, 'observed_trials': obs_trials,
            'observed_hit_rate': round(obs_rate, 3),
            'null_mean_hit_rate': round(float(null_arr.mean()), 3),
            'null_p95_hit_rate': round(float(np.percentile(null_arr, 95)), 3),
            'lift': round(obs_rate - float(null_arr.mean()), 3),
            'empirical_p_one_sided': round(empirical_p, 4),
        })
    df = pd.DataFrame(rows)
    atomic_to_csv(df, T2_EVENT_MULTI_BOOTSTRAP_PATH, index=False)
    if verbose:
        print("\n=== Multi-anchor bootstrap empirical p-values ===")
        print(df.to_string(index=False))
    return df
