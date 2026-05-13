"""
Sec 10.5 PART F + Sec 10.6 PART J - Reverse-direction lookup + random-cp null.

PART F: For each (phone x route), find closest event in 21-event KNOWN_EVENTS
        database to each detected cp.
PART J: Compare observed match rate to a random-cp permutation null (1000
        permutations preserving cps count per asin).
"""

import numpy as np
import pandas as pd

from ..config.hyperparams import RANDOM_STATE
from ..utils.io import atomic_to_csv
from .events import (
    EVENT_ANCHORS, KNOWN_EVENTS, T2_EVENT_VALIDATION_DIR,
)


T2_EVENT_REVERSE_PATH = T2_EVENT_VALIDATION_DIR / "event_reverse_lookup.csv"
T2_EVENT_REVERSE_NULL_PATH = T2_EVENT_VALIDATION_DIR / "event_reverse_null_baseline.csv"

N_REVERSE_NULL_DEFAULT = 1000


def reverse_lookup(route_cps_dict, records_dict, anchors=EVENT_ANCHORS,
                   known_events=KNOWN_EVENTS, verbose=True):
    """Sec 10.5 PART F: each (phone x route x cp) -> closest known event.

    Returns DataFrame with one row per (asin, route, cp).
    """
    rows = []
    for asin, info in anchors.items():
        if asin not in records_dict:
            continue
        phone_known = known_events.get(asin, {}).get('events', [])
        if not phone_known:
            continue
        if verbose:
            print(f"\n{info['title']} ({asin}) - {len(phone_known)} candidate events in database")
        for route_name, cps_per_asin in route_cps_dict.items():
            cps = cps_per_asin.get(asin, [])
            if not cps:
                continue
            r = records_dict[asin]
            dates = pd.to_datetime(r['dates'])
            for cp_idx in cps:
                if not (0 <= cp_idx < len(dates)):
                    continue
                cp_date = dates[cp_idx]
                dists = [(abs((d - cp_date).days) / 30.5, lbl, d, strength)
                         for d, lbl, strength in phone_known]
                min_d, best_lbl, best_d, best_strength = min(dists, key=lambda x: x[0])
                verdict = ("STRONG MATCH" if min_d <= 1 else
                           "GOOD MATCH" if min_d <= 2 else
                           "WEAK MATCH" if min_d <= 4 else "no clear match")
                rows.append({
                    'asin': asin, 'phone': info['title'], 'route': route_name,
                    'cp_date': cp_date.strftime('%Y-%m'),
                    'closest_event_date': best_d.strftime('%Y-%m-%d'),
                    'closest_event_label': best_lbl,
                    'closest_event_strength': best_strength,
                    'distance_months': round(min_d, 1),
                    'verdict': verdict,
                })
                if verbose:
                    print(f"    {route_name:14s} cp @ {cp_date.strftime('%Y-%m')}  "
                          f"-> closest event {best_d.strftime('%Y-%m')} '{best_lbl[:38]}'  "
                          f"(+/-{min_d:.1f} mo, {verdict})")
    df = pd.DataFrame(rows)
    atomic_to_csv(df, T2_EVENT_REVERSE_PATH, index=False)
    if verbose:
        print()
        print("=== Summary: cps explained by KNOWN_EVENTS database ===")
        for route_name in route_cps_dict:
            sub = df[df['route'] == route_name]
            if sub.empty:
                continue
            n_total = len(sub)
            n_strong = int((sub['distance_months'] <= 1).sum())
            n_good = int((sub['distance_months'] <= 2).sum())
            n_weak = int((sub['distance_months'] <= 4).sum())
            print(f"  {route_name:14s} {n_total} total cps  |  "
                  f"strong (<=1mo): {n_strong}/{n_total} ({n_strong/n_total:.0%})  |  "
                  f"good (<=2mo): {n_good}/{n_total} ({n_good/n_total:.0%})  |  "
                  f"weak (<=4mo): {n_weak}/{n_total} ({n_weak/n_total:.0%})")
    return df


def compute_match_rate(route_cps_dict_input, records_dict, anchors=EVENT_ANCHORS,
                       known_events=KNOWN_EVENTS, tolerance_months=2):
    """Fraction of cps within `tolerance_months` of any KNOWN_EVENT for that phone."""
    known_dates_per_asin = {
        asin: [d for d, _, _ in info['events']]
        for asin, info in known_events.items()
    }
    total_cps = 0
    total_matches = 0
    for asin, info in anchors.items():
        if asin not in records_dict:
            continue
        events = known_dates_per_asin.get(asin, [])
        if not events:
            continue
        cps = route_cps_dict_input.get(asin, [])
        dates = pd.to_datetime(records_dict[asin]['dates'])
        for cp_idx in cps:
            if not (0 <= cp_idx < len(dates)):
                continue
            cp_date = dates[cp_idx]
            min_d = min(abs((d - cp_date).days) / 30.5 for d in events)
            total_cps += 1
            if min_d <= tolerance_months:
                total_matches += 1
    rate = (total_matches / total_cps) if total_cps > 0 else 0.0
    return rate, total_matches, total_cps


def random_cp_permutation_null(route_cps_dict, records_dict, anchors=EVENT_ANCHORS,
                                known_events=KNOWN_EVENTS, n_permutations=N_REVERSE_NULL_DEFAULT,
                                tolerances=(1, 2, 4), force_rerun=False, verbose=True):
    """Sec 10.6 PART J: null distribution from random-cp permutations.

    For each route, repeatedly:
      1. For each asin, sample N=len(route's cps for that asin) random positions
         uniformly in [0, n_obs).
      2. Compute match rate vs KNOWN_EVENTS at each tolerance.
    Returns DataFrame with (route, tolerance, observed_match_rate, null_mean, null_p95,
    lift, empirical_p_one_sided).
    """
    if not force_rerun and T2_EVENT_REVERSE_NULL_PATH.exists():
        df = pd.read_csv(T2_EVENT_REVERSE_NULL_PATH)
        if verbose:
            print(f"Loaded reverse-direction null from cache: {len(df)} routextol rows")
        return df

    rng = np.random.default_rng(seed=RANDOM_STATE)

    known_dates_per_asin = {
        asin: [d for d, _, _ in info['events']]
        for asin, info in known_events.items()
    }

    if verbose:
        print(f"Running {n_permutations}-round random-cp null simulation per route...")

    null_match_rates = {(rn, tol): [] for rn in route_cps_dict for tol in tolerances}
    for _ in range(n_permutations):
        for route_name, cps_per_asin in route_cps_dict.items():
            # Sample a random-cps version for each asin
            random_cps = {}
            for asin in anchors:
                if asin not in records_dict:
                    continue
                cps = cps_per_asin.get(asin, [])
                n_cps = len(cps)
                if n_cps == 0:
                    random_cps[asin] = []
                    continue
                n_obs = len(records_dict[asin]['series_norm'])
                if n_obs <= 0:
                    random_cps[asin] = []
                    continue
                random_cps[asin] = sorted(rng.integers(0, n_obs, size=n_cps).tolist())
            for tol in tolerances:
                rate, _, _ = compute_match_rate(random_cps, records_dict, anchors,
                                                 known_events, tolerance_months=tol)
                null_match_rates[(route_name, tol)].append(rate)

    rows = []
    for (route_name, tol), null_rates in null_match_rates.items():
        obs_rate, n_match, n_total = compute_match_rate(
            route_cps_dict.get(route_name, {}), records_dict, anchors,
            known_events, tolerance_months=tol)
        if not null_rates:
            continue
        null_arr = np.array(null_rates)
        empirical_p = float((null_arr >= obs_rate).mean())
        rows.append({
            'route': route_name, 'tolerance_months': tol,
            'observed_match_rate': round(obs_rate, 3),
            'n_match': n_match, 'n_total': n_total,
            'null_mean': round(float(null_arr.mean()), 3),
            'null_p95': round(float(np.percentile(null_arr, 95)), 3),
            'lift': round(obs_rate - float(null_arr.mean()), 3),
            'empirical_p_one_sided': round(empirical_p, 3),
        })
    df = pd.DataFrame(rows)
    atomic_to_csv(df, T2_EVENT_REVERSE_NULL_PATH, index=False)
    if verbose:
        print("\n=== Reverse-direction null baseline ===")
        print(df.to_string(index=False))
    return df
