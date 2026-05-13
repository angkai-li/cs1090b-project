"""
Sec 10.5 PART G + Sec 10.6 PART I - Cohen's d + Hedges' g with bootstrap CI.

Hedges' g is Cohen's d x J small-sample correction (Hedges & Olkin 1985):
    J = 1 - 3 / (4 * df - 1),   df = n1 + n2 - 2
For n=6+6=12 -> J ~ 0.929 (shrinks |d| by ~7%, removes upward bias).
"""

import numpy as np
import pandas as pd

from ..utils.io import atomic_to_csv
from .events import (
    BOOTSTRAP_CI_N, EVENT_ANCHORS, T2_EVENT_VALIDATION_DIR, WINDOW_MONTHS,
)


T2_EVENT_MAGNITUDE_PATH = T2_EVENT_VALIDATION_DIR / "event_magnitude_cohend.csv"
T2_EVENT_HEDGES_PATH = T2_EVENT_VALIDATION_DIR / "event_magnitude_hedges_g.csv"


def cohens_d(pre, post):
    """Standardized mean difference: (post - pre) / pooled_std."""
    n1, n2 = len(pre), len(post)
    if n1 < 2 or n2 < 2:
        return float('nan')
    var1, var2 = np.var(pre, ddof=1), np.var(post, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return float('nan')
    return (np.mean(post) - np.mean(pre)) / pooled


def hedges_j(n1, n2):
    """Small-sample correction factor: J = 1 - 3/(4*df-1)."""
    df = n1 + n2 - 2
    if df <= 0:
        return 1.0
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def hedges_g(pre, post):
    d = cohens_d(pre, post)
    if np.isnan(d):
        return float('nan')
    return d * hedges_j(len(pre), len(post))


def bootstrap_ci_cohen(pre, post, n_resamples=BOOTSTRAP_CI_N, alpha=0.05, seed=42):
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = []
    n1, n2 = len(pre), len(post)
    for _ in range(n_resamples):
        b_pre = rng.choice(pre, size=n1, replace=True)
        b_post = rng.choice(post, size=n2, replace=True)
        d = cohens_d(b_pre, b_post)
        if not np.isnan(d):
            ds.append(d)
    if len(ds) < 100:
        return (float('nan'), float('nan'))
    return (float(np.percentile(ds, 100 * alpha / 2)),
            float(np.percentile(ds, 100 * (1 - alpha / 2))))


def bootstrap_ci_hedges(pre, post, n_resamples=BOOTSTRAP_CI_N, alpha=0.05, seed=42):
    """Bootstrap 95% CI for Hedges' g."""
    rng = np.random.default_rng(seed)
    gs = []
    n1, n2 = len(pre), len(post)
    for _ in range(n_resamples):
        b_pre = rng.choice(pre, size=n1, replace=True)
        b_post = rng.choice(post, size=n2, replace=True)
        g = hedges_g(b_pre, b_post)
        if not np.isnan(g):
            gs.append(g)
    if len(gs) < 100:
        return (float('nan'), float('nan'))
    return (float(np.percentile(gs, 100 * alpha / 2)),
            float(np.percentile(gs, 100 * (1 - alpha / 2))))


def _extract_window(records_dict, asin, event_date, window_months=WINDOW_MONTHS):
    """Extract (pre, post) series arrays for +/-window_months around event_date."""
    r = records_dict[asin]
    dates = pd.to_datetime(r['dates'])
    series = r['series_norm']
    delta_months = pd.Series((dates - event_date).total_seconds() / (3600 * 24 * 30.5))
    pre_mask = (delta_months >= -window_months) & (delta_months < 0)
    post_mask = (delta_months >= 0) & (delta_months <= window_months)
    return series[pre_mask.values].astype(float), series[post_mask.values].astype(float)


def compute_cohen_table(records_dict, anchors=EVENT_ANCHORS, window_months=WINDOW_MONTHS):
    """Sec 10.5 PART G: Cohen's d for each anchor event. Returns DataFrame."""
    rows = []
    for asin, info in anchors.items():
        if asin not in records_dict:
            continue
        event_date = info['event_date']
        pre, post = _extract_window(records_dict, asin, event_date, window_months)
        n_pre, n_post = len(pre), len(post)
        if n_pre < 2 or n_post < 2:
            rows.append({
                'asin': asin, 'phone': info['title'],
                'event_label': info['event_label'],
                'event_date': event_date.strftime('%Y-%m-%d'),
                'n_pre': n_pre, 'n_post': n_post,
                'mean_pre': round(float(pre.mean()), 3) if n_pre else float('nan'),
                'mean_post': round(float(post.mean()), 3) if n_post else float('nan'),
                'cohen_d': float('nan'),
                'ci_lower': float('nan'), 'ci_upper': float('nan'),
                'interpretation': 'INSUFFICIENT DATA (n_pre or n_post < 2)',
            })
            continue
        d = cohens_d(pre, post)
        ci_lo, ci_hi = bootstrap_ci_cohen(pre, post)
        abs_d = abs(d)
        direction = "DROP" if d < 0 else "RISE"
        size = "trivial" if abs_d < 0.2 else "small" if abs_d < 0.5 else \
               "medium" if abs_d < 0.8 else "LARGE"
        sig = "(CI excludes 0)" if (ci_lo > 0 or ci_hi < 0) else "(CI includes 0 - not sig)"
        rows.append({
            'asin': asin, 'phone': info['title'],
            'event_label': info['event_label'],
            'event_date': event_date.strftime('%Y-%m-%d'),
            'n_pre': n_pre, 'n_post': n_post,
            'mean_pre': round(float(pre.mean()), 3),
            'mean_post': round(float(post.mean()), 3),
            'cohen_d': round(float(d), 3),
            'ci_lower': round(ci_lo, 3), 'ci_upper': round(ci_hi, 3),
            'interpretation': f"{size} {direction} {sig}",
        })
    df = pd.DataFrame(rows)
    atomic_to_csv(df, T2_EVENT_MAGNITUDE_PATH, index=False)
    return df


def compute_hedges_table(records_dict, anchors=EVENT_ANCHORS, window_months=WINDOW_MONTHS,
                        verbose=True):
    """Sec 10.6 PART I: Hedges' g + J correction for each anchor. Returns DataFrame."""
    rows = []
    for asin, info in anchors.items():
        if asin not in records_dict:
            continue
        event_date = info['event_date']
        pre, post = _extract_window(records_dict, asin, event_date, window_months)
        n_pre, n_post = len(pre), len(post)
        if n_pre < 2 or n_post < 2:
            rows.append({
                'asin': asin, 'phone': info['title'], 'event_label': info['event_label'],
                'event_date': event_date.strftime('%Y-%m-%d'),
                'n_pre': n_pre, 'n_post': n_post,
                'cohen_d': float('nan'), 'hedges_g': float('nan'),
                'j_correction': float('nan'),
                'ci_lower_g': float('nan'), 'ci_upper_g': float('nan'),
                'interpretation': 'INSUFFICIENT DATA',
            })
            continue
        d = cohens_d(pre, post)
        j = hedges_j(n_pre, n_post)
        g = d * j
        ci_lo, ci_hi = bootstrap_ci_hedges(pre, post)
        abs_g = abs(g)
        direction = "DROP" if g < 0 else "RISE"
        size = "trivial" if abs_g < 0.2 else "small" if abs_g < 0.5 else \
               "medium" if abs_g < 0.8 else "LARGE"
        sig = "(CI excludes 0)" if (ci_lo > 0 or ci_hi < 0) else "(CI includes 0)"
        rows.append({
            'asin': asin, 'phone': info['title'], 'event_label': info['event_label'],
            'event_date': event_date.strftime('%Y-%m-%d'),
            'n_pre': n_pre, 'n_post': n_post,
            'cohen_d': round(float(d), 3),
            'hedges_g': round(float(g), 3),
            'j_correction': round(float(j), 3),
            'ci_lower_g': round(ci_lo, 3), 'ci_upper_g': round(ci_hi, 3),
            'interpretation': f"{size} {direction} {sig}",
        })
    df = pd.DataFrame(rows)
    atomic_to_csv(df, T2_EVENT_HEDGES_PATH, index=False)
    if verbose:
        print(df.to_string(index=False))
        print("\nJ correction factor < 1: shrinks |d| -> unbiased estimate per Hedges & Olkin (1985).")
    return df
