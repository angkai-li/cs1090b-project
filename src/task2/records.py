"""
Task 2 data preparation (Sec 9.1 + Sec 9.3):
  - Build DeBERTa-weighted product-month / product-week panels (Sec 9.1)
  - Per-series z-score normalization (RevIN-style)
  - Per-product embedding IDs + <RARE> bucket
  - Hierarchical L2 shrinkage scale (1/sqrt(n_p))
  - Assemble training records (one entry per product)
"""

import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from ..config.hyperparams import (
    ALPHA, BETA, EMBED_DIM, LAMBDA_L2, MIN_OBS_PER_MONTH, MIN_OBS_PER_WEEK,
    MIN_REVIEWS_TASK2, RARE_THRESHOLD_MONTHS,
)
from ..config.paths import (
    CLEAN_CACHE_META_PATH,
    T2_PANEL_MONTHLY_META_PATH, T2_PANEL_MONTHLY_PATH,
    T2_PANEL_WEEKLY_PATH,
    T2_RECORDS_META_PATH, T2_RECORDS_PATH,
)
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv, atomic_to_pickle, atomic_write_text, read_json_or_none
from ..utils.metrics import weighted_mean_with_fallback


T2_PANEL_VERSION = 't2_panel'
T2_DATAPREP_VERSION = 't2_dataprep'


# ====================================================================
# Sec 9.1 DeBERTa-weighted panels (product-month + product-week)
# ====================================================================
def task2_panel_cache_is_valid():
    return T2_PANEL_MONTHLY_PATH.exists() and T2_PANEL_WEEKLY_PATH.exists()


def build_period_panel(df_in, freq, min_obs):
    """Aggregate per-review DeBERTa scores to product-period level.

    Args:
        df_in: DataFrame with cols asin, review_date, score_deberta_v3_base_lora,
               sentiment, overall, verified, vote, review_weight
        freq:  pandas Grouper freq str ('ME' for month-end, 'W' for week-end)
        min_obs: drop (asin, period) rows with num_reviews < min_obs

    Returns:
        DataFrame with columns: asin, review_date, deberta_sentiment_naive/weighted,
        rating_sentiment_weighted, rating_mean, num_reviews, verified_ratio,
        helpful_vote_mean, sentiment_std, review_weight_sum
    """
    panel = (
        df_in
        .groupby(['asin', pd.Grouper(key='review_date', freq=freq)])
        .apply(lambda g: pd.Series({
            'deberta_sentiment_naive':    g['score_deberta_v3_base_lora'].mean(),
            'deberta_sentiment_weighted': weighted_mean_with_fallback(
                g, 'score_deberta_v3_base_lora', 'review_weight'),
            'rating_sentiment_weighted':  weighted_mean_with_fallback(
                g, 'sentiment', 'review_weight'),
            'rating_mean':                g['overall'].mean(),
            'num_reviews':                len(g),
            'verified_ratio':             g['verified'].mean(),
            'helpful_vote_mean':          g['vote'].mean(),
            'sentiment_std':              g['score_deberta_v3_base_lora'].std(),
            'review_weight_sum':          g['review_weight'].sum(),
        }))
        .reset_index()
    )
    return panel[panel['num_reviews'] >= min_obs].copy()


def build_or_load_panels(df, force_rerun=False, verbose=True):
    """Build monthly + weekly DeBERTa panels, or load from cache.

    Returns (monthly_panel, weekly_panel).
    """
    if not force_rerun and task2_panel_cache_is_valid():
        monthly = pd.read_csv(T2_PANEL_MONTHLY_PATH, parse_dates=['review_date'])
        weekly  = pd.read_csv(T2_PANEL_WEEKLY_PATH,  parse_dates=['review_date'])
        if verbose:
            print(f"Loaded Task 2 panels from cache.")
            print(f"  Monthly: {monthly.shape}  Weekly: {weekly.shape}")
        return monthly, weekly

    t0 = time.time()
    required = {
        'asin', 'review_date',
        'score_deberta_v3_base_lora', 'score_tfidf_ridge', 'sentiment',
        'overall', 'verified', 'vote', 'review_weight',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot build Task 2 panels. Missing columns: {sorted(missing)}. "
                         "Run Task 1 (DeBERTa scoring) and ensure cleaning has review_weight.")

    counts = df['asin'].value_counts()
    target_asins = counts[counts >= MIN_REVIEWS_TASK2].index
    df_target = df[df['asin'].isin(target_asins)].copy()
    if verbose:
        print(f"Target products (>= {MIN_REVIEWS_TASK2} reviews): {len(target_asins)}")
        print(f"Total reviews from target products: {len(df_target):,}")

    monthly = build_period_panel(df_target, freq='ME', min_obs=MIN_OBS_PER_MONTH)
    weekly  = build_period_panel(df_target, freq='W',  min_obs=MIN_OBS_PER_WEEK)
    atomic_to_csv(monthly, T2_PANEL_MONTHLY_PATH, index=False)
    atomic_to_csv(weekly,  T2_PANEL_WEEKLY_PATH,  index=False)

    config = {
        'version': T2_PANEL_VERSION,
        'min_reviews_task2': MIN_REVIEWS_TASK2,
        'min_obs_per_month': MIN_OBS_PER_MONTH,
        'min_obs_per_week':  MIN_OBS_PER_WEEK,
        'clean_cache_meta':  read_json_or_none(CLEAN_CACHE_META_PATH),
    }
    atomic_write_text(T2_PANEL_MONTHLY_META_PATH, json.dumps({'t2_panel_config': config}, indent=2))
    if verbose:
        print(f"\nMonthly panel: {monthly.shape}  Saved to {T2_PANEL_MONTHLY_PATH}")
        print(f"Weekly panel:  {weekly.shape}  Saved to {T2_PANEL_WEEKLY_PATH}")
        print(f"Task 2 panel elapsed: {format_elapsed(time.time() - t0)}")
    return monthly, weekly


# ====================================================================
# Sec 9.3 Weighted sentiment score + shrinkage helpers
# ====================================================================
def weighted_sentiment_score(s, n, alpha=ALPHA, beta=BETA):
    """Volume-weighted sentiment with smooth scaling to [-1, +1].

    raw = (alpha * log1p(n) + beta) * sentiment
    out = tanh(raw)

    Larger n_reviews -> stronger weight; beta provides minimum weight for low-n periods.
    """
    raw = (alpha * np.log1p(n) + beta) * s
    return np.tanh(raw)


def shrinkage_loss(embed_module, scale, lam=LAMBDA_L2):
    """L2 weight-decay scaled by 1/sqrt(n_p) per product (hierarchical shrinkage).

    Stein-James / Empirical Bayes style: products with few obs shrink toward zero
    more aggressively, products with many obs barely shrink.
    """
    weight = embed_module.embed.weight if hasattr(embed_module, "embed") else embed_module.weight
    norms = weight.pow(2).sum(dim=1)
    return lam * (scale.to(norms.device) * norms).sum()


# ====================================================================
# Sec 9.3 Build training records (one entry per product)
# ====================================================================
def task2_records_cache_is_valid():
    return T2_RECORDS_PATH.exists() and T2_PANEL_MONTHLY_PATH.exists()


def build_records_from_panel(monthly_panel, rare_threshold_months=RARE_THRESHOLD_MONTHS,
                             embed_dim=EMBED_DIM, verbose=True):
    """Build the per-product training records list from a monthly panel.

    Pipeline (Sec 9.3):
      1. Per-series z-score normalization (RevIN-style)
      2. Per-product embedding ID assignment, with <RARE> bucket (ID=0)
      3. Hierarchical L2 shrinkage scale (1/sqrt(n_p)) for embedding weight decay
      4. Assemble records (each: asin, asin_id, series_norm, series_raw, etc.)

    Returns:
      records (list of dict), asin_to_id (dict), n_products (int),
      weight_decay_scale (torch.Tensor of shape [n_products]).
    """
    monthly_panel = monthly_panel.copy()
    monthly_panel['weighted_sentiment'] = weighted_sentiment_score(
        monthly_panel['deberta_sentiment_weighted'],
        monthly_panel['num_reviews'],
    )

    # Step 1: Per-series z-score normalization
    sigma_floor = monthly_panel.groupby('asin')['weighted_sentiment'].std().quantile(0.10)
    monthly_panel['mu_p'] = monthly_panel.groupby('asin')['weighted_sentiment'].transform('mean')
    monthly_panel['sigma_p'] = (
        monthly_panel.groupby('asin')['weighted_sentiment'].transform('std').clip(lower=sigma_floor)
    )
    monthly_panel['sentiment_norm'] = (
        (monthly_panel['weighted_sentiment'] - monthly_panel['mu_p']) / monthly_panel['sigma_p']
    )
    if verbose:
        print(f"sentiment_norm: mean={monthly_panel['sentiment_norm'].mean():.4f}, "
              f"std={monthly_panel['sentiment_norm'].std():.4f}")

    # Step 2: <RARE> bucket (ID=0 for short series)
    asin_obs = monthly_panel.groupby('asin').size()
    unique_asins = list(asin_obs.index)
    rare_asins = set(asin_obs[asin_obs < rare_threshold_months].index)
    asin_to_id = {a: 0 for a in rare_asins}
    non_rare_sorted = sorted([a for a in unique_asins if a not in rare_asins])
    asin_to_id.update({a: i + 1 for i, a in enumerate(non_rare_sorted)})
    n_products = max(asin_to_id.values()) + 1
    if verbose:
        print(f"Total IDs: {n_products} (1 RARE + {n_products-1} non-RARE)")
        print(f"RARE bucket size: {len(rare_asins)}")

    # Step 3: Hierarchical L2 shrinkage scale
    n_per_product = torch.zeros(n_products, dtype=torch.float32)
    for asin, pid in asin_to_id.items():
        n_per_product[pid] += float(asin_obs.get(asin, 0))
    n_per_product = n_per_product.clamp(min=1.0)
    weight_decay_scale = 1.0 / torch.sqrt(n_per_product)

    # Step 4: Build records (mix as independent - product identity carried by embedding)
    records = []
    for asin, group in monthly_panel.groupby('asin'):
        asin_id = asin_to_id.get(asin, 0)
        g = group.sort_values('review_date').reset_index(drop=True)
        records.append({
            'asin': asin, 'asin_id': asin_id,
            'series_norm':    g['sentiment_norm'].values.astype(np.float32),
            'series_raw':     g['weighted_sentiment'].values.astype(np.float32),
            'num_reviews':    g['num_reviews'].values.astype(np.float32),
            'rating_mean':    g['rating_mean'].values.astype(np.float32),
            'verified_ratio': g['verified_ratio'].values.astype(np.float32),
            'helpful_mean':   g['helpful_vote_mean'].values.astype(np.float32),
            'dates':          g['review_date'].values,
            'mu_p':           float(g['mu_p'].iloc[0]),
            'sigma_p':        float(g['sigma_p'].iloc[0]),
        })
    if verbose:
        print(f"Built {len(records)} product records.")
        print(f"Total panel rows: {sum(len(r['series_norm']) for r in records):,}")
    return records, asin_to_id, n_products, weight_decay_scale


def build_or_load_records(monthly_panel=None, force_rerun=False, verbose=True):
    """Build records or load from cache. Returns (records, asin_to_id, n_products, weight_decay_scale).

    On cache hit, monthly_panel arg is ignored. On cache miss, monthly_panel is required.
    """
    if not force_rerun and task2_records_cache_is_valid():
        t0 = time.time()
        with open(T2_RECORDS_PATH, 'rb') as f:
            records = pickle.load(f)
        asin_to_id = {r['asin']: r['asin_id'] for r in records}
        n_products = max(r['asin_id'] for r in records) + 1
        n_per_product = torch.zeros(n_products, dtype=torch.float32)
        for r in records:
            n_per_product[r['asin_id']] += float(len(r['series_norm']))
        n_per_product = n_per_product.clamp(min=1.0)
        weight_decay_scale = 1.0 / torch.sqrt(n_per_product)
        if verbose:
            print(f"Loaded {len(records)} product records from cache ({format_elapsed(time.time() - t0)})")
            print(f"Total panel rows: {sum(len(r['series_norm']) for r in records):,}")
        return records, asin_to_id, n_products, weight_decay_scale

    if monthly_panel is None:
        raise ValueError("Cache miss: monthly_panel is required to build records.")

    t0 = time.time()
    records, asin_to_id, n_products, weight_decay_scale = build_records_from_panel(
        monthly_panel, verbose=verbose)
    atomic_to_pickle(records, T2_RECORDS_PATH)

    config = {
        'version': T2_DATAPREP_VERSION,
        'alpha': ALPHA, 'beta': BETA,
        'embed_dim': EMBED_DIM,
        'rare_threshold_months': RARE_THRESHOLD_MONTHS,
        'lambda_l2': LAMBDA_L2,
        't2_panel_meta': read_json_or_none(T2_PANEL_MONTHLY_META_PATH),
    }
    atomic_write_text(T2_RECORDS_META_PATH, json.dumps({'t2_dataprep_config': config}, indent=2))
    if verbose:
        print(f"Data prep elapsed: {format_elapsed(time.time() - t0)}")
    return records, asin_to_id, n_products, weight_decay_scale
