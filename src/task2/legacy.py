"""
Task 2 Sec 9.2 legacy baseline: multivariate PELT(rbf) on 7-feature panel.

7 features: task1_score_0_1, amazon_sentiment_0_1, star_rating, review_count,
            verified_ratio, task1_score_variance, helpful_vote_mean.

Kept as an ablation comparator to the cleaner univariate Route 1 (PELT-L2) in Sec 9.4.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config.paths import (
    CLEAN_CACHE_META_PATH,
    T2_LEGACY_CP_PATH, T2_LEGACY_META_PATH, T2_LEGACY_PANEL_PATH,
)
from ..config.runtime import HAS_RUPTURES
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none


T2_LEGACY_VERSION = 't2_legacy'
T2_LEGACY_FEATURES = [
    'avg_task1_score_0_1', 'avg_amazon_sentiment_0_1', 'avg_star_rating',
    'review_count', 'verified_ratio', 'task1_score_variance', 'helpful_vote_mean',
]
LEGACY_MIN_PERIODS_PER_PRODUCT = 6
LEGACY_MIN_SEGMENT_SIZE = 2
LEGACY_PEN = 3.0


def _pick_first_existing(frame, candidates):
    return next((col for col in candidates if col in frame.columns), None)


def _to_unit_interval(series):
    """Min-max normalize to [0, 1]. NaN handling: returns 0.5 for degenerate cases."""
    series = pd.to_numeric(series, errors='coerce')
    if series.dropna().empty:
        return series
    smin, smax = series.min(), series.max()
    if pd.notna(smin) and pd.notna(smax) and 0 <= smin <= smax <= 1:
        return series
    if pd.isna(smin) or pd.isna(smax) or smin == smax:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - smin) / (smax - smin)


def _sentiment_band(score):
    if pd.isna(score): return 'Unknown'
    if score < 0.3:    return 'Low'
    if score < 0.7:    return 'Medium'
    return 'High'


def _rating_band(rating):
    if pd.isna(rating): return 'Unknown'
    if rating < 2.5:    return 'Low'
    if rating < 4.0:    return 'Medium'
    return 'High'


def task2_legacy_cache_is_valid():
    return T2_LEGACY_CP_PATH.exists() and T2_LEGACY_PANEL_PATH.exists()


def build_legacy_monthly_panel(df):
    """Aggregate per-review df to a (asin, month) panel with 7 multivariate features."""
    df_in = df.copy()

    score_col = _pick_first_existing(
        df_in, ['score_deberta_v3_base_lora', 'score_tfidf_ridge', 'score_vader', 'sentiment'])
    if score_col is None:
        raise RuntimeError(
            "Legacy Task 2 baseline needs one of: score_deberta_v3_base_lora, "
            "score_tfidf_ridge, score_vader, sentiment.")
    print(f"Legacy Task 2 baseline using score column: {score_col}")

    df_in['task1_score_raw'] = pd.to_numeric(df_in[score_col], errors='coerce')
    df_in['task1_score_0_1'] = _to_unit_interval(df_in['task1_score_raw'])
    df_in['amazon_sentiment_raw'] = pd.to_numeric(df_in['sentiment'], errors='coerce')
    df_in['amazon_sentiment_0_1'] = _to_unit_interval(df_in['amazon_sentiment_raw'])
    df_in['rating_numeric'] = pd.to_numeric(df_in['overall'], errors='coerce')
    df_in['verified_numeric'] = df_in['verified'].astype(float)
    df_in['helpful_numeric'] = pd.to_numeric(df_in['vote'], errors='coerce')
    df_in['time_period'] = df_in['review_date'].dt.to_period('M').dt.to_timestamp()

    panel = (
        df_in.groupby(['asin', 'time_period'])
        .agg(
            avg_task1_score_0_1=('task1_score_0_1', 'mean'),
            avg_amazon_sentiment_0_1=('amazon_sentiment_0_1', 'mean'),
            avg_star_rating=('rating_numeric', 'mean'),
            review_count=('task1_score_raw', 'size'),
            verified_ratio=('verified_numeric', 'mean'),
            task1_score_variance=('task1_score_raw', 'var'),
            helpful_vote_mean=('helpful_numeric', 'mean'),
        )
        .reset_index()
    )
    for col in ['verified_ratio', 'task1_score_variance', 'helpful_vote_mean']:
        panel[col] = panel[col].fillna(0.0)
    return panel


def segment_with_rbf_pelt(panel, features=T2_LEGACY_FEATURES,
                          min_periods=LEGACY_MIN_PERIODS_PER_PRODUCT,
                          min_segment_size=LEGACY_MIN_SEGMENT_SIZE,
                          penalty=LEGACY_PEN):
    """Run multivariate PELT(rbf) per product. Returns segments DataFrame."""
    if not HAS_RUPTURES:
        raise ImportError("Task 2 baseline requires `ruptures`. Install: pip install ruptures==1.1.9")
    import ruptures as rpt

    rows = []
    for asin, group in panel.groupby('asin'):
        group = group.sort_values('time_period').reset_index(drop=True)
        if len(group) < min_periods:
            continue
        X = group[features].replace([np.inf, -np.inf], np.nan)
        usable_cols = [col for col in features if X[col].notna().any()]
        if not usable_cols:
            change_points = [len(group)]
        else:
            X = X[usable_cols].fillna(X[usable_cols].median(numeric_only=True)).fillna(0.0)
            X_scaled = StandardScaler().fit_transform(X)
            try:
                model = rpt.Pelt(model='rbf', min_size=min_segment_size).fit(X_scaled)
                change_points = model.predict(pen=penalty)
            except Exception:
                change_points = [len(group)]

        start_idx, phase_id = 0, 1
        for end_idx in change_points:
            segment = group.iloc[start_idx:end_idx].copy()
            if segment.empty:
                continue
            task1_score = float(np.clip(segment['avg_task1_score_0_1'].mean(), 0, 1))
            avg_star = float(segment['avg_star_rating'].mean()) if segment['avg_star_rating'].notna().any() else np.nan
            t1_label = _sentiment_band(task1_score)
            amazon_label = _rating_band(avg_star)
            rows.append({
                'asin': asin, 'phase_id': phase_id,
                'start_date': segment['time_period'].min(),
                'end_date':   segment['time_period'].max(),
                'n_periods':  len(segment),
                'n_reviews':  int(segment['review_count'].sum()),
                'task1_score_0_1_mean': task1_score,
                'avg_star_rating': avg_star,
                'verified_ratio_mean': float(segment['verified_ratio'].mean()),
                'helpful_vote_mean':  float(segment['helpful_vote_mean'].mean()),
                'task1_phase_label':  t1_label,
                'amazon_phase_label': amazon_label,
                'combined_phase_label': f'{t1_label} Task1 / {amazon_label} Amazon rating phase',
                'change_point_end_index': int(end_idx),
            })
            start_idx = end_idx
            phase_id += 1
    return pd.DataFrame(rows)


def run_legacy_baseline(df, force_rerun=False, verbose=True):
    """Run the legacy rbf-PELT baseline, or load from cache.

    Returns (panel, segments) DataFrames.
    """
    if not force_rerun and task2_legacy_cache_is_valid():
        segments = pd.read_csv(T2_LEGACY_CP_PATH)
        if verbose:
            print(f"Loaded Task 2 legacy baseline from cache: {segments.shape}")
        panel = pd.read_csv(T2_LEGACY_PANEL_PATH) if T2_LEGACY_PANEL_PATH.exists() else None
        return panel, segments

    t0 = time.time()
    panel = build_legacy_monthly_panel(df)
    if verbose:
        print(f"Legacy monthly panel shape: {panel.shape}")
    segments = segment_with_rbf_pelt(panel)
    if verbose:
        print(f"Legacy segmented phases: {len(segments)}")

    atomic_to_csv(panel, T2_LEGACY_PANEL_PATH, index=False)
    atomic_to_csv(segments, T2_LEGACY_CP_PATH, index=False)

    config = {
        'version': T2_LEGACY_VERSION,
        'pelt_model': 'rbf', 'pelt_min_size': LEGACY_MIN_SEGMENT_SIZE, 'pelt_pen': LEGACY_PEN,
        'features': ['task1_score_0_1', 'amazon_sentiment_0_1', 'star_rating',
                     'review_count', 'verified_ratio', 'task1_score_variance', 'helpful_vote_mean'],
        'clean_cache_meta': read_json_or_none(CLEAN_CACHE_META_PATH),
    }
    atomic_write_text(T2_LEGACY_META_PATH, json.dumps({'t2_legacy_config': config}, indent=2))
    if verbose:
        print(f"Legacy baseline elapsed: {format_elapsed(time.time() - t0)}")
    return panel, segments
