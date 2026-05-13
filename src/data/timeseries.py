"""
Rating-based daily sentiment time series (Sec 5).

Uses `overall` ratings transformed to [-1, 1] as the sentiment proxy.
Aggregated per day with both naive average and weighted average.
"""

import json

import pandas as pd

from ..config.paths import (
    CLEAN_CACHE_META_PATH,
    T1_DAILY_DIR,
    T1_DAILY_RATING_PATH,
)
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none
from ..utils.metrics import weighted_mean_with_fallback


T1_RATING_DAILY_VERSION = 't1_rating_daily'


def _build_config():
    return {
        'version': T1_RATING_DAILY_VERSION,
        'clean_cache_meta': read_json_or_none(CLEAN_CACHE_META_PATH),
    }


def task1_rating_daily_cache_is_valid():
    """True iff the daily rating CSV exists. File-existence only - set
    `force_rerun=True` in build_or_load() to override."""
    return T1_DAILY_RATING_PATH.exists()


def build_daily_rating_series(df):
    """Aggregate the cleaned df to a daily rating-based sentiment time series.

    Returns a DataFrame with columns:
      review_day, rating_sentiment_naive, rating_sentiment_weighted, num_reviews
    """
    daily_naive = (
        df.groupby('review_day')['sentiment']
        .mean()
        .reset_index(name='rating_sentiment_naive')
    )

    daily_weighted = (
        df.groupby('review_day')[['sentiment', 'informativeness_log']]
        .apply(lambda g: weighted_mean_with_fallback(g, 'sentiment'))
        .reset_index(name='rating_sentiment_weighted')
    )

    daily_counts = (
        df.groupby('review_day')
        .size()
        .reset_index(name='num_reviews')
    )

    return (
        daily_naive
        .merge(daily_weighted, on='review_day')
        .merge(daily_counts, on='review_day')
        .sort_values('review_day')
        .reset_index(drop=True)
    )


def build_or_load(df, force_rerun=False):
    """Build the daily rating-based series, or load from cache if available.

    Returns the time-series DataFrame. On cache hit, df is not used.
    """
    if not force_rerun and task1_rating_daily_cache_is_valid():
        ts = pd.read_csv(T1_DAILY_RATING_PATH)
        ts['review_day'] = pd.to_datetime(ts['review_day'])
        return ts

    ts = build_daily_rating_series(df)
    atomic_to_csv(ts, T1_DAILY_RATING_PATH, index=False)
    atomic_write_text(T1_DAILY_DIR / 'rating.meta.json',
                      json.dumps({'t1_rating_daily_config': _build_config()}, indent=2))
    return ts
