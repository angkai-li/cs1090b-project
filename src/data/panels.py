"""
Product-level panels: weekly rating-based panel for popular products (Sec 6).

The weekly rating panel feeds into the EDA visualization in Sec 6. The Task 2
DeBERTa panels (monthly + weekly with auxiliary labels) are built later in
src/task2/records.py - they require the DeBERTa scores.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from ..config.paths import (
    CLEAN_CACHE_META_PATH,
    FIG_EDA,
    T1_RATING_WEEKLY_PATH,
    T1_RATING_WEEKLY_META_PATH,
)
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none
from ..utils.metrics import weighted_mean_with_fallback


T1_RATING_WEEKLY_VERSION = 't1_rating_weekly'
RATING_WEEKLY_MIN_REVIEWS_DEFAULT = 500


def _build_config(min_reviews):
    return {
        'version': T1_RATING_WEEKLY_VERSION,
        'min_reviews': min_reviews,
        'freq': 'W',
        'clean_cache_meta': read_json_or_none(CLEAN_CACHE_META_PATH),
    }


def task1_rating_weekly_cache_is_valid():
    """True iff the weekly rating panel CSV exists."""
    return T1_RATING_WEEKLY_PATH.exists()


def build_rating_weekly_panel(df, min_reviews=RATING_WEEKLY_MIN_REVIEWS_DEFAULT):
    """Build a weekly rating-based panel for products with >= min_reviews reviews.

    Returns (panel, popular_products_index, df_popular).
      - panel: DataFrame with (asin, review_date, rating_sentiment_weighted,
               rating_sentiment_naive, num_reviews)
      - popular_products_index: pd.Index of selected asins
      - df_popular: subset of df for the popular products only
    """
    product_counts = df['asin'].value_counts()
    popular_products = product_counts[product_counts >= min_reviews].index

    if len(popular_products) == 0:
        raise ValueError(f"No products meet min_reviews={min_reviews}. Lower the threshold.")

    df_popular = df[df['asin'].isin(popular_products)].copy()

    weighted_panel = (
        df_popular
        .groupby(['asin', pd.Grouper(key='review_date', freq='W')])
        .apply(lambda g: weighted_mean_with_fallback(g, 'sentiment'))
        .reset_index(name='rating_sentiment_weighted')
    )

    naive_panel = (
        df_popular
        .groupby(['asin', pd.Grouper(key='review_date', freq='W')])['sentiment']
        .mean()
        .reset_index(name='rating_sentiment_naive')
    )

    count_panel = (
        df_popular
        .groupby(['asin', pd.Grouper(key='review_date', freq='W')])
        .size()
        .reset_index(name='num_reviews')
    )

    panel = (
        weighted_panel
        .merge(naive_panel, on=['asin', 'review_date'], how='outer')
        .merge(count_panel, on=['asin', 'review_date'], how='outer')
        .sort_values(['asin', 'review_date'])
        .reset_index(drop=True)
    )
    panel = panel[panel['num_reviews'] > 0].copy()
    return panel, popular_products, df_popular


def build_or_load_rating_weekly(df, min_reviews=RATING_WEEKLY_MIN_REVIEWS_DEFAULT,
                                force_rerun=False):
    """Build weekly rating panel or load from cache.

    Returns (panel, popular_products, df_popular).
    """
    if not force_rerun and task1_rating_weekly_cache_is_valid():
        panel = pd.read_csv(T1_RATING_WEEKLY_PATH)
        panel['review_date'] = pd.to_datetime(panel['review_date'])
        popular_products = pd.Index(panel['asin'].drop_duplicates())
        df_popular = df[df['asin'].isin(popular_products)].copy()
        return panel, popular_products, df_popular

    panel, popular_products, df_popular = build_rating_weekly_panel(df, min_reviews=min_reviews)
    atomic_to_csv(panel, T1_RATING_WEEKLY_PATH, index=False)
    atomic_write_text(
        T1_RATING_WEEKLY_META_PATH,
        json.dumps({'t1_rating_weekly_config': _build_config(min_reviews)}, indent=2),
    )
    return panel, popular_products, df_popular


def plot_weekly_sentiment_one_product(panel, asin, save_path=None, show=True):
    """Visualize one product's weekly sentiment trajectory (naive + weighted + volume).

    Saves to FIG_EDA/'weekly_sentiment_top_product.png' by default.
    """
    sub = panel[panel['asin'] == asin]
    fig = plt.figure(figsize=(14, 6))
    plt.plot(
        sub['review_date'], sub['rating_sentiment_naive'],
        label='Naive Sentiment (Simple Average)',
        color='gray', alpha=0.6, linestyle='--',
    )
    plt.plot(
        sub['review_date'], sub['rating_sentiment_weighted'],
        label='Weighted Sentiment (Informativeness Adjusted)',
        color='blue', linewidth=2,
    )
    plt.bar(
        sub['review_date'],
        sub['num_reviews'] / sub['num_reviews'].max(),
        alpha=0.2, label='Review Volume (scaled)',
    )
    plt.title(f'Weekly Sentiment Time Series for Product {asin}')
    plt.xlabel('Date')
    plt.ylabel('Sentiment [-1, 1]')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIG_EDA / 'weekly_sentiment_top_product.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
