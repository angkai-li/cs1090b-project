"""
Cleaning + feature engineering for Amazon Cell Phone reviews.

Pipeline:
  1. Keep 8 raw cols, drop reviews without text
  2. Numeric coercion for `vote` (handles "1,234" string format)
  3. Composite `review_weight` = 1 + log1p(vote_capped) + ALPHA_VERIFIED * verified
  4. Parse `unixReviewTime` -> `review_date` + `review_day`
  5. Map `overall` in {1..5} -> `sentiment` in [-1, +1] via (overall-3)/2
  6. Build `text` = summary + ". " + reviewText
  7. Sort by (asin, review_date)
"""

import json

import numpy as np
import pandas as pd

from ..config.hyperparams import CLEANING_VERSION, ALPHA_VERIFIED
from ..config.paths import CLEAN_CACHE_PATH, CLEAN_CACHE_META_PATH
from ..utils.io import (
    atomic_to_pickle,
    atomic_write_text,
    raw_file_signature,
)
from .load import clean_cache_is_valid, load_json_lines


_REQUIRED_COLS = [
    'overall', 'reviewText', 'summary', 'unixReviewTime',
    'vote', 'image', 'verified', 'asin',
]


def clean_and_engineer(df_raw):
    """Run the full cleaning + feature-engineering pipeline.

    Args:
        df_raw: DataFrame with the 8 required Amazon-review columns.
    Returns:
        Cleaned DataFrame with derived columns (review_weight, sentiment, text, ...).
    """
    missing_cols = set(_REQUIRED_COLS).difference(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Missing expected raw columns: {sorted(missing_cols)}")

    df = df_raw[_REQUIRED_COLS].copy()
    df = df.dropna(subset=['reviewText']).copy()
    df['summary'] = df['summary'].fillna('')

    df['vote'] = pd.to_numeric(
        df['vote'].fillna(0).astype(str).str.replace(',', '', regex=False),
        errors='coerce',
    ).fillna(0).astype(int)

    df['informativeness'] = df['vote']
    df['informativeness_log'] = np.log1p(df['vote'])

    # === Composite review weight ===
    # 75% of reviews have vote=0, so a pure-log1p(vote) weight collapses to zero
    # for 3/4 of data and triggers fallback equal-weighting. Composite weight
    # = 1 (baseline) + log1p(vote_capped) + ALPHA_VERIFIED * verified ensures
    # every review participates and credits verified-buyer reviews (Mudambi &
    # Schuff MISQ 2010, Ghose & Ipeirotis IEEE TKDE 2011).
    df['vote_capped'] = df['vote'].clip(upper=df['vote'].quantile(0.99))
    df['verified'] = df['verified'].fillna(False).astype(bool)
    df['review_weight'] = (
        1.0
        + np.log1p(df['vote_capped'])
        + ALPHA_VERIFIED * df['verified'].astype(float)
    )

    df['has_image'] = df['image'].notna().astype(int)
    df = df.drop(columns=['image'])

    df['review_date'] = pd.to_datetime(df['unixReviewTime'], unit='s')
    df['review_day'] = df['review_date'].dt.floor('D')

    df['sentiment'] = (df['overall'] - 3) / 2

    df['text'] = (
        df['summary'].fillna('').astype(str).str.strip()
        + '. '
        + df['reviewText'].fillna('').astype(str).str.strip()
    ).str.strip()

    df = df.sort_values(['asin', 'review_date']).reset_index(drop=True)
    return df


def save_clean_cache(df, raw_path):
    """Save the cleaned df + meta.json to CLEAN_CACHE_PATH (atomic)."""
    cache_meta = {
        'cleaning_version': CLEANING_VERSION,
        'raw_signature': raw_file_signature(raw_path),
        'rows': int(len(df)),
        'columns': df.columns.tolist(),
    }
    atomic_to_pickle(df, CLEAN_CACHE_PATH)
    atomic_write_text(CLEAN_CACHE_META_PATH, json.dumps(cache_meta, indent=2))


def load_or_clean(raw_path):
    """Load cleaned df from cache if valid, else build it from raw and cache it.

    Returns the cleaned DataFrame.
    """
    if clean_cache_is_valid(raw_path):
        return pd.read_pickle(CLEAN_CACHE_PATH)

    df_raw = load_json_lines(raw_path)
    df = clean_and_engineer(df_raw)
    save_clean_cache(df, raw_path)
    return df
