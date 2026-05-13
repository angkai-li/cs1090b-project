"""
Shared 70/10/20 train/val/test split (stratified by `overall` rating).

Used by ALL Task 1 models (VADER, TF-IDF+Ridge, DeBERTa+LoRA) so the test
set is identical across models, enabling fair MAE/RMSE comparison.

Cached as `split.csv` keyed by df_model row id (row in df after sentiment filter).
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config.hyperparams import RANDOM_STATE
from ..config.paths import (
    CLEAN_CACHE_META_PATH,
    T1_SPLIT_PATH,
    T1_SPLIT_META_PATH,
)
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none


T1_SPLIT_VERSION = 't1_split'
T1_VAL_SIZE = 0.10
T1_TEST_SIZE = 0.20


def filter_to_modeling_subset(df):
    """Keep only rows with valid sentiment + non-trivial text. Returns df_model
    with a stable `df_row_id` column referencing the original df index.
    """
    df_model = df[['text', 'overall', 'sentiment']].copy()
    df_model = df_model[
        df_model['sentiment'].notna()
        & df_model['overall'].notna()
        & (df_model['text'].str.len() > 1)
    ].copy()
    df_model = df_model.reset_index(drop=False).rename(columns={'index': 'df_row_id'})
    return df_model


def split_cache_is_valid():
    """True iff the cached split.csv exists. File-existence only."""
    return T1_SPLIT_PATH.exists()


def build_or_load_split(df_model, val_size=T1_VAL_SIZE, test_size=T1_TEST_SIZE,
                       random_state=RANDOM_STATE, force_rerun=False):
    """Get the cached 70/10/20 split, or build it from scratch.

    Args:
        df_model: DataFrame filtered to modeling rows (see filter_to_modeling_subset)
        val_size: fraction of all data going to val (default 0.10)
        test_size: fraction of all data going to test (default 0.20)
        random_state: split seed (default RANDOM_STATE)
        force_rerun: ignore cache and rebuild

    Returns:
        DataFrame with columns ['df_model_row_id', 'split'] where split in {train, val, test}
    """
    if not force_rerun and split_cache_is_valid():
        return pd.read_csv(T1_SPLIT_PATH)

    stratify_labels = df_model['overall']
    if stratify_labels.value_counts().min() < 2:
        stratify_labels = None

    train_val_idx, test_idx = train_test_split(
        df_model.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    train_val_stratify = df_model.loc[train_val_idx, 'overall']
    if train_val_stratify.value_counts().min() < 2:
        train_val_stratify = None

    val_fraction_of_train_val = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        random_state=random_state,
        stratify=train_val_stratify,
    )

    split_df = pd.DataFrame({
        'df_model_row_id': np.concatenate([train_idx, val_idx, test_idx]),
        'split': ['train'] * len(train_idx) + ['val'] * len(val_idx) + ['test'] * len(test_idx),
    }).sort_values('df_model_row_id').reset_index(drop=True)

    split_config = {
        'version': T1_SPLIT_VERSION,
        'random_state': random_state,
        'val_size': val_size,
        'test_size': test_size,
        'rows': int(len(df_model)),
        'clean_cache_meta': read_json_or_none(CLEAN_CACHE_META_PATH),
    }
    atomic_to_csv(split_df, T1_SPLIT_PATH, index=False)
    atomic_write_text(T1_SPLIT_META_PATH, json.dumps({'t1_split_config': split_config}, indent=2))
    return split_df


def apply_split(df_model, split_df):
    """Apply a split DataFrame to df_model, returning (train_df, val_df, test_df).

    Each returned DataFrame has columns ['text', 'overall', 'sentiment', 'df_row_id'].
    """
    train_idx = split_df.loc[split_df['split'] == 'train', 'df_model_row_id'].to_numpy()
    val_idx   = split_df.loc[split_df['split'] == 'val',   'df_model_row_id'].to_numpy()
    test_idx  = split_df.loc[split_df['split'] == 'test',  'df_model_row_id'].to_numpy()

    train_df = df_model.loc[train_idx].copy()
    val_df   = df_model.loc[val_idx].copy()
    test_df  = df_model.loc[test_idx].copy()
    return train_df, val_df, test_df
