"""
Balanced helpfulness dataset for NLP modeling (Sec 7).

Downsamples each helpfulness class (vote==0 vs vote>0) to the smaller class size,
producing a 50/50 binary classification dataset. Separate from the time-series
pipeline - used only by the helpfulness NLP experiment.
"""

import json

import pandas as pd
from sklearn.utils import resample

from ..config.hyperparams import RANDOM_STATE
from ..config.paths import (
    CLEAN_CACHE_META_PATH,
    T1_NLP_META_PATH,
    T1_NLP_TRAIN_PATH,
)
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none


T1_NLP_VERSION = 't1_nlp'


def _build_config():
    return {
        'version': T1_NLP_VERSION,
        'random_state': RANDOM_STATE,
        'clean_cache_meta': read_json_or_none(CLEAN_CACHE_META_PATH),
    }


def task1_nlp_cache_is_valid():
    """True iff the NLP train CSV exists."""
    return T1_NLP_TRAIN_PATH.exists()


def build_balanced_helpfulness(df, random_state=RANDOM_STATE):
    """Downsample each helpfulness class to balance them 50/50.

    Adds `is_helpful` column to df if missing. Returns the balanced DataFrame.
    """
    if 'is_helpful' not in df.columns:
        df['is_helpful'] = (df['vote'] > 0).astype(int)

    df_class_0 = df[df['is_helpful'] == 0]
    df_class_1 = df[df['is_helpful'] == 1]

    print(f"Original Class 0 (0 votes): {len(df_class_0)}")
    print(f"Original Class 1 (>0 votes): {len(df_class_1)}")

    if len(df_class_0) == 0 or len(df_class_1) == 0:
        raise ValueError("Both helpfulness classes are required for balanced NLP training.")

    n_balanced = min(len(df_class_0), len(df_class_1))

    df_class_0_down = resample(df_class_0, replace=False, n_samples=n_balanced,
                               random_state=random_state)
    df_class_1_down = resample(df_class_1, replace=False, n_samples=n_balanced,
                               random_state=random_state)

    df_nlp_train = (
        pd.concat([df_class_0_down, df_class_1_down])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return df_nlp_train


def build_or_load(df, random_state=RANDOM_STATE, force_rerun=False):
    """Build the balanced NLP dataset, or load from cache if available."""
    # Ensure the source df has `is_helpful` so the caller can use it downstream
    if 'is_helpful' not in df.columns:
        df['is_helpful'] = (df['vote'] > 0).astype(int)

    if not force_rerun and task1_nlp_cache_is_valid():
        df_nlp_train = pd.read_csv(T1_NLP_TRAIN_PATH)
        for date_col in ['review_date', 'review_day']:
            if date_col in df_nlp_train.columns:
                df_nlp_train[date_col] = pd.to_datetime(df_nlp_train[date_col])
        return df_nlp_train

    df_nlp_train = build_balanced_helpfulness(df, random_state=random_state)
    atomic_to_csv(df_nlp_train, T1_NLP_TRAIN_PATH, index=False)
    atomic_write_text(T1_NLP_META_PATH, json.dumps({'t1_nlp_config': _build_config()}, indent=2))
    return df_nlp_train
