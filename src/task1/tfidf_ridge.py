"""
TF-IDF + Ridge regression baseline for Task 1.

Pipeline:
  TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
    -> Ridge(alpha=1.0)

Trained on the shared 70/10/20 split. Predictions clipped to [-1, +1] to match
the sentiment scale.
"""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from ..config.hyperparams import SENTIMENT_MIN, SENTIMENT_MAX
from ..config.paths import T1_RIDGE_MODEL_PATH
from ..utils.io import atomic_to_pickle
from ..utils.metrics import rmse


DEFAULT_CONFIG = {
    'tfidf_max_features': 10000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_sublinear_tf': True,
    'ridge_alpha': 1.0,
}


def build_pipeline(config=None):
    """Construct an untrained TF-IDF + Ridge sklearn Pipeline."""
    if config is None:
        config = DEFAULT_CONFIG
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=config['tfidf_max_features'],
            ngram_range=tuple(config['tfidf_ngram_range']),
            sublinear_tf=config['tfidf_sublinear_tf'],
        )),
        ('regressor', Ridge(alpha=config['ridge_alpha'])),
    ])


def train(X_train, y_train, config=None, verbose=True):
    """Fit a fresh Ridge pipeline. Returns the trained pipeline."""
    pipeline = build_pipeline(config)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    if verbose:
        from ..utils.format import format_elapsed
        print("Ridge training elapsed:", format_elapsed(time.perf_counter() - t0))
    return pipeline


def ridge_cache_is_valid():
    """True iff the trained Ridge pickle exists."""
    return T1_RIDGE_MODEL_PATH.exists()


def load_cached_pipeline():
    """Load a previously-trained Ridge pipeline from disk."""
    with open(T1_RIDGE_MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def train_or_load(X_train, y_train, config=None, force_rerun=False, verbose=True):
    """Train or load the Ridge pipeline. Returns (pipeline, came_from_cache: bool)."""
    if not force_rerun and ridge_cache_is_valid():
        return load_cached_pipeline(), True
    pipeline = train(X_train, y_train, config, verbose=verbose)
    atomic_to_pickle(pipeline, T1_RIDGE_MODEL_PATH)
    if verbose:
        print(f"Saved Ridge baseline model to {T1_RIDGE_MODEL_PATH}")
    return pipeline, False


def score_clipped(pipeline, texts):
    """Predict + clip to [-1, +1]. Returns numpy array."""
    preds = pipeline.predict(texts)
    return np.clip(preds, SENTIMENT_MIN, SENTIMENT_MAX)


def evaluate_on_test(pipeline, test_df):
    """Compute MAE/RMSE on the test split.

    Returns dict with keys: preds, mae, rmse.
    """
    preds = score_clipped(pipeline, test_df['text'])
    y_true = test_df['sentiment']
    return {
        'preds': preds,
        'mae': float(mean_absolute_error(y_true, preds)),
        'rmse': float(rmse(y_true, preds)),
    }
