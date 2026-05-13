"""
VADER (rule-based sentiment) baseline for Task 1.

Lightweight: no training. Just `vaderSentiment.SentimentIntensityAnalyzer.polarity_scores()['compound']`.
The compound score is already on [-1, +1] which matches our sentiment scale, so
no scaling needed.

Optional: skipped if `vaderSentiment` package isn't installed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ..utils.metrics import rmse


def _make_analyzer():
    """Lazy import of VADER (skipped if not installed)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def make_score_function():
    """Return a callable that scores a single text. Handles errors -> NaN."""
    analyzer = _make_analyzer()

    def vader_score(text):
        try:
            return analyzer.polarity_scores(str(text))['compound']
        except Exception:
            return np.nan

    return vader_score


def score_corpus(texts):
    """Score a pandas Series of texts. Returns a numpy array of scores."""
    fn = make_score_function()
    return texts.apply(fn).to_numpy()


def evaluate_on_test(test_df):
    """Run VADER on the test split and return MAE/RMSE + the predictions.

    Args:
        test_df: DataFrame with columns 'text' and 'sentiment' (the true label).
    Returns:
        dict with keys: preds, valid_mask, mae, rmse.
    """
    fn = make_score_function()
    preds = test_df['text'].apply(fn).to_numpy()
    valid = ~np.isnan(preds)
    y_true = test_df['sentiment'].to_numpy()
    mae_val = mean_absolute_error(y_true[valid], preds[valid])
    rmse_val = rmse(y_true[valid], preds[valid])
    return {
        'preds': preds,
        'valid_mask': valid,
        'mae': float(mae_val),
        'rmse': float(rmse_val),
    }
