"""
Numerical metrics: RMSE, weighted average, correlation report, etc.

These are general-purpose math helpers, kept separate from format.py
(text/table formatting) so plot code can import metrics without pulling in
table-rendering logic.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def weighted_mean_with_fallback(group, value_col, weight_col=None):
    """Weighted mean of `value_col` weighted by `weight_col`, with safe fallbacks.

    Default weight column is `review_weight` (added during Sec 2 cleaning). Falls back
    to `informativeness_log` for backward compatibility.

    If weights sum to zero or all values are NaN, falls back to the unweighted
    mean (or returns NaN if no valid values).
    """
    if weight_col is None:
        if 'review_weight' in group.columns:
            weight_col = 'review_weight'
        else:
            weight_col = 'informativeness_log'
    values = group[value_col].astype(float)
    weights = group[weight_col].fillna(0).astype(float)

    valid = values.notna() & weights.notna()
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan
    if weights.sum() <= 0:
        return values.mean()

    return np.average(values, weights=weights)


def correlation_report(x, y):
    """Pearson + Spearman correlations between two arrays.

    Returns a dict with: n, pearson_r, pearson_p, spearman_r, spearman_p.
    Skips NaN pairs.
    """
    from scipy import stats
    mask = ~(pd.isna(x) | pd.isna(y))
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 2:
        return {'n': len(x), 'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {'n': int(len(x)),
            'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp)}


def pvalue_text(value):
    """Format a p-value for display: <0.001, <0.01, or 3-decimal."""
    if value is None or pd.isna(value):
        return '-'
    if value < 0.001:
        return '<0.001'
    if value < 0.01:
        return '<0.01'
    return f'{value:.3f}'
