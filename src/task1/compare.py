"""
Task 1 model time-series comparison + visualization (Sec 8.3).

Compares daily Ridge vs DeBERTa scores: correlations, differences, robustness
to low-volume days. Produces 4 main report plots + 4 diagnostic plots.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.hyperparams import SENTIMENT_MIN, SENTIMENT_MAX
from ..config.paths import (
    FIG_T1_COMPARE_DIAG,
    FIG_T1_COMPARE_MAIN,
    T1_DAILY_DEBERTA_PATH,
    T1_DAILY_BASELINE_COMBINED_PATH,
    T1_DAILY_MODEL_COMPARISON_PATH,
    T1_DISAGREE_HIGH_VOL_PATH,
    T1_DISAGREE_TOP_PATH,
    T1_DISAGREE_VOL_SUMMARY_PATH,
    T1_KEY_FINDINGS_PATH,
    T1_MONTHLY_COMPARISON_PATH,
    T1_PLOT_MANIFEST_PATH,
    T1_SUMMARY_BY_VARIANT_PATH,
    T1_VOLUME_ROBUSTNESS_PATH,
)
from ..utils.format import display_path, format_console_grid
from ..utils.io import atomic_to_csv


def load_daily_series(baseline_path=None, deberta_path=None):
    """Load + validate baseline daily + DeBERTa daily score tables."""
    baseline_path = baseline_path or T1_DAILY_BASELINE_COMBINED_PATH
    deberta_path = deberta_path or T1_DAILY_DEBERTA_PATH
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline daily file: {baseline_path}")
    if not deberta_path.exists():
        raise FileNotFoundError(f"Missing DeBERTa daily file: {deberta_path}")

    daily_baseline = pd.read_csv(baseline_path)
    daily_deberta = pd.read_csv(deberta_path)
    for name, frame in [('baseline', daily_baseline), ('deberta', daily_deberta)]:
        if 'review_day' not in frame.columns:
            raise ValueError(f"Daily {name} table missing review_day column.")
        frame['review_day'] = pd.to_datetime(frame['review_day'])
    return daily_baseline, daily_deberta


def build_comparison_frame(daily_baseline, daily_deberta):
    """Merge baseline + DeBERTa daily tables, compute diffs + calendar features."""
    required_baseline_cols = [
        'review_day', 'score_tfidf_ridge_naive', 'score_tfidf_ridge_weighted'
    ]
    required_deberta_cols = [
        'review_day', 'score_deberta_v3_base_lora_naive',
        'score_deberta_v3_base_lora_weighted', 'num_reviews',
    ]
    missing_b = [c for c in required_baseline_cols if c not in daily_baseline.columns]
    missing_d = [c for c in required_deberta_cols if c not in daily_deberta.columns]
    if missing_b:
        raise ValueError(f"Baseline daily missing columns: {missing_b}")
    if missing_d:
        raise ValueError(f"DeBERTa daily missing columns: {missing_d}")

    ts_compare = daily_baseline[required_baseline_cols].merge(
        daily_deberta[required_deberta_cols], on='review_day', how='inner'
    ).sort_values('review_day')

    comparison_cols = [
        'score_tfidf_ridge_naive', 'score_tfidf_ridge_weighted',
        'score_deberta_v3_base_lora_naive', 'score_deberta_v3_base_lora_weighted',
        'num_reviews',
    ]
    for col in comparison_cols:
        ts_compare[col] = pd.to_numeric(ts_compare[col], errors='coerce')
    ts_compare = ts_compare.dropna(subset=comparison_cols).reset_index(drop=True)

    if ts_compare.empty:
        raise ValueError("No overlapping daily rows between baseline and DeBERTa.")

    for variant in ['naive', 'weighted']:
        ridge_col = f'score_tfidf_ridge_{variant}'
        deberta_col = f'score_deberta_v3_base_lora_{variant}'
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        ts_compare[diff_col] = ts_compare[deberta_col] - ts_compare[ridge_col]
        ts_compare[f'abs_diff_{variant}'] = ts_compare[diff_col].abs()
        ts_compare[f'mean_score_{variant}'] = (ts_compare[deberta_col] + ts_compare[ridge_col]) / 2

    ts_compare['year'] = ts_compare['review_day'].dt.year
    ts_compare['month'] = ts_compare['review_day'].dt.to_period('M').dt.to_timestamp()
    return ts_compare


def _correlation_report(x, y):
    try:
        from scipy import stats
    except ImportError:
        stats = None

    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, int(len(x))
    if stats is not None:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
    else:
        pearson_r = x.corr(y, method='pearson')
        spearman_r = x.corr(y, method='spearman')
        pearson_p = np.nan
        spearman_p = np.nan
    return pearson_r, pearson_p, spearman_r, spearman_p, int(len(x))


def _summary_row(frame, variant, group_label, min_reviews=None):
    ridge_col = f'score_tfidf_ridge_{variant}'
    deberta_col = f'score_deberta_v3_base_lora_{variant}'
    diff_col = f'diff_deberta_minus_ridge_{variant}'
    abs_diff_col = f'abs_diff_{variant}'
    pearson_r, pearson_p, spearman_r, spearman_p, n_corr = _correlation_report(
        frame[ridge_col], frame[deberta_col])
    diff = frame[diff_col]
    abs_diff = frame[abs_diff_col]
    return {
        'Group': group_label,
        'Min Reviews': '-' if min_reviews is None else int(min_reviews),
        'Variant': variant,
        'Days': int(len(frame)),
        'Correlation N': n_corr,
        'Total Reviews': int(frame['num_reviews'].sum()) if len(frame) else 0,
        'Median Reviews/Day': frame['num_reviews'].median() if len(frame) else np.nan,
        'Pearson r': pearson_r,
        'Pearson p': pearson_p,
        'Spearman r': spearman_r,
        'Spearman p': spearman_p,
        'Diff Mean': diff.mean(),
        'Diff Median': diff.median(),
        'Diff Std': diff.std(),
        'Mean Abs Diff': abs_diff.mean(),
        'Diff P05': diff.quantile(0.05),
        'Diff P95': diff.quantile(0.95),
        'Diff Min': diff.min(),
        'Diff Max': diff.max(),
        'Max Abs Diff': abs_diff.max(),
    }


def _weighted_average(values, weights):
    values = pd.to_numeric(values, errors='coerce')
    weights = pd.to_numeric(weights, errors='coerce')
    valid = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        return np.nan
    if weights.sum() <= 0:
        return values.mean()
    return np.average(values, weights=weights)


def build_summary_tables(ts_compare, high_volume_threshold=25, top_n=25,
                         volume_thresholds=(5, 10, 25, 50, 100)):
    """Build all summary DataFrames. Returns a dict of named tables."""
    comparison_summary = pd.DataFrame([
        _summary_row(ts_compare, variant, 'all days') for variant in ['naive', 'weighted']
    ])

    volume_summary_rows = []
    for min_r in volume_thresholds:
        subset = ts_compare[ts_compare['num_reviews'] >= min_r].copy()
        if len(subset) == 0:
            continue
        for variant in ['naive', 'weighted']:
            volume_summary_rows.append(
                _summary_row(subset, variant, f'num_reviews >= {min_r}', min_reviews=min_r))
    volume_summary = pd.DataFrame(volume_summary_rows)

    key_findings_rows = []
    for variant in ['naive', 'weighted']:
        overall = comparison_summary.loc[comparison_summary['Variant'] == variant].iloc[0]
        high_vol = volume_summary[
            (volume_summary['Variant'] == variant) &
            (volume_summary['Min Reviews'] == high_volume_threshold)
        ]
        high_vol = high_vol.iloc[0] if len(high_vol) else overall
        key_findings_rows.extend([
            {'Finding': 'Daily model agreement', 'Variant': variant, 'Metric': 'Pearson r',
             'Value': overall['Pearson r'],
             'Interpretation': 'Higher means DeBERTa and Ridge move together over time.'},
            {'Finding': 'Rank-order agreement', 'Variant': variant, 'Metric': 'Spearman r',
             'Value': overall['Spearman r'],
             'Interpretation': 'Higher means the two models rank high/low sentiment days similarly.'},
            {'Finding': 'Average level bias', 'Variant': variant, 'Metric': 'Mean DeBERTa - Ridge',
             'Value': overall['Diff Mean'],
             'Interpretation': 'Near zero means no large systematic level shift.'},
            {'Finding': 'Typical daily disagreement', 'Variant': variant,
             'Metric': 'Mean absolute difference', 'Value': overall['Mean Abs Diff'],
             'Interpretation': 'Average absolute gap between daily DeBERTa and Ridge scores.'},
            {'Finding': 'High-volume robustness', 'Variant': variant,
             'Metric': f'Pearson r, num_reviews >= {high_volume_threshold}',
             'Value': high_vol['Pearson r'],
             'Interpretation': 'Agreement after excluding low-volume days.'},
            {'Finding': 'Largest local disagreement', 'Variant': variant,
             'Metric': 'Max absolute difference', 'Value': overall['Max Abs Diff'],
             'Interpretation': 'Largest single-day model gap, inspect top-disagreement table.'},
        ])
    key_findings = pd.DataFrame(key_findings_rows)

    top_frames = []
    for variant in ['naive', 'weighted']:
        ridge_col = f'score_tfidf_ridge_{variant}'
        deberta_col = f'score_deberta_v3_base_lora_{variant}'
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        abs_diff_col = f'abs_diff_{variant}'
        top = ts_compare.sort_values(abs_diff_col, ascending=False).head(top_n)[[
            'review_day', 'num_reviews', ridge_col, deberta_col, diff_col, abs_diff_col,
        ]].copy().rename(columns={
            ridge_col: 'ridge_score', deberta_col: 'deberta_score',
            diff_col: 'deberta_minus_ridge', abs_diff_col: 'abs_difference',
        })
        top.insert(0, 'variant', variant)
        top_frames.append(top)
    top_disagreement_days = pd.concat(top_frames, ignore_index=True)

    hv_subset = ts_compare[ts_compare['num_reviews'] >= high_volume_threshold].copy()
    hv_frames = []
    for variant in ['naive', 'weighted']:
        if len(hv_subset) == 0:
            continue
        ridge_col = f'score_tfidf_ridge_{variant}'
        deberta_col = f'score_deberta_v3_base_lora_{variant}'
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        abs_diff_col = f'abs_diff_{variant}'
        hv = hv_subset.sort_values(abs_diff_col, ascending=False).head(top_n)[[
            'review_day', 'num_reviews', ridge_col, deberta_col, diff_col, abs_diff_col,
        ]].copy().rename(columns={
            ridge_col: 'ridge_score', deberta_col: 'deberta_score',
            diff_col: 'deberta_minus_ridge', abs_diff_col: 'abs_difference',
        })
        hv.insert(0, 'variant', variant)
        hv_frames.append(hv)
    high_volume_top_disagreement_days = (
        pd.concat(hv_frames, ignore_index=True) if hv_frames
        else pd.DataFrame(columns=top_disagreement_days.columns)
    )

    disagreement_volume_summary = pd.DataFrame([
        {
            'Variant': variant,
            'Top Days': int(len(top_disagreement_days[top_disagreement_days['variant'] == variant])),
            'Median Reviews in Top Disagreements':
                top_disagreement_days[top_disagreement_days['variant'] == variant]['num_reviews'].median(),
            'Share with <5 Reviews':
                (top_disagreement_days[top_disagreement_days['variant'] == variant]['num_reviews'] < 5).mean(),
            'Share with <10 Reviews':
                (top_disagreement_days[top_disagreement_days['variant'] == variant]['num_reviews'] < 10).mean(),
            'Share with <25 Reviews':
                (top_disagreement_days[top_disagreement_days['variant'] == variant]['num_reviews'] < 25).mean(),
            'Max Abs Difference':
                top_disagreement_days[top_disagreement_days['variant'] == variant]['abs_difference'].max(),
        }
        for variant in ['naive', 'weighted']
    ])

    monthly_rows = []
    for month, group in ts_compare.groupby('month', sort=True):
        row = {
            'month': month, 'days': int(group['review_day'].nunique()),
            'num_reviews': int(group['num_reviews'].sum()),
            'median_reviews_per_day': group['num_reviews'].median(),
        }
        for variant in ['naive', 'weighted']:
            ridge_col = f'score_tfidf_ridge_{variant}'
            deberta_col = f'score_deberta_v3_base_lora_{variant}'
            row[f'ridge_{variant}_daily_mean'] = group[ridge_col].mean()
            row[f'deberta_{variant}_daily_mean'] = group[deberta_col].mean()
            row[f'diff_{variant}_daily_mean'] = group[f'diff_deberta_minus_ridge_{variant}'].mean()
            row[f'ridge_{variant}_review_weighted'] = _weighted_average(group[ridge_col], group['num_reviews'])
            row[f'deberta_{variant}_review_weighted'] = _weighted_average(group[deberta_col], group['num_reviews'])
            row[f'diff_{variant}_review_weighted'] = (
                row[f'deberta_{variant}_review_weighted'] - row[f'ridge_{variant}_review_weighted']
            )
        monthly_rows.append(row)
    monthly_compare = pd.DataFrame(monthly_rows)

    return {
        'comparison_summary': comparison_summary,
        'volume_summary': volume_summary,
        'key_findings': key_findings,
        'top_disagreement_days': top_disagreement_days,
        'high_volume_top_disagreement_days': high_volume_top_disagreement_days,
        'disagreement_volume_summary': disagreement_volume_summary,
        'monthly_compare': monthly_compare,
    }


def add_rolling_features(ts_compare, windows=None):
    """Add 30d/90d rolling means + 365d rolling correlation to ts_compare in place."""
    if windows is None:
        windows = {'30d': 30, '90d': 90}
    for variant in ['naive', 'weighted']:
        ridge_col = f'score_tfidf_ridge_{variant}'
        deberta_col = f'score_deberta_v3_base_lora_{variant}'
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        for label, w in windows.items():
            min_periods = max(3, min(w // 3, 30))
            ts_compare[f'{ridge_col}_roll_{label}'] = ts_compare[ridge_col].rolling(w, min_periods=min_periods).mean()
            ts_compare[f'{deberta_col}_roll_{label}'] = ts_compare[deberta_col].rolling(w, min_periods=min_periods).mean()
            ts_compare[f'{diff_col}_roll_{label}'] = ts_compare[diff_col].rolling(w, min_periods=min_periods).mean()
        ts_compare[f'rolling_corr_{variant}_365d'] = (
            ts_compare[ridge_col].rolling(365, min_periods=90).corr(ts_compare[deberta_col])
        )
    return ts_compare


def _save_plot(fig, path, plot_role, title, description, plot_records=None, show=True):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    if plot_records is not None:
        plot_records.append({
            'Role': plot_role, 'Title': title,
            'Description': description, 'Path': display_path(path),
        })
    if show:
        plt.show()


def plot_main_figures(ts_compare, monthly_compare, volume_summary, plot_records, show=True):
    """4 main report plots: rolling trend, monthly trend, hexbin agreement, volume robustness."""
    # 1. Rolling daily trend
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, variant, title in zip(
        axes, ['naive', 'weighted'],
        ['Naive Daily Sentiment: 30/90-Day Rolling Mean',
         'Weighted Daily Sentiment: 30/90-Day Rolling Mean'],
    ):
        ridge_col = f'score_tfidf_ridge_{variant}'
        deberta_col = f'score_deberta_v3_base_lora_{variant}'
        ax.plot(ts_compare['review_day'], ts_compare[ridge_col], color='steelblue', alpha=0.10, linewidth=0.35)
        ax.plot(ts_compare['review_day'], ts_compare[deberta_col], color='tomato', alpha=0.10, linewidth=0.35)
        ax.plot(ts_compare['review_day'], ts_compare[f'{ridge_col}_roll_30d'], color='steelblue', alpha=0.85, linewidth=1.2, label='Ridge 30-day')
        ax.plot(ts_compare['review_day'], ts_compare[f'{deberta_col}_roll_30d'], color='tomato', alpha=0.85, linewidth=1.2, label='DeBERTa 30-day')
        ax.plot(ts_compare['review_day'], ts_compare[f'{ridge_col}_roll_90d'], color='navy', alpha=0.85, linewidth=2.0, label='Ridge 90-day')
        ax.plot(ts_compare['review_day'], ts_compare[f'{deberta_col}_roll_90d'], color='darkred', alpha=0.85, linewidth=2.0, label='DeBERTa 90-day')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Sentiment Score')
        ax.set_ylim(SENTIMENT_MIN - 0.08, SENTIMENT_MAX + 0.08)
        ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.25)
    fig.suptitle('Smoothed Daily Sentiment Trends: TF-IDF vs DeBERTa', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_MAIN / 'main_rolling_daily_trend.png',
               'main', 'Smoothed daily sentiment trends',
               'Primary trend plot: raw daily series in the background plus 30/90-day rolling means.',
               plot_records, show=show)

    # 2. Monthly trend
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, variant, title in zip(
        axes, ['naive', 'weighted'],
        ['Monthly Naive Sentiment', 'Monthly Weighted Sentiment'],
    ):
        ridge_col = f'ridge_{variant}_review_weighted'
        deberta_col = f'deberta_{variant}_review_weighted'
        ax.plot(monthly_compare['month'], monthly_compare[ridge_col], color='steelblue', linewidth=1.6, label='TF-IDF + Ridge')
        ax.plot(monthly_compare['month'], monthly_compare[deberta_col], color='tomato', linewidth=1.6, label='DeBERTa-v3-base + LoRA')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Sentiment Score')
        ax.set_ylim(SENTIMENT_MIN - 0.08, SENTIMENT_MAX + 0.08)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.25)
    fig.suptitle('Monthly Sentiment Trends: TF-IDF vs DeBERTa', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_MAIN / 'main_monthly_trend.png',
               'main', 'Monthly sentiment trends',
               'Report-friendly monthly version of the daily sentiment time series.',
               plot_records, show=show)

    # 3. Hexbin agreement
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, variant, title in zip(axes, ['naive', 'weighted'], ['Naive Density', 'Weighted Density']):
        ridge_col = f'score_tfidf_ridge_{variant}'
        deberta_col = f'score_deberta_v3_base_lora_{variant}'
        hb = ax.hexbin(ts_compare[ridge_col], ts_compare[deberta_col],
                       gridsize=45, mincnt=1, cmap='viridis')
        lims = [
            min(ts_compare[ridge_col].min(), ts_compare[deberta_col].min()) - 0.02,
            max(ts_compare[ridge_col].max(), ts_compare[deberta_col].max()) + 0.02,
        ]
        ax.plot(lims, lims, 'w--', linewidth=1.0, label='y = x')
        r, _, _, _, _ = _correlation_report(ts_compare[ridge_col], ts_compare[deberta_col])
        ax.set_title(f'{title} (r={r:.3f})', fontsize=12)
        ax.set_xlabel('TF-IDF + Ridge')
        ax.set_ylabel('DeBERTa-v3-base + LoRA')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=9)
        fig.colorbar(hb, ax=ax, label='Days')
    fig.suptitle('Daily Score Agreement: TF-IDF vs DeBERTa', fontsize=13)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_MAIN / 'main_model_agreement_hexbin.png',
               'main', 'Daily score agreement density',
               'Compact density view showing whether daily DeBERTa scores align with Ridge.',
               plot_records, show=show)

    # 4. Volume robustness
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, ylabel, title in [
        (axes[0], 'Pearson r', 'Pearson r', 'Agreement by minimum daily review count'),
        (axes[1], 'Mean Abs Diff', 'Mean absolute difference', 'Disagreement by minimum daily review count'),
    ]:
        for variant, color in [('naive', 'steelblue'), ('weighted', 'tomato')]:
            sub = volume_summary[volume_summary['Variant'] == variant]
            ax.plot(sub['Min Reviews'], sub[metric], marker='o', linewidth=1.5, color=color, label=variant)
        ax.set_xscale('log')
        ax.set_xlabel('Minimum daily reviews')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle('Robustness to Low-Volume Days', fontsize=13)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_MAIN / 'main_volume_robustness.png',
               'main', 'Robustness to low-volume days',
               'Whether model agreement improves when low-review-count days are excluded.',
               plot_records, show=show)


def plot_diagnostic_figures(ts_compare, plot_records, show=True):
    """4 diagnostic plots: difference series, distribution, Bland-Altman, rolling corr."""
    # 1. Difference series
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, variant, title in zip(axes, ['naive', 'weighted'],
                                  ['Naive: DeBERTa - TF-IDF', 'Weighted: DeBERTa - TF-IDF']):
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        ax.plot(ts_compare['review_day'], ts_compare[diff_col], linewidth=0.45, color='purple', alpha=0.22, label='Daily diff')
        ax.plot(ts_compare['review_day'], ts_compare[f'{diff_col}_roll_30d'], linewidth=1.2, color='darkviolet', alpha=0.9, label='30-day mean')
        ax.plot(ts_compare['review_day'], ts_compare[f'{diff_col}_roll_90d'], linewidth=2.0, color='black', alpha=0.8, label='90-day mean')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axhline(ts_compare[diff_col].mean(), color='red', linewidth=0.8, linestyle=':',
                   label=f'mean={ts_compare[diff_col].mean():.4f}')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Score Difference')
        ax.legend(fontsize=8, ncol=4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.25)
    fig.suptitle('Daily Score Difference: DeBERTa - TF-IDF', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_DIAG / 'diagnostic_difference_series.png',
               'diagnostic', 'Daily model difference series',
               'Shows when DeBERTa is locally higher or lower than Ridge.', plot_records, show=show)

    # 2. Difference distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, variant, title in zip(axes, ['naive', 'weighted'], ['Naive Difference', 'Weighted Difference']):
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        values = ts_compare[diff_col].dropna()
        ax.hist(values, bins=70, color='mediumpurple', alpha=0.75, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.9)
        ax.axvline(values.mean(), color='red', linestyle=':', linewidth=1.2,
                   label=f'mean={values.mean():.4f}')
        ax.axvline(values.median(), color='darkgreen', linestyle='-.', linewidth=1.0,
                   label=f'median={values.median():.4f}')
        ax.set_title(title)
        ax.set_xlabel('DeBERTa - TF-IDF')
        ax.set_ylabel('Days')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle('Distribution of Daily Model Differences', fontsize=13)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_DIAG / 'diagnostic_difference_distribution.png',
               'diagnostic', 'Distribution of daily differences',
               'Whether model differences are centered near zero or driven by outliers.',
               plot_records, show=show)

    # 3. Bland-Altman
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, variant, title in zip(axes, ['naive', 'weighted'], ['Naive Agreement', 'Weighted Agreement']):
        mean_col = f'mean_score_{variant}'
        diff_col = f'diff_deberta_minus_ridge_{variant}'
        values = ts_compare[diff_col].dropna()
        mean_diff = values.mean()
        sd_diff = values.std()
        upper = mean_diff + 1.96 * sd_diff
        lower = mean_diff - 1.96 * sd_diff
        sc = ax.scatter(
            ts_compare[mean_col], ts_compare[diff_col],
            c=np.log1p(ts_compare['num_reviews']),
            cmap='plasma', s=18, alpha=0.45, edgecolors='none',
        )
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=1.0, label=f'mean={mean_diff:.3f}')
        ax.axhline(upper, color='black', linestyle='--', linewidth=0.9, label='mean +/- 1.96 SD')
        ax.axhline(lower, color='black', linestyle='--', linewidth=0.9)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Average of Two Model Scores')
        ax.set_ylabel('DeBERTa - TF-IDF')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label='log1p(num_reviews)')
    fig.suptitle('Bland-Altman Agreement: Daily Model Scores', fontsize=13)
    _save_plot(fig, FIG_T1_COMPARE_DIAG / 'diagnostic_bland_altman.png',
               'diagnostic', 'Bland-Altman agreement',
               'Whether model disagreement changes across the sentiment scale.',
               plot_records, show=show)

    # 4. Rolling correlation
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for variant, color in [('naive', 'steelblue'), ('weighted', 'tomato')]:
        ax.plot(ts_compare['review_day'], ts_compare[f'rolling_corr_{variant}_365d'],
                color=color, linewidth=1.5, label=f'{variant} 365-day rolling r')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('365-Day Rolling Correlation: TF-IDF vs DeBERTa')
    ax.set_ylabel('Rolling Pearson r')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_plot(fig, FIG_T1_COMPARE_DIAG / 'diagnostic_rolling_correlation.png',
               'diagnostic', 'Rolling model correlation',
               'Whether agreement is stable over time.', plot_records, show=show)


def save_all_outputs(ts_compare, tables, plot_records):
    """Write all CSVs + plot manifest. Returns dict of output paths."""
    output_paths = {
        'daily_comparison': T1_DAILY_MODEL_COMPARISON_PATH,
        'key_findings': T1_KEY_FINDINGS_PATH,
        'summary_by_variant': T1_SUMMARY_BY_VARIANT_PATH,
        'summary_by_review_volume': T1_VOLUME_ROBUSTNESS_PATH,
        'top_disagreement_days': T1_DISAGREE_TOP_PATH,
        'top_high_volume_disagreement_days': T1_DISAGREE_HIGH_VOL_PATH,
        'top_disagreement_volume_summary': T1_DISAGREE_VOL_SUMMARY_PATH,
        'monthly_comparison': T1_MONTHLY_COMPARISON_PATH,
        'plot_manifest': T1_PLOT_MANIFEST_PATH,
    }
    atomic_to_csv(ts_compare, output_paths['daily_comparison'], index=False)
    atomic_to_csv(tables['key_findings'], output_paths['key_findings'], index=False)
    atomic_to_csv(tables['comparison_summary'], output_paths['summary_by_variant'], index=False)
    atomic_to_csv(tables['volume_summary'], output_paths['summary_by_review_volume'], index=False)
    atomic_to_csv(tables['top_disagreement_days'], output_paths['top_disagreement_days'], index=False)
    atomic_to_csv(tables['high_volume_top_disagreement_days'], output_paths['top_high_volume_disagreement_days'], index=False)
    atomic_to_csv(tables['disagreement_volume_summary'], output_paths['top_disagreement_volume_summary'], index=False)
    atomic_to_csv(tables['monthly_compare'], output_paths['monthly_comparison'], index=False)
    atomic_to_csv(pd.DataFrame(plot_records), output_paths['plot_manifest'], index=False)
    return output_paths


def display_summary_tables(tables, high_volume_threshold=25):
    """Print the key results tables to console."""
    key_findings = tables['key_findings']
    print("\n=== Key Findings for Report ===")
    kf_display = key_findings.copy()
    kf_display['Value'] = kf_display['Value'].map(lambda v: '-' if pd.isna(v) else f'{v:.4f}')
    print(format_console_grid(kf_display[['Finding', 'Variant', 'Metric', 'Value']]))
    print("\nNote: weighted scores use review weights, so a few high-weight reviews can create "
          "larger daily disagreement even when overall trend agreement remains strong.")

    print("\n=== Volume Robustness Summary ===")
    volume_display_cols = ['Min Reviews', 'Variant', 'Days', 'Median Reviews/Day',
                           'Pearson r', 'Spearman r', 'Mean Abs Diff', 'Diff P05', 'Diff P95']
    print(format_console_grid(tables['volume_summary'][volume_display_cols].round(4)))

    print("\n=== Top 10 Disagreement Days ===")
    print(format_console_grid(tables['top_disagreement_days'].head(10).round(4)))

    print(f"\n=== Top 10 High-Volume Disagreement Days (num_reviews >= {high_volume_threshold}) ===")
    hv = tables['high_volume_top_disagreement_days']
    if len(hv):
        print(format_console_grid(hv.head(10).round(4)))
    else:
        print('No high-volume days found for this threshold.')
