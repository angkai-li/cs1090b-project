"""
Text / table formatting helpers (no numerics - see metrics.py for math).

Functions here are about display: elapsed-time strings, count formatting,
ASCII console grids, model-comparison tables.
"""

from pathlib import Path

import pandas as pd


def format_elapsed(seconds):
    """Format a duration in seconds as a human-readable string."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m {seconds:04.1f}s"


def display_path(path, base_dir=None):
    """Convert an absolute path to a project-relative display string.

    If base_dir is given and path is under it, returns the relative portion.
    Otherwise returns the absolute path string.
    """
    if path is None:
        return ''
    path = Path(path)
    if base_dir is not None:
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            pass
    return str(path)


def format_count_for_display(value):
    """Format an integer count with thousands-separator commas. Handles NaN."""
    if value is None:
        return '-'
    try:
        if pd.isna(value):
            return '-'
    except (TypeError, ValueError):
        pass
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def format_console_grid(frame):
    """Render a DataFrame as an ASCII grid for console display."""
    display_frame = frame.copy().fillna('-')
    if display_frame.empty:
        return '(empty)'

    string_frame = display_frame.astype(str)
    columns = list(string_frame.columns)
    widths = []
    for col in columns:
        values = string_frame[col].tolist()
        widths.append(max([len(str(col))] + [len(value) for value in values]))

    border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
    header = '|' + '|'.join(f" {col:<{width}} " for col, width in zip(columns, widths)) + '|'
    lines = [border, header, border]
    for _, row in string_frame.iterrows():
        lines.append('|' + '|'.join(f" {row[col]:<{width}} " for col, width in zip(columns, widths)) + '|')
    lines.append(border)
    return '\n'.join(lines)


# ----------------------------------------------------------------------
# Model-comparison + per-model summary table helpers (Task 1)
# ----------------------------------------------------------------------

def make_model_result_row(model, mae, rmse, run_label, modeling_rows, train_rows,
                          validation_rows, test_rows, split_label):
    """Build a single row of the model-comparison table."""
    return {
        'Model': model,
        'Run': run_label,
        'Modeling Rows': modeling_rows,
        'Train Rows': train_rows,
        'Validation Rows': validation_rows,
        'Test Rows': test_rows,
        'Split': split_label,
        'MAE': float(mae),
        'RMSE': float(rmse),
    }


def print_model_comparison_table(results, title='Model Comparison'):
    """Print a model-comparison table to console + return the DataFrame."""
    results_df = pd.DataFrame(results).copy()
    preferred_cols = [
        'Model', 'Run', 'Modeling Rows', 'Train Rows', 'Validation Rows',
        'Test Rows', 'Split', 'MAE', 'RMSE',
    ]
    ordered_cols = [c for c in preferred_cols if c in results_df.columns]
    ordered_cols += [c for c in results_df.columns if c not in ordered_cols]
    results_df = results_df[ordered_cols]

    display_df = results_df.copy()
    for col in ['Modeling Rows', 'Train Rows', 'Validation Rows', 'Test Rows']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_count_for_display)
    for col in ['MAE', 'RMSE']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').map(
                lambda v: '-' if pd.isna(v) else f"{v:.4f}"
            )

    print(f"\n=== {title} ===")
    print(format_console_grid(display_df))
    return results_df


def make_daily_score_summary_row(model, run_label, daily_frame, naive_col, weighted_col):
    """Build a row of the daily-score summary table (one per Model/Run)."""
    row = {'Model': model, 'Run': run_label}
    for label, col in [('Naive', naive_col), ('Weighted', weighted_col)]:
        values = pd.to_numeric(daily_frame[col], errors='coerce')
        row[f'{label} Mean'] = values.mean()
        row[f'{label} Std'] = values.std()
        row[f'{label} Min'] = values.min()
        row[f'{label} Max'] = values.max()
    return row


def print_daily_score_summary_table(results, title='Daily Score Summary'):
    """Print daily-score summary + return DataFrame."""
    results_df = pd.DataFrame(results).copy()
    preferred_cols = [
        'Model', 'Run',
        'Naive Mean', 'Naive Std', 'Naive Min', 'Naive Max',
        'Weighted Mean', 'Weighted Std', 'Weighted Min', 'Weighted Max',
    ]
    ordered_cols = [c for c in preferred_cols if c in results_df.columns]
    ordered_cols += [c for c in results_df.columns if c not in ordered_cols]
    results_df = results_df[ordered_cols]

    display_df = results_df.copy()
    metric_cols = [c for c in display_df.columns if c not in {'Model', 'Run'}]
    for col in metric_cols:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').map(
            lambda v: '-' if pd.isna(v) else f"{v:.4f}"
        )

    print(f"\n=== {title} ===")
    print(format_console_grid(display_df))
    return results_df


def make_time_series_output_row(model, run_label, score_column, daily_naive_column,
                                daily_weighted_column, daily_rows, review_score_file,
                                daily_score_file):
    """Build a row of the time-series output-manifest table."""
    return {
        'Model': model,
        'Run': run_label,
        'Score Column': score_column,
        'Daily Naive Column': daily_naive_column,
        'Daily Weighted Column': daily_weighted_column,
        'Daily Rows': daily_rows,
    }


def print_time_series_output_table(results, title='Time Series Outputs'):
    """Print time-series output manifest table + return DataFrame."""
    results_df = pd.DataFrame(results).copy()
    preferred_cols = [
        'Model', 'Run', 'Score Column',
        'Daily Naive Column', 'Daily Weighted Column', 'Daily Rows',
    ]
    ordered_cols = [c for c in preferred_cols if c in results_df.columns]
    ordered_cols += [c for c in results_df.columns if c not in ordered_cols]
    results_df = results_df[ordered_cols]

    display_df = results_df.copy()
    if 'Daily Rows' in display_df.columns:
        display_df['Daily Rows'] = display_df['Daily Rows'].apply(format_count_for_display)

    print(f"\n=== {title} ===")
    print(format_console_grid(display_df))
    return results_df
