"""
Sec 10.6 PART L - Multiplicity correction.

  BH-FDR (Benjamini-Hochberg 1995): controls false discovery rate (less conservative).
  Bonferroni-Holm (Holm 1979):      controls family-wise error rate (more conservative).

Applied per-family: A_single_anchor (24 tests), B_multi_anchor (24 tests),
C_reverse_direction (12 tests).
"""

import numpy as np
import pandas as pd

from ..utils.io import atomic_to_csv
from .events import T2_EVENT_VALIDATION_DIR


T2_EVENT_FDR_PATH = T2_EVENT_VALIDATION_DIR / "event_fdr_adjusted.csv"


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR adjustment. Returns adjusted p-values (same length, same order)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    sorted_p = p[order]
    raw_adj = sorted_p * n / ranks
    # Enforce monotonicity by reverse cumulative minimum
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    # Reorder back to original
    adj = np.empty(n, dtype=float)
    adj[order] = adj_sorted
    return adj


def bonferroni_holm(p_values):
    """Bonferroni-Holm step-down FWER correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.minimum(sorted_p * (n - np.arange(n)), 1.0)
    # Enforce monotonicity (step-down: never decreases)
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj = np.empty(n, dtype=float)
    adj[order] = adj_sorted
    return adj


def apply_corrections(p_df, family_col='family', p_col='raw_p'):
    """Apply per-family BH-FDR + Bonferroni-Holm to a DataFrame.

    Adds columns: adj_p_BH, adj_p_Holm, sig_BH_05, sig_Holm_05.
    Returns the augmented DataFrame.
    """
    out = p_df.copy()
    out['adj_p_BH'] = np.nan
    out['adj_p_Holm'] = np.nan
    for family, group in out.groupby(family_col):
        p_arr = group[p_col].values
        bh = bh_fdr(p_arr)
        holm = bonferroni_holm(p_arr)
        out.loc[group.index, 'adj_p_BH'] = np.round(bh, 4)
        out.loc[group.index, 'adj_p_Holm'] = np.round(holm, 4)
    out['sig_BH_05'] = out['adj_p_BH'] < 0.05
    out['sig_Holm_05'] = out['adj_p_Holm'] < 0.05
    return out


def build_combined_family_table(single_anchor_emp_p_df, multi_anchor_bootstrap_df,
                                 reverse_null_df):
    """Combine A_single_anchor + B_multi_anchor + C_reverse_direction p-values
    into a single DataFrame ready for per-family BH-FDR / Holm correction."""
    rows = []
    for _, row in single_anchor_emp_p_df.iterrows():
        rows.append({
            'family': 'A_single_anchor',
            'route': row['route'],
            'tolerance_months': row['tolerance_months'],
            'observed': row['observed_hit_rate'],
            'null_mean': row['null_mean_hit_rate'],
            'raw_p': row['empirical_p_one_sided'],
        })
    for _, row in multi_anchor_bootstrap_df.iterrows():
        rows.append({
            'family': 'B_multi_anchor',
            'route': row['route'],
            'tolerance_months': row['tolerance_months'],
            'observed': row['observed_hit_rate'],
            'null_mean': row['null_mean_hit_rate'],
            'raw_p': row['empirical_p_one_sided'],
        })
    for _, row in reverse_null_df.iterrows():
        rows.append({
            'family': 'C_reverse_direction',
            'route': row['route'],
            'tolerance_months': row['tolerance_months'],
            'observed': row['observed_match_rate'],
            'null_mean': row['null_mean'],
            'raw_p': row['empirical_p_one_sided'],
        })
    return pd.DataFrame(rows)


def run_fdr_corrections(single_anchor_emp_p_df, multi_anchor_bootstrap_df,
                        reverse_null_df, force_rerun=False, verbose=True):
    """Compute family-wide BH + Holm corrections. Returns adjusted DataFrame."""
    if not force_rerun and T2_EVENT_FDR_PATH.exists():
        adjusted = pd.read_csv(T2_EVENT_FDR_PATH)
        if verbose:
            print(f"Loaded FDR corrections from cache: {len(adjusted)} rows")
            for family in ['A_single_anchor', 'B_multi_anchor', 'C_reverse_direction']:
                sub = adjusted[adjusted['family'] == family]
                n_sig_bh = int(sub['sig_BH_05'].sum())
                bh_min = sub['adj_p_BH'].min() if len(sub) else float('nan')
                print(f"  Family {family}: {len(sub)} tests, BH sig {n_sig_bh}, min adj_p_BH={bh_min:.4f}")
        return adjusted

    combined = build_combined_family_table(
        single_anchor_emp_p_df, multi_anchor_bootstrap_df, reverse_null_df)
    adjusted = apply_corrections(combined)
    atomic_to_csv(adjusted, T2_EVENT_FDR_PATH, index=False)

    if verbose:
        for family in ['A_single_anchor', 'B_multi_anchor', 'C_reverse_direction']:
            sub = adjusted[adjusted['family'] == family]
            n_total = len(sub)
            n_sig_bh = int(sub['sig_BH_05'].sum())
            n_sig_holm = int(sub['sig_Holm_05'].sum())
            bh_min = sub['adj_p_BH'].min() if len(sub) else float('nan')
            holm_min = sub['adj_p_Holm'].min() if len(sub) else float('nan')
            print(f"\n  Family {family}: {n_total} tests")
            print(f"    BH-FDR        {n_sig_bh}/{n_total} sig at alpha=0.05  "
                  f"(min adj_p_BH={bh_min:.4f})")
            print(f"    Bonferroni-Holm {n_sig_holm}/{n_total} sig at alpha=0.05  "
                  f"(min adj_p_Holm={holm_min:.4f})")

        print("\n=== Top 10 lowest BH-adjusted p-values (across all families) ===")
        top10 = adjusted.nsmallest(10, 'adj_p_BH')[
            ['family', 'route', 'tolerance_months', 'observed', 'null_mean',
             'raw_p', 'adj_p_BH', 'adj_p_Holm', 'sig_BH_05', 'sig_Holm_05']
        ]
        print(top10.to_string(index=False))
    return adjusted
