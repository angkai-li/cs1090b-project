"""
Sec 10 plots: 5-phone overlays, bootstrap curves, Hedges' g forest, reverse-direction bar.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .events import (
    EVENT_ANCHORS, T2_EVENT_VALIDATION_DIR, TOLERANCE_PRIMARY, TOLERANCES_EXTENDED,
)


T2_EVENT_FIG_PATH = T2_EVENT_VALIDATION_DIR / "event_overlays.png"
T2_EVENT_BOOTSTRAP_FIG_PATH = T2_EVENT_VALIDATION_DIR / "event_bootstrap_curves.png"
T2_EVENT_FOREST_FIG_PATH = T2_EVENT_VALIDATION_DIR / "event_forest_plot.png"
T2_EVENT_REVERSE_BAR_FIG_PATH = T2_EVENT_VALIDATION_DIR / "event_reverse_observed_vs_null.png"


ROUTE_COLORS = {
    'PELT (R1)':    '#1f77b4',
    'AutoCPD (R2)': '#ff7f0e',
    'TST (R3)':     '#2ca02c',
    'MOMENT (R4)':  '#d62728',
}

ROUTE_LINESTYLES = {
    'PELT (R1)':    (0, (5, 2)),
    'AutoCPD (R2)': (0, (3, 1, 1, 1)),
    'TST (R3)':     (0, (1, 1)),
    'MOMENT (R4)':  (0, (4, 1, 1, 1, 1, 1)),
}


def plot_overlay_5phones(records_dict, route_cps_dict, anchors=EVENT_ANCHORS,
                         tolerance=TOLERANCE_PRIMARY, save_path=None, show=True):
    """Sec 10 main plot: 5 phones x sentiment trace + 4 routes' cps overlay + event red line."""
    save_path = save_path or T2_EVENT_FIG_PATH
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=False)
    for ax, (asin, info) in zip(axes, anchors.items()):
        if asin not in records_dict:
            ax.set_title(f"{info['title']} ({asin}) - RECORD MISSING")
            ax.text(0.5, 0.5, 'NOT IN MONTHLY PANEL', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='red')
            continue
        r = records_dict[asin]
        dates = pd.to_datetime(r['dates'])
        series = r['series_norm']
        event_date = info['event_date']
        ax.plot(dates, series, color='#444444', linewidth=1.4, alpha=0.85, zorder=2)
        ax.fill_between(dates, series.min() if len(series) else 0, series,
                        alpha=0.10, color='#888888', zorder=1)
        band_lo = event_date - pd.Timedelta(days=tolerance * 30)
        band_hi = event_date + pd.Timedelta(days=tolerance * 30)
        ax.axvspan(band_lo, band_hi, color='#ffcccc', alpha=0.45, zorder=0)
        ax.axvline(event_date, color='#cc0000', linewidth=2.4, linestyle='-', alpha=0.95, zorder=4)
        for route_name, color in ROUTE_COLORS.items():
            cps = route_cps_dict.get(route_name, {}).get(asin, [])
            for cp_idx in cps:
                if 0 <= cp_idx < len(dates):
                    ax.axvline(dates[cp_idx], color=color, linewidth=1.6,
                               linestyle=ROUTE_LINESTYLES[route_name],
                               alpha=0.80, zorder=3)
        ax.set_title(
            f"{info['title']} ({asin})  |  Event: {info['event_label']} @ {event_date.date()}"
            f"  |  Strength: {info['strength']}",
            fontsize=10,
        )
        ax.set_ylabel('sentiment_norm')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())

    handles = [
        Line2D([0], [0], color='#cc0000', linewidth=2.4, label='Documented event date'),
        Patch(facecolor='#ffcccc', alpha=0.45, label=f'+/-{tolerance}-month tolerance band'),
    ]
    for n, c in ROUTE_COLORS.items():
        handles.append(Line2D([0], [0], color=c, linewidth=1.6,
                              linestyle=ROUTE_LINESTYLES[n], label=n))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.005),
               ncol=3, fontsize=9, frameon=True)
    plt.suptitle('Sec 10 Event-anchored validation: 5 phones x 4 methods cps overlay',
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_bootstrap_curves(ext_agg_df, emp_p_df, n_sample_per_round=5,
                          n_bootstrap=10000, route_names=None,
                          tolerances=TOLERANCES_EXTENDED, save_path=None, show=True):
    """Sec 10.4 plot: observed vs null hit-rate curves with shaded p95 band."""
    save_path = save_path or T2_EVENT_BOOTSTRAP_FIG_PATH
    route_names = route_names or list(ROUTE_COLORS.keys())
    fig, ax = plt.subplots(figsize=(11, 6))
    for route_name in route_names:
        obs = ext_agg_df[ext_agg_df['route'] == route_name].sort_values('tolerance_months')
        nul = emp_p_df[emp_p_df['route'] == route_name].sort_values('tolerance_months')
        if obs.empty:
            continue
        c = ROUTE_COLORS.get(route_name, 'gray')
        ax.plot(obs['tolerance_months'], obs['hit_rate'],
                marker='o', color=c, linewidth=2, label=f'{route_name} (observed)')
        if not nul.empty:
            ax.plot(nul['tolerance_months'], nul['null_mean_hit_rate'],
                    linestyle=':', color=c, linewidth=1.4, alpha=0.7,
                    label=f'{route_name} (null mean)')
            ax.fill_between(nul['tolerance_months'],
                            nul['null_mean_hit_rate'],
                            nul['null_p95_hit_rate'],
                            color=c, alpha=0.10)
    ax.set_xlabel('Tolerance +/- months')
    ax.set_ylabel('Hit rate')
    ax.set_title(f'Sec 10.4 Hit-rate vs tolerance (observed = solid, null bootstrap = dotted +/- 95th pct shaded)\n'
                 f'N={n_sample_per_round} per round x {n_bootstrap} rounds = '
                 f'{n_bootstrap*n_sample_per_round:,} null trials per route x tolerance',
                 fontsize=10)
    ax.set_xticks(tolerances)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_forest_hedges_g(hedges_df, save_path=None, show=True):
    """Sec 10.6 plot: forest plot of Hedges' g with 95% CI per phone."""
    save_path = save_path or T2_EVENT_FOREST_FIG_PATH
    plot_df = hedges_df.dropna(subset=['hedges_g']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 1.0 * len(plot_df))))
    y_positions = list(range(len(plot_df)))
    for i, row in plot_df.iterrows():
        g = row['hedges_g']
        ci_lo = row['ci_lower_g']
        ci_hi = row['ci_upper_g']
        sig = (ci_hi < 0 or ci_lo > 0)
        color = '#cc0000' if sig else '#999999'
        ax.errorbar(g, i, xerr=[[g - ci_lo], [ci_hi - g]],
                    fmt='s', color=color, markersize=11, capsize=6,
                    elinewidth=2.4, capthick=2)
        ax.text(ci_hi + 0.10, i,
                f"g={g:.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]  n={row['n_pre']}+{row['n_post']}",
                va='center', fontsize=9, color='#222222')
    ax.axvline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{r['phone']}\n  {r['event_label'][:42]}"
                        for _, r in plot_df.iterrows()], fontsize=9)
    ax.set_xlabel("Hedges' g  (standardized mean diff: post - pre,  +/-6mo windows)", fontsize=10)
    ax.set_title("Sec 10.6 Forest plot: sentiment magnitude shift per anchor event\n"
                 "(red = CI excludes 0, gray = inconclusive)", fontsize=11)
    ax.grid(axis='x', alpha=0.25)
    ax.invert_yaxis()
    ax.set_xlim(min(plot_df['ci_lower_g'].min() - 0.5, -1),
                max(plot_df['ci_upper_g'].max() + 2.0, 2))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_reverse_observed_vs_null(reverse_null_df, tolerance_months=2,
                                  save_path=None, show=True):
    """Sec 10.6 plot: bar chart of observed match rate vs random-cp null per route."""
    save_path = save_path or T2_EVENT_REVERSE_BAR_FIG_PATH
    sub = reverse_null_df[reverse_null_df['tolerance_months'] == tolerance_months].copy()
    if sub.empty:
        return None
    sub = sub.sort_values('observed_match_rate', ascending=False)
    routes = sub['route'].tolist()
    observed = sub['observed_match_rate'].values
    null_mean = sub['null_mean'].values

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(routes))
    w = 0.35
    bars_obs = ax.bar(x - w/2, observed, w, label='Observed',
                       color=[ROUTE_COLORS.get(r, 'gray') for r in routes])
    bars_null = ax.bar(x + w/2, null_mean, w, label='Null (random cps)',
                        color='lightgray', edgecolor='black')
    for bar, val in zip(bars_obs, observed):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_null, null_mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='dimgray')
    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=0, fontsize=9)
    ax.set_ylabel(f'Match rate <=+/-{tolerance_months}mo')
    ax.set_title(f'Sec 10.6 Reverse-direction: observed cps match KNOWN_EVENTS vs random-cp null '
                 f'(+/-{tolerance_months}mo)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, max(observed.max(), null_mean.max()) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
