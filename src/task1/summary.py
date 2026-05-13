"""
Final output manifest (Sec 8.4): verify every artifact this notebook produces exists.

Run at the end as a sanity check that nothing got skipped.
"""

from pathlib import Path

from ..config.paths import (
    CELL_PHONE_PATH,
    CLEAN_CACHE_PATH, CLEAN_CACHE_META_PATH,
    DATA_DIR,
    FIG_T1_COMPARE_DIAG, FIG_T1_COMPARE_MAIN,
    T1_BASELINE_DAILY_SUMMARY_PATH, T1_BASELINE_META_PATH,
    T1_BASELINE_RESULTS_PATH, T1_BASELINE_TS_OUTPUTS_PATH,
    T1_DAILY_BASELINE_COMBINED_PATH, T1_DAILY_DEBERTA_PATH,
    T1_DAILY_MODEL_COMPARISON_PATH, T1_DAILY_RATING_PATH, T1_DAILY_RIDGE_PATH,
    T1_DAILY_SUMMARY_PATH, T1_DAILY_VADER_PATH,
    T1_DEBERTA_CHECKPOINT_DIR, T1_DEBERTA_FINAL_DIR, T1_DEBERTA_PROGRESS_PATH,
    T1_DEBERTA_SUCCESS_PATH, T1_DEBERTA_TEST_METRICS_PATH,
    T1_DEBERTA_TIMING_LOG_PATH, T1_DEBERTA_TIMING_PATH, T1_DEBERTA_TRAIN_CONFIG_PATH,
    T1_DISAGREE_HIGH_VOL_PATH, T1_DISAGREE_TOP_PATH, T1_DISAGREE_VOL_SUMMARY_PATH,
    T1_KEY_FINDINGS_PATH, T1_MODEL_COMPARISON_PATH, T1_MONTHLY_COMPARISON_PATH,
    T1_NLP_META_PATH, T1_NLP_TRAIN_PATH,
    T1_PER_REVIEW_BASELINE_COMBINED_PATH, T1_PER_REVIEW_DEBERTA_PATH,
    T1_PER_REVIEW_RIDGE_PATH, T1_PER_REVIEW_VADER_PATH,
    T1_PLOT_MANIFEST_PATH,
    T1_RATING_WEEKLY_META_PATH, T1_RATING_WEEKLY_PATH,
    T1_RIDGE_MODEL_PATH,
    T1_SPLIT_META_PATH, T1_SPLIT_PATH,
    T1_SUMMARY_BY_VARIANT_PATH, T1_TS_OUTPUTS_PATH, T1_VOLUME_ROBUSTNESS_PATH,
    T2_EVAL_META_PATH, T2_HYBRID_CP_PATH, T2_HYBRID_META_PATH,
    T2_LEGACY_CP_PATH, T2_LEGACY_META_PATH, T2_LEGACY_PANEL_PATH,
    T2_PANEL_MONTHLY_META_PATH, T2_PANEL_MONTHLY_PATH, T2_PANEL_WEEKLY_PATH,
    T2_PHASE_PROFILES_PATH,
    T2_R1_CP_PATH, T2_R1_LOOKUP_PATH, T2_R1_META_PATH,
    T2_R2_CP_PATH, T2_R2_META_PATH, T2_R2_MODEL_PATH, T2_R2_PROBS_PATH,
    T2_R3_CP_PATH, T2_R3_META_PATH, T2_R3_MODEL_PATH, T2_R3_PROBS_PATH,
    T2_R4_CP_PATH, T2_R4_LORA_DIR, T2_R4_META_PATH, T2_R4_PROBS_PATH, T2_R4_SIDE_PATH,
    T2_RECORDS_META_PATH, T2_RECORDS_PATH,
    T2_ROUTE_CONSISTENCY_PATH, T2_ROUTE_TRAIN_DATA_PATH,
    VIDEO_GAMES_PATH,
)
from ..config.runtime import HAS_VADER


def expected_outputs(include_vader=None):
    """Return list of Path objects expected to exist after a full run."""
    if include_vader is None:
        include_vader = HAS_VADER
    paths = [
        # (2) cache/, derived data
        CLEAN_CACHE_PATH, CLEAN_CACHE_META_PATH,
        T1_SPLIT_PATH, T1_SPLIT_META_PATH,
        T1_RATING_WEEKLY_PATH, T1_RATING_WEEKLY_META_PATH,
        T1_NLP_TRAIN_PATH, T1_NLP_META_PATH,

        # (3) models/, trained weights
        T1_RIDGE_MODEL_PATH,
        T1_DEBERTA_CHECKPOINT_DIR, T1_DEBERTA_FINAL_DIR,
        T1_DEBERTA_SUCCESS_PATH, T1_DEBERTA_TRAIN_CONFIG_PATH,
        T1_DEBERTA_TIMING_PATH, T1_DEBERTA_TIMING_LOG_PATH,

        # (4) outputs/task1/
        T1_PER_REVIEW_RIDGE_PATH, T1_PER_REVIEW_DEBERTA_PATH,
        T1_PER_REVIEW_BASELINE_COMBINED_PATH,
        T1_DAILY_RATING_PATH, T1_DAILY_RIDGE_PATH,
        T1_DAILY_DEBERTA_PATH, T1_DAILY_BASELINE_COMBINED_PATH,
        T1_DEBERTA_PROGRESS_PATH,
        T1_BASELINE_RESULTS_PATH, T1_BASELINE_TS_OUTPUTS_PATH,
        T1_BASELINE_DAILY_SUMMARY_PATH, T1_BASELINE_META_PATH,
        T1_MODEL_COMPARISON_PATH, T1_TS_OUTPUTS_PATH, T1_DAILY_SUMMARY_PATH,
        T1_DAILY_MODEL_COMPARISON_PATH, T1_MONTHLY_COMPARISON_PATH,
        T1_KEY_FINDINGS_PATH, T1_SUMMARY_BY_VARIANT_PATH,
        T1_VOLUME_ROBUSTNESS_PATH, T1_PLOT_MANIFEST_PATH,
        T1_DEBERTA_TEST_METRICS_PATH,
        T1_DISAGREE_TOP_PATH, T1_DISAGREE_HIGH_VOL_PATH, T1_DISAGREE_VOL_SUMMARY_PATH,

        # (5) figures/task1_*
        FIG_T1_COMPARE_MAIN / 'main_rolling_daily_trend.png',
        FIG_T1_COMPARE_MAIN / 'main_monthly_trend.png',
        FIG_T1_COMPARE_MAIN / 'main_model_agreement_hexbin.png',
        FIG_T1_COMPARE_MAIN / 'main_volume_robustness.png',
        FIG_T1_COMPARE_DIAG / 'diagnostic_difference_series.png',
        FIG_T1_COMPARE_DIAG / 'diagnostic_difference_distribution.png',
        FIG_T1_COMPARE_DIAG / 'diagnostic_bland_altman.png',
        FIG_T1_COMPARE_DIAG / 'diagnostic_rolling_correlation.png',

        # (4) outputs/task2_*
        T2_LEGACY_PANEL_PATH, T2_LEGACY_CP_PATH, T2_LEGACY_META_PATH,
        T2_RECORDS_PATH, T2_RECORDS_META_PATH, T2_ROUTE_TRAIN_DATA_PATH,
        T2_PANEL_MONTHLY_PATH, T2_PANEL_WEEKLY_PATH, T2_PANEL_MONTHLY_META_PATH,
        T2_R1_CP_PATH, T2_R1_LOOKUP_PATH, T2_R1_META_PATH,
        T2_R2_CP_PATH, T2_R2_PROBS_PATH, T2_R2_META_PATH, T2_R2_MODEL_PATH,
        T2_R3_CP_PATH, T2_R3_PROBS_PATH, T2_R3_META_PATH, T2_R3_MODEL_PATH,
        T2_R4_CP_PATH, T2_R4_PROBS_PATH, T2_R4_META_PATH,
        T2_R4_LORA_DIR, T2_R4_SIDE_PATH,
        T2_HYBRID_CP_PATH, T2_HYBRID_META_PATH,
        T2_ROUTE_CONSISTENCY_PATH, T2_PHASE_PROFILES_PATH, T2_EVAL_META_PATH,
    ]
    if include_vader:
        paths.append(T1_PER_REVIEW_VADER_PATH)
        paths.append(T1_DAILY_VADER_PATH)
    return paths


def verify_all(include_vader=None):
    """Print which expected outputs exist. Returns (ok_count, missing_count)."""
    paths = expected_outputs(include_vader=include_vader)
    print("=== Expected outputs ===")
    ok = 0
    missing = 0
    for p in paths:
        status = "exists" if p.exists() else "MISSING"
        if p.exists():
            ok += 1
        else:
            missing += 1
        print(f"  {status:8s}  {p}")
    print(f"\nSummary: {ok} present, {missing} missing")
    return ok, missing


def check_data_dir_hygiene():
    """Verify data/ only contains expected raw input files. Returns True if clean."""
    raw_root_files = {CELL_PHONE_PATH.name}
    if VIDEO_GAMES_PATH.exists():
        raw_root_files.add(VIDEO_GAMES_PATH.name)

    print("\n=== data/ hygiene ===")
    unexpected_files = sorted(
        p.name for p in DATA_DIR.iterdir() if p.is_file() and p.name not in raw_root_files
    )
    unexpected_dirs = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"Expected raw files: {sorted(raw_root_files)}")
    if unexpected_files or unexpected_dirs:
        print("Unexpected files in data/:", unexpected_files)
        print("Unexpected dirs in data/:", unexpected_dirs)
        print("(Old derived dirs from earlier runs may still be present; safe to delete since "
              "cache/ models/ outputs/ figures/ are now top-level.)")
        return False
    print("OK: data/ contains only raw inputs. Derived data is in cache/, models/, outputs/, figures/.")
    return True
