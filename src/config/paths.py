"""
Project directory layout - all path constants, single source of truth.

Five top-level dirs under PROJECT_DIR. Each has ONE responsibility:
  data/       raw inputs (read-only, never written to)
  cache/      derived data (regenerable, safe to delete)
  models/     trained weights (long-lived artifacts)
  outputs/    data products (predictions, scores, eval tables)
  figures/    plots (for the report)

Conventions:
  - Every artifact lives in a dedicated subdir with a sibling `meta.json` describing
    the run that produced it (cache-validity check).
  - File names within a context-bearing path are minimal: a path like
    `outputs/task1/per_review/deberta_lora.csv` doesn't need to repeat "task1"
    or "deberta" in the file name.
  - System caches (matplotlib config, HuggingFace Hub) live under cache/ so the
    entire cache/ folder is safe to nuke without losing trained weights.

Importing this module:
  - Triggers _find_project_root() and chdir() to PROJECT_DIR
  - Creates all ALL_DIRS via mkdir(parents=True, exist_ok=True)
  - Sets a few env vars (matplotlib, HF) - note these env vars are ALSO set by
    src.config.env.setup_environment(), which must run first
"""

import os
from pathlib import Path


def _find_project_root():
    """Locate project root by searching for data/Cell_Phones_and_Accessories_5.json.

    Handles three layouts:
      (a) Local: cwd is already the project root.
      (b) Zip-extracted on a server: cwd is parent of '1090b-project/', chdir into it.
      (c) Renamed/nested folder: scan immediate subfolders for the dataset.
    Falls back to cwd (downstream code will raise a clear FileNotFoundError).
    """
    target = Path('data') / 'Cell_Phones_and_Accessories_5.json'
    cwd = Path.cwd()
    if (cwd / target).exists():
        return cwd
    for name in ('1090b-project', 'project', 'main'):
        cand = cwd / name
        if (cand / target).exists():
            return cand
    for sub in sorted(cwd.iterdir()) if cwd.exists() else []:
        if sub.is_dir() and not sub.name.startswith(('.', '_')):
            if (sub / target).exists():
                return sub
    return cwd


PROJECT_DIR = _find_project_root()
if Path.cwd() != PROJECT_DIR:
    _cwd_before = Path.cwd()
    os.chdir(PROJECT_DIR)
    print(f"chdir -> {PROJECT_DIR}  (was {_cwd_before})")

DATA_DIR    = PROJECT_DIR if PROJECT_DIR.name == "data" else PROJECT_DIR / "data"
CACHE_DIR   = PROJECT_DIR / "cache"
MODELS_DIR  = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = PROJECT_DIR / "figures"


# === (1) Raw inputs (read-only) ============================================
CELL_PHONE_PATH  = DATA_DIR / "Cell_Phones_and_Accessories_5.json"
VIDEO_GAMES_PATH = DATA_DIR / "Video_Games_5.json"


# === (2) Cache: derived data (regenerable) ================================
# Cleaned dataframe
CACHE_CLEANED_DIR     = CACHE_DIR / "cleaned"
CLEAN_CACHE_PATH      = CACHE_CLEANED_DIR / "df.pkl"
CLEAN_CACHE_META_PATH = CACHE_CLEANED_DIR / "meta.json"

# Train/val/test split indices (shared across Task 1 models for fair comparison)
T1_SPLIT_DIR        = CACHE_DIR / "splits"
T1_SPLIT_PATH       = T1_SPLIT_DIR / "split.csv"
T1_SPLIT_META_PATH  = T1_SPLIT_DIR / "meta.json"

# Time-series panels (rating-proxy + DeBERTa-based, daily/weekly/monthly)
PANEL_DIR = CACHE_DIR / "panels"
T1_RATING_DAILY_PATH         = PANEL_DIR / "rating_daily_ts.csv"
T1_RATING_DAILY_META_PATH    = PANEL_DIR / "rating_daily_ts.meta.json"
T1_RATING_WEEKLY_PATH        = PANEL_DIR / "rating_weekly_panel.csv"
T1_RATING_WEEKLY_META_PATH   = PANEL_DIR / "rating_weekly_panel.meta.json"
T2_PANEL_MONTHLY_PATH        = PANEL_DIR / "deberta_monthly_panel.csv"
T2_PANEL_MONTHLY_META_PATH   = PANEL_DIR / "deberta_monthly_panel.meta.json"
T2_PANEL_WEEKLY_PATH         = PANEL_DIR / "deberta_weekly_panel.csv"
T2_PANEL_WEEKLY_META_PATH    = PANEL_DIR / "deberta_weekly_panel.meta.json"

# Balanced helpfulness dataset (separate pipeline)
T1_NLP_DIR        = CACHE_DIR / "nlp_balanced"
T1_NLP_TRAIN_PATH = T1_NLP_DIR / "train.csv"
T1_NLP_META_PATH  = T1_NLP_DIR / "meta.json"

# Task 2 per-product training records (monthly)
T2_RECORDS_DIR       = CACHE_DIR / "task2_records_monthly"
T2_RECORDS_PATH      = T2_RECORDS_DIR / "records.pkl"
T2_RECORDS_META_PATH = T2_RECORDS_DIR / "meta.json"

# Synthetic + weak-label training set shared across Routes 2/3/4 (monthly)
T2_ROUTE_TRAIN_DATA_PATH = CACHE_DIR / "route_train_data_monthly.npz"

# System caches (separated from project caches so users can wipe without confusion)
MPL_CONFIG_DIR = CACHE_DIR / "mpl_config"
HF_HUB_DIR     = CACHE_DIR / "hf_hub"

# Set MPL/HF env vars (idempotent; env.py also sets these earlier)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(HF_HUB_DIR))
os.environ.setdefault("HF_HOME", str(HF_HUB_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_DIR / "huggingface" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HUB_DIR / "huggingface" / "datasets"))

# torch.compile inductor cache to project disk
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR / "torch_inductor"))


# === (3) Models: trained weights ===========================================
T1_RIDGE_MODEL_DIR  = MODELS_DIR / "task1_tfidf_ridge"
T1_RIDGE_MODEL_PATH = T1_RIDGE_MODEL_DIR / "pipeline.pkl"
T1_RIDGE_META_PATH  = T1_RIDGE_MODEL_DIR / "meta.json"

T1_DEBERTA_MODEL_DIR         = MODELS_DIR / "task1_deberta_lora"
T1_DEBERTA_FINAL_DIR         = T1_DEBERTA_MODEL_DIR / "final_model"
T1_DEBERTA_SUCCESS_PATH      = T1_DEBERTA_FINAL_DIR / "_SUCCESS"
T1_DEBERTA_CHECKPOINT_DIR    = T1_DEBERTA_MODEL_DIR / "checkpoints"
T1_DEBERTA_LOG_DIR           = T1_DEBERTA_MODEL_DIR / "tb_logs"
T1_DEBERTA_TRAIN_CONFIG_PATH = T1_DEBERTA_MODEL_DIR / "train_config.json"
T1_DEBERTA_TIMING_PATH       = T1_DEBERTA_MODEL_DIR / "run_timing.json"
T1_DEBERTA_TIMING_LOG_PATH   = T1_DEBERTA_MODEL_DIR / "run_timing_log.csv"

T2_R2_MODEL_DIR  = MODELS_DIR / "task2_route2_autocpd_monthly"
T2_R2_MODEL_PATH = T2_R2_MODEL_DIR / "model.pt"

T2_R3_MODEL_DIR  = MODELS_DIR / "task2_route3_tst_monthly"
T2_R3_MODEL_PATH = T2_R3_MODEL_DIR / "model.pt"

T2_R4_MODEL_DIR = MODELS_DIR / "task2_route4_moment_monthly"
T2_R4_LORA_DIR  = T2_R4_MODEL_DIR / "lora_adapter"
T2_R4_SIDE_PATH = T2_R4_MODEL_DIR / "side_channel.pt"


# === (4) Outputs: data products (predictions, scores, eval tables) =========
# Task 1, per-review scores (1.1M rows each)
T1_PER_REVIEW_DIR                    = OUTPUTS_DIR / "task1" / "per_review"
T1_PER_REVIEW_VADER_PATH             = T1_PER_REVIEW_DIR / "vader.csv"
T1_PER_REVIEW_RIDGE_PATH             = T1_PER_REVIEW_DIR / "tfidf_ridge.csv"
T1_PER_REVIEW_DEBERTA_PATH           = T1_PER_REVIEW_DIR / "deberta_lora.csv"
T1_PER_REVIEW_BASELINE_COMBINED_PATH = T1_PER_REVIEW_DIR / "baseline_combined.csv"

# Task 1, daily aggregated scores
T1_DAILY_DIR                    = OUTPUTS_DIR / "task1" / "daily"
T1_DAILY_RATING_PATH            = T1_DAILY_DIR / "rating.csv"
T1_DAILY_VADER_PATH             = T1_DAILY_DIR / "vader.csv"
T1_DAILY_RIDGE_PATH             = T1_DAILY_DIR / "tfidf_ridge.csv"
T1_DAILY_DEBERTA_PATH           = T1_DAILY_DIR / "deberta_lora.csv"
T1_DAILY_BASELINE_COMBINED_PATH = T1_DAILY_DIR / "baseline_combined.csv"

# Task 1, streaming progress (live during inference)
T1_DEBERTA_PROGRESS_PATH = T1_PER_REVIEW_DIR / "deberta_progress.csv"

# Task 1, evaluation tables
T1_EVAL_DIR                     = OUTPUTS_DIR / "task1" / "evaluation"
T1_BASELINE_RESULTS_PATH        = T1_EVAL_DIR / "baseline_results.csv"
T1_BASELINE_TS_OUTPUTS_PATH     = T1_EVAL_DIR / "baseline_ts_outputs.csv"
T1_BASELINE_DAILY_SUMMARY_PATH  = T1_EVAL_DIR / "baseline_daily_summary.csv"
T1_BASELINE_META_PATH           = T1_EVAL_DIR / "baseline_meta.json"
T1_MODEL_COMPARISON_PATH        = T1_EVAL_DIR / "model_comparison.csv"
T1_TS_OUTPUTS_PATH              = T1_EVAL_DIR / "ts_outputs.csv"
T1_DAILY_SUMMARY_PATH           = T1_EVAL_DIR / "daily_score_summary.csv"
T1_KEY_FINDINGS_PATH            = T1_EVAL_DIR / "key_findings.csv"
T1_VOLUME_ROBUSTNESS_PATH       = T1_EVAL_DIR / "volume_robustness.csv"
T1_DAILY_MODEL_COMPARISON_PATH  = T1_EVAL_DIR / "daily_model_comparison.csv"
T1_MONTHLY_COMPARISON_PATH      = T1_EVAL_DIR / "monthly_model_comparison.csv"
T1_SUMMARY_BY_VARIANT_PATH      = T1_EVAL_DIR / "summary_by_variant.csv"
T1_PLOT_MANIFEST_PATH           = T1_EVAL_DIR / "plot_manifest.csv"
T1_DEBERTA_TEST_METRICS_PATH    = T1_EVAL_DIR / "deberta_test_metrics.json"

T1_DISAGREE_DIR              = T1_EVAL_DIR / "disagreement"
T1_DISAGREE_TOP_PATH         = T1_DISAGREE_DIR / "top_days.csv"
T1_DISAGREE_HIGH_VOL_PATH    = T1_DISAGREE_DIR / "high_volume_days.csv"
T1_DISAGREE_VOL_SUMMARY_PATH = T1_DISAGREE_DIR / "volume_summary.csv"

# Task 2, legacy multivariate baseline (Sec 9.2)
T2_LEGACY_DIR        = OUTPUTS_DIR / "task2_legacy_monthly"
T2_LEGACY_PANEL_PATH = T2_LEGACY_DIR / "panel.csv"
T2_LEGACY_CP_PATH    = T2_LEGACY_DIR / "change_points.csv"
T2_LEGACY_META_PATH  = T2_LEGACY_DIR / "meta.json"

# Task 2, 4-route outputs (Sec 9.4-Sec 9.7)
T2_ROUTES_DIR = OUTPUTS_DIR / "task2_routes"

T2_R1_DIR         = T2_ROUTES_DIR / "route1_classical_monthly"
T2_R1_CP_PATH     = T2_R1_DIR / "change_points.csv"
T2_R1_LOOKUP_PATH = T2_R1_DIR / "pelt_lookup.pkl"
T2_R1_META_PATH   = T2_R1_DIR / "meta.json"

T2_R2_DIR        = T2_ROUTES_DIR / "route2_autocpd_monthly"
T2_R2_CP_PATH    = T2_R2_DIR / "change_points.csv"
T2_R2_PROBS_PATH = T2_R2_DIR / "probs.pkl"
T2_R2_META_PATH  = T2_R2_DIR / "meta.json"

T2_R3_DIR        = T2_ROUTES_DIR / "route3_tst_monthly"
T2_R3_CP_PATH    = T2_R3_DIR / "change_points.csv"
T2_R3_PROBS_PATH = T2_R3_DIR / "probs.pkl"
T2_R3_META_PATH  = T2_R3_DIR / "meta.json"

T2_R4_DIR        = T2_ROUTES_DIR / "route4_moment_monthly"
T2_R4_CP_PATH    = T2_R4_DIR / "change_points.csv"
T2_R4_PROBS_PATH = T2_R4_DIR / "probs.pkl"
T2_R4_META_PATH  = T2_R4_DIR / "meta.json"

# Task 2, hybrid PELT-style decoding (Sec 9.8) - symmetric monthly + weekly
T2_HYBRID_MONTHLY_DIR       = OUTPUTS_DIR / "task2_hybrid_monthly"
T2_HYBRID_MONTHLY_CP_PATH   = T2_HYBRID_MONTHLY_DIR / "change_points.csv"
T2_HYBRID_MONTHLY_META_PATH = T2_HYBRID_MONTHLY_DIR / "meta.json"
T2_HYBRID_WEEKLY_DIR        = OUTPUTS_DIR / "task2_hybrid_weekly"
T2_HYBRID_WEEKLY_CP_PATH    = T2_HYBRID_WEEKLY_DIR / "change_points.csv"
T2_HYBRID_WEEKLY_META_PATH  = T2_HYBRID_WEEKLY_DIR / "meta.json"

# Task 2, evaluation + phase profiling (Sec 9.9) - symmetric monthly + weekly
T2_EVAL_MONTHLY_DIR                = OUTPUTS_DIR / "task2_evaluation_monthly"
T2_ROUTE_CONSISTENCY_MONTHLY_PATH  = T2_EVAL_MONTHLY_DIR / "route_consistency.csv"
T2_PHASE_PROFILES_MONTHLY_PATH     = T2_EVAL_MONTHLY_DIR / "phase_profiles.csv"
T2_PHASE_CLUSTERED_MONTHLY_PATH    = T2_EVAL_MONTHLY_DIR / "phase_profiles_clustered.csv"
T2_FINAL_FINDINGS_MONTHLY_PATH     = T2_EVAL_MONTHLY_DIR / "final_findings.csv"
T2_EVAL_MONTHLY_META_PATH          = T2_EVAL_MONTHLY_DIR / "meta.json"
T2_EVAL_WEEKLY_DIR                 = OUTPUTS_DIR / "task2_evaluation_weekly"
T2_ROUTE_CONSISTENCY_WEEKLY_PATH   = T2_EVAL_WEEKLY_DIR / "route_consistency.csv"
T2_PHASE_PROFILES_WEEKLY_PATH      = T2_EVAL_WEEKLY_DIR / "phase_profiles.csv"
T2_PHASE_CLUSTERED_WEEKLY_PATH     = T2_EVAL_WEEKLY_DIR / "phase_profiles_clustered.csv"
T2_FINAL_FINDINGS_WEEKLY_PATH      = T2_EVAL_WEEKLY_DIR / "final_findings.csv"
T2_EVAL_WEEKLY_META_PATH           = T2_EVAL_WEEKLY_DIR / "meta.json"

# Backwards-compat aliases (point to monthly variants)
T2_HYBRID_DIR              = T2_HYBRID_MONTHLY_DIR
T2_HYBRID_CP_PATH          = T2_HYBRID_MONTHLY_CP_PATH
T2_HYBRID_META_PATH        = T2_HYBRID_MONTHLY_META_PATH
T2_EVAL_DIR                = T2_EVAL_MONTHLY_DIR
T2_ROUTE_CONSISTENCY_PATH  = T2_ROUTE_CONSISTENCY_MONTHLY_PATH
T2_PHASE_PROFILES_PATH     = T2_PHASE_PROFILES_MONTHLY_PATH
T2_PHASE_CLUSTERED_PATH    = T2_PHASE_CLUSTERED_MONTHLY_PATH
T2_FINAL_FINDINGS_PATH     = T2_FINAL_FINDINGS_MONTHLY_PATH
T2_EVAL_META_PATH          = T2_EVAL_MONTHLY_META_PATH


# === (5) Figures ============================================================
FIG_EDA              = FIGURES_DIR / "eda"
FIG_T1_BASELINE      = FIGURES_DIR / "task1_baseline"
FIG_T1_DEBERTA       = FIGURES_DIR / "task1_deberta"
FIG_T1_COMPARE_MAIN  = FIGURES_DIR / "task1_comparison_main"
FIG_T1_COMPARE_DIAG  = FIGURES_DIR / "task1_comparison_diagnostic"
FIG_T2_LEGACY_MONTHLY = FIGURES_DIR / "task2_legacy_monthly"
FIG_T2_R1_MONTHLY     = FIGURES_DIR / "task2_route1_monthly"
FIG_T2_R2_MONTHLY     = FIGURES_DIR / "task2_route2_monthly"
FIG_T2_R3_MONTHLY     = FIGURES_DIR / "task2_route3_monthly"
FIG_T2_R4_MONTHLY     = FIGURES_DIR / "task2_route4_monthly"
FIG_T2_HYBRID_MONTHLY = FIGURES_DIR / "task2_hybrid_monthly"
FIG_T2_EVAL_MONTHLY   = FIGURES_DIR / "task2_evaluation_monthly"
FIG_T2_PHASES_MONTHLY = FIGURES_DIR / "task2_phases_monthly"


# === Auto-create all directories ==========================================
ALL_DIRS = [
    DATA_DIR, CACHE_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR,
    CACHE_CLEANED_DIR, T1_SPLIT_DIR, PANEL_DIR, T1_NLP_DIR,
    T2_RECORDS_DIR, MPL_CONFIG_DIR, HF_HUB_DIR,
    T1_RIDGE_MODEL_DIR, T1_DEBERTA_MODEL_DIR, T1_DEBERTA_FINAL_DIR,
    T1_DEBERTA_CHECKPOINT_DIR, T1_DEBERTA_LOG_DIR,
    T2_R2_MODEL_DIR, T2_R3_MODEL_DIR, T2_R4_MODEL_DIR, T2_R4_LORA_DIR,
    T1_PER_REVIEW_DIR, T1_DAILY_DIR, T1_EVAL_DIR, T1_DISAGREE_DIR,
    T2_LEGACY_DIR, T2_ROUTES_DIR,
    T2_R1_DIR, T2_R2_DIR, T2_R3_DIR, T2_R4_DIR,
    T2_HYBRID_MONTHLY_DIR, T2_HYBRID_WEEKLY_DIR,
    T2_EVAL_MONTHLY_DIR, T2_EVAL_WEEKLY_DIR,
    FIG_EDA, FIG_T1_BASELINE, FIG_T1_DEBERTA, FIG_T1_COMPARE_MAIN, FIG_T1_COMPARE_DIAG,
    FIG_T2_LEGACY_MONTHLY, FIG_T2_R1_MONTHLY, FIG_T2_R2_MONTHLY,
    FIG_T2_R3_MONTHLY, FIG_T2_R4_MONTHLY,
    FIG_T2_HYBRID_MONTHLY, FIG_T2_EVAL_MONTHLY, FIG_T2_PHASES_MONTHLY,
]
for _d in ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)
