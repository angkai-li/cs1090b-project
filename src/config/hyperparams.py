"""
All hyperparameters - single source of truth.

Imported by config layer + all downstream modules. Keep this file flat (constants
only, no logic). If you need an ablation, override locally in the relevant cell
with a clear comment, do NOT mutate these.
"""

# ====================================================================
# Cleaning
# ====================================================================
CLEANING_VERSION = "cleaning_v5"
SENTIMENT_MIN = -1
SENTIMENT_MAX = 1


# ====================================================================
# Task 2 filter / aggregation
# ====================================================================
MIN_REVIEWS_TASK2 = 300       # 274 products satisfy this threshold
MIN_OBS_PER_MONTH = 5
MIN_OBS_PER_WEEK  = 3


# ====================================================================
# Review weighting (data/clean.py: review_weight)
# ====================================================================
ALPHA_VERIFIED = 0.5


# ====================================================================
# Weighted sentiment aggregation (task2/records.py: weighted_sentiment_score)
# ====================================================================
ALPHA = 0.3
BETA  = 0.5


# ====================================================================
# Product embedding + hierarchical L2 shrinkage
# ====================================================================
EMBED_DIM             = 4
RARE_THRESHOLD_MONTHS = 12       # products with < this many effective months -> ID=0 (RARE)
LAMBDA_L2             = 1e-3


# ====================================================================
# Routes 2/3/4 windowing
# ====================================================================
WINDOW_LEN_R2 = 24
WINDOW_LEN_R3 = 24
WINDOW_LEN_R4 = 24      # MOMENT change-point detection window

R1_MIN_SIZE          = 4    # PELT minimum segment size (Route 1 + Hybrid)
R1_MIN_SERIES_MONTHS = 12   # skip products with < 12 monthly observations
R1_MIN_SERIES_WEEKS  = 24   # skip products with < 24 weekly obs (~6 months)


# ====================================================================
# Synthetic data + Route 4 MOMENT
# ====================================================================
N_SYNTHETIC       = 50_000
MOMENT_TARGET_LEN = 512


# ====================================================================
# Random state
# ====================================================================
RANDOM_STATE = 1090
