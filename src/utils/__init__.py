"""src.utils - shared helpers (IO, losses, training, metrics, formatting).

Typical usage:
    from src.utils.io import atomic_to_csv, read_json_or_none
    from src.utils.losses import FocalLoss
    from src.utils.training import BestStateTracker, split_train_val
    from src.utils.metrics import rmse, weighted_mean_with_fallback
    from src.utils.format import format_elapsed, print_model_comparison_table
"""
