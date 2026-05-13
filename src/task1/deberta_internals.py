"""
DeBERTa-v3-base + LoRA internals: constants, callbacks, cache helpers, metrics.

All the wiring details that don't belong in the notebook. The public API is in
src/task1/deberta.py - this module is "private" (importable but not called
directly from the notebook).
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.hyperparams import RANDOM_STATE, SENTIMENT_MIN, SENTIMENT_MAX
from ..config.paths import (
    T1_DEBERTA_TIMING_LOG_PATH,
)
from ..utils.io import atomic_to_csv, read_json_or_none
from ..utils.format import format_elapsed


# ====================================================================
# Config constants - must match the values used to produce existing _SUCCESS
# markers. DO NOT rename fields without also adjusting the saved
# _SUCCESS / train_config.json files (or cache will be invalidated).
# ====================================================================
DEBERTA_MODEL_NAME = 'microsoft/deberta-v3-base'
DEBERTA_TUNING_METHOD = 'lora'
DEBERTA_MAX_LENGTH = 256
DEBERTA_MAX_ROWS = None  # Full-data run. Set to 50000 for a pilot.
# Up from L4's 4 epochs: at batch=128 we get 2x fewer gradient updates per epoch
# than L4's batch=64, so +50% epochs preserves total update count at fixed LR.
DEBERTA_EPOCHS = 6
# Sized for LoRA-optimal regime, not max VRAM. Thinking Machines (2025/9, 'LoRA
# Without Regret') shows LoRA degrades noticeably above batch ~128; 96 GB headroom
# matters less than gradient-update count + noise scale.
# Peak VRAM at batch 128 (fp32): ~62 GB on RTX PRO 6000.
DEBERTA_BATCH_SIZE_GPU = 128
DEBERTA_BATCH_SIZE_CPU = 8
DEBERTA_EVAL_BATCH_SIZE_GPU = 256
DEBERTA_EVAL_BATCH_SIZE_CPU = 8
DEBERTA_GRADIENT_ACCUMULATION_STEPS = 1
DEBERTA_DATALOADER_NUM_WORKERS_GPU = 2  # lowered from 8 - some py3.12 + transformers>=4.49 setups hang on higher worker counts
DEBERTA_DATALOADER_NUM_WORKERS_CPU = 0
# Updated math (batch=128, epochs=6, train=789K):
#   total steps = 6 * (789K / 128) ~ 37K
#   eval_steps  = 2000  -> ~18 evals  (tight early-stopping signal, patience=3 fits)
#   save_steps  = 8000  -> ~4 mid-training saves + final  (lean disk usage)
# load_best_model_at_end requires save_steps multiple of eval_steps (8000/2000=4 OK).
DEBERTA_EVAL_STEPS = 2000
DEBERTA_CHECKPOINT_STEPS = 8000
DEBERTA_FULL_PREDICT_CHUNK_ROWS = 200_000
DEBERTA_WARMUP_RATIO = 0.06
DEBERTA_MAX_GRAD_NORM = 1.0
DEBERTA_NONFINITE_METRIC_PENALTY = float(SENTIMENT_MAX - SENTIMENT_MIN)
DEBERTA_LOAD_BEST_MODEL_AT_END = True
DEBERTA_USE_EARLY_STOPPING = True
DEBERTA_REQUIRE_EARLY_STOPPING = True
DEBERTA_EARLY_STOPPING_PATIENCE = 3
DEBERTA_EARLY_STOPPING_THRESHOLD = 0.0

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# In DeBERTa-v3, 'dense' matches attention.output.dense (O proj) + 2 FFN dense layers,
# so this expands to {Q, K, V, O, FFN_up, FFN_down}.
# Trainable: 3.25M (1.73% of 188M), up from QV-only 1.18M (0.64%).
LORA_TARGET_MODULES = ['query_proj', 'key_proj', 'value_proj', 'dense']
LORA_MODULES_TO_SAVE = ['classifier', 'pooler']
# LoRA learning rate: HF Kaggle DeBERTa-v3 LoRA standard for ~1M-row datasets.
LORA_LEARNING_RATE = 1e-4

# Resume controls
DEBERTA_RESUME_TRAINING = True
DEBERTA_REUSE_FINAL_MODEL = True
DEBERTA_REUSE_EXISTING_FULL_SCORES = True
DEBERTA_FORCE_RETRAIN = False
DEBERTA_FORCE_RESCORE = False
DEBERTA_REUSE_TEST_METRICS = True

# Memory-only settings that don't change the trained weights, so they're
# allowed to differ between cache-write and cache-read.
RESUME_COMPATIBLE_CONFIG_KEYS = {
    'batch_size', 'eval_batch_size', 'gradient_accumulation_steps',
    'eval_steps', 'checkpoint_steps',
}


def build_t1_deberta_config(batch_size, eval_batch_size, gradient_accumulation_steps):
    """Build the canonical config dict used for _SUCCESS marker comparison.

    Pass in the runtime-resolved batch sizes (GPU vs CPU) so the cache marker
    captures what was actually used. Does NOT change the trained weights - these
    fields are in RESUME_COMPATIBLE_CONFIG_KEYS.
    """
    return {
        'model_name': DEBERTA_MODEL_NAME,
        'tuning_method': DEBERTA_TUNING_METHOD,
        'max_length': DEBERTA_MAX_LENGTH,
        'max_rows': DEBERTA_MAX_ROWS,
        'split_mode': 'shared_70_10_20_split' if DEBERTA_MAX_ROWS is None else 'independent_pilot_70_10_20_split',
        'epochs': DEBERTA_EPOCHS,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'checkpoint_steps': DEBERTA_CHECKPOINT_STEPS,
        'eval_steps': DEBERTA_EVAL_STEPS,
        'warmup_ratio': DEBERTA_WARMUP_RATIO,
        'max_grad_norm': DEBERTA_MAX_GRAD_NORM,
        'nonfinite_metric_penalty': DEBERTA_NONFINITE_METRIC_PENALTY,
        'load_best_model_at_end': DEBERTA_LOAD_BEST_MODEL_AT_END,
        'use_early_stopping': DEBERTA_USE_EARLY_STOPPING,
        'require_early_stopping': DEBERTA_REQUIRE_EARLY_STOPPING,
        'early_stopping_patience': DEBERTA_EARLY_STOPPING_PATIENCE,
        'early_stopping_threshold': DEBERTA_EARLY_STOPPING_THRESHOLD,
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT,
        'lora_target_modules': LORA_TARGET_MODULES,
        'lora_modules_to_save': LORA_MODULES_TO_SAVE,
        'learning_rate': LORA_LEARNING_RATE,
        'random_state': RANDOM_STATE,
        'sentiment_min': SENTIMENT_MIN,
        'sentiment_max': SENTIMENT_MAX,
    }


def effective_train_batch(config):
    """Effective batch size = per-device batch x gradient-accumulation steps."""
    try:
        return int(config.get('batch_size')) * int(config.get('gradient_accumulation_steps'))
    except (TypeError, ValueError):
        return None


# ====================================================================
# Checkpoint discovery helpers
# ====================================================================
def model_weight_exists(path):
    """True iff one of the standard model-weight filenames exists in `path`."""
    path = Path(path)
    return any(
        (path / filename).exists()
        for filename in [
            'model.safetensors', 'pytorch_model.bin',
            'adapter_model.safetensors', 'adapter_model.bin',
        ]
    )


def lora_adapter_is_complete(path):
    """True iff `path` has both adapter_config.json and a weight file."""
    path = Path(path)
    return (path / 'adapter_config.json').exists() and model_weight_exists(path)


def deberta_final_model_is_complete(path):
    """True iff `path` has _SUCCESS marker + complete LoRA adapter."""
    path = Path(path)
    return (path / '_SUCCESS').exists() and lora_adapter_is_complete(path)


def checkpoint_step(path):
    """Extract the step number from a 'checkpoint-NNNN' directory name."""
    try:
        return int(Path(path).name.split('-')[-1])
    except ValueError:
        return -1


def checkpoint_is_complete(path):
    """True iff `path` is a complete HF Trainer checkpoint."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / 'trainer_state.json').exists()
        and (path / 'optimizer.pt').exists()
        and (path / 'scheduler.pt').exists()
        and model_weight_exists(path)
    )


def get_last_complete_checkpoint(output_dir):
    """Return the path of the highest-step complete checkpoint, or None."""
    output_dir = Path(output_dir)
    checkpoints = [
        path for path in output_dir.glob('checkpoint-*')
        if checkpoint_is_complete(path)
    ]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=checkpoint_step))


# ====================================================================
# Predict-output sanitization + metric computation
# ====================================================================
def sanitize_deberta_predictions(predictions, context):
    """Replace non-finite predictions with safe values, clip to [-1, +1].

    Returns (cleaned_predictions, n_nonfinite_replaced).
    """
    values = np.asarray(predictions, dtype=np.float32).reshape(-1)
    nonfinite_mask = ~np.isfinite(values)
    nonfinite_count = int(nonfinite_mask.sum())
    if nonfinite_count:
        print(
            f"WARNING: {context} produced {nonfinite_count:,} / {len(values):,} non-finite predictions. "
            "Replacing them for logging/scoring, but this usually means the run was numerically unstable."
        )
    values = np.nan_to_num(values, nan=0.0, posinf=SENTIMENT_MAX, neginf=SENTIMENT_MIN)
    return values.clip(SENTIMENT_MIN, SENTIMENT_MAX), nonfinite_count


def make_compute_metrics_fn():
    """Build a HuggingFace-compatible compute_metrics callback.

    Computes MAE and RMSE on the validation set, handling non-finite predictions
    by replacing them with the penalty value.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def compute_deberta_metrics(eval_pred):
        predictions, labels = eval_pred
        preds, nonfinite_count = sanitize_deberta_predictions(predictions, context='DeBERTa validation')
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        valid_mask = np.isfinite(labels) & np.isfinite(preds)
        if not valid_mask.any():
            return {
                'mae': DEBERTA_NONFINITE_METRIC_PENALTY,
                'rmse': DEBERTA_NONFINITE_METRIC_PENALTY,
                'nonfinite_predictions': float(nonfinite_count),
            }
        mae_value = mean_absolute_error(labels[valid_mask], preds[valid_mask])
        rmse_value = np.sqrt(mean_squared_error(labels[valid_mask], preds[valid_mask]))
        if nonfinite_count:
            mae_value = max(mae_value, DEBERTA_NONFINITE_METRIC_PENALTY)
            rmse_value = max(rmse_value, DEBERTA_NONFINITE_METRIC_PENALTY)
        return {
            'mae': float(mae_value),
            'rmse': float(rmse_value),
            'nonfinite_predictions': float(nonfinite_count),
        }

    return compute_deberta_metrics


# ====================================================================
# Timing logging - writes one row per stage to T1_DEBERTA_TIMING_LOG_PATH
# ====================================================================
def _timing_safe_value(value):
    if value is None:
        return ''
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return ''
        return float(value)
    if isinstance(value, (bool, str)):
        return value
    return str(value)


def make_timing_logger(session_id, batch_size, eval_batch_size, max_rows):
    """Build a per-session timing logger.

    Returns a function `log_event(stage, seconds=None, status='complete',
    step=None, epoch=None, extra=None)` that appends a CSV row per call.
    """

    def log_event(stage, seconds=None, status='complete', step=None, epoch=None, extra=None):
        event = {
            'session_id': session_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'stage': stage,
            'status': status,
            'seconds': np.nan if seconds is None else float(seconds),
            'elapsed': '' if seconds is None else format_elapsed(seconds),
            'global_step': np.nan if step is None else int(step),
            'epoch': np.nan if epoch is None else float(epoch),
            'max_rows': 'full' if max_rows is None else int(max_rows),
            'batch_size': int(batch_size),
            'eval_batch_size': int(eval_batch_size),
            'learning_rate': float(LORA_LEARNING_RATE),
        }
        if extra:
            event.update({k: _timing_safe_value(v) for k, v in extra.items()})

        try:
            if T1_DEBERTA_TIMING_LOG_PATH.exists():
                log = pd.read_csv(T1_DEBERTA_TIMING_LOG_PATH)
            else:
                log = pd.DataFrame()
            event_frame = pd.DataFrame([event])
            log = event_frame if log.empty else pd.concat([log, event_frame], ignore_index=True)
            atomic_to_csv(log, T1_DEBERTA_TIMING_LOG_PATH, index=False)
        except Exception as err:
            print(f"WARNING: Could not update DeBERTa timing log: {err}")

    return log_event


# ====================================================================
# Callbacks (lazy-imported to avoid pulling transformers at import time)
# ====================================================================
def make_timing_callback(log_event):
    """Build a TrainerCallback that records timing events."""
    from transformers import TrainerCallback

    class DebertaTimingCallback(TrainerCallback):
        def __init__(self):
            self.train_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start = time.perf_counter()
            log_event('training', status='started',
                      step=getattr(state, 'global_step', None),
                      epoch=getattr(state, 'epoch', None))

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            seconds = None if self.train_start is None else time.perf_counter() - self.train_start
            extra = metrics.copy() if metrics else {}
            log_event('validation', seconds=seconds, status='complete',
                      step=getattr(state, 'global_step', None),
                      epoch=getattr(state, 'epoch', None), extra=extra)

        def on_save(self, args, state, control, **kwargs):
            seconds = None if self.train_start is None else time.perf_counter() - self.train_start
            log_event('checkpoint_save', seconds=seconds, status='complete',
                      step=getattr(state, 'global_step', None),
                      epoch=getattr(state, 'epoch', None),
                      extra={'checkpoint_dir': Path(args.output_dir) / f"checkpoint-{getattr(state, 'global_step', 'unknown')}"})

        def on_train_end(self, args, state, control, **kwargs):
            seconds = None if self.train_start is None else time.perf_counter() - self.train_start
            log_event('training', seconds=seconds, status='complete',
                      step=getattr(state, 'global_step', None),
                      epoch=getattr(state, 'epoch', None),
                      extra={
                          'best_model_checkpoint': getattr(state, 'best_model_checkpoint', ''),
                          'best_metric': getattr(state, 'best_metric', ''),
                      })

    return DebertaTimingCallback()


def make_nan_detect_callback():
    """Build a TrainerCallback that halts training on non-finite loss/grad_norm.

    Saves you from waiting 15 min until the first eval (step 2000) to discover
    the model NaN'd at step 100.
    """
    from transformers import TrainerCallback

    class NaNDetectCallback(TrainerCallback):
        def __init__(self):
            self.tripped = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or self.tripped:
                return
            for key in ('loss', 'grad_norm'):
                value = logs.get(key)
                if value is None:
                    continue
                try:
                    if not np.isfinite(float(value)):
                        self.tripped = True
                        control.should_training_stop = True
                        print(
                            f"\n[FATAL] Non-finite {key}={value} at step {state.global_step}. "
                            "Halting to avoid NaN-poisoning downstream steps.\n"
                            "fp32 mode is already on (forced in Sec 0.1 setup), so this is NOT a precision "
                            "overflow. Likely causes: corrupted labels (check for inf/NaN in y_train), "
                            "out-of-range token IDs, or numerical issue in custom code. "
                            "Inspect data and recent code changes before rerunning."
                        )
                        return
                except (TypeError, ValueError):
                    continue

    return NaNDetectCallback()


# ====================================================================
# Test-cache identity (for cache validity check)
# ====================================================================
def row_id_signature(values):
    """Stable signature of an index array - for verifying test split membership."""
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return {'count': 0, 'first': None, 'last': None, 'sha256': ''}
    return {
        'count': int(len(values)),
        'first': int(values[0]),
        'last': int(values[-1]),
        'sha256': hashlib.sha256(values.tobytes()).hexdigest(),
    }


def final_model_identity(marker):
    """Reduce a _SUCCESS marker JSON to the fields used for cache identity."""
    marker = marker or {}
    return {
        'status': marker.get('status'),
        't1_deberta_config': marker.get('t1_deberta_config'),
        'best_model_checkpoint': marker.get('best_model_checkpoint'),
        'best_metric': marker.get('best_metric'),
        'final_global_step': marker.get('final_global_step'),
    }


def task1_deberta_test_cache_is_valid(cache, expected, model_updated_this_run, reuse_test_metrics):
    """True iff the test-metrics cache matches the current run identity.

    Forgives differences in memory-only config keys (batch_size etc.) because
    they don't change the trained weights, making the cache portable across
    CPU/GPU machines.
    """
    if not reuse_test_metrics or model_updated_this_run:
        return False
    if cache is None or cache.get('status') != 'complete':
        return False
    for key, expected_value in expected.items():
        cached_value = cache.get(key)
        if key == 't1_deberta_config' and isinstance(cached_value, dict) and isinstance(expected_value, dict):
            diff_keys = {k for k in set(expected_value) | set(cached_value)
                         if expected_value.get(k) != cached_value.get(k)}
            if diff_keys and not diff_keys.issubset(RESUME_COMPATIBLE_CONFIG_KEYS):
                return False
            continue
        if key == 'model_identity' and isinstance(cached_value, dict) and isinstance(expected_value, dict):
            inner_cur = expected_value.get('t1_deberta_config') or {}
            inner_old = cached_value.get('t1_deberta_config') or {}
            inner_diff = {k for k in set(inner_cur) | set(inner_old)
                          if inner_cur.get(k) != inner_old.get(k)}
            if inner_diff and not inner_diff.issubset(RESUME_COMPATIBLE_CONFIG_KEYS):
                return False
            for sub_key in set(cached_value) | set(expected_value):
                if sub_key == 't1_deberta_config':
                    continue
                if cached_value.get(sub_key) != expected_value.get(sub_key):
                    return False
            continue
        if cached_value != expected_value:
            return False
    required_metric_keys = ['mae', 'rmse', 'nonfinite_predictions']
    return all(k in cache for k in required_metric_keys)


# ====================================================================
# Resume compatibility check
# ====================================================================
def check_resume_compatibility(last_checkpoint, final_model_exists, current_config,
                               train_config_path, force_retrain):
    """Decide whether to resume from `last_checkpoint` based on config compat.

    Returns the checkpoint path (possibly None if config drift forces a clean run).
    Also prints diagnostic messages.
    """
    if force_retrain:
        return last_checkpoint  # caller controls overwrite

    previous_config = read_json_or_none(train_config_path)
    if previous_config is None and train_config_path.exists():
        print("WARNING: Previous DeBERTa LoRA config file is unreadable, continuing with current config.")
        return last_checkpoint
    if previous_config and last_checkpoint and not final_model_exists:
        changed_keys = [k for k, v in current_config.items() if previous_config.get(k) != v]
        if changed_keys:
            incompatible_keys = [k for k in changed_keys if k not in RESUME_COMPATIBLE_CONFIG_KEYS]
            previous_eff = effective_train_batch(previous_config)
            current_eff = effective_train_batch(current_config)
            batch_changed = any(k in changed_keys for k in ['batch_size', 'gradient_accumulation_steps'])
            compatible_batch_change = (not batch_changed) or (previous_eff == current_eff)
            if incompatible_keys or not compatible_batch_change:
                print("WARNING: DeBERTa LoRA run config changed since the saved checkpoint:", changed_keys)
                print("Ignoring old checkpoint for this run.")
                return None
            print("DeBERTa LoRA run config changed only in resume-compatible memory settings:", changed_keys)
            print("Continuing from checkpoint because the effective train batch size is unchanged.")
    return last_checkpoint
