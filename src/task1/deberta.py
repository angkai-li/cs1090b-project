"""
DeBERTa-v3-base + LoRA fine-tuning for sentiment regression (Task 1).

Public API:
    prepare_modeling_subset(df)          -> df_deberta with df_row_id col
    apply_shared_split(df_deberta, ...)  -> (train_df, val_df, test_df)
    train_or_load(...)                   -> trained Trainer (resumes if cache hit)
    compute_test_metrics(trainer, ...)   -> dict with mae/rmse/nonfinite_count
    score_full_dataset(trainer, df, ...) -> df with score_deberta_v3_base_lora col
    aggregate_daily(df)                  -> daily TS DataFrame

All cache logic lives in deberta_internals.py. This module orchestrates them.
"""

import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.hyperparams import RANDOM_STATE, SENTIMENT_MIN, SENTIMENT_MAX
from ..config.paths import (
    T1_DEBERTA_CHECKPOINT_DIR,
    T1_DEBERTA_FINAL_DIR,
    T1_DEBERTA_LOG_DIR,
    T1_DEBERTA_PROGRESS_PATH,
    T1_DEBERTA_SUCCESS_PATH,
    T1_DEBERTA_TEST_METRICS_PATH,
    T1_DEBERTA_TIMING_PATH,
    T1_DEBERTA_TRAIN_CONFIG_PATH,
    T1_DAILY_DEBERTA_PATH,
    T1_PER_REVIEW_DEBERTA_PATH,
    T1_SPLIT_PATH,
)
from ..config.runtime import USE_CUDA, device
from ..utils.format import format_elapsed
from ..utils.io import atomic_to_csv, atomic_write_text, read_json_or_none
from ..utils.metrics import weighted_mean_with_fallback
from . import deberta_internals as _internals


# ====================================================================
# Step 1: filter df -> modeling subset
# ====================================================================
def prepare_modeling_subset(df):
    """Filter df to rows with valid sentiment + non-trivial text.

    Adds a stable `df_row_id` column referencing the original df index.
    Identical filter to splits.filter_to_modeling_subset() so the shared
    70/10/20 split is portable across baseline + DeBERTa.
    """
    df_deberta = df[['text', 'overall', 'sentiment']].copy()
    df_deberta['text'] = df_deberta['text'].fillna('').astype(str)
    df_deberta = df_deberta[
        df_deberta['sentiment'].notna()
        & df_deberta['overall'].notna()
        & (df_deberta['text'].str.len() > 1)
    ].copy()
    df_deberta = df_deberta.reset_index(drop=False).rename(columns={'index': 'df_row_id'})
    return df_deberta


def apply_shared_split(df_deberta):
    """Apply the cached shared 70/10/20 split to df_deberta. Returns (train, val, test) DataFrames."""
    if not T1_SPLIT_PATH.exists():
        raise FileNotFoundError("Missing shared split cache. Build it via build_shared_70_10_20_split() in src/task1/splits.py first.")

    shared_split = pd.read_csv(T1_SPLIT_PATH)
    expected_ids = set(range(len(df_deberta)))
    cached_ids = set(shared_split['df_model_row_id'].astype(int))
    if cached_ids != expected_ids:
        raise ValueError("Shared split rows don't match current DeBERTa modeling rows. Rerun baseline.")
    if set(shared_split['split']) != {'train', 'val', 'test'}:
        raise ValueError("Shared split must contain train/val/test. Rerun baseline.")

    train_idx = shared_split.loc[shared_split['split'] == 'train', 'df_model_row_id'].astype(int).to_numpy()
    val_idx   = shared_split.loc[shared_split['split'] == 'val',   'df_model_row_id'].astype(int).to_numpy()
    test_idx  = shared_split.loc[shared_split['split'] == 'test',  'df_model_row_id'].astype(int).to_numpy()

    return (
        df_deberta.loc[train_idx].copy(),
        df_deberta.loc[val_idx].copy(),
        df_deberta.loc[test_idx].copy(),
    )


# ====================================================================
# Step 2: tokenizer + datasets
# ====================================================================
def build_tokenizer(reuse_final_adapter=True):
    """Load the DeBERTa-v3 tokenizer (from final adapter dir if it exists, else from HF Hub)."""
    from transformers import AutoTokenizer

    if (reuse_final_adapter and _internals.deberta_final_model_is_complete(T1_DEBERTA_FINAL_DIR)):
        source = str(T1_DEBERTA_FINAL_DIR)
    else:
        source = _internals.DEBERTA_MODEL_NAME
    return AutoTokenizer.from_pretrained(source, use_fast=True)


def build_datasets(train_df, val_df, test_df, tokenizer):
    """Build HF Datasets for train/val/test. Tokenized + label-renamed (sentiment -> labels)."""
    from datasets import Dataset

    def to_ds(df):
        return Dataset.from_pandas(
            pd.DataFrame({
                'text': df['text'].to_numpy(),
                'labels': df['sentiment'].astype(np.float32).to_numpy(),
            }),
            preserve_index=False,
        )

    def tokenize_batch(batch):
        return tokenizer(batch['text'], truncation=True, max_length=_internals.DEBERTA_MAX_LENGTH)

    # num_proc=1: fast tokenizer is Rust-parallel internally; num_proc>1 fights it (HF datasets#620)
    train_ds = to_ds(train_df).map(tokenize_batch, batched=True, remove_columns=['text'], num_proc=1)
    val_ds   = to_ds(val_df).map(tokenize_batch,   batched=True, remove_columns=['text'], num_proc=1)
    test_ds  = to_ds(test_df).map(tokenize_batch,  batched=True, remove_columns=['text'], num_proc=1)
    return train_ds, val_ds, test_ds


# ====================================================================
# Step 3: build / reuse model
# ====================================================================
def build_or_load_model(skip_training):
    """Build a fresh LoRA model OR load from the final adapter directory.

    Returns (model, base_model). The base_model is kept separate so callers can
    free it from VRAM after training.
    """
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    # See deberta_internals.DEBERTA_MODEL_NAME for full rationale on:
    #   - attn_implementation='eager' (sdpa NaNs at step 2000 with bf16+LoRA)
    #   - dtype=torch.float32 (deberta-v3-base pytorch_model.bin ships fp16, force-align)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        _internals.DEBERTA_MODEL_NAME,
        num_labels=1,
        problem_type='regression',
        attn_implementation='eager',
        dtype=torch.float32,
    )

    if skip_training:
        model = PeftModel.from_pretrained(base_model, str(T1_DEBERTA_FINAL_DIR))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        print("Using saved final DeBERTa LoRA adapter, skipping training.")
    else:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=_internals.LORA_R,
            lora_alpha=_internals.LORA_ALPHA,
            lora_dropout=_internals.LORA_DROPOUT,
            target_modules=_internals.LORA_TARGET_MODULES,
            modules_to_save=_internals.LORA_MODULES_TO_SAVE,
            bias='none',
        )
        model = get_peft_model(base_model, lora_config)

    # Cast trainable params to fp32 (safety net for fp16+GradScaler / mixed-dtype adapter loads)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return model, base_model


# ====================================================================
# Step 4: build trainer
# ====================================================================
def build_trainer(model, tokenizer, train_dataset, val_dataset, batch_size, eval_batch_size,
                  dataloader_num_workers, callbacks):
    """Build a HuggingFace Trainer with all the careful kwargs probed from the
    installed transformers version."""
    from transformers import (
        DataCollatorWithPadding,
        EarlyStoppingCallback as _EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    try:
        EarlyStoppingCallback = _EarlyStoppingCallback
    except Exception:
        EarlyStoppingCallback = None

    training_kwargs = {
        'output_dir': str(T1_DEBERTA_CHECKPOINT_DIR),
        'overwrite_output_dir': _internals.DEBERTA_FORCE_RETRAIN,
        'learning_rate': _internals.LORA_LEARNING_RATE,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': eval_batch_size,
        'num_train_epochs': _internals.DEBERTA_EPOCHS,
        'weight_decay': 0.01,
        'warmup_ratio': _internals.DEBERTA_WARMUP_RATIO,
        'max_grad_norm': _internals.DEBERTA_MAX_GRAD_NORM,
        'gradient_accumulation_steps': _internals.DEBERTA_GRADIENT_ACCUMULATION_STEPS,
        'logging_dir': str(T1_DEBERTA_LOG_DIR),
        'logging_steps': 100,
        'save_strategy': 'steps',
        'save_steps': _internals.DEBERTA_CHECKPOINT_STEPS,
        'save_total_limit': 2,
        'load_best_model_at_end': _internals.DEBERTA_LOAD_BEST_MODEL_AT_END,
        'metric_for_best_model': 'rmse',
        'greater_is_better': False,
        # Mixed precision: bf16 NaN's on DeBERTa-v3, fp32 is mandatory.
        'fp16': False,
        'bf16': False,
        'dataloader_num_workers': dataloader_num_workers,
        'dataloader_pin_memory': USE_CUDA,
        'eval_accumulation_steps': 16,
        'report_to': 'none',
        'seed': RANDOM_STATE,
    }
    if USE_CUDA:
        training_kwargs['optim'] = 'adamw_torch_fused'

    # Probe TrainingArguments signature (transformers version compat)
    ta_signature = inspect.signature(TrainingArguments.__init__).parameters
    if 'torch_empty_cache_steps' in ta_signature:
        training_kwargs['torch_empty_cache_steps'] = _internals.DEBERTA_CHECKPOINT_STEPS
    if 'eval_strategy' in ta_signature:
        training_kwargs['eval_strategy'] = 'steps'
    elif 'evaluation_strategy' in ta_signature:
        training_kwargs['evaluation_strategy'] = 'steps'
    elif _internals.DEBERTA_REQUIRE_EARLY_STOPPING:
        raise RuntimeError(
            "This transformers version doesn't support step-based eval. "
            "Upgrade: pip install -U transformers accelerate"
        )
    if 'eval_steps' in ta_signature:
        training_kwargs['eval_steps'] = _internals.DEBERTA_EVAL_STEPS
    if 'group_by_length' in ta_signature:
        training_kwargs['group_by_length'] = True
    if 'save_safetensors' in ta_signature:
        training_kwargs['save_safetensors'] = True
    if 'dataloader_prefetch_factor' in ta_signature and USE_CUDA:
        training_kwargs['dataloader_prefetch_factor'] = 2

    # Drop unsupported kwargs if the installed version has no **kwargs
    supports_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in ta_signature.values()
    )
    if not supports_var_kwargs:
        unsupported = sorted(k for k in training_kwargs if k not in ta_signature)
        if unsupported:
            print("Dropping unsupported TrainingArguments:", unsupported)
            for k in unsupported:
                training_kwargs.pop(k)

    training_args = TrainingArguments(**training_kwargs)
    actual_load_best = bool(getattr(training_args, 'load_best_model_at_end', False))
    if _internals.DEBERTA_LOAD_BEST_MODEL_AT_END and not actual_load_best:
        raise RuntimeError(
            "This transformers version doesn't support load_best_model_at_end. "
            "Upgrade: pip install -U transformers accelerate"
        )

    # Append early-stopping callback if available
    all_callbacks = list(callbacks)
    if _internals.DEBERTA_USE_EARLY_STOPPING and EarlyStoppingCallback is not None and actual_load_best:
        es_kwargs = {'early_stopping_patience': _internals.DEBERTA_EARLY_STOPPING_PATIENCE}
        es_signature = inspect.signature(EarlyStoppingCallback.__init__).parameters
        if 'early_stopping_threshold' in es_signature:
            es_kwargs['early_stopping_threshold'] = _internals.DEBERTA_EARLY_STOPPING_THRESHOLD
        all_callbacks.append(EarlyStoppingCallback(**es_kwargs))
        print(
            "Early stopping enabled:",
            f"metric=eval_rmse, patience={_internals.DEBERTA_EARLY_STOPPING_PATIENCE},",
            f"threshold={_internals.DEBERTA_EARLY_STOPPING_THRESHOLD}",
        )

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': val_dataset,
        'data_collator': DataCollatorWithPadding(tokenizer=tokenizer),
        'compute_metrics': _internals.make_compute_metrics_fn(),
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in trainer_signature:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
    if all_callbacks:
        trainer_kwargs['callbacks'] = all_callbacks

    return Trainer(**trainer_kwargs)


# ====================================================================
# Step 5: high-level train_or_load
# ====================================================================
def resolve_runtime_config():
    """Pick batch_size + dataloader_num_workers based on GPU availability."""
    batch_size = _internals.DEBERTA_BATCH_SIZE_GPU if USE_CUDA else _internals.DEBERTA_BATCH_SIZE_CPU
    eval_batch_size = _internals.DEBERTA_EVAL_BATCH_SIZE_GPU if USE_CUDA else _internals.DEBERTA_EVAL_BATCH_SIZE_CPU
    dl_workers = _internals.DEBERTA_DATALOADER_NUM_WORKERS_GPU if USE_CUDA else _internals.DEBERTA_DATALOADER_NUM_WORKERS_CPU
    return {
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'gradient_accumulation_steps': _internals.DEBERTA_GRADIENT_ACCUMULATION_STEPS,
        'dataloader_num_workers': dl_workers,
    }


def train_or_load(train_df, val_df, test_df, force_retrain=False):
    """Train (with resume if checkpoint exists) or load the final adapter.

    Returns dict with:
        trainer       - HF Trainer (use for predict + free after)
        tokenizer     - tokenizer (use for chunked scoring)
        test_dataset  - tokenized test dataset
        train_dataset, val_dataset
        timing        - dict of timing measurements
        model_updated_this_run - bool
        final_global_step - int
        config        - the t1_deberta_config dict used
    """
    import torch
    from transformers import set_seed
    set_seed(RANDOM_STATE)

    rt = resolve_runtime_config()
    t1_config = _internals.build_t1_deberta_config(
        batch_size=rt['batch_size'],
        eval_batch_size=rt['eval_batch_size'],
        gradient_accumulation_steps=rt['gradient_accumulation_steps'],
    )

    # Step 1: discover existing checkpoint
    T1_DEBERTA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    T1_DEBERTA_LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_checkpoint = None
    if _internals.DEBERTA_RESUME_TRAINING:
        last_checkpoint = _internals.get_last_complete_checkpoint(T1_DEBERTA_CHECKPOINT_DIR)
    final_model_exists = _internals.deberta_final_model_is_complete(T1_DEBERTA_FINAL_DIR)

    # Step 2: check resume compatibility (config drift)
    last_checkpoint = _internals.check_resume_compatibility(
        last_checkpoint, final_model_exists, t1_config,
        T1_DEBERTA_TRAIN_CONFIG_PATH, force_retrain or _internals.DEBERTA_FORCE_RETRAIN,
    )

    # Step 3: check if final adapter is reusable
    if final_model_exists:
        final_marker = read_json_or_none(T1_DEBERTA_SUCCESS_PATH)
        previous_config = final_marker.get('t1_deberta_config') if final_marker else None
        if previous_config is None:
            print("Final LoRA adapter exists but no marker found, it will not be reused.")
            final_model_exists = False
        elif previous_config != t1_config:
            diff_keys = {k for k in set(t1_config) | set(previous_config)
                         if t1_config.get(k) != previous_config.get(k)}
            if diff_keys.issubset(_internals.RESUME_COMPATIBLE_CONFIG_KEYS):
                print("Final LoRA adapter config differs only in memory-only settings:", sorted(diff_keys))
                print("Reusing existing weights (these fields don't change the trained model).")
            else:
                weight_diff = sorted(diff_keys - _internals.RESUME_COMPATIBLE_CONFIG_KEYS)
                print("Final LoRA adapter exists but was created with a different config/seed, won't reuse.")
                print("  Weight-affecting fields that differ:", weight_diff)
                final_model_exists = False
    atomic_write_text(T1_DEBERTA_TRAIN_CONFIG_PATH, json.dumps(t1_config, indent=2))

    print("LoRA checkpoint directory:", T1_DEBERTA_CHECKPOINT_DIR)
    print("LoRA final adapter directory:", T1_DEBERTA_FINAL_DIR)
    print("Last complete LoRA checkpoint:", last_checkpoint or "none")
    print("Complete final LoRA adapter exists:", final_model_exists)

    # Step 4: build tokenizer + datasets + model
    skip_training = (final_model_exists and _internals.DEBERTA_REUSE_FINAL_MODEL
                     and not force_retrain and not _internals.DEBERTA_FORCE_RETRAIN)
    tokenizer = build_tokenizer(reuse_final_adapter=skip_training)
    train_dataset, val_dataset, test_dataset = build_datasets(train_df, val_df, test_df, tokenizer)
    model, base_model = build_or_load_model(skip_training=skip_training)
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    # Step 5: build callbacks + trainer
    session_id = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_event = _internals.make_timing_logger(
        session_id,
        batch_size=rt['batch_size'],
        eval_batch_size=rt['eval_batch_size'],
        max_rows=_internals.DEBERTA_MAX_ROWS,
    )
    timing_cb = _internals.make_timing_callback(log_event)
    nan_cb = _internals.make_nan_detect_callback()
    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset,
                            batch_size=rt['batch_size'],
                            eval_batch_size=rt['eval_batch_size'],
                            dataloader_num_workers=rt['dataloader_num_workers'],
                            callbacks=[timing_cb, nan_cb])

    if USE_CUDA:
        torch.cuda.empty_cache()
    resume_checkpoint = last_checkpoint if (last_checkpoint and _internals.DEBERTA_RESUME_TRAINING
                                            and not skip_training) else None
    resume_step = _internals.checkpoint_step(resume_checkpoint) if resume_checkpoint else 0

    timing = {}
    model_updated_this_run = False

    if not skip_training:
        log_event('cell', status='started',
                  extra={'last_checkpoint': last_checkpoint or '',
                         'final_model_exists': final_model_exists,
                         'train_config_path': T1_DEBERTA_TRAIN_CONFIG_PATH})
        train_start = time.perf_counter()
        if resume_checkpoint:
            print(f"Resuming DeBERTa LoRA training from checkpoint: {resume_checkpoint}")
        else:
            print("Starting DeBERTa LoRA training from the base model.")
        try:
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        except Exception as err:
            timing['training_seconds_before_error'] = time.perf_counter() - train_start
            log_event('training', seconds=timing['training_seconds_before_error'],
                      status='error', step=getattr(trainer.state, 'global_step', None),
                      epoch=getattr(trainer.state, 'epoch', None), extra={'error': repr(err)})
            raise
        timing['training_seconds'] = time.perf_counter() - train_start
        log_event('training', seconds=timing['training_seconds'], status='complete',
                  step=getattr(trainer.state, 'global_step', None),
                  epoch=getattr(trainer.state, 'epoch', None), extra=train_result.metrics)
        print("Training metrics:", train_result.metrics)
        print("DeBERTa training elapsed:", format_elapsed(timing['training_seconds']))

        final_step = int(getattr(trainer.state, 'global_step', 0) or 0)
        if nan_cb.tripped:
            raise RuntimeError(
                f"NaNDetectCallback halted training at step {final_step}. "
                "Refusing to save NaN-poisoned weights as final adapter."
            )
        model_updated_this_run = final_step > resume_step
        if model_updated_this_run:
            print(f"DeBERTa training advanced from step {resume_step} to {final_step}.")
        else:
            print(f"DeBERTa checkpoint was already at step {final_step}, weights unchanged.")

        trainer.save_model(str(T1_DEBERTA_FINAL_DIR))
        tokenizer.save_pretrained(str(T1_DEBERTA_FINAL_DIR))
        marker_payload = {
            'status': 'complete',
            't1_deberta_config': t1_config,
            'metrics': train_result.metrics,
            'timing': {k: float(v) for k, v in timing.items()},
            'best_model_checkpoint': getattr(trainer.state, 'best_model_checkpoint', None),
            'best_metric': getattr(trainer.state, 'best_metric', None),
            'resume_checkpoint': resume_checkpoint,
            'resume_checkpoint_step': int(resume_step),
            'final_global_step': int(final_step),
            'model_updated_this_run': bool(model_updated_this_run),
        }
        atomic_write_text(T1_DEBERTA_SUCCESS_PATH, json.dumps(marker_payload, indent=2))
        if marker_payload['best_model_checkpoint']:
            print("Best validation checkpoint:", marker_payload['best_model_checkpoint'])
            print("Best validation metric:", marker_payload['best_metric'])
        print(f"Saved complete final DeBERTa LoRA adapter to {T1_DEBERTA_FINAL_DIR}")
    else:
        log_event('training', seconds=0.0, status='skipped', extra={'reason': 'reused_final_model'})

    return {
        'trainer': trainer,
        'tokenizer': tokenizer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'log_event': log_event,
        'config': t1_config,
        'model_updated_this_run': model_updated_this_run,
        'timing': timing,
    }


# ====================================================================
# Step 6: test metrics with cache check
# ====================================================================
def compute_test_metrics(state, test_df):
    """Compute MAE/RMSE on test split, with cache support.

    Reads from T1_DEBERTA_TEST_METRICS_PATH if the cached identity matches.
    Otherwise runs trainer.predict() on test_dataset.
    Returns dict {mae, rmse, nonfinite_count, from_cache}.
    """
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    trainer = state['trainer']
    test_dataset = state['test_dataset']
    log_event = state['log_event']
    timing = state['timing']

    final_marker = read_json_or_none(T1_DEBERTA_SUCCESS_PATH) or {}
    expected = {
        't1_deberta_config': state['config'],
        'model_identity': _internals.final_model_identity(final_marker),
        'test_rows': int(len(test_df)),
        'test_split_label': 'shared 70/10/20',
        'test_row_id_signature': _internals.row_id_signature(test_df['df_row_id'].to_numpy()),
    }

    cached = read_json_or_none(T1_DEBERTA_TEST_METRICS_PATH)
    if _internals.task1_deberta_test_cache_is_valid(
        cached, expected,
        state['model_updated_this_run'],
        _internals.DEBERTA_REUSE_TEST_METRICS,
    ):
        timing['test_prediction_seconds'] = 0.0
        log_event('test_prediction', seconds=0.0, status='loaded_existing',
                  extra={'test_rows': len(test_df), 'cache_path': T1_DEBERTA_TEST_METRICS_PATH})
        print(f"Loaded cached DeBERTa test metrics from {T1_DEBERTA_TEST_METRICS_PATH}")
        return {
            'mae': float(cached['mae']),
            'rmse': float(cached['rmse']),
            'nonfinite_count': int(cached.get('nonfinite_predictions', 0)),
            'from_cache': True,
        }

    if state['model_updated_this_run']:
        print("Model updated this run, recomputing DeBERTa test metrics.")
    elif cached is not None:
        print("Existing DeBERTa test-metric cache is stale, recomputing.")

    if USE_CUDA:
        torch.cuda.empty_cache()
    test_start = time.perf_counter()
    test_output = trainer.predict(test_dataset)
    timing['test_prediction_seconds'] = time.perf_counter() - test_start
    log_event('test_prediction', seconds=timing['test_prediction_seconds'], status='complete')
    print("DeBERTa test prediction elapsed:", format_elapsed(timing['test_prediction_seconds']))

    preds, nonfinite_count = _internals.sanitize_deberta_predictions(
        test_output.predictions, context='DeBERTa test prediction')
    y_test = test_df['sentiment'].astype(np.float32).to_numpy()
    mae = float(mean_absolute_error(y_test, preds))
    rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
    if nonfinite_count:
        mae = max(mae, _internals.DEBERTA_NONFINITE_METRIC_PENALTY)
        rmse_val = max(rmse_val, _internals.DEBERTA_NONFINITE_METRIC_PENALTY)

    cache_payload = {
        'status': 'complete',
        **expected,
        'mae': mae,
        'rmse': rmse_val,
        'nonfinite_predictions': int(nonfinite_count),
        'created_at': pd.Timestamp.now().isoformat(),
    }
    atomic_write_text(T1_DEBERTA_TEST_METRICS_PATH, json.dumps(cache_payload, indent=2))
    print(f"Saved DeBERTa test-metric cache to {T1_DEBERTA_TEST_METRICS_PATH}")
    return {'mae': mae, 'rmse': rmse_val, 'nonfinite_count': nonfinite_count, 'from_cache': False}


# ====================================================================
# Step 7: chunked full-data scoring (with progress checkpointing)
# ====================================================================
SCORE_COL = 'score_deberta_v3_base_lora'


def score_full_dataset(df, state, force_rescore=False):
    """Score all rows of df with DeBERTa LoRA. Mutates df in place + returns it.

    Reads progress from T1_DEBERTA_PROGRESS_PATH on cache hit; chunks of
    DEBERTA_FULL_PREDICT_CHUNK_ROWS rows at a time. Skips entirely if a complete
    cached score file already exists.
    """
    import torch
    from datasets import Dataset

    trainer = state['trainer']
    tokenizer = state['tokenizer']
    log_event = state['log_event']
    timing = state['timing']

    if state['model_updated_this_run']:
        force_rescore = True

    full_start = time.perf_counter()
    loaded_existing = False

    if (_internals.DEBERTA_REUSE_EXISTING_FULL_SCORES and not force_rescore
            and T1_PER_REVIEW_DEBERTA_PATH.exists()):
        try:
            existing = pd.read_csv(T1_PER_REVIEW_DEBERTA_PATH, usecols=[SCORE_COL])
            score_values = pd.to_numeric(existing[SCORE_COL], errors='coerce')
            if len(score_values) == len(df) and score_values.notna().all():
                df[SCORE_COL] = score_values.clip(SENTIMENT_MIN, SENTIMENT_MAX).to_numpy()
                loaded_existing = True
                timing['full_scoring_seconds'] = time.perf_counter() - full_start
                log_event('full_scoring', seconds=timing['full_scoring_seconds'],
                          status='loaded_existing',
                          extra={'scored_rows': len(df), 'total_rows': len(df)})
                print(f"Loaded existing full DeBERTa LoRA scores from {T1_PER_REVIEW_DEBERTA_PATH}")
                print("DeBERTa full scoring load elapsed:", format_elapsed(timing['full_scoring_seconds']))
        except Exception as err:
            print(f"Existing DeBERTa LoRA score file couldn't be reused: {err}")

    if not loaded_existing:
        df[SCORE_COL] = np.nan
        progress_df = pd.DataFrame(columns=['row_id', SCORE_COL])

        if T1_DEBERTA_PROGRESS_PATH.exists() and not force_rescore:
            try:
                progress_df = pd.read_csv(T1_DEBERTA_PROGRESS_PATH)
                if {'row_id', SCORE_COL}.issubset(progress_df.columns):
                    progress_df = progress_df[['row_id', SCORE_COL]].copy()
                    progress_df['row_id'] = pd.to_numeric(progress_df['row_id'], errors='coerce')
                    progress_df[SCORE_COL] = pd.to_numeric(progress_df[SCORE_COL], errors='coerce')
                    progress_df = progress_df.dropna()
                    progress_df['row_id'] = progress_df['row_id'].astype(int)
                    progress_df = progress_df[progress_df['row_id'].between(0, len(df) - 1)]
                    progress_df = progress_df.drop_duplicates('row_id', keep='last')
                    df.loc[progress_df['row_id'], SCORE_COL] = progress_df[SCORE_COL].clip(
                        SENTIMENT_MIN, SENTIMENT_MAX).to_numpy()
                    print(f"Loaded DeBERTa LoRA scoring progress: {len(progress_df)} / {len(df)} rows")
                else:
                    progress_df = pd.DataFrame(columns=['row_id', SCORE_COL])
            except Exception as err:
                print(f"Ignoring unreadable DeBERTa LoRA progress file: {err}")
                progress_df = pd.DataFrame(columns=['row_id', SCORE_COL])

        remaining_idx = df.index[df[SCORE_COL].isna()].to_numpy()
        print(f"Remaining DeBERTa LoRA rows to score: {len(remaining_idx)}")

        def tokenize_batch(batch):
            return tokenizer(batch['text'], truncation=True,
                             max_length=_internals.DEBERTA_MAX_LENGTH)

        chunk = _internals.DEBERTA_FULL_PREDICT_CHUNK_ROWS
        for start in range(0, len(remaining_idx), chunk):
            idx = remaining_idx[start:start + chunk]
            chunk_frame = pd.DataFrame({
                'text': df.loc[idx, 'text'].fillna('').astype(str).to_numpy()
            })
            chunk_ds = Dataset.from_pandas(chunk_frame, preserve_index=False).map(
                tokenize_batch, batched=True, remove_columns=['text'], num_proc=1)

            if USE_CUDA:
                torch.cuda.empty_cache()
            chunk_out = trainer.predict(chunk_ds)
            chunk_scores, chunk_nonfin = _internals.sanitize_deberta_predictions(
                chunk_out.predictions,
                context=f'DeBERTa full scoring rows {int(idx[0])}-{int(idx[-1])}')
            df.loc[idx, SCORE_COL] = chunk_scores

            chunk_progress = pd.DataFrame({'row_id': idx, SCORE_COL: chunk_scores})
            progress_df = (chunk_progress if progress_df.empty
                           else pd.concat([progress_df, chunk_progress], ignore_index=True))
            progress_df['row_id'] = progress_df['row_id'].astype(int)
            progress_df = progress_df.drop_duplicates('row_id', keep='last').sort_values('row_id')
            atomic_to_csv(progress_df, T1_DEBERTA_PROGRESS_PATH, index=False)
            log_event('full_scoring_progress', seconds=time.perf_counter() - full_start,
                      status='progress',
                      extra={'scored_rows': len(progress_df), 'total_rows': len(df),
                             'chunk_rows': len(idx), 'nonfinite_predictions': chunk_nonfin})
            print(f"Saved DeBERTa LoRA scoring progress: {len(progress_df)} / {len(df)} rows")

    if df[SCORE_COL].isna().any():
        raise RuntimeError("Some DeBERTa LoRA scores are still missing. Rerun this cell.")
    if 'full_scoring_seconds' not in timing:
        timing['full_scoring_seconds'] = time.perf_counter() - full_start
        log_event('full_scoring', seconds=timing['full_scoring_seconds'], status='complete',
                  extra={'scored_rows': int(df[SCORE_COL].notna().sum()), 'total_rows': len(df)})
        print("DeBERTa full scoring elapsed:", format_elapsed(timing['full_scoring_seconds']))

    return df


# ====================================================================
# Step 8: daily aggregation
# ====================================================================
def aggregate_daily(df):
    """Aggregate DeBERTa per-review scores to a daily time series."""
    def agg(group):
        return pd.Series({
            f'{SCORE_COL}_naive': group[SCORE_COL].mean(),
            f'{SCORE_COL}_weighted': weighted_mean_with_fallback(group, SCORE_COL),
            'num_reviews': len(group),
        })

    return df.groupby('review_day')[[SCORE_COL, 'informativeness_log']].apply(agg).reset_index()


# ====================================================================
# Step 9: save all outputs
# ====================================================================
def save_full_outputs(df, daily_ts, test_metrics, state, timing_path=T1_DEBERTA_TIMING_PATH):
    """Save per-review CSV + daily-aggregation CSV + update timing files."""
    export_cols = [
        'asin', 'unixReviewTime', 'review_date', 'review_day',
        'overall', 'sentiment', 'informativeness_log', SCORE_COL,
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    atomic_to_csv(df[export_cols], T1_PER_REVIEW_DEBERTA_PATH, index=False)
    atomic_to_csv(daily_ts, T1_DAILY_DEBERTA_PATH, index=False)

    log_event = state['log_event']
    timing = state['timing']
    timing['cell_seconds'] = sum(v for v in timing.values() if isinstance(v, (int, float))) + 0.001
    log_event('cell', seconds=timing['cell_seconds'], status='complete')
    atomic_write_text(timing_path, json.dumps({k: float(v) for k, v in timing.items()}, indent=2))

    # Append timing to _SUCCESS marker
    final_marker = read_json_or_none(T1_DEBERTA_SUCCESS_PATH) or {}
    final_marker['timing'] = {k: float(v) for k, v in timing.items()}
    atomic_write_text(T1_DEBERTA_SUCCESS_PATH, json.dumps(final_marker, indent=2))
    print(f"Saved DeBERTa LoRA review-level score file to {T1_PER_REVIEW_DEBERTA_PATH}")
    print(f"Saved DeBERTa LoRA daily score file to {T1_DAILY_DEBERTA_PATH}")
