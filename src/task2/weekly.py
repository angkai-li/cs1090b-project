"""
Task 2 Sec 9.W Weekly pipeline (full parallel of Sec 9.3-Sec 9.9 + Sec 9.2 legacy).

Provides a single `run_weekly_pipeline(df)` entry that mirrors monthly behaviour
on the weekly DeBERTa panel. Reuses the monthly model classes and helpers - the
only differences are: (1) input panel is weekly, (2) cache paths use _weekly
suffix, (3) min-series threshold is R1_MIN_SERIES_WEEKS=24 instead of R1_MIN_SERIES_MONTHS=12.

For brevity this module relies on existing monthly utilities. See individual
route modules for the underlying training/inference logic.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config.hyperparams import (
    ALPHA, BETA, EMBED_DIM, LAMBDA_L2, MOMENT_TARGET_LEN, N_SYNTHETIC,
    R1_MIN_SERIES_WEEKS, R1_MIN_SIZE, RANDOM_STATE, RARE_THRESHOLD_MONTHS,
    WINDOW_LEN_R2, WINDOW_LEN_R3, WINDOW_LEN_R4,
)
from ..config.paths import (
    T2_HYBRID_WEEKLY_CP_PATH, T2_HYBRID_WEEKLY_META_PATH,
    T2_LEGACY_DIR, T2_PANEL_WEEKLY_PATH,
    T2_R1_DIR, T2_R2_DIR, T2_R3_DIR, T2_R4_DIR,
    T2_R2_MODEL_DIR, T2_R3_MODEL_DIR, T2_R4_MODEL_DIR,
    T2_RECORDS_DIR, T2_RECORDS_PATH, T2_ROUTE_CONSISTENCY_WEEKLY_PATH,
    T2_PHASE_CLUSTERED_WEEKLY_PATH, T2_PHASE_PROFILES_WEEKLY_PATH,
    T2_EVAL_WEEKLY_META_PATH, T2_EVAL_WEEKLY_DIR,
    T2_ROUTE_TRAIN_DATA_PATH,
)
from ..config.runtime import device, HAS_MOMENT, HAS_RUPTURES, HAS_TRANSFORMERS, USE_BF16, USE_CUDA
from ..utils.format import format_elapsed
from ..utils.io import (
    atomic_savez, atomic_save_pretrained, atomic_to_csv, atomic_to_pickle,
    atomic_torch_save, atomic_write_text, read_json_or_none,
)
from ..utils.losses import FocalLoss
from ..utils.training import BestStateTracker, split_train_val
from .autocpd import AutoCPDClassifier, build_training_data, detect_cps_autocpd
from .evaluate import run_evaluation
from .hybrid import run_hybrid_decoding
from .legacy import build_legacy_monthly_panel  # we'll adapt for weekly
from .moment import (
    _MomentSideChannel, _build_moment_base, _reload_best_lora_model,
    detect_cps_moment, make_moment_input_mask, prepare_moment_input,
)
from .pelt import run_route1
from .records import (
    build_records_from_panel, shrinkage_loss, weighted_sentiment_score,
)
from .synth import empirical_sigma_norm
from .tst import (
    TSTEncoder, detect_cps_tst, train_phase1_mvp, train_phase2_finetune,
)


# ====================================================================
# Path translation: monthly -> weekly
# ====================================================================
def _to_weekly(p):
    """Replace '_monthly' with '_weekly' in path; fall back to appending."""
    p = Path(p)
    s = str(p)
    if '_monthly' in s:
        return Path(s.replace('_monthly', '_weekly'))
    if p.suffix:
        return p.parent / (p.stem + '_weekly' + p.suffix)
    return p.parent / (p.name + '_weekly')


# Weekly path constants
T2_RECORDS_WEEKLY_DIR = _to_weekly(T2_RECORDS_DIR)
T2_RECORDS_WEEKLY_PATH = T2_RECORDS_WEEKLY_DIR / 'records.pkl'
T2_ROUTE_TRAIN_DATA_WEEKLY_PATH = _to_weekly(T2_ROUTE_TRAIN_DATA_PATH)
T2_R1_WEEKLY_DIR = _to_weekly(T2_R1_DIR)
T2_R1_WEEKLY_CP_PATH = T2_R1_WEEKLY_DIR / 'change_points.csv'
T2_R1_WEEKLY_LOOKUP_PATH = T2_R1_WEEKLY_DIR / 'pelt_lookup.pkl'
T2_R1_WEEKLY_META_PATH = T2_R1_WEEKLY_DIR / 'meta.json'

T2_R2_WEEKLY_DIR = _to_weekly(T2_R2_DIR)
T2_R2_WEEKLY_CP_PATH = T2_R2_WEEKLY_DIR / 'change_points.csv'
T2_R2_WEEKLY_PROBS_PATH = T2_R2_WEEKLY_DIR / 'probs.pkl'
T2_R2_WEEKLY_MODEL_DIR = _to_weekly(T2_R2_MODEL_DIR)
T2_R2_WEEKLY_MODEL_PATH = T2_R2_WEEKLY_MODEL_DIR / 'model.pt'

T2_R3_WEEKLY_DIR = _to_weekly(T2_R3_DIR)
T2_R3_WEEKLY_CP_PATH = T2_R3_WEEKLY_DIR / 'change_points.csv'
T2_R3_WEEKLY_PROBS_PATH = T2_R3_WEEKLY_DIR / 'probs.pkl'
T2_R3_WEEKLY_MODEL_DIR = _to_weekly(T2_R3_MODEL_DIR)
T2_R3_WEEKLY_MODEL_PATH = T2_R3_WEEKLY_MODEL_DIR / 'model.pt'

T2_R4_WEEKLY_DIR = _to_weekly(T2_R4_DIR)
T2_R4_WEEKLY_CP_PATH = T2_R4_WEEKLY_DIR / 'change_points.csv'
T2_R4_WEEKLY_PROBS_PATH = T2_R4_WEEKLY_DIR / 'probs.pkl'
T2_R4_WEEKLY_MODEL_DIR = _to_weekly(T2_R4_MODEL_DIR)
T2_R4_WEEKLY_LORA_DIR = T2_R4_WEEKLY_MODEL_DIR / 'lora_adapter'
T2_R4_WEEKLY_SIDE_PATH = T2_R4_WEEKLY_MODEL_DIR / 'side_channel.pt'

T2_LEGACY_WEEKLY_DIR = _to_weekly(T2_LEGACY_DIR)
T2_LEGACY_WEEKLY_PANEL_PATH = T2_LEGACY_WEEKLY_DIR / 'weekly_panel.csv'
T2_LEGACY_WEEKLY_CP_PATH = T2_LEGACY_WEEKLY_DIR / 'change_points.csv'
T2_LEGACY_WEEKLY_META_PATH = T2_LEGACY_WEEKLY_DIR / 'meta.json'


# ====================================================================
# Stage 1 - Weekly data prep (records + PELT weak labels + training data)
# ====================================================================
def prepare_weekly_data(force_rerun=False, verbose=True):
    """Build weekly records from cached weekly panel; run PELT for weak labels;
    assemble X_train_weekly / y_train_weekly / asin_train_weekly."""
    T2_RECORDS_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    T2_R1_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

    weekly_panel_path = T2_PANEL_WEEKLY_PATH
    if not weekly_panel_path.exists():
        raise FileNotFoundError(f"Missing weekly panel: {weekly_panel_path}")

    if (not force_rerun and T2_RECORDS_WEEKLY_PATH.exists()
            and T2_R1_WEEKLY_LOOKUP_PATH.exists()
            and T2_ROUTE_TRAIN_DATA_WEEKLY_PATH.exists()):
        with open(T2_RECORDS_WEEKLY_PATH, 'rb') as f:
            records_weekly = pickle.load(f)
        with open(T2_R1_WEEKLY_LOOKUP_PATH, 'rb') as f:
            pelt_lookup_weekly = pickle.load(f)
        d = np.load(T2_ROUTE_TRAIN_DATA_WEEKLY_PATH)
        X_train_w = d['X_train']
        y_train_w = d['y_train']
        asin_train_w = d['asin_train']
        n_products_weekly = max(r['asin_id'] for r in records_weekly) + 1
        n_per = torch.zeros(n_products_weekly, dtype=torch.float32)
        for r in records_weekly:
            n_per[r['asin_id']] += float(len(r['series_norm']))
        n_per = n_per.clamp(min=1.0)
        weight_decay_scale_w = 1.0 / torch.sqrt(n_per)
        if verbose:
            print(f"Loaded weekly data prep + R1 from cache: {len(records_weekly)} records, "
                  f"{X_train_w.shape[0]} training rows  (cache hit)")
        return dict(records_weekly=records_weekly, pelt_lookup_weekly=pelt_lookup_weekly,
                    X_train_weekly=X_train_w, y_train_weekly=y_train_w,
                    asin_train_weekly=asin_train_w, n_products_weekly=n_products_weekly,
                    weight_decay_scale_weekly=weight_decay_scale_w)

    t0 = time.time()
    if verbose:
        print(f"Loaded weekly panel: {weekly_panel_path}")
    weekly_panel = pd.read_csv(weekly_panel_path, parse_dates=['review_date'])

    # Build records using same logic as monthly (z-score norm + RARE bucket + records)
    records_weekly, asin_to_id_w, n_products_weekly, weight_decay_scale_w = (
        build_records_from_panel(weekly_panel, rare_threshold_months=RARE_THRESHOLD_MONTHS,
                                  embed_dim=EMBED_DIM, verbose=verbose)
    )
    atomic_to_pickle(records_weekly, T2_RECORDS_WEEKLY_PATH)

    # Run PELT on weekly series (with min_series_len = R1_MIN_SERIES_WEEKS)
    classical_df_w, pelt_lookup_weekly, _ = run_route1(
        records_weekly, n_jobs=8, run_calibration=False,
        run_penalty_sensitivity=False, min_series_len=R1_MIN_SERIES_WEEKS,
        cp_path=T2_R1_WEEKLY_CP_PATH, lookup_path=T2_R1_WEEKLY_LOOKUP_PATH,
        meta_path=T2_R1_WEEKLY_META_PATH, out_dir=T2_R1_WEEKLY_DIR,
        verbose=verbose,
    )

    sigma_w = empirical_sigma_norm(records_weekly)
    if verbose:
        print(f"Empirical sigma of weekly sentiment_norm: {sigma_w:.4f}")

    # Build training data (synthetic + weak labels)
    X_train_w, y_train_w, asin_train_w = build_training_data(
        records_weekly, pelt_lookup_weekly, n_products_weekly,
        window_len=WINDOW_LEN_R2, sigma_norm=sigma_w, verbose=verbose,
    )
    atomic_savez(T2_ROUTE_TRAIN_DATA_WEEKLY_PATH,
                 X_train=X_train_w, y_train=y_train_w, asin_train=asin_train_w)

    if verbose:
        print(f"Weekly data prep + R1 elapsed: {format_elapsed(time.time() - t0)}")
    return dict(records_weekly=records_weekly, pelt_lookup_weekly=pelt_lookup_weekly,
                X_train_weekly=X_train_w, y_train_weekly=y_train_w,
                asin_train_weekly=asin_train_w, n_products_weekly=n_products_weekly,
                weight_decay_scale_weekly=weight_decay_scale_w)


# ====================================================================
# Stage 2 - Weekly Route 2 AutoCPD
# ====================================================================
def train_route2_weekly(weekly_state, max_epochs=60, patience=10, batch_size=512,
                        lr=1e-3, weight_decay=1e-4, force_rerun=False, verbose=True):
    """Train weekly AutoCPD. Reuses monthly AutoCPDClassifier."""
    T2_R2_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    T2_R2_WEEKLY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (not force_rerun and T2_R2_WEEKLY_PROBS_PATH.exists()
            and T2_R2_WEEKLY_MODEL_PATH.exists()):
        with open(T2_R2_WEEKLY_PROBS_PATH, 'rb') as f:
            probs_w = pickle.load(f)
        if verbose:
            print(f"Loaded Weekly R2 from cache ({len(probs_w)} products). "
                  "To force rebuild, delete the model/probs/csv files.")
        return probs_w

    t0 = time.time()
    n_products_w = weekly_state['n_products_weekly']
    X_train_w = weekly_state['X_train_weekly']
    y_train_w = weekly_state['y_train_weekly']
    asin_train_w = weekly_state['asin_train_weekly']
    weight_decay_scale_w = weekly_state['weight_decay_scale_weekly']
    records_weekly = weekly_state['records_weekly']

    if verbose:
        print("\n=== Weekly Route 2: AutoCPD training + inference ===")
    (X_tr, y_tr, a_tr), (X_va, y_va, a_va), _, _ = split_train_val(
        X_train_w, y_train_w, asin_train_w, val_frac=0.10, seed=RANDOM_STATE)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(a_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(a_va), torch.from_numpy(y_va)),
        batch_size=batch_size, shuffle=False, num_workers=0)

    model = AutoCPDClassifier(n_products=n_products_w).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    tracker = BestStateTracker(model)
    patience_counter = 0
    for epoch in range(max_epochs):
        model.train()
        train_sum = 0.0
        for xb, ab, yb in train_loader:
            xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
            out = model(xb, ab)
            loss = loss_fn(out, yb) + shrinkage_loss(model.embed, weight_decay_scale_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item() * xb.size(0)
        train_loss = train_sum / len(train_loader.dataset)
        scheduler.step()
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, ab, yb in val_loader:
                xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
                val_sum += loss_fn(model(xb, ab), yb).item() * xb.size(0)
        val_loss = val_sum / len(val_loader.dataset)
        improved = tracker.update(val_loss, epoch + 1)
        patience_counter = 0 if improved else patience_counter + 1
        if verbose and ((epoch + 1) % 5 == 0 or improved or patience_counter >= patience):
            mark = " *new best*" if improved else ""
            print(f"  epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}{mark}")
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    tracker.restore()
    if verbose:
        print(f"  Weekly R2: {tracker.summary()}")
    atomic_torch_save(model.state_dict(), T2_R2_WEEKLY_MODEL_PATH)

    probs_w = {}
    half = WINDOW_LEN_R2 // 2
    for r in records_weekly:
        cps, probs = detect_cps_autocpd(model, r['series_norm'], r['asin_id'], window=WINDOW_LEN_R2)
        aligned = np.zeros(len(r['series_norm']), dtype=np.float32)
        for i, p in enumerate(probs):
            idx = i + half
            if 0 <= idx < len(aligned):
                aligned[idx] = p
        probs_w[r['asin']] = aligned
    atomic_to_pickle(probs_w, T2_R2_WEEKLY_PROBS_PATH)
    if verbose:
        print(f"  Weekly R2 elapsed: {format_elapsed(time.time() - t0)}")
    return probs_w


# ====================================================================
# Stage 3 - Weekly Route 3 TST
# ====================================================================
def train_route3_weekly(weekly_state, force_rerun=False, verbose=True):
    T2_R3_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    T2_R3_WEEKLY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (not force_rerun and T2_R3_WEEKLY_PROBS_PATH.exists()
            and T2_R3_WEEKLY_MODEL_PATH.exists()):
        with open(T2_R3_WEEKLY_PROBS_PATH, 'rb') as f:
            probs_w = pickle.load(f)
        if verbose:
            print(f"Loaded Weekly R3 from cache ({len(probs_w)} products).")
        return probs_w

    t0 = time.time()
    n_products_w = weekly_state['n_products_weekly']
    records_weekly = weekly_state['records_weekly']
    X_train_w = weekly_state['X_train_weekly']
    y_train_w = weekly_state['y_train_weekly']
    asin_train_w = weekly_state['asin_train_weekly']
    weight_decay_scale_w = weekly_state['weight_decay_scale_weekly']

    if verbose:
        print("\n=== Weekly Route 3: TST + MVP training + inference ===")

    model = train_phase1_mvp(records_weekly, n_products=n_products_w, verbose=verbose)
    atomic_torch_save(model.state_dict(), T2_R3_WEEKLY_MODEL_DIR / 'phase1_only.pt')
    model = train_phase2_finetune(model, X_train_w, y_train_w, asin_train_w,
                                   weight_decay_scale_w, verbose=verbose)
    atomic_torch_save(model.state_dict(), T2_R3_WEEKLY_MODEL_PATH)

    probs_w = {}
    half = WINDOW_LEN_R3 // 2
    for r in records_weekly:
        cps, probs = detect_cps_tst(model, r['series_norm'], r['asin_id'], window=WINDOW_LEN_R3)
        aligned = np.zeros(len(r['series_norm']), dtype=np.float32)
        for i, p in enumerate(probs):
            idx = i + half
            if 0 <= idx < len(aligned):
                aligned[idx] = p
        probs_w[r['asin']] = aligned
    atomic_to_pickle(probs_w, T2_R3_WEEKLY_PROBS_PATH)
    if verbose:
        print(f"  Weekly R3 elapsed: {format_elapsed(time.time() - t0)}")
    return probs_w


# ====================================================================
# Stage 4 - Weekly Route 4 MOMENT
# ====================================================================
def train_route4_weekly(weekly_state, max_epochs=14, patience=4, force_rerun=False, verbose=True):
    T2_R4_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    T2_R4_WEEKLY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Cache hit takes priority over CUDA availability - a CPU-only machine can still
    # load pre-trained MOMENT weights and probs computed on a GPU machine.
    if (not force_rerun and T2_R4_WEEKLY_LORA_DIR.exists()
            and T2_R4_WEEKLY_SIDE_PATH.exists()
            and T2_R4_WEEKLY_PROBS_PATH.exists()):
        with open(T2_R4_WEEKLY_PROBS_PATH, 'rb') as f:
            probs_w = pickle.load(f)
        if verbose:
            print(f"Loaded Weekly R4 from cache ({len(probs_w)} products).")
        return probs_w

    if not (HAS_MOMENT and HAS_TRANSFORMERS and USE_CUDA):
        if verbose:
            print("Skipping Weekly Route 4 (MOMENT/transformers/CUDA missing)")
        return {}

    from peft import LoraConfig, PeftModel, get_peft_model

    t0 = time.time()
    n_products_w = weekly_state['n_products_weekly']
    records_weekly = weekly_state['records_weekly']
    X_train_w = weekly_state['X_train_weekly']
    y_train_w = weekly_state['y_train_weekly']
    asin_train_w = weekly_state['asin_train_weekly']
    weight_decay_scale_w = weekly_state['weight_decay_scale_weekly']

    if verbose:
        print("\n=== Weekly Route 4: MOMENT + LoRA training + inference ===")

    base = _build_moment_base().to(device)
    lora_cfg = LoraConfig(r=32, lora_alpha=64,
                          target_modules=["q", "k", "v", "o"],
                          lora_dropout=0.05, bias="none")
    model = get_peft_model(base, lora_cfg)
    if verbose:
        model.print_trainable_parameters()
    side = _MomentSideChannel(n_products=n_products_w).to(device)
    focal = FocalLoss(alpha_pos=0.75, gamma=2.0)

    X_moment = np.stack([prepare_moment_input(w, MOMENT_TARGET_LEN) for w in X_train_w])
    (X_tr, y_tr, a_tr), (X_va, y_va, a_va), _, _ = split_train_val(
        X_moment, y_train_w, asin_train_w, val_frac=0.10, seed=RANDOM_STATE)

    amp_enabled = USE_BF16
    batch_size = 192 if amp_enabled else 32
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(a_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True, pin_memory=USE_CUDA, num_workers=2)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(a_va), torch.from_numpy(y_va)),
        batch_size=batch_size, shuffle=False, pin_memory=USE_CUDA, num_workers=2)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(side.parameters()), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    tracker_side = BestStateTracker(side)
    best_epoch, best_val, patience_counter = -1, float('inf'), 0

    for epoch in range(max_epochs):
        model.train()
        side.train()
        train_sum = 0.0
        for xb, ab, yb in train_loader:
            xb = xb.to(device).unsqueeze(1)
            ab = ab.to(device)
            yb = yb.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                input_mask = make_moment_input_mask(WINDOW_LEN_R4, xb.shape[0], xb.device)
                final_logits = model(x_enc=xb, input_mask=input_mask).logits + side(ab)
                cls_loss = focal(final_logits, yb)
            loss = cls_loss + shrinkage_loss(side.embed, weight_decay_scale_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item() * xb.size(0)
        train_loss = train_sum / len(train_loader.dataset)
        scheduler.step()

        model.eval()
        side.eval()
        val_sum = 0.0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                              enabled=amp_enabled):
            for xb, ab, yb in val_loader:
                xb = xb.to(device).unsqueeze(1)
                ab = ab.to(device)
                yb = yb.to(device)
                input_mask = make_moment_input_mask(WINDOW_LEN_R4, xb.shape[0], xb.device)
                final_logits = model(x_enc=xb, input_mask=input_mask).logits + side(ab)
                val_sum += focal(final_logits, yb).item() * xb.size(0)
        val_loss = val_sum / len(val_loader.dataset)
        improved = tracker_side.update(val_loss, epoch + 1)
        patience_counter = 0 if improved else patience_counter + 1
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            atomic_save_pretrained(model, T2_R4_WEEKLY_LORA_DIR)
        if verbose:
            print(f"  MOMENT epoch {epoch+1}/{max_epochs}  train_cls={train_loss:.4f}  "
                  f"val_cls={val_loss:.4f}" + (" *new best*" if improved else ""))
        if patience_counter >= patience:
            break
    tracker_side.restore()
    if verbose:
        print(f"  Weekly R4: best epoch={best_epoch}  val_cls={best_val:.4f}")

    # Reload best LoRA
    base2 = _build_moment_base().to(device)
    model = PeftModel.from_pretrained(base2, str(T2_R4_WEEKLY_LORA_DIR)).to(device)
    atomic_torch_save(side.state_dict(), T2_R4_WEEKLY_SIDE_PATH)

    # Inference
    probs_w = {}
    half = WINDOW_LEN_R4 // 2
    for r in records_weekly:
        cps, probs = detect_cps_moment(
            model, side, r['series_norm'], r['asin_id'],
            window=WINDOW_LEN_R4, target_len=MOMENT_TARGET_LEN, amp_enabled=amp_enabled)
        aligned = np.zeros(len(r['series_norm']), dtype=np.float32)
        for i, p in enumerate(probs):
            idx = i + half
            if 0 <= idx < len(aligned):
                aligned[idx] = p
        probs_w[r['asin']] = aligned
    atomic_to_pickle(probs_w, T2_R4_WEEKLY_PROBS_PATH)
    if verbose:
        print(f"  Weekly R4 elapsed: {format_elapsed(time.time() - t0)}")
    return probs_w


# ====================================================================
# Stage 5/6 - Weekly hybrid + evaluation (reuses monthly functions)
# ====================================================================
def run_weekly_hybrid(weekly_state, neural_probs_w, force_rerun=False, verbose=True):
    return run_hybrid_decoding(
        weekly_state['records_weekly'], neural_probs_w,
        cp_path=T2_HYBRID_WEEKLY_CP_PATH, meta_path=T2_HYBRID_WEEKLY_META_PATH,
        min_series_len=R1_MIN_SERIES_WEEKS, force_rerun=force_rerun, verbose=verbose,
    )


def run_weekly_evaluation(weekly_state, classical_df_w, force_rerun=False, verbose=True):
    return run_evaluation(
        weekly_state['records_weekly'], classical_df_w, weekly_state['pelt_lookup_weekly'],
        r2_probs_path=T2_R2_WEEKLY_PROBS_PATH, r3_probs_path=T2_R3_WEEKLY_PROBS_PATH,
        r4_probs_path=T2_R4_WEEKLY_PROBS_PATH,
        r2_cp_path=T2_R2_WEEKLY_CP_PATH, r3_cp_path=T2_R3_WEEKLY_CP_PATH,
        r4_cp_path=T2_R4_WEEKLY_CP_PATH,
        consistency_path=T2_ROUTE_CONSISTENCY_WEEKLY_PATH,
        phase_profiles_path=T2_PHASE_PROFILES_WEEKLY_PATH,
        phase_clustered_path=T2_PHASE_CLUSTERED_WEEKLY_PATH,
        meta_path=T2_EVAL_WEEKLY_META_PATH,
        min_series_len=R1_MIN_SERIES_WEEKS, granularity_label='Weekly',
        force_rerun=force_rerun, verbose=verbose,
    )


# ====================================================================
# Stage 7 - Weekly legacy rbf-PELT (weekly aggregation)
# ====================================================================
def run_weekly_legacy(df, force_rerun=False, verbose=True):
    """Run rbf-PELT baseline on a (asin, week) panel."""
    T2_LEGACY_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    # Cache hit takes priority - load even if ruptures isn't installed locally.
    if (not force_rerun and T2_LEGACY_WEEKLY_CP_PATH.exists()
            and T2_LEGACY_WEEKLY_PANEL_PATH.exists()):
        seg = pd.read_csv(T2_LEGACY_WEEKLY_CP_PATH)
        if verbose:
            print(f"Loaded weekly legacy from cache: {seg.shape}")
        return seg
    if not HAS_RUPTURES:
        if verbose:
            print("Skipping weekly legacy (ruptures missing)")
        return None

    t0 = time.time()
    # Build weekly panel: just like build_legacy_monthly_panel but with W groupby
    df_in = df.copy()
    score_col = next((c for c in ['score_deberta_v3_base_lora', 'score_tfidf_ridge',
                                   'score_vader', 'sentiment'] if c in df_in.columns), None)
    if score_col is None:
        raise RuntimeError("Legacy weekly baseline needs Task 1 scores")

    def _to_unit(s):
        s = pd.to_numeric(s, errors='coerce')
        if s.dropna().empty: return s
        mn, mx = s.min(), s.max()
        if pd.notna(mn) and pd.notna(mx) and 0 <= mn <= mx <= 1: return s
        if pd.isna(mn) or pd.isna(mx) or mn == mx: return pd.Series(0.5, index=s.index, dtype=float)
        return (s - mn) / (mx - mn)
    df_in['task1_score_raw'] = pd.to_numeric(df_in[score_col], errors='coerce')
    df_in['task1_score_0_1'] = _to_unit(df_in['task1_score_raw'])
    df_in['amazon_sentiment_raw'] = pd.to_numeric(df_in['sentiment'], errors='coerce')
    df_in['amazon_sentiment_0_1'] = _to_unit(df_in['amazon_sentiment_raw'])
    df_in['rating_numeric'] = pd.to_numeric(df_in['overall'], errors='coerce')
    df_in['verified_numeric'] = df_in['verified'].astype(float)
    df_in['helpful_numeric'] = pd.to_numeric(df_in['vote'], errors='coerce')

    weekly_panel = (
        df_in.groupby(['asin', pd.Grouper(key='review_date', freq='W')])
        .agg(
            avg_task1_score_0_1=('task1_score_0_1', 'mean'),
            avg_amazon_sentiment_0_1=('amazon_sentiment_0_1', 'mean'),
            avg_star_rating=('rating_numeric', 'mean'),
            review_count=('task1_score_raw', 'size'),
            verified_ratio=('verified_numeric', 'mean'),
            task1_score_variance=('task1_score_raw', 'var'),
            helpful_vote_mean=('helpful_numeric', 'mean'),
        )
        .reset_index()
        .rename(columns={'review_date': 'time_period'})
    )
    for col in ['verified_ratio', 'task1_score_variance', 'helpful_vote_mean']:
        weekly_panel[col] = weekly_panel[col].fillna(0.0)

    # Run rbf-PELT per product (reusing legacy.segment_with_rbf_pelt)
    from .legacy import segment_with_rbf_pelt
    seg_df = segment_with_rbf_pelt(weekly_panel)
    atomic_to_csv(weekly_panel, T2_LEGACY_WEEKLY_PANEL_PATH, index=False)
    atomic_to_csv(seg_df, T2_LEGACY_WEEKLY_CP_PATH, index=False)
    atomic_write_text(T2_LEGACY_WEEKLY_META_PATH,
                      json.dumps({'t2_legacy_weekly_config': {'version': 't2_legacy_weekly'}}, indent=2))
    if verbose:
        print(f"Weekly legacy elapsed: {format_elapsed(time.time() - t0)}")
        print(f"Saved weekly legacy outputs to {T2_LEGACY_WEEKLY_DIR}")
    return seg_df


# ====================================================================
# Stage 8 - Cross-granularity summary
# ====================================================================
def cross_granularity_summary(pelt_lookup_monthly, pelt_lookup_weekly,
                              records_monthly, records_weekly, verbose=True):
    monthly_avg = np.mean([len(v) for v in pelt_lookup_monthly.values()])
    weekly_avg = np.mean([len(v) for v in pelt_lookup_weekly.values()])
    monthly_obs = sum(len(r['series_norm']) for r in records_monthly)
    weekly_obs = sum(len(r['series_norm']) for r in records_weekly)
    if verbose:
        print("\n=== Cross-Granularity Summary (Monthly vs Weekly) ===")
        print(f"  PELT(L2) avg boundaries/product:")
        print(f"    monthly: {monthly_avg:.2f}  (over {len(pelt_lookup_monthly)} products)")
        print(f"    weekly : {weekly_avg:.2f}  (over {len(pelt_lookup_weekly)} products)")
        print(f"  Total observations: monthly={monthly_obs:,}  weekly={weekly_obs:,}  "
              f"(~{weekly_obs/max(monthly_obs,1):.1f}x)")
    return dict(
        monthly_avg_cps=monthly_avg, weekly_avg_cps=weekly_avg,
        monthly_obs=monthly_obs, weekly_obs=weekly_obs,
    )


# ====================================================================
# Top-level orchestrator
# ====================================================================
def run_weekly_pipeline(df, pelt_lookup_monthly, records_monthly,
                       force_rerun=False, verbose=True):
    """Run the full weekly pipeline Sec 9.W.3-Sec 9.W.11. Returns dict with all results."""
    t0 = time.time()
    if verbose:
        print("\n========== Weekly Task 2 Pipeline ==========")
    weekly_state = prepare_weekly_data(force_rerun=force_rerun, verbose=verbose)
    classical_df_w = pd.read_csv(T2_R1_WEEKLY_CP_PATH)

    autocpd_probs_w = train_route2_weekly(weekly_state, force_rerun=force_rerun, verbose=verbose)
    tst_probs_w = train_route3_weekly(weekly_state, force_rerun=force_rerun, verbose=verbose)
    moment_probs_w = train_route4_weekly(weekly_state, force_rerun=force_rerun, verbose=verbose)

    # Hybrid uses autocpd_probs_w
    hybrid_df = run_weekly_hybrid(weekly_state, autocpd_probs_w,
                                   force_rerun=force_rerun, verbose=verbose)

    # Evaluation
    eval_df, phase_df = run_weekly_evaluation(
        weekly_state, classical_df_w, force_rerun=force_rerun, verbose=verbose)

    # Legacy
    legacy_df = run_weekly_legacy(df, force_rerun=force_rerun, verbose=verbose)

    # Cross-granularity summary
    summary = cross_granularity_summary(
        pelt_lookup_monthly, weekly_state['pelt_lookup_weekly'],
        records_monthly, weekly_state['records_weekly'], verbose=verbose)

    if verbose:
        print(f"\nWeekly pipeline elapsed: {format_elapsed(time.time() - t0)}")
        print("=== Task 2 weekly pipeline complete. ===")

    return dict(weekly_state=weekly_state, classical_df_weekly=classical_df_w,
                autocpd_probs_weekly=autocpd_probs_w,
                tst_probs_weekly=tst_probs_w,
                moment_probs_weekly=moment_probs_w,
                hybrid_df_weekly=hybrid_df,
                eval_df_weekly=eval_df, phase_df_weekly=phase_df,
                legacy_df_weekly=legacy_df, cross_granularity=summary)
