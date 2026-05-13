"""
Task 2 Sec 9.6 Route 3 - TST-style BERT encoder + MVP pretraining.

Two-phase training:
  Phase 1 (MVP / Masked Value Pretraining): bidirectional masked imputation on
    all real product series. Product-agnostic.
  Phase 2 (Boundary classification): classify whether window center is a boundary,
    using synthetic + PELT weak labels. Per-product additive bias added DeepAR-style
    to hidden states: h_{p,t} = TransformerEncoder(InputProj(x) + PE(t)) + e_p.
"""

import json
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config.hyperparams import LAMBDA_L2, RANDOM_STATE, WINDOW_LEN_R3
from ..config.paths import (
    T2_R2_META_PATH, T2_R3_CP_PATH, T2_R3_DIR, T2_R3_META_PATH,
    T2_R3_MODEL_DIR, T2_R3_MODEL_PATH, T2_R3_PROBS_PATH,
)
from ..config.runtime import device
from ..utils.format import format_elapsed
from ..utils.io import (
    atomic_to_csv, atomic_to_pickle, atomic_torch_save,
    atomic_write_text, read_json_or_none,
)
from ..utils.training import BestStateTracker, split_train_val
from .records import shrinkage_loss


T2_R3_VERSION = 't2_r3'
D_MODEL = 64
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TSTEncoder(nn.Module):
    """TST-style bidirectional encoder for univariate time series.

    - Phase 1 head: imputation (per-position regression, product-agnostic)
    - Phase 2 head: classification (window-level binary, with DeepAR-style
      additive product bias: h_{p,t} = TransformerEncoder(InputProj(x) + PE(t)) + e_p).
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dropout=DROPOUT, max_len=128, n_products=None):
        super().__init__()
        assert n_products is not None, "TSTEncoder needs n_products to size embedding"
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.imputation_head = nn.Linear(d_model, 1)
        self.classification_head = nn.Linear(d_model, 2)
        self.embed = nn.Embedding(n_products, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, mode='classification', asin_id=None):
        h = x.unsqueeze(-1)            # (B, T, 1)
        h = self.input_proj(h)
        h = self.pos_enc(h)
        if mode == 'classification' and asin_id is not None:
            e = self.embed(asin_id)
            h = h + e.unsqueeze(1)
        h = self.encoder(h)
        if mode == 'imputation':
            return self.imputation_head(h).squeeze(-1)
        pooled = h.mean(dim=1)
        return self.classification_head(pooled)


def detect_cps_tst(model, series, asin_id, window=WINDOW_LEN_R3, threshold=0.5, device_=None):
    """Sliding-window inference. Returns (cps_list, probs_array)."""
    device_ = device_ or device
    if len(series) < window:
        return [], np.array([])
    model.eval()
    n_win = len(series) - window + 1
    windows = torch.tensor(np.array([series[i:i+window] for i in range(n_win)]),
                           dtype=torch.float32).to(device_)
    asin_ids = torch.full((n_win,), int(asin_id), dtype=torch.long, device=device_)
    with torch.no_grad():
        logits = model(windows, mode='classification', asin_id=asin_ids)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    cps = [int(i + window // 2) for i, p in enumerate(probs) if p > threshold]
    return cps, probs


def task2_r3_cache_is_valid():
    return T2_R3_CP_PATH.exists() and T2_R3_MODEL_PATH.exists() and T2_R3_PROBS_PATH.exists()


def train_phase1_mvp(records, window_len=WINDOW_LEN_R3, mask_ratio=0.15,
                    max_epochs=40, patience=6, lr=1e-3, batch_size=128,
                    n_products=None, verbose=True):
    """Phase 1: masked-value pretraining on all real product series."""
    pretrain_windows = []
    for r in records:
        series = r['series_norm']
        for i in range(0, len(series) - window_len + 1, 4):  # stride=4 to dedupe
            pretrain_windows.append(series[i:i+window_len])
    pretrain_X = np.array(pretrain_windows, dtype=np.float32)
    if verbose:
        print(f"Pretrain windows: {pretrain_X.shape}")

    model = TSTEncoder(n_products=n_products).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    placeholder = np.zeros(len(pretrain_X), dtype=np.int64)
    (X_tr, _,), (X_va, _,), _, _ = split_train_val(
        pretrain_X, placeholder, val_frac=0.10, seed=RANDOM_STATE)
    if verbose:
        print(f"  Phase 1 train/val split: {len(X_tr)} train / {len(X_va)} val")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va)),
                            batch_size=batch_size, shuffle=False)

    # Deterministic val mask for comparable val_loss across epochs
    val_gen = torch.Generator().manual_seed(RANDOM_STATE)
    val_mask = torch.rand(X_va.shape, generator=val_gen) < mask_ratio

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    tracker = BestStateTracker(model)
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_sum = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            mask = (torch.rand_like(xb) < mask_ratio)
            x_masked = xb.clone()
            x_masked[mask] = 0.0
            x_pred = model(x_masked, mode='imputation')
            loss = mse_loss(x_pred[mask], xb[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item() * xb.size(0)
        train_loss = train_sum / len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_sum, val_count = 0.0, 0
        offset = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                mb = val_mask[offset:offset + xb.size(0)].to(device)
                x_masked = xb.clone()
                x_masked[mb] = 0.0
                x_pred = model(x_masked, mode='imputation')
                if mb.any():
                    val_sum += mse_loss(x_pred[mb], xb[mb]).item() * mb.sum().item()
                    val_count += mb.sum().item()
                offset += xb.size(0)
        val_loss = val_sum / max(val_count, 1)

        improved = tracker.update(val_loss, epoch + 1)
        patience_counter = 0 if improved else patience_counter + 1
        last_lr = scheduler.get_last_lr()[0]
        if verbose and ((epoch + 1) % 5 == 0 or improved or patience_counter >= patience):
            mark = " *new best*" if improved else ""
            print(f"  Phase1 epoch {epoch+1}/{max_epochs}  train_recon={train_loss:.4f}  "
                  f"val_recon={val_loss:.4f}  lr={last_lr:.2e}{mark}")
        if patience_counter >= patience:
            if verbose:
                print(f"  Phase 1 early stopping at epoch {epoch+1} (best at epoch {tracker.best_epoch})")
            break

    tracker.restore()
    if verbose:
        print(f"  Route 3 Phase 1: {tracker.summary()}")
    return model


def train_phase2_finetune(model, X_train, y_train, asin_train, weight_decay_scale,
                          max_epochs=35, patience=8, lr=3e-4, batch_size=256, verbose=True):
    """Phase 2: boundary classification fine-tune with product embedding."""
    (X_tr, y_tr, a_tr), (X_va, y_va, a_va), _, _ = split_train_val(
        X_train, y_train, asin_train, val_frac=0.10, seed=RANDOM_STATE)
    if verbose:
        print(f"  Phase 2 train/val split: {len(X_tr)} train / {len(X_va)} val")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(a_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(a_va), torch.from_numpy(y_va)),
        batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    ce_loss = nn.CrossEntropyLoss()
    tracker = BestStateTracker(model)
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_sum = 0.0
        for xb, ab, yb in train_loader:
            xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
            logits = model(xb, mode='classification', asin_id=ab)
            cls_loss = ce_loss(logits, yb)
            reg_loss = shrinkage_loss(model.embed, weight_decay_scale)
            loss = cls_loss + reg_loss
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
                val_sum += ce_loss(model(xb, mode='classification', asin_id=ab), yb).item() * xb.size(0)
        val_loss = val_sum / len(val_loader.dataset)

        improved = tracker.update(val_loss, epoch + 1)
        patience_counter = 0 if improved else patience_counter + 1
        last_lr = scheduler.get_last_lr()[0]
        if verbose and ((epoch + 1) % 5 == 0 or improved or patience_counter >= patience):
            mark = " *new best*" if improved else ""
            print(f"  Phase2 epoch {epoch+1}/{max_epochs}  train_cls={train_loss:.4f}  "
                  f"val_cls={val_loss:.4f}  lr={last_lr:.2e}{mark}")
        if patience_counter >= patience:
            if verbose:
                print(f"  Phase 2 early stopping at epoch {epoch+1} (best at epoch {tracker.best_epoch})")
            break

    tracker.restore()
    if verbose:
        print(f"  Route 3 Phase 2: {tracker.summary()}")
    return model


def run_route3_inference(model, records, window_len=WINDOW_LEN_R3):
    """Sliding-window inference. Returns (cps_df, probs_dict)."""
    results = []
    probs_dict = {}
    half = window_len // 2
    for r in records:
        cps, probs = detect_cps_tst(model, r['series_norm'], r['asin_id'], window=window_len)
        results.append({
            'asin': r['asin'], 'asin_id': r['asin_id'], 'n_obs': len(r['series_norm']),
            'tst_cps': json.dumps(cps),
            'tst_probs_max':  float(np.max(probs))  if len(probs) else 0.0,
            'tst_probs_mean': float(np.mean(probs)) if len(probs) else 0.0,
        })
        aligned = np.zeros(len(r['series_norm']), dtype=np.float32)
        for i, p in enumerate(probs):
            idx = i + half
            if 0 <= idx < len(aligned):
                aligned[idx] = p
        probs_dict[r['asin']] = aligned
    return pd.DataFrame(results), probs_dict


def train_or_load_route3(records, X_train, y_train, asin_train, n_products, weight_decay_scale,
                         force_rerun=False, verbose=True):
    """Run full Route 3 pipeline (Phase 1 + Phase 2 + inference), or load from cache."""
    T2_R3_DIR.mkdir(parents=True, exist_ok=True)
    T2_R3_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rerun and task2_r3_cache_is_valid():
        model = TSTEncoder(n_products=n_products).to(device)
        model.load_state_dict(torch.load(T2_R3_MODEL_PATH, map_location=device))
        cps_df = pd.read_csv(T2_R3_CP_PATH)
        with open(T2_R3_PROBS_PATH, 'rb') as f:
            probs_dict = pickle.load(f)
        if verbose:
            print(f"Loaded Route 3 from cache: {len(cps_df)} products")
            print("Route 3 done (from cache).")
        return {'model': model, 'cps_df': cps_df, 'probs_dict': probs_dict}

    t0 = time.time()
    if verbose:
        print("Route 3 Phase 1: MVP masked imputation pretraining")
    model = train_phase1_mvp(records, n_products=n_products, verbose=verbose)
    phase1_path = T2_R3_MODEL_DIR / 'phase1_only.pt'
    atomic_torch_save(model.state_dict(), phase1_path)
    if verbose:
        print(f"  [safety] Phase 1 best weights saved to {phase1_path}")
        print("Route 3 Phase 2: boundary classification fine-tune (with product embedding)")

    model = train_phase2_finetune(model, X_train, y_train, asin_train, weight_decay_scale,
                                   verbose=verbose)
    atomic_torch_save(model.state_dict(), T2_R3_MODEL_PATH)
    if verbose:
        print(f"  [safety] Phase 2 best weights saved to {T2_R3_MODEL_PATH} (pre-inference)")

    cps_df, probs_dict = run_route3_inference(model, records)
    atomic_to_csv(cps_df, T2_R3_CP_PATH, index=False)
    atomic_torch_save(model.state_dict(), T2_R3_MODEL_PATH)
    atomic_to_pickle(probs_dict, T2_R3_PROBS_PATH)

    config = {
        'version': T2_R3_VERSION,
        'window_len': WINDOW_LEN_R3,
        'embed_dim_for_model': 'd_model=64',
        'lambda_l2': LAMBDA_L2,
        'arch': {'d_model': 64, 'n_heads': 8, 'n_layers': 3, 'dim_ff': 128, 'dropout': 0.2},
        'phase1': {'mask_ratio': 0.15, 'max_epochs': 40, 'patience': 6, 'lr': 1e-3,
                   'batch_size': 128, 'scheduler': 'CosineAnnealingLR(eta_min=1e-5)'},
        'phase2': {'max_epochs': 35, 'patience': 8, 'lr': 3e-4,
                   'batch_size': 256, 'scheduler': 'CosineAnnealingLR(eta_min=1e-5)'},
        't2_r2_meta': read_json_or_none(T2_R2_META_PATH),
    }
    atomic_write_text(T2_R3_META_PATH, json.dumps({'t2_r3_config': config}, indent=2))
    if verbose:
        print(f"Saved Route 3 model weights to {T2_R3_MODEL_PATH}")
        print(f"Route 3 elapsed: {format_elapsed(time.time() - t0)}")
        print(f"Route 3 done. Saved to {T2_R3_DIR}")
    return {'model': model, 'cps_df': cps_df, 'probs_dict': probs_dict}
