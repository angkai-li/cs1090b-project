"""
Task 2 Sec 9.5 Route 2 - AutoCPD-style: 2-layer MLP + product embedding.

Reimplementation of Li, Horvath, Wang, Yau JRSS-B 2024 `general_simple_nn`,
extended with product embedding (concat to MLP input).

Training data:
  50K synthetic AR(1) windows (50/50 no-change vs mean-shift)
  + all per-product PELT weak labels (~2K monthly, ~13K weekly)

Cache cascade: model.pt -> probs.pkl -> change_points.csv -> route_train_data.npz
The training data (X_train, y_train, asin_train) is also reused by Routes 3/4.
"""

import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config.hyperparams import (
    EMBED_DIM, N_SYNTHETIC, RANDOM_STATE, WINDOW_LEN_R2,
)
from ..config.paths import (
    T2_R1_META_PATH, T2_R2_CP_PATH, T2_R2_DIR, T2_R2_META_PATH,
    T2_R2_MODEL_PATH, T2_R2_PROBS_PATH,
    T2_RECORDS_META_PATH, T2_ROUTE_TRAIN_DATA_PATH,
)
from ..config.runtime import device
from ..utils.format import format_elapsed
from ..utils.io import (
    atomic_savez, atomic_to_csv, atomic_to_pickle, atomic_torch_save,
    atomic_write_text, read_json_or_none,
)
from ..utils.training import BestStateTracker, split_train_val
from .records import shrinkage_loss
from .synth import generate_window


T2_R2_VERSION = 't2_r2'


class AutoCPDClassifier(nn.Module):
    """PyTorch port of AutoCPD `general_simple_nn` (Li et al. JRSS-B 2024).

    Concatenates per-product embedding to the input window before MLP.
        Linear(window+embed_dim -> 64) -> BN -> GELU -> Dropout
      -> Linear(64 -> 64)               -> BN -> GELU -> Dropout
      -> Linear(64 -> 2)
    """
    def __init__(self, window_len=WINDOW_LEN_R2, hidden=64,
                 n_products=None, embed_dim=EMBED_DIM):
        super().__init__()
        assert n_products is not None, "AutoCPDClassifier needs n_products to size embedding"
        self.embed = nn.Embedding(n_products, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(window_len + embed_dim, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, asin_id):
        e = self.embed(asin_id)
        return self.net(torch.cat([x, e], dim=-1))


def detect_cps_autocpd(model, series, asin_id, window=WINDOW_LEN_R2,
                       threshold=0.5, device_=None):
    """Run sliding-window prediction. Returns (cps_list, probs_array)."""
    device_ = device_ or device
    if len(series) < window:
        return [], np.array([])
    model.eval()
    n_win = len(series) - window + 1
    windows = torch.tensor(np.array([series[i:i+window] for i in range(n_win)]),
                           dtype=torch.float32).to(device_)
    asin_ids = torch.full((n_win,), int(asin_id), dtype=torch.long, device=device_)
    with torch.no_grad():
        probs = torch.softmax(model(windows, asin_ids), dim=-1)[:, 1].cpu().numpy()
    cps = [int(i + window // 2) for i, p in enumerate(probs) if p > threshold]
    return cps, probs


def build_training_data(records, pelt_lookup, n_products, window_len=WINDOW_LEN_R2,
                        n_synthetic=N_SYNTHETIC, sigma_norm=1.0, verbose=True):
    """Build the shared X_train / y_train / asin_train tensors for R2/R3/R4.

    Mix:
      - n_synthetic synthetic AR(1) windows (50/50 no-change vs mean-shift),
        with random non-RARE asin_id (gradient spread across many products).
      - All per-product sliding windows from records, labeled via PELT weak labels
        (positive iff window center <= 1 from any PELT cp).

    Returns (X_train, y_train, asin_train) np.ndarrays.
    """
    np.random.seed(RANDOM_STATE)

    if verbose:
        print("Generating synthetic windows...")
    X_syn = np.zeros((n_synthetic, window_len), dtype=np.float32)
    y_syn = np.zeros(n_synthetic, dtype=np.int64)
    for i in range(n_synthetic):
        X_syn[i], y_syn[i] = generate_window(window_len, phi=0.3, sigma=sigma_norm)

    non_rare_ids = np.array([i for i in range(1, n_products)], dtype=np.int64)
    if len(non_rare_ids) == 0:
        non_rare_ids = np.array([0], dtype=np.int64)
    asin_syn = np.random.choice(non_rare_ids, size=n_synthetic, replace=True)
    if verbose:
        print(f"Synthetic: {X_syn.shape}, positive rate: {y_syn.mean():.3f}, "
              f"asin pool: {len(non_rare_ids)}")

    weak_X, weak_y, weak_asin = [], [], []
    for r in records:
        series = r['series_norm']
        bkps = pelt_lookup.get(r['asin'], [])
        for i in range(len(series) - window_len + 1):
            center = i + window_len // 2
            is_bd = any(abs(center - bp) <= 1 for bp in bkps)
            weak_X.append(series[i:i+window_len])
            weak_y.append(int(is_bd))
            weak_asin.append(r['asin_id'])
    weak_X = np.array(weak_X, dtype=np.float32)
    weak_y = np.array(weak_y, dtype=np.int64)
    weak_asin = np.array(weak_asin, dtype=np.int64)
    if weak_X.ndim == 1:
        weak_X = weak_X.reshape(0, window_len)
    if verbose:
        pos_rate = weak_y.mean() if len(weak_y) > 0 else 0.0
        print(f"Weak labels: {weak_X.shape}, positive rate: {pos_rate:.3f}")

    X_train = np.concatenate([X_syn, weak_X])
    y_train = np.concatenate([y_syn, weak_y])
    asin_train = np.concatenate([asin_syn, weak_asin])
    if verbose:
        print(f"Combined training set: {X_train.shape}  asin_train: {asin_train.shape}")
    return X_train, y_train, asin_train


def task2_r2_cache_is_valid():
    return (T2_R2_CP_PATH.exists() and T2_R2_MODEL_PATH.exists()
            and T2_R2_PROBS_PATH.exists() and T2_ROUTE_TRAIN_DATA_PATH.exists())


def train_route2(records, pelt_lookup, n_products, weight_decay_scale, sigma_norm,
                 max_epochs=60, patience=10, batch_size=512, lr=1e-3,
                 weight_decay=1e-4, verbose=True):
    """Train Route 2 from scratch. Returns (model, X_train, y_train, asin_train)."""
    X_train, y_train, asin_train = build_training_data(
        records, pelt_lookup, n_products, sigma_norm=sigma_norm, verbose=verbose)

    (X_tr, y_tr, a_tr), (X_va, y_va, a_va), _, _ = split_train_val(
        X_train, y_train, asin_train, val_frac=0.10, seed=RANDOM_STATE)
    if verbose:
        print(f"Route 2 train/val split: {len(X_tr)} train / {len(X_va)} val")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(a_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(a_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AutoCPDClassifier(n_products=n_products).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    tracker = BestStateTracker(model)
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, ab, yb in train_loader:
            xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
            out = model(xb, ab)
            cls_loss = loss_fn(out, yb)
            reg_loss = shrinkage_loss(model.embed, weight_decay_scale)
            loss = cls_loss + reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / len(train_ds)
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, ab, yb in val_loader:
                xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
                val_loss_sum += loss_fn(model(xb, ab), yb).item() * xb.size(0)
        val_loss = val_loss_sum / len(val_ds)

        improved = tracker.update(val_loss, epoch + 1)
        patience_counter = 0 if improved else patience_counter + 1
        last_lr = scheduler.get_last_lr()[0]
        if verbose and ((epoch + 1) % 5 == 0 or improved or patience_counter >= patience):
            mark = " *new best*" if improved else ""
            print(f"  epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  lr={last_lr:.2e}{mark}")
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best at epoch {tracker.best_epoch})")
            break

    tracker.restore()
    if verbose:
        print(f"  Route 2: {tracker.summary()}")
    return model, X_train, y_train, asin_train


def run_route2_inference(model, records, window_len=WINDOW_LEN_R2):
    """Sliding-window inference on all products. Returns (cps_df, probs_dict)."""
    results = []
    probs_dict = {}
    half = window_len // 2
    for r in records:
        cps, probs = detect_cps_autocpd(model, r['series_norm'], r['asin_id'], window=window_len)
        results.append({
            'asin': r['asin'], 'asin_id': r['asin_id'], 'n_obs': len(r['series_norm']),
            'autocpd_cps': json.dumps(cps),
            'autocpd_probs_max': float(np.max(probs)) if len(probs) else 0.0,
            'autocpd_probs_mean': float(np.mean(probs)) if len(probs) else 0.0,
        })
        aligned = np.zeros(len(r['series_norm']), dtype=np.float32)
        for i, p in enumerate(probs):
            idx = i + half
            if 0 <= idx < len(aligned):
                aligned[idx] = p
        probs_dict[r['asin']] = aligned
    return pd.DataFrame(results), probs_dict


def train_or_load_route2(records, pelt_lookup, n_products, weight_decay_scale,
                         sigma_norm, force_rerun=False, verbose=True):
    """Run the full Route 2 pipeline or load from cache.

    Returns dict with: model, cps_df, probs_dict, X_train, y_train, asin_train.
    """
    T2_R2_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rerun and task2_r2_cache_is_valid():
        model = AutoCPDClassifier(n_products=n_products).to(device)
        model.load_state_dict(torch.load(T2_R2_MODEL_PATH, map_location=device))
        cps_df = pd.read_csv(T2_R2_CP_PATH)
        with open(T2_R2_PROBS_PATH, 'rb') as f:
            probs_dict = pickle.load(f)
        d = np.load(T2_ROUTE_TRAIN_DATA_PATH)
        if verbose:
            print(f"Loaded Route 2 from cache: {len(cps_df)} products")
            print(f"  X_train: {d['X_train'].shape}  positive rate: {d['y_train'].mean():.3f}")
            print("Route 2 done (from cache).")
        return {
            'model': model, 'cps_df': cps_df, 'probs_dict': probs_dict,
            'X_train': d['X_train'], 'y_train': d['y_train'], 'asin_train': d['asin_train'],
        }

    t0 = time.time()
    if verbose:
        print(f"Empirical sigma of sentiment_norm: {sigma_norm:.4f}")
        print(f"Route 2 using device: {device}")

    model, X_train, y_train, asin_train = train_route2(
        records, pelt_lookup, n_products, weight_decay_scale, sigma_norm, verbose=verbose)
    atomic_torch_save(model.state_dict(), T2_R2_MODEL_PATH)
    if verbose:
        print(f"  [safety] Best AutoCPD weights saved to {T2_R2_MODEL_PATH} (pre-inference)")

    cps_df, probs_dict = run_route2_inference(model, records)
    atomic_to_csv(cps_df, T2_R2_CP_PATH, index=False)
    atomic_torch_save(model.state_dict(), T2_R2_MODEL_PATH)
    atomic_to_pickle(probs_dict, T2_R2_PROBS_PATH)
    atomic_savez(T2_ROUTE_TRAIN_DATA_PATH,
                 X_train=X_train, y_train=y_train, asin_train=asin_train)

    config = {
        'version': T2_R2_VERSION,
        'window_len': WINDOW_LEN_R2, 'n_synthetic': N_SYNTHETIC,
        'embed_dim': EMBED_DIM,
        'arch': {'hidden': 64, 'dropout': 0.3, 'activation': 'GELU'},
        'training': {'lr': 1e-3, 'weight_decay': 1e-4, 'max_epochs': 60,
                     'patience': 10, 'batch_size': 512,
                     'scheduler': 'CosineAnnealingLR(eta_min=1e-5)'},
        't2_dataprep_meta': read_json_or_none(T2_RECORDS_META_PATH),
        't2_r1_meta': read_json_or_none(T2_R1_META_PATH),
    }
    atomic_write_text(T2_R2_META_PATH, json.dumps({'t2_r2_config': config}, indent=2))
    if verbose:
        print(f"Saved Route 2 predictions to {T2_R2_CP_PATH}")
        print(f"Saved Route 2 model weights to {T2_R2_MODEL_PATH}")
        print(f"Route 2 elapsed: {format_elapsed(time.time() - t0)}")
        print("Route 2 done.")

    return {
        'model': model, 'cps_df': cps_df, 'probs_dict': probs_dict,
        'X_train': X_train, 'y_train': y_train, 'asin_train': asin_train,
    }
