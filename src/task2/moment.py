"""
Task 2 Sec 9.7 Route 4 - MOMENT-1-large + LoRA + 2-layer MLP side-channel.

Foundation model approach: AutonLab/MOMENT-1-large (385M params, pretrained on
1.13B time points) with:
  - LoRA r=32 on q/k/v/o attention projections (~1.63% trainable)
  - 2-layer MLP per-product side-channel `embed_dim -> 64 -> n_classes` (zero-init
    final layer so initial behavior matches MOMENT-only)
  - FocalLoss(alpha_pos=0.75, gamma=2.0) for class imbalance
  - Left-pad inputs to 512 with edge replication (series[0]); set input_mask to
    mark padding so MOMENT's T5 attention focuses on the 24 real positions.

Skipped automatically when momentfm, transformers, or CUDA is missing.
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
    EMBED_DIM, MOMENT_TARGET_LEN, RANDOM_STATE, WINDOW_LEN_R4,
)
from ..config.paths import (
    T2_R2_META_PATH, T2_R4_CP_PATH, T2_R4_DIR, T2_R4_LORA_DIR,
    T2_R4_META_PATH, T2_R4_PROBS_PATH, T2_R4_SIDE_PATH,
)
from ..config.runtime import device, USE_BF16, USE_CUDA, HAS_MOMENT, HAS_TRANSFORMERS
from ..utils.format import format_elapsed
from ..utils.io import (
    atomic_save_pretrained, atomic_to_csv, atomic_to_pickle, atomic_torch_save,
    atomic_write_text, read_json_or_none,
)
from ..utils.losses import FocalLoss
from ..utils.training import BestStateTracker, split_train_val
from .records import shrinkage_loss


T2_R4_VERSION = 't2_r4'


def prepare_moment_input(series, target_len=MOMENT_TARGET_LEN):
    """Left-pad a short series to target_len with edge replication (series[0]).

    Edge replication is recommended by Google TimesFM for short-sequence handling
    in time-series foundation models - preserves the starting baseline rather
    than introducing an artificial "stable plateau" via mean padding.
    """
    L = len(series)
    if L >= target_len:
        return series[-target_len:]
    return np.concatenate([
        np.full(target_len - L, float(series[0]), dtype=np.float32),
        series,
    ])


def make_moment_input_mask(real_len, batch_size, device_, target_len=MOMENT_TARGET_LEN):
    """Build (batch_size, target_len) mask with 1s at real positions, 0s at padding.

    Since prepare_moment_input pads on the LEFT, real positions are at the END.
    Without this mask, MOMENT attends equally to all 512 positions including
    the 488 padded positions - wasted attention.
    """
    mask = torch.zeros((batch_size, target_len), dtype=torch.float32, device=device_)
    mask[:, -real_len:] = 1.0
    return mask


class _MomentSideChannel(nn.Module):
    """Per-product side-channel logit bias for MOMENT classification.

    2-layer MLP `embed_dim -> 64 -> n_classes`. The final layer is zero-initialized
    so the model starts equivalent to MOMENT-only and learns the side-channel
    from gradient signal.
    """
    def __init__(self, n_products, embed_dim=EMBED_DIM, hidden_dim=64, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(n_products, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.bias_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )
        nn.init.zeros_(self.bias_proj[-1].weight)
        nn.init.zeros_(self.bias_proj[-1].bias)

    def forward(self, asin_id):
        return self.bias_proj(self.embed(asin_id))


def _build_moment_base():
    """Construct + .init() the MOMENT pipeline. Must call init() BEFORE .to(device)."""
    from momentfm import MOMENTPipeline
    base = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification', 'n_channels': 1, 'num_class': 2,
            'freeze_encoder': True, 'reduction': 'mean',
        },
    )
    base.init()
    return base


def task2_r4_cache_is_valid():
    return (T2_R4_LORA_DIR.exists() and T2_R4_SIDE_PATH.exists()
            and T2_R4_PROBS_PATH.exists() and T2_R4_CP_PATH.exists())


def detect_cps_moment(model, side, series, asin_id, window=WINDOW_LEN_R4, threshold=0.5,
                     target_len=MOMENT_TARGET_LEN, amp_enabled=False):
    if len(series) < window:
        return [], np.array([])
    n_win = len(series) - window + 1
    windows = np.array([series[i:i+window] for i in range(n_win)])
    windows_padded = np.stack([prepare_moment_input(w, target_len) for w in windows])
    x = torch.from_numpy(windows_padded).to(device).unsqueeze(1)
    asin_ids = torch.full((n_win,), int(asin_id), dtype=torch.long, device=device)
    model.eval()
    side.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                          enabled=amp_enabled):
        input_mask = make_moment_input_mask(window, x.shape[0], x.device, target_len)
        moment_logits = model(x_enc=x, input_mask=input_mask).logits
        side_logits = side(asin_ids)
        logits = moment_logits + side_logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    cps = [int(i + window // 2) for i, p in enumerate(probs) if p > threshold]
    return cps, probs


def train_route4(X_train, y_train, asin_train, n_products, weight_decay_scale,
                 max_epochs=14, patience=4, lr=5e-4, weight_decay=0.01,
                 window_len=WINDOW_LEN_R4, target_len=MOMENT_TARGET_LEN,
                 verbose=True):
    """Train MOMENT + LoRA + side-channel from scratch.

    Returns (moment_model_with_lora, side_module, best_epoch, best_val_loss).
    """
    from peft import LoraConfig, get_peft_model

    moment_base = _build_moment_base().to(device)
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q", "k", "v", "o"],
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(moment_base, lora_cfg)
    if verbose:
        model.print_trainable_parameters()

    side = _MomentSideChannel(n_products=n_products).to(device)
    focal = FocalLoss(alpha_pos=0.75, gamma=2.0)
    n_side = sum(p.numel() for p in side.parameters() if p.requires_grad)
    if verbose:
        print(f"  +side-channel embedding/bias_proj: {n_side} params")
        print("Preparing MOMENT inputs (padding to 512)...")
    X_moment = np.stack([prepare_moment_input(w, target_len) for w in X_train])

    (X_tr, y_tr, a_tr), (X_va, y_va, a_va), _, _ = split_train_val(
        X_moment, y_train, asin_train, val_frac=0.10, seed=RANDOM_STATE)
    if verbose:
        print(f"  MOMENT train/val split: {len(X_tr)} train / {len(X_va)} val")

    amp_enabled = USE_BF16
    batch_size = 192 if amp_enabled else 32
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(a_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True, pin_memory=USE_CUDA, num_workers=2)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(a_va), torch.from_numpy(y_va)),
        batch_size=batch_size, shuffle=False, pin_memory=USE_CUDA, num_workers=2)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(side.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    tracker_side = BestStateTracker(side)
    best_epoch, best_val = -1, float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        side.train()
        train_sum = 0.0
        for xb, ab, yb in train_loader:
            xb = xb.to(device).unsqueeze(1)
            ab = ab.to(device)
            yb = yb.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                input_mask = make_moment_input_mask(window_len, xb.shape[0], xb.device, target_len)
                moment_logits = model(x_enc=xb, input_mask=input_mask).logits
                side_logits = side(ab)
                final_logits = moment_logits + side_logits
                cls_loss = focal(final_logits, yb)
            reg_loss = shrinkage_loss(side.embed, weight_decay_scale)
            loss = cls_loss + reg_loss
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
                input_mask = make_moment_input_mask(window_len, xb.shape[0], xb.device, target_len)
                moment_logits = model(x_enc=xb, input_mask=input_mask).logits
                final_logits = moment_logits + side(ab)
                val_sum += focal(final_logits, yb).item() * xb.size(0)
        val_loss = val_sum / len(val_loader.dataset)

        improved = tracker_side.update(val_loss, epoch + 1)
        patience_counter = 0 if improved else patience_counter + 1
        last_lr = scheduler.get_last_lr()[0]
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            atomic_save_pretrained(model, T2_R4_LORA_DIR)
        if verbose:
            print(f"  MOMENT epoch {epoch+1}/{max_epochs}  train_cls={train_loss:.4f}  "
                  f"val_cls={val_loss:.4f}  lr={last_lr:.2e}"
                  + (" *new best*" if improved else ""))
        if patience_counter >= patience:
            if verbose:
                print(f"  MOMENT early stopping at epoch {epoch+1} (best at epoch {best_epoch})")
            break

    tracker_side.restore()
    return model, side, best_epoch, best_val


def _reload_best_lora_model():
    """Reload MOMENT base + LoRA adapter at best epoch (saved during training)."""
    from peft import PeftModel
    base = _build_moment_base().to(device)
    return PeftModel.from_pretrained(base, str(T2_R4_LORA_DIR)).to(device)


def run_route4_inference(model, side, records, window_len=WINDOW_LEN_R4,
                         target_len=MOMENT_TARGET_LEN, amp_enabled=False):
    """Sliding-window inference. Returns (cps_df, probs_dict)."""
    results = []
    probs_dict = {}
    half = window_len // 2
    for r in records:
        cps, probs = detect_cps_moment(
            model, side, r['series_norm'], r['asin_id'],
            window=window_len, target_len=target_len, amp_enabled=amp_enabled)
        results.append({
            'asin': r['asin'], 'asin_id': r['asin_id'], 'n_obs': len(r['series_norm']),
            'moment_cps': json.dumps(cps),
            'moment_probs_max':  float(np.max(probs))  if len(probs) else 0.0,
            'moment_probs_mean': float(np.mean(probs)) if len(probs) else 0.0,
        })
        aligned = np.zeros(len(r['series_norm']), dtype=np.float32)
        for i, p in enumerate(probs):
            idx = i + half
            if 0 <= idx < len(aligned):
                aligned[idx] = p
        probs_dict[r['asin']] = aligned
    return pd.DataFrame(results), probs_dict


def train_or_load_route4(records, X_train, y_train, asin_train, n_products,
                         weight_decay_scale, force_rerun=False, verbose=True):
    """Run full Route 4 pipeline or load from cache.

    Returns dict with model, side, cps_df, probs_dict. If MOMENT/CUDA missing,
    returns {'model': None, 'side': None, 'cps_df': None, 'probs_dict': {}}.
    """
    T2_R4_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rerun and task2_r4_cache_is_valid():
        cps_df = pd.read_csv(T2_R4_CP_PATH)
        with open(T2_R4_PROBS_PATH, 'rb') as f:
            probs_dict = pickle.load(f)
        if verbose:
            print(f"Loaded Route 4 from cache: {len(cps_df)} products")
        return {'model': None, 'side': None, 'cps_df': cps_df, 'probs_dict': probs_dict}

    if not (HAS_MOMENT and HAS_TRANSFORMERS and USE_CUDA):
        if not HAS_MOMENT:
            print("Skipping Route 4 (momentfm not installed, see Sec 0.1 setup)")
        elif not HAS_TRANSFORMERS:
            print("Skipping Route 4 (transformers/peft not installed)")
        else:
            print("Skipping Route 4 (no CUDA GPU)")
        atomic_write_text(T2_R4_META_PATH, json.dumps({'t2_r4_config': _build_config()}, indent=2))
        return {'model': None, 'side': None, 'cps_df': None, 'probs_dict': {}}

    t0 = time.time()
    model, side, best_epoch, best_val = train_route4(
        X_train, y_train, asin_train, n_products, weight_decay_scale, verbose=verbose)

    # Reload best LoRA from disk (already saved during training at best epoch)
    model = _reload_best_lora_model()
    if verbose:
        print(f"  Route 4: best epoch={best_epoch}  val_cls={best_val:.4f}")
    atomic_torch_save(side.state_dict(), T2_R4_SIDE_PATH)
    if verbose:
        print("  [safety] MOMENT best LoRA + side-channel saved (pre-inference)")

    cps_df, probs_dict = run_route4_inference(
        model, side, records, amp_enabled=USE_BF16)
    atomic_to_csv(cps_df, T2_R4_CP_PATH, index=False)
    atomic_save_pretrained(model, T2_R4_LORA_DIR)
    atomic_torch_save(side.state_dict(), T2_R4_SIDE_PATH)
    atomic_to_pickle(probs_dict, T2_R4_PROBS_PATH)
    atomic_write_text(T2_R4_META_PATH, json.dumps({'t2_r4_config': _build_config()}, indent=2))
    if verbose:
        print(f"Saved Route 4 LoRA adapter to {T2_R4_LORA_DIR}")
        print(f"Saved Route 4 side-channel to {T2_R4_SIDE_PATH}")
        print(f"Route 4 elapsed: {format_elapsed(time.time() - t0)}")
        print(f"Route 4 done. Saved to {T2_R4_DIR}")
    return {'model': model, 'side': side, 'cps_df': cps_df, 'probs_dict': probs_dict}


def _build_config():
    return {
        'version': T2_R4_VERSION,
        'base_model': 'AutonLab/MOMENT-1-large',
        'target_len': MOMENT_TARGET_LEN,
        'lora': {'r': 32, 'alpha': 64, 'target_modules': ['q', 'k', 'v', 'o'],
                 'dropout': 0.05, 'bias': 'none', 'task_type': None},
        'side_channel': {'embed_dim': EMBED_DIM, 'hidden_dim': 64,
                         'arch': 'mlp_2layer', 'init': 'final_layer_zeros'},
        'loss': 'FocalLoss(alpha_pos=0.75, gamma=2.0)',
        'padding': 'edge_replicate (series[0])',
        'training': {'lr': 5e-4, 'weight_decay': 0.01, 'max_epochs': 14, 'patience': 4,
                     'batch_size_bf16': 192, 'batch_size_fp32': 32,
                     'scheduler': 'CosineAnnealingLR(eta_min=1e-5)'},
        't2_r2_meta': read_json_or_none(T2_R2_META_PATH),
    }
