"""
Training utilities: best-state tracking, train/val splits, GPU cleanup.

Shared across Task 2 Routes 2/3/4 (and any other neural training loops).
"""

import gc

import numpy as np
import torch


class BestStateTracker:
    """Track best-validation-loss model state in memory across epochs.

    Used by Routes 2/3/4 to restore the best epoch's weights instead of using
    the last epoch's weights (which may be slightly overfit).
    """

    def __init__(self, model):
        self.model = model
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_state = None

    def update(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in self.model.state_dict().items()}
            return True
        return False

    def restore(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            return True
        return False

    def summary(self):
        if self.best_state is None:
            return 'no validation tracked'
        return f'best epoch={self.best_epoch}  val_loss={self.best_loss:.4f}'


def split_train_val(X, y, *extras, val_frac=0.10, seed=1090):
    """Random train/val split with deterministic seed for cross-route consistency.

    Same seed + same X = identical split across Routes 2/3/4, so models are
    evaluated on the same val set and comparable.

    Returns (train_arrays, val_arrays, train_idx, val_idx).
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    val_size = max(1, int(round(val_frac * n)))
    val_idx = rng.choice(n, size=val_size, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]
    train_arrays = (X[train_idx], y[train_idx]) + tuple(e[train_idx] for e in extras)
    val_arrays   = (X[val_idx],   y[val_idx])   + tuple(e[val_idx]   for e in extras)
    return train_arrays, val_arrays, train_idx, val_idx


def cleanup_gpu(namespace=None, *var_names):
    """Free named variables (if they exist in the given namespace) and empty CUDA cache.

    Call at the end of each training cell to release VRAM before the next stage.

    Usage from a notebook cell:
        cleanup_gpu(globals(), 'model', 'trainer', 'optimizer')

    Or just collect garbage + empty cache (without freeing specific names):
        cleanup_gpu()
    """
    if namespace is not None:
        for name in var_names:
            if name in namespace:
                del namespace[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
