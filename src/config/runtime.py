"""
Runtime state: optional-dependency probes, random seeds, device detection.

Importing this module:
  - Probes HAS_VADER / HAS_TRANSFORMERS / HAS_RUPTURES / HAS_CLASPY / HAS_MOMENT
  - Sets random seeds once for Python / NumPy / PyTorch / HF transformers
  - Detects device (CPU / CUDA) and bf16/fp16 availability
  - Suppresses noisy transformers/datasets logging

Must be imported AFTER `src.config.env.setup_environment()` because HAS_* probes
trigger transformers/peft import.
"""

import importlib.util
import logging
import os
import random

import numpy as np
import torch

from .hyperparams import RANDOM_STATE


# ====================================================================
# Optional dependency probes
# ====================================================================
HAS_VADER = importlib.util.find_spec('vaderSentiment') is not None

try:
    from datasets import Dataset  # noqa: F401
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed as hf_set_seed,
    )
    try:
        from transformers import EarlyStoppingCallback  # noqa: F401
    except ImportError:
        EarlyStoppingCallback = None
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    hf_set_seed = None

try:
    import ruptures as rpt  # noqa: F401
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    from claspy.segmentation import BinaryClaSPSegmentation  # noqa: F401
    HAS_CLASPY = True
except ImportError:
    HAS_CLASPY = False

try:
    from momentfm import MOMENTPipeline  # noqa: F401
    HAS_MOMENT = True
except ImportError:
    HAS_MOMENT = False


# ====================================================================
# Random seeds - single seeding pass
# ====================================================================
def set_all_seeds(seed=RANDOM_STATE):
    """Seed Python, NumPy, PyTorch (CPU+CUDA), and HF transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if HAS_TRANSFORMERS and hf_set_seed is not None:
        hf_set_seed(seed)


set_all_seeds(RANDOM_STATE)


# ====================================================================
# Device detection + low-level perf knobs
# ====================================================================
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if USE_CUDA else 'CPU'
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
USE_FP16 = USE_CUDA and not USE_BF16

if USE_CUDA:
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cudnn.benchmark caches the fastest kernel per input shape.
    # group_by_length keeps shapes ~stable per bucket, so net positive ~3-8%.
    # If you observe slowdown (very dynamic shapes), set this False.
    torch.backends.cudnn.benchmark = True


# ====================================================================
# Logging: reduce noise from transformers/datasets
# ====================================================================
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ====================================================================
# Status banner (call from notebook to display)
# ====================================================================
def print_status():
    """Print runtime status: device, seed, optional dep flags."""
    if not HAS_TRANSFORMERS:
        print("transformers/peft/datasets not installed; DeBERTa (Sec 8.2) and "
              "MOMENT (Route 4) will be skipped.")
    if not HAS_RUPTURES:
        print("ruptures not installed; Route 1 PELT will be skipped. "
              "Install: pip install ruptures==1.1.9")
    if not HAS_CLASPY:
        print("claspy not installed; Route 1 ClaSP will be skipped. "
              "Install: pip install claspy>=0.2.6")
    if not HAS_MOMENT:
        print("momentfm not installed; Route 4 will be skipped. "
              "Install: pip install momentfm")
    print(f"  Device:        {CUDA_DEVICE_NAME}  (bf16={USE_BF16}, fp16={USE_FP16})")
    print(f"  Random seed:   {RANDOM_STATE}")
    print(f"  Optional deps: VADER={HAS_VADER}  transformers={HAS_TRANSFORMERS}  "
          f"ruptures={HAS_RUPTURES}  claspy={HAS_CLASPY}  moment={HAS_MOMENT}")
