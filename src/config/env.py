"""
Environment setup: HF mirror, env vars, pip install, dependency verification.

This module MUST be imported and called BEFORE any transformers / huggingface_hub
/ torch import. Once those are imported they cache env-var values, so setting
them later has no effect on the cache location / mirror redirect.

Usage in notebook (Cell Sec 0):
    from src.config.env import setup_environment
    setup_environment()
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path


def _set_hf_env_vars():
    """Set HF mirror + cache locations BEFORE any HF import.

    Uses cwd-relative paths (cwd = project root after jupyter launches there).
    Idempotent: setdefault skips if already set.
    """
    hf_root = Path.cwd() / 'cache' / 'hf_hub'
    mpl_root = Path.cwd() / 'cache' / 'mpl_config'

    # Mirror redirect: hf-mirror.com is a community proxy of huggingface.co
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    # HF cache locations (read at huggingface_hub import time)
    os.environ.setdefault('XDG_CACHE_HOME',     str(hf_root))
    os.environ.setdefault('HF_HOME',            str(hf_root / 'huggingface'))
    os.environ.setdefault('HF_HUB_CACHE',       str(hf_root / 'huggingface' / 'hub'))
    os.environ.setdefault('HF_DATASETS_CACHE',  str(hf_root / 'huggingface' / 'datasets'))

    # torch / matplotlib env vars (read at their respective import time)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM',  'true')
    os.environ.setdefault('MPLCONFIGDIR',            str(mpl_root))


def _pip(packages):
    """Install packages via the current kernel's Python (sys.executable).

    Idempotent: takes ~5-10s when everything is already installed.
    """
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         '--root-user-action=ignore', *packages],
        check=True,
    )


def _verify(pkg, hint):
    """Hard-verify a critical package can be imported. Fail loud with reinstall hint."""
    try:
        mod = __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"\n[FATAL] '{pkg}' install verification failed: {type(e).__name__}: {e}\n"
            f"  Reinstall manually:  pip install --force-reinstall --no-cache-dir {hint}\n"
            f"  (or check the pip output above for the underlying build/dep error)"
        ) from e
    version = getattr(mod, '__version__', '?')
    print(f"  ok  {pkg:<10s}  version={version}")


# DeBERTa fp32 forced: bf16 + DeBERTa-v3 NaN's at step 100 (effective lr ~2.3e-6,
# well within warmup). Root cause: PR #35336 fixed deberta v1 bf16 NaN but NOT
# deberta_v2 (deberta-v3 uses modeling_deberta_v2.py). Disentangled attention's
# c2c+c2p+p2c score sum overflows bf16's 7-bit mantissa on Blackwell.
# Pure fp32 is mathematically NaN-proof (23-bit mantissa) at ~2x training cost.
DEBERTA_BF16_AVAILABLE = False


def setup_environment(install=True, verify=True, prewarm=False):
    """One-shot environment bootstrap. Call FIRST in the notebook.

    1. Sets HF env vars (mirror, cache paths) - must run before HF imports
    2. (optional) Installs all pip deps idempotently
    3. (optional) Hard-verifies critical packages (peft, ruptures, momentfm)
    4. (optional) Pre-warms HF model downloads (off by default)

    Args:
        install:  if True, run `_pip([...])` for all required packages
        verify:   if True, hard-verify peft / ruptures / momentfm
        prewarm:  if True, pre-download DeBERTa + MOMENT (~2 GB, slow on bad net)
    """
    _set_hf_env_vars()

    if install:
        # Build tools first: py3.12 needs setuptools>=69 (older crashes
        # with 'pkgutil has no attribute ImpImporter' when building wheels)
        _pip(['--upgrade', 'pip', 'setuptools', 'wheel', 'cython'])

        # Scientific Python (usually present but ensured)
        _pip(['pandas', 'numpy>=1.26', 'matplotlib', 'seaborn', 'scipy',
              'scikit-learn', 'einops', 'numba'])

        # Task 1 NLP baseline
        _pip(['vaderSentiment'])

        # HuggingFace stack
        # transformers>=4.45 needed for torch_compile + dataloader_persistent_workers
        # peft>=0.13 needed for Blackwell sm_120 LoRA stability
        _pip([
            'transformers>=4.49',
            'peft>=0.13',
            'datasets',
            'accelerate',
            'sentencepiece',
        ])
        print("transformers>=4.49 installed; DeBERTa forced to fp32 "
              "(bf16 NaN'd at step 100, see env.py comment)")

        # Task 2 change-point detection: ruptures + claspy ship py3.12 wheels
        _pip([
            '--only-binary=:all:',
            'ruptures>=1.1.10',
            'claspy>=0.2.6',
        ])

        # momentfm: PyPI 0.1.4 pins numpy==1.25.2 which has no py3.12 wheel.
        # Install --no-deps to skip the broken pin (our numpy 1.26+ is fine).
        _pip(['--no-deps', '--no-build-isolation', 'momentfm'])

    if verify:
        print("Verifying critical packages:")
        _verify('peft',     'peft>=0.13')
        _verify('ruptures', 'ruptures>=1.1.10')
        _verify('momentfm', 'momentfm')
        print("\nAll required packages ready.")

    # Disk-space pre-check
    check_dir = Path.cwd()
    free_gb = shutil.disk_usage(check_dir).free / (1024 ** 3)
    if free_gb < 10:
        print(f"WARNING: only {free_gb:.1f} GB free in {check_dir}. "
              "Recommend >=10 GB for full pipeline (peak ~5-8 GB).")
    else:
        print(f"Disk space ok: {free_gb:.1f} GB free in {check_dir}")

    if prewarm:
        _prewarm_hf_models()


def _prewarm_hf_models():
    """Pre-download HF models (~2 GB). Off by default; flip when on bad network."""
    print("\nPre-warming HF caches (downloads ~2 GB, takes 1-3 min on fast network)...")
    try:
        from transformers import AutoModelForSequenceClassification
        AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=1, problem_type="regression")
        print("  ok  microsoft/deberta-v3-base")
    except Exception as e:
        print(f"  FAIL DeBERTa pre-warm: {type(e).__name__}: {e}")
    try:
        from momentfm import MOMENTPipeline
        MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={'task_name': 'classification', 'n_channels': 1, 'num_class': 2,
                          'freeze_encoder': True, 'reduction': 'mean'})
        print("  ok  AutonLab/MOMENT-1-large")
    except Exception as e:
        print(f"  FAIL MOMENT pre-warm: {type(e).__name__}: {e}")
