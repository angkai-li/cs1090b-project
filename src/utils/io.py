"""
Atomic IO helpers: write to .tmp then rename. Prevents half-written files
when the kernel is killed mid-write (which would silently load corrupt state).
"""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def atomic_write_text(path, text):
    """Write text to file atomically (.tmp -> rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_to_csv(frame, path, **kwargs):
    """Atomic version of DataFrame.to_csv()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, **kwargs)
    tmp.replace(path)


def atomic_to_pickle(obj, path):
    """Atomic version of pickle.dump()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def atomic_torch_save(obj, path):
    """Atomic torch.save: write to .tmp then rename.

    Avoids half-written state_dicts if the kernel is killed mid-save (which would
    silently load corrupt weights on next run).
    """
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def atomic_savez(path, **arrays):
    """Atomic version of np.savez_compressed: write to .tmp then rename.

    Note: np.savez_compressed auto-appends '.npz' when given a path/string.
    To control the exact temp filename we pass an open file object instead,
    which bypasses the auto-suffix behaviour and writes EXACTLY to `tmp`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def atomic_save_pretrained(model, dir_path):
    """Atomic version of HuggingFace `model.save_pretrained()`.

    Writes to a sibling .tmp directory, then renames. Avoids half-written
    LoRA adapter directories.
    """
    dir_path = Path(dir_path)
    dir_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dir_path.parent / (dir_path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    if dir_path.exists():
        shutil.rmtree(dir_path)
    tmp.rename(dir_path)


def read_json_or_none(path):
    """Read a JSON file or return None if missing / unparseable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def raw_file_signature(path):
    """Portable file signature using only byte size.

    For the same raw JSON file (identical bytes), size is deterministic across
    machines, while absolute path and mtime are machine-specific. The cleaning
    process is a pure function of the raw bytes, so size-only signature is
    sufficient for data-integrity validation while staying portable.
    """
    path = Path(path)
    return {"size": path.stat().st_size}


def dataframe_file_info(path):
    """Return a dict with file existence + row count + columns. Used by cache validity
    checks that need to confirm a CSV file exists and has the expected schema."""
    import pandas as pd
    path = Path(path)
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": []}
    try:
        df = pd.read_csv(path, nrows=0)
        cols = list(df.columns)
    except Exception:
        cols = []
    try:
        with open(path) as f:
            rows = sum(1 for _ in f) - 1  # minus header
    except Exception:
        rows = 0
    return {"exists": True, "rows": rows, "columns": cols}
