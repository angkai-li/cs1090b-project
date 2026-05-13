"""
Raw data loading + cleaned-cache validity.

Two layers:
  - `load_json_lines(path)`: read a JSON-lines file into a DataFrame.
  - `clean_cache_is_valid(raw_path)`: check whether the cleaned-df cache at
    CLEAN_CACHE_PATH was built from a byte-identical raw file.
  - `load_raw_if_needed(raw_path)`: convenience - returns None if cache valid,
    else loads the raw JSON.
"""

import json
from pathlib import Path

import pandas as pd

from ..config.paths import CLEAN_CACHE_PATH, CLEAN_CACHE_META_PATH
from ..utils.io import read_json_or_none, raw_file_signature


def load_json_lines(path):
    """Read a JSON-lines file (one JSON object per line) into a DataFrame."""
    rows = []
    with open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def clean_cache_is_valid(raw_path):
    """True iff the cleaned-df cache exists and matches the raw file's byte size.

    File-existence + raw-file signature (data integrity). The raw_signature check
    detects upstream data changes (e.g. new dataset version) and is NOT a
    code-version check - to force rebuild, delete CLEAN_CACHE_PATH.
    """
    if not CLEAN_CACHE_PATH.exists() or not CLEAN_CACHE_META_PATH.exists():
        return False
    if not Path(raw_path).exists():
        return False
    meta = read_json_or_none(CLEAN_CACHE_META_PATH)
    if meta is None:
        return False
    return meta.get("raw_signature") == raw_file_signature(raw_path)


def load_raw_if_needed(raw_path):
    """Load raw JSON if the cleaned cache isn't usable; otherwise return None.

    Use this when the next stage will load `df` from CLEAN_CACHE_PATH on cache hit.
    Returns the raw DataFrame on cache miss so cleaning can rebuild it.
    """
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Missing data file: {raw_path}")
    if clean_cache_is_valid(raw_path):
        return None
    return load_json_lines(raw_path)
