"""
ClaSP wrapper (Route 1 secondary classical baseline).

ClaSP (Ermshaus, Schafer, Leser DAMI 2023) is parameter-free time-series
segmentation. Used as a second-opinion classical change-point detector alongside
PELT in Route 1. Outputs go to the `clasp_cps` column of change_points.csv.

The actual call is inlined inside pelt._process_one() to keep that worker
self-contained for joblib pickling. This module just exposes a single helper
for direct (non-parallel) use.
"""

import numpy as np


def detect_cps_clasp(series):
    """Run ClaSP on a 1-D series. Returns a list of int change-point positions.

    Returns [] if claspy isn't installed or if ClaSP fails (rare).
    """
    try:
        from claspy.segmentation import BinaryClaSPSegmentation
    except ImportError:
        return []
    try:
        clasp = BinaryClaSPSegmentation()
        clasp.fit(np.asarray(series, dtype=float))
        return list(map(int, clasp.change_points))
    except Exception:
        return []
