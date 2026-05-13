"""
Sec 10 Event database + helper functions.

EVENT_ANCHORS: 5 phones x 1 primary documented event each (the 5-anchor design)
KNOWN_EVENTS:  5 phones x 3-6 events each (21 events total, for Sec 10.5 reverse-direction
               + Sec 10.5 multi-anchor analyses)
"""

import pickle

import numpy as np
import pandas as pd

from ..config.paths import OUTPUTS_DIR, T2_RECORDS_PATH


T2_EVENT_VALIDATION_DIR = OUTPUTS_DIR / "task2_event_validation"


# ====================================================================
# Single primary event per phone (Sec 10 5-anchor design)
# ====================================================================
EVENT_ANCHORS = {
    'B00GXHPN1U': {
        'title': 'BLU Advance',
        'event_label': 'Adups firmware disclosure',
        'event_date': pd.Timestamp('2016-11-15'),
        'strength': 'HIGH',
    },
    'B00E45043A': {
        'title': 'Nokia Lumia 520',
        'event_label': 'Windows Phone 8.1 EOL',
        'event_date': pd.Timestamp('2017-07-11'),
        'strength': 'HIGH',
    },
    'B00IZ1XJ3Q': {
        'title': 'Samsung Galaxy S5',
        'event_label': 'Note 7 discontinuation (brand shock)',
        'event_date': pd.Timestamp('2016-10-11'),
        'strength': 'MEDIUM',
    },
    'B00MWI4IN8': {
        'title': 'Motorola Moto G2',
        'event_label': 'Moto G3 launch (successor)',
        'event_date': pd.Timestamp('2015-07-28'),
        'strength': 'MEDIUM-LOW',
    },
    'B00A29WCA0': {
        'title': 'Samsung Galaxy S3 Mini',
        'event_label': 'Galaxy S5 launch (successor)',
        'event_date': pd.Timestamp('2014-04-11'),
        'strength': 'LOW',
    },
}


# ====================================================================
# Curated 21-event database (literature-verified dates, for Sec 10.5)
# ====================================================================
KNOWN_EVENTS = {
    'B00GXHPN1U': {
        'title': 'BLU Advance',
        'events': [
            ('2015-03-15', 'BLU Studio Energy launch (~approx)',                  'LOW'),
            ('2016-11-15', 'Kryptowire Adups firmware disclosure',                'HIGH'),
            ('2017-07-31', 'Amazon temporarily suspends BLU phone sales',         'HIGH'),
        ],
    },
    'B00E45043A': {
        'title': 'Nokia Lumia 520',
        'events': [
            ('2014-04-25', 'Microsoft acquires Nokia mobile (rebranding starts)', 'MEDIUM'),
            ('2014-08-07', 'Lumia 530 launch (successor)',                        'MEDIUM'),
            ('2014-11-14', 'Lumia 535 launch (first Microsoft-branded Lumia)',    'MEDIUM'),
            ('2015-04-09', 'Lumia 640 launch (Lumia rebrand complete)',           'MEDIUM'),
            ('2017-07-11', 'Windows Phone 8.1 end-of-support (OS dies)',          'HIGH'),
        ],
    },
    'B00IZ1XJ3Q': {
        'title': 'Samsung Galaxy S5',
        'events': [
            ('2015-04-10', 'Galaxy S6 launch (direct successor)',                 'MEDIUM'),
            ('2015-08-13', 'Galaxy Note 5 launch (same-brand flagship)',          'LOW'),
            ('2016-03-11', 'Galaxy S7 launch (cannibalization)',                  'MEDIUM'),
            ('2016-09-02', 'Galaxy Note 7 first recall',                          'MEDIUM'),
            ('2016-10-11', 'Galaxy Note 7 second recall (sentiment turning pt)',  'MEDIUM'),
            ('2017-04-21', 'Galaxy S8 launch',                                    'MEDIUM'),
        ],
    },
    'B00MWI4IN8': {
        'title': 'Motorola Moto G2',
        'events': [
            ('2014-10-30', 'Lenovo acquires Motorola Mobility from Google',       'MEDIUM'),
            ('2015-07-28', 'Moto G3 launch (direct successor)',                   'MEDIUM'),
            ('2016-05-17', 'Moto G4 launch',                                      'LOW'),
        ],
    },
    'B00A29WCA0': {
        'title': 'Samsung Galaxy S3 Mini',
        'events': [
            ('2014-04-11', 'Galaxy S5 launch (immediate-superior successor)',     'LOW'),
            ('2014-09-03', 'Galaxy Note 4 launch',                                'LOW'),
            ('2014-12-11', 'Galaxy A3/A5 announcement (budget-line shift)',       'LOW'),
            ('2015-04-10', 'Galaxy S6 launch',                                    'LOW'),
        ],
    },
}

# Convert string dates -> pd.Timestamp once at import
for _asin, _info in KNOWN_EVENTS.items():
    _info['events'] = [(pd.Timestamp(d), lbl, s) for d, lbl, s in _info['events']]


TOLERANCES = [2, 3, 4]
TOLERANCES_EXTENDED = [2, 3, 4, 5, 6, 7]
TOLERANCE_PRIMARY = 3
WINDOW_MONTHS = 6
BOOTSTRAP_CI_N = 1000


def load_records():
    """Load monthly records from T2_RECORDS_PATH."""
    if not T2_RECORDS_PATH.exists():
        raise FileNotFoundError(f"{T2_RECORDS_PATH} missing - run Sec 9.3 first to materialise records.pkl")
    with open(T2_RECORDS_PATH, 'rb') as f:
        return pickle.load(f)


def records_by_asin(records):
    return {r['asin']: r for r in records}


def ix_to_date(asin, ix, records_dict):
    """Convert cp integer-index to pd.Timestamp via stored dates array."""
    r = records_dict.get(asin)
    if r is None:
        return None
    dates = pd.to_datetime(r['dates'])
    if 0 <= ix < len(dates):
        return dates[ix]
    return None


def months_apart(d1, d2):
    """Calendar-month difference between two timestamps (always non-negative)."""
    return abs((d1.year - d2.year) * 12 + (d1.month - d2.month))


def random_hit_prob(n_obs, tol, n_cps):
    """P(at least 1 of n_cps cps placed uniformly in [0, n_obs) lands within +/-tol of fixed point)."""
    if n_cps == 0 or n_obs <= 0:
        return 0.0
    p_single = min(1.0, (2 * tol + 1) / n_obs)
    return 1.0 - (1.0 - p_single) ** n_cps


def load_route_cps_dict(r1_path, r2_path, r3_path, r4_path):
    """Load route_cps_dict from the 4 route CSV files. Returns dict route_name -> {asin: cps_list}."""
    import json
    from pathlib import Path

    def _load(csv_path, cp_col):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  WARN: {csv_path.name} missing - route excluded")
            return {}
        df = pd.read_csv(csv_path)
        if cp_col not in df.columns:
            print(f"  WARN: column '{cp_col}' not in {csv_path.name} - route excluded")
            return {}
        return dict(zip(df['asin'], df[cp_col].map(json.loads)))

    return {
        'PELT (R1)':    _load(r1_path, 'pelt_cps'),
        'AutoCPD (R2)': _load(r2_path, 'autocpd_cps'),
        'TST (R3)':     _load(r3_path, 'tst_cps'),
        'MOMENT (R4)':  _load(r4_path, 'moment_cps'),
    }
