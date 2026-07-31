"""SWELL-KW loader (the Kaggle HRV-feature release).

That release ships precomputed HRV features rather than raw intervals, so
there is nothing to detect — but it also means two of Echo's eight features
cannot be reconstructed:

    rr_slope  - needs the ordered interval series, not summary statistics
    coverage  - window completeness is not recorded

Both are filled with neutral values, which is a real limitation: a model
trained here has learned to ignore drift velocity, and drift velocity is the
feature the V1.1 regression roadmap is built on. Use SWELL for cross-checking
the static HRV features, not as a primary corpus.

STATE MAPPING:
    no stress    -> GREEN
    interruption -> AMBER
    time pressure-> RED
"""
import numpy as np
import pandas as pd

from harmonize import AMBER, GREEN, RED

CONDITION_MAP = {
    "no stress": GREEN, "no_stress": GREEN, "nostress": GREEN,
    "interruption": AMBER,
    "time pressure": RED, "time_pressure": RED, "timepressure": RED,
}

# SWELL column -> Echo feature index
COLUMN_MAP = {
    "MEAN_RR": 0, "HR": 1, "SDRR": 2, "RMSSD": 3, "pNN50": 4,
}


def load_swell(root):
    files = sorted(root.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV under {root}")

    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    cond_col = next((c for c in df.columns if c.lower() in
                     ("condition", "label", "class")), None)
    if cond_col is None:
        raise KeyError(f"no condition column in {list(df.columns)[:12]}")

    subj_col = next((c for c in df.columns if c.lower() in
                     ("subject", "subject_id", "id", "participant")), None)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise KeyError(f"missing expected HRV columns: {missing}")

    labels = df[cond_col].astype(str).str.strip().str.lower().map(CONDITION_MAP)
    keep = labels.notna()
    df, labels = df[keep], labels[keep].astype(int)

    n = len(df)
    X = np.zeros((n, 8), dtype=np.float64)
    for col, idx in COLUMN_MAP.items():
        X[:, idx] = df[col].to_numpy(dtype=np.float64)

    X[:, 4] = np.where(X[:, 4] > 1.0, X[:, 4] / 100.0, X[:, 4])   # percent -> fraction
    X[:, 5] = 0.0                                                  # rr_slope unavailable
    X[:, 6] = np.divide(X[:, 2], X[:, 0], out=np.zeros(n), where=X[:, 0] > 0)
    X[:, 7] = 1.0                                                  # coverage unknown

    groups = (df[subj_col].astype(str).to_numpy() if subj_col
              else np.zeros(n, dtype=str))
    print(f"  swell: {n} rows, rr_slope and coverage synthesised")
    return X, labels.to_numpy(), groups
