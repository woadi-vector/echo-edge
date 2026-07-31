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
import re

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


REQUIRED = ("MEAN_RR", "RMSSD")

# Per-participant files only: p1.csv ... p25.csv. The Kaggle release also
# ships combined/train/test/validation/unseen repackagings of the SAME rows,
# and a set of WESAD-derived CSVs. Loading those multiplies every observation
# several times over and mixes two corpora, which produces leakage rather than
# data. Subject identity also comes from these filenames.
PARTICIPANT = re.compile(r"^p(\d+)\.csv$", re.IGNORECASE)


def load_swell(root):
    """Find the SWELL CSVs under root, ignoring everything else.

    The dataset directory usually holds several corpora side by side, so this
    tries every CSV and keeps only files that carry the expected HRV columns.
    Empty and unreadable files are skipped rather than fatal.
    """
    candidates = [q for q in sorted(root.glob("**/*.csv"))
                  if PARTICIPANT.match(q.name)]
    if not candidates:
        raise FileNotFoundError(
            f"no per-participant SWELL files (p1.csv ... pNN.csv) under {root}")

    # Validate BEFORE deduplicating. `databases` can hold several different
    # p*.csv sets — the SWELL HRV features, and unrelated per-participant
    # files sharing the same names. Deduplicating on filename first would
    # discard the right copy whenever the wrong one sorts earlier.
    frames, used, skipped, seen = [], [], [], set()
    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            skipped.append(f"{path.name}: {type(exc).__name__}")
            continue
        frame.columns = [c.strip() for c in frame.columns]
        if not all(c in frame.columns for c in REQUIRED):
            skipped.append(f"{path.name}: no HRV columns ({path.parent.name})")
            continue
        key = path.name.lower()
        if key in seen:
            skipped.append(f"{path.name}: duplicate copy")
            continue
        seen.add(key)
        frame["__subject"] = path.stem.lower()   # subject from filename
        frames.append(frame)
        used.append(path.name)

    if not frames:
        raise FileNotFoundError(
            f"no SWELL-shaped CSV under {root}\n  checked: "
            + "\n  checked: ".join(skipped[:12]))

    print(f"  swell participants: {len(used)} files "
          f"({used[0]} ... {used[-1]})")
    if skipped:
        print(f"  skipped {len(skipped)} unreadable file(s)")

    df = pd.concat(frames, ignore_index=True)

    cond_col = next((c for c in df.columns if c.lower() in
                     ("condition", "label", "class")), None)
    if cond_col is None:
        raise KeyError(f"no condition column in {list(df.columns)[:12]}")

    subj_col = "__subject"

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

    groups = df[subj_col].astype(str).to_numpy()
    print(f"  swell: {n} rows, rr_slope and coverage synthesised")
    return X, labels.to_numpy(), groups
