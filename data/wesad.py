"""WESAD loader.

Expects the standard layout: <root>/S2/S2.pkl, <root>/S3/S3.pkl, ...
Each pickle holds synchronised chest (RespiBAN, 700 Hz) and wrist (Empatica)
signals plus a per-sample label array.

WESAD labels: 0 undefined, 1 baseline, 2 stress, 3 amusement, 4 meditation,
5-7 ignore. We use the chest ECG, detect R-peaks, and window the resulting RR
series.

STATE MAPPING (a modelling decision, not ground truth):
    baseline, amusement -> GREEN
    meditation          -> GREEN
    stress (TSST)       -> RED
There is no native AMBER in WESAD. Acute lab stress is binary by design, so a
model trained on WESAD alone will not learn the intermediate state that
matters most operationally. Pair it with a dataset that has graded load.
"""
import pickle

import numpy as np

from ecg import quality, rr_from_ecg
from harmonize import GREEN, RED, window_rr

FS_CHEST = 700.0
LABEL_MAP = {1: GREEN, 2: RED, 3: GREEN, 4: GREEN}
MIN_QUALITY = 0.85


def load_wesad(root):
    files = sorted(root.glob("S*/S*.pkl"))
    if not files:
        files = sorted(root.glob("**/S*.pkl"))
    if not files:
        raise FileNotFoundError(f"no S*.pkl under {root}")

    X, y, groups = [], [], []
    for path in files:
        subject = path.stem
        with open(path, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")

        ecg = np.asarray(d["signal"]["chest"]["ECG"]).ravel()
        labels = np.asarray(d["label"]).ravel()

        for raw_label, state in LABEL_MAP.items():
            mask = labels == raw_label
            if mask.sum() < FS_CHEST * 90:      # need at least 90 s
                continue
            # Take the longest contiguous run of this label.
            idx = np.flatnonzero(mask)
            breaks = np.flatnonzero(np.diff(idx) > 1)
            runs = np.split(idx, breaks + 1)
            seg = max(runs, key=len)

            rr = rr_from_ecg(ecg[seg], FS_CHEST)
            q = quality(rr)
            if q < MIN_QUALITY:
                print(f"  {subject} label={raw_label}: detection quality {q:.2f}, skipped")
                continue

            xs, ys = window_rr(rr, state)
            X.extend(xs)
            y.extend(ys)
            groups.extend([subject] * len(xs))

        print(f"  {subject}: {len(X)} cumulative windows")

    return X, y, groups
