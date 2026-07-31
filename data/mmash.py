"""MMASH loader (PhysioNet).

Layout: <root>/user_1/RR.csv, user_2/RR.csv, ... with an `ibi_s` column of
inter-beat intervals in seconds. Native RR data, so no detection step — the
same representation the Polar H10 emits.

MMASH has no stress labels attached to the interval stream. It carries
questionnaire scores (Daily Stress Inventory, PANAS, anxiety) at the
subject-day level. Rather than invent per-window labels, this loader returns
every window as GREEN by default and expects you to supply a labelling
function for anything else.

USE THIS FOR BASELINE STRUCTURE, NOT SUPERVISION. Its value is showing what
ordinary 24-hour variation looks like in RR terms — the distribution a
readiness score should sit inside. Also note the cohort is 22 young adult
males, which is a poor match for scoring female athletes.
"""
import numpy as np
import pandas as pd

from harmonize import GREEN, window_rr


def load_mmash(root, label_fn=None):
    files = sorted(root.glob("**/RR.csv"))
    if not files:
        raise FileNotFoundError(f"no RR.csv under {root}")

    X, y, groups = [], [], []
    for path in files:
        subject = path.parent.name
        df = pd.read_csv(path)
        col = next((c for c in df.columns if "ibi" in c.lower()), None)
        if col is None:
            print(f"  {subject}: no ibi column, skipped")
            continue

        rr = df[col].to_numpy(dtype=np.float64) * 1000.0    # seconds -> ms
        rr = rr[(rr >= 250.0) & (rr <= 2500.0)]

        label = GREEN if label_fn is None else label_fn(subject)
        xs, ys = window_rr(rr, label)
        X.extend(xs)
        y.extend(ys)
        groups.extend([subject] * len(xs))
        print(f"  {subject}: {len(xs)} windows")

    return X, y, groups
