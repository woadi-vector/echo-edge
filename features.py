"""Reference feature extractor.

This is the training-side twin of `echo_features()` in core/echo.c. The two
must agree, or the model is scored on different numbers than it was trained
on. `make parity` compares them on shared vectors and fails on drift.
"""
import numpy as np

FEATURE_NAMES = [
    "mean_rr", "mean_hr", "sdnn", "rmssd",
    "pnn50", "rr_slope", "hr_cv", "coverage",
]
N_FEATURES = len(FEATURE_NAMES)

MIN_BEATS = 20
RR_MIN, RR_MAX = 250.0, 2500.0
WINDOW_MS = 60000.0


def clean(rr):
    """Drop implausible intervals — ectopic beats and dropped detections."""
    rr = np.asarray(rr, dtype=np.float64)
    return rr[(rr >= RR_MIN) & (rr <= RR_MAX)]


def extract(rr, window_ms=WINDOW_MS):
    """RR intervals in ms -> feature vector. Returns None if underfilled."""
    rr = clean(rr)
    if rr.size < MIN_BEATS:
        return None

    n = rr.size
    mean_rr = rr.mean()
    sdnn = np.sqrt(max((rr**2).mean() - mean_rr**2, 0.0))
    d = np.diff(rr)
    rmssd = np.sqrt((d**2).sum() / (n - 1)) if n > 1 else 0.0
    pnn50 = float((np.abs(d) > 50.0).sum()) / (n - 1) if n > 1 else 0.0

    i = np.arange(n, dtype=np.float64)
    denom = n * (i**2).sum() - i.sum() ** 2
    slope = (n * (i * rr).sum() - i.sum() * rr.sum()) / denom if denom else 0.0

    coverage = min(rr.sum() / window_ms, 1.0)

    return np.array([
        mean_rr,
        60000.0 / mean_rr,
        sdnn,
        rmssd,
        pnn50,
        slope,
        sdnn / mean_rr if mean_rr else 0.0,
        coverage,
    ], dtype=np.float64)
