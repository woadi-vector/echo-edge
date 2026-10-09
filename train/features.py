"""Reference feature extractor.

This is the training-side twin of `echo_features()` in core/echo.c. The two
must agree, or the model is scored on different numbers than it was trained
on. `make parity` compares them on shared vectors and fails on drift.

This module is a faithful port of the C, not an approximation of it. It
reproduces three things the earlier version left out, each of which made the
two sides disagree:

  1. The relative jump filter. The C rejects any interval differing from the
     last accepted one by more than RR_JUMP. Absolute bounds alone are not
     enough: a dropped beat merges two intervals into one of roughly double
     length, which at 73 bpm lands inside the plausible range.
  2. The rejection-run cap. Holding the anchor through a run of rejections is
     right for an isolated artifact, but a sustained step above RR_JUMP would
     otherwise reject every subsequent beat forever. After MAX_REJECT_RUN
     consecutive relative rejections the level is treated as real and the
     anchor moves.
  3. Window eviction. The C holds only the trailing WINDOW_MS of signal,
     evicting from the front once the span exceeds it. Without this a long
     series yields features over the whole recording on one side and over the
     last minute on the other.

Any change to echo_window_push or echo_features belongs here in the same
commit, and `make parity` is what catches it when it does not.
"""
import numpy as np

FEATURE_NAMES = [
    "mean_rr", "mean_hr", "sdnn", "rmssd",
    "pnn50", "rr_slope", "hr_cv", "coverage",
]
N_FEATURES = len(FEATURE_NAMES)

# These must match core/echo.c exactly.
MIN_BEATS = 20
RR_MIN, RR_MAX = 250.0, 2500.0
RR_JUMP = 0.25
MAX_REJECT_RUN = 4
MAX_BEATS = 512
WINDOW_MS = 60000.0


def push_all(rr, window_ms=WINDOW_MS):
    """Port of echo_window_push over a whole series.

    Returns (held, gap, n_accepted, n_rejected): the intervals the C window
    would be holding after the last push, the per-interval gap flags, and the
    accept and reject counts.
    """
    rr = np.asarray(rr, dtype=np.float64)
    held, gaps = [], []
    last = 0.0
    pending_gap = 0
    reject_run = 0
    n_acc = n_rej = 0
    span = 0.0

    for x in rr:
        if not (RR_MIN <= x <= RR_MAX):
            n_rej += 1
            pending_gap = 1
            continue

        if last > 0.0:
            ratio = x / last
            if ratio < 1.0 - RR_JUMP or ratio > 1.0 + RR_JUMP:
                if reject_run < MAX_REJECT_RUN:
                    reject_run += 1
                    n_rej += 1
                    pending_gap = 1
                    continue
                pending_gap = 1      # re-anchoring is itself a discontinuity

        reject_run = 0
        last = x
        n_acc += 1

        if len(held) == MAX_BEATS:
            span -= held[0]
            held.pop(0)
            gaps.pop(0)

        held.append(float(x))
        gaps.append(pending_gap)
        pending_gap = 0
        span += float(x)

        while len(held) > MIN_BEATS and span - held[0] > window_ms:
            span -= held[0]
            held.pop(0)
            gaps.pop(0)

    return (np.array(held, dtype=np.float64),
            np.array(gaps, dtype=bool), n_acc, n_rej)


def clean(rr):
    """Deprecated. Kept so older callers do not break.

    Returns the held intervals and a mask that is all True, because filtering
    now happens inside push_all and the held series contains no rejects.
    """
    held, _gaps, _a, _r = push_all(rr)
    return held, np.ones(held.size, dtype=bool)


def extract(rr, window_ms=WINDOW_MS):
    """RR intervals in ms -> feature vector. Returns None if underfilled."""
    held, gaps, _n_acc, _n_rej = push_all(rr, window_ms)
    n = held.size
    if n < MIN_BEATS:
        return None

    mean_rr = held.mean()
    # Centred, matching the C. The uncentred identity is unstable in float32
    # and the two sides must compute the same thing, not merely agree in
    # double precision where the instability is invisible.
    sdnn = np.sqrt(((held - mean_rr) ** 2).mean())

    # Successive differences only between beats adjacent in time. A beat
    # rejected between i-1 and i makes their difference meaningless, so it is
    # suppressed. The denominator counts differences actually taken, not
    # n - 1, which overcounts once any interval has been dropped.
    take = ~gaps[1:]
    d = np.diff(held)[take]
    nd = d.size
    rmssd = np.sqrt((d ** 2).sum() / nd) if nd else 0.0
    pnn50 = float((np.abs(d) > 50.0).sum()) / nd if nd else 0.0

    i = np.arange(n, dtype=np.float64)
    di = i - i.mean()
    dx = held - mean_rr
    sum_di2 = (di ** 2).sum()
    slope = (di * dx).sum() / sum_di2 if sum_di2 else 0.0

    coverage = min(held.sum() / window_ms, 1.0)

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
