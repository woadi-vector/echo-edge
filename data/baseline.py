"""Per-operator baseline normalization.

A global z-score asks "is this heart rate high?" A per-operator baseline asks
"is this heart rate high *for them*?" Only the second question is answerable,
because a resting rate of 52 and one of 78 are both normal for their owners.

The deployable form is an enrollment step: record a short quiet period once
per operator, store their baseline vector, and thereafter express every
reading as deviation from it. Two minutes at intake, then never again.

WHAT THIS ASSUMES
    The enrollment windows are actually taken at rest. In WESAD that holds —
    the protocol opens with a 20-minute baseline block and the loader emits
    windows in protocol order. On a corpus where the recording does not start
    calm, enrollment has to be selected rather than assumed, or the baseline
    is anchored to whatever the person happened to be doing.

Enrollment windows are dropped from the returned set. They would sit at
exactly zero deviation and inflate the GREEN class with degenerate rows.
"""
import numpy as np

DEFAULT_ENROLL = 8   # windows; at 15 s step this is under three minutes


def fit_scale(X):
    """Population spread per feature, used as the common denominator.

    Dividing by each subject's *own* spread would erase genuine differences in
    reactivity — a person whose HRV barely moves would be rescaled to look as
    volatile as anyone else. The offset is personal; the scale stays shared.
    """
    scale = np.asarray(X, dtype=np.float64).std(axis=0)
    scale[scale == 0.0] = 1.0
    return scale


def personalize(X, y, groups, n_enroll=DEFAULT_ENROLL, scale=None, rest_label=None):
    """Re-express each subject's windows as deviation from their own baseline.

    rest_label=None takes each subject's first n_enroll windows, which is
    correct only when the recording is known to open at rest (WESAD does).
    Passing a label instead draws enrollment from that subject's windows
    carrying it — the faithful model of field enrollment, where you ask the
    operator to sit still and therefore know the period is resting.

    Returns (X, y, groups) with enrollment windows removed.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if scale is None:
        scale = fit_scale(X)

    out_X, out_y, out_g = [], [], []
    dropped = []

    for g in dict.fromkeys(groups):          # preserves first-seen order
        mask = groups == g
        Xi, yi = X[mask], y[mask]
        if Xi.shape[0] <= n_enroll + 4:
            dropped.append(g)
            continue

        if rest_label is None:
            enroll_idx = np.arange(n_enroll)
        else:
            rest = np.flatnonzero(yi == rest_label)
            if rest.size < n_enroll:
                dropped.append(g)
                continue
            enroll_idx = rest[:n_enroll]

        baseline = Xi[enroll_idx].mean(axis=0)
        keep = np.setdiff1d(np.arange(Xi.shape[0]), enroll_idx)
        out_X.append((Xi[keep] - baseline) / scale)
        out_y.append(yi[keep])
        out_g.extend([g] * keep.size)

    if dropped:
        print(f"  baseline: {len(dropped)} subject(s) too short to enroll: {dropped}")
    if not out_X:
        raise RuntimeError("no subject had enough windows to enroll")

    return np.vstack(out_X), np.concatenate(out_y), np.array(out_g)
