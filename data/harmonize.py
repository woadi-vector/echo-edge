"""Dataset harmonizer.

Every corpus becomes the same thing: windows of RR intervals with a state
label, fed through the one feature extractor the C core mirrors. Adding a
dataset means writing one loader, not touching the model.

    from harmonize import load
    X, y, groups = load("wesad", "/path/to/WESAD")

`groups` carries the subject ID for every window. Use it — random splits on
windows leak, because consecutive windows from one subject are nearly
identical. Split by subject or the accuracy number is fiction.

State mapping is a modelling decision, not a fact about the data. Each loader
declares its own and says so out loud.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "train"))
from features import extract  # noqa: E402

GREEN, AMBER, RED = 0, 1, 2
WINDOW_MS = 60000.0
STEP_MS = 15000.0   # 45 s overlap between consecutive windows


def window_rr(rr, label, window_ms=WINDOW_MS, step_ms=STEP_MS):
    """Slide a window over one continuous RR series, emitting feature vectors."""
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 20:
        return [], []

    edges = np.concatenate([[0.0], np.cumsum(rr)])
    total = edges[-1]
    X, y = [], []
    start = 0.0
    while start + window_ms <= total:
        lo = np.searchsorted(edges, start)
        hi = np.searchsorted(edges, start + window_ms)
        f = extract(rr[lo:hi], window_ms=window_ms)
        if f is not None:
            X.append(f)
            y.append(label)
        start += step_ms
    return X, y


def load(name, root):
    """Dispatch to a loader. Returns (X, y, groups) as arrays."""
    if name == "wesad":
        from wesad import load_wesad as fn
    elif name == "swell":
        from swell import load_swell as fn
    elif name == "mmash":
        from mmash import load_mmash as fn
    elif name == "synthetic":
        from train import synthesize
        X, y = synthesize()
        return X, y, np.arange(len(X))     # every window its own group
    else:
        raise ValueError(f"unknown dataset: {name}")

    X, y, g = fn(pathlib.Path(root))
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    if X.size == 0:
        raise RuntimeError(f"{name}: no usable windows found under {root}")
    return X, y, g


def summarize(name, X, y, groups):
    counts = {s: int((y == i).sum()) for i, s in enumerate(["GREEN", "AMBER", "RED"])}
    print(f"{name}: {len(X)} windows, {len(set(groups))} subjects, {counts}")
