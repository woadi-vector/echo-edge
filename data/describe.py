"""Print per-feature distributions for a dataset.

NOTE: this file must not be called inspect.py — numpy imports the standard
library module of that name, and a local file shadows it.

Use before training, not after. Two corpora that look similar in accuracy can
be wildly different in scale, and a model trained on one will not transfer to
the other no matter how good its cross-validation looked. This is how you find
that out in thirty seconds instead of after a deployment.
"""
import argparse
import sys

import numpy as np

import harmonize
sys.path.insert(0, "../train")
from features import FEATURE_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("root")
    args = ap.parse_args()

    X, y, g = harmonize.load(args.dataset, args.root)
    X = np.asarray(X)
    harmonize.summarize(args.dataset, X, np.asarray(y), np.asarray(g))

    print(f"\n{'feature':<10} {'mean':>10} {'sd':>10} {'p5':>10} {'p95':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f"{name:<10} {col.mean():>10.2f} {col.std():>10.2f} "
              f"{np.percentile(col, 5):>10.2f} {np.percentile(col, 95):>10.2f}")

    # Between-subject spread of each feature's mean. Large values are the
    # argument for per-operator baselining.
    print(f"\n{'feature':<10} {'between-subject sd of mean':>28}")
    for i, name in enumerate(FEATURE_NAMES):
        per = [X[np.asarray(g) == s][:, i].mean() for s in set(g)]
        print(f"{name:<10} {np.std(per):>28.2f}")


if __name__ == "__main__":
    main()
