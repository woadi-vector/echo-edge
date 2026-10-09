"""Capacity sweep: how much forest does 2391 windows actually support?

This varies depth and tree count, records held-out subject accuracy and the
resulting model size, and restores the chosen configuration at the end so the
repository is never left holding a sweep artifact.

The question that prompted it was whether 100 trees at depth 15 is overfitting,
since in-sample accuracy is 0.998 against 0.833 held out. The answer, on WESAD,
is no: held-out accuracy rises monotonically from 0.763 at depth 4 to 0.832 at
depth 15 with no plateau. A large in-sample gap is what a random forest does by
construction, since every tree is grown to purity on its bootstrap sample, and
it is not by itself evidence of harmful overfitting. The held-out curve is the
only thing that answers the question. Keep that in mind before reading a
training number as a diagnosis.

    python train/sweep.py --root "C:\\path\\to\\WESAD"

Nothing is committed and nothing is decided here. It prints a table.
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train" / "train.py"
MODEL = ROOT / "core" / "echo_model.h"

ACC = re.compile(r"held-out subject accuracy:\s*([\d.]+)\s*\+/-\s*([\d.]+)")
BASE = re.compile(r"majority-class baseline:\s*([\d.]+)")
NODES = re.compile(r"#define ECHO_N_NODES\s+(\d+)")
LEAVES = re.compile(r"#define ECHO_N_LEAVES\s+(\d+)")


def run(root, dataset, trees, depth):
    out = subprocess.run(
        [sys.executable, str(TRAIN), "--dataset", dataset, "--root", root,
         "--trees", str(trees), "--depth", str(depth)],
        capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"train.py failed at trees={trees} depth={depth}")

    m, b = ACC.search(out.stdout), BASE.search(out.stdout)
    h = MODEL.read_text(errors="replace")
    n = int(NODES.search(h).group(1))
    l = int(LEAVES.search(h).group(1))
    # Packed sizes as echo_model.h declares them: a node is one float plus
    # three int16; a leaf is three floats.
    kb = (n * 10 + l * 12) / 1024.0
    return (float(m.group(1)), float(m.group(2)),
            float(b.group(1)) if b else float("nan"), n, l, kb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="wesad")
    ap.add_argument("--depths", default="4,6,8,10,12,15")
    ap.add_argument("--trees", default="100")
    ap.add_argument("--restore-trees", type=int, default=100)
    ap.add_argument("--restore-depth", type=int, default=15)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    trees = [int(t) for t in args.trees.split(",")]

    rows = []
    print(f"{'trees':>6s} {'depth':>6s} {'held-out':>9s} {'+/-':>6s} "
          f"{'nodes':>8s} {'leaves':>7s} {'KB':>8s}")
    for t in trees:
        for d in depths:
            acc, sd, base, n, l, kb = run(args.root, args.dataset, t, d)
            rows.append((t, d, acc, sd, n, l, kb))
            print(f"{t:6d} {d:6d} {acc:9.3f} {sd:6.3f} {n:8d} {l:7d} {kb:8.0f}")

    best = max(rows, key=lambda r: r[2])
    print(f"\nmajority-class baseline: {base:.3f}")
    print(f"highest held-out: {best[2]:.3f} at trees={best[0]} depth={best[1]}, "
          f"{best[6]:.0f} KB")

    # Marginal gain per setting, which is what the decision actually turns on.
    #
    # An earlier version of this script reported the smallest configuration
    # within one fold standard deviation of the best, and that rule is wrong.
    # Every configuration is scored on the identical folds with the identical
    # subjects, so a comparison between two of them is paired and its variance
    # is much smaller than the between-subject spread the fold SD measures.
    # Using the fold SD as a tolerance makes differences look unresolvable when
    # they are not, and on WESAD it recommended depth 6 over depth 15 across a
    # real 0.046 gap. Marginal gain against marginal size is reported instead,
    # and the choice is left to whoever knows the storage constraint.
    print("\nmarginal gain per step, in the order swept:")
    prev = None
    for r in rows:
        if prev is not None and r[0] == prev[0]:
            print(f"  depth {prev[1]:2d} -> {r[1]:2d}: "
                  f"{r[2] - prev[2]:+.3f} accuracy for {r[6] - prev[6]:+.0f} KB")
        prev = r
    print("\nNo storage constraint means take the highest accuracy. A constraint "
          "means read the table above and decide what the last points cost.")

    print(f"\nrestoring trees={args.restore_trees} depth={args.restore_depth}")
    run(args.root, args.dataset, args.restore_trees, args.restore_depth)
    print("core/echo_model.h and bench/testvectors.h are back to that setting")


if __name__ == "__main__":
    main()
