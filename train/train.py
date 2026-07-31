"""Train the Echo readiness classifier and export it as portable C.

Input:  a CSV of RR-interval windows, or the synthetic generator below.
Output: core/echo_model.h  (scaler constants + flattened forest)
        bench/testvectors.h (parity fixtures shared with the C core)

PLACEHOLDER DATA WARNING
------------------------
`synthesize()` produces physiologically shaped but *fabricated* data so the
pipeline runs end to end on day one. It is not evidence of anything. Replace
it with WESAD/SWELL-derived windows, or real Polar H10 captures, before any
accuracy number leaves this repo.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))
from features import FEATURE_NAMES, N_FEATURES, extract  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATES = ["GREEN", "AMBER", "RED"]


def band(proba, lo, hi):
    """Map p(RED) to a three-state readout.

    A two-class corpus cannot teach AMBER, but the model still reports how
    confident it is. Calling the uncertain middle AMBER is honest labelling of
    what the model actually knows — "transitioning or unclear" — rather than
    an invented class. Below lo is GREEN, above hi is RED, between is AMBER.
    """
    p_red = np.asarray(proba)
    out = np.full(p_red.shape, 1, dtype=int)
    out[p_red < lo] = 0
    out[p_red > hi] = 2
    return out


def synthesize(n_per_class=1200, seed=7):
    """Fabricated RR windows with class-dependent HR and HRV structure."""
    rng = np.random.default_rng(seed)
    profiles = [
        # (mean HR, HRV scale, drift ms/beat)
        (62.0, 55.0, 0.0),    # GREEN: low HR, high variability, stable
        (88.0, 28.0, -0.35),  # AMBER: elevated, variability compressing
        (118.0, 11.0, -0.9),  # RED:   high HR, variability collapsed
    ]
    X, y = [], []
    for label, (hr, hrv, drift) in enumerate(profiles):
        for _ in range(n_per_class):
            beats = int(rng.integers(45, 110))
            base = 60000.0 / (hr + rng.normal(0, 6))
            noise = rng.normal(0, max(hrv + rng.normal(0, 6), 3.0), beats)
            trend = drift * np.arange(beats) * rng.uniform(0.5, 1.5)
            rr = base + noise + trend
            f = extract(rr)
            if f is not None:
                X.append(f)
                y.append(label)
    return np.array(X), np.array(y)


def emit_tree(t, classes, leaf_probs):
    """One sklearn tree -> depth-first preorder packed nodes.

    Preorder means the left child always lands immediately after its parent,
    so the C traversal only stores an offset to the right child. Nodes are
    (threshold, right_offset, feature, class); feature -1 marks a leaf.

    `classes` maps sklearn's internal column index to the actual class label.
    They diverge whenever the training data is missing a class — with only
    GREEN and RED present, column 1 means RED, not AMBER. Emitting the column
    index instead of the label silently mislabels every prediction.

    Leaves store an index into a shared probability table rather than a hard
    class. sklearn's predict() averages the per-tree class distributions and
    takes the argmax of the mean; majority-voting each tree's own argmax is a
    different estimator and disagrees on impure leaves. Since a leaf never
    uses `right_off`, the index rides in that field for free.
    """
    out = []

    def walk(i):
        idx = len(out)
        if t.children_left[i] == -1:
            v = np.asarray(t.value[i][0], dtype=np.float64)
            total = v.sum()
            dist = v / total if total > 0 else np.full(len(v), 1.0 / len(v))
            full = np.zeros(3, dtype=np.float64)
            for col, label in enumerate(classes):
                full[label] = dist[col]
            leaf_probs.append(full)
            out.append([0.0, len(leaf_probs) - 1, -1, 0])
            return idx
        out.append([float(t.threshold[i]), 0, int(t.feature[i]), 0])
        walk(t.children_left[i])            # lands at idx + 1
        right = walk(t.children_right[i])
        out[idx][1] = right - idx
        return idx

    walk(0)
    return out


def flatten(forest):
    """Forest -> one contiguous packed node array plus per-tree roots."""
    classes = np.asarray(forest.classes_, dtype=int)
    nodes, roots, leaf_probs = [], [], []
    for est in forest.estimators_:
        roots.append(len(nodes))
        nodes.extend(emit_tree(est.tree_, classes, leaf_probs))
    if len(leaf_probs) > 32767:
        raise ValueError("too many leaves for an int16 index; reduce --depth")
    for thr, off, feat, klass in nodes:
        if not -32768 <= off <= 32767:
            raise ValueError(f"right offset {off} exceeds int16; reduce --depth")
    return nodes, roots, np.array(leaf_probs)


def node_array(nodes, f):
    rows = [f"    {{ {f(thr)}, {off}, {feat}, {klass} }},"
            for thr, off, feat, klass in nodes]
    body = "\n".join(rows)
    return f"static const echo_node_t ECHO_NODES[{len(nodes)}] = {{\n{body}\n}};\n"


def carray(name, ctype, values, fmt=str, per_line=12):
    rows = []
    for i in range(0, len(values), per_line):
        rows.append("    " + " ".join(fmt(v) + "," for v in values[i:i + per_line]))
    body = "\n".join(rows)
    return f"static const {ctype} {name}[{len(values)}] = {{\n{body}\n}};\n"


def export_header(path, scaler, forest, model_id, pop_scale=None,
                  amber_band=None):
    nodes, roots, leaf_probs = flatten(forest)
    def f(v):
        s = f"{v:.9g}"
        if not any(c in s for c in ".eE"):
            s += ".0"
        return s + "f"

    inv_scale = [1.0 / s if s else 0.0 for s in scaler.scale_]
    baselined = pop_scale is not None
    pop_inv = ([1.0 / s if s else 0.0 for s in pop_scale] if baselined
               else [1.0] * N_FEATURES)

    out = [
        "/* GENERATED by train/train.py — do not edit by hand. */",
        "#ifndef ECHO_MODEL_H",
        "#define ECHO_MODEL_H",
        '#include "echo.h"',
        "",
        f'#define ECHO_MODEL_ID "{model_id}"',
        f"#define ECHO_N_TREES {len(roots)}",
        f"#define ECHO_N_NODES {len(nodes)}",
        f"#define ECHO_BASELINED {1 if baselined else 0}",
        f"#define ECHO_AMBER_BAND {1 if amber_band else 0}",
        f"#define ECHO_BAND_LO {f(amber_band[0]) if amber_band else '0.0f'}",
        f"#define ECHO_BAND_HI {f(amber_band[1]) if amber_band else '1.0f'}",
        "",
        carray("ECHO_POP_INV_SCALE", "float", pop_inv, f),
        carray("ECHO_SCALER_MEAN", "float", list(scaler.mean_), f),
        carray("ECHO_SCALER_INV_SCALE", "float", inv_scale, f),
        f"#define ECHO_N_LEAVES {len(leaf_probs)}",
        "",
        carray("ECHO_TREE_ROOT", "int32_t", roots),
        node_array(nodes, f),
        carray("ECHO_LEAF_PROB", "float", list(leaf_probs.ravel()), f),
        "#endif /* ECHO_MODEL_H */",
    ]
    path.write_text("\n".join(out))


def _cf(v):
    s = f"{v:.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def export_testvectors(path, X, y_pred, n=64):
    rows = []
    for i in range(min(n, len(X))):
        vals = ", ".join(_cf(v) for v in X[i])
        rows.append(f"    {{ {{ {vals} }}, {int(y_pred[i])} }},")
    body = "\n".join(rows)
    path.write_text(
        "/* GENERATED by train/train.py — parity fixtures. */\n"
        "#ifndef ECHO_TESTVECTORS_H\n#define ECHO_TESTVECTORS_H\n\n"
        "typedef struct { float f[8]; int expect; } echo_fixture_t;\n\n"
        f"static const echo_fixture_t ECHO_FIXTURES[{min(n, len(X))}] = {{\n{body}\n}};\n\n"
        "#endif\n"
    )


def fit(X, y, args):
    scaler = StandardScaler().fit(X)
    forest = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.depth,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    ).fit(scaler.transform(X), y)
    return scaler, forest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dataset", default="synthetic",
                    help="synthetic | wesad | swell | mmash")
    ap.add_argument("--root", default=None, help="dataset directory")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--baseline", type=int, default=0, metavar="N",
                    help="per-operator baselining: enroll each subject on N "
                         "windows (0 = off)")
    ap.add_argument("--amber-band", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None,
                    help="derive AMBER from classifier confidence: p(RED) "
                         "below LO is GREEN, above HI is RED, between is AMBER")
    ap.add_argument("--enroll-rest", action="store_true",
                    help="draw enrollment from GREEN-labelled windows instead "
                         "of the first N; use when the recording does not "
                         "reliably begin at rest")
    args = ap.parse_args()

    if args.dataset == "synthetic":
        X, y = synthesize(seed=args.seed)
        groups = np.arange(len(X))
    else:
        if not args.root:
            ap.error("--root is required for a real dataset")
        import harmonize
        X, y, groups = harmonize.load(args.dataset, args.root)

    pop_scale = None
    if args.baseline > 0:
        from baseline import fit_scale, personalize
        n_before = len(X)
        pop_scale = fit_scale(X)
        X, y, groups = personalize(X, y, groups, n_enroll=args.baseline,
                                   scale=pop_scale,
                                   rest_label=0 if args.enroll_rest else None)
        print(f"  baselined against each subject's first {args.baseline} windows "
              f"({n_before} -> {len(X)} windows)")

    present = sorted(set(y))
    names = [STATES[i] for i in present]
    print(f"{args.dataset}: {len(X)} windows, {len(set(groups))} groups, "
          f"classes present: {names}")

    # Subject-wise cross-validation. Windows overlap by 45 s, so a random
    # split leaks near-duplicate windows across the boundary and reports a
    # number that will not survive contact with a new person.
    n_folds = min(args.folds, len(set(groups)))
    if n_folds >= 2:
        accs = []
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
            sc, fo = fit(X[tr], y[tr], args)
            accs.append((fo.predict(sc.transform(X[te])) == y[te]).mean())
        accs = np.array(accs)
        print(f"held-out subject accuracy: {accs.mean():.3f} "
              f"+/- {accs.std():.3f}  (folds: {np.round(accs, 3)})")

        # Report against the majority-class rate, which is the number that
        # tells you whether the model learned anything at all.
        base = max((y == c).mean() for c in present)
        print(f"majority-class baseline:   {base:.3f}")

        sc, fo = fit(X, y, args)
        pred = fo.predict(sc.transform(X))
        print(classification_report(y, pred, labels=present,
                                    target_names=names, digits=3,
                                    zero_division=0))
        print("(report above is in-sample; trust the held-out figure)")
    else:
        sc, fo = fit(X, y, args)
        pred = fo.predict(sc.transform(X))

    if args.amber_band:
        lo, hi = args.amber_band
        red_col = list(fo.classes_).index(2) if 2 in fo.classes_ else None
        if red_col is None:
            print("  amber-band: no RED class in this model, skipping")
            args.amber_band = None
        else:
            p_red = fo.predict_proba(sc.transform(X))[:, red_col]
            banded = band(p_red, lo, hi)
            n = len(banded)
            print(f"\namber band [{lo}, {hi}] applied to p(RED):")
            for i, name in enumerate(STATES):
                share = (banded == i).mean()
                print(f"  {name:<6} {share:6.1%} of windows", end="")
                if i == 1 and share > 0:
                    # What the band is actually catching, by true label.
                    true = y[banded == 1]
                    frac_red = (true == 2).mean()
                    print(f"   (of these, {frac_red:.1%} are truly RED)", end="")
                print()
            print(f"  windows: {n}")
            print("  NOTE: AMBER here is a confidence band, not a learned "
                  "class. It reports uncertainty, and is unvalidated against "
                  "graded physiological load.")

    if args.dataset == "synthetic":
        print("NOTE: trained on synthetic data. Accuracy above is not evidence.")

    model_id = hashlib.sha256(
        json.dumps({"trees": args.trees, "depth": args.depth, "seed": args.seed,
                    "data": args.dataset, "baseline": args.baseline,
                    "feats": FEATURE_NAMES}).encode()
    ).hexdigest()[:12]

    export_header(ROOT / "core" / "echo_model.h", sc, fo, model_id, pop_scale,
                  args.amber_band)
    fixture_pred = pred[:64]
    if args.amber_band:
        red_col = list(fo.classes_).index(2)
        p64 = fo.predict_proba(sc.transform(X[:64]))[:, red_col]
        fixture_pred = band(p64, args.amber_band[0], args.amber_band[1])
    export_testvectors(ROOT / "bench" / "testvectors.h", X[:64], fixture_pred)
    print(f"wrote core/echo_model.h  (model {model_id}, {args.trees} trees, "
          f"depth {args.depth}, data={args.dataset})")
    print("wrote bench/testvectors.h")


if __name__ == "__main__":
    main()
