"""Compare the C feature extractor against the Python twin.

Builds nothing. Expects bench-parity to exist; `make parity` builds it first.

The series below are chosen to exercise the paths where the two sides have
historically drifted: gap-spanning successive differences, the pnn50
denominator, the relative jump filter, the rejection-run cap, and window
eviction. A random battery is added so drift in untested corners still shows.

Tolerance is relative, because the C computes in float32 and the Python in
float64. A disagreement larger than TOL is a logic difference, not rounding.
"""
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from features import FEATURE_NAMES, WINDOW_MS, extract  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / "bench-parity"
TOL = 2e-4          # relative; float32 carries about 7 significant digits


def battery():
    """(label, rr series). Deterministic."""
    rng = np.random.default_rng(20261009)
    out = []

    out.append(("constant, no artifact", np.full(41, 1000.0)))
    out.append(("step across rejected beat",
                np.r_[np.full(20, 1000.0), 3000.0, np.full(20, 1200.0)]))
    out.append(("same step, no rejection",
                np.r_[np.full(20, 1000.0), np.full(20, 1200.0)]))
    out.append(("alternating + rejection",
                np.r_[np.tile([1000.0, 1100.0], 15), 3000.0,
                      np.tile([1000.0, 1100.0], 15)]))
    out.append(("30 percent step, run cap",
                np.r_[np.full(20, 1000.0), np.full(21, 1300.0)]))
    out.append(("below the absolute floor",
                np.r_[np.full(20, 1000.0), 100.0, np.full(20, 1000.0)]))
    out.append(("above the absolute ceiling",
                np.r_[np.full(20, 1000.0), 4000.0, np.full(20, 1000.0)]))
    out.append(("exactly MIN_BEATS", np.full(20, 900.0)))
    out.append(("one short of MIN_BEATS", np.full(19, 900.0)))

    # Longer than the window, so eviction runs.
    out.append(("120 s, eviction", 1000.0 + rng.normal(0, 30, 120)))
    out.append(("300 s, heavy eviction", 850.0 + rng.normal(0, 40, 350)))

    # Posture-style step and a dropout-and-reacquire.
    out.append(("supine to standing",
                np.r_[60000.0 / 60 + rng.normal(0, 50, 60),
                      60000.0 / 95 + rng.normal(0, 25, 60)]))
    out.append(("dropout, reacquire fast",
                np.r_[60000.0 / 70 + rng.normal(0, 45, 40),
                      60000.0 / 150 + rng.normal(0, 10, 60)]))

    # Random battery across plausible rates and variabilities.
    for k in range(40):
        hr = rng.uniform(45, 150)
        sd = rng.uniform(3, 80)
        n = int(rng.integers(25, 200))
        rr = 60000.0 / hr + rng.normal(0, sd, n)
        if rng.random() < 0.4:                    # sprinkle artifacts
            for _ in range(int(rng.integers(1, 5))):
                rr[rng.integers(0, n)] *= rng.choice([0.4, 2.1, 0.2, 3.0])
        out.append((f"random {k:02d}: {hr:.0f} bpm, SD {sd:.0f}", rr))

    return out


def main():
    if not BIN.exists():
        print(f"missing {BIN}; run `make parity`", file=sys.stderr)
        return 2

    cases = battery()
    payload = "\n".join(",".join(f"{v:.10g}" for v in rr) for _, rr in cases)
    res = subprocess.run([str(BIN), str(WINDOW_MS)], input=payload,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        return 2
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    if len(lines) != len(cases):
        print(f"expected {len(cases)} output lines, got {len(lines)}",
              file=sys.stderr)
        return 2

    worst = 0.0
    worst_where = ""
    failures = []
    for (label, rr), line in zip(cases, lines):
        py = extract(rr)
        if line == "UNDERFILLED":
            if py is not None:
                failures.append(f"{label}: C underfilled, Python returned a vector")
            continue
        if py is None:
            failures.append(f"{label}: Python underfilled, C returned a vector")
            continue
        c = np.array([float(v) for v in line.split(",")])
        denom = np.where(np.abs(py) > 1e-9, np.abs(py), 1.0)
        rel = np.abs(c - py) / denom
        j = int(np.argmax(rel))
        if rel[j] > worst:
            worst, worst_where = rel[j], f"{label} / {FEATURE_NAMES[j]}"
        bad = np.where(rel > TOL)[0]
        for b in bad:
            failures.append(
                f"{label}: {FEATURE_NAMES[b]} C={c[b]:.6g} Python={py[b]:.6g} "
                f"rel={rel[b]:.2e}")

    print(f"parity: {len(cases)} series, {len(FEATURE_NAMES)} features each")
    print(f"largest relative difference: {worst:.2e} ({worst_where})")
    if failures:
        print(f"\nFAIL: {len(failures)} disagreement(s) above {TOL:.0e}")
        for f in failures[:25]:
            print(f"  {f}")
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more")
        return 1
    print(f"PASS: every feature agrees within {TOL:.0e} relative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
