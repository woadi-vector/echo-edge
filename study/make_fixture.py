"""Generate a schema-exact synthetic echo_*.csv so the analyzer can be exercised
before any real session exists. FABRICATED DATA — for pipeline testing only, it
is not evidence of anything. Mirrors the export header in docs/app.js.
"""
import argparse
import csv
import datetime as dt
import pathlib

import numpy as np

HEADER = ["timestamp", "elapsed_s", "participant", "note", "activity", "mark",
          "temp_f", "humidity_pct", "wbgt_f", "setting", "signal_quality",
          "state", "raw_state", "confidence", "p_green", "p_amber", "p_red",
          "mean_rr", "mean_hr", "sdnn", "rmssd", "pnn50", "rr_slope",
          "hr_cv", "coverage", "model"]

STATES = ["GREEN", "AMBER", "RED"]

# (note, activity, minutes, mean HR, readiness bias) — a plausible session arc.
STAGES = [
    ("rest",          "rest",   4, 58,  0.85),
    ("pre-training",  "rest",   3, 74,  0.55),
    ("post-training", "active", 5, 108, 0.20),
    ("recovery",      "rest",   5, 82,  0.45),
]


def probs(readiness, rng):
    """Turn a latent readiness (0..1) into GREEN/AMBER/RED vote shares."""
    p_red = np.clip(1 - readiness + rng.normal(0, 0.08), 0.01, 0.98)
    p_amber = np.clip(0.25 - abs(readiness - 0.5) * 0.3 + rng.normal(0, 0.05), 0.01, 0.6)
    p_green = max(0.01, 1 - p_red - p_amber)
    tot = p_green + p_amber + p_red
    return p_green / tot, p_amber / tot, p_red / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="P01")
    ap.add_argument("--out", default="echo_P01_fixture.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step", type=float, default=5.0, help="seconds/window")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = dt.datetime(2026, 10, 1, 14, 22, 0)
    elapsed = 0.0
    raw_hold, last_raw = 0, "GREEN"
    rows = []
    model = "fixture000000"

    for note, activity, minutes, hr, bias in STAGES:
        n = int(minutes * 60 / args.step)
        for k in range(n):
            readiness = float(np.clip(bias + rng.normal(0, 0.08), 0.02, 0.98))
            pg, pa, pr = probs(readiness, rng)
            raw = STATES[int(np.argmax([pg, pa, pr]))]
            # crude hysteresis: reported state lags raw by a few windows
            if raw == last_raw:
                raw_hold += 1
            else:
                raw_hold, last_raw = 0, raw
            state = raw if raw_hold >= 3 else (rows[-1][11] if rows else "GREEN")
            conf = max(pg, pa, pr)
            hr_k = hr + rng.normal(0, 4)
            mean_rr = 60000.0 / hr_k
            sdnn = max(4.0, 60 * readiness + rng.normal(0, 6))
            rmssd = max(3.0, 55 * readiness + rng.normal(0, 6))
            pnn50 = 0.0 if hr_k > 90 else max(0.0, 0.4 * readiness + rng.normal(0, 0.05))
            quality = round(float(np.clip(rng.normal(0.95, 0.03), 0.6, 1.0)), 3)
            mark = "stage_change" if k == 0 and note == "post-training" else ""
            wbgt = 82 + rng.normal(0, 1)
            rows.append([
                (t0 + dt.timedelta(seconds=elapsed)).isoformat(timespec="seconds"),
                f"{elapsed:.1f}", args.pid, note, activity, mark,
                f"{88.0:.0f}", f"{55.0:.0f}", f"{wbgt:.1f}", "field",
                f"{quality:.3f}", state, raw, f"{conf:.4f}",
                f"{pg:.4f}", f"{pa:.4f}", f"{pr:.4f}",
                f"{mean_rr:.4f}", f"{hr_k:.4f}", f"{sdnn:.4f}", f"{rmssd:.4f}",
                f"{pnn50:.4f}", f"{rng.normal(0,1.5):.4f}",
                f"{sdnn/mean_rr:.4f}", f"{min(1.0, 0.95+rng.normal(0,0.02)):.4f}",
                model,
            ])
            elapsed += args.step

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} windows, {len(HEADER)} columns")


if __name__ == "__main__":
    main()
