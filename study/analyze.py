"""Echo Edge — study session analyzer (Track A: early-warning behaviour).

Reads the CSV files the browser client exports (echo_P01_<stamp>.csv) and turns
a folder of them into a readable report: per-condition state breakdown, a
continuous readiness trace, signal-quality and hysteresis diagnostics, and the
early-warning metrics from the improvement plan — lead/lag around load onsets,
alert rate per hour, a rest-vs-load AUC, and a threshold sweep (the Rothman
Figure-2 tradeoff).

This runs on exported logs; it does NOT re-run the model. The client already
logged the classified state and the class vote shares, so the continuous
readiness previewed here is computed from those, not re-inferred.

    python3 study/analyze.py <file-or-dir> [options]

WHAT IS AND IS NOT A CLAIM
    In device-shakedown data there are no adverse events, so "lead time" here
    measures how the score BEHAVES around protocol-stage transitions, not how
    early it predicts injury. It is a behaviour diagnostic until the approved
    study attaches real outcomes. The code says so wherever it prints such a
    number.

Dependencies: numpy and the standard library only. No pandas, matching the repo.
"""
import argparse
import csv
import glob
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

# --- The export contract, mirrored from docs/app.js exportCSV(). -------------
# Header order is fixed there; we read by name, not position, so an added
# column does not silently shift everything, but we assert the ones we need.
FEATURES = ["mean_rr", "mean_hr", "sdnn", "rmssd",
            "pnn50", "rr_slope", "hr_cv", "coverage"]
REQUIRED = ["elapsed_s", "participant", "note", "activity", "mark",
            "signal_quality", "state", "raw_state",
            "p_green", "p_amber", "p_red"]

STATES = ["GREEN", "AMBER", "RED"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# Default condition semantics for the shakedown protocol vocabulary. LOW is the
# resting/parasympathetic side, HIGH is the loaded side. Anything unrecognised
# is reported but left out of the rest-vs-load contrast.
LOW_CONDITIONS = {"rest", "recovery", "baseline", "seated", "quiet"}
HIGH_CONDITIONS = {"pre-training", "post-training", "training", "load",
                   "stress", "post", "exertion"}


def readiness_from(p_green, p_amber, p_red):
    """Continuous 0-100 readiness from the logged vote shares.

    readiness = 100 * (p_green + 0.5 * p_amber). This is the Track-B scalar,
    computed here from what the client already logged so the study gets a
    continuous, trendable value with no firmware change. RED pulls it to 0,
    GREEN to 100, the uncertain AMBER band to the middle.
    """
    tot = p_green + p_amber + p_red
    if tot <= 0:
        return float("nan")
    g, a = p_green / tot, p_amber / tot
    return 100.0 * (g + 0.5 * a)


def _f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def parse_file(path):
    """One echo_*.csv -> list of typed row dicts, in file order."""
    rows = []
    with open(path, "r", newline="", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return rows
        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{pathlib.Path(path).name}: missing columns {missing}")
        for raw in reader:
            state = (raw.get("state") or "").strip().upper()
            rawst = (raw.get("raw_state") or "").strip().upper()
            pg, pa, pr = _f(raw, "p_green"), _f(raw, "p_amber"), _f(raw, "p_red")
            rows.append({
                "file": pathlib.Path(path).name,
                "pid": (raw.get("participant") or "").strip(),
                "elapsed": _f(raw, "elapsed_s"),
                "note": (raw.get("note") or "").strip().lower(),
                "activity": (raw.get("activity") or "").strip().lower(),
                "mark": (raw.get("mark") or "").strip(),
                "quality": _f(raw, "signal_quality"),
                "state": state,
                "raw_state": rawst,
                "state_i": STATE_IDX.get(state, -1),
                "raw_i": STATE_IDX.get(rawst, -1),
                "p_green": pg, "p_amber": pa, "p_red": pr,
                "readiness": readiness_from(pg, pa, pr),
                "mean_hr": _f(raw, "mean_hr"),
                "rmssd": _f(raw, "rmssd"),
                "model": (raw.get("model") or "").strip(),
            })
    return rows


def load(paths):
    """Expand files/dirs to a flat, file-grouped list of sessions."""
    files = []
    for p in paths:
        pp = pathlib.Path(p)
        if pp.is_dir():
            files += sorted(glob.glob(str(pp / "echo_*.csv")))
        else:
            files.append(str(pp))
    sessions = []
    for f in files:
        try:
            rows = parse_file(f)
        except ValueError as e:
            print(f"  skipped: {e}", file=sys.stderr)
            continue
        if rows:
            sessions.append((pathlib.Path(f).name, rows))
    return sessions


# --- Metrics -----------------------------------------------------------------

def auc(pos, neg):
    """AUC via the Mann-Whitney U statistic. pos/neg are score arrays where
    higher means more likely to be the positive (load) class."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[:pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return u / (pos.size * neg.size)


def sustained_alerts(states, hold):
    """Indices where a RED run of >= hold windows first becomes sustained.

    Mirrors the reporting-layer hysteresis idea: a single-window RED is noise;
    a held RED is a signal. Returns the index at which each sustained run
    crosses the hold threshold."""
    onsets, run = [], 0
    for i, s in enumerate(states):
        if s == "RED":
            run += 1
            if run == hold:
                onsets.append(i - hold + 1)
        else:
            run = 0
    return onsets


def dwell_hours(rows):
    """Elapsed span of a row group, in hours, from elapsed_s."""
    e = [r["elapsed"] for r in rows if not math.isnan(r["elapsed"])]
    if len(e) < 2:
        return 0.0
    return max(0.0, (max(e) - min(e)) / 3600.0)


def state_mix(rows):
    n = len(rows)
    if n == 0:
        return (0, 0, 0)
    c = [0, 0, 0]
    for r in rows:
        if r["state_i"] >= 0:
            c[r["state_i"]] += 1
    return tuple(100.0 * x / n for x in c)


def transitions(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def cond_class(note):
    if note in LOW_CONDITIONS:
        return "low"
    if note in HIGH_CONDITIONS:
        return "high"
    return "other"


# --- Reporting ---------------------------------------------------------------

def report_session(name, rows, args):
    out = []
    pid = rows[0]["pid"] or "?"
    models = sorted({r["model"] for r in rows if r["model"]})
    dur_min = dwell_hours(rows) * 60.0
    out.append(f"\n=== {name}   participant {pid}   {len(rows)} windows"
               f"   {dur_min:.1f} min ===")
    if len(models) > 1:
        out.append(f"  ! multiple model IDs in one file: {models} "
                   f"(sessions should not span a retrain)")

    # Signal quality — the protocol calls sustained <0.9 poor contact.
    q = np.array([r["quality"] for r in rows], float)
    q = q[~np.isnan(q)]
    if q.size:
        accepted = float((q >= args.quality).mean())
        flag = "  <-- poor contact" if accepted < 0.9 else ""
        out.append(f"  signal quality: mean {q.mean():.3f}, "
                   f"{accepted*100:.0f}% >= {args.quality}{flag}")

    # Exertion gate: 'active' windows are where the UI withholds state; their
    # HRV is the known-unreliable regime, so we surface how much of the session
    # sat there.
    active = sum(1 for r in rows if r["activity"] == "active")
    if active:
        out.append(f"  exertion-gated (activity=active): {active} "
                   f"windows ({100.0*active/len(rows):.0f}%) — HRV unreliable here")

    # Hysteresis effect: transitions in the reported state vs the raw state.
    raw_seq = [r["raw_state"] for r in rows if r["raw_i"] >= 0]
    st_seq = [r["state"] for r in rows if r["state_i"] >= 0]
    if raw_seq and st_seq:
        out.append(f"  state transitions: {transitions(st_seq)} reported "
                   f"vs {transitions(raw_seq)} raw "
                   f"(hysteresis absorbed {transitions(raw_seq)-transitions(st_seq)})")

    # Per-condition breakdown, grouped by the chosen field.
    by = defaultdict(list)
    for r in rows:
        by[r[args.by] or "(blank)"].append(r)
    out.append(f"  by {args.by}:")
    out.append(f"    {'condition':<16}{'win':>5}{'min':>7}"
               f"{'HR':>7}{'readiness':>11}   G/A/R %")
    for cond, rs in by.items():
        hr = np.nanmean([r["mean_hr"] for r in rs])
        rd = np.nanmean([r["readiness"] for r in rs])
        g, a, rr = state_mix(rs)
        out.append(f"    {cond:<16}{len(rs):>5}{dwell_hours(rs)*60:>7.1f}"
                   f"{hr:>7.0f}{rd:>11.1f}   {g:.0f}/{a:.0f}/{rr:.0f}")

    # Alert rate per hour, RED (sustained) and RED-or-AMBER.
    for label, pred in (("RED", lambda r: r["state"] == "RED"),
                        ("RED+AMBER", lambda r: r["state"] in ("RED", "AMBER"))):
        alerts = sum(1 for r in rows if pred(r))
        h = dwell_hours(rows)
        rate = alerts / h if h > 0 else float("nan")
        out.append(f"  {label} windows: {alerts} ({rate:.1f}/hour)")

    # Operator marks — the in-app event mechanism. Report the score around each.
    marks = [(i, r) for i, r in enumerate(rows) if r["mark"]]
    if marks:
        out.append("  operator marks:")
        for i, r in marks:
            lo = max(0, i - 3)
            hi = min(len(rows), i + 4)
            near = np.nanmean([rows[j]["readiness"] for j in range(lo, hi)])
            out.append(f"    t+{r['elapsed']:.0f}s  mark='{r['mark']}'  "
                       f"state={r['state']}  readiness~{near:.0f}")

    # Rest-vs-load discrimination + lead/lag. Behaviour diagnostic on shakedown
    # data (no real outcomes), stated as such.
    low = [r["readiness"] for r in rows if cond_class(r["note"]) == "low"]
    high = [r["readiness"] for r in rows if cond_class(r["note"]) == "high"]
    if low and high:
        # Positive class = load; readiness is inverse risk, so score on -readiness.
        a = auc([-x for x in high], [-x for x in low])
        out.append(f"  rest-vs-load AUC (readiness): {a:.3f}   "
                   f"[behaviour, not outcome — shakedown has no events]")
        out.append(f"    load readiness {np.nanmean(high):.1f} "
                   f"vs rest {np.nanmean(low):.1f}")

        # Lead/lag: first sustained RED relative to first load-condition onset.
        note_class = [cond_class(r["note"]) for r in rows]
        onset = next((i for i, c in enumerate(note_class) if c == "high"), None)
        if onset is not None:
            al = sustained_alerts([r["state"] for r in rows], args.hold)
            after = [i for i in al if i >= onset]
            if after:
                dt = rows[after[0]]["elapsed"] - rows[onset]["elapsed"]
                sign = "after" if dt >= 0 else "before"
                out.append(f"    first sustained RED {abs(dt):.0f}s {sign} load onset")
            else:
                out.append("    no sustained RED after load onset")
    return "\n".join(out), by


def report_sweep(all_rows, args):
    """Rothman Figure-2 analog: sweep the p_red alert threshold and show the
    alert-volume vs load-recall tradeoff across the whole corpus."""
    p_red = np.array([r["p_red"] for r in all_rows], float)
    is_load = np.array([cond_class(r["note"]) == "high" for r in all_rows])
    is_rest = np.array([cond_class(r["note"]) == "low" for r in all_rows])
    if not is_load.any() or not is_rest.any():
        return "\n(threshold sweep skipped: need both rest and load windows)"
    lines = ["\n=== threshold sweep (p_red)  —  alert volume vs load recall ===",
             f"    {'thr':>5}{'alert%':>9}{'load recall':>13}{'rest FPR':>10}"]
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fire = p_red >= thr
        vol = 100.0 * fire.mean()
        recall = 100.0 * (fire & is_load).sum() / max(is_load.sum(), 1)
        fpr = 100.0 * (fire & is_rest).sum() / max(is_rest.sum(), 1)
        lines.append(f"    {thr:>5.1f}{vol:>9.1f}{recall:>13.1f}{fpr:>10.1f}")
    lines.append("    (pick the threshold where load recall stays high and "
                 "rest FPR is tolerable — the alarm-fatigue knee)")
    return "\n".join(lines)


def write_windows_csv(path, sessions):
    """Tidy per-window CSV with the computed readiness added, for plotting."""
    cols = ["file", "pid", "elapsed", "note", "activity", "mark", "quality",
            "raw_state", "state", "p_green", "p_amber", "p_red", "readiness",
            "mean_hr", "rmssd", "model"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for _, rows in sessions:
            for r in rows:
                w.writerow([r[c] for c in cols])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="echo_*.csv files or a folder")
    ap.add_argument("--by", default="note", choices=["note", "activity"],
                    help="condition field to group by (default: note)")
    ap.add_argument("--quality", type=float, default=0.85,
                    help="signal-quality floor for the accepted-fraction figure")
    ap.add_argument("--hold", type=int, default=3,
                    help="consecutive RED windows before an alert is 'sustained'")
    ap.add_argument("--out", default=None,
                    help="write a tidy per-window CSV (with readiness) here")
    args = ap.parse_args()

    sessions = load(args.paths)
    if not sessions:
        ap.error("no readable echo_*.csv sessions found")

    print(f"loaded {len(sessions)} session file(s)")
    all_rows = []
    per_pid = defaultdict(list)
    for name, rows in sessions:
        text, _ = report_session(name, rows, args)
        print(text)
        all_rows += rows
        per_pid[rows[0]["pid"]] += rows

    print(report_sweep(all_rows, args))

    print(f"\n=== corpus: {len(all_rows)} windows, {len(per_pid)} participant(s) ===")
    for pid, rows in sorted(per_pid.items()):
        g, a, r = state_mix(rows)
        rd = np.nanmean([x["readiness"] for x in rows])
        print(f"  {pid or '?':<8} {len(rows):>5} win   "
              f"readiness {rd:5.1f}   G/A/R {g:.0f}/{a:.0f}/{r:.0f}")

    if args.out:
        write_windows_csv(args.out, sessions)
        print(f"\nwrote per-window CSV -> {args.out}")


if __name__ == "__main__":
    main()
