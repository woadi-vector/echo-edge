"""Schumann front-end validation.

Validates the R-peak detector in data/ecg.py against Schumann et al. 2025's
published reference values, before any of the four processing-convention arms
are run.

Deliberately does NOT use echo_features. RMSSD is computed directly in numpy
from the detected RR series, so the gap-spanning successive-difference defect
in echo_features cannot contaminate the check.

Usage
-----
  # step 1: one file
  python schumann_check.py --file path/to/01_1.txt

  # step 2: one subject, both sessions
  python schumann_check.py --file path/to/01_1.txt path/to/01_2.txt

  # step 3: the full set
  python schumann_check.py --dir path/to/schumann --batch out.csv

Reference (Schumann et al. 2025, full 15 min, Pan-Tompkins + adaptive filter):
  RMSSD ICC 0.75, 95% CI [0.63, 0.84], CV 29.6%
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

REF_ICC = 0.75
REF_ICC_CI = (0.63, 0.84)
REF_CV = 29.6

RR_PLAUSIBLE = (300.0, 2000.0)   # the memo's sanity window, not a filter
REFRACTORY_S = 0.25


def load_signal(path, col=2):
    """Read a whitespace or comma delimited text file, return one column.

    col is 1-indexed to match how the channel was described.
    """
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue          # header or comment
            rows.append(vals)

    if not rows:
        raise ValueError(f"{path}: no numeric rows found")

    width = max(len(r) for r in rows)
    if col > width:
        raise ValueError(f"{path}: asked for column {col}, file has {width}")

    return np.array([r[col - 1] for r in rows if len(r) >= col], dtype=np.float64)


def detect_peaks(ecg, fs):
    """Pan-Tompkins style detection, mirroring data/ecg.py.

    Returns peak sample indices. No interval filtering is applied here so the
    caller can see the unfiltered series.
    """
    nyq = 0.5 * fs
    hi = min(15.0, nyq * 0.95)
    b, a = butter(2, [5.0 / nyq, hi / nyq], btype="band")
    filtered = filtfilt(b, a, ecg)

    squared = np.diff(filtered, prepend=filtered[0]) ** 2
    win = max(int(0.150 * fs), 1)
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    thresh = np.percentile(integrated, 98) * 0.35
    peaks, _ = find_peaks(integrated, height=thresh,
                          distance=int(REFRACTORY_S * fs))
    return peaks


def rmssd(rr):
    """Root mean square of successive differences, in ms.

    Computed on the series as given. If the caller has removed intervals, the
    difference spanning the removal is spurious, which is precisely why this
    check runs on the unfiltered series.
    """
    if rr.size < 2:
        return float("nan")
    d = np.diff(rr)
    return float(np.sqrt(np.mean(d ** 2)))


def analyse(path, fs, col):
    ecg = load_signal(path, col)
    duration_s = ecg.size / fs
    peaks = detect_peaks(ecg, fs)
    rr = np.diff(peaks) / fs * 1000.0

    if rr.size < 2:
        return {"file": path.name, "error": "fewer than 2 beats detected"}

    lo, hi = RR_PLAUSIBLE
    out_of_range = int(((rr < lo) | (rr > hi)).sum())

    # Successive-ratio check: a dropped beat shows up as a near-doubling.
    ratio = rr[1:] / rr[:-1]
    jumpy = int(((ratio < 0.7) | (ratio > 1.3)).sum())

    return {
        "file": path.name,
        "duration_min": duration_s / 60.0,
        "beats": int(peaks.size),
        "mean_rr_ms": float(rr.mean()),
        "mean_hr_bpm": 60000.0 / float(rr.mean()),
        "rr_min_ms": float(rr.min()),
        "rr_max_ms": float(rr.max()),
        "out_of_range": out_of_range,
        "jumpy_pairs": jumpy,
        "rmssd_ms": rmssd(rr),
        "sdnn_ms": float(rr.std()),
    }


def show(r):
    if "error" in r:
        print(f"{r['file']}: ERROR {r['error']}")
        return
    print(f"\n{r['file']}")
    print(f"  duration        {r['duration_min']:8.2f} min")
    print(f"  beats detected  {r['beats']:8d}      (expect ~800-1100 over 15 min at rest)")
    print(f"  mean RR         {r['mean_rr_ms']:8.1f} ms")
    print(f"  mean HR         {r['mean_hr_bpm']:8.1f} bpm")
    print(f"  RR range        {r['rr_min_ms']:8.1f} to {r['rr_max_ms']:.1f} ms")
    print(f"  outside 300-2000{r['out_of_range']:8d}      (nonzero means detection misses)")
    print(f"  jumpy pairs     {r['jumpy_pairs']:8d}      (>30% beat-to-beat change)")
    print(f"  RMSSD           {r['rmssd_ms']:8.2f} ms")
    print(f"  SDNN            {r['sdnn_ms']:8.2f} ms")


def icc_2_1_absolute(pairs):
    """ICC(2,1), two-way random, absolute agreement, single measures.

    pairs: array of shape (n_subjects, 2).
    """
    x = np.asarray(pairs, dtype=np.float64)
    n, k = x.shape
    grand = x.mean()

    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom)


def cv_percent(pairs):
    """Within-subject coefficient of variation, as a percentage."""
    x = np.asarray(pairs, dtype=np.float64)
    sd_within = np.sqrt(((x[:, 0] - x[:, 1]) ** 2).mean() / 2.0)
    return float(100.0 * sd_within / x.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs="*", type=pathlib.Path, default=[])
    ap.add_argument("--dir", type=pathlib.Path)
    ap.add_argument("--batch", type=pathlib.Path,
                    help="write per-file results here and compute ICC/CV")
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--col", type=int, default=2, help="1-indexed ECG column")
    args = ap.parse_args()

    paths = list(args.file)
    if args.dir:
        paths += sorted(args.dir.glob("*_[12].txt"))
    if not paths:
        ap.error("give --file or --dir")

    results = []
    for p in paths:
        try:
            r = analyse(p, args.fs, args.col)
        except Exception as exc:
            r = {"file": p.name, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        if len(paths) <= 4:
            show(r)
        elif "error" in r:
            print(f"  {r['file']}: ERROR {r['error']}")
        else:
            print(f"  {r['file']}: {r['beats']} beats, "
                  f"RMSSD {r['rmssd_ms']:.2f} ms, "
                  f"{r['out_of_range']} out of range")

    if not args.batch:
        return

    good = {r["file"]: r for r in results if "error" not in r}
    subjects = {}
    for name, r in good.items():
        stem = name.rsplit(".", 1)[0]
        if "_" not in stem:
            continue
        sid, sess = stem.rsplit("_", 1)
        subjects.setdefault(sid, {})[sess] = r["rmssd_ms"]

    complete = {s: v for s, v in subjects.items() if "1" in v and "2" in v}
    incomplete = sorted(set(subjects) - set(complete))
    if incomplete:
        print(f"\nincomplete pairs excluded: {incomplete}")

    with open(args.batch, "w") as fh:
        fh.write("file,beats,mean_hr_bpm,rmssd_ms,sdnn_ms,out_of_range,jumpy_pairs\n")
        for r in results:
            if "error" in r:
                fh.write(f"{r['file']},,,,,,\n")
                continue
            fh.write(f"{r['file']},{r['beats']},{r['mean_hr_bpm']:.2f},"
                     f"{r['rmssd_ms']:.4f},{r['sdnn_ms']:.4f},"
                     f"{r['out_of_range']},{r['jumpy_pairs']}\n")
    print(f"\nwrote {args.batch}")

    if len(complete) < 3:
        print("not enough complete pairs for ICC")
        return

    sids = sorted(complete)
    pairs = np.array([[complete[s]["1"], complete[s]["2"]] for s in sids])

    icc = icc_2_1_absolute(pairs)
    cv = cv_percent(pairs)

    print(f"\ncomplete pairs      {len(sids)}")
    print(f"RMSSD ICC(2,1)      {icc:.3f}     reference {REF_ICC} "
          f"CI [{REF_ICC_CI[0]}, {REF_ICC_CI[1]}]")
    print(f"within-subject CV   {cv:.1f}%     reference {REF_CV}%")

    inside = REF_ICC_CI[0] <= icc <= REF_ICC_CI[1]
    print(f"\nICC {'falls inside' if inside else 'falls OUTSIDE'} the published CI")

    # Schumann named 51_2 as the high-ectopy recording; it should stand out.
    flagged = sorted(good.values(), key=lambda r: -r["jumpy_pairs"])[:5]
    print("\nmost artifact-laden recordings by jumpy pairs:")
    for r in flagged:
        print(f"  {r['file']:12s} {r['jumpy_pairs']:5d} jumpy, "
              f"{r['out_of_range']:4d} out of range, RMSSD {r['rmssd_ms']:.2f}")


if __name__ == "__main__":
    main()
