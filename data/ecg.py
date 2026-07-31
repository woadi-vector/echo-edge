"""ECG -> RR intervals.

WESAD gives raw 700 Hz chest ECG; the Polar H10 gives RR intervals directly.
To train on one and infer on the other, the ECG has to be reduced to the same
representation. This is a Pan-Tompkins-style detector: bandpass to isolate the
QRS complex, differentiate, square, integrate, then pick peaks with a
physiological refractory period.

Clean lab ECG is the easy case. This is not a general-purpose detector and
would need work before being pointed at ambulatory recordings.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

REFRACTORY_S = 0.25   # no two beats closer than 250 ms
RR_MIN, RR_MAX = 250.0, 2500.0


def bandpass(x, fs, lo=5.0, hi=15.0, order=2):
    nyq = 0.5 * fs
    hi = min(hi, nyq * 0.95)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def rr_from_ecg(ecg, fs=700.0):
    """Raw ECG samples -> RR intervals in milliseconds."""
    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    if ecg.size < fs * 5:
        return np.array([])

    filtered = bandpass(ecg, fs)
    squared = np.diff(filtered, prepend=filtered[0]) ** 2

    win = max(int(0.150 * fs), 1)                      # 150 ms integration
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # Threshold on the recording's own distribution rather than a fixed value,
    # since amplitude varies with electrode placement and subject.
    thresh = np.percentile(integrated, 98) * 0.35
    peaks, _ = find_peaks(integrated,
                          height=thresh,
                          distance=int(REFRACTORY_S * fs))
    if peaks.size < 3:
        return np.array([])

    rr = np.diff(peaks) / fs * 1000.0
    return rr[(rr >= RR_MIN) & (rr <= RR_MAX)]


def quality(rr):
    """Rough confidence that detection worked. Sane recordings sit above 0.9.

    Counts the fraction of intervals within 30% of their predecessor —
    missed or doubled beats show up as large jumps.
    """
    if rr.size < 3:
        return 0.0
    ratio = rr[1:] / rr[:-1]
    return float(np.mean((ratio > 0.7) & (ratio < 1.3)))
