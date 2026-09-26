"""
Classical ECG feature extraction.

No machine-learning model is used here. Features are calculated only from
the supplied ECG signal and the sampling rate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import signal


def detect_r_peaks(
    ecg: Any,
    sampling_rate_hz: float,
    refractory_period_s: float = 0.25,
) -> np.ndarray:
    """
    Detect R-peak candidates with scipy.signal.find_peaks.

    Both ECG polarities are tested so an inverted lead can be handled.
    Prominence is used instead of an absolute amplitude threshold, and a
    250 ms refractory period prevents double-counting a QRS complex.
    """
    x = np.asarray(ecg, dtype=float).reshape(-1)
    fs = float(sampling_rate_hz)

    if x.size < 3 or not np.all(np.isfinite(x)) or fs <= 0:
        return np.array([], dtype=int)

    scale = float(np.std(x))
    if scale <= np.finfo(float).eps:
        return np.array([], dtype=int)

    distance = max(1, int(round(refractory_period_s * fs)))
    prominence = max(0.5 * scale, 0.25)

    candidates = []
    for polarity, candidate_signal in (("positive", x), ("negative", -x)):
        peaks, props = signal.find_peaks(
            candidate_signal,
            distance=distance,
            prominence=prominence,
        )
        prom = np.asarray(props.get("prominences", []), dtype=float)
        if len(peaks) == 0:
            candidates.append((polarity, peaks, prom, np.array([], dtype=int)))
            continue

        # Keep peaks with prominence close to the dominant QRS prominence.
        # This removes smaller T-wave/artifact peaks without using a patient-
        # specific amplitude assumption.
        strong_cutoff = max(0.70 * float(np.max(prom)), 0.25)
        strong = peaks[prom >= strong_cutoff]
        strong_prom = prom[prom >= strong_cutoff]
        candidates.append((polarity, peaks, prom, strong))

    viable = [c for c in candidates if len(c[3]) > 0]
    if not viable:
        return np.array([], dtype=int)

    # Select the polarity with the most consistently prominent peaks. If
    # tied, prefer fewer peaks to avoid counting secondary deflections.
    def score(c):
        _, _, prom, strong = c
        strong_prom = prom[np.isin(
            np.arange(len(prom)),
            np.flatnonzero(prom >= max(0.70 * np.max(prom), 0.25))
        )]
        return (len(strong), float(np.median(strong_prom)) if len(strong_prom) else 0.0, -len(c[1]))

    best = max(viable, key=score)
    return np.asarray(best[3], dtype=int)


def _rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def assess_signal_quality(
    original: np.ndarray,
    processed: np.ndarray,
    sampling_rate_hz: float,
) -> Dict[str, Any]:
    """
    Transparent non-clinical quality heuristic.

    It considers finite/flat samples and whether the processed signal has a
    stable sequence of detected R peaks. It is not a validated clinical
    signal-quality index.
    """
    if original.size < 3 or processed.size != original.size:
        return {"status": "Undetermined", "reason": "Insufficient signal data."}

    orig_std = float(np.std(original))
    if orig_std <= np.finfo(float).eps:
        return {"status": "Poor", "reason": "Very low signal variation."}

    flat_fraction = float(
        np.mean(
            np.isclose(
                np.diff(original),
                0.0,
                atol=max(orig_std * 1e-8, 1e-12),
            )
        )
    )

    peaks = detect_r_peaks(processed, sampling_rate_hz)
    if len(peaks) < 2:
        if flat_fraction < 0.02:
            return {
                "status": "Undetermined",
                "reason": "Fewer than two reliable R peaks were detected.",
                "flat_sample_fraction": _rounded(flat_fraction, 4),
            }
        return {"status": "Poor", "reason": "Signal contains substantial flat segments."}

    rr = np.diff(peaks) / float(sampling_rate_hz)
    rr_cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else np.inf

    if flat_fraction < 0.05 and rr_cv < 0.10:
        status = "Good"
    elif flat_fraction < 0.15 and rr_cv < 0.20:
        status = "Fair"
    else:
        status = "Poor"

    return {
        "status": status,
        "reason": "Heuristic based on flat-sample fraction and R-R interval consistency.",
        "flat_sample_fraction": _rounded(flat_fraction, 4),
        "rr_coefficient_of_variation": _rounded(rr_cv, 4),
    }


def extract_features(
    processed_signal: Any,
    sampling_rate_hz: float,
    original_signal: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract requested ECG features; unavailable values are returned as null."""
    x = np.asarray(processed_signal, dtype=float).reshape(-1)
    fs = float(sampling_rate_hz)

    if x.size == 0 or not np.all(np.isfinite(x)) or fs <= 0:
        raise ValueError("A finite processed ECG and valid sampling rate are required.")

    peaks = detect_r_peaks(x, fs)

    rr_intervals = np.diff(peaks) / fs if len(peaks) >= 2 else np.array([], dtype=float)
    mean_rr = float(np.mean(rr_intervals)) if rr_intervals.size else None
    heart_rate = (60.0 / mean_rr) if mean_rr and mean_rr > 0 else None

    original = np.asarray(original_signal, dtype=float).reshape(-1) if original_signal is not None else x
    quality = assess_signal_quality(original, x, fs)

    return {
        "mean_amplitude": _rounded(np.mean(x)),
        "standard_deviation": _rounded(np.std(x)),
        "minimum_amplitude": _rounded(np.min(x)),
        "maximum_amplitude": _rounded(np.max(x)),
        "signal_range": _rounded(np.ptp(x)),
        "rms": _rounded(np.sqrt(np.mean(np.square(x)))),
        "r_peak_count": int(len(peaks)),
        "r_peak_indices": peaks.tolist(),
        "rr_intervals_s": [_rounded(v, 4) for v in rr_intervals],
        "mean_rr_interval_s": _rounded(mean_rr, 4),
        "heart_rate_bpm": _rounded(heart_rate, 2),
        "signal_quality": quality,
        "feature_status": {
            "r_peaks": "calculated" if len(peaks) else "unreliable_or_not_detected",
            "rr_intervals": "calculated" if rr_intervals.size else "unavailable_less_than_two_r_peaks",
            "heart_rate": "calculated" if heart_rate is not None else "unavailable_less_than_two_r_peaks",
        },
    }
