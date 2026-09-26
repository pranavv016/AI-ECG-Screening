"""
ECG preprocessing utilities for the AI-Assisted ECG Screening System.

This module intentionally uses classical digital signal processing only.
It does not perform disease classification or machine-learning inference.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
from scipy import signal


@dataclass
class PreprocessingConfig:
    """Configurable ECG preprocessing parameters."""

    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    filter_order: int = 4
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0
    apply_notch: bool = True
    normalize: bool = True


def validate_ecg_input(
    samples: Any,
    sampling_rate_hz: Optional[float],
    min_samples: int = 50,
) -> tuple[np.ndarray, float]:
    """Validate and convert ECG samples and sampling rate."""
    if samples is None:
        raise ValueError("ECG samples are required.")

    x = np.asarray(samples, dtype=float).reshape(-1)

    if x.size < min_samples:
        raise ValueError(f"At least {min_samples} ECG samples are required.")
    if not np.all(np.isfinite(x)):
        raise ValueError("ECG samples must contain only finite numeric values.")
    if np.ptp(x) == 0:
        raise ValueError("ECG signal is constant and cannot be processed.")

    if sampling_rate_hz is None or not np.isfinite(sampling_rate_hz):
        raise ValueError(
            "A valid sampling rate is required for preprocessing. "
            "Provide a time column with sufficiently uniform sampling."
        )

    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    return x, fs




def preprocess_ml_beat(samples: Any, expected_samples: int = 187) -> np.ndarray:
    """Apply the exact per-beat normalization used by the training pipeline.

    This intentionally does not apply the display/analysis band-pass filter:
    the trained classifier was built from per-beat standardized 187-sample
    waveforms. Keeping this transformation centralized prevents training and
    inference preprocessing drift.
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size != expected_samples:
        raise ValueError(
            f"ML inference expects exactly {expected_samples} ECG samples."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("ECG samples must contain only finite numeric values.")
    if np.ptp(x) == 0:
        raise ValueError("ECG signal is constant and cannot be classified.")

    std = float(np.std(x))
    if std <= np.finfo(float).eps:
        raise ValueError("ECG signal has near-zero variance and cannot be classified.")

    return (x - float(np.mean(x))) / (std + 1e-8)


def _safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering with a useful error for signals that are too short."""
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError as exc:
        # SciPy's default padding requires a minimum signal length.
        # Gustafsson is not available for SOS in all SciPy versions, so
        # fall back to a single zero-phase-compatible method only when safe.
        padlen = 3 * (2 * len(sos) + 1)
        if len(x) <= padlen:
            raise ValueError(
                f"ECG signal is too short for the selected filter settings "
                f"({len(x)} samples; need more than {padlen})."
            ) from exc
        raise


def preprocess_ecg(
    samples: Any,
    sampling_rate_hz: Optional[float],
    config: Optional[PreprocessingConfig] = None,
) -> Dict[str, Any]:
    """
    Preprocess an ECG without machine learning.

    Steps:
      1. Validate finite numeric samples and sampling rate.
      2. Zero-phase Butterworth band-pass filtering to reduce baseline wander
         and high-frequency noise while preserving waveform timing.
      3. Optional zero-phase notch filter for mains interference.
      4. Optional z-score normalization.

    The original samples are returned unchanged.
    """
    cfg = config or PreprocessingConfig()
    x, fs = validate_ecg_input(samples, sampling_rate_hz)

    if cfg.lowcut_hz <= 0:
        raise ValueError("lowcut_hz must be greater than zero.")
    if cfg.highcut_hz <= cfg.lowcut_hz:
        raise ValueError("highcut_hz must be greater than lowcut_hz.")
    nyquist = fs / 2.0
    if cfg.highcut_hz >= nyquist:
        raise ValueError(
            f"highcut_hz ({cfg.highcut_hz:g} Hz) must be below Nyquist "
            f"frequency ({nyquist:g} Hz)."
        )
    if cfg.filter_order < 1:
        raise ValueError("filter_order must be at least 1.")

    # A single band-pass stage handles both baseline-wander reduction
    # (< lowcut) and high-frequency noise (> highcut).
    sos = signal.butter(
        cfg.filter_order,
        [cfg.lowcut_hz, cfg.highcut_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    processed = _safe_sosfiltfilt(sos, x)

    notch_applied = False
    if cfg.apply_notch and cfg.notch_hz is not None and 0 < cfg.notch_hz < nyquist:
        b_notch, a_notch = signal.iirnotch(cfg.notch_hz, cfg.notch_q, fs=fs)
        processed = signal.filtfilt(b_notch, a_notch, processed)
        notch_applied = True

    mean_before_norm = float(np.mean(processed))
    std_before_norm = float(np.std(processed, ddof=0))

    if cfg.normalize:
        if std_before_norm <= np.finfo(float).eps:
            raise ValueError("Processed ECG has near-zero variance; normalization failed.")
        processed = (processed - mean_before_norm) / std_before_norm

    # Guard against numerical problems introduced by filtering.
    if not np.all(np.isfinite(processed)):
        raise ValueError("Preprocessing produced non-finite values.")

    info = {
        "method": "zero-phase Butterworth band-pass + optional notch + optional z-score normalization",
        "baseline_wander_reduction": f"High-pass cutoff {cfg.lowcut_hz:g} Hz",
        "band_pass": {
            "low_hz": cfg.lowcut_hz,
            "high_hz": cfg.highcut_hz,
            "order": cfg.filter_order,
            "phase": "zero-phase (forward/backward)"
        },
        "notch": {
            "enabled": bool(cfg.apply_notch),
            "frequency_hz": cfg.notch_hz,
            "q": cfg.notch_q,
            "applied": notch_applied,
        },
        "normalization": {
            "enabled": bool(cfg.normalize),
            "method": "z-score" if cfg.normalize else "none",
        },
        "sampling_rate_hz": fs,
        "input_samples": int(x.size),
    }

    return {
        "original_signal": x.tolist(),
        "processed_signal": processed.tolist(),
        "sampling_rate_hz": fs,
        "preprocessing_info": info,
    }
