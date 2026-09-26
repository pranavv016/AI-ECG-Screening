from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .preprocessing import PreprocessingConfig, preprocess_ecg, validate_ecg_input
    from .features import extract_features
    from .classifier import ECGClassifier, MODEL_PATH, SCALER_PATH
except ImportError:
    from preprocessing import PreprocessingConfig, preprocess_ecg, validate_ecg_input
    from features import extract_features
    from classifier import ECGClassifier, MODEL_PATH, SCALER_PATH


BASE = Path(__file__).resolve().parents[1]
FRONTEND = BASE / "frontend"

app = FastAPI(
    title="AI-Assisted ECG Screening System",
    version="Part 6",
    description="FastAPI communication layer for the Python ECG preprocessing and ML pipeline.",
)

# Local-development CORS. Restrict these origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ECGRequest(BaseModel):
    samples: List[float] = Field(..., min_length=3)
    time: Optional[List[float]] = None
    sampling_rate_hz: Optional[float] = None
    filename: Optional[str] = None


class ProcessingConfigRequest(BaseModel):
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    filter_order: int = 4
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0
    apply_notch: bool = True
    normalize: bool = True


class AnalyzeRequest(ECGRequest):
    config: Optional[ProcessingConfigRequest] = None


def infer_sampling_rate(time: Optional[List[float]]) -> Optional[float]:
    if time is None or len(time) < 2:
        return None

    t = np.asarray(time, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        return None

    intervals = np.diff(t)
    mean_dt = float(np.mean(intervals))
    if mean_dt <= 0:
        return None

    relative_deviation = float(np.max(np.abs(intervals - mean_dt)) / mean_dt)
    return 1.0 / mean_dt if relative_deviation <= 0.001 else None


def get_classifier() -> ECGClassifier:
    """Load the saved model lazily so training can happen while the API is stopped."""
    try:
        return ECGClassifier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Trained ML model is not installed. Run `python -m backend.train` after placing the MIT-BIH-derived training/test CSVs in data/.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Saved ML model could not be loaded: {exc}",
        ) from exc


@app.get("/api/health")
def health() -> dict[str, Any]:
    model_ready = MODEL_PATH.exists() and SCALER_PATH.exists()
    return {
        "status": "ok",
        "api": "ready",
        "model_loaded": model_ready,
        "model_artifacts": {
            "classifier": MODEL_PATH.name,
            "scaler": SCALER_PATH.name,
        },
        "message": (
            "FastAPI, ECG preprocessing, feature extraction, and ML inference are ready."
            if model_ready
            else "FastAPI, ECG preprocessing, and feature extraction are ready; ML model artifacts are not installed."
        ),
    }


@app.post("/api/upload-ecg")
def upload_ecg(payload: ECGRequest) -> dict[str, Any]:
    """Validate ECG data received from the frontend.

    The endpoint does not persist files. It is the communication-layer
    validation step before /api/analyze and /api/predict.
    """
    try:
        fs = payload.sampling_rate_hz or infer_sampling_rate(payload.time)
        x, _ = validate_ecg_input(payload.samples, fs, min_samples=3)
        if payload.time is not None and len(payload.time) != len(payload.samples):
            raise ValueError("The time array must have the same number of values as samples.")

        series = pd.Series(x)
        return {
            "status": "ok",
            "filename": payload.filename,
            "samples": int(x.size),
            "sampling_rate_hz": float(fs),
            "valid": True,
            "summary": {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            },
            "message": "ECG received and validated successfully.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze")
def analyze_ecg(payload: AnalyzeRequest) -> dict[str, Any]:
    sampling_rate = payload.sampling_rate_hz or infer_sampling_rate(payload.time)
    if sampling_rate is None:
        raise HTTPException(
            status_code=400,
            detail="A valid sampling rate is required. Upload a time,amplitude CSV with uniform timestamps or provide sampling_rate_hz.",
        )
    if payload.time is not None and len(payload.time) != len(payload.samples):
        raise HTTPException(status_code=400, detail="The time array must match the ECG sample count.")

    try:
        cfg_data = payload.config.model_dump() if payload.config else {}
        config = PreprocessingConfig(**cfg_data)

        result = preprocess_ecg(payload.samples, sampling_rate, config=config)
        feature_result = extract_features(
            result["processed_signal"],
            result["sampling_rate_hz"],
            original_signal=result["original_signal"],
        )

        return {
            "status": "ok",
            "filename": payload.filename,
            "original_signal": result["original_signal"],
            "processed_signal": result["processed_signal"],
            "sampling_rate_hz": result["sampling_rate_hz"],
            "preprocessing_info": result["preprocessing_info"],
            "features": feature_result,
            "machine_learning": False,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ECG processing failed: {exc}") from exc


@app.post("/api/predict")
def predict_ecg(payload: ECGRequest) -> dict[str, Any]:
    """Run the actual saved ML model on one 187-sample training-format beat."""
    if len(payload.samples) != 187:
        raise HTTPException(
            status_code=400,
            detail="ML prediction requires exactly 187 ECG samples because the trained model was trained on 187-sample heartbeat segments.",
        )

    try:
        model = get_classifier()
        result = model.predict(payload.samples)

        # These are independently calculated display features. The model itself
        # uses the `model_features` returned by the classifier.
        display_signal = np.asarray(result["processed_signal"], dtype=float)
        display_features = extract_features(
            display_signal,
            125.0,
            original_signal=np.asarray(payload.samples, dtype=float),
        )

        return {
            "status": "ok",
            "classification": result["classification"],
            **({"confidence": result["confidence"]} if "confidence" in result else {}),
            "class_probabilities": result.get("class_probabilities"),
            "features": {
                "heart_rate": display_features["heart_rate_bpm"],
                "r_peak_count": display_features["r_peak_count"],
                "mean_rr_interval_s": display_features["mean_rr_interval_s"],
                "signal_quality": display_features["signal_quality"],
            },
            "model_features": result["model_features"],
            "processed_ecg": result["processed_signal"],
            "model_type": result["model_type"],
            "dataset_task": result["dataset_task"],
            "training_matched_preprocessing": result["preprocessing"],
            "clinical_diagnosis": False,
            "warning": "This is an educational/research screening result and is not a medical diagnosis.",
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ECG prediction failed: {exc}") from exc


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return {
            "model_available": False,
            "message": "No trained model artifacts are installed.",
            "expected_samples": 187,
            "task": "NORMAL vs ABNORMAL",
        }

    try:
        model = get_classifier()
        bundle = model.bundle
        estimator = bundle["model"]
        return {
            "model_available": True,
            "model_type": type(estimator).__name__,
            "task": "NORMAL vs ABNORMAL",
            "expected_samples": int(bundle.get("n_signal_samples", 187)),
            "feature_version": bundle.get("feature_version"),
            "sampling_rate_hz": bundle.get("sampling_rate_hz"),
            "class_names": bundle.get("class_names"),
            "original_label_mapping": bundle.get("original_label_mapping"),
            "supports_probability": hasattr(estimator, "predict_proba"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model information could not be loaded: {exc}") from exc


# Keep the old endpoint as a compatibility alias for the previous frontend.
@app.post("/api/classify")
def classify_compat(payload: ECGRequest):
    return predict_ecg(payload)


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
