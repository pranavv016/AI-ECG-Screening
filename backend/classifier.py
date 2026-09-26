from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .preprocessing import preprocess_ml_beat
from .train import extract_features_from_beat, training_feature_names, N_SIGNAL


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ecg_classifier.joblib"
SCALER_PATH = ROOT / "models" / "ecg_scaler.joblib"


class ECGClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                "Trained ECG model is not available. Run `python -m backend.train` first."
            )
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.bundle = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if self.bundle.get("n_signal_samples") != N_SIGNAL:
            raise ValueError("Saved model expects an incompatible ECG sample length.")
        if self.bundle.get("feature_version") != "ecg_binary_v1":
            raise ValueError("Saved model uses an unsupported feature version.")

    def predict(self, samples: Any) -> dict[str, Any]:
        raw = np.asarray(samples, dtype=float).reshape(-1)
        xz = preprocess_ml_beat(raw, expected_samples=N_SIGNAL)

        features = extract_features_from_beat(raw).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        model = self.bundle["model"]

        prediction = int(model.predict(features_scaled)[0])
        result: dict[str, Any] = {
            "classification": self.bundle.get("class_names", {0: "NORMAL", 1: "ABNORMAL"})[prediction],
            "class_id": prediction,
            "model_type": type(model).__name__,
            "dataset_task": "MIT-BIH-derived heartbeat NORMAL vs ABNORMAL",
            "clinical_diagnosis": False,
            "preprocessing": "per-beat z-score standardization (training-matched)",
            "processed_signal": xz.tolist(),
            "model_features": {
                name: float(value)
                for name, value in zip(training_feature_names(), features[0])
            },
        }

        # Confidence is returned only when the fitted model genuinely exposes
        # class probabilities. No confidence is fabricated for models that do not.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == len(probabilities):
                probability_map = {
                    str(self.bundle.get("class_names", {0: "NORMAL", 1: "ABNORMAL"})[int(cls)]): float(prob)
                    for cls, prob in zip(classes, probabilities)
                }
                result["confidence"] = float(max(probability_map.values()))
                result["class_probabilities"] = probability_map

        return result
