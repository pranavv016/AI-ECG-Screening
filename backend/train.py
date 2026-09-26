"""
Train the ECG NORMAL vs ABNORMAL classifier.

Dataset:
    MIT-BIH Arrhythmia Database-derived heartbeat CSVs in the format used by
    the public "ECG Heartbeat Categorization Dataset":
        - 187 ECG samples per heartbeat
        - final column = original MIT-BIH-derived class code 0..4

Mapping:
    0 -> NORMAL (N)
    1,2,3,4 -> ABNORMAL (S,V,F,Q)

This is a beat-pattern screening benchmark, not a clinical diagnostic model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from .preprocessing import preprocess_ml_beat
except ImportError:
    from preprocessing import preprocess_ml_beat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "mitbih_train.csv"
DEFAULT_TEST = ROOT / "data" / "mitbih_test.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ecg_classifier.joblib"
SCALER_PATH = MODEL_DIR / "ecg_scaler.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

FS = 125.0
N_SIGNAL = 187


def validate_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{name} is empty.")
    if df.shape[1] != N_SIGNAL + 1:
        raise ValueError(
            f"{name} must contain {N_SIGNAL} signal columns + 1 label column "
            f"(expected {N_SIGNAL + 1}, got {df.shape[1]})."
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad = int(df.isna().sum().sum())
        raise ValueError(f"{name} contains {bad} non-numeric/missing values.")

    labels = df.iloc[:, -1].astype(int)
    allowed = {0, 1, 2, 3, 4}
    found = set(labels.unique())
    if not found.issubset(allowed):
        raise ValueError(f"{name} contains unsupported labels: {sorted(found - allowed)}")

    if not np.isfinite(df.iloc[:, :-1].to_numpy()).all():
        raise ValueError(f"{name} contains non-finite ECG samples.")

    return df


def bandpower(x: np.ndarray, fs: float, low: float, high: float) -> float:
    freqs, psd = signal.periodogram(x, fs=fs)
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def training_feature_names() -> list[str]:
    names = [
        "mean_standardized", "std_standardized", "min_standardized",
        "max_standardized", "range_standardized", "rms_standardized",
        "skewness", "kurtosis", "mean_abs_derivative",
        "std_derivative", "max_abs_derivative", "dominant_frequency_hz",
        "bandpower_0_5_4hz", "bandpower_4_8hz", "bandpower_8_15hz",
        "bandpower_15_30hz",
        "quantile_0.05", "quantile_0.10", "quantile_0.25",
        "quantile_0.50", "quantile_0.75", "quantile_0.90", "quantile_0.95",
    ]
    names.extend([f"morphology_sample_{i}" for i in range(0, N_SIGNAL, 4)])
    return names


def extract_features_from_beat(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Extract transparent, classical features from one 187-sample beat."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != N_SIGNAL or not np.isfinite(x).all():
        raise ValueError("Each ECG beat must contain 187 finite samples.")

    # Exact per-beat normalization shared by training and inference.
    xz = preprocess_ml_beat(x, expected_samples=N_SIGNAL)
    dx = np.diff(xz)

    q = np.quantile(xz, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    freqs, psd = signal.periodogram(xz, fs=fs)
    dominant_freq = float(freqs[np.argmax(psd[1:]) + 1]) if len(psd) > 1 else 0.0

    # Fixed, deterministic morphology/energy features.
    features = [
        np.mean(xz),
        np.std(xz),
        np.min(xz),
        np.max(xz),
        np.ptp(xz),
        np.sqrt(np.mean(xz**2)),
        stats.skew(xz, bias=False),
        stats.kurtosis(xz, bias=False),
        np.mean(np.abs(dx)),
        np.std(dx),
        np.max(np.abs(dx)),
        dominant_freq,
        bandpower(xz, fs, 0.5, 4.0),
        bandpower(xz, fs, 4.0, 8.0),
        bandpower(xz, fs, 8.0, 15.0),
        bandpower(xz, fs, 15.0, 30.0),
    ]
    features.extend(q.tolist())

    # Downsampled morphology keeps the overall waveform shape while limiting
    # dimensionality and overfitting compared with using all 187 raw samples.
    features.extend(xz[::4].tolist())
    return np.asarray(features, dtype=float)


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    signals = df.iloc[:, :N_SIGNAL].to_numpy(dtype=float)
    original_labels = df.iloc[:, -1].to_numpy(dtype=int)
    X = np.vstack([extract_features_from_beat(row) for row in signals])
    y = (original_labels != 0).astype(int)  # 0=NORMAL, 1=ABNORMAL
    return X, y


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found at {path}. "
            "See README.md for the dataset setup procedure."
        )
    return validate_dataset(pd.read_csv(path, header=None), name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIT-BIH-derived ECG binary classifier.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    print("Loading datasets...")
    train_df = load_csv(args.train, "Training")
    test_df = load_csv(args.test, "Testing")

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print("Original training labels:", train_df.iloc[:, -1].value_counts().sort_index().to_dict())
    print("Binary training labels:", (train_df.iloc[:, -1] != 0).value_counts().sort_index().to_dict())

    print("Extracting features...")
    X_train, y_train = make_features(train_df)
    X_test, y_test = make_features(test_df)

    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Both training and testing sets must contain NORMAL and ABNORMAL samples.")

    # Fit preprocessing only on training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random forest is a strong, interpretable baseline for mixed handcrafted
    # morphology/statistical features and does not require linear separability.
    model = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    print("Training Random Forest...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = {
        "dataset": "MIT-BIH Arrhythmia Database-derived heartbeat CSV",
        "task": "NORMAL vs ABNORMAL",
        "normal_definition": "original class 0 / N",
        "abnormal_definition": "original classes 1,2,3,4 / S,V,F,Q",
        "sampling_rate_hz": FS,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["NORMAL", "ABNORMAL"],
            zero_division=0,
            output_dict=True,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "sampling_rate_hz": FS,
        "feature_version": "ecg_binary_v1",
        "n_signal_samples": N_SIGNAL,
        "class_names": {0: "NORMAL", 1: "ABNORMAL"},
        "original_label_mapping": {
            0: "N / normal beat",
            1: "S / supraventricular ectopic beat",
            2: "V / ventricular ectopic beat",
            3: "F / fusion beat",
            4: "Q / unclassifiable beat",
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nEvaluation")
    print("----------")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("Confusion matrix [NORMAL, ABNORMAL]:")
    print(np.asarray(metrics["confusion_matrix"]))

    print("\nSaved:")
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(METRICS_PATH)

    # Required load-and-predict smoke test on a real held-out ECG row.
    loaded_bundle = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    sample_x = extract_features_from_beat(test_df.iloc[0, :N_SIGNAL].to_numpy(float)).reshape(1, -1)
    sample_pred = loaded_bundle["model"].predict(loaded_scaler.transform(sample_x))[0]
    sample_label = int(test_df.iloc[0, -1])
    print("\nSaved-model smoke test")
    print("----------------------")
    print(f"Sample original class: {sample_label}")
    print(f"Saved model prediction : {'ABNORMAL' if sample_pred else 'NORMAL'}")
    print("Saved model loaded successfully and produced a prediction.")


if __name__ == "__main__":
    main()
