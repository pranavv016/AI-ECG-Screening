# AI-Assisted ECG Screening System

Educational/research prototype for ECG signal processing and a supervised
machine-learning screening benchmark.

> **Important:** This is not a clinical diagnostic device. The classifier
> predicts only the two benchmark classes **NORMAL** and **ABNORMAL** for
> segmented ECG heartbeats. It does not diagnose a disease or a patient.

## Part 5 — machine learning

### Dataset

The training pipeline uses the **MIT-BIH Arrhythmia Database-derived ECG
Heartbeat Categorization Dataset** representation: each row contains a
segmented heartbeat with 187 signal samples followed by the original
MIT-BIH-derived class code 0–4. The public dataset description identifies its
source as PhysioNet's MIT-BIH Arrhythmia Database and reports a sampling
frequency of 125 Hz for the distributed heartbeat segments.

The original MIT-BIH Arrhythmia Database is an open-access PhysioNet database
of two-channel ambulatory ECG recordings from 47 subjects. Its files are
available under the **Open Data Commons Attribution License v1.0**. When the
original PhysioNet data are used, cite the database and PhysioNet as required.

For this project, the CSV representation should be kept outside GitHub unless
you have separately verified the redistribution terms of the copy you are
using. Do not commit a downloaded patient/ECG dataset to this repository by
default.

### Original labels

| Original code | Meaning |
|---|---|
| 0 / N | Normal beat |
| 1 / S | Supraventricular ectopic beat |
| 2 / V | Ventricular ectopic beat |
| 3 / F | Fusion of ventricular and normal beat |
| 4 / Q | Unclassifiable beat |

These meanings follow the standard PhysioNet annotation terminology.

### NORMAL / ABNORMAL mapping

The binary task deliberately does **not** invent disease labels:

- `0 / N` → `NORMAL`
- `1 / S`, `2 / V`, `3 / F`, `4 / Q` → `ABNORMAL`

Therefore "ABNORMAL" means that the original beat annotation is not the
normal-beat class. It does **not** mean a particular disease is present.

### Why this baseline

A **Random Forest** is used as the baseline because it works well with a
moderate number of handcrafted signal/morphology features, is fast enough for
a student project, and does not require a deep-learning stack.

The feature pipeline is deterministic:

1. Validate 187 finite ECG samples.
2. Standardize each beat for morphology.
3. Calculate amplitude, dispersion, skewness, kurtosis and derivative features.
4. Calculate periodogram-derived frequency/band-power features.
5. Add quantiles and a downsampled morphology representation.
6. Fit `StandardScaler` on training data only.
7. Train `RandomForestClassifier(class_weight="balanced")`.

### Train/test split

The supplied MIT-BIH-derived CSV distribution already has separate training and
testing files. The pipeline uses:

- `data/mitbih_train.csv` for training.
- `data/mitbih_test.csv` for held-out testing.

This is preferable to randomly mixing all heartbeat rows before evaluation.
The benchmark files do not expose a patient identifier suitable for a
patient-independent split, so the reported result must **not** be interpreted
as a patient-level clinical performance estimate.

## Setup

1. Obtain the MIT-BIH-derived heartbeat CSV files from a legitimate source and
   verify its terms of use. A commonly distributed representation is the public
   ECG Heartbeat Categorization Dataset, whose data source is listed as
   PhysioNet's MIT-BIH Arrhythmia Database.
2. Put the two files here:

```text
AI-ECG-Screening/
├── data/
│   ├── mitbih_train.csv
│   └── mitbih_test.csv
```

Each file must have **188 columns with no header**:
187 ECG samples + 1 original class label.

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Train:

```bash
python -m backend.train
```

or:

```bash
python backend/train.py
```

The script validates the data, extracts features, trains the classifier,
evaluates it, saves the model, and then reloads the saved artifacts and makes
one prediction on a held-out test heartbeat.

### Saved artifacts

After successful training:

```text
models/
├── ecg_classifier.joblib
├── ecg_scaler.joblib
└── evaluation_metrics.json
```

The `.joblib` files are generated artifacts and should normally be kept out of
Git history unless you deliberately want to distribute that exact model.

### Evaluation

The training script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion matrix
- Per-class precision/recall/F1 in the classification report

Because the original dataset is highly imbalanced, **macro F1 and class-specific
recall should be considered alongside accuracy**.

No clinical accuracy or diagnostic performance is claimed.

## Backend API

Run:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### `/api/analyze`

Existing endpoint. It performs signal validation, filtering, normalization and
classical feature extraction.

### `/api/classify`

New ML endpoint.

It expects one segmented heartbeat containing exactly **187 samples**, matching
the representation used during training.

Example request body:

```json
{
  "samples": [0.01, 0.02, 0.03]
}
```

The example above is intentionally too short; a real request must contain all
187 samples.

The response contains `NORMAL` or `ABNORMAL` and the model's class probability
output. The probability is a model probability, not a medical certainty or
clinical risk score.

If the trained artifacts are absent, the endpoint returns HTTP 503 instead of
pretending to classify the ECG.

## Current project structure

```text
AI-ECG-Screening/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── classifier.py
├── data/
│   ├── sample_ecg.csv
│   ├── mitbih_train.csv        # user downloads locally; do not commit by default
│   └── mitbih_test.csv         # user downloads locally; do not commit by default
├── models/
│   ├── ecg_classifier.joblib   # generated after training
│   ├── ecg_scaler.joblib       # generated after training
│   └── evaluation_metrics.json
├── frontend/
├── requirements.txt
└── README.md
```

## Reproducibility

The classifier uses `random_state=42`. Feature extraction and scaling are
deterministic. The training script prints dataset sizes and original-label
distributions before training.

## Dataset citations / attribution

- MIT-BIH Arrhythmia Database, PhysioNet, DOI 10.13026/C2F305.
- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.*
  IEEE Engineering in Medicine and Biology Magazine. 2001;20(3):45–50.
- Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.*
  Circulation. 2000;101(23):e215–e220.

If you redistribute the original PhysioNet files or substantial adapted
material, follow the applicable Open Data Commons Attribution License terms
and retain the required attribution/license information.

## Safety and scope

This project is an educational/research ML experiment. It is not validated for
clinical use, diagnosis, triage, treatment, or medical decision-making.


## Part 6 — FastAPI frontend/ML integration

The frontend now uses the following flow when **Analyze ECG** is clicked:

```text
ECG CSV
  -> POST /api/upload-ecg
  -> POST /api/analyze
       -> preprocessing.py
       -> features.py
  -> POST /api/predict (only for exactly 187 samples)
       -> training-matched per-beat normalization
       -> same feature extractor used during training
       -> saved Random Forest model
  -> JSON result -> frontend
```

### API endpoints

- `GET /api/health` — API/model readiness.
- `POST /api/upload-ecg` — validates ECG data received from the frontend.
- `POST /api/analyze` — preprocessing, processed waveform, and classical ECG features.
- `POST /api/predict` — actual saved ML inference on exactly 187 samples.
- `GET /api/model-info` — saved model metadata and supported task.
- `POST /api/classify` — compatibility alias for `/api/predict`.

The ML endpoint never invents a confidence value. It returns `confidence` only when
the loaded estimator exposes `predict_proba`.

### Important model/input constraint

The supplied project archive does **not** contain `models/ecg_classifier.joblib`,
`models/ecg_scaler.joblib`, `data/mitbih_train.csv`, or `data/mitbih_test.csv`.
Therefore the API correctly reports the ML model as unavailable until the trained
artifacts are installed. It will return HTTP 503 rather than fabricate a result.

The supplied demo ECG is 2,000 samples long, while the trained benchmark expects
one 187-sample heartbeat. The frontend therefore processes and visualizes the
whole demo signal but does not silently truncate it and claim that a model result
came from a valid trained heartbeat.

### Verification performed

- Python backend modules compile successfully.
- Frontend JavaScript passes `node --check`.
- `GET /api/health`: passed.
- `POST /api/upload-ecg` using `data/sample_ecg.csv`: passed.
- `POST /api/analyze` using `data/sample_ecg.csv`: passed; 2,000 processed samples
  returned and ECG features were calculated.
- `GET /api/model-info`: passed and correctly reported that no model artifact is
  installed.
- `POST /api/predict` with 187 samples: passed the endpoint contract and correctly
  returned HTTP 503 because no trained model artifact is present.
- `POST /api/predict` with the wrong length: correctly returned HTTP 400.

A real ML prediction and frontend verification cannot honestly be claimed from this
archive until the exact trained model/scaler artifacts (or the legitimate training
dataset needed to train them) are supplied.
