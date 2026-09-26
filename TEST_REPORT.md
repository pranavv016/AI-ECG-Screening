# Integration Test Report

Date: 2026-09-24

## Passed

- `python -m compileall backend`
- `node --check frontend/app.js`
- `GET /api/health` -> 200
- `POST /api/upload-ecg` with `data/sample_ecg.csv` -> 200
- `POST /api/analyze` with `data/sample_ecg.csv` -> 200
  - 2,000 input samples
  - 2,000 processed samples
  - heart-rate feature calculated
- `GET /api/model-info` -> 200
- `POST /api/predict` with 187 samples -> 503 (correct, because the archive has no trained model artifact)
- `POST /api/predict` with 3 samples -> 400 (correct validation)

## Blocked verification

The uploaded project contains no `ecg_classifier.joblib` or `ecg_scaler.joblib`, and no
MIT-BIH-derived `mitbih_train.csv` / `mitbih_test.csv`. Consequently, generating or
claiming an ML prediction would be fabricated. The server is deliberately fail-safe
and refuses ML inference until the trained artifacts are installed.
