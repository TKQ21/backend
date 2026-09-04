# ElectriPredict ML backend

```
Lovable Frontend -> CSV Upload -> FastAPI -> Python ML Pipeline
   -> Linear / Ridge / DecisionTree / RandomForest / XGBoost / LightGBM / CatBoost / KNN / Baseline
   -> Metrics (MAE, MSE, RMSE, R2, MAPE) -> Best model by validation RMSE
   -> MLflow (params, metrics, experiments, model versions) -> artifacts -> Docker
```

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
mlflow ui           # optional, http://localhost:5000
```

Point the frontend at it with `VITE_ML_API_URL=http://localhost:8000` (default already).

## Docker

```bash
docker build -t electripredict-api ./backend
docker run -p 8000:8000 electripredict-api
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/ml/health` | liveness + currently registered model |
| POST | `/api/ml/train` | multipart CSV upload, returns `{job_id}` and trains in the background |
| GET | `/api/ml/status/{job_id}` | `queued \| running \| succeeded \| failed`, progress %, message |
| GET | `/api/ml/result/{job_id}` | full comparison, metrics, importances, forecast |
| GET | `/api/ml/runs` | tracked run history (also mirrored to MLflow) |
| GET | `/api/ml/metrics` | metrics of the saved artifact |
| POST | `/api/ml/predict` | single-row inference from the registered model |

## Notes

- Training runs in a background task; the API stays responsive and the browser only polls.
- All models learn the delta vs `lag_1` (stationary target); the level is added back on predict.
- The split is strictly chronological (70/15/15) — a time series is never shuffled.
- XGBoost / LightGBM / CatBoost / MLflow are optional: if a library is missing the pipeline
  skips that model (or MLflow logging) instead of failing.