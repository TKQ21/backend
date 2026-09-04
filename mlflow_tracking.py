"""MLflow experiment tracking + model registry. No-ops cleanly if MLflow is absent."""
from __future__ import annotations

from typing import Any

EXPERIMENT = "electricity-demand-forecast"
REGISTERED_NAME = "electricity-demand-forecaster"


def log_comparison(payload: dict[str, Any], best_est: Any) -> dict[str, Any]:
    """Log every model as a run, then register the best one.

    Returns a dict describing the tracking state so the UI can show real MLflow
    information (or an honest "not available" when MLflow is not installed).
    """
    info: dict[str, Any] = {
        "available": False,
        "experiment": EXPERIMENT,
        "registeredName": REGISTERED_NAME,
        "version": None,
        "runId": None,
        "runStatus": "not-logged",
        "trackingUri": None,
        "artifactUri": None,
        "childRuns": [],
        "note": "MLflow is not installed on the backend; the model artifact is still saved to backend/artifacts/.",
    }
    try:
        import mlflow
        import mlflow.sklearn
    except Exception:
        return info

    try:
        mlflow.set_experiment(EXPERIMENT)
        info["available"] = True
        info["trackingUri"] = mlflow.get_tracking_uri()
        info["note"] = None
        best_name = payload["best"]["name"]
        for r in payload["results"]:
            if "val" not in r:
                continue
            with mlflow.start_run(run_name=f"{payload['runId']}-{r['name']}") as run:
                mlflow.log_params({k: v for k, v in r.get("params", {}).items()})
                mlflow.set_tags({"algorithm": r["name"], "family": r["family"],
                                 "is_best": str(r["name"] == best_name)})
                mlflow.log_metrics({f"val_{k}": v for k, v in r["val"].items() if v is not None})
                mlflow.log_metrics({f"test_{k}": v for k, v in r["test"].items() if v is not None})
                mlflow.log_metric("train_seconds", r["trainSeconds"])
                info["childRuns"].append({"name": r["name"], "runId": run.info.run_id,
                                          "status": "FINISHED"})

        with mlflow.start_run(run_name=f"{payload['runId']}-BEST-{best_name}") as run:
            mlflow.set_tags({"selection_metric": "validation_rmse", "algorithm": best_name})
            mlflow.log_metrics({f"val_{k}": v for k, v in payload["best"]["val"].items() if v is not None})
            mlflow.log_metrics({f"test_{k}": v for k, v in payload["best"]["test"].items() if v is not None})
            mlflow.sklearn.log_model(best_est, "model", registered_model_name=REGISTERED_NAME)
            info["runId"] = run.info.run_id
            info["runStatus"] = "FINISHED"
            info["artifactUri"] = f"{run.info.artifact_uri}/model"
        try:
            from mlflow.tracking import MlflowClient
            versions = MlflowClient().search_model_versions(f"name='{REGISTERED_NAME}'")
            info["version"] = max(int(v.version) for v in versions) if versions else None
        except Exception:
            pass
        return info
    except Exception as exc:
        info["runStatus"] = "FAILED"
        info["note"] = f"MLflow logging failed: {str(exc)[:200]}"
        return info
