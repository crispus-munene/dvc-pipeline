import mlflow
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import dvclive
import os
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
)
from mlflow.tracking import MlflowClient

from data_loading import load_data
from splitting import train_test_split_data
from preprocessing import build_pipeline
from utils import load_params


def train_and_log(params):
    # --- Logging setup ---
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

    # --- MLflow setup ---
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = MlflowClient()
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    run_name = params["mlflow"]["run_name"]
    artifact_path = params["mlflow"]["artifact_path"]

    # --- Prepare output dirs ---
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    os.makedirs("outputs/dvclive", exist_ok=True)

    # Unique suffix per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Data ---
    df = load_data(params["data"]["path"])
    X_train, X_test, y_train, y_test = train_test_split_data(
        df, test_size=params["data"]["test_size"]
    )

    categorical_cols = X_train.select_dtypes(include="O").columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude="O").columns.tolist()
    model = build_pipeline(categorical_cols, numeric_cols, params["model"])

    # --- Training + Logging ---
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Dynamically evaluate metrics
        metrics = {}
        for metric in params["model"]["metrics"]:
            if metric == "accuracy":
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
            elif metric == "precision":
                metrics["precision"] = precision_score(y_test, y_pred)
            elif metric == "recall":
                metrics["recall"] = recall_score(y_test, y_pred)
            elif metric == "f1":
                metrics["f1"] = f1_score(y_test, y_pred)

        # Confusion Matrix
        df_cm = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=np.unique(y_test),
            index=np.unique(y_test)
        )
        sns.heatmap(df_cm, cmap="Blues", annot=True)
        cm_path = f"outputs/confusion_matrices/confusion_matrix_{timestamp}.png"
        plt.savefig(cm_path)
        plt.close()

        # Save metrics JSON
        metrics_path = f"outputs/metrics/metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        # DVCLive logging
        with dvclive.Live("outputs/dvclive", report=None) as live:
            for k, v in metrics.items():
                live.log_metric(k, v)
            live.next_step()

        # MLflow logging
        run_id = mlflow.active_run().info.run_id
        mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
        mlflow.log_params(params["model"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_path)

        # Register + promote to production
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(
            model_uri=model_uri, name=params["model"]["name"]
        )
        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage="production"
        )
