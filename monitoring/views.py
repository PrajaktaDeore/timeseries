from pathlib import Path
import os
import sys

import mlflow
import mlflow.pyfunc
import pandas as pd
from django.shortcuts import render
from mlflow.tracking import MlflowClient
from statsmodels.tsa.arima.model import ARIMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_REGISTERED_MODELS,
    configure_mlflow,
)

LINEAR_FEATURES = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]


def _first_metric(run, keys):
    for key in keys:
        if key in run.data.metrics:
            return run.data.metrics[key]
    return None


def _display_stage(stage_value):
    if not stage_value or stage_value.lower() == "none":
        return "Unstaged"
    return stage_value


def _latest_close_from_processed_csv():
    processed_csv = PROJECT_ROOT / "data" / "processed" / "btcusd_processed.csv"
    if not processed_csv.exists():
        return None

    df = pd.read_csv(processed_csv)
    if df.empty or "Close" not in df.columns:
        return None

    return float(df["Close"].iloc[-1])


def _load_processed_df():
    processed_csv = PROJECT_ROOT / "data" / "processed" / "btcusd_processed.csv"
    if not processed_csv.exists():
        return None

    df = pd.read_csv(processed_csv)
    if df.empty:
        return None

    return df


def _format_prediction(value):
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


def _prediction_from_experiment_runs(client, experiment_name, metric_keys):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None, "No MLflow experiment found"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    for run in runs:
        metric_value = _first_metric(run, metric_keys)
        if metric_value is not None:
            return float(metric_value), f"MLflow run {run.info.run_id[:8]}"

    return None, "No prediction metric logged"


def _linear_regression_fallback_prediction():
    df = _load_processed_df()
    if df is None:
        return None, "Fallback unavailable: processed data missing"

    if not all(feature in df.columns for feature in LINEAR_FEATURES):
        return _latest_close_from_processed_csv(), "Fallback: latest Close price"

    try:
        model = mlflow.pyfunc.load_model("models:/BTCUSD_Linear_Regression/latest")
        latest_features = df[LINEAR_FEATURES].tail(1)
        prediction = model.predict(latest_features)
        return float(prediction[0]), "Fallback: latest registered Linear Regression model"
    except Exception:
        return _latest_close_from_processed_csv(), "Fallback: latest Close price"


def _arima_fallback_prediction():
    df = _load_processed_df()
    if df is None or "Close" not in df.columns or len(df["Close"]) < 30:
        return _latest_close_from_processed_csv(), "Fallback: latest Close price"

    try:
        model = ARIMA(df["Close"], order=(5, 1, 0))
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
        return float(prediction.iloc[0]), "Fallback: ARIMA one-step forecast"
    except Exception:
        return _latest_close_from_processed_csv(), "Fallback: latest Close price"


def _get_model_predictions(client):
    model_predictions = []

    arima_value, arima_source = _prediction_from_experiment_runs(
        client,
        "BTCUSD_ARIMA_Forecasting",
        [
            "predicted_close_1h",
            "prediction_1h",
            "predicted_close",
            "predicted_price",
            "prediction",
        ],
    )
    if arima_value is None:
        arima_value, arima_source = _arima_fallback_prediction()
    model_predictions.append(
        {
            "name": "ARIMA",
            "prediction": _format_prediction(arima_value),
            "source": arima_source,
        }
    )

    lstm_value, lstm_source = _prediction_from_experiment_runs(
        client,
        "BTCUSD_LSTM_Forecasting",
        [
            "predicted_close_1h",
            "prediction_1h",
            "predicted_close",
            "predicted_price",
            "prediction",
        ],
    )
    if lstm_value is None:
        lstm_value = _latest_close_from_processed_csv()
        lstm_source = "Fallback: latest Close price"
    model_predictions.append(
        {
            "name": "LSTM",
            "prediction": _format_prediction(lstm_value),
            "source": lstm_source,
        }
    )

    linear_value, linear_source = _prediction_from_experiment_runs(
        client,
        "BTCUSD_Linear_Regression",
        [
            "predicted_close_1h",
            "prediction_1h",
            "predicted_close",
            "predicted_price",
            "prediction",
        ],
    )
    if linear_value is None:
        linear_value, linear_source = _linear_regression_fallback_prediction()
    model_predictions.append(
        {
            "name": "Linear Regression",
            "prediction": _format_prediction(linear_value),
            "source": linear_source,
        }
    )

    return model_predictions


def dashboard_overview(request):
    latest_prediction = "N/A"
    latest_prediction_source = "No prediction metric logged yet"
    model_mse = "N/A"
    runs_data = []
    model_predictions = []
    recent_labels = []
    recent_close_values = []
    latest_versions = []
    mlflow_error = None

    try:
        configure_mlflow()
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = MlflowClient()

        model_predictions = _get_model_predictions(client)

        experiment_ids = []
        for experiment_name in DEFAULT_EXPERIMENTS:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_ids.append(experiment.experiment_id)

        all_runs = []
        if experiment_ids:
            all_runs = client.search_runs(
                experiment_ids=experiment_ids,
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=200,
            )

            for run in all_runs[:10]:
                mse_value = _first_metric(run, ["test_mse", "mse"])
                runs_data.append(
                    {
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "model_type": run.data.params.get("model_type", "N/A"),
                        "mse": mse_value if mse_value is not None else "N/A",
                        "start_time": pd.to_datetime(run.info.start_time, unit="ms").strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                )

            runs_with_mse = [
                (run, _first_metric(run, ["test_mse", "mse"]))
                for run in all_runs
                if _first_metric(run, ["test_mse", "mse"]) is not None
            ]
            if runs_with_mse:
                best_run, best_mse = min(runs_with_mse, key=lambda item: item[1])
                model_mse = f"{best_mse:.4f}"

            runs_with_prediction = [
                run
                for run in all_runs
                if _first_metric(
                    run,
                    ["prediction", "latest_prediction", "predicted_price", "predicted_close"],
                )
                is not None
            ]
            if runs_with_prediction:
                latest_prediction_run = runs_with_prediction[0]
                latest_prediction_value = _first_metric(
                    latest_prediction_run,
                    ["prediction", "latest_prediction", "predicted_price", "predicted_close"],
                )
                latest_prediction = f"{latest_prediction_value:,.2f}"
                model_type = latest_prediction_run.data.params.get("model_type", "Model")
                latest_prediction_source = f"From latest {model_type} run metric"

        for model_name in DEFAULT_REGISTERED_MODELS:
            try:
                registered_model = client.get_registered_model(model_name)
                for version in registered_model.latest_versions:
                    latest_versions.append(
                        {
                            "model_name": model_name,
                            "version": version.version,
                            "current_stage": _display_stage(version.current_stage),
                        }
                    )
            except Exception:
                continue

    except Exception as exc:
        mlflow_error = str(exc)
        latest_close = _latest_close_from_processed_csv()
        model_predictions = [
            {
                "name": "ARIMA",
                "prediction": _format_prediction(latest_close),
                "source": "MLflow unavailable: showing latest Close price",
            },
            {
                "name": "LSTM",
                "prediction": _format_prediction(latest_close),
                "source": "MLflow unavailable: showing latest Close price",
            },
            {
                "name": "Linear Regression",
                "prediction": _format_prediction(latest_close),
                "source": "MLflow unavailable: showing latest Close price",
            },
        ]

    first_available_prediction = next(
        (item for item in model_predictions if item["prediction"] != "N/A"),
        None,
    )
    if first_available_prediction:
        latest_prediction = first_available_prediction["prediction"]
        latest_prediction_source = f"{first_available_prediction['name']} next 1-hour prediction"

    if latest_prediction == "N/A":
        latest_close = _latest_close_from_processed_csv()
        if latest_close is not None:
            latest_prediction = f"{latest_close:,.2f}"
            latest_prediction_source = "Fallback: latest Close price from processed data"

    processed_df = _load_processed_df()
    if processed_df is not None and "Close" in processed_df.columns:
        tail_df = processed_df.tail(90).copy()
        if "Date" in tail_df.columns:
            recent_labels = [str(item) for item in tail_df["Date"].tolist()]
        else:
            recent_labels = [str(idx) for idx in tail_df.index.tolist()]
        recent_close_values = [float(value) for value in tail_df["Close"].tolist()]

    context = {
        "latest_prediction": latest_prediction,
        "latest_prediction_source": latest_prediction_source,
        "model_accuracy": model_mse,
        "runs": runs_data,
        "latest_versions": latest_versions,
        "model_predictions": model_predictions,
        "recent_labels": recent_labels,
        "recent_close_values": recent_close_values,
        "mlflow_error": mlflow_error,
    }

    return render(request, "dashboard/overview.html", context)


def drift_monitoring(request):
    df = _load_processed_df()
    status = "No Data"
    status_class = "status-other"
    baseline_mean = None
    recent_mean = None
    drift_pct = None
    latest_close = _latest_close_from_processed_csv()
    close_values_for_hist = []

    if df is not None and "Close" in df.columns and len(df["Close"]) >= 60:
        close = df["Close"].astype(float)
        close_values_for_hist = [float(v) for v in close.tail(500).tolist()]
        recent_window = close.tail(30)
        baseline_window = close.iloc[-60:-30]

        recent_mean = float(recent_window.mean())
        baseline_mean = float(baseline_window.mean())
        if baseline_mean != 0:
            drift_pct = ((recent_mean - baseline_mean) / baseline_mean) * 100
        else:
            drift_pct = 0.0

        if abs(drift_pct) >= 5:
            status = "Drift Alert"
            status_class = "status-alert"
        elif abs(drift_pct) >= 2:
            status = "Watch"
            status_class = "status-watch"
        else:
            status = "Stable"
            status_class = "status-stable"

    context = {
        "status": status,
        "status_class": status_class,
        "baseline_mean": f"{baseline_mean:,.2f}" if baseline_mean is not None else "N/A",
        "recent_mean": f"{recent_mean:,.2f}" if recent_mean is not None else "N/A",
        "drift_pct": f"{drift_pct:.2f}%" if drift_pct is not None else "N/A",
        "latest_close": f"{latest_close:,.2f}" if latest_close is not None else "N/A",
        "close_values_for_hist": close_values_for_hist,
    }
    return render(request, "dashboard/drift_monitoring.html", context)
