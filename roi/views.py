import os
import sys
from math import sqrt
from pathlib import Path

import mlflow
import pandas as pd
from django.shortcuts import render
from mlflow.tracking import MlflowClient
from statsmodels.tsa.arima.model import ARIMA

from .models import ROIMetric


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MLFLOW_TRACKING_URI, configure_mlflow


MODEL_EXPERIMENTS = [
    ("ARIMA", "BTCUSD_ARIMA_Forecasting"),
    ("LSTM", "BTCUSD_LSTM_Forecasting"),
    ("Linear Regression", "BTCUSD_Linear_Regression"),
]


def _first_metric(run, metric_keys):
    for key in metric_keys:
        if key in run.data.metrics:
            return run.data.metrics[key]
    return None


def _latest_close_from_processed_csv():
    processed_csv = PROJECT_ROOT / "data" / "processed" / "btcusd_processed.csv"
    if not processed_csv.exists():
        return None

    df = pd.read_csv(processed_csv)
    if df.empty or "Close" not in df.columns:
        return None

    return float(df["Close"].iloc[-1])


def _arima_fallback_prediction():
    processed_csv = PROJECT_ROOT / "data" / "processed" / "btcusd_processed.csv"
    if not processed_csv.exists():
        return _latest_close_from_processed_csv()

    df = pd.read_csv(processed_csv)
    if df.empty or "Close" not in df.columns or len(df["Close"]) < 30:
        return _latest_close_from_processed_csv()

    try:
        model = ARIMA(df["Close"], order=(5, 1, 0))
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
        return float(prediction.iloc[0])
    except Exception:
        return _latest_close_from_processed_csv()


def _search_recent_runs(client, experiment_name, max_results=50):
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []

    return client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )


def _first_available_metric(runs, metric_keys):
    for run in runs:
        value = _first_metric(run, metric_keys)
        if value is not None:
            return float(value), run
    return None, None


def _build_model_comparison():
    rows = []

    try:
        configure_mlflow()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
        client = MlflowClient()
    except Exception:
        client = None

    for model_name, experiment_name in MODEL_EXPERIMENTS:
        if client is None:
            rows.append(
                {
                    "model_name": model_name,
                    "mse": "N/A",
                    "mae": "N/A",
                    "rmse": "N/A",
                    "predicted_close": "N/A",
                    "run_time": "N/A",
                }
            )
            continue

        runs = _search_recent_runs(client, experiment_name)
        if not runs:
            rows.append(
                {
                    "model_name": model_name,
                    "mse": "N/A",
                    "mae": "N/A",
                    "rmse": "N/A",
                    "predicted_close": "N/A",
                    "run_time": "N/A",
                }
            )
            continue

        run = runs[0]
        mse, mse_run = _first_available_metric(runs, ["test_mse", "mse"])
        mae, mae_run = _first_available_metric(runs, ["mae", "test_mae"])
        rmse, rmse_run = _first_available_metric(runs, ["rmse", "test_rmse"])
        predicted_close, predicted_run = _first_available_metric(
            runs,
            [
                "predicted_close_1h",
                "prediction_1h",
                "predicted_close",
                "predicted_price",
                "prediction",
            ],
        )

        if rmse is None and mse is not None:
            rmse = sqrt(mse)

        if predicted_close is None:
            if model_name == "ARIMA":
                predicted_close = _arima_fallback_prediction()
            else:
                predicted_close = _latest_close_from_processed_csv()

        metric_run = predicted_run or rmse_run or mae_run or mse_run or run
        run_time = pd.to_datetime(metric_run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M")

        rows.append(
            {
                "model_name": model_name,
                "mse": f"{float(mse):.6f}" if mse is not None else "N/A",
                "mae": f"{float(mae):.6f}" if mae is not None else "N/A",
                "rmse": f"{float(rmse):.6f}" if rmse is not None else "N/A",
                "predicted_close": f"{float(predicted_close):,.2f}" if predicted_close is not None else "N/A",
                "run_time": run_time,
            }
        )

    return rows


def roi_index(request):
    """
    Displays simulated ROI and financial impact of the forecasting model.
    """
    latest_roi = ROIMetric.objects.order_by("-calculated_at").first()

    # If no data exists, create a dummy entry for demonstration
    if not latest_roi:
        latest_roi = ROIMetric.objects.create(
            model_version="1.0",
            period="Last 30 Days",
            simulated_profit_usd=12500.50,
            risk_reduction_pct=15.4,
        )

    context = {
        "latest_roi": latest_roi,
        "history": ROIMetric.objects.all().order_by("-calculated_at")[:10],
        "model_comparison": _build_model_comparison(),
    }
    return render(request, "dashboard/roi_index.html", context)
