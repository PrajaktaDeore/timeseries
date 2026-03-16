import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from django.shortcuts import render
from mlflow.tracking import MlflowClient

from .models import ROIMetric


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
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

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
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
        mse = _first_metric(run, ["test_mse", "mse"])
        mae = _first_metric(run, ["mae", "test_mae"])
        rmse = _first_metric(run, ["rmse", "test_rmse"])
        predicted_close = _first_metric(
            run, ["predicted_close_1h", "prediction_1h", "predicted_close", "predicted_price", "prediction"]
        )
        run_time = pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M")

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
    latest_roi = ROIMetric.objects.order_by('-calculated_at').first()
    
    # If no data exists, create a dummy entry for demonstration
    if not latest_roi:
        latest_roi = ROIMetric.objects.create(
            model_version="1.0",
            period="Last 30 Days",
            simulated_profit_usd=12500.50,
            risk_reduction_pct=15.4,
        )

    context = {
        'latest_roi': latest_roi,
        'history': ROIMetric.objects.all().order_by('-calculated_at')[:10],
        'model_comparison': _build_model_comparison(),
    }
    return render(request, 'dashboard/roi_index.html', context)
