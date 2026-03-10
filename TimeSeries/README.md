# TimeSeries

This folder is a workspace wrapper for the main BTCUSD forecasting / MLOps project located in `timeseries/`.

## What this project does

The `timeseries/` project is an end-to-end, production-style ML workflow for **forecasting BTCUSD** from historical market data. It combines:

- **Data ingestion + preprocessing** (writes processed datasets into `timeseries/data/processed/`)
- **Model training** (baseline scripts like `timeseries/src/models/linear_regression.py`, with room for ARIMA/RNN variants)
- **Experiment tracking with MLflow** (local tracking DB at `timeseries/mlflow.db`, runs/artifacts under `timeseries/mlruns/`)
- **A Django dashboard** to view results and serve the UI (`timeseries/dashboard/`)

The main entry point (`timeseries/run_project.py`) bootstraps the environment by running Django migrations, ingesting data if missing, optionally creating an initial MLflow training run, and then starting the dashboard server.

## Quickstart

```powershell
cd .\timeseries
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_project.py
```

## Project docs

- Primary documentation: `timeseries/README.md`
- Entry point script: `timeseries/run_project.py`
