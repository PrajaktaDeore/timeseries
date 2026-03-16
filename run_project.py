import argparse
import os
import subprocess
import sys

import mlflow
from mlflow.tracking import MlflowClient

from project_config import PROCESSED_BTCUSD_CSV, configure_mlflow


BOOTSTRAP_TRAINING_STEPS = [
    ("BTCUSD_Linear_Regression", "src/models/linear_regression.py"),
    ("BTCUSD_ARIMA_Forecasting", "src/models/arima_model.py"),
    ("BTCUSD_LSTM_Forecasting", "src/models/lstm_moel.py"),
]


def run_step(args, cwd=None, env=None):
    print(f"\n==> Running: {' '.join(args)}")
    subprocess.run(args, cwd=cwd, env=env, check=True)


def has_any_runs():
    mlflow.set_tracking_uri(configure_mlflow())
    client = MlflowClient()
    for experiment_name in (
        "BTCUSD_Forecasting",
        "BTCUSD_Linear_Regression",
        "BTCUSD_ARIMA_Forecasting",
    ):
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            continue
        runs = client.search_runs([experiment.experiment_id], max_results=1)
        if runs:
            return True
    return False


def experiment_has_runs(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return False

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
    )
    return bool(runs)


def bootstrap_project(env, skip_train=False, train_missing_models=False):
    run_step([sys.executable, "dashboard/manage.py", "migrate"], env=env)

    if not PROCESSED_BTCUSD_CSV.exists():
        run_step([sys.executable, "src/data/ingestion.py"], env=env)

    if skip_train:
        return

    mlflow.set_tracking_uri(env["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    if train_missing_models:
        for experiment_name, script_path in BOOTSTRAP_TRAINING_STEPS:
            if experiment_has_runs(client, experiment_name):
                continue

            try:
                run_step([sys.executable, script_path], env=env)
            except subprocess.CalledProcessError as exc:
                print(f"Warning: bootstrap training failed for {experiment_name}: {exc}")
    elif not has_any_runs():
        run_step([sys.executable, "src/models/linear_regression.py"], env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap and run the BTCUSD forecasting dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not create a quick baseline training run if MLflow is empty.",
    )
    parser.add_argument(
        "--train-missing-models",
        action="store_true",
        help="Train each missing model experiment once during bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Run migrations/data bootstrap/training and exit without starting the server.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = configure_mlflow()

    bootstrap_project(
        env=env,
        skip_train=args.skip_train,
        train_missing_models=args.train_missing_models,
    )

    if args.bootstrap_only:
        return

    run_step(
        [
            sys.executable,
            "dashboard/manage.py",
            "runserver",
            f"{args.host}:{args.port}",
        ],
        env=env,
    )


if __name__ == "__main__":
    main()
