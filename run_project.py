import argparse
import os
import subprocess
import sys

import mlflow
from mlflow.tracking import MlflowClient

from project_config import PROCESSED_BTCUSD_CSV, configure_mlflow


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
        "--train-script",
        default="src/models/linear_regression.py",
        help="Training script used for initial bootstrap.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = configure_mlflow()

    run_step([sys.executable, "dashboard/manage.py", "migrate"], env=env)

    if not PROCESSED_BTCUSD_CSV.exists():
        run_step([sys.executable, "src/data/ingestion.py"], env=env)

    if not args.skip_train and not has_any_runs():
        run_step([sys.executable, args.train_script], env=env)

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