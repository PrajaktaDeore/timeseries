from pathlib import Path
import os
import platform
import re
import shutil
import sqlite3
from datetime import datetime
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_BTCUSD_CSV = RAW_DATA_DIR / "btc-usd_historical.csv"
PROCESSED_BTCUSD_CSV = PROCESSED_DATA_DIR / "btcusd_processed.csv"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_RUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_ARTIFACT_ROOT = MLFLOW_RUNS_DIR.resolve().as_uri()
DEFAULT_EXPERIMENTS = [
    "BTCUSD_Forecasting",
    "BTCUSD_Linear_Regression",
    "BTCUSD_ARIMA_Forecasting",
    "BTCUSD_LSTM_Forecasting",
    "BTCUSD_RNN_Forecasting",
]
DEFAULT_REGISTERED_MODELS = [
    "BTCUSD_RNN_Model",
    "BTCUSD_Linear_Regression",
    "BTCUSD_LSTM_Model",
]


def ensure_project_root_on_path():
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_WINDOWS_ARTIFACT_RE = re.compile(r"file:(?:/{2,3})?[A-Za-z]:")


def _mlflow_db_has_windows_artifact_locations(db_path: Path) -> bool:
    if not db_path.exists():
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute("SELECT artifact_location FROM experiments")
            for (artifact_location,) in cursor.fetchall():
                if not artifact_location:
                    continue
                if _WINDOWS_ARTIFACT_RE.search(str(artifact_location)):
                    return True
            return False
        finally:
            conn.close()
    except Exception:
        return False


def ensure_mlflow_store():
    """
    Ensures MLflow writes artifacts to a sane location on the current OS.

    This project ships a pre-created `mlflow.db` from Windows sometimes. That DB can contain
    `artifact_location = file:///C:/...` which breaks on Linux (tries to write to `/C:`).
    On non-Windows hosts, detect this and rotate the DB so a fresh store is created.
    """
    MLFLOW_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if platform.system().lower() != "windows" and _mlflow_db_has_windows_artifact_locations(MLFLOW_DB_PATH):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = PROJECT_ROOT / f"mlflow.windows-backup.{timestamp}.db"
        shutil.move(str(MLFLOW_DB_PATH), str(backup_path))


def configure_mlflow():
    ensure_project_root_on_path()
    ensure_mlflow_store()

    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        client = MlflowClient()
        for experiment_name in DEFAULT_EXPERIMENTS:
            if client.get_experiment_by_name(experiment_name) is None:
                client.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACT_ROOT)
    except Exception:
        # Keep configuration best-effort; the caller may only need the URI string.
        pass

    return os.environ["MLFLOW_TRACKING_URI"]
