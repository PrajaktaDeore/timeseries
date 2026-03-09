import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import warnings
from pathlib import Path
import os

# Set up MLflow
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "btcusd_processed.csv"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
mlflow.set_experiment("BTCUSD_ARIMA_Forecasting")

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

def train_arima_model(p=5, d=1, q=0):
    """
    Trains an ARIMA model on BTCUSD Close price.
    p: Lag order
    d: Degree of differencing
    q: Order of moving average
    """
    # Load processed data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # ARIMA uses univariate time series
    series = df['Close']
    
    # Split data (80% train, 20% test)
    split_point = int(len(series) * 0.8)
    train, test = series[0:split_point], series[split_point:]
    
    with mlflow.start_run(run_name="ARIMA_Model"):
        # Log parameters
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)
        
        # Fit model
        print(f"Training ARIMA({p},{d},{q})...")
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast
        forecast_result = model_fit.forecast(steps=len(test))
        predictions = forecast_result
        
        # Calculate MSE
        mse = mean_squared_error(test, predictions)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        
        # Log model (Note: logging statsmodels objects directly can be tricky with mlflow.sklearn, 
        # so we often save it as a generic python function or artifact, but here we try basic logging)
        # For simplicity in this demo, we'll log the parameters and metrics primarily.
        
        print(f"ARIMA Training complete.")
        print(f"MSE: {mse:.4f}")
        
        return model_fit, mse

if __name__ == "__main__":
    try:
        train_arima_model()
    except Exception as e:
        print(f"Training failed: {e}")
