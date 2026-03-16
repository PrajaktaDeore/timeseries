import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import PROCESSED_BTCUSD_CSV, configure_mlflow

configure_mlflow()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Set up MLflow
mlflow.set_experiment("BTCUSD_Linear_Regression")

def train_linear_regression(test_size=0.2):
    """
    Trains a Linear Regression model to predict the next day's Close price.
    Uses 'Close', 'MA7', 'MA21', 'Daily_Return', 'Volume' as features.
    """
    # Load processed data
    if not PROCESSED_BTCUSD_CSV.exists():
        print(f"Error: {PROCESSED_BTCUSD_CSV} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(PROCESSED_BTCUSD_CSV, index_col=0, parse_dates=True)
    
    # Feature Engineering for Linear Regression
    # We want to predict 'Close' price of the NEXT day using TODAY's features
    # df['Target'] = df['Close'].shift(-1)

#if want data hourly
    df['Target'] = df['Close'].shift(-60)
    df = df.dropna()
    
    features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']
    X = df[features]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    with mlflow.start_run(run_name="Linear_Regression_Baseline"):
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features", features)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        latest_features = X.tail(1)
        next_prediction = float(model.predict(latest_features)[0])
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("predicted_close", next_prediction)
        mlflow.log_metric("predicted_close_1h", next_prediction)
        mlflow.log_metric("prediction_1h", next_prediction)
        
        # Log model with registry
        model_info = mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="BTCUSD_Linear_Regression"
        )
        
        print(f"Linear Regression Training complete.")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"Next predicted Close: {next_prediction:.2f}")
        print(f"Model registered as 'BTCUSD_Linear_Regression' at: {model_info.model_uri}")
        
        return model, mse, mae

if __name__ == "__main__":
    try:
        train_linear_regression()
    except Exception as e:
        print(f"Training failed: {e}")
