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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "btcusd_processed.csv"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

# Set up MLflow
mlflow.set_experiment("BTCUSD_Linear_Regression")

def train_linear_regression(test_size=0.2):
    """
    Trains a Linear Regression model to predict the next day's Close price.
    Uses 'Close', 'MA7', 'MA21', 'Daily_Return', 'Volume' as features.
    """
    # Load processed data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
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
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        
        # Log model with registry
        model_info = mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="BTCUSD_Linear_Regression"
        )
        
        print(f"Linear Regression Training complete.")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"Model registered as 'BTCUSD_Linear_Regression' at: {model_info.model_uri}")
        
        return model, mse, mae

if __name__ == "__main__":
    try:
        train_linear_regression()
    except Exception as e:
        print(f"Training failed: {e}")
