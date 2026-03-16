import argparse
import os
from pathlib import Path
import sys

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN
from tensorflow.keras.models import Sequential


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import PROCESSED_BTCUSD_CSV, configure_mlflow

configure_mlflow()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

EXPERIMENT_NAME = "BTCUSD_RNN_Forecasting"
REGISTERED_MODEL_NAME = "BTCUSD_RNN_Model"
FEATURES = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]


def prepare_sequences(df, window_size, horizon_steps):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])

    x_data, y_data = [], []
    max_start = len(scaled) - horizon_steps
    for idx in range(window_size, max_start):
        x_data.append(scaled[idx - window_size : idx])
        y_data.append(scaled[idx + horizon_steps, 0])

    return np.array(x_data), np.array(y_data), scaler


def inverse_close(values, scaler):
    return values * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]


def build_rnn_model(input_shape, units=64, dropout=0.2):
    model = Sequential(
        [
            Input(shape=input_shape),
            SimpleRNN(units=units, return_sequences=True),
            Dropout(dropout),
            SimpleRNN(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def train_rnn(window_size=60, horizon_steps=60, epochs=10, batch_size=32, units=64, dropout=0.2):
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not PROCESSED_BTCUSD_CSV.exists():
        raise FileNotFoundError(f"{PROCESSED_BTCUSD_CSV} not found. Run ingestion first.")

    df = pd.read_csv(PROCESSED_BTCUSD_CSV, index_col=0, parse_dates=True)
    x_data, y_data, scaler = prepare_sequences(
        df=df, window_size=window_size, horizon_steps=horizon_steps
    )
    if len(x_data) < 30:
        raise ValueError("Not enough rows to train RNN. Increase input data size.")

    split_idx = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]

    with mlflow.start_run(run_name="RNN_1H_Baseline"):
        mlflow.log_param("model_type", "RNN")
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("horizon_steps", horizon_steps)
        mlflow.log_param("target_horizon", "next_1_hour")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("units", units)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("features", FEATURES)

        model = build_rnn_model(
            input_shape=(x_train.shape[1], x_train.shape[2]),
            units=units,
            dropout=dropout,
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=1,
        )

        pred_scaled = model.predict(x_test, verbose=0).flatten()
        y_test_actual = inverse_close(y_test, scaler)
        pred_actual = inverse_close(pred_scaled, scaler)

        mse = mean_squared_error(y_test_actual, pred_actual)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test_actual, pred_actual)

        latest_window = x_data[-1:]
        next_1h_scaled = float(model.predict(latest_window, verbose=0).flatten()[0])
        next_1h_pred = float(inverse_close(np.array([next_1h_scaled]), scaler)[0])

        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("predicted_close_1h", next_1h_pred)
        mlflow.log_metric("prediction_1h", next_1h_pred)
        mlflow.log_metric("train_loss_last", float(history.history["loss"][-1]))
        if "val_loss" in history.history:
            mlflow.log_metric("val_loss_last", float(history.history["val_loss"][-1]))

        model_info = mlflow.tensorflow.log_model(
            model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("RNN Training complete.")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        print(f"Next 1-hour predicted Close: {next_1h_pred:.2f}")
        print(f"Model registered as '{REGISTERED_MODEL_NAME}' at: {model_info.model_uri}")
        return model


def main():
    parser = argparse.ArgumentParser(
        description="Train BTCUSD SimpleRNN model and predict next 1-hour close."
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    train_rnn(
        window_size=args.window_size,
        horizon_steps=args.horizon_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}")
