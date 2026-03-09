import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "dashboard" / "static" / "reports" / "drift_report.html"

def check_data_drift(reference_data_path, current_data_path, report_save_path):
    """
    Checks for data drift between reference (training) and current data using Evidently.
    """
    if not os.path.exists(reference_data_path) or not os.path.exists(current_data_path):
        print("Data files for drift detection not found.")
        return False

    ref_df = pd.read_csv(reference_data_path)
    cur_df = pd.read_csv(current_data_path)
    
    # We only want to compare the feature columns
    features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']
    
    # Create the report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])

    data_drift_report.run(reference_data=ref_df[features], current_data=cur_df[features])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
    
    # Save report as HTML
    data_drift_report.save_html(report_save_path)
    print(f"Drift report saved to {report_save_path}")
    
    # Return drift status (summary)
    return data_drift_report.as_dict()['metrics'][0]['result']['dataset_drift']

if __name__ == "__main__":
    # For demonstration, we'll use the same processed data split in two
    processed_path = DATA_DIR / "btcusd_processed.csv"
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        # Simulate reference and current data
        mid_point = len(df) // 2
        ref = df.iloc[:mid_point]
        cur = df.iloc[mid_point:]
        
        reference_path = DATA_DIR / "reference_data.csv"
        current_path = DATA_DIR / "current_data.csv"
        ref.to_csv(reference_path, index=False)
        cur.to_csv(current_path, index=False)
        
        drift_detected = check_data_drift(
            reference_path,
            current_path,
            REPORT_PATH
        )
        print(f"Drift detected: {drift_detected}")
