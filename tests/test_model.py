#Practical
import os
import pandas as pd
import yaml
import subprocess
import joblib
from sklearn.metrics import mean_squared_error
import sys
import numpy as np
# Included so that this test file can read src/preprocessing_pipeline.py
sys.path.append("src/") 
from ML2_Asg_Pipeline import preprocess

def test_preprocess_runs_successfully():
    with open("configs/python-app.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    output_path = cfg["2012_path"]
    # Remove existing output if present
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run the script (same as CI)
    result = subprocess.run(
        ["python", "src/ML_Asg.py", "--config", "configs/python-app.yaml"],
        capture_output=True,
        text=True
    )

    # Check it ran without crashing
    assert result.returncode == 0, f"Preprocess failed: {result.stderr}"

    # Check output file exists
    assert os.path.exists(output_path), "Cleaned CSV not created."

def test_quality_gate():
    # Load saved model
    model = joblib.load("best_model.pkl")

    # Load baseline dataset (2011)
    df_2011 = preprocess("data/day_2011.csv")
    X_2011, y_2011 = df_2011.drop(columns=["cnt"]), df_2011["cnt"]
    baseline_preds = model.predict(X_2011)
    baseline_mse = mean_squared_error(y_2011, baseline_preds)
    baseline_rmse = np.sqrt(baseline_mse)

    # Load evaluation dataset (2012)
    df_2012 = preprocess("data/day_2012.csv")
    X_2012, y_2012 = df_2012.drop(columns=["cnt"]), df_2012["cnt"]
    preds = model.predict(X_2012)
    mse = mean_squared_error(y_2012, preds)
    rmse = np.sqrt(mse)

    # Define threshold (e.g., 5% tolerance)
    threshold = baseline_rmse * 1.05 #did not specify how much to do
    print(f"Baseline RMSE: {baseline_rmse:.2f}, 2012 RMSE: {rmse:.2f}, Threshold: {threshold:.2f}")

    # Quality Gate check
    assert rmse <= threshold, "❌ Quality Gate FAILED: Model performance degraded."
    print("✅ Quality Gate PASSED: Model performance acceptable.")

