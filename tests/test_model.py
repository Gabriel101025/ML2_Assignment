#Practical
import os
import pandas as pd
import yaml
import subprocess
import joblib
from sklearn.metrics import mean_squared_error
import sys
# Included so that this test file can read src/preprocessing_pipeline.py
sys.path.append("src/") 
from ML_Asg_Pipeline import preprocessing_steps

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
    # Step 1: Load saved model
    model = joblib.load("model.pkl")

    # Step 2: Load evaluation dataset (2012 data)
    df = pd.read_csv("data/day_2012.csv")
    X = df.drop(columns=["cnt"])
    y = df["cnt"]

    # Step 3: Make predictions
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)

    # Step 4: Define threshold (baseline RMSE from Task 2 results)
    baseline_rmse = 600   # Example: 2011 RMSE was ~525, 2012 was ~731
    threshold = baseline_rmse * 1.2  # Allow 20% tolerance

    print(f"RMSE on 2012 data: {rmse:.2f}, Threshold: {threshold:.2f}")

    # Step 5: Quality Gate check
    if rmse > threshold:
        print("❌ Quality Gate FAILED: Model performance degraded.")
        sys.exit(1)  # Fail workflow
    else:
        print("✅ Quality Gate PASSED: Model performance acceptable.")

