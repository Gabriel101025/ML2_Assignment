import os
import pandas as pd
import yaml
import subprocess
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys
# Included so that this test file can read src/ML2_Asg_Pipeline.py
sys.path.append("src/") 
from ML2_Asg_Pipeline import preprocess


def test_quality_gate():
    model = joblib.load("best_model.pkl")

    # Preprocess both datasets
    df_2011 = preprocess("data/day_2011.csv")
    df_2012 = preprocess("data/day_2012.csv")

    # Evaluate on 2011
    X_2011 = df_2011.drop(columns=["cnt"])
    y_2011 = df_2011["cnt"].values
    X_train_2011, X_test_2011, y_train_2011, y_test_2011 = train_test_split(X_2011, y_2011, test_size=0.2, random_state=42)
    model.fit(X_train_2011, y_train_2011)
    rf_preds_2011 = model.predict(X_test_2011)
    rf_rmse_2011 = np.sqrt(mean_squared_error(y_test_2011, rf_preds_2011))
    rf_mae_2011 = mean_absolute_error(y_test_2011, rf_preds_2011)
    rf_r2_2011 = r2_score(y_test_2011, rf_preds_2011)

    # Evaluate on 2012
    X_2012 = df_2012.drop(columns=['cnt']) 
    y_2012 = df_2012['cnt']
    X_train_2012, X_test_2012, y_train_2012, y_test_2012 = train_test_split(X_2012, y_2012, test_size=0.2, random_state=42)
    model.fit(X_train_2012, y_train_2012)
    rf_preds_2012 = model.predict(X_test_2012)
    rf_rmse_2012 = np.sqrt(mean_squared_error(y_test_2012, rf_preds_2012))
    rf_mae_2012 = mean_absolute_error(y_test_2012, rf_preds_2012)
    rf_r2_2012 = r2_score(y_test_2012, rf_preds_2012)


    #Compare
    print("2011 Random Forest:", rf_rmse_2011, rf_mae_2011, rf_r2_2011)
    print("2012 Random Forest:", rf_rmse_2012, rf_mae_2012, rf_r2_2012)

    threshold = rf_rmse_2011 * 1.2
    print(f"Baseline RMSE: {rf_rmse_2011:.2f}, 2012 RMSE: {rf_rmse_2012:.2f}, Threshold: {threshold:.2f}")

    assert rf_rmse_2012 <= threshold, f"Quality Gate FAILED: RMSE {rf_rmse_2012:.2f} > {threshold:.2f}"


