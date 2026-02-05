import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ML2_Asg_Pipeline import preprocess
import numpy as np

def main():
    model = joblib.load("model.pkl")
    #Set datasets
    df_2011 = preprocess("data/day_2011.csv")
    df_2012 = preprocess("data/day_2012.csv")

    # Evaluate on 2011
    X_2011 = df_2011.drop(columns=["cnt"])
    y_2011 = df_2011["cnt"].values
    preds_2011 = model.predict(X_2011)
    mse_2011 = mean_squared_error(y_2011, preds_2011)
    rmse_2011 = np.sqrt(mse_2011)

    # Evaluate on 2012
    X_2012 = df_2012.drop(columns=["cnt"])
    y_2012 = df_2012["cnt"].values
    preds_2012 = model.predict(X_2012)
    mse_2012 = mean_squared_error(y_2012, preds_2012)
    rmse_2012 = np.sqrt(mse_2012)
    #Compare
    print(f"2011 RMSE: {rmse_2011:.2f}")
    print(f"2012 RMSE: {rmse_2012:.2f}")

if __name__ == "__main__":
    main()







