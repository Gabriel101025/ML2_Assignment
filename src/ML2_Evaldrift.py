import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ML2_Asg_Pipeline import preprocess

def main():
    model = joblib.load("model.pkl")

    df_2011 = preprocess("data/day_2011.csv")
    df_2012 = preprocess("data/day_2012.csv")

    # Evaluate on 2011
    X_2011 = df_2011.drop(columns=["cnt"])
    y_2011 = df_2011["cnt"]
    preds_2011 = model.predict(X_2011)
    rmse_2011 = mean_squared_error(y_2011, preds_2011, squared=False)

    # Evaluate on 2012
    X_2012 = df_2012.drop(columns=["cnt"])
    y_2012 = df_2012["cnt"]
    preds_2012 = model.predict(X_2012)
    rmse_2012 = mean_squared_error(y_2012, preds_2012, squared=False)

    print(f"2011 RMSE: {rmse_2011:.2f}")
    print(f"2012 RMSE: {rmse_2012:.2f}")

if __name__ == "__main__":
    main()
