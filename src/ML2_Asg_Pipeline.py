import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    if "dteday" in df.columns:
        df = df.drop(columns=["dteday"])
    return df

def add_features(df):
    # Example engineered feature
    if "mnth" in df.columns and "weekday" in df.columns:
        df["day_of_month"] = (df.index % 30) + 1
    return df

def preprocess(path):
    df = load_data(path)
    df = add_features(df)
    return df
