#Preprocess
from sklearn.preprocessing import OneHotEncoder
import argparse, json, os
import pandas as pd

def basic_quality_checks(df: pd.DataFrame):
    assert len(df) > 0, "Empty DataFrame after preprocessing"

    # Example rule: no nulls in key columns
    for col in ["id"]:
        if col in df.columns:
            assert df[col].notna().all(), f"Nulls found in required column: {col}"
def preprocessing_steps(df, raw_path, out_path):

    # 1) Convert dteday to datetime
    df['dteday'] = pd.to_datetime(df['dteday'], format="%d/%m/%Y")
    # Extract day of month
    df['day_of_month'] = df['dteday'].dt.day
    # Drop raw date 
    df = df.drop(columns=['dteday'])

    # 2) One hot Encoding Categorical Columns
    # One-hot encode categorical columns
    categorical_cols = ['season', 'weathersit', 'mnth', 'weekday']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_df.info()
    # Replace categorical columns
    df = pd.concat([df.drop(columns=categorical_cols).reset_index(drop=True), encoded_df], axis=1)

    # Quality checks
    basic_quality_checks(df)

    # Save
    df.to_csv(out_path, index=False)

    # Emit a tiny metadata file that downstream steps can read
    meta = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "source": raw_path,
        "output": out_path,
    }
    meta_path = out_path.replace(".csv", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved cleaned data to {out_path}")
    print(f"Saved metadata to {meta_path}")
    return df