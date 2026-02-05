import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess(filepath):
    df = pd.read_csv(filepath)

    # Convert dteday to datetime
    df['dteday'] = pd.to_datetime(df['dteday'], format="%d/%m/%Y")

    # Extract day of month
    df['day_of_month'] = df['dteday'].dt.day

    # Drop raw date
    df = df.drop(columns=['dteday'])

    # One-hot encode categorical columns
    categorical_cols = ['season', 'weathersit', 'mnth', 'weekday']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Replace categorical columns
    df = pd.concat([df.drop(columns=categorical_cols).reset_index(drop=True), encoded_df], axis=1)

    return df
