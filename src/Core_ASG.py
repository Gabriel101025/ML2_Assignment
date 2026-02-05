#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris, make_classification
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("task 1 Model Development and Experiment Design")

#Task 1

df = pd.read_csv("day_2011.csv")
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
# Separate features and target 
X = df.drop(columns=['cnt']) 
y = df['cnt']
mlflow.set_experiment("task 1 Model Development and Experiment Design")
#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Defining models
from sklearn.linear_model import LinearRegression, Ridge
lr_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
#Train and eval models with RMSE, MAE and R square
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate_model(model, X_train, X_test, y_train, y_test): 
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2
# Evaluate both linear regression and ridge regression
lr_rmse, lr_mae, lr_r2 = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
ridge_rmse, ridge_mae, ridge_r2 = evaluate_model(ridge_model, X_train, X_test, y_train, y_test)
import matplotlib.pyplot as plt
preds = lr_model.predict(X_test)
residuals = y_test - preds
preds = ridge.predict(X_test)
residuals = y_test - preds
#Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)
preds = rf.predict(X_test)
residuals = y_test - preds

#task 2

#Load new dataset
df_2012 = pd.read_csv("day_2012.csv")
# Convert dteday to datetime
df_2012['dteday'] = pd.to_datetime(df_2012['dteday'], format="%d/%m/%Y")
# Extract day of month
df_2012['day_of_month'] = df_2012['dteday'].dt.day
# Drop raw date 
df_2012 = df_2012.drop(columns=['dteday'])
# One-hot encode categorical columns
encoded_2012 = encoder.fit_transform(df_2012[categorical_cols])
encoded_df_2012 = pd.DataFrame(encoded_2012, columns=encoder.get_feature_names_out(categorical_cols))
encoded_df_2012.info()
# Replace categorical columns
df_2012 = pd.concat([df_2012.drop(columns=categorical_cols).reset_index(drop=True), encoded_df_2012], axis=1)
#Evaluate model on drifted data
# Separate features and target 
X_2012 = df_2012.drop(columns=['cnt']) 
y_2012 = df_2012['cnt']
#Train test split
X_train_2012, X_test_2012, y_train_2012, y_test_2012 = train_test_split(X_2012, y_2012, test_size=0.2, random_state=42)
rf.fit(X_train_2012, y_train_2012)
rf_preds_2012 = rf.predict(X_test_2012)
rf_rmse_2012 = np.sqrt(mean_squared_error(y_test_2012, rf_preds_2012))
rf_mae_2012 = mean_absolute_error(y_test_2012, rf_preds_2012)
rf_r2_2012 = r2_score(y_test_2012, rf_preds_2012)

