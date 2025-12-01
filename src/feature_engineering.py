"""Feature engineering and selection utilities.

- Feature scaling
- Lasso and RandomForest based selection
- Save selected feature list
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_processed(path=None):
    if path is None:
        path = os.path.join(PROCESSED_DIR, "secom_processed.csv")
    return pd.read_csv(path)


def scale_features(X, scaler_path=None):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return Xs, scaler


def select_features_lasso_rf(X, y, top_k=30):
    # Lasso selection
    lasso = LassoCV(cv=5, n_jobs=-1)
    lasso.fit(X, y)
    lasso_coefs = np.abs(lasso.coef_)

    # RandomForest selection
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    # combine rankings
    combined = lasso_coefs + rf_imp
    idx = np.argsort(combined)[::-1][:top_k]
    return idx, combined


if __name__ == "__main__":
    df = load_processed()
    # assume last column is label if named 'class' or similar
    if "class" in df.columns:
        y = df["class"].values
        X = df.drop(columns=["class"]) 
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    Xnum = X[numeric_cols].fillna(0).values
    Xs, scaler = scale_features(Xnum)
    idx, combined = select_features_lasso_rf(Xs, y, top_k=30)
    selected_cols = numeric_cols[idx]
    print("Selected features:", selected_cols.tolist())
