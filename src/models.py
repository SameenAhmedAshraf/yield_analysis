"""Modeling: anomaly detection and classification."""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_processed(path=None):
    if path is None:
        path = os.path.join(PROCESSED_DIR, "secom_processed.csv")
    return pd.read_csv(path)


def anomaly_detection(X, contamination=0.05):
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)
    # -1 anomaly, 1 normal
    return preds, iso


def train_classifier(X, y, use_smote=True):
    # ensure y is 1d array
    y = np.asarray(y)

    # determine if labels are numeric; if not, encode
    if y.dtype.kind not in "biufc":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    n_classes = len(np.unique(y_train))
    if use_smote and n_classes == 2:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif use_smote and n_classes > 2:
        # SMOTE for multiclass can be used but may require strategy; skip by default to avoid issues
        use_smote = False

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba_raw = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    # choose averaging strategy based on number of classes
    if n_classes == 2:
        avg = 'binary'
    else:
        avg = 'macro'

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)

    # compute AUC appropriately
    auc = None
    if y_proba_raw is not None:
        try:
            if n_classes == 2:
                # take probability of positive class
                y_score = y_proba_raw[:, 1]
                auc = roc_auc_score(y_test, y_score)
            else:
                # multiclass: pass full probability matrix
                auc = roc_auc_score(y_test, y_proba_raw, multi_class='ovr', average='macro')
        except Exception:
            auc = None

    return clf, {"precision": p, "recall": r, "f1": f, "auc": auc}, (X_test, y_test, y_pred, y_proba_raw)


if __name__ == "__main__":
    df = load_processed()
    if "class" in df.columns:
        y = df["class"].values
        X = df.drop(columns=["class"]) 
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]

    # encode non-numeric labels to integers
    if y.dtype.kind not in "biufc":
        le_main = LabelEncoder()
        y = le_main.fit_transform(y)
    else:
        # ensure integers
        try:
            y = y.astype(int)
        except Exception:
            pass

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    Xnum = X[numeric_cols].fillna(0).values
    preds, iso = anomaly_detection(Xnum)
    clf, metrics, test_data = train_classifier(Xnum, y)
    
    # Save with pickle (lighter than joblib)
    model_path = os.path.join(PROCESSED_DIR, "rf_model.pkl")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(clf, f, protocol=2)
        print(f"Model saved to {model_path}")
    except OSError as e:
        print(f"Warning: Could not save model due to disk space: {e}")
    
    print("Metrics:", metrics)
