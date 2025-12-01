"""ETL pipeline for UCI SECOM dataset with Murata-inspired augmentations.

- Loads raw `uci-secom.csv` from data/raw
- Adds Murata-style categorical columns
- Renames a selection of sensor features to business-relevant names
- Handles missing values (interpolation/median)
- Writes cleaned CSVs into data/processed
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

RAW_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "uci-secom.csv")
# If raw path doesn't exist (your CSV may be at repo root), fall back to repository root file
if not os.path.exists(RAW_PATH):
    alt = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "uci-secom.csv"))
    if os.path.exists(alt):
        RAW_PATH = alt

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def augment_murata_columns(df, random_state=42):
    np.random.seed(random_state)
    n = len(df)
    df = df.copy()
    df["plant_location"] = np.random.choice(["Plant_A", "Plant_B", "Plant_C"], size=n)
    df["product_family"] = np.random.choice([
        "Capacitor_MLCC",
        "Inductor_Coil",
        "SAW_Filter",
        "Resistor",
    ], size=n)
    df["shift"] = np.random.choice(["Day", "Night", "Weekend"], size=n)
    df["batch_id"] = np.random.randint(1, 101, size=n)
    return df


def rename_sensors(df, user_map=None):
    """Rename 3-5 numeric columns to business names. If user_map provided, use it. Otherwise auto-select numeric columns.

    Returns df, rename_map
    """
    df = df.copy()

    business_names = [
        "air_temperature",
        "water_temperature",
        "particle_count",
        "humidity",
        "pressure",
    ]

    if user_map:
        rename_map = user_map
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # exclude label if present (assuming last column is label 'Pass/Fail' or 'class')
        if "class" in numeric_cols:
            numeric_cols.remove("class")
        if "passfail" in numeric_cols:
            numeric_cols.remove("passfail")

        # choose up to 5 columns from numeric_cols
        selected = numeric_cols[: min(len(numeric_cols), len(business_names))]
        rename_map = {orig: new for orig, new in zip(selected, business_names)}

    df = df.rename(columns=rename_map)
    return df, rename_map


def impute_missing(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Try linear interpolation for numeric
    try:
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    except Exception:
        pass
    # Then median impute any remaining
    imp = SimpleImputer(strategy="median")
    df[numeric_cols] = imp.fit_transform(df[numeric_cols])
    return df


def create_summary_metrics(df):
    df = df.copy()
    total = len(df)
    
    # Get the Pass/Fail column (should be last now)
    if "Pass/Fail" in df.columns:
        # Real UCI SECOM data: -1 = fail, 1 = pass
        labels = df["Pass/Fail"].fillna(0).astype(int)
        # Map: -1 (fail) → 1, 1 (pass) → 0 for counting
        fails = (labels == -1).sum()
        passes = (labels == 1).sum()
    elif "class" in df.columns:
        labels = df["class"].fillna(0).astype(int)
        fails = (labels == -1).sum()
        passes = (labels == 1).sum()
    elif "Status" in df.columns:
        labels = df["Status"].fillna(0).astype(int)
        fails = (labels == 1).sum()
        passes = (labels == 0).sum()
    else:
        # Fallback: assume last column is label
        labels = df[df.columns[-1]].fillna(0).astype(int)
        fails = (labels < 0).sum()
        passes = total - fails

    yield_rate = passes / total * 100
    dpm = fails / total * 1_000_000
    return {"total": total, "passes": passes, "fails": fails, "yield_rate": yield_rate, "dpm": dpm}


def save_processed(df, filename="secom_processed.csv"):
    # Reorder columns: move Pass/Fail to end (label should be last)
    df = df.copy()
    if "Pass/Fail" in df.columns:
        pass_fail = df.pop("Pass/Fail")
        df["Pass/Fail"] = pass_fail
    
    out = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = load_data()
    df = augment_murata_columns(df)
    df, rename_map = rename_sensors(df)
    df = impute_missing(df)
    metrics = create_summary_metrics(df)
    out = save_processed(df)
    print(f"Saved processed CSV to {out}")
    print("Rename map:", rename_map)
    print("Summary metrics:", metrics)
    print(f"  ✓ Total units: {metrics['total']}")
    print(f"  ✓ Pass: {metrics['passes']} ({metrics['yield_rate']:.1f}%)")
    print(f"  ✓ Fail: {metrics['fails']} (DPM: {metrics['dpm']:.0f})")
