"""
Student Performance Prediction — ML Training Module
====================================================
Loads the UCI Student Performance (Mathematics) dataset, trains a
Linear Regression model, and exposes helper functions consumed by the
Streamlit front-end.

Dataset : student-mat.csv  (semicolon-separated)
Features: studytime, failures, absences, G1, G2
Target  : G3 (final grade, 0–20)
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["studytime", "failures", "absences", "G1", "G2"]
TARGET_COL = "G3"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load *student-mat.csv* and return the full DataFrame.

    Parameters
    ----------
    csv_path : str, optional
        Explicit path to the CSV file.  When *None* the function looks in
        ``data/student-mat.csv`` relative to the project root (one level
        above this module's directory).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV cannot be located.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "student-mat.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Please download 'student-mat.csv' from the UCI Machine Learning "
            "Repository and place it inside the 'data/' folder."
        )

    df = pd.read_csv(csv_path, sep=";")

    # Validate that the required columns exist
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from the dataset: {missing}"
        )

    return df

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """Train a Linear Regression model and return everything the app needs.

    Returns
    -------
    dict with keys:
        model        – fitted LinearRegression instance
        X_train      – training features
        X_test       – testing features
        y_train      – training target
        y_test       – testing target
        y_pred       – predictions on the test set
        mae          – Mean Absolute Error on the test set
        r2           – R² Score on the test set
        feature_cols – list of feature column names
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "r2": r2,
        "feature_cols": FEATURE_COLS,
    }

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_grade(model, studytime: int, failures: int, absences: int,
                  g1: int, g2: int) -> float:
    """Return the predicted G3 grade (clipped to 0–20).

    Parameters
    ----------
    model : LinearRegression
        Already-fitted model returned by :func:`train_model`.
    studytime, failures, absences, g1, g2 : int
        Student input features.

    Returns
    -------
    float
        Predicted final grade rounded to two decimal places.
    """
    features = pd.DataFrame(
        [[studytime, failures, absences, g1, g2]], columns=FEATURE_COLS
    )
    prediction = model.predict(features)[0]
    prediction = float(np.clip(prediction, 0, 20))
    return round(prediction, 2)
