"""
Audi Price Prediction Script
----------------------------------

This script reads the provided ``audi_challenge.csv`` dataset and trains
several machine‑learning models to predict missing car prices.  It performs
feature engineering on categorical and numerical columns, evaluates a
variety of regression algorithms using a hold‑out validation set, and
reports the root mean square error (RMSE) for each model.  Finally, the
best model is refit on all available training data and used to predict
prices for records where the ``price (euro)`` column is missing.

Models considered:

* Linear Regression with Ridge regularisation
* Random Forest Regressor
* Gradient Boosting Regressor
* Extreme Gradient Boosting (XGBRegressor)

The script outputs a summary of the RMSE for each model and writes the
predicted prices for missing entries back to disk as ``predicted_prices.csv``.

Usage
-----

Run the script from the command line:

.. code:: bash

   python audi_price_prediction.py

"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
    _has_xgb = True
except Exception:
    # XGBoost may not be available; in that case we fall back to None
    XGBRegressor = None  # type: ignore
    _has_xgb = False


def load_data(path: str) -> pd.DataFrame:
    """Load the Audi dataset from a CSV file.

    Parameters
    ----------
    path: str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training (non‑missing price) and prediction (missing price).

    Parameters
    ----------
    df: pd.DataFrame
        The raw data.

    Returns
    -------
    train_df: pd.DataFrame
        Rows where the target price is present.
    pred_df: pd.DataFrame
        Rows where the target price is missing and must be predicted.
    """
    df = df.copy()
    # rename columns for easier handling
    df = df.rename(columns={
        "price (euro)": "price",
        "age (year)": "age",
        "fuel type": "fuel_type",
        "transmission": "transmission",
        "bodystyle": "bodystyle",
        "car model": "car_model",
        "mileage (km)": "mileage",
        "car description": "car_description"
    })

    train_df = df[df["price"].notna()].reset_index(drop=True)
    pred_df = df[df["price"].isna()].reset_index(drop=True)
    return train_df, pred_df


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """Build a preprocessing pipeline for numeric and categorical features.

    Parameters
    ----------
    df: pd.DataFrame
        The data containing features.

    Returns
    -------
    preprocessor: ColumnTransformer
        Transformer for preprocessing numeric and categorical columns.
    numeric_features: list
        Names of numeric columns.
    categorical_features: list
        Names of categorical columns.
    """
    # For this task we ignore the free‑text 'car_description' to keep the
    # feature space manageable.  See documentation for an explanation.
    features = df.drop(columns=["price", "car_description", "index"], errors="ignore")

    # Determine numeric and categorical columns
    numeric_features = [col for col in features.columns if features[col].dtype != object]
    categorical_features = [col for col in features.columns if features[col].dtype == object]

    # Preprocessing for numeric features: impute missing values and scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical features: impute missing values and one‑hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Dict[str, float]:
    """Train and evaluate multiple regression models.

    Parameters
    ----------
    X_train: pd.DataFrame
        Training features.
    y_train: pd.Series
        Target prices.
    preprocessor: ColumnTransformer
        The preprocessing pipeline to apply to the features.

    Returns
    -------
    Dict[str, float]
        Mapping of model names to their RMSE on the validation set.
    """
    # Split training data into train/validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    results: Dict[str, float] = {}

    # Define models to evaluate
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_split=2, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    if _has_xgb:
        # Use a modest configuration for XGBoost to avoid excessive training time
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    for name, model in models.items():
        # Create a pipeline that applies the preprocessor then the model
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        # Fit the model
        pipeline.fit(X_tr, y_tr)
        # Predict on validation set
        y_pred = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results[name] = rmse

    return results


def train_best_model(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, best_model_name: str):
    """Train the best model on the full training data.

    Parameters
    ----------
    X: pd.DataFrame
        Features.
    y: pd.Series
        Target prices.
    preprocessor: ColumnTransformer
        Preprocessing pipeline.
    best_model_name: str
        Name of the best performing model.

    Returns
    -------
    Pipeline
        The fitted pipeline ready for prediction.
    """
    model_map = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=3000, max_depth=None, min_samples_split=2, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    if _has_xgb:
        model_map["XGBRegressor"] = XGBRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    best_model = model_map[best_model_name]
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", best_model),
    ])
    pipeline.fit(X, y)
    return pipeline


def main():
    data_path = os.path.join(os.path.dirname(__file__), "audi_challenge.csv")
    df = load_data(data_path)
    train_df, pred_df = preprocess_data(df)

    # Separate features and target
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]

    # Build preprocessor
    preprocessor, numeric_features, categorical_features = build_preprocessor(train_df)
    # Evaluate models on validation split
    results = evaluate_models(X_train, y_train, preprocessor)

    # Print evaluation results
    print("Model evaluation results (RMSE on validation set):")
    for model_name, rmse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {model_name:15s}: {rmse:.2f}")

    # Choose the best model based on RMSE
    best_model_name = min(results, key=results.get)
    print(f"\nBest model based on validation RMSE: {best_model_name}")

    # Train the best model on the full training data
    pipeline = train_best_model(X_train, y_train, preprocessor, best_model_name)

    # Predict missing prices
    if not pred_df.empty:
        X_pred = pred_df.drop(columns=["price"])
        pred_prices = pipeline.predict(X_pred)
        pred_df = pred_df.copy()
        pred_df["predicted_price"] = pred_prices
        # Write predictions to CSV
        output_path = os.path.join(os.path.dirname(__file__), "predicted_prices.csv")
        pred_df.to_csv(output_path, index=False)
        print(f"\nPredictions for {len(pred_df)} records with missing prices have been saved to {output_path}.")
    else:
        print("No missing prices found in the dataset. No predictions to make.")


if __name__ == "__main__":
    main()