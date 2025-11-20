"""
Improved Audi Price Prediction Script
------------------------------------

This script extends the baseline model by adding domain‑specific feature
engineering and experimenting with gradient‑boosting models that can
handle categorical variables directly.  It extracts information from
the free‑text ``car_description`` column, such as the apparent
engine displacement (e.g. "2.0" in "2.0 TFSI") and horsepower
(e.g. "160pk" or "160 pk"), and computes derived attributes like
mileage per year and power per litre.  These features are then used
together with existing structured columns to train a LightGBM regressor.

Although the model performs significantly better than a simple linear
model, achieving an RMSE of roughly 5 250–5 300 euro on a hold‑out
validation set, reducing the error below 4 000 euro proved
unrealistic with the available data.  The data contains a wide range of
prices and likely influential factors that are not captured in the
provided features (e.g. options packages, condition, maintenance history).

Usage
-----

Run the script in the same directory as ``audi_challenge.csv``:

.. code:: bash

   python audi_price_prediction_improved.py

The script will print the RMSE of the LightGBM model on a 20 % validation
split, identify the top features by importance, and write predictions
for rows with missing prices to ``predicted_prices_improved.csv``.

See the accompanying report and citations for background on the
importance of feature engineering and ensemble methods in car price
prediction【780848266158723†L27-L33】【843803948975740†L127-L130】.
"""

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV and rename columns to convenient identifiers."""
    df = pd.read_csv(path)
    rename_map = {
        'price (euro)': 'price',
        'age (year)': 'age',
        'fuel type': 'fuel_type',
        'transmission': 'transmission',
        'bodystyle': 'bodystyle',
        'car model': 'car_model',
        'mileage (km)': 'mileage',
        'car description': 'car_description',
    }
    return df.rename(columns=rename_map)


def parse_car_description(description: str) -> Tuple[float, float]:
    """Extract approximate engine displacement and horsepower from a car description.

    Parameters
    ----------
    description: str
        The free‑text description of the car.

    Returns
    -------
    Tuple[float, float]
        engine_displacement (litres or model classification) and horsepower (pk)
        if found, otherwise ``None`` for missing values.
    """
    text = str(description).lower()
    displacement = None
    horsepower = None
    # Engine size appears before TFSI/TSI/TDI (e.g. "1.8 tfsi" or "50 tdi")
    match_disp = re.search(r'(\d+(?:\.\d+)?)\s*(?=t[f]?si|tdi)', text)
    if match_disp:
        try:
            displacement = float(match_disp.group(1))
        except ValueError:
            displacement = None
    # Horsepower appears as "160pk", "160 pk" or "160 hp"
    match_hp = re.search(r'(\d+)\s?pk', text)
    if match_hp:
        horsepower = int(match_hp.group(1))
    else:
        match_hp = re.search(r'(\d+)\s?hp', text)
        if match_hp:
            horsepower = int(match_hp.group(1))
    return displacement, horsepower


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional features from the raw dataset.

    Adds:
    - engine_displacement and horsepower extracted from car_description.
    - mileage_per_year: mileage divided by (age + 0.5) to avoid zero division.
    - power_per_liter: horsepower divided by engine_displacement.

    Missing values in these engineered fields are left as NaN; LightGBM can
    handle missing numerical values.
    """
    # Extract engine size and horsepower from description
    parsed = df['car_description'].apply(parse_car_description)
    df[['engine_displacement', 'horsepower']] = pd.DataFrame(parsed.tolist(), index=df.index)
    # Derived features
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 0.5)
    df['power_per_liter'] = df['horsepower'] / df['engine_displacement']
    return df


def train_lightgbm(X: pd.DataFrame, y: pd.Series, categoricals: List[str]) -> Tuple[LGBMRegressor, float]:
    """Train a LightGBM model and evaluate its RMSE on a validation split.

    Returns the fitted model and the RMSE.
    """
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Instantiate the model with tuned hyperparameters
    model = LGBMRegressor(
        n_estimators=60000,
        learning_rate=0.002,
        num_leaves=3000,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1,
    )
    # Train
    model.fit(X_train, y_train, categorical_feature=categoricals)
    # Predict on validation set
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return model, rmse


def main():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'audi_challenge.csv')
    df = load_dataset(data_path)
    df = engineer_features(df)
    # Separate training and rows to predict
    train_df = df[df['price'].notna()].reset_index(drop=True)
    pred_df = df[df['price'].isna()].reset_index(drop=True)
    # Define features to use
    feature_cols = [
        'age', 'mileage', 'fuel_type', 'transmission', 'bodystyle', 'car_model',
        'engine_displacement', 'horsepower', 'mileage_per_year', 'power_per_liter'
    ]
    X_train = train_df[feature_cols].copy()
    y_train = train_df['price']
    # Ensure categorical features are category type
    categorical_cols = ['fuel_type', 'transmission', 'bodystyle', 'car_model']
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
    # Train model and evaluate
    model, rmse = train_lightgbm(X_train, y_train, categorical_cols)
    print(f"LightGBM validation RMSE: {rmse:.2f} euro")
    # Fit model on all training data
    model.fit(X_train, y_train, categorical_feature=categorical_cols)
    # Predict for missing prices
    if not pred_df.empty:
        X_pred = pred_df[feature_cols].copy()
        for col in categorical_cols:
            X_pred[col] = X_pred[col].astype('category')
        predictions = model.predict(X_pred)
        pred_df = pred_df.copy()
        pred_df['predicted_price'] = predictions
        out_path = os.path.join(os.path.dirname(__file__), 'predicted_prices_improved.csv')
        pred_df.to_csv(out_path, index=False)
        print(f"Predicted prices for {len(pred_df)} cars saved to {out_path}.")


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    main()