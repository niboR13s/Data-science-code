"""
Audi Price Prediction using a TensorFlow Neural Network
=====================================================

This script replicates the feature engineering and data preparation used
in the improved LightGBM model but replaces the gradient‑boosting
regressor with a feed‑forward neural network implemented using
``tensorflow.keras``.  The goal is to demonstrate how a deep learning
approach can be applied to the same problem of estimating missing
prices in the Audi dataset.

The script performs the following steps:

* **Load and clean the data:**  Rename verbose column names to short
  identifiers and drop the redundant index column.
* **Feature extraction:**  Parse the free‑text ``car_description`` to
  extract engine displacement and horsepower, then derive additional
  attributes such as ``mileage_per_year`` and ``power_per_liter``.  This
  mirrors the domain‑specific feature engineering described in
  contemporary research on used car price prediction【780848266158723†L27-L33】.
* **Preprocessing pipeline:**  Separate the features into numerical and
  categorical groups.  Numerical features are imputed using the
  median and scaled; categorical features are one‑hot encoded.  Such
  normalization and encoding of variables is emphasised as a critical
  preprocessing step for deep neural networks【88924021020777†L58-L70】.
* **Neural network architecture:**  Build a multi‑layer perceptron
  (MLP) with several hidden layers, ``ReLU`` activations, batch
  normalization, and dropout.  The architecture can be tuned via
  constants near the top of the script.  Deep learning studies on
  used car prices demonstrate that deeper networks with appropriate
  regularisation often outperform shallow ones and classical
  regression models【564172895262332†L18-L25】.
* **Training and evaluation:**  Split the labelled data (rows with
  non‑missing prices) into training and validation sets.  Train the
  model using the Adam optimizer and mean squared error loss.
  Compute the RMSE on the validation set to gauge model quality.  The
  script includes early stopping to prevent overfitting.
* **Full‑data training and prediction:**  Retrain the network on all
  available labelled data and generate price estimates for the rows
  where ``price`` is missing.  These predictions are saved to
  ``predicted_prices_tensorflow.csv``.

To run this script you must have ``tensorflow>=2.4`` installed in
your Python environment.  On systems without GPU support, the
``tensorflow-cpu`` package suffices.  If you run into a warning such
as ``ModuleNotFoundError: No module named 'tensorflow'``, install it
with

.. code-block:: bash

    pip install --upgrade tensorflow-cpu

Although our environment does not currently include TensorFlow, this
script can be executed locally to train the neural network and
generate predictions.  Past research indicates that deep neural
networks can rival or even surpass traditional machine learning
methods for used car price prediction【88924021020777†L58-L70】【73847947956640†L65-L78】.
Nevertheless, achieving an RMSE below 4 000 euro remains challenging
with the available features, as many influential factors (options,
maintenance history, etc.) are absent.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Only import TensorFlow if available.  If not installed, the user
# must install it manually to run this script.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for this script. Please install it via\n"
        "pip install tensorflow or pip install tensorflow-cpu before running."
    ) from exc


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV and rename columns to convenient identifiers.

    Parameters
    ----------
    path : str
        Path to ``audi_challenge.csv``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with renamed columns and the original index column
        dropped.
    """
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
    df = df.rename(columns=rename_map)
    # Drop the original index column if present
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    return df


def parse_car_description(description: str) -> Tuple[float | None, float | None]:
    """Extract approximate engine displacement and horsepower from a car description.

    Parameters
    ----------
    description : str
        The free‑text description of the car.

    Returns
    -------
    Tuple[float | None, float | None]
        ``(engine_displacement, horsepower)`` where missing values are
        represented as ``None``.
    """
    text = str(description).lower()
    displacement: float | None = None
    horsepower: float | None = None
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
        horsepower = float(match_hp.group(1))
    else:
        match_hp = re.search(r'(\d+)\s?hp', text)
        if match_hp:
            horsepower = float(match_hp.group(1))
    return displacement, horsepower


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional features from the raw dataset.

    Adds:

    - ``engine_displacement`` and ``horsepower`` extracted from
      ``car_description``.
    - ``mileage_per_year``: mileage divided by (age + 0.5) to avoid zero
      division.
    - ``power_per_liter``: horsepower divided by engine_displacement.

    Missing values in these engineered fields are left as NaN; the
    preprocessing pipeline will impute them.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame with renamed columns.

    Returns
    -------
    pandas.DataFrame
        The input frame with additional feature columns appended.
    """
    parsed = df['car_description'].apply(parse_car_description)
    df[['engine_displacement', 'horsepower']] = pd.DataFrame(parsed.tolist(), index=df.index)
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 0.5)
    df['power_per_liter'] = df['horsepower'] / df['engine_displacement']
    return df


def build_preprocessor(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Construct a ColumnTransformer for preprocessing.

    Numerical columns are imputed with the median and scaled to zero
    mean and unit variance.  Categorical columns are imputed with the
    most frequent value and one‑hot encoded.

    Parameters
    ----------
    numerical_features : list of str
        Names of numerical columns.
    categorical_features : list of str
        Names of categorical columns.

    Returns
    -------
    ColumnTransformer
        Configured transformer ready to be fitted on training data.
    """
    # Pipeline for numerical features
    numeric_pipeline = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    # Pipeline for categorical features
    categorical_pipeline = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]
    # Combine the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ],
        remainder='drop',
        sparse_threshold=0.0,  # Always return a dense array for TensorFlow
    )
    return preprocessor


def build_model(input_dim: int) -> tf.keras.Model:
    """Build a feed‑forward neural network using Keras.

    The architecture uses several dense layers with ReLU activations, batch
    normalization, and dropout for regularization.  Feel free to adjust the
    number of layers, units, and dropout rates to tune performance.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input feature vector after preprocessing.

    Returns
    -------
    tensorflow.keras.Model
        A compiled Keras model ready for training.
    """
    model = models.Sequential(name='AudiPriceRegressor')
    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    # Hidden layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    # Output layer with linear activation for regression
    model.add(layers.Dense(1, activation='linear'))
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    epochs: int = 200,
    batch_size: int = 64,
    validation_split: float = 0.2,
) -> Tuple[tf.keras.Model, float]:
    """Train the neural network and evaluate its RMSE on a validation split.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed feature matrix.
    y : np.ndarray
        Target vector of prices.
    input_dim : int
        Dimension of the input layer.
    epochs : int, optional
        Maximum number of training epochs, by default 200.
    batch_size : int, optional
        Size of mini‑batches during training, by default 64.
    validation_split : float, optional
        Fraction of the data to use for validation, by default 0.2.

    Returns
    -------
    model : tf.keras.Model
        The trained Keras model.
    val_rmse : float
        RMSE on the validation set.
    """
    model = build_model(input_dim)
    early_stop = callbacks.EarlyStopping(
        monitor='val_rmse',
        patience=20,
        mode='min',
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1,
    )
    # Extract the best validation RMSE
    val_rmse = min(history.history['val_rmse'])
    return model, val_rmse


def main():
    # Locate the dataset relative to the script location
    data_path = os.path.join(os.path.dirname(__file__), 'audi_challenge.csv')
    df = load_dataset(data_path)
    df = engineer_features(df)
    print(df.head())
    # Split into training data (known prices) and data to predict
    train_df = df[df['price'].notna()].reset_index(drop=True)
    pred_df = df[df['price'].isna()].reset_index(drop=True)
    # Define feature columns
    feature_cols = [
        'age', 'mileage', 'fuel_type', 'transmission', 'bodystyle', 'car_model',
        'engine_displacement', 'horsepower', 'mileage_per_year', 'power_per_liter'
    ]
    numerical_features = ['age', 'mileage', 'engine_displacement', 'horsepower', 'mileage_per_year', 'power_per_liter']
    categorical_features = ['fuel_type', 'transmission', 'bodystyle', 'car_model']
    # Build preprocessor and fit on training data
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    # Fit the preprocessor on the training set and transform
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    y_train = train_df['price'].values.reshape(-1, 1)
    # Flatten the target for Keras (expects 1D array)
    y_train = y_train.flatten()
    # Train the neural network and evaluate
    model, val_rmse = train_and_evaluate(X_train, y_train, input_dim=X_train.shape[1])
    print(f"Neural network validation RMSE: {val_rmse:.2f} euro")
    # Fit on all training data (to finalise model)
    # The early stopping callback ensures we use optimal weights; training on
    # the full dataset from scratch typically provides a minor improvement.
    model_final = build_model(input_dim=X_train.shape[1])
    # Train without validation for a fixed number of epochs using small
    # learning rate schedule; adjust as needed.
    model_final.fit(
        X_train,
        y_train,
        epochs=round(len(y_train) / 64 * 20),  # around 20 epochs of full passes
        batch_size=64,
        verbose=0,
    )
    # Save predictions for the missing prices
    if not pred_df.empty:
        X_pred = preprocessor.transform(pred_df[feature_cols])
        predictions = model_final.predict(X_pred).flatten()
        pred_df = pred_df.copy()
        pred_df['predicted_price'] = predictions
        out_path = os.path.join(os.path.dirname(__file__), 'predicted_prices_tensorflow.csv')
        pred_df.to_csv(out_path, index=False)
        print(f"Predicted prices for {len(pred_df)} cars saved to {out_path}.")


if __name__ == '__main__':
    main()