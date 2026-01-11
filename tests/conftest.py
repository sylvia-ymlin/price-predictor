"""
Pytest Configuration and Fixtures

This module provides shared fixtures for all test modules.
Fixtures include sample data, trained models, and common test utilities.

Author: Your Name
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_regression_data():
    """
    Generate sample regression data for testing.
    
    Returns:
        Tuple containing X_train, X_test, y_train, y_test DataFrames/Series.
    """
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42,
    )
    
    # Convert to pandas DataFrames/Series with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # Split into train/test
    split_idx = int(len(X_df) * 0.8)
    X_train = X_df.iloc[:split_idx]
    X_test = X_df.iloc[split_idx:]
    y_train = y_series.iloc[:split_idx]
    y_test = y_series.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_dataframe():
    """
    Generate a sample DataFrame for testing data processing functions.
    
    Returns:
        pd.DataFrame with mixed numerical and categorical columns.
    """
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "numerical_1": np.random.randn(n_samples) * 100 + 500,
        "numerical_2": np.random.randn(n_samples) * 50 + 200,
        "numerical_3": np.random.exponential(scale=1000, size=n_samples),
        "categorical_1": np.random.choice(["A", "B", "C"], size=n_samples),
        "categorical_2": np.random.choice(["X", "Y"], size=n_samples),
        "target": np.random.randn(n_samples) * 10000 + 200000,
    })
    
    # Add some missing values
    df.loc[5:10, "numerical_1"] = np.nan
    df.loc[15:20, "numerical_2"] = np.nan
    
    return df


@pytest.fixture
def sample_dataframe_with_outliers():
    """
    Generate a sample DataFrame with outliers for testing outlier detection.
    
    Returns:
        pd.DataFrame with numerical columns containing outliers.
    """
    np.random.seed(42)
    n_samples = 100
    
    # Normal data
    normal_data = np.random.randn(n_samples - 5) * 10 + 100
    
    # Add outliers
    outliers = np.array([500, -300, 600, -400, 700])
    data = np.concatenate([normal_data, outliers])
    
    df = pd.DataFrame({
        "feature": data,
        "target": np.random.randn(n_samples) * 100 + 1000,
    })
    
    return df


@pytest.fixture
def trained_linear_model(sample_regression_data):
    """
    Provide a trained linear regression model for testing.
    
    Returns:
        Pipeline: A fitted sklearn pipeline with StandardScaler and LinearRegression.
    """
    X_train, _, y_train, _ = sample_regression_data
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture
def housing_like_dataframe():
    """
    Generate a DataFrame similar to the Ames Housing dataset structure.
    
    Returns:
        pd.DataFrame mimicking housing data.
    """
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "Lot Area": np.random.randint(5000, 20000, size=n_samples),
        "Overall Qual": np.random.randint(1, 10, size=n_samples),
        "Year Built": np.random.randint(1950, 2020, size=n_samples),
        "Gr Liv Area": np.random.randint(800, 3000, size=n_samples),
        "Full Bath": np.random.randint(1, 4, size=n_samples),
        "Garage Cars": np.random.randint(0, 4, size=n_samples),
        "SalePrice": np.random.randint(100000, 500000, size=n_samples),
    })
    
    return df


# Utility functions for tests
def assert_dataframe_shape_preserved(original: pd.DataFrame, transformed: pd.DataFrame):
    """
    Assert that the transformation preserved the DataFrame shape.
    
    Parameters:
        original: Original DataFrame
        transformed: Transformed DataFrame
    """
    assert len(original) == len(transformed), "Row count should be preserved"


def assert_no_missing_values(df: pd.DataFrame):
    """
    Assert that a DataFrame has no missing values.
    
    Parameters:
        df: DataFrame to check
    """
    assert df.isnull().sum().sum() == 0, "DataFrame should have no missing values"


def assert_metrics_dict_valid(metrics: dict, expected_keys: list = None):
    """
    Assert that a metrics dictionary is valid.
    
    Parameters:
        metrics: Dictionary to validate
        expected_keys: Optional list of expected keys
    """
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert len(metrics) > 0, "Metrics should not be empty"
    
    for key, value in metrics.items():
        assert isinstance(key, str), "Metric keys should be strings"
        assert isinstance(value, (int, float)), f"Metric values should be numeric, got {type(value)}"
    
    if expected_keys:
        for key in expected_keys:
            assert key in metrics, f"Expected key '{key}' not found in metrics"
