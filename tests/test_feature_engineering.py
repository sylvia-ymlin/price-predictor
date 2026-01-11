"""
Feature Engineering Tests

Unit tests for the feature engineering module.
Tests all feature transformation strategies: Log, Standard Scaling, MinMax, OneHot.

Author: Your Name
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)


class TestLogTransformation:
    """Tests for the LogTransformation strategy."""

    def test_log_transformation_basic(self, sample_dataframe):
        """Test that log transformation is applied correctly."""
        df = sample_dataframe.copy()
        features = ["numerical_1", "numerical_2"]
        
        # Make values positive for log transformation
        df["numerical_1"] = df["numerical_1"].abs() + 1
        df["numerical_2"] = df["numerical_2"].abs() + 1
        
        transformer = FeatureEngineer(LogTransformation(features=features))
        transformed = transformer.apply_feature_engineering(df)
        
        # Check that transformation was applied
        assert not transformed[features].equals(df[features])
        
        # Verify calculation for a few points
        # expected = np.log1p(df[features])
        # pd.testing.assert_frame_equal(transformed[features], expected)
        
        # Simplified check: strictly less if x > 0
        mask = df[features] > 0
        if mask.any().any():
             # Check explicit numpy calculation
             expected = np.log1p(df[features])
             pd.testing.assert_frame_equal(transformed[features], expected)

    def test_log_transformation_preserves_other_columns(self, sample_dataframe):
        """Test that non-transformed columns are preserved."""
        df = sample_dataframe.copy()
        df["numerical_1"] = df["numerical_1"].abs() + 1
        
        transformer = FeatureEngineer(LogTransformation(features=["numerical_1"]))
        transformed = transformer.apply_feature_engineering(df)
        
        # Check that other numerical columns are unchanged
        pd.testing.assert_series_equal(
            transformed["numerical_3"], df["numerical_3"], check_names=True
        )

    def test_log_transformation_handles_zeros(self):
        """Test that log1p handles zero values correctly."""
        df = pd.DataFrame({
            "value": [0, 1, 10, 100],
            "other": [1, 2, 3, 4],
        })
        
        transformer = FeatureEngineer(LogTransformation(features=["value"]))
        transformed = transformer.apply_feature_engineering(df)
        
        # log1p(0) should be 0
        assert transformed["value"].iloc[0] == 0


class TestStandardScaling:
    """Tests for the StandardScaling strategy."""

    def test_standard_scaling_basic(self, sample_dataframe):
        """Test that standard scaling normalizes data correctly."""
        df = sample_dataframe.dropna()  # Remove NaN for this test
        features = ["numerical_1", "numerical_2"]
        
        transformer = FeatureEngineer(StandardScaling(features=features))
        transformed = transformer.apply_feature_engineering(df)
        
        # Scaled values should have mean ≈ 0 and std ≈ 1
        for feature in features:
            assert abs(transformed[feature].mean()) < 0.01
            assert abs(transformed[feature].std() - 1) < 0.01

    def test_standard_scaling_preserves_shape(self, sample_dataframe):
        """Test that standard scaling preserves DataFrame shape."""
        df = sample_dataframe.dropna()
        features = ["numerical_1"]
        
        transformer = FeatureEngineer(StandardScaling(features=features))
        transformed = transformer.apply_feature_engineering(df)
        
        assert transformed.shape == df.shape


class TestMinMaxScaling:
    """Tests for the MinMaxScaling strategy."""

    def test_minmax_scaling_default_range(self, sample_dataframe):
        """Test that MinMax scaling scales to [0, 1] by default."""
        df = sample_dataframe.dropna()
        features = ["numerical_1", "numerical_2"]
        
        transformer = FeatureEngineer(MinMaxScaling(features=features))
        transformed = transformer.apply_feature_engineering(df)
        
        for feature in features:
            assert transformed[feature].min() >= 0
            assert transformed[feature].max() <= 1

    def test_minmax_scaling_custom_range(self, sample_dataframe):
        """Test that MinMax scaling works with custom range."""
        df = sample_dataframe.dropna()
        features = ["numerical_1"]
        
        transformer = FeatureEngineer(
            MinMaxScaling(features=features, feature_range=(-1, 1))
        )
        transformed = transformer.apply_feature_engineering(df)
        
        assert transformed["numerical_1"].min() >= -1
        assert transformed["numerical_1"].max() <= 1


class TestOneHotEncoding:
    """Tests for the OneHotEncoding strategy."""

    def test_onehot_encoding_creates_columns(self, sample_dataframe):
        """Test that one-hot encoding creates new binary columns."""
        df = sample_dataframe.dropna()
        features = ["categorical_1"]
        
        transformer = FeatureEngineer(OneHotEncoding(features=features))
        transformed = transformer.apply_feature_engineering(df)
        
        # Original categorical column should be removed
        assert "categorical_1" not in transformed.columns
        
        # New columns should be created (drop='first' leaves n-1 columns)
        onehot_cols = [c for c in transformed.columns if c.startswith("categorical_1_")]
        assert len(onehot_cols) == 2  # 3 categories - 1 = 2

    def test_onehot_encoding_binary_values(self, sample_dataframe):
        """Test that one-hot encoded values are binary (0 or 1)."""
        df = sample_dataframe.dropna()
        features = ["categorical_1"]
        
        transformer = FeatureEngineer(OneHotEncoding(features=features))
        transformed = transformer.apply_feature_engineering(df)
        
        onehot_cols = [c for c in transformed.columns if c.startswith("categorical_1_")]
        for col in onehot_cols:
            assert set(transformed[col].unique()).issubset({0.0, 1.0})


class TestFeatureEngineer:
    """Tests for the FeatureEngineer context class."""

    def test_set_strategy(self, sample_dataframe):
        """Test that strategy can be changed dynamically."""
        df = sample_dataframe.dropna()
        df["numerical_1"] = df["numerical_1"].abs() + 1
        
        # Start with log transformation
        engineer = FeatureEngineer(LogTransformation(features=["numerical_1"]))
        result1 = engineer.apply_feature_engineering(df)
        
        # Switch to standard scaling
        engineer.set_strategy(StandardScaling(features=["numerical_1"]))
        result2 = engineer.apply_feature_engineering(df)
        
        # Results should be different
        assert not result1["numerical_1"].equals(result2["numerical_1"])

    def test_accepts_different_strategies(self, sample_dataframe):
        """Test that FeatureEngineer accepts all strategy types."""
        df = sample_dataframe.dropna()
        strategies = [
            LogTransformation(features=["numerical_3"]),
            StandardScaling(features=["numerical_1"]),
            MinMaxScaling(features=["numerical_2"]),
            OneHotEncoding(features=["categorical_1"]),
        ]
        
        for strategy in strategies:
            engineer = FeatureEngineer(strategy)
            transformed = engineer.apply_feature_engineering(df)
            assert transformed is not None
