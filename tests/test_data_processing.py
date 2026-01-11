"""
Data Processing Tests

Unit tests for data processing modules including data ingestion,
missing value handling, outlier detection, and data splitting.

Author: Your Name
"""

import numpy as np
import pandas as pd
import pytest

from src.data_splitter import (
    DataSplitter,
    SimpleTrainTestSplitStrategy,
)
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from src.outlier_detection import (
    IQROutlierDetection,
    OutlierDetector,
    ZScoreOutlierDetection,
)


class TestMissingValueHandling:
    """Tests for missing value handling strategies."""

    def test_drop_missing_values(self, sample_dataframe):
        """Test dropping rows with missing values."""
        df = sample_dataframe.copy()
        original_rows = len(df)
        
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
        cleaned = handler.handle_missing_values(df)
        
        assert len(cleaned) < original_rows
        assert cleaned.isnull().sum().sum() == 0

    def test_fill_missing_with_mean(self, sample_dataframe):
        """Test filling missing values with mean."""
        df = sample_dataframe.copy()
        
        handler = MissingValueHandler(FillMissingValuesStrategy(method="mean"))
        filled = handler.handle_missing_values(df)
        
        # Check that numeric columns have no missing values
        numeric_cols = df.select_dtypes(include="number").columns
        assert filled[numeric_cols].isnull().sum().sum() == 0

    def test_fill_missing_with_median(self, sample_dataframe):
        """Test filling missing values with median."""
        df = sample_dataframe.copy()
        
        handler = MissingValueHandler(FillMissingValuesStrategy(method="median"))
        filled = handler.handle_missing_values(df)
        
        numeric_cols = df.select_dtypes(include="number").columns
        assert filled[numeric_cols].isnull().sum().sum() == 0

    def test_fill_missing_with_mode(self, sample_dataframe):
        """Test filling missing values with mode."""
        df = sample_dataframe.copy()
        
        handler = MissingValueHandler(FillMissingValuesStrategy(method="mode"))
        filled = handler.handle_missing_values(df)
        
        # All columns should have no missing values after mode fill
        assert filled.isnull().sum().sum() == 0

    def test_fill_missing_with_constant(self, sample_dataframe):
        """Test filling missing values with a constant."""
        df = sample_dataframe.copy()
        constant_value = -999
        
        handler = MissingValueHandler(
            FillMissingValuesStrategy(method="constant", fill_value=constant_value)
        )
        filled = handler.handle_missing_values(df)
        
        assert filled.isnull().sum().sum() == 0
        # Check that the constant was used
        assert (filled == constant_value).any().any()

    def test_set_strategy(self, sample_dataframe):
        """Test switching between strategies."""
        df = sample_dataframe.copy()
        
        handler = MissingValueHandler(DropMissingValuesStrategy())
        dropped = handler.handle_missing_values(df)
        
        handler.set_strategy(FillMissingValuesStrategy(method="mean"))
        filled = handler.handle_missing_values(df)
        
        # Dropped should have fewer rows
        assert len(dropped) < len(filled)


class TestOutlierDetection:
    """Tests for outlier detection strategies."""

    def test_zscore_detects_outliers(self, sample_dataframe_with_outliers):
        """Test that Z-score method detects outliers."""
        df = sample_dataframe_with_outliers.copy()
        
        detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
        outliers = detector.detect_outliers(df[["feature"]])
        
        # Should detect some outliers
        assert outliers.any().any()

    def test_iqr_detects_outliers(self, sample_dataframe_with_outliers):
        """Test that IQR method detects outliers."""
        df = sample_dataframe_with_outliers.copy()
        
        detector = OutlierDetector(IQROutlierDetection())
        outliers = detector.detect_outliers(df[["feature"]])
        
        # Should detect some outliers
        assert outliers.any().any()

    def test_handle_outliers_remove(self, sample_dataframe_with_outliers):
        """Test removing outliers from dataset."""
        df = sample_dataframe_with_outliers.copy()
        original_len = len(df)
        
        detector = OutlierDetector(ZScoreOutlierDetection(threshold=2))
        cleaned = detector.handle_outliers(df[["feature"]], method="remove")
        
        # Should have fewer rows
        assert len(cleaned) < original_len

    def test_handle_outliers_cap(self, sample_dataframe_with_outliers):
        """Test capping outliers in dataset."""
        df = sample_dataframe_with_outliers.copy()
        original_len = len(df)
        
        detector = OutlierDetector(ZScoreOutlierDetection())
        capped = detector.handle_outliers(df[["feature"]], method="cap")
        
        # Should preserve row count
        assert len(capped) == original_len

    def test_set_strategy(self, sample_dataframe_with_outliers):
        """Test switching between detection strategies."""
        df = sample_dataframe_with_outliers.copy()
        
        detector = OutlierDetector(ZScoreOutlierDetection())
        zscore_outliers = detector.detect_outliers(df[["feature"]])
        
        detector.set_strategy(IQROutlierDetection())
        iqr_outliers = detector.detect_outliers(df[["feature"]])
        
        # Both should produce valid results (may differ in exact count)
        assert isinstance(zscore_outliers, pd.DataFrame)
        assert isinstance(iqr_outliers, pd.DataFrame)


class TestDataSplitting:
    """Tests for data splitting strategies."""

    def test_simple_train_test_split(self, housing_like_dataframe):
        """Test simple train-test split."""
        df = housing_like_dataframe.copy()
        
        splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2))
        X_train, X_test, y_train, y_test = splitter.split(df, target_column="SalePrice")
        
        # Check correct proportions
        total = len(df)
        assert len(X_train) == pytest.approx(total * 0.8, abs=1)
        assert len(X_test) == pytest.approx(total * 0.2, abs=1)
        
        # Target should be separated
        assert "SalePrice" not in X_train.columns
        assert "SalePrice" not in X_test.columns

    def test_split_preserves_features(self, housing_like_dataframe):
        """Test that split preserves feature columns."""
        df = housing_like_dataframe.copy()
        expected_features = [c for c in df.columns if c != "SalePrice"]
        
        splitter = DataSplitter(SimpleTrainTestSplitStrategy())
        X_train, X_test, _, _ = splitter.split(df, target_column="SalePrice")
        
        assert list(X_train.columns) == expected_features
        assert list(X_test.columns) == expected_features

    def test_random_state_reproducibility(self, housing_like_dataframe):
        """Test that random state ensures reproducibility."""
        df = housing_like_dataframe.copy()
        
        splitter = DataSplitter(SimpleTrainTestSplitStrategy(random_state=42))
        X_train1, _, _, _ = splitter.split(df, target_column="SalePrice")
        
        splitter2 = DataSplitter(SimpleTrainTestSplitStrategy(random_state=42))
        X_train2, _, _, _ = splitter2.split(df, target_column="SalePrice")
        
        pd.testing.assert_frame_equal(X_train1, X_train2)

    def test_different_test_sizes(self, housing_like_dataframe):
        """Test different test size configurations."""
        df = housing_like_dataframe.copy()
        
        for test_size in [0.1, 0.2, 0.3]:
            splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=test_size))
            X_train, X_test, _, _ = splitter.split(df, target_column="SalePrice")
            
            expected_test = int(len(df) * test_size)
            assert len(X_test) == pytest.approx(expected_test, abs=1)
