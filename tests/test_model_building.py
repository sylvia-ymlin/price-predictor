"""
Model Building Tests

Unit tests for the model building module.
Tests all model strategies: Linear Regression, XGBoost, Random Forest.

Author: Your Name
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model_building import (
    LinearRegressionStrategy,
    ModelBuilder,
    RandomForestStrategy,
    get_model_strategy,
)


class TestLinearRegressionStrategy:
    """Tests for the LinearRegressionStrategy."""

    def test_builds_pipeline(self, sample_regression_data):
        """Test that strategy builds a valid sklearn Pipeline."""
        X_train, _, y_train, _ = sample_regression_data
        
        strategy = LinearRegressionStrategy()
        model = strategy.build_and_train_model(X_train, y_train)
        
        assert isinstance(model, Pipeline)
        assert "scaler" in model.named_steps
        assert "model" in model.named_steps

    def test_model_can_predict(self, sample_regression_data):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, _ = sample_regression_data
        
        strategy = LinearRegressionStrategy()
        model = strategy.build_and_train_model(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predictions_are_reasonable(self, sample_regression_data):
        """Test that predictions are in a reasonable range."""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        strategy = LinearRegressionStrategy()
        model = strategy.build_and_train_model(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Predictions should be finite
        assert np.all(np.isfinite(predictions))
        
        # Predictions should be somewhat correlated with actual values
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        assert correlation > 0.5  # Should have positive correlation

    def test_raises_on_invalid_input(self):
        """Test that strategy raises TypeError for invalid input."""
        strategy = LinearRegressionStrategy()
        
        with pytest.raises(TypeError):
            strategy.build_and_train_model([[1, 2], [3, 4]], [1, 2])  # Not DataFrame/Series


class TestRandomForestStrategy:
    """Tests for the RandomForestStrategy."""

    def test_builds_pipeline(self, sample_regression_data):
        """Test that strategy builds a valid sklearn Pipeline."""
        X_train, _, y_train, _ = sample_regression_data
        
        strategy = RandomForestStrategy(n_estimators=10, random_state=42)
        model = strategy.build_and_train_model(X_train, y_train)
        
        assert isinstance(model, Pipeline)
        assert "scaler" in model.named_steps
        assert "model" in model.named_steps

    def test_model_can_predict(self, sample_regression_data):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, _ = sample_regression_data
        
        strategy = RandomForestStrategy(n_estimators=10, random_state=42)
        model = strategy.build_and_train_model(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_custom_hyperparameters(self, sample_regression_data):
        """Test that custom hyperparameters are applied."""
        X_train, _, y_train, _ = sample_regression_data
        
        n_estimators = 20
        max_depth = 5
        
        strategy = RandomForestStrategy(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model = strategy.build_and_train_model(X_train, y_train)
        
        rf_model = model.named_steps["model"]
        assert rf_model.n_estimators == n_estimators
        assert rf_model.max_depth == max_depth


class TestXGBoostStrategy:
    """Tests for the XGBoostStrategy (if XGBoost is installed)."""

    @pytest.fixture
    def xgboost_available(self):
        """Check if XGBoost is available."""
        try:
            import xgboost
            return True
        except ImportError:
            return False

    def test_builds_pipeline(self, sample_regression_data, xgboost_available):
        """Test that strategy builds a valid sklearn Pipeline."""
        if not xgboost_available:
            pytest.skip("XGBoost not installed")
        
        from src.model_building import XGBoostStrategy
        
        X_train, _, y_train, _ = sample_regression_data
        
        strategy = XGBoostStrategy(n_estimators=10, random_state=42)
        model = strategy.build_and_train_model(X_train, y_train)
        
        assert isinstance(model, Pipeline)
        assert "scaler" in model.named_steps
        assert "model" in model.named_steps

    def test_model_can_predict(self, sample_regression_data, xgboost_available):
        """Test that trained model can make predictions."""
        if not xgboost_available:
            pytest.skip("XGBoost not installed")
        
        from src.model_building import XGBoostStrategy
        
        X_train, X_test, y_train, _ = sample_regression_data
        
        strategy = XGBoostStrategy(n_estimators=10, random_state=42)
        model = strategy.build_and_train_model(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestModelBuilder:
    """Tests for the ModelBuilder context class."""

    def test_builds_model_with_strategy(self, sample_regression_data):
        """Test that ModelBuilder correctly uses strategy."""
        X_train, _, y_train, _ = sample_regression_data
        
        builder = ModelBuilder(LinearRegressionStrategy())
        model = builder.build_model(X_train, y_train)
        
        assert isinstance(model, Pipeline)

    def test_set_strategy(self, sample_regression_data):
        """Test that strategy can be changed."""
        X_train, _, y_train, _ = sample_regression_data
        
        builder = ModelBuilder(LinearRegressionStrategy())
        
        # Build with linear regression
        model1 = builder.build_model(X_train, y_train)
        
        # Switch to random forest
        builder.set_strategy(RandomForestStrategy(n_estimators=10))
        model2 = builder.build_model(X_train, y_train)
        
        # Models should be different types
        assert type(model1.named_steps["model"]) != type(model2.named_steps["model"])


class TestGetModelStrategy:
    """Tests for the get_model_strategy factory function."""

    def test_linear_regression(self):
        """Test factory creates LinearRegressionStrategy."""
        strategy = get_model_strategy("linear_regression")
        assert isinstance(strategy, LinearRegressionStrategy)

    def test_random_forest(self):
        """Test factory creates RandomForestStrategy."""
        strategy = get_model_strategy("random_forest", n_estimators=50)
        assert isinstance(strategy, RandomForestStrategy)

    def test_invalid_model_type(self):
        """Test factory raises for unknown model type."""
        with pytest.raises(ValueError) as exc_info:
            get_model_strategy("invalid_model")
        
        assert "Unknown model type" in str(exc_info.value)

    def test_passes_kwargs(self):
        """Test factory passes kwargs to strategy."""
        strategy = get_model_strategy("random_forest", n_estimators=200, max_depth=10)
        assert strategy.n_estimators == 200
        assert strategy.max_depth == 10
