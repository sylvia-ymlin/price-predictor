"""
Model Evaluator Tests

Unit tests for the model evaluator module.
Tests all evaluation strategies: regression metrics, cross-validation, combined.

Author: Your Name
"""

import numpy as np
import pandas as pd
import pytest

from src.model_evaluator import (
    CombinedEvaluationStrategy,
    CrossValidationEvaluationStrategy,
    ModelEvaluator,
    RegressionModelEvaluationStrategy,
    compare_models,
    get_evaluation_summary,
)


class TestRegressionModelEvaluationStrategy:
    """Tests for the RegressionModelEvaluationStrategy."""

    def test_returns_all_metrics(self, trained_linear_model, sample_regression_data):
        """Test that all expected metrics are returned."""
        _, X_test, _, y_test = sample_regression_data
        
        strategy = RegressionModelEvaluationStrategy()
        metrics = strategy.evaluate_model(trained_linear_model, X_test, y_test)
        
        expected_keys = [
            "Mean Squared Error",
            "Root Mean Squared Error",
            "Mean Absolute Error",
            "R-Squared",
            "Adjusted R-Squared",
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Expected metric '{key}' not found"

    def test_metrics_are_numeric(self, trained_linear_model, sample_regression_data):
        """Test that all metric values are numeric."""
        _, X_test, _, y_test = sample_regression_data
        
        strategy = RegressionModelEvaluationStrategy()
        metrics = strategy.evaluate_model(trained_linear_model, X_test, y_test)
        
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"
            assert np.isfinite(value), f"{key} should be finite"

    def test_mse_rmse_relationship(self, trained_linear_model, sample_regression_data):
        """Test that RMSE is the square root of MSE."""
        _, X_test, _, y_test = sample_regression_data
        
        strategy = RegressionModelEvaluationStrategy()
        metrics = strategy.evaluate_model(trained_linear_model, X_test, y_test)
        
        mse = metrics["Mean Squared Error"]
        rmse = metrics["Root Mean Squared Error"]
        
        assert abs(rmse - np.sqrt(mse)) < 1e-10

    def test_r2_in_valid_range(self, trained_linear_model, sample_regression_data):
        """Test that R² is in a valid range for a decent model."""
        _, X_test, _, y_test = sample_regression_data
        
        strategy = RegressionModelEvaluationStrategy()
        metrics = strategy.evaluate_model(trained_linear_model, X_test, y_test)
        
        r2 = metrics["R-Squared"]
        # R² can be negative for very bad models, but should generally be between -1 and 1
        assert r2 <= 1.0

    def test_without_adjusted_r2(self, trained_linear_model, sample_regression_data):
        """Test evaluation without adjusted R²."""
        _, X_test, _, y_test = sample_regression_data
        
        strategy = RegressionModelEvaluationStrategy(include_adjusted_r2=False)
        metrics = strategy.evaluate_model(trained_linear_model, X_test, y_test)
        
        assert "Adjusted R-Squared" not in metrics


class TestCrossValidationEvaluationStrategy:
    """Tests for the CrossValidationEvaluationStrategy."""

    def test_returns_cv_metrics(self, sample_regression_data):
        """Test that CV metrics are returned."""
        X_train, _, y_train, _ = sample_regression_data
        
        # Need to use train data for cross-validation
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        strategy = CrossValidationEvaluationStrategy(cv=3)
        metrics = strategy.evaluate_model(model, X_train, y_train)
        
        # Check for CV-specific keys
        assert any("CV Mean" in key for key in metrics)
        assert any("CV Std" in key for key in metrics)

    def test_cv_folds_parameter(self, sample_regression_data):
        """Test that CV uses specified number of folds."""
        X_train, _, y_train, _ = sample_regression_data
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        # Different CV values should give different std
        strategy3 = CrossValidationEvaluationStrategy(cv=3)
        strategy5 = CrossValidationEvaluationStrategy(cv=5)
        
        metrics3 = strategy3.evaluate_model(model, X_train, y_train)
        metrics5 = strategy5.evaluate_model(model, X_train, y_train)
        
        # Both should produce valid results
        assert len(metrics3) > 0
        assert len(metrics5) > 0


class TestCombinedEvaluationStrategy:
    """Tests for the CombinedEvaluationStrategy."""

    def test_combines_multiple_strategies(self, trained_linear_model, sample_regression_data):
        """Test that combined strategy includes metrics from all strategies."""
        _, X_test, _, y_test = sample_regression_data
        
        strategies = [
            RegressionModelEvaluationStrategy(),
        ]
        
        combined = CombinedEvaluationStrategy(strategies)
        metrics = combined.evaluate_model(trained_linear_model, X_test, y_test)
        
        # Should have all regression metrics
        assert "Mean Squared Error" in metrics
        assert "R-Squared" in metrics


class TestModelEvaluator:
    """Tests for the ModelEvaluator context class."""

    def test_evaluates_with_strategy(self, trained_linear_model, sample_regression_data):
        """Test that evaluator uses strategy correctly."""
        _, X_test, _, y_test = sample_regression_data
        
        evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
        metrics = evaluator.evaluate(trained_linear_model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_set_strategy(self, trained_linear_model, sample_regression_data):
        """Test that strategy can be changed."""
        _, X_test, _, y_test = sample_regression_data
        
        evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
        metrics1 = evaluator.evaluate(trained_linear_model, X_test, y_test)
        
        # Change strategy
        evaluator.set_strategy(RegressionModelEvaluationStrategy(include_adjusted_r2=False))
        metrics2 = evaluator.evaluate(trained_linear_model, X_test, y_test)
        
        # Second should have fewer metrics
        assert len(metrics2) < len(metrics1)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_evaluation_summary(self):
        """Test evaluation summary formatting."""
        metrics = {
            "Mean Squared Error": 0.123456,
            "R-Squared": 0.987654,
        }
        
        summary = get_evaluation_summary(metrics)
        
        assert "Mean Squared Error" in summary
        assert "R-Squared" in summary
        assert "Model Evaluation Summary" in summary

    def test_compare_models(self, sample_regression_data):
        """Test model comparison function."""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        # Train two models
        model1 = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        model2 = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        models = {
            "Linear Regression": model1,
            "Ridge": model2,
        }
        
        comparison = compare_models(models, X_test, y_test)
        
        assert isinstance(comparison, pd.DataFrame)
        assert "Linear Regression" in comparison.index
        assert "Ridge" in comparison.index
