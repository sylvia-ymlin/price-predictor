"""
Model Evaluator Module

This module provides a flexible evaluation framework using the Strategy Pattern.
It supports multiple evaluation strategies for regression models, including
standard metrics and cross-validation.

Design Pattern: Strategy Pattern
- ModelEvaluationStrategy: Abstract base class defining the evaluation interface
- RegressionModelEvaluationStrategy: Concrete strategy for regression metrics
- CrossValidationEvaluationStrategy: Concrete strategy for cross-validation evaluation
- ModelEvaluator: Context class that uses strategies to evaluate models

Author: Your Name
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    """
    Abstract base class for model evaluation strategies.

    This class defines the interface that all concrete evaluation strategies
    must implement. It follows the Strategy Pattern, allowing different
    evaluation methods to be used interchangeably.
    """

    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Abstract method to evaluate a model.

        Args:
            model (RegressorMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Comprehensive Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    """
    Comprehensive regression evaluation strategy.

    Computes multiple metrics including MSE, RMSE, MAE, R², and Adjusted R²
    for thorough model performance assessment.
    """

    def __init__(self, include_adjusted_r2: bool = True):
        """
        Initialize the regression evaluation strategy.

        Args:
            include_adjusted_r2 (bool): Whether to compute Adjusted R². Defaults to True.
        """
        self.include_adjusted_r2 = include_adjusted_r2

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluates a regression model using comprehensive metrics.

        Args:
            model (RegressorMixin): The trained regression model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Dict[str, float]: Dictionary containing:
                - Mean Squared Error (MSE)
                - Root Mean Squared Error (RMSE)
                - Mean Absolute Error (MAE)
                - R-Squared (R²)
                - Adjusted R-Squared (if enabled)
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "R-Squared": r2,
        }

        if self.include_adjusted_r2:
            n_samples = len(y_test)
            n_features = X_test.shape[1] if hasattr(X_test, "shape") else len(X_test.columns)

            # Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
            if n_samples > n_features + 1:
                adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
                metrics["Adjusted R-Squared"] = adjusted_r2
            else:
                logging.warning(
                    "Cannot compute Adjusted R²: insufficient samples relative to features."
                )

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics


# Concrete Strategy for Cross-Validation Evaluation
class CrossValidationEvaluationStrategy(ModelEvaluationStrategy):
    """
    Cross-validation evaluation strategy.

    Evaluates model performance using k-fold cross-validation for more
    robust performance estimates.
    """

    def __init__(
        self,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
        return_train_score: bool = False,
    ):
        """
        Initialize the cross-validation evaluation strategy.

        Args:
            cv (int): Number of cross-validation folds. Defaults to 5.
            scoring (str): Scoring metric for cross-validation.
                Options: "neg_mean_squared_error", "neg_mean_absolute_error", "r2".
            return_train_score (bool): Whether to include training scores. Defaults to False.
        """
        self.cv = cv
        self.scoring = scoring
        self.return_train_score = return_train_score

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluates a regression model using cross-validation.

        Note: For cross-validation, X_test and y_test should actually be the
        full dataset (or training set) to perform proper k-fold splits.

        Args:
            model (RegressorMixin): The trained regression model.
            X_test (pd.DataFrame): The data features for cross-validation.
            y_test (pd.Series): The data labels/target for cross-validation.

        Returns:
            Dict[str, float]: Dictionary containing:
                - CV Mean Score
                - CV Std Score
                - CV Min Score
                - CV Max Score
        """
        logging.info(f"Performing {self.cv}-fold cross-validation with {self.scoring}.")

        scores = cross_val_score(
            model,
            X_test,
            y_test,
            cv=self.cv,
            scoring=self.scoring,
        )

        # Convert negative scores to positive for metrics like neg_mean_squared_error
        if self.scoring.startswith("neg_"):
            scores = -scores
            metric_name = self.scoring.replace("neg_", "").replace("_", " ").title()
        else:
            metric_name = self.scoring.replace("_", " ").title()

        metrics = {
            f"CV Mean {metric_name}": float(np.mean(scores)),
            f"CV Std {metric_name}": float(np.std(scores)),
            f"CV Min {metric_name}": float(np.min(scores)),
            f"CV Max {metric_name}": float(np.max(scores)),
        }

        logging.info(f"Cross-Validation Results: {metrics}")
        return metrics


# Concrete Strategy for Combined Evaluation
class CombinedEvaluationStrategy(ModelEvaluationStrategy):
    """
    Combined evaluation strategy.

    Combines multiple evaluation strategies to provide a comprehensive
    view of model performance.
    """

    def __init__(self, strategies: Optional[List[ModelEvaluationStrategy]] = None):
        """
        Initialize the combined evaluation strategy.

        Args:
            strategies (List[ModelEvaluationStrategy], optional): List of strategies
                to combine. Defaults to RegressionModelEvaluationStrategy.
        """
        if strategies is None:
            strategies = [RegressionModelEvaluationStrategy()]
        self.strategies = strategies

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluates a model using all combined strategies.

        Args:
            model (RegressorMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Dict[str, float]: Combined dictionary of all evaluation metrics.
        """
        combined_metrics = {}

        for strategy in self.strategies:
            metrics = strategy.evaluate_model(model, X_test, y_test)
            combined_metrics.update(metrics)

        return combined_metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    """
    Context class for model evaluation using the Strategy Pattern.

    This class allows switching between different evaluation strategies
    at runtime, providing flexibility in how models are assessed.
    """

    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific evaluation strategy.

        Args:
            strategy (ModelEvaluationStrategy): The strategy to use for evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy) -> None:
        """
        Sets a new strategy for the ModelEvaluator.

        Args:
            strategy (ModelEvaluationStrategy): The new strategy to use.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Executes the model evaluation using the current strategy.

        Args:
            model (RegressorMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


# Utility Functions
def get_evaluation_summary(metrics: Dict[str, float]) -> str:
    """
    Creates a formatted summary string of evaluation metrics.
    
    Parameters:
        metrics (Dict[str, float]): Dictionary of metric names and values.
    
    Returns:
        str: Formatted multi-line summary of metrics.
    """
    lines = ["=" * 50, "Model Evaluation Summary", "=" * 50]
    for name, value in metrics.items():
        lines.append(f"  {name}: {value:.6f}")
    lines.append("=" * 50)
    return "\n".join(lines)


def compare_models(
    models: Dict[str, RegressorMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: Optional[ModelEvaluationStrategy] = None,
) -> pd.DataFrame:
    """
    Compares multiple models using the same evaluation strategy.
    
    Parameters:
        models (Dict[str, RegressorMixin]): Dictionary of model names to models.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.
        strategy (ModelEvaluationStrategy, optional): Evaluation strategy to use.
    
    Returns:
        pd.DataFrame: DataFrame comparing all models across metrics.
    """
    if strategy is None:
        strategy = RegressionModelEvaluationStrategy()
    
    evaluator = ModelEvaluator(strategy)
    results = {}
    
    for name, model in models.items():
        logging.info(f"Evaluating model: {name}")
        metrics = evaluator.evaluate(model, X_test, y_test)
        results[name] = metrics
    
    return pd.DataFrame(results).T


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Example 1: Basic regression evaluation
    # evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # metrics = evaluator.evaluate(model, X_test, y_test)
    # print(get_evaluation_summary(metrics))

    # Example 2: Cross-validation evaluation
    # cv_evaluator = ModelEvaluator(CrossValidationEvaluationStrategy(cv=5))
    # cv_metrics = cv_evaluator.evaluate(model, X_train, y_train)

    # Example 3: Combined evaluation
    # combined_strategy = CombinedEvaluationStrategy([
    #     RegressionModelEvaluationStrategy(),
    #     CrossValidationEvaluationStrategy(cv=5),
    # ])
    # combined_evaluator = ModelEvaluator(combined_strategy)
    # all_metrics = combined_evaluator.evaluate(model, X_test, y_test)

    # Example 4: Compare multiple models
    # models = {
    #     "Linear Regression": lr_model,
    #     "XGBoost": xgb_model,
    #     "Random Forest": rf_model,
    # }
    # comparison_df = compare_models(models, X_test, y_test)
    # print(comparison_df)

    pass
