"""
Model Evaluator Step

ZenML step for evaluating trained ML models with comprehensive metrics.
Supports standard regression metrics, cross-validation, and MLflow logging.

Author: Your Name
"""

import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step

from src.model_evaluator import (
    ModelEvaluator,
    RegressionModelEvaluationStrategy,
    get_evaluation_summary,
)


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates the trained model using comprehensive regression metrics.

    This step computes multiple evaluation metrics and logs them to MLflow
    for experiment tracking.

    Parameters:
        trained_model (Pipeline): The trained pipeline containing preprocessing
            and model steps.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data labels/target.

    Returns:
        Tuple[dict, float]: A tuple containing:
            - dict: Dictionary of all evaluation metrics
            - float: Mean Squared Error value

    Metrics Computed:
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Error (MAE)
        - R-Squared (R²)
        - Adjusted R-Squared

    Raises:
        TypeError: If X_test is not a DataFrame or y_test is not a Series.
        ValueError: If evaluation metrics are not returned as a dictionary.
    """
    # Input validation
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Starting model evaluation...")

    # Apply the preprocessing to test data
    logging.info("Applying preprocessing to the test data.")
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Initialize the evaluator with comprehensive regression strategy
    evaluator = ModelEvaluator(
        strategy=RegressionModelEvaluationStrategy(include_adjusted_r2=True)
    )

    # Perform the evaluation on the underlying model
    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    # Validate return type
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")

    # Log metrics to MLflow if active
    try:
        if mlflow.active_run():
            for metric_name, metric_value in evaluation_metrics.items():
                # Convert metric name to MLflow-friendly format
                mlflow_name = metric_name.lower().replace(" ", "_").replace("-", "_")
                mlflow.log_metric(mlflow_name, metric_value)
            logging.info("Metrics logged to MLflow.")
    except Exception as e:
        logging.warning(f"Could not log metrics to MLflow: {e}")

    # Print evaluation summary
    summary = get_evaluation_summary(evaluation_metrics)
    logging.info(f"\n{summary}")

    # Extract MSE for backward compatibility
    mse = evaluation_metrics.get("Mean Squared Error", 0.0)

    return evaluation_metrics, mse
