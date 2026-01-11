"""
Model Building Step

ZenML step for building and training ML models using various strategies.
Supports Linear Regression, XGBoost, and Random Forest with MLflow tracking.

Author: Your Name
"""

import logging
from typing import Annotated, Literal

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client

from src.model_building import (
    LinearRegressionStrategy,
    RandomForestStrategy,
    XGBoostStrategy,
    get_model_strategy,
)

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

from zenml import Model

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "linear_regression",
    enable_tuning: bool = False,
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a regression model using scikit-learn pipelines.

    This step supports multiple model types and automatically handles
    categorical and numerical preprocessing.

    Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.
        model_type (str): Type of model to train. Options:
            - "linear_regression" (default): Standard linear regression
            - "xgboost": XGBoost gradient boosting
            - "random_forest": Random Forest ensemble

    Returns:
        Pipeline: The trained scikit-learn pipeline including preprocessing
            and the selected regression model.

    Raises:
        TypeError: If X_train is not a DataFrame or y_train is not a Series.
        ValueError: If an unknown model_type is provided.

    Example:
        >>> pipeline = model_building_step(X_train, y_train, model_type="xgboost")
    """
    # Input validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")
    logging.info(f"Selected model type: {model_type}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Get the appropriate model based on model_type
    model_mapping = {
        "linear_regression": ("Linear Regression", lambda: __import__('sklearn.linear_model', fromlist=['LinearRegression']).LinearRegression()),
        "xgboost": ("XGBoost", lambda: _get_xgboost_model()),
        "random_forest": ("Random Forest", lambda: __import__('sklearn.ensemble', fromlist=['RandomForestRegressor']).RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    }

    if model_type not in model_mapping:
        available = ", ".join(model_mapping.keys())
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")

    model_name, model_factory = model_mapping[model_type]

    # Define the model training pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_factory()),
        ]
    )

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging for scikit-learn
        mlflow.sklearn.autolog()

        # Log model type as a parameter
        mlflow.log_param("model_type", model_type)

        if enable_tuning:
            logging.info("Hyperparameter tuning enabled. Performing GridSearchCV...")
            from sklearn.model_selection import GridSearchCV
            
            # Define parameter grids based on model type
            param_grids = {
                "xgboost": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.1, 0.2],
                    "model__max_depth": [3, 5, 7],
                },
                "random_forest": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [10, 20, None],
                    "model__min_samples_split": [2, 5],
                },
                "linear_regression": {
                    "model__fit_intercept": [True, False],
                }
            }
            
            grid_params = param_grids.get(model_type, {})
            
            search = GridSearchCV(
                pipeline,
                grid_params,
                cv=3,
                scoring="r2",
                verbose=1,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            
            logging.info(f"Best parameters: {search.best_params_}")
            logging.info(f"Best cross-validation score: {search.best_score_:.4f}")
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_score", search.best_score_)
            
            # The best estimator is the pipeline with best params
            pipeline = search.best_estimator_
            logging.info(f"Model tuning completed for {model_name}.")
            
        else:
            logging.info(f"Building and training the {model_name} model (no tuning).")
            pipeline.fit(X_train, y_train)
            logging.info("Model training completed.")

        # Log the columns that the model expects
        if len(categorical_cols) > 0:
            onehot_encoder = (
                pipeline.named_steps["preprocessor"]
                .transformers_[1][1]
                .named_steps["onehot"]
            )
            # Ensure the encoder is fitted before accessing feature names
            # If loaded from best_estimator_, it is already fitted
            
            expected_columns = numerical_cols.tolist() + list(
                onehot_encoder.get_feature_names_out(categorical_cols)
            )
        else:
            expected_columns = numerical_cols.tolist()

        logging.info(f"Model expects {len(expected_columns)} features after preprocessing.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline


def _get_xgboost_model():
    """Helper function to import and create XGBoost model."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )
