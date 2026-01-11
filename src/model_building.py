"""
Model Building Module

This module provides a flexible model building framework using the Strategy Pattern.
It supports multiple regression algorithms including Linear Regression, XGBoost,
and Random Forest, with optional hyperparameter tuning.

Design Pattern: Strategy Pattern
- ModelBuildingStrategy: Abstract base class defining the training interface
- LinearRegressionStrategy: Standard linear regression with scaling
- XGBoostStrategy: Gradient boosting with XGBoost
- RandomForestStrategy: Ensemble learning with Random Forest
- HyperparameterTuningStrategy: GridSearchCV wrapper for any base model
- ModelBuilder: Context class that uses strategies to build models

Author: Your Name
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    """
    Abstract base class for model building strategies.
    
    This class defines the interface that all concrete model building strategies
    must implement. It follows the Strategy Pattern, allowing different algorithms
    to be used interchangeably.
    """
    
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Linear Regression
class LinearRegressionStrategy(ModelBuildingStrategy):
    """
    Linear Regression strategy with optional feature scaling.
    
    This strategy builds a pipeline with StandardScaler and LinearRegression,
    providing a simple but interpretable baseline model.
    """
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize the Linear Regression strategy.
        
        Parameters:
            fit_intercept (bool): Whether to calculate the intercept for the model.
        """
        self.fit_intercept = fit_intercept

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a linear regression model with feature scaling.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            Pipeline: A scikit-learn pipeline with StandardScaler and LinearRegression.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression(fit_intercept=self.fit_intercept)),
        ])

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")
        
        return pipeline


# Concrete Strategy for XGBoost Regression
class XGBoostStrategy(ModelBuildingStrategy):
    """
    XGBoost Gradient Boosting strategy for regression.
    
    This strategy uses XGBoost, a powerful gradient boosting algorithm that
    often achieves state-of-the-art results on structured/tabular data.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize the XGBoost strategy with hyperparameters.
        
        Parameters:
            n_estimators (int): Number of boosting rounds.
            learning_rate (float): Step size shrinkage to prevent overfitting.
            max_depth (int): Maximum depth of each tree.
            subsample (float): Subsample ratio of the training instances.
            colsample_bytree (float): Subsample ratio of columns for each tree.
            random_state (int): Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains an XGBoost regression model.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            Pipeline: A scikit-learn pipeline with StandardScaler and XGBRegressor.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing XGBoost Regressor model.")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbosity=0,  # Suppress XGBoost warnings
            )),
        ])

        logging.info("Training XGBoost model.")
        pipeline.fit(X_train, y_train)
        logging.info("XGBoost model training completed.")
        
        return pipeline


# Concrete Strategy for Random Forest Regression
class RandomForestStrategy(ModelBuildingStrategy):
    """
    Random Forest strategy for regression.
    
    This strategy uses Random Forest, an ensemble method that trains multiple
    decision trees and averages their predictions for robust results.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        """
        Initialize the Random Forest strategy with hyperparameters.
        
        Parameters:
            n_estimators (int): Number of trees in the forest.
            max_depth (int, optional): Maximum depth of each tree. None means unlimited.
            min_samples_split (int): Minimum samples required to split an internal node.
            min_samples_leaf (int): Minimum samples required at a leaf node.
            max_features (str): Number of features to consider for the best split.
            random_state (int): Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a Random Forest regression model.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            Pipeline: A scikit-learn pipeline with StandardScaler and RandomForestRegressor.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Random Forest Regressor model.")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
            )),
        ])

        logging.info("Training Random Forest model.")
        pipeline.fit(X_train, y_train)
        logging.info("Random Forest model training completed.")
        
        return pipeline


# Concrete Strategy for Hyperparameter Tuning
class HyperparameterTuningStrategy(ModelBuildingStrategy):
    """
    Hyperparameter tuning strategy using GridSearchCV.
    
    This strategy wraps any base model and performs grid search cross-validation
    to find the optimal hyperparameters.
    """
    
    def __init__(
        self,
        base_model: RegressorMixin,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
        n_jobs: int = -1,
    ):
        """
        Initialize the hyperparameter tuning strategy.
        
        Parameters:
            base_model (RegressorMixin): The base model to tune.
            param_grid (Dict[str, List[Any]]): Dictionary with parameter names as keys
                and lists of parameter values to try.
            cv (int): Number of cross-validation folds.
            scoring (str): Scoring metric for cross-validation.
            n_jobs (int): Number of parallel jobs (-1 uses all cores).
        """
        self.base_model = base_model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Performs hyperparameter tuning and trains the best model.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            Pipeline: A scikit-learn pipeline with the best model from grid search.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info(f"Initializing GridSearchCV with {self.cv}-fold cross-validation.")
        logging.info(f"Parameter grid: {self.param_grid}")

        # Create pipeline with scaling and the base model
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", self.base_model),
        ])

        # Adjust param_grid keys to match pipeline naming
        adjusted_param_grid = {
            f"model__{key}": value for key, value in self.param_grid.items()
        }

        grid_search = GridSearchCV(
            pipeline,
            adjusted_param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1,
        )

        logging.info("Starting hyperparameter search...")
        grid_search.fit(X_train, y_train)
        
        logging.info(f"Best parameters found: {grid_search.best_params_}")
        logging.info(f"Best cross-validation score: {-grid_search.best_score_:.4f} MSE")
        
        return grid_search.best_estimator_


# Context Class for Model Building
class ModelBuilder:
    """
    Context class for model building using the Strategy Pattern.
    
    This class allows switching between different model building strategies
    at runtime, providing flexibility in model selection.
    """
    
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific strategy.

        Parameters:
            strategy (ModelBuildingStrategy): The strategy to use for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy) -> None:
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
            strategy (ModelBuildingStrategy): The new strategy to use.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Executes model building using the current strategy.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


# Factory function for creating model strategies
def get_model_strategy(
    model_type: str = "linear_regression",
    **kwargs
) -> ModelBuildingStrategy:
    """
    Factory function to create model building strategies.
    
    Parameters:
        model_type (str): Type of model to create. Options:
            - "linear_regression": Linear Regression (default)
            - "xgboost": XGBoost Regressor
            - "random_forest": Random Forest Regressor
        **kwargs: Additional keyword arguments passed to the strategy constructor.
    
    Returns:
        ModelBuildingStrategy: The requested model building strategy.
    
    Raises:
        ValueError: If an unknown model_type is provided.
    
    Example:
        >>> strategy = get_model_strategy("xgboost", n_estimators=200)
        >>> model_builder = ModelBuilder(strategy)
        >>> model = model_builder.build_model(X_train, y_train)
    """
    strategies = {
        "linear_regression": LinearRegressionStrategy,
        "xgboost": XGBoostStrategy,
        "random_forest": RandomForestStrategy,
    }
    
    if model_type not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. Available options: {available}"
        )
    
    return strategies[model_type](**kwargs)


# Example usage
if __name__ == "__main__":
    # Example DataFrame (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')
    # X_train = df.drop(columns=['target_column'])
    # y_train = df['target_column']

    # Example 1: Using Linear Regression Strategy
    # model_builder = ModelBuilder(LinearRegressionStrategy())
    # trained_model = model_builder.build_model(X_train, y_train)

    # Example 2: Using XGBoost Strategy
    # model_builder = ModelBuilder(XGBoostStrategy(n_estimators=200, learning_rate=0.05))
    # trained_model = model_builder.build_model(X_train, y_train)

    # Example 3: Using Random Forest Strategy
    # model_builder = ModelBuilder(RandomForestStrategy(n_estimators=200, max_depth=10))
    # trained_model = model_builder.build_model(X_train, y_train)

    # Example 4: Using Hyperparameter Tuning
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.1],
    # }
    # from xgboost import XGBRegressor
    # tuning_strategy = HyperparameterTuningStrategy(
    #     base_model=XGBRegressor(),
    #     param_grid=param_grid,
    #     cv=5
    # )
    # model_builder = ModelBuilder(tuning_strategy)
    # best_model = model_builder.build_model(X_train, y_train)

    # Example 5: Using factory function
    # strategy = get_model_strategy("xgboost", n_estimators=200)
    # model_builder = ModelBuilder(strategy)
    # model = model_builder.build_model(X_train, y_train)

    pass
