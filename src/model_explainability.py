"""
Model Explainability Module

This module provides model interpretability using SHAP (SHapley Additive exPlanations).
SHAP values explain individual predictions by computing the contribution of each feature.

Key capabilities:
1. Global feature importance - which features matter most overall
2. Local explanations - why a specific prediction was made
3. Visualization generation - summary plots, waterfall plots, force plots
"""

import logging
import os
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelExplainer:
    """
    A class to generate SHAP-based explanations for trained models.
    
    Supports tree-based models (XGBoost, RandomForest) and linear models.
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize the explainer with a trained model and training data.
        
        Parameters:
            model: Trained sklearn-compatible model or Pipeline
            X_train: Training features used to fit the model
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type."""
        # Handle sklearn Pipeline
        if hasattr(self.model, 'named_steps'):
            # Get the final estimator from pipeline
            estimator = list(self.model.named_steps.values())[-1]
        else:
            estimator = self.model
        
        model_type = type(estimator).__name__
        logging.info(f"Initializing SHAP explainer for model type: {model_type}")
        
        # Select appropriate explainer
        if model_type in ['XGBRegressor', 'XGBClassifier', 'RandomForestRegressor', 
                          'RandomForestClassifier', 'GradientBoostingRegressor']:
            self.explainer = shap.TreeExplainer(estimator)
        elif model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            self.explainer = shap.LinearExplainer(estimator, self.X_train)
        else:
            # Fallback to KernelExplainer (slower but universal)
            logging.warning(f"Using KernelExplainer for {model_type}. This may be slow.")
            if hasattr(self.model, 'predict'):
                self.explainer = shap.KernelExplainer(self.model.predict, 
                                                       shap.sample(self.X_train, 100))
            else:
                raise ValueError(f"Cannot create explainer for model type: {model_type}")
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for the given data.
        
        Parameters:
            X: Features to explain
            
        Returns:
            SHAP values array with shape (n_samples, n_features)
        """
        # For Pipeline, need to transform data first
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
            X_transformed = self.model.named_steps['scaler'].transform(X)
            X_df = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
        else:
            X_df = X
        
        self.shap_values = self.explainer.shap_values(X_df)
        return self.shap_values
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Parameters:
            X: Features to compute importance from
            
        Returns:
            DataFrame with features sorted by importance
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Generate SHAP summary plot showing feature importance and impact direction.
        
        Parameters:
            X: Features to explain
            save_path: Optional path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Summary plot saved to: {save_path}")
        plt.close()
    
    def plot_bar(self, X: pd.DataFrame, save_path: Optional[str] = None, 
                 max_features: int = 15) -> None:
        """
        Generate bar plot of feature importance.
        
        Parameters:
            X: Features to explain
            save_path: Optional path to save the plot
            max_features: Maximum number of features to show
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, plot_type="bar", 
                         max_display=max_features, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Bar plot saved to: {save_path}")
        plt.close()
    
    def explain_prediction(self, X_single: pd.DataFrame, 
                           save_path: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Explain a single prediction using SHAP waterfall plot.
        
        Parameters:
            X_single: Single row DataFrame with features
            save_path: Optional path to save the plot
            
        Returns:
            Tuple of (shap_values, expected_value)
        """
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
            X_transformed = self.model.named_steps['scaler'].transform(X_single)
            X_df = pd.DataFrame(X_transformed, columns=X_single.columns)
        else:
            X_df = X_single
        
        shap_values_single = self.explainer.shap_values(X_df)
        expected_value = self.explainer.expected_value
        
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
        
        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_values_single[0],
            base_values=expected_value,
            data=X_single.values[0],
            feature_names=X_single.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Waterfall plot saved to: {save_path}")
        plt.close()
        
        return shap_values_single[0], expected_value


def generate_explainability_report(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "./docs/explainability"
) -> pd.DataFrame:
    """
    Generate a complete explainability report with visualizations.
    
    Parameters:
        model: Trained model or Pipeline
        X_train: Training features
        X_test: Test features to explain
        output_dir: Directory to save plots
        
    Returns:
        DataFrame with feature importance rankings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Initializing model explainer...")
    explainer = ModelExplainer(model, X_train)
    
    logging.info("Computing SHAP values for test set...")
    explainer.compute_shap_values(X_test)
    
    logging.info("Generating summary plot...")
    explainer.plot_summary(X_test, save_path=f"{output_dir}/shap_summary.png")
    
    logging.info("Generating feature importance bar plot...")
    explainer.plot_bar(X_test, save_path=f"{output_dir}/shap_importance.png")
    
    logging.info("Generating example prediction explanation...")
    explainer.explain_prediction(
        X_test.iloc[[0]], 
        save_path=f"{output_dir}/shap_waterfall_example.png"
    )
    
    importance_df = explainer.get_feature_importance(X_test)
    importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    logging.info(f"Feature importance saved to: {output_dir}/feature_importance.csv")
    
    logging.info("Explainability report generation complete.")
    return importance_df


if __name__ == "__main__":
    # Example usage (requires trained model and data)
    print("Model Explainability Module")
    print("Usage: from src.model_explainability import generate_explainability_report")
    print("       importance = generate_explainability_report(model, X_train, X_test)")
