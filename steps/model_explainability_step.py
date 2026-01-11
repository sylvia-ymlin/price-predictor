"""
Model Explainability Step

ZenML step for generating SHAP-based model explanations.
Produces feature importance rankings and visualization artifacts.
"""

import pandas as pd
from zenml import step, ArtifactConfig
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_explainability_step(
    trained_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate model explainability report using SHAP.
    
    Parameters:
        trained_model: The trained model (Pipeline or estimator)
        X_train: Training features for explainer initialization
        X_test: Test features to explain
        
    Returns:
        DataFrame with feature importance rankings
    """
    from src.model_explainability import generate_explainability_report
    
    logger.info("Starting model explainability analysis...")
    
    importance_df = generate_explainability_report(
        model=trained_model,
        X_train=X_train,
        X_test=X_test,
        output_dir="./docs/explainability"
    )
    
    # Log top features
    top_features = importance_df.head(5)
    logger.info("Top 5 most important features:")
    for _, row in top_features.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df
