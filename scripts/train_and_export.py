"""
Train and export a standalone model for deployment.
This script trains an XGBoost model and saves it as a joblib file
that can be bundled into the Docker image.
"""

import os
import sys
import logging
import zipfile
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Key features to use for prediction
FEATURE_COLUMNS = [
    "Gr Liv Area",
    "Overall Qual", 
    "Total Bsmt SF",
    "Garage Cars",
    "Year Built",
    "1st Flr SF",
    "Full Bath",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Area"
]


def load_data(zip_path: str) -> pd.DataFrame:
    """Load data from the archive.zip file."""
    logger.info(f"Loading data from {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the actual data file inside
        file_list = z.namelist()
        logger.info(f"Files in archive: {file_list}")
        
        # Look for csv or excel file
        data_file = None
        for f in file_list:
            if f.endswith('.csv') or f.endswith('.xlsx'):
                data_file = f
                break
        
        if data_file is None:
            raise ValueError("No CSV or Excel file found in archive")
        
        logger.info(f"Reading {data_file}")
        if data_file.endswith('.csv'):
            df = pd.read_csv(z.open(data_file))
        else:
            df = pd.read_excel(z.open(data_file))
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess data for training."""
    logger.info("Preprocessing data...")
    
    # Check available columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    logger.info(f"Using features: {available_features}")
    
    if "SalePrice" not in df.columns:
        raise ValueError("SalePrice column not found")
    
    # Select features and target
    X = df[available_features].copy()
    y = df["SalePrice"].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Log transform target for better distribution
    y = np.log1p(y)
    
    logger.info(f"Features shape: {X.shape}")
    return X, y, available_features


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train XGBoost model with StandardScaler."""
    logger.info("Training model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline with scaler and XGBoost
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgboost", XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"Train R²: {train_score:.4f}")
    logger.info(f"Test R²: {test_score:.4f}")
    
    # Calculate MSE on test set
    from sklearn.metrics import mean_squared_error
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    
    metrics = {
        "train_r2": train_score,
        "test_r2": test_score,
        "test_mse": mse,
        "test_rmse": rmse
    }
    
    return pipeline, metrics


def save_model(pipeline, features: list, metrics: dict, output_dir: str):
    """Save model and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.joblib")
    meta_path = os.path.join(output_dir, "model_meta.joblib")
    
    # Save model
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "features": features,
        "metrics": metrics,
        "model_type": "XGBoost",
        "version": os.environ.get("MODEL_VERSION", "1.0.0")
    }
    joblib.dump(metadata, meta_path)
    logger.info(f"Metadata saved to {meta_path}")
    
    return model_path, meta_path


def main():
    """Main training pipeline."""
    # Paths
    data_path = "data/archive.zip"
    output_dir = "models"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess
    X, y, features = preprocess_data(df)
    
    # Train
    pipeline, metrics = train_model(X, y)
    
    # Save
    save_model(pipeline, features, metrics, output_dir)
    
    logger.info("Training complete!")
    logger.info(f"Model metrics: {metrics}")


if __name__ == "__main__":
    main()
