"""
Housing Prices Predictor API

A production-ready FastAPI service for housing price predictions.
Supports both real MLflow models and demo mode for testing/deployment verification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from prometheus_fastapi_instrumentator import Instrumentator

# Setup structured logging
from src.logger import setup_logger
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Housing Prices Predictor API",
    description="""
A production-ready API for predicting housing prices.

## Features
- Real-time price predictions
- Health monitoring endpoint
- Model version tracking
- Prometheus metrics at /metrics

## Model Info
The model is trained on the Ames Housing Dataset using XGBoost.
    """,
    version="1.0.0",
)

# Instrument app with Prometheus
Instrumentator().instrument(app).expose(app)

# Global variables
model = None
model_version = None
demo_mode = False
model_features = []


# Request/Response schemas
class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: Dict[str, Any] = Field(
        ...,
        example={
            "Gr Liv Area": 1500,
            "Overall Qual": 7,
            "Total Bsmt SF": 1000,
            "Garage Cars": 2,
            "Year Built": 2005
        }
    )

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions (MLflow format)."""
    dataframe_records: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predicted_price: float
    model_version: str
    demo_mode: bool
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[float]
    model_version: str
    demo_mode: bool

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    service: str
    model_loaded: bool
    demo_mode: bool


def calculate_demo_price(features: Dict[str, Any]) -> float:
    """
    Calculate a realistic demo price based on feature values.
    Uses simplified linear approximation of actual model behavior.
    """
    base_price = 150000
    
    # Living area: ~$100 per sq ft
    living_area = features.get("Gr Liv Area", 1500)
    price = base_price + (living_area * 100)
    
    # Quality: ~$20k per quality point above 5
    quality = features.get("Overall Qual", 5)
    price += (quality - 5) * 20000
    
    # Basement: ~$50 per sq ft
    basement = features.get("Total Bsmt SF", 0)
    price += basement * 50
    
    # Garage: ~$15k per car
    garage = features.get("Garage Cars", 0)
    price += garage * 15000
    
    # Age adjustment: -$1k per year for homes older than 2000
    year_built = features.get("Year Built", 2000)
    if year_built < 2000:
        price -= (2000 - year_built) * 1000
    else:
        price += (year_built - 2000) * 500
    
    return max(50000, min(price, 800000))  # Clamp to reasonable range


@app.on_event("startup")
async def load_model():
    """Load the model on startup. Priority: local joblib -> MLflow -> demo mode."""
    global model, model_version, demo_mode, model_features
    
    # Check if running in demo mode via environment variable
    if os.getenv("DEMO_MODE", "false").lower() == "true":
        logger.info("Running in DEMO mode (set by environment variable)")
        demo_mode = True
        model_version = "demo-1.0.0"
        return
    
    # Try 1: Load local joblib model (bundled in Docker)
    try:
        import joblib
        model_path = "models/model.joblib"
        meta_path = "models/model_meta.joblib"
        
        if os.path.exists(model_path):
            logger.info(f"Loading local model from {model_path}")
            model = joblib.load(model_path)
            
            if os.path.exists(meta_path):
                metadata = joblib.load(meta_path)
                model_version = f"xgboost-{metadata.get('version', '1.0.0')}"
                model_features = metadata.get('features', [])
                logger.info(f"Model features: {model_features}")
            else:
                model_version = "xgboost-1.0.0"
                model_features = []
            
            demo_mode = False
            logger.info(f"Local model loaded successfully: {model_version}")
            return
            
    except Exception as e:
        logger.warning(f"Failed to load local model: {e}")
    
    # Try 2: Load from MLflow Model Registry
    try:
        import mlflow.pyfunc
        
        model_name = "prices_predictor"
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"
        
        logger.info(f"Attempting to load model from MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = f"{model_name}@{stage}"
        demo_mode = False
        logger.info(f"MLflow model loaded successfully: {model_version}")
        return
        
    except Exception as e:
        logger.warning(f"Failed to load MLflow model: {e}")
    
    # Fallback: Demo mode
    logger.info("Falling back to DEMO mode")
    demo_mode = True
    model_version = "demo-1.0.0"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend interface."""
    from src.frontend import FRONTEND_HTML
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint."""
    return {
        "service": "Housing Prices Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "service": "prices-predictor",
        "model_loaded": model is not None,
        "demo_mode": demo_mode
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction for housing price.
    
    In demo mode, returns a calculated estimate based on key features.
    In production mode, uses the loaded XGBoost model.
    """
    try:
        features = request.features
        
        if demo_mode or model is None:
            predicted_price = calculate_demo_price(features)
        else:
            import pandas as pd
            import numpy as np
            
            # Prepare features in correct order
            if model_features:
                # Fill missing features with 0
                feature_dict = {f: features.get(f, 0) for f in model_features}
                df = pd.DataFrame([feature_dict])[model_features]
            else:
                df = pd.DataFrame([features])
            
            # Predict (model outputs log1p transformed price)
            log_predictions = model.predict(df)
            # Inverse transform: expm1 reverses log1p
            predicted_price = float(np.expm1(log_predictions[0]))
        
        return {
            "predicted_price": round(predicted_price, 2),
            "model_version": model_version or "unknown",
            "demo_mode": demo_mode,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple records.
    Compatible with MLflow serving format (dataframe_records).
    """
    if not request.dataframe_records:
        raise HTTPException(status_code=400, detail="No data provided.")
    
    try:
        if demo_mode or model is None:
            predictions = [calculate_demo_price(record) for record in request.dataframe_records]
        else:
            import pandas as pd
            df = pd.DataFrame(request.dataframe_records)
            predictions = model.predict(df).tolist()
        
        return {
            "predictions": [round(p, 2) for p in predictions],
            "model_version": model_version or "unknown",
            "demo_mode": demo_mode
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
