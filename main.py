"""
FastAPI Application for ML Model Predictions
Provides REST API endpoints for product success prediction

This is a SEPARATE service from Streamlit and runs independently.
Streamlit deployment is NOT affected by this file.

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Product Success Prediction API",
    description="ML API for predicting product success for The Whole Truth Foods",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow requests from any origin (useful for web apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    """Input schema for prediction endpoint"""
    price: float = Field(..., description="Product price in INR", example=1047)
    discount: float = Field(..., description="Discount percentage (0-100)", example=0)
    category: str = Field(..., description="Product category", example="Dark Chocolate")
    ingredients_count: int = Field(..., description="Number of ingredients", example=4)
    has_dates: int = Field(..., description="Contains dates (0 or 1)", example=1)
    has_cocoa: int = Field(..., description="Contains cocoa (0 or 1)", example=1)
    has_protein: int = Field(..., description="Contains protein (0 or 1)", example=0)
    packaging_type: str = Field(..., description="Type of packaging", example="Paper-based")
    season: str = Field(..., description="Season", example="Winter")
    customer_gender: float = Field(..., description="Customer gender (0=Male, 1=Female)", example=1.0)
    age_numeric: float = Field(..., description="Customer age (18-100)", example=55)
    shelf_life: float = Field(..., description="Shelf life in months", example=12)
    clean_label: int = Field(..., description="Clean label certified (0 or 1)", example=1)

    class Config:
        schema_extra = {
            "example": {
                "price": 1047,
                "discount": 0,
                "category": "Dark Chocolate",
                "ingredients_count": 4,
                "has_dates": 1,
                "has_cocoa": 1,
                "has_protein": 0,
                "packaging_type": "Paper-based",
                "season": "Winter",
                "customer_gender": 1.0,
                "age_numeric": 55,
                "shelf_life": 12,
                "clean_label": 1
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction endpoint"""
    prediction: int = Field(..., description="Predicted class (0=Failure, 1=Success)")
    probability: float = Field(..., description="Probability of success (0-1)")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    model_used: str = Field(..., description="Name of the model used")


# Global variables for model and preprocessor
model = None
preprocessor = None
model_name = "KNN"  # Default model to use


def load_model_and_preprocessor():
    """Load the trained model and preprocessor from disk"""
    global model, preprocessor, model_name
    
    model_dir = "dsmodelpickl+preprocessor"
    
    try:
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info("✅ Preprocessor loaded successfully")
        
        # Load model (using KNN as default - best performing non-overfitting model)
        model_path = os.path.join(model_dir, "knn_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"✅ Model ({model_name}) loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model/preprocessor: {str(e)}")
        return False


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    logger.info("🚀 Starting FastAPI application...")
    success = load_model_and_preprocessor()
    if success:
        logger.info("✅ Application ready to serve predictions")
    else:
        logger.error("❌ Application started but model loading failed")


@app.get("/")
def read_root():
    """Root endpoint - Welcome message"""
    return {
        "message": "Welcome to the Model Prediction API!",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None
    
    return {
        "status": "healthy" if (model_loaded and preprocessor_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "model_name": model_name if model_loaded else None
    }


@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features_required": [
            "price", "discount", "category", "ingredients_count",
            "has_dates", "has_cocoa", "has_protein", "packaging_type",
            "season", "customer_gender", "age_numeric", "shelf_life", "clean_label"
        ],
        "output": {
            "prediction": "0 (Failure) or 1 (Success)",
            "probability": "Float between 0 and 1",
            "confidence": "Low, Medium, or High"
        }
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Make a prediction for product success
    
    Args:
        input_data: Product and customer features
    
    Returns:
        Prediction result with probability and confidence
    """
    # Check if model is loaded
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Check /health endpoint."
        )
    
    try:
        # Convert input to dictionary
        features_dict = input_data.dict()
        
        # Create DataFrame with proper column names
        features_df = pd.DataFrame([features_dict])
        
        # Log the input
        logger.info(f"Received prediction request: {features_dict}")
        
        # Apply preprocessing
        features_processed = preprocessor.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_processed)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_processed)
            probability = float(proba[0][1])  # Probability of success (class 1)
        else:
            # For models without predict_proba
            probability = float(prediction)
        
        # Determine confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = "High"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        result = {
            "prediction": int(prediction),
            "probability": round(probability, 4),
            "confidence": confidence,
            "model_used": model_name
        }
        
        logger.info(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch-predict")
def batch_predict(inputs: list[PredictionInput]):
    """
    Make predictions for multiple inputs at once
    
    Args:
        inputs: List of product and customer features
    
    Returns:
        List of prediction results
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded"
        )
    
    try:
        results = []
        for input_data in inputs:
            result = predict(input_data)
            results.append(result)
        
        return {
            "predictions": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
