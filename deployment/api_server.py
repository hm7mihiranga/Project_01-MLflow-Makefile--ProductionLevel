"""
FastAPI server for Telco Customer Churn Prediction API.

This module provides a RESTful API for churn prediction using the trained model.
"""

import os
import sys
import logging
import uvicorn
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
import pandas as pd

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

try:
    from model_inference import ModelInference
    import config
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    print("Make sure you're running from the project root directory")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-ready API for predicting customer churn in telecommunications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model inference instance
model_inference = None


class CustomerData(BaseModel):
    """Pydantic model for customer data validation."""
    
    customerID: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen flag (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, le=100, description="Tenure in months")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines status")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Online security service")
    OnlineBackup: str = Field(..., description="Online backup service")
    DeviceProtection: str = Field(..., description="Device protection service")
    TechSupport: str = Field(..., description="Tech support service")
    StreamingTV: str = Field(..., description="Streaming TV service")
    StreamingMovies: str = Field(..., description="Streaming movies service")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges amount")
    TotalCharges: str = Field(..., description="Total charges (can be string or numeric)")
    
    class Config:
        schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85"
            }
        }


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    
    success: bool = Field(..., description="Prediction success status")
    customer_id: str = Field(..., description="Customer identifier")
    prediction: int = Field(..., description="Churn prediction (0=Retain, 1=Churn)")
    churn_status: str = Field(..., description="Human-readable churn status")
    churn_probability: float = Field(..., description="Probability of churn")
    retain_probability: float = Field(..., description="Probability of retention")
    confidence_score: float = Field(..., description="Confidence score (0-100%)")
    risk_level: str = Field(..., description="Risk categorization")
    recommendation: str = Field(..., description="Business recommendation")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")
    model_type: str = Field(..., description="Model type used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Pydantic model for batch prediction requests."""
    
    customers: List[CustomerData] = Field(..., description="List of customer data")
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "customerID": "DEMO-001",
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "Yes",
                        "Dependents": "No",
                        "tenure": 1,
                        "PhoneService": "No",
                        "MultipleLines": "No phone service",
                        "InternetService": "DSL",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 29.85,
                        "TotalCharges": "29.85"
                    }
                ]
            }
        }


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_inference
    
    logger.info("="*60)
    logger.info("STARTING TELCO CHURN PREDICTION API")
    logger.info("="*60)
    
    try:
        # Initialize model inference
        logger.info("Loading model inference engine...")
        model_inference = ModelInference()
        logger.info("✓ Model inference engine loaded successfully")
        
        logger.info("API startup completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error("API will start but predictions will fail")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model_inference
    
    return {
        "status": "healthy",
        "model_loaded": model_inference is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """
    Predict churn for a single customer.
    
    Args:
        customer_data: Customer information for prediction
        
    Returns:
        PredictionResponse: Churn prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    global model_inference
    
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic model to dict
        customer_dict = customer_data.dict()
        
        # Make prediction
        result = model_inference.predict_single_customer(customer_dict)
        
        # Determine risk level and recommendation
        churn_prob = result['churn_probability']
        if churn_prob > 0.7:
            risk_level = "HIGH RISK"
            recommendation = "Immediate retention campaign required"
        elif churn_prob > 0.4:
            risk_level = "MEDIUM RISK"
            recommendation = "Monitor closely and engage proactively"
        else:
            risk_level = "LOW RISK"
            recommendation = "Continue standard service level"
        
        # Create response
        response = PredictionResponse(
            success=True,
            customer_id=result['customer_id'],
            prediction=result['prediction'],
            churn_status=result['churn_status'],
            churn_probability=result['churn_probability'],
            retain_probability=result['retain_probability'],
            confidence_score=result['confidence_score'],
            risk_level=risk_level,
            recommendation=recommendation,
            prediction_time_ms=result['prediction_time_ms'],
            model_type=result['model_type'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed for customer {result['customer_id']}: {result['churn_status']}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch_churn(batch_request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Args:
        batch_request: Batch of customer data for prediction
        
    Returns:
        List of prediction results
        
    Raises:
        HTTPException: If batch prediction fails
    """
    global model_inference
    
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic models to list of dicts
        customers_data = [customer.dict() for customer in batch_request.customers]
        
        # Make batch prediction
        results = model_inference.predict_batch(customers_data)
        
        # Format responses
        responses = []
        for result in results:
            if result.get('prediction', -1) != -1:  # Successful prediction
                churn_prob = result['churn_probability']
                if churn_prob > 0.7:
                    risk_level = "HIGH RISK"
                    recommendation = "Immediate retention campaign required"
                elif churn_prob > 0.4:
                    risk_level = "MEDIUM RISK"
                    recommendation = "Monitor closely and engage proactively"
                else:
                    risk_level = "LOW RISK"
                    recommendation = "Continue standard service level"
                
                response = PredictionResponse(
                    success=True,
                    customer_id=result['customer_id'],
                    prediction=result['prediction'],
                    churn_status=result['churn_status'],
                    churn_probability=result['churn_probability'],
                    retain_probability=result['retain_probability'],
                    confidence_score=result['confidence_score'],
                    risk_level=risk_level,
                    recommendation=recommendation,
                    prediction_time_ms=result['prediction_time_ms'],
                    model_type=result['model_type'],
                    timestamp=datetime.now().isoformat()
                )
            else:  # Failed prediction
                response = PredictionResponse(
                    success=False,
                    customer_id=result['customer_id'],
                    prediction=-1,
                    churn_status="Error",
                    churn_probability=0.0,
                    retain_probability=0.0,
                    confidence_score=0.0,
                    risk_level="UNKNOWN",
                    recommendation="Prediction failed - check data quality",
                    prediction_time_ms=0.0,
                    model_type="N/A",
                    timestamp=datetime.now().isoformat()
                )
            
            responses.append(response)
        
        logger.info(f"Batch prediction completed for {len(customers_data)} customers")
        return {
            "success": True,
            "total_customers": len(customers_data),
            "successful_predictions": len([r for r in responses if r.success]),
            "predictions": responses,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded model."""
    global model_inference
    
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Get model information
        model_type = type(model_inference.model).__name__
        
        return {
            "model_loaded": True,
            "model_type": model_type,
            "model_path": model_inference.model_path,
            "encoders_loaded": len(model_inference.encoders),
            "batch_size": getattr(model_inference, 'batch_size', 1000),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


if __name__ == "__main__":
    # Run the API server
    logger.info("Starting Telco Churn Prediction API server...")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8080,
        log_level="info",
        reload=False
    )