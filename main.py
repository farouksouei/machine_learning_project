"""
Enhanced FastAPI Application for Customer Churn Prediction

This API provides comprehensive machine learning capabilities including:
- Model training with multiple algorithms
- Hyperparameter optimization
- Model evaluation and comparison
- Prediction endpoints with confidence scores
- Metrics tracking and experiment management
- Model versioning and deployment
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path as PathParam
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator

# ML and data processing imports
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Local imports
from inference import predict, get_available_models, set_active_model, get_active_model_id
from train import ChurnTrainer, train_churn_model
from save_load import ModelRegistry, create_metrics_directory
from data_loader import load_data

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Customer Churn Prediction API",
    description="""
    A comprehensive machine learning API for customer churn prediction featuring:
    
    ## Features
    - **Multiple ML Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks
    - **Hyperparameter Optimization**: Grid search and random search with cross-validation
    - **Advanced Preprocessing**: Feature engineering, outlier handling, class imbalance correction
    - **Model Evaluation**: Comprehensive metrics, ROC curves, feature importance analysis
    - **Experiment Tracking**: Full experiment logging and comparison
    - **Model Versioning**: Model registry with version control
    - **Real-time Predictions**: Fast inference with confidence scores
    
    ## Endpoints
    - `/predict` - Make predictions on customer data
    - `/train` - Train models with advanced ML techniques
    - `/models` - Model management and comparison
    - `/experiments` - Experiment tracking and analysis
    - `/metrics` - Performance metrics and visualizations
    """,
    version="2.0.0",
    contact={
        "name": "ML Engineering Team",
        "email": "ml-team@company.com"
    }
)

# Add CORS middleware - wildcard configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # ← allows all origins
    allow_credentials=True,        # important if you use cookies/auth headers
    allow_methods=["*"],           # allows GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"],           # allows any header (Content-Type, Authorization, etc.)
)

# Initialize components
model_registry = ModelRegistry()
create_metrics_directory()

# Global variables for current model
current_trainer = None
training_status = {"status": "idle", "progress": 0, "message": ""}


# Pydantic models for request/response validation
class CustomerData(BaseModel):
    """Customer data for churn prediction."""
    
    # Demographic information
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen flag (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    
    # Service information
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Internet service type (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(..., description="Online security (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Online backup (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Device protection (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Tech support (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Streaming TV (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Streaming movies (Yes/No/No internet service)")
    
    # Contract information
    Contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    
    # Financial information
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: Union[float, str] = Field(..., description="Total charges (numeric or string)")
    
    @validator('TotalCharges')
    def validate_total_charges(cls, v):
        """Convert TotalCharges to float, handling empty strings."""
        if isinstance(v, str):
            if v.strip() == '' or v.strip().lower() == 'nan':
                return 0.0
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid TotalCharges value: {v}")
        return float(v)
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
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
                "TotalCharges": 358.2
            }
        }


class TrainingRequest(BaseModel):
    """Request model for training configuration."""
    
    models_to_train: Optional[List[str]] = Field(
        default=None,
        description="List of models to train. If None, trains all available models."
    )
    optimize_hyperparameters: bool = Field(
        default=True,
        description="Whether to perform hyperparameter optimization"
    )
    search_type: str = Field(
        default="random",
        description="Search strategy: 'grid' or 'random'"
    )
    n_iter: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Number of iterations for random search"
    )
    use_ensemble: bool = Field(
        default=True,
        description="Whether to create ensemble models"
    )
    advanced_preprocessing: bool = Field(
        default=True,
        description="Whether to use advanced preprocessing techniques"
    )
    feature_engineering: bool = Field(
        default=True,
        description="Whether to create interaction features"
    )
    handle_imbalance: bool = Field(
        default=True,
        description="Whether to handle class imbalance"
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="Name for the training run (auto-generated if None)"
    )
    
    @validator('models_to_train')
    def validate_models(cls, v):
        """Validate model names."""
        if v is not None:
            valid_models = [
                'logistic', 'random_forest', 'gradient_boosting',
                'xgboost', 'lightgbm', 'svm', 'neural_network'
            ]
            invalid_models = [m for m in v if m not in valid_models]
            if invalid_models:
                raise ValueError(f"Invalid models: {invalid_models}. Valid options: {valid_models}")
        return v
    
    @validator('search_type')
    def validate_search_type(cls, v):
        """Validate search type."""
        if v not in ['grid', 'random']:
            raise ValueError("search_type must be 'grid' or 'random'")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    churn: bool = Field(..., description="Predicted churn (True/False)")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    risk_factors: List[str] = Field(..., description="Key risk factors identified")
    model_version: str = Field(..., description="Version of model used")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")


class TrainingResponse(BaseModel):
    """Response model for training results."""
    
    status: str = Field(..., description="Training status")
    models_trained: List[str] = Field(..., description="List of models that were trained")
    best_model: str = Field(..., description="Name of the best performing model")
    best_score: float = Field(..., description="Best validation score achieved")
    training_time: float = Field(..., description="Total training time in seconds")
    message: str = Field(..., description="Additional information")


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Advanced Customer Churn Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Multiple ML algorithms",
            "Hyperparameter optimization",
            "Advanced preprocessing",
            "Model evaluation and comparison",
            "Experiment tracking",
            "Model versioning"
        ],
        "endpoints": {
            "prediction": "/predict",
            "training": "/train",
            "models": "/models",
            "experiments": "/experiments",
            "metrics": "/metrics"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_registry_status": "operational",
        "experiment_tracker_status": "operational"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(
    customer_data: CustomerData,
    model_id: Optional[str] = Query(default=None, description="Model ID to use for prediction. If not specified, uses the active model.")
):
    """
    Predict customer churn with confidence scores and risk analysis.

    This endpoint uses the specified model (or active model) to make predictions
    and provides detailed analysis including confidence levels and risk factors.

    You can specify a model_id to use a specific model, or leave it empty to use
    the currently active model.
    """
    try:
        # Convert Pydantic model to dictionary
        data_dict = customer_data.dict()

        # Make prediction using the inference module with optional model_id
        basic_prediction = predict(data_dict, model_id=model_id)

        # Check for errors in prediction
        if "error" in basic_prediction:
            raise HTTPException(status_code=500, detail=basic_prediction["error"])

        # Get probability from the prediction
        churn_prob = basic_prediction.get("probability", 0.5)

        # Determine confidence level based on probability
        if churn_prob < 0.3 or churn_prob > 0.7:
            confidence = "High"
        elif churn_prob < 0.4 or churn_prob > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Identify risk factors (simplified logic)
        risk_factors = []
        if customer_data.Contract == "Month-to-month":
            risk_factors.append("Short-term contract")
        if customer_data.tenure < 12:
            risk_factors.append("Low tenure")
        if customer_data.MonthlyCharges > 70:
            risk_factors.append("High monthly charges")
        if customer_data.PaymentMethod == "Electronic check":
            risk_factors.append("Electronic check payment")

        return PredictionResponse(
            churn=basic_prediction["churn"],
            churn_probability=churn_prob,
            confidence=confidence,
            risk_factors=risk_factors,
            model_version=basic_prediction.get("model_id", "unknown"),
            prediction_timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_models(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Train machine learning models with advanced techniques.
    
    This endpoint initiates model training with the specified configuration.
    Training runs in the background and progress can be monitored via the
    /train/status endpoint.
    """
    try:
        # Update training status
        training_status["status"] = "starting"
        training_status["message"] = "Initializing training pipeline..."
        
        # Start training in background
        background_tasks.add_task(
            run_training_pipeline,
            training_request
        )
        
        return TrainingResponse(
            status="started",
            models_trained=[],
            best_model="pending",
            best_score=0.0,
            training_time=0.0,
            message="Training started in background. Check /train/status for progress."
        )
        
    except Exception as e:
        training_status["status"] = "failed"
        training_status["message"] = f"Training failed to start: {str(e)}"
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")


@app.get("/train/status", tags=["Training"])
async def get_training_status():
    """Get current training status and progress."""
    return training_status


async def run_training_pipeline(training_request: TrainingRequest):
    """
    Background task to run the training pipeline.
    
    This function executes the actual training process and updates
    the global training status.
    """
    global current_trainer
    
    try:
        training_status["status"] = "running"
        training_status["progress"] = 10
        training_status["message"] = "Loading and preprocessing data..."
        
        start_time = datetime.now()
        
        # Initialize trainer
        trainer = ChurnTrainer()
        
        # Load and split data
        trainer.load_and_split_data()
        training_status["progress"] = 20
        training_status["message"] = "Data loaded and split successfully"
        
        # Setup preprocessing
        trainer.setup_preprocessing(
            advanced=training_request.advanced_preprocessing,
            feature_engineering=training_request.feature_engineering,
            handle_imbalance=training_request.handle_imbalance
        )
        training_status["progress"] = 30
        training_status["message"] = "Preprocessing pipeline configured"
        
        # Train models
        training_status["message"] = "Training models..."
        results = trainer.train_multiple_models(
            model_names=training_request.models_to_train,
            optimize_hyperparameters=training_request.optimize_hyperparameters
        )
        training_status["progress"] = 70
        
        training_status["progress"] = 80
        training_status["message"] = "Evaluating models on test set..."
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_on_test_set()
        
        # Generate comparison report
        comparison_df = trainer.generate_model_comparison_report()
        
        training_status["progress"] = 90
        training_status["message"] = "Saving results..."
        
        # Update global trainer
        current_trainer = trainer
        
        # Calculate training time
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Update final status
        training_status["status"] = "completed"
        training_status["progress"] = 100
        training_status["message"] = f"Training completed successfully. Best model: {trainer.best_model_name}"
        training_status["best_model"] = trainer.best_model_name
        training_status["best_score"] = trainer.models[trainer.best_model_name]['metrics']['val_roc_auc']
        training_status["training_time"] = training_time
        training_status["models_trained"] = list(trainer.models.keys())
        
    except Exception as e:
        training_status["status"] = "failed"
        training_status["message"] = f"Training failed: {str(e)}"
        print(f"Training error: {str(e)}")


@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models and their performance metrics."""
    try:
        models_df = model_registry.list_models()
        if models_df.empty:
            return {"message": "No models registered yet", "models": []}
        
        return {
            "models": models_df.to_dict('records'),
            "total_models": len(models_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/compare", tags=["Models"])
async def compare_models(
    metric: str = Query(default="val_roc_auc", description="Metric to use for comparison")
):
    """Compare all models by a specific metric."""
    try:
        comparison_df = model_registry.compare_models(metric=metric)
        if comparison_df.empty:
            return {"message": "No models available for comparison", "comparison": []}
        
        return {
            "comparison": comparison_df.to_dict('records'),
            "metric_used": metric,
            "best_model": comparison_df.iloc[0].to_dict() if not comparison_df.empty else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")


@app.get("/metrics/comparison", tags=["Metrics"])
async def get_model_comparison():
    """Get model comparison metrics."""
    comparison_path = "metrics/model_comparison.csv"

    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=404, detail="Model comparison not available")

    try:
        comparison_df = pd.read_csv(comparison_path)
        return {
            "comparison": comparison_df.to_dict('records'),
            "summary": {
                "total_models": len(comparison_df),
                "best_model": comparison_df.iloc[0]['Model'] if not comparison_df.empty else None,
                "best_score": comparison_df.iloc[0]['Val_ROC_AUC'] if not comparison_df.empty else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load comparison: {str(e)}")


# Model Selection Endpoints

@app.get("/models/active", tags=["Models"])
async def get_active_model_endpoint():
    """Get the currently active model for predictions."""
    active_id = get_active_model_id()
    if active_id is None:
        return {
            "active_model": None,
            "message": "No active model set. The best model will be used for predictions."
        }
    return {
        "active_model": active_id,
        "message": f"Active model is set to: {active_id}"
    }


@app.post("/models/active/{model_id}", tags=["Models"])
async def set_active_model_endpoint(model_id: str):
    """
    Set the active model for predictions.

    Args:
        model_id: The model ID to set as active (e.g., "logistic_v20260112_221304")
    """
    available = get_available_models()
    if model_id not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available models: {available}"
        )

    success = set_active_model(model_id)
    if success:
        return {
            "status": "success",
            "message": f"Active model set to: {model_id}",
            "active_model": model_id
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to set active model")


@app.get("/models/available", tags=["Models"])
async def list_available_models():
    """List all available model IDs that can be used for predictions."""
    available = get_available_models()
    active = get_active_model_id()

    return {
        "available_models": available,
        "total": len(available),
        "active_model": active
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")