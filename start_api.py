"""
Simple startup script for the Enhanced Customer Churn Prediction API.

This script starts the FastAPI server with the enhanced ML capabilities.
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting Enhanced Customer Churn Prediction API")
    print("=" * 50)
    print("Features:")
    print("- Multiple ML algorithms (Logistic, RF, XGBoost, LightGBM, etc.)")
    print("- Hyperparameter optimization")
    print("- Advanced preprocessing and feature engineering")
    print("- Comprehensive model evaluation")
    print("- Experiment tracking and model versioning")
    print("- Real-time predictions with confidence scores")
    print("=" * 50)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )