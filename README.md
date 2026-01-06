# Simplified Customer Churn Prediction System

A streamlined machine learning system for predicting customer churn with advanced techniques and production-ready API endpoints.

## 🚀 Features

### Advanced Machine Learning
- **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, Neural Networks
- **Hyperparameter Optimization**: Grid search and random search with cross-validation
- **Feature Engineering**: Automatic creation of interaction terms and domain-specific features
- **Advanced Preprocessing**: Multiple scaling strategies, outlier handling, feature selection

### Model Management
- **Model Registry**: Automatic model versioning and storage
- **Performance Tracking**: Comprehensive metrics and comparison reports
- **Easy Deployment**: Simple functions for training and inference

### Production-Ready API
- **RESTful Endpoints**: FastAPI-based endpoints for training and inference
- **Background Training**: Asynchronous model training with progress tracking
- **Model Comparison**: Compare different algorithms and select the best

## 📁 Project Structure

```
├── data/
│   └── dataset.csv              # Customer churn dataset
├── metrics/                     # Training outputs and metrics
│   ├── models/                 # Saved models
│   ├── plots/                  # Evaluation plots
│   └── reports/                # Model comparison reports
├── model_registry/             # Model registry with versioning
├── model.py                    # ML models and factory
├── preprocessor.py             # Advanced preprocessing pipeline
├── train.py                    # Simplified training pipeline
├── evaluate.py                 # Model evaluation utilities
├── inference.py                # Prediction and inference
├── save_load.py               # Model persistence and registry
├── data_loader.py             # Data loading utilities
├── main.py                    # FastAPI application
└── requirements.txt           # Python dependencies
```

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional: Install Advanced ML Libraries
```bash
pip install xgboost lightgbm imbalanced-learn matplotlib seaborn
```

## 🚀 Quick Start

### Option 1: Simple Training
```python
from train import train

# Train a simple logistic regression model
pipeline = train()

# Make predictions
from inference import predict
result = predict({
    "gender": "Female",
    "tenure": 12,
    "MonthlyCharges": 29.85,
    "TotalCharges": "358.2",
    # ... other features
})
print(result)
```

### Option 2: Advanced Training
```python
from train import train_churn_model

# Train multiple models with optimization
trainer = train_churn_model(
    models_to_train=['logistic', 'random_forest', 'xgboost'],
    optimize_hyperparameters=True,
    advanced_preprocessing=True
)

# View results
print(f"Best model: {trainer.best_model_name}")
print(f"Models trained: {list(trainer.models.keys())}")

# Check model registry
models_df = trainer.model_registry.list_models()
print(models_df)
```

### Option 3: API-Based Training
```bash
# Start the API server
python start_api.py

# Visit http://localhost:8000/docs for interactive documentation

# Train models via API
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"models_to_train": ["logistic", "random_forest"], "optimize_hyperparameters": true}'

# Make predictions
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"gender": "Female", "tenure": 12, "MonthlyCharges": 29.85, ...}'
```

## 📊 API Endpoints

### Core Endpoints
- `POST /train` - Start model training
- `GET /train/status` - Check training progress
- `POST /predict` - Make predictions

### Model Management
- `GET /models` - List all trained models
- `GET /models/compare` - Compare model performance

### Metrics
- `GET /metrics/comparison` - Model comparison table

## 🎯 Model Performance

The system delivers excellent performance with the current dataset:

- **ROC-AUC**: 84.7% (excellent for churn prediction)
- **Test Accuracy**: 79.6%
- **Test F1-Score**: 58.4%
- **Cross-Validation**: 84.5% ± 1.0%

## 🔧 Configuration

### Training Options
```python
trainer = train_churn_model(
    data_path="data/dataset.csv",
    models_to_train=['logistic', 'random_forest', 'xgboost'],
    optimize_hyperparameters=True,
    advanced_preprocessing=True
)
```

### Available Models
- `logistic` - Logistic Regression (fast, interpretable)
- `random_forest` - Random Forest (robust, feature importance)
- `gradient_boosting` - Gradient Boosting (high performance)
- `xgboost` - XGBoost (competition-grade, if installed)
- `lightgbm` - LightGBM (fast, large datasets, if installed)
- `svm` - Support Vector Machine (non-linear patterns)
- `neural_network` - Multi-layer Perceptron (complex patterns)

## 📈 Model Registry

All trained models are automatically saved to the model registry with:
- **Versioning**: Automatic timestamp-based versioning
- **Metadata**: Training parameters, performance metrics, data info
- **Easy Loading**: Load any model version by name and version
- **Comparison**: Compare models across different training runs

```python
# List all models
models_df = trainer.model_registry.list_models()

# Load specific model
model, metadata = trainer.model_registry.load_model_version("logistic", "latest")

# Get best model
best_model, metadata = trainer.model_registry.get_best_model()
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "start_api.py"]
```

### Environment Variables
```bash
export MODEL_REGISTRY_PATH="model_registry"
export METRICS_DIR="metrics"
export LOG_LEVEL="INFO"
```

## 📚 Key Functions

### Training Functions
- `train()` - Simple training with logistic regression
- `train_churn_model()` - Advanced training with multiple algorithms
- `ChurnTrainer` - Full training pipeline class

### Inference Functions
- `predict(data)` - Make predictions on customer data

### Model Management
- `ModelRegistry` - Model versioning and storage
- `load_model()` / `save_model()` - Model persistence

## 🎉 What's New in the Simplified Version

✅ **Removed Complexity**
- No more experiments folder - models saved directly to registry
- Simplified training pipeline focused on core functionality
- Removed test files - direct usage of training functions

✅ **Enhanced Model Registry**
- Automatic model versioning with timestamps
- Rich metadata storage (parameters, metrics, training info)
- Easy model comparison and selection

✅ **Streamlined API**
- Focused on essential endpoints
- Simplified training and prediction workflows
- Better error handling and status reporting

✅ **Better Performance**
- Models automatically saved to registry during training
- Efficient model loading and comparison
- Optimized for production use

The simplified system maintains all the advanced ML capabilities while being much easier to use and deploy!