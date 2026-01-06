# Quick Start Guide

Get up and running with the Customer Churn Prediction System in just a few minutes. This guide covers the essential steps to make your first prediction and train your first model.

## 🚀 5-Minute Quick Start

### Step 1: Verify Installation

```bash
# Ensure you're in the project directory
cd customer-churn-prediction

# Activate virtual environment (if using one)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Verify installation
python -c "import sklearn, pandas, fastapi; print('✅ Ready to go!')"
```

### Step 2: Start the API Server

```bash
# Start the FastAPI server
python start_api.py
```

You should see output like:
```
Starting Enhanced Customer Churn Prediction API
==================================================
Features:
- Multiple ML algorithms (Logistic, RF, XGBoost, LightGBM, etc.)
- Hyperparameter optimization
- Advanced preprocessing and feature engineering
- Comprehensive model evaluation
- Experiment tracking and model versioning
- Real-time predictions with confidence scores
==================================================
API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health
==================================================
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 3: Test the API

Open your browser and visit:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Or test with curl:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-06T...",
  "model_registry_status": "operational",
  "experiment_tracker_status": "operational"
}
```

### Step 4: Make Your First Prediction

#### Option A: Using the Interactive API Documentation

1. Go to http://localhost:8000/docs
2. Find the `/predict` endpoint
3. Click "Try it out"
4. Use this sample data:

```json
{
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
```

5. Click "Execute"

#### Option B: Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

#### Option C: Using Python

```python
import requests

# Customer data
customer_data = {
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

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=customer_data
)

print(response.json())
```

Expected response:
```json
{
  "churn": false,
  "churn_probability": 0.35,
  "confidence": "medium",
  "risk_factors": [
    "Short-term contract",
    "Electronic check payment"
  ],
  "model_version": "1.0",
  "prediction_timestamp": "2024-01-06T..."
}
```

## 🎯 Training Your First Model

### Option 1: Simple Training (Python Script)

```python
# simple_training.py
from train import train

# Train a simple logistic regression model
print("Training simple model...")
pipeline = train()
print("✅ Training completed!")

# Test the trained model
from inference import predict

result = predict({
    "gender": "Female",
    "tenure": 12,
    "MonthlyCharges": 29.85,
    "TotalCharges": "358.2",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    # ... add other required fields
})

print(f"Prediction: {result}")
```

### Option 2: Advanced Training (API)

```bash
# Train multiple models with optimization
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "models_to_train": ["logistic", "random_forest"],
       "optimize_hyperparameters": true,
       "advanced_preprocessing": true,
       "feature_engineering": true
     }'
```

### Option 3: Advanced Training (Python)

```python
from train import ChurnTrainer

# Initialize trainer
trainer = ChurnTrainer()

# Load and split data
trainer.load_and_split_data()

# Setup advanced preprocessing
trainer.setup_preprocessing(
    advanced=True,
    feature_engineering=True,
    handle_imbalance=False
)

# Train multiple models
results = trainer.train_multiple_models(
    model_names=['logistic', 'random_forest', 'gradient_boosting'],
    optimize_hyperparameters=True
)

# Evaluate on test set
test_metrics = trainer.evaluate_on_test_set()

# Generate comparison report
comparison_df = trainer.generate_model_comparison_report()
print(comparison_df)
```

## 📊 Understanding the Results

### Prediction Response Fields

```json
{
  "churn": false,                    // Binary prediction (true/false)
  "churn_probability": 0.35,         // Probability of churn (0-1)
  "confidence": "medium",            // Confidence level (low/medium/high)
  "risk_factors": [...],             // Key risk factors identified
  "model_version": "1.0",            // Model version used
  "prediction_timestamp": "..."      // When prediction was made
}
```

### Confidence Levels
- **High**: Probability < 0.3 or > 0.7 (model is very confident)
- **Medium**: Probability 0.4-0.6 or 0.3-0.4/0.6-0.7 (moderate confidence)
- **Low**: Probability 0.4-0.6 (model is uncertain)

### Risk Factors
Common risk factors the model identifies:
- Short-term contracts (month-to-month)
- High monthly charges
- Electronic check payments
- Low tenure (new customers)
- Lack of additional services

## 🔍 Exploring the System

### 1. Check Available Models

```bash
curl http://localhost:8000/models
```

### 2. View Model Comparison

```bash
curl http://localhost:8000/models/compare
```

### 3. Get Training Status

```bash
# Start training
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{}'

# Check status
curl http://localhost:8000/train/status
```

### 4. View Metrics

```bash
curl http://localhost:8000/metrics/comparison
```

## 🛠️ Common Use Cases

### Use Case 1: Batch Predictions

```python
import pandas as pd
import requests

# Load customer data
customers = pd.read_csv('customer_data.csv')

# Make predictions for each customer
predictions = []
for _, customer in customers.iterrows():
    response = requests.post(
        "http://localhost:8000/predict",
        json=customer.to_dict()
    )
    predictions.append(response.json())

# Convert to DataFrame
results_df = pd.DataFrame(predictions)
print(f"High-risk customers: {results_df['churn'].sum()}")
```

### Use Case 2: Model Comparison

```python
from train import ChurnTrainer

trainer = ChurnTrainer()
trainer.load_and_split_data()
trainer.setup_preprocessing(advanced=True)

# Train multiple models
models = ['logistic', 'random_forest', 'xgboost']
results = trainer.train_multiple_models(models)

# Compare performance
comparison = trainer.generate_model_comparison_report()
print("Best model:", trainer.best_model_name)
print("Best score:", trainer.models[trainer.best_model_name]['metrics']['val_roc_auc'])
```

### Use Case 3: Custom Preprocessing

```python
from preprocessor import AdvancedPreprocessor
from data_loader import load_data

# Load data
df = load_data('data/dataset.csv')
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Custom preprocessing
preprocessor = AdvancedPreprocessor(
    scaling_strategy='robust',
    feature_engineering=True,
    handle_outliers=True
)

# Fit and transform
X_processed = preprocessor.fit_transform(X, y)
print(f"Original features: {X.shape[1]}")
print(f"Processed features: {X_processed.shape[1]}")
```

## ⚡ Performance Tips

### 1. Speed Up Training
```python
# Use fewer models for faster training
trainer.train_multiple_models(['logistic', 'random_forest'])

# Disable hyperparameter optimization for speed
trainer.train_multiple_models(optimize_hyperparameters=False)

# Use simpler preprocessing
trainer.setup_preprocessing(advanced=False)
```

### 2. Improve Accuracy
```python
# Enable all advanced features
trainer.setup_preprocessing(
    advanced=True,
    feature_engineering=True,
    handle_imbalance=True
)

# Train more models
models = ['logistic', 'random_forest', 'gradient_boosting', 'xgboost']
trainer.train_multiple_models(models, optimize_hyperparameters=True)
```

### 3. Monitor API Performance
```bash
# Check API health
curl http://localhost:8000/health

# Monitor training progress
curl http://localhost:8000/train/status
```

## 🚨 Troubleshooting Quick Issues

### Issue 1: API Won't Start
```bash
# Check if port 8000 is in use
netstat -an | grep 8000

# Use different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Issue 2: Prediction Errors
- Ensure all required fields are provided
- Check data types match the expected format
- Verify numeric fields are actually numeric

### Issue 3: Training Fails
```python
# Check data file exists
import os
print(os.path.exists('data/dataset.csv'))

# Check data format
from data_loader import load_data
df = load_data('data/dataset.csv')
print(df.head())
print(df.info())
```

### Issue 4: Memory Issues
```python
# Use smaller dataset for testing
df_sample = df.sample(n=1000)

# Disable advanced features
trainer.setup_preprocessing(advanced=False, feature_engineering=False)
```

## 🎉 What's Next?

Now that you have the system running:

1. **Explore the API**: Visit http://localhost:8000/docs for full API documentation
2. **Understand the Data**: Read [Data Pipeline](04-data-pipeline.md) to understand the input format
3. **Learn About Models**: Check [Model Architecture](05-model-architecture.md) for algorithm details
4. **Advanced Training**: Read [Training Pipeline](06-training-pipeline.md) for comprehensive training options
5. **Production Deployment**: See [Deployment Guide](12-deployment.md) for production setup

### Sample Workflows

**For Data Scientists:**
1. Start with [Training Pipeline](06-training-pipeline.md)
2. Explore [Evaluation System](07-evaluation.md)
3. Learn [Feature Engineering](11-feature-engineering.md)

**For Developers:**
1. Study [API Documentation](09-api-documentation.md)
2. Check [Code Examples](18-examples.md)
3. Review [API Reference](16-api-reference.md)

**For DevOps:**
1. Read [Deployment Guide](12-deployment.md)
2. Study [Monitoring](13-monitoring.md)
3. Check [Performance Optimization](15-performance.md)

---

**Congratulations!** 🎉 You've successfully set up and tested the Customer Churn Prediction System. Continue with the detailed documentation to explore advanced features and customization options.