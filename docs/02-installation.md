# Installation Guide

This guide provides step-by-step instructions for setting up the Customer Churn Prediction System in different environments.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for dependencies and models
- **OS**: Windows, macOS, or Linux

### Required Knowledge
- Basic Python programming
- Command line interface usage
- Understanding of virtual environments (recommended)

## 🚀 Quick Installation

### Option 1: Standard Installation

1. **Clone or Download the Project**
   ```bash
   # If using git
   git clone <repository-url>
   cd customer-churn-prediction
   
   # Or download and extract the project files
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # Using venv
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import sklearn, pandas, fastapi; print('Installation successful!')"
   ```

### Option 2: Development Installation

For development with optional advanced features:

```bash
# Install all dependencies including optional ones
pip install -r requirements.txt

# Install optional advanced ML libraries
pip install xgboost lightgbm imbalanced-learn

# Install visualization libraries
pip install matplotlib seaborn

# Install interpretability libraries (optional)
pip install shap lime
```

## 📦 Dependency Overview

### Core Dependencies
```
numpy>=1.21.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
scikit-learn>=1.0.0    # Machine learning algorithms
scipy>=1.7.0           # Scientific computing
fastapi>=0.70.0        # Web framework
uvicorn>=0.15.0        # ASGI server
pydantic>=1.8.0        # Data validation
joblib>=1.1.0          # Model persistence
```

### Advanced ML Libraries (Optional)
```
xgboost>=1.5.0         # Advanced gradient boosting
lightgbm>=3.3.0        # Fast gradient boosting
imbalanced-learn>=0.8.0 # Class imbalance handling
```

### Visualization Libraries (Optional)
```
matplotlib>=3.5.0      # Basic plotting
seaborn>=0.11.0        # Statistical visualization
```

### Interpretability Libraries (Optional)
```
shap>=0.40.0           # Model interpretability
lime>=0.2.0            # Local interpretability
```

### Development Libraries (Optional)
```
pytest>=6.0.0          # Testing framework
pytest-asyncio>=0.18.0 # Async testing
httpx>=0.23.0          # HTTP client for testing
```

## 🔧 Environment Setup

### 1. Python Environment

**Check Python Version**
```bash
python --version
# Should be 3.9 or higher
```

**Install Python (if needed)**
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: Use Homebrew: `brew install python`
- **Linux**: Use package manager: `sudo apt install python3.9`

### 2. Virtual Environment Setup

**Why Use Virtual Environments?**
- Isolate project dependencies
- Avoid conflicts with system packages
- Easy dependency management
- Reproducible environments

**Create and Activate**
```bash
# Create virtual environment
python -m venv churn_env

# Activate (Windows)
churn_env\Scripts\activate

# Activate (macOS/Linux)
source churn_env/bin/activate

# Verify activation (should show virtual env path)
which python
```

### 3. Dependency Installation

**Install Core Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Install Optional Dependencies**
```bash
# For advanced ML features
pip install xgboost lightgbm imbalanced-learn

# For visualization
pip install matplotlib seaborn

# For model interpretability
pip install shap lime

# For development
pip install pytest pytest-asyncio httpx
```

**Verify Installation**
```bash
python -c "
import sklearn
import pandas as pd
import fastapi
import numpy as np
print('Core dependencies installed successfully!')
print(f'scikit-learn: {sklearn.__version__}')
print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
"
```

## 🗂️ Project Structure Setup

After installation, your project structure should look like this:

```
customer-churn-prediction/
├── data/
│   └── dataset.csv              # Customer data
├── docs/                        # Documentation (this folder)
├── metrics/                     # Training outputs
│   ├── models/                 # Saved models
│   ├── plots/                  # Visualizations
│   └── reports/                # Analysis reports
├── model_registry/             # Model versioning
│   ├── models/                 # Versioned models
│   └── metadata/               # Model metadata
├── .venv/                      # Virtual environment (if created)
├── requirements.txt            # Dependencies
├── main.py                     # FastAPI application
├── train.py                    # Training pipeline
├── model.py                    # ML models
├── preprocessor.py             # Data preprocessing
├── evaluate.py                 # Model evaluation
├── inference.py                # Prediction engine
├── save_load.py               # Model persistence
├── data_loader.py             # Data loading
└── start_api.py               # API startup script
```

## ✅ Installation Verification

### 1. Basic Functionality Test

```bash
# Test data loading
python -c "
from data_loader import load_data
df = load_data('data/dataset.csv')
print(f'Data loaded: {df.shape}')
"

# Test model factory
python -c "
from model import ChurnModelFactory
factory = ChurnModelFactory()
models = factory.get_available_models()
print(f'Available models: {models}')
"

# Test preprocessing
python -c "
from preprocessor import build_preprocessor
print('Preprocessor created successfully!')
"
```

### 2. API Test

```bash
# Start the API server
python start_api.py

# In another terminal, test the API
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

### 3. Training Test

```bash
# Run a quick training test
python -c "
from train import train
pipeline = train()
print('Training completed successfully!')
"
```

## 🐛 Troubleshooting Installation

### Common Issues

#### 1. Python Version Issues
**Problem**: `ImportError` or compatibility errors
**Solution**: 
```bash
# Check Python version
python --version

# If version is < 3.9, install newer Python
# Then recreate virtual environment
```

#### 2. Dependency Conflicts
**Problem**: Package version conflicts
**Solution**:
```bash
# Clear pip cache
pip cache purge

# Reinstall with specific versions
pip install --force-reinstall -r requirements.txt
```

#### 3. XGBoost/LightGBM Installation Issues
**Problem**: Compilation errors on Windows/macOS
**Solution**:
```bash
# Use conda instead of pip
conda install xgboost lightgbm

# Or install pre-compiled wheels
pip install --only-binary=all xgboost lightgbm
```

#### 4. Memory Issues During Installation
**Problem**: Installation fails due to memory constraints
**Solution**:
```bash
# Install packages one by one
pip install numpy pandas
pip install scikit-learn
pip install fastapi uvicorn
# ... continue with remaining packages
```

#### 5. Permission Issues
**Problem**: Permission denied errors
**Solution**:
```bash
# Install in user directory
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Platform-Specific Issues

#### Windows
- **Issue**: Long path names
- **Solution**: Enable long path support or use shorter directory names
- **Issue**: Visual C++ compiler missing
- **Solution**: Install Microsoft Visual C++ Build Tools

#### macOS
- **Issue**: Xcode command line tools missing
- **Solution**: `xcode-select --install`
- **Issue**: M1 chip compatibility
- **Solution**: Use conda or install ARM64 compatible wheels

#### Linux
- **Issue**: Missing system libraries
- **Solution**: Install development packages
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

## 🔄 Environment Management

### Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade scikit-learn

# Check for outdated packages
pip list --outdated
```

### Freezing Dependencies

```bash
# Create exact dependency snapshot
pip freeze > requirements-exact.txt

# Install from exact snapshot
pip install -r requirements-exact.txt
```

### Multiple Environments

```bash
# Create environment for different Python versions
python3.9 -m venv env39
python3.10 -m venv env310

# Create environment for different use cases
python -m venv env-dev      # Development
python -m venv env-prod     # Production
python -m venv env-test     # Testing
```

## 🚀 Next Steps

After successful installation:

1. **Verify Data**: Ensure `data/dataset.csv` exists and is accessible
2. **Run Quick Start**: Follow the [Quick Start Guide](03-quick-start.md)
3. **Explore API**: Visit `http://localhost:8000/docs` after starting the API
4. **Read Documentation**: Continue with component-specific documentation

### Development Setup

If you plan to modify the code:

```bash
# Install development dependencies
pip install pytest pytest-asyncio httpx black flake8

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Production Setup

For production deployment:

```bash
# Install production-only dependencies
pip install --no-dev -r requirements.txt

# Set environment variables
export MODEL_REGISTRY_PATH="model_registry"
export METRICS_DIR="metrics"
export LOG_LEVEL="INFO"
```

---

**Next**: Continue to [Quick Start Guide](03-quick-start.md) to run your first predictions.