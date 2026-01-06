# Model Architecture Documentation

This document provides comprehensive coverage of the machine learning models and algorithms used in the Customer Churn Prediction System. The system implements a factory pattern to provide access to multiple algorithms with consistent interfaces and advanced optimization capabilities.

## 🏗️ Architecture Overview

### Model Factory Pattern

The system uses a factory pattern (`ChurnModelFactory`) to provide unified access to multiple machine learning algorithms:

```python
from model import ChurnModelFactory

# Initialize factory
factory = ChurnModelFactory(random_state=42)

# Get available models
models = factory.get_available_models()
print(f"Available models: {models}")

# Create specific model
logistic_model = factory.get_model('logistic')
random_forest = factory.get_model('random_forest')
```

### Supported Algorithms

The system supports 7 different machine learning algorithms, each optimized for binary classification:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential ensemble method
4. **XGBoost** - Advanced gradient boosting (optional)
5. **LightGBM** - Fast gradient boosting (optional)
6. **Support Vector Machine** - Kernel-based classifier
7. **Neural Network** - Multi-layer perceptron

## 🤖 Individual Model Details

### 1. Logistic Regression

**Purpose**: Linear baseline model for interpretable predictions

**Characteristics**:
- Fast training and prediction
- Highly interpretable coefficients
- Good performance on linearly separable data
- Robust to outliers when regularized

**Implementation**:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)
```

**Hyperparameters**:
```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],           # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],   # Regularization type
    'solver': ['liblinear', 'saga'],         # Optimization algorithm
    'class_weight': [None, 'balanced']       # Handle class imbalance
}
```

**When to Use**:
- Need interpretable results
- Linear relationships in data
- Fast training required
- Baseline model for comparison

**Performance Characteristics**:
- Training time: Very fast (seconds)
- Memory usage: Low
- Interpretability: High
- Typical ROC-AUC: 0.82-0.86

### 2. Random Forest

**Purpose**: Ensemble method combining multiple decision trees

**Characteristics**:
- Handles non-linear relationships well
- Built-in feature importance
- Resistant to overfitting
- Works well with mixed data types

**Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
```

**Hyperparameters**:
```python
param_grid = {
    'n_estimators': [100, 200, 300],         # Number of trees
    'max_depth': [10, 20, None],             # Tree depth
    'min_samples_split': [2, 5, 10],         # Min samples to split node
    'min_samples_leaf': [1, 2, 4],           # Min samples in leaf
    'max_features': ['sqrt', 'log2', None],  # Features per split
    'class_weight': [None, 'balanced']
}
```

**Feature Importance**:
```python
# Get feature importance after training
importances = model.feature_importances_
feature_names = ['feature_1', 'feature_2', ...]

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

**When to Use**:
- Non-linear relationships expected
- Feature importance needed
- Robust model required
- Mixed data types

**Performance Characteristics**:
- Training time: Moderate (minutes)
- Memory usage: Moderate
- Interpretability: Medium (via feature importance)
- Typical ROC-AUC: 0.83-0.87

### 3. Gradient Boosting

**Purpose**: Sequential ensemble that builds models to correct previous errors

**Characteristics**:
- High predictive accuracy
- Handles complex patterns
- Sequential learning process
- Good generalization

**Implementation**:
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    random_state=42
)
```

**Hyperparameters**:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],      # Step size shrinkage
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]            # Fraction of samples per tree
}
```

**Learning Curve Analysis**:
```python
# Monitor training progress
train_scores = model.train_score_
validation_scores = []  # Would need validation set

plt.plot(range(1, len(train_scores) + 1), train_scores, label='Training')
plt.xlabel('Boosting Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**When to Use**:
- High accuracy required
- Complex patterns in data
- Willing to trade interpretability for performance
- Sufficient training data available

**Performance Characteristics**:
- Training time: Moderate to slow
- Memory usage: Moderate
- Interpretability: Low to medium
- Typical ROC-AUC: 0.84-0.88

### 4. XGBoost (Optional)

**Purpose**: Advanced gradient boosting with optimizations

**Characteristics**:
- State-of-the-art performance
- Built-in regularization
- Handles missing values
- Parallel processing

**Installation**:
```bash
pip install xgboost
```

**Implementation**:
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
```

**Hyperparameters**:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],           # Min sum of weights in child
    'gamma': [0, 0.1, 0.2],                  # Min loss reduction for split
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],     # Feature sampling
    'reg_alpha': [0, 0.1, 0.5],              # L1 regularization
    'reg_lambda': [1, 1.5, 2]                # L2 regularization
}
```

**Advanced Features**:
```python
# Early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)

# Feature importance (multiple types)
importance_gain = model.feature_importances_  # Default: gain
importance_weight = model.get_booster().get_score(importance_type='weight')
importance_cover = model.get_booster().get_score(importance_type='cover')
```

**When to Use**:
- Maximum performance required
- Competition or production setting
- Large datasets
- Complex feature interactions

**Performance Characteristics**:
- Training time: Fast to moderate (optimized)
- Memory usage: Moderate
- Interpretability: Medium (via SHAP)
- Typical ROC-AUC: 0.85-0.90

### 5. LightGBM (Optional)

**Purpose**: Fast gradient boosting optimized for large datasets

**Characteristics**:
- Very fast training
- Low memory usage
- Handles categorical features natively
- Good accuracy

**Installation**:
```bash
pip install lightgbm
```

**Implementation**:
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    random_state=42,
    verbose=-1  # Suppress warnings
)
```

**Hyperparameters**:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, -1],              # -1 means no limit
    'num_leaves': [31, 50, 100],             # Max leaves per tree
    'min_child_samples': [20, 30, 50],       # Min samples in leaf
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}
```

**Categorical Feature Handling**:
```python
# LightGBM can handle categorical features directly
categorical_features = ['gender', 'Contract', 'PaymentMethod']

model.fit(
    X_train, y_train,
    categorical_feature=categorical_features
)
```

**When to Use**:
- Large datasets (>10K samples)
- Fast training required
- Memory constraints
- Categorical features present

**Performance Characteristics**:
- Training time: Very fast
- Memory usage: Low
- Interpretability: Medium
- Typical ROC-AUC: 0.84-0.89

### 6. Support Vector Machine (SVM)

**Purpose**: Kernel-based classifier for complex decision boundaries

**Characteristics**:
- Effective in high-dimensional spaces
- Memory efficient
- Versatile (different kernels)
- Good for small to medium datasets

**Implementation**:
```python
from sklearn.svm import SVC

model = SVC(
    random_state=42,
    probability=True  # Enable probability estimates
)
```

**Hyperparameters**:
```python
param_grid = {
    'C': [0.1, 1, 10, 100],                  # Regularization parameter
    'kernel': ['rbf', 'poly', 'sigmoid'],    # Kernel type
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
    'class_weight': [None, 'balanced']
}
```

**Kernel Selection**:
```python
# RBF kernel (most common)
svm_rbf = SVC(kernel='rbf', gamma='scale')

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)

# Linear kernel (for linearly separable data)
svm_linear = SVC(kernel='linear')
```

**When to Use**:
- High-dimensional data
- Clear margin of separation
- Small to medium datasets
- Non-linear patterns

**Performance Characteristics**:
- Training time: Slow for large datasets
- Memory usage: Moderate
- Interpretability: Low
- Typical ROC-AUC: 0.81-0.85

### 7. Neural Network (MLP)

**Purpose**: Multi-layer perceptron for complex pattern recognition

**Characteristics**:
- Can learn complex non-linear patterns
- Flexible architecture
- Universal function approximator
- Requires careful tuning

**Implementation**:
```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    random_state=42,
    max_iter=1000
)
```

**Hyperparameters**:
```python
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Network architecture
    'activation': ['relu', 'tanh', 'logistic'],                   # Activation function
    'alpha': [0.0001, 0.001, 0.01],                              # L2 regularization
    'learning_rate': ['constant', 'adaptive'],                    # Learning rate schedule
    'learning_rate_init': [0.001, 0.01, 0.1]                     # Initial learning rate
}
```

**Architecture Examples**:
```python
# Simple network
simple_mlp = MLPClassifier(hidden_layer_sizes=(50,))

# Deep network
deep_mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25))

# Wide network
wide_mlp = MLPClassifier(hidden_layer_sizes=(200,))
```

**When to Use**:
- Complex non-linear patterns
- Large datasets available
- Feature interactions important
- Willing to sacrifice interpretability

**Performance Characteristics**:
- Training time: Moderate to slow
- Memory usage: Moderate to high
- Interpretability: Very low
- Typical ROC-AUC: 0.82-0.87

## 🔧 Hyperparameter Optimization

### Grid Search vs Random Search

The system supports both optimization strategies:

```python
# Grid Search - exhaustive search
best_model, best_params = factory.optimize_hyperparameters(
    model_name='random_forest',
    X_train=X_train,
    y_train=y_train,
    search_type='grid',
    cv=5
)

# Random Search - faster, often equally effective
best_model, best_params = factory.optimize_hyperparameters(
    model_name='random_forest',
    X_train=X_train,
    y_train=y_train,
    search_type='random',
    n_iter=50,
    cv=5
)
```

### Custom Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Custom parameter distributions for random search
custom_param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Custom optimization
search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=custom_param_dist,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

## 🎯 Ensemble Methods

### Voting Classifier

The system can create ensemble models that combine multiple algorithms:

```python
# Create voting ensemble
ensemble = factory.create_voting_ensemble(['logistic', 'random_forest', 'gradient_boosting'])

# Train ensemble
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

### Custom Ensemble

```python
from sklearn.ensemble import VotingClassifier

# Create custom ensemble
estimators = [
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
]

ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft'  # Use predicted probabilities
)

# Train and evaluate
ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
```

## 📊 Model Selection Strategy

### Performance-Based Selection

```python
def select_best_model(models_dict, metric='val_roc_auc'):
    """
    Select the best model based on a specific metric.
    
    Args:
        models_dict: Dictionary of trained models with metrics
        metric: Metric to use for selection
        
    Returns:
        Best model name and score
    """
    best_score = -1
    best_model = None
    
    for model_name, model_info in models_dict.items():
        score = model_info['metrics'][metric]
        if score > best_score:
            best_score = score
            best_model = model_name
    
    return best_model, best_score

# Usage
best_model_name, best_score = select_best_model(trainer.models)
print(f"Best model: {best_model_name} (Score: {best_score:.4f})")
```

### Multi-Criteria Selection

```python
def select_model_multi_criteria(models_dict, weights=None):
    """
    Select model based on multiple criteria.
    
    Args:
        models_dict: Dictionary of trained models
        weights: Dictionary of metric weights
        
    Returns:
        Best model based on weighted score
    """
    if weights is None:
        weights = {
            'val_roc_auc': 0.4,
            'val_f1': 0.3,
            'val_precision': 0.2,
            'val_recall': 0.1
        }
    
    model_scores = {}
    
    for model_name, model_info in models_dict.items():
        weighted_score = 0
        for metric, weight in weights.items():
            if metric in model_info['metrics']:
                weighted_score += model_info['metrics'][metric] * weight
        
        model_scores[model_name] = weighted_score
    
    best_model = max(model_scores, key=model_scores.get)
    return best_model, model_scores[best_model]
```

## 🔍 Model Interpretability

### Feature Importance

```python
def get_feature_importance(model, feature_names):
    """Extract feature importance from different model types."""
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
    else:
        # Use permutation importance as fallback
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(model, X_test, y_test)
        importances = perm_importance.importances_mean
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

# Usage
importance_df = get_feature_importance(model, feature_names)
print(importance_df.head(10))
```

### SHAP Integration (Optional)

```python
try:
    import shap
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])  # Explain first 100 samples
    
    # Summary plot
    shap.summary_plot(shap_values, X_test[:100])
    
    # Feature importance plot
    shap.summary_plot(shap_values, X_test[:100], plot_type="bar")
    
except ImportError:
    print("SHAP not available. Install with: pip install shap")
```

## ⚡ Performance Optimization

### Model-Specific Optimizations

```python
# Random Forest optimization
rf_optimized = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,  # Use all cores
    random_state=42
)

# XGBoost optimization
xgb_optimized = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)
```

### Memory Optimization

```python
# For large datasets, use incremental learning where possible
from sklearn.linear_model import SGDClassifier

# Stochastic Gradient Descent for large datasets
sgd_model = SGDClassifier(
    loss='log',  # Logistic regression
    random_state=42
)

# Partial fit for streaming data
for batch in data_batches:
    sgd_model.partial_fit(batch_X, batch_y, classes=[0, 1])
```

## 🚀 Best Practices

### 1. Model Selection Guidelines

```python
def recommend_model(dataset_size, interpretability_needed, training_time_constraint):
    """
    Recommend best model based on constraints.
    
    Args:
        dataset_size: Size of dataset ('small', 'medium', 'large')
        interpretability_needed: Whether interpretability is important
        training_time_constraint: Time constraint ('fast', 'medium', 'slow')
        
    Returns:
        Recommended model name
    """
    if interpretability_needed:
        return 'logistic'
    
    if dataset_size == 'small' and training_time_constraint == 'fast':
        return 'logistic'
    elif dataset_size == 'medium' and training_time_constraint == 'medium':
        return 'random_forest'
    elif dataset_size == 'large' and training_time_constraint == 'fast':
        return 'lightgbm'
    elif training_time_constraint == 'slow':
        return 'xgboost'
    else:
        return 'gradient_boosting'
```

### 2. Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold for imbalanced datasets
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Multiple metrics evaluation
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for model_name, model in models.items():
    print(f"\n{model_name} Cross-Validation Results:")
    for score in scoring:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=score)
        print(f"  {score}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 3. Model Persistence

```python
# Save model with metadata
model_metadata = {
    'model_type': 'RandomForestClassifier',
    'hyperparameters': model.get_params(),
    'feature_names': feature_names,
    'training_date': datetime.now().isoformat(),
    'performance_metrics': {
        'roc_auc': 0.85,
        'f1_score': 0.72,
        'accuracy': 0.81
    }
}

# Save using joblib (recommended for sklearn models)
import joblib
joblib.dump({
    'model': model,
    'metadata': model_metadata
}, 'model_with_metadata.pkl')

# Load model
loaded_data = joblib.load('model_with_metadata.pkl')
loaded_model = loaded_data['model']
loaded_metadata = loaded_data['metadata']
```

---

**Next**: Continue to [Training Pipeline](06-training-pipeline.md) to understand how these models are trained and optimized in the system.