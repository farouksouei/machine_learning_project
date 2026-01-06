"""
Advanced Machine Learning Models for Customer Churn Prediction

This module implements multiple ML algorithms with hyperparameter optimization,
ensemble methods, and advanced techniques for binary classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")


class ChurnModelFactory:
    """
    Factory class for creating and configuring various ML models for churn prediction.
    
    Supports multiple algorithms with hyperparameter optimization:
    - Logistic Regression (baseline)
    - Random Forest (ensemble)
    - Gradient Boosting (ensemble)
    - XGBoost (if available)
    - LightGBM (if available)
    - Support Vector Machine
    - Neural Network (MLP)
    - Voting Ensemble
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model factory.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.param_grids = {}
        self._setup_models()
    
    def _setup_models(self) -> None:
        """Setup all available models with their parameter grids."""
        
        # 1. Logistic Regression - Linear baseline model
        self.models['logistic'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.param_grids['logistic'] = {
            'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
            'solver': ['liblinear', 'saga'],  # Optimization algorithm
            'class_weight': [None, 'balanced']  # Handle class imbalance
        }
        
        # 2. Random Forest - Ensemble of decision trees
        self.models['random_forest'] = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.param_grids['random_forest'] = {
            'n_estimators': [100, 200, 300],  # Number of trees
            'max_depth': [10, 20, None],  # Tree depth
            'min_samples_split': [2, 5, 10],  # Min samples to split node
            'min_samples_leaf': [1, 2, 4],  # Min samples in leaf
            'max_features': ['sqrt', 'log2', None],  # Features per split
            'class_weight': [None, 'balanced']
        }
        
        # 3. Gradient Boosting - Sequential ensemble
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            random_state=self.random_state
        )
        self.param_grids['gradient_boosting'] = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]  # Fraction of samples per tree
        }
        
        # 4. XGBoost - Advanced gradient boosting (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            self.param_grids['xgboost'] = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],  # Min sum of weights in child
                'gamma': [0, 0.1, 0.2],  # Min loss reduction for split
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],  # Feature sampling
                'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
                'reg_lambda': [1, 1.5, 2]  # L2 regularization
            }
        
        # 5. LightGBM - Fast gradient boosting (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1  # Suppress warnings
            )
            self.param_grids['lightgbm'] = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],  # -1 means no limit
                'num_leaves': [31, 50, 100],  # Max leaves per tree
                'min_child_samples': [20, 30, 50],  # Min samples in leaf
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
        
        # 6. Support Vector Machine - Kernel-based classifier
        self.models['svm'] = SVC(
            random_state=self.random_state,
            probability=True  # Enable probability estimates
        )
        self.param_grids['svm'] = {
            'C': [0.1, 1, 10, 100],  # Regularization parameter
            'kernel': ['rbf', 'poly', 'sigmoid'],  # Kernel type
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
            'class_weight': [None, 'balanced']
        }
        
        # 7. Neural Network - Multi-layer perceptron
        self.models['neural_network'] = MLPClassifier(
            random_state=self.random_state,
            max_iter=1000
        )
        self.param_grids['neural_network'] = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Network architecture
            'activation': ['relu', 'tanh', 'logistic'],  # Activation function
            'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
            'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
            'learning_rate_init': [0.001, 0.01, 0.1]  # Initial learning rate
        }
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific model by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Sklearn-compatible model instance
            
        Raises:
            ValueError: If model name is not recognized
        """
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not available. Choose from: {available_models}")
        
        return self.models[model_name]
    
    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameter grid for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameters and their possible values
        """
        if model_name not in self.param_grids:
            raise ValueError(f"Parameter grid for '{model_name}' not available")
        
        return self.param_grids[model_name]
    
    def create_voting_ensemble(self, model_names: Optional[list] = None) -> VotingClassifier:
        """
        Create a voting ensemble from multiple models.
        
        Args:
            model_names: List of model names to include in ensemble.
                        If None, uses all available models.
        
        Returns:
            VotingClassifier ensemble
        """
        if model_names is None:
            model_names = ['logistic', 'random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                model_names.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                model_names.append('lightgbm')
        
        # Create estimators list for voting classifier
        estimators = []
        for name in model_names:
            if name in self.models:
                estimators.append((name, self.models[name]))
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use predicted probabilities
        )
    
    def optimize_hyperparameters(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        scoring: str = 'roc_auc',
        search_type: str = 'grid',
        n_iter: int = 50
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters for a specific model using cross-validation.
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            n_iter: Number of iterations for random search
            
        Returns:
            Tuple of (best_model, best_parameters)
        """
        model = self.get_model(model_name)
        param_grid = self.get_param_grid(model_name)
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")
        
        # Fit the search
        print(f"Optimizing {model_name} hyperparameters using {search_type} search...")
        search.fit(X_train, y_train)
        
        print(f"Best {scoring} score: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
    
    def get_available_models(self) -> list:
        """Get list of all available model names."""
        return list(self.models.keys())
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary mapping class labels to weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))


def get_model(model_name: str = 'logistic') -> Any:
    """
    Convenience function to get a model instance.
    
    Args:
        model_name: Name of the model to create
        
    Returns:
        Model instance
    """
    factory = ChurnModelFactory()
    return factory.get_model(model_name)


def get_ensemble_model() -> VotingClassifier:
    """
    Convenience function to get a voting ensemble model.
    
    Returns:
        VotingClassifier with multiple algorithms
    """
    factory = ChurnModelFactory()
    return factory.create_voting_ensemble()