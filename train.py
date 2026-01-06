"""
Simplified Training Pipeline for Customer Churn Prediction

This module provides a streamlined training pipeline with:
- Multiple ML algorithms with hyperparameter optimization
- Advanced preprocessing and feature engineering
- Model evaluation and comparison
- Model registry for saving trained models
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline

# Local imports
from data_loader import load_data
from preprocessor import AdvancedPreprocessor, build_preprocessor
from model import ChurnModelFactory
from save_load import ModelRegistry, create_metrics_directory


class ChurnTrainer:
    """
    Simplified training pipeline for customer churn prediction.
    
    Features:
    - Multiple ML algorithms with hyperparameter optimization
    - Advanced preprocessing and feature engineering
    - Model evaluation and comparison
    - Model registry integration
    """
    
    def __init__(
        self,
        data_path: str = "data/dataset.csv",
        target_column: str = "Churn",
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5
    ):
        """
        Initialize the training pipeline.
        
        Args:
            data_path: Path to the dataset
            target_column: Name of the target column
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Initialize components
        self.model_factory = ChurnModelFactory(random_state=random_state)
        self.model_registry = ModelRegistry()
        self.preprocessor = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Create metrics directory
        create_metrics_directory("metrics")
    
    def load_and_split_data(self) -> None:
        """Load data and create train/validation/test splits."""
        print("Loading and splitting data...")
        
        # Load data
        df = load_data(self.data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        
        # Handle target encoding
        if df[self.target_column].dtype == 'object':
            y = df[self.target_column].map({"Yes": 1, "No": 0})
            if y.isnull().any():
                unique_values = df[self.target_column].unique()
                print(f"Target values: {unique_values}")
                y = (df[self.target_column] == unique_values[1]).astype(int)
        else:
            y = df[self.target_column]
        
        # Create train/test split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y
        )
        
        # Create train/validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.validation_size,
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
    
    def setup_preprocessing(
        self,
        advanced: bool = True,
        feature_engineering: bool = True,
        handle_imbalance: bool = False
    ) -> None:
        """
        Setup the preprocessing pipeline.
        
        Args:
            advanced: Whether to use advanced preprocessing
            feature_engineering: Whether to create interaction features
            handle_imbalance: Whether to handle class imbalance
        """
        print("Setting up preprocessing pipeline...")
        
        # Identify column types
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.X_train.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        if advanced:
            # Use advanced preprocessor
            self.preprocessor = AdvancedPreprocessor(
                feature_engineering=feature_engineering,
                handle_imbalance=handle_imbalance,
                random_state=self.random_state
            )
        else:
            # Use simple preprocessor
            self.preprocessor = build_preprocessor(numerical_cols, categorical_cols)
        
        # Fit preprocessor on training data
        self.preprocessor.fit(self.X_train, self.y_train)
    
    def train_single_model(
        self,
        model_name: str,
        optimize_hyperparameters: bool = True,
        save_to_registry: bool = True
    ) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            optimize_hyperparameters: Whether to optimize hyperparameters
            save_to_registry: Whether to save the model to registry
            
        Returns:
            Dictionary containing model and metrics
        """
        print(f"\nTraining {model_name}...")
        
        # Transform training data
        X_train_processed = self.preprocessor.transform(self.X_train)
        X_val_processed = self.preprocessor.transform(self.X_val)
        
        # Handle class imbalance if using advanced preprocessor
        if hasattr(self.preprocessor, 'handle_class_imbalance'):
            X_train_processed, y_train_processed = self.preprocessor.handle_class_imbalance(
                X_train_processed, self.y_train.values
            )
        else:
            y_train_processed = self.y_train.values
        
        if optimize_hyperparameters:
            # Optimize hyperparameters
            model, best_params = self.model_factory.optimize_hyperparameters(
                model_name=model_name,
                X_train=X_train_processed,
                y_train=y_train_processed,
                cv=self.cv_folds,
                search_type='random',
                n_iter=30
            )
        else:
            # Use default model
            model = self.model_factory.get_model(model_name)
            model.fit(X_train_processed, y_train_processed)
            best_params = {}
        
        # Evaluate model
        metrics = self._evaluate_model(
            model, X_train_processed, y_train_processed,
            X_val_processed, self.y_val.values, model_name
        )
        
        # Store results
        result = {
            'model': model,
            'best_params': best_params,
            'metrics': metrics
        }
        
        self.models[model_name] = result
        
        # Update best model
        if self.best_model is None or metrics['val_roc_auc'] > self.models[self.best_model_name]['metrics']['val_roc_auc']:
            self.best_model = model
            self.best_model_name = model_name
        
        # Save to model registry
        if save_to_registry:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_registry.register_model(
                model=model,
                model_name=model_name,
                version=version,
                metrics=metrics,
                metadata={
                    'best_params': best_params,
                    'preprocessing': 'advanced' if hasattr(self.preprocessor, 'feature_engineering') else 'simple',
                    'data_shape': self.X_train.shape,
                    'trained_at': datetime.now().isoformat()
                },
                tags=['churn_prediction', 'production']
            )
        
        return result
    
    def train_multiple_models(
        self,
        model_names: Optional[List[str]] = None,
        optimize_hyperparameters: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and compare their performance.
        
        Args:
            model_names: List of model names to train (if None, trains selected models)
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of model results
        """
        if model_names is None:
            # Train a selection of good models by default
            available_models = self.model_factory.get_available_models()
            model_names = ['logistic', 'random_forest', 'gradient_boosting']
            # Add XGBoost and LightGBM if available
            if 'xgboost' in available_models:
                model_names.append('xgboost')
            if 'lightgbm' in available_models:
                model_names.append('lightgbm')
        
        print(f"Training {len(model_names)} models: {model_names}")
        
        results = {}
        for model_name in model_names:
            try:
                result = self.train_single_model(
                    model_name=model_name,
                    optimize_hyperparameters=optimize_hyperparameters
                )
                results[model_name] = result
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find and print best model
        self._find_best_model(results)
        
        return results
    
    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Prediction probabilities (if available)
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
        except:
            y_train_proba = y_train_pred
            y_val_proba = y_val_pred
        
        # Calculate metrics
        metrics = {
            # Training metrics
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='binary'),
            'train_recall': recall_score(y_train, y_train_pred, average='binary'),
            'train_f1': f1_score(y_train, y_train_pred, average='binary'),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            
            # Validation metrics
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='binary'),
            'val_recall': recall_score(y_val, y_val_pred, average='binary'),
            'val_f1': f1_score(y_val, y_val_pred, average='binary'),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba),
        }
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=self.cv_folds,
                scoring='roc_auc', n_jobs=-1
            )
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()
        except:
            metrics['cv_roc_auc_mean'] = metrics['val_roc_auc']
            metrics['cv_roc_auc_std'] = 0.0
        
        # Print metrics
        print(f"{model_name} Results:")
        print(f"  Validation ROC-AUC: {metrics['val_roc_auc']:.4f}")
        print(f"  Validation F1: {metrics['val_f1']:.4f}")
        print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
        
        return metrics
    
    def _find_best_model(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Find the best model based on validation ROC-AUC."""
        best_score = -1
        best_name = None
        
        for model_name, result in results.items():
            score = result['metrics']['val_roc_auc']
            if score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = results[best_name]['model'] if best_name else None
        
        print(f"\nBest model: {best_name} (ROC-AUC: {best_score:.4f})")
    
    def evaluate_on_test_set(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate the best model (or specified model) on the test set.
        
        Args:
            model_name: Name of model to evaluate (if None, uses best model)
            
        Returns:
            Dictionary of test metrics
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]['model']
        
        if model is None:
            raise ValueError("No model available for testing")
        
        print(f"\nEvaluating {model_name} on test set...")
        
        # Transform test data
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Predictions
        y_test_pred = model.predict(X_test_processed)
        
        try:
            y_test_proba = model.predict_proba(X_test_processed)[:, 1]
        except:
            y_test_proba = y_test_pred
        
        # Calculate test metrics
        test_metrics = {
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'test_precision': precision_score(self.y_test, y_test_pred, average='binary'),
            'test_recall': recall_score(self.y_test, y_test_pred, average='binary'),
            'test_f1': f1_score(self.y_test, y_test_pred, average='binary'),
            'test_roc_auc': roc_auc_score(self.y_test, y_test_proba),
        }
        
        # Print test results
        print("Test Set Results:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        return test_metrics
    
    def generate_model_comparison_report(self) -> pd.DataFrame:
        """
        Generate a comparison report of all trained models.
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.models:
            raise ValueError("No models trained yet")
        
        comparison_data = []
        for model_name, result in self.models.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'Val_ROC_AUC': metrics['val_roc_auc'],
                'Val_F1': metrics['val_f1'],
                'Val_Accuracy': metrics['val_accuracy'],
                'Val_Precision': metrics['val_precision'],
                'Val_Recall': metrics['val_recall'],
                'CV_ROC_AUC_Mean': metrics['cv_roc_auc_mean'],
                'CV_ROC_AUC_Std': metrics['cv_roc_auc_std']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val_ROC_AUC', ascending=False)
        
        # Save comparison report
        report_path = os.path.join("metrics", 'model_comparison.csv')
        comparison_df.to_csv(report_path, index=False)
        
        print("\nModel Comparison Report:")
        print(comparison_df.round(4))
        print(f"Comparison report saved to: {report_path}")
        
        return comparison_df


def train_churn_model(
    data_path: str = "data/dataset.csv",
    models_to_train: Optional[List[str]] = None,
    optimize_hyperparameters: bool = True,
    advanced_preprocessing: bool = True
) -> ChurnTrainer:
    """
    Main training function with sensible defaults.
    
    Args:
        data_path: Path to the dataset
        models_to_train: List of models to train (if None, trains selected models)
        optimize_hyperparameters: Whether to optimize hyperparameters
        advanced_preprocessing: Whether to use advanced preprocessing
        
    Returns:
        Trained ChurnTrainer instance
    """
    # Initialize trainer
    trainer = ChurnTrainer(data_path=data_path)
    
    # Load and split data
    trainer.load_and_split_data()
    
    # Setup preprocessing
    trainer.setup_preprocessing(
        advanced=advanced_preprocessing,
        feature_engineering=advanced_preprocessing,
        handle_imbalance=False  # Keep simple for now
    )
    
    # Train models
    trainer.train_multiple_models(
        model_names=models_to_train,
        optimize_hyperparameters=optimize_hyperparameters
    )
    
    # Evaluate best model on test set
    if trainer.best_model is not None:
        trainer.evaluate_on_test_set()
    
    # Generate comparison report
    trainer.generate_model_comparison_report()
    
    return trainer


def train() -> Any:
    """
    Simple training function for backward compatibility.
    
    Returns:
        Trained pipeline
    """
    trainer = train_churn_model(
        models_to_train=['logistic'],  # Only train logistic regression for speed
        optimize_hyperparameters=False,
        advanced_preprocessing=False
    )
    
    # Return the best model wrapped in a pipeline for compatibility
    numerical_cols = trainer.X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = trainer.X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Create simple pipeline
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(numerical_cols, categorical_cols)),
        ("model", trainer.best_model)
    ])
    
    # Fit the pipeline on the full training data
    X_full = pd.concat([trainer.X_train, trainer.X_val])
    y_full = pd.concat([trainer.y_train, trainer.y_val])
    pipeline.fit(X_full, y_full)
    
    return pipeline