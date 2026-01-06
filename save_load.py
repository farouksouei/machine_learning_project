"""
Enhanced Model and Metrics Persistence Module

This module provides comprehensive functionality for:
- Model serialization and loading with versioning
- Metrics tracking and storage
- Experiment management and organization
- Model metadata and configuration persistence
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def create_metrics_directory(metrics_dir: str = "metrics") -> str:
    """
    Create metrics directory structure for organizing results.
    
    Args:
        metrics_dir: Base directory for metrics storage
        
    Returns:
        Path to created metrics directory
    """
    # Create main metrics directory
    Path(metrics_dir).mkdir(exist_ok=True)
    
    # Create subdirectories for organization
    subdirs = ['models', 'plots', 'reports']
    for subdir in subdirs:
        Path(metrics_dir, subdir).mkdir(exist_ok=True)
    
    return metrics_dir


def save_model(
    model: Any,
    path: str = "churn_model.pkl",
    metadata: Optional[Dict[str, Any]] = None,
    use_joblib: bool = True
) -> str:
    """
    Save a trained model with optional metadata.
    
    Args:
        model: Trained model object
        path: Path to save the model
        metadata: Optional metadata dictionary
        use_joblib: Whether to use joblib (recommended) or pickle
        
    Returns:
        Path where model was saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Save model
    if use_joblib:
        joblib.dump(model, path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata if provided
    if metadata:
        metadata_path = path.replace('.pkl', '_metadata.json')
        save_metadata(metadata, metadata_path)
    
    print(f"Model saved to: {path}")
    return path


def load_model(
    path: str = "churn_model.pkl",
    load_metadata: bool = False
) -> Union[Any, tuple]:
    """
    Load a trained model with optional metadata.
    
    Args:
        path: Path to the saved model
        load_metadata: Whether to also load metadata
        
    Returns:
        Loaded model, or tuple of (model, metadata) if load_metadata=True
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Load model
    try:
        model = joblib.load(path)
    except:
        # Fallback to pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
    
    if load_metadata:
        metadata_path = path.replace('.pkl', '_metadata.json')
        metadata = load_metadata_file(metadata_path) if os.path.exists(metadata_path) else {}
        return model, metadata
    
    return model


def save_metrics(
    metrics: Dict[str, Union[float, int, str]],
    path: str,
    timestamp: bool = True
) -> str:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        path: Path to save metrics
        timestamp: Whether to add timestamp to metrics
        
    Returns:
        Path where metrics were saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Add timestamp if requested
    if timestamp:
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Save metrics
    with open(path, 'w') as f:
        json.dump(convert_numpy(metrics), f, indent=2)
    
    print(f"Metrics saved to: {path}")
    return path


def load_metrics(path: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        path: Path to metrics file
        
    Returns:
        Dictionary of loaded metrics
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    
    with open(path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def save_metadata(
    metadata: Dict[str, Any],
    path: str
) -> str:
    """
    Save model or experiment metadata.
    
    Args:
        metadata: Metadata dictionary
        path: Path to save metadata
        
    Returns:
        Path where metadata was saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Add system information
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['python_version'] = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    with open(path, 'w') as f:
        json.dump(convert_numpy(metadata), f, indent=2)
    
    return path


def load_metadata_file(path: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file.
    
    Args:
        path: Path to metadata file
        
    Returns:
        Dictionary of loaded metadata
    """
    if not os.path.exists(path):
        return {}
    
    with open(path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


class ModelRegistry:
    """
    Model registry for managing multiple model versions and experiments.
    
    Features:
    - Model versioning and tracking
    - Experiment organization
    - Model comparison and selection
    - Metadata management
    """
    
    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store model registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        # Create registry structure
        (self.registry_dir / "models").mkdir(exist_ok=True)
        (self.registry_dir / "experiments").mkdir(exist_ok=True)
        (self.registry_dir / "metadata").mkdir(exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry metadata."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "experiments": {},
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
    
    def _save_registry(self) -> None:
        """Save registry metadata."""
        self.registry["updated_at"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Version string (e.g., "1.0", "v2.1")
            metrics: Model performance metrics
            metadata: Additional metadata
            tags: Optional tags for categorization
            
        Returns:
            Model ID for the registered model
        """
        model_id = f"{model_name}_v{version}"
        model_dir = self.registry_dir / "models" / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        save_model(model, str(model_path))
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        save_metrics(metrics, str(metrics_path))
        
        # Prepare model metadata
        model_metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
            "tags": tags or [],
            "metadata": metadata or {}
        }
        
        # Save model metadata
        metadata_path = model_dir / "metadata.json"
        save_metadata(model_metadata, str(metadata_path))
        
        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {}
        
        self.registry["models"][model_name][version] = {
            "model_id": model_id,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "tags": tags or []
        }
        
        self._save_registry()
        
        print(f"Model registered: {model_id}")
        return model_id
    
    def load_model_version(
        self,
        model_name: str,
        version: str = "latest"
    ) -> tuple:
        """
        Load a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version to load ("latest" for most recent)
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        versions = self.registry["models"][model_name]
        
        if version == "latest":
            # Get the most recent version
            latest_version = max(versions.keys(), key=lambda v: versions[v]["registered_at"])
            version = latest_version
        
        if version not in versions:
            available_versions = list(versions.keys())
            raise ValueError(f"Version '{version}' not found. Available versions: {available_versions}")
        
        model_id = versions[version]["model_id"]
        model_dir = self.registry_dir / "models" / model_id
        
        # Load model and metadata
        model_path = model_dir / "model.pkl"
        metadata_path = model_dir / "metadata.json"
        
        model = load_model(str(model_path))
        metadata = load_metadata_file(str(metadata_path))
        
        return model, metadata
    
    def list_models(self) -> pd.DataFrame:
        """
        List all registered models.
        
        Returns:
            DataFrame with model information
        """
        models_data = []
        
        for model_name, versions in self.registry["models"].items():
            for version, info in versions.items():
                row = {
                    "model_name": model_name,
                    "version": version,
                    "model_id": info["model_id"],
                    "registered_at": info["registered_at"],
                    "tags": ", ".join(info["tags"])
                }
                
                # Add key metrics
                metrics = info.get("metrics", {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        row[metric_name] = metric_value
                
                models_data.append(row)
        
        return pd.DataFrame(models_data)
    
    def compare_models(
        self,
        model_names: Optional[List[str]] = None,
        metric: str = "val_roc_auc"
    ) -> pd.DataFrame:
        """
        Compare models by a specific metric.
        
        Args:
            model_names: List of model names to compare (if None, compares all)
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with model comparison
        """
        models_df = self.list_models()
        
        if model_names:
            models_df = models_df[models_df["model_name"].isin(model_names)]
        
        if metric in models_df.columns:
            models_df = models_df.sort_values(metric, ascending=False)
        
        return models_df
    
    def get_best_model(
        self,
        model_name: Optional[str] = None,
        metric: str = "val_roc_auc"
    ) -> tuple:
        """
        Get the best model based on a metric.
        
        Args:
            model_name: Specific model name (if None, considers all models)
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model, metadata) for the best model
        """
        models_df = self.list_models()
        
        if model_name:
            models_df = models_df[models_df["model_name"] == model_name]
        
        if models_df.empty:
            raise ValueError("No models found")
        
        if metric not in models_df.columns:
            raise ValueError(f"Metric '{metric}' not found in model data")
        
        best_row = models_df.loc[models_df[metric].idxmax()]
        
        return self.load_model_version(
            best_row["model_name"],
            best_row["version"]
        )

