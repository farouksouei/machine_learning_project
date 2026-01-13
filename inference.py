"""
Inference module for customer churn prediction.

This module handles loading models from the registry and making predictions.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional

from save_load import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model registry
model_registry = ModelRegistry()

# Cache for loaded models
_model_cache: Dict[str, Any] = {}


def get_model(model_id: Optional[str] = None):
    """
    Get a model from the registry.

    Args:
        model_id: Specific model ID to load. If None, uses the active model.

    Returns:
        Loaded model (pipeline) or None if no model available.
    """
    global _model_cache

    # Determine which model to load
    if model_id is None:
        model_id = model_registry.get_active_model_id()

    if model_id is None:
        # No active model set, try to get the best model
        try:
            model, metadata = model_registry.get_best_model()
            model_id = metadata.get('model_id', 'best')
            _model_cache[model_id] = model
            logger.info(f"Loaded best model: {model_id}")
            return model
        except Exception as e:
            logger.warning(f"No models available in registry: {e}")
            return None

    # Check cache first
    if model_id in _model_cache:
        return _model_cache[model_id]

    # Load from registry
    try:
        model, metadata = model_registry.load_model_by_id(model_id)
        _model_cache[model_id] = model
        logger.info(f"Loaded model from registry: {model_id}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        return None


def clear_model_cache():
    """Clear the model cache to force reloading."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")


def predict(input_data: dict, model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Predict customer churn.

    Args:
        input_data: Dictionary containing customer features
        model_id: Optional model ID to use for prediction. If None, uses active model.

    Returns:
        Dictionary with churn prediction
    """
    logger.info(f"Received prediction request (model: {model_id or 'active'})")

    model = get_model(model_id)

    if model is None:
        logger.warning("No model available for prediction. Returning fallback prediction.")
        return {
            "churn": False,
            "confidence": "low",
            "message": "No model available. Please train a model first."
        }

    try:
        df = pd.DataFrame([input_data])
        logger.debug(f"Input data converted to DataFrame with shape: {df.shape}")

        prediction = model.predict(df)[0]
        logger.info(f"Model prediction: {prediction}")

        # Try to get probability if available
        try:
            proba = model.predict_proba(df)[0, 1]
            confidence = "high" if proba > 0.7 or proba < 0.3 else "medium" if proba > 0.6 or proba < 0.4 else "low"
            logger.info(f"Prediction probability: {proba:.4f} (confidence: {confidence})")
        except Exception as proba_error:
            logger.warning(f"Could not get prediction probability: {proba_error}")
            proba = 0.5
            confidence = "medium"

        result = {
            "churn": bool(prediction),
            "probability": float(proba),
            "confidence": confidence,
            "model_id": model_id or model_registry.get_active_model_id() or "best"
        }
        logger.info(f"Successfully completed prediction: churn={result['churn']}, probability={result['probability']:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return {"churn": False, "confidence": "low", "error": str(e)}


def get_available_models() -> list:
    """Get list of available model IDs in the registry."""
    return model_registry.get_model_ids()


def set_active_model(model_id: str) -> bool:
    """
    Set the active model for predictions.

    Args:
        model_id: Model ID to set as active

    Returns:
        True if successful, False otherwise
    """
    try:
        model_registry.set_active_model(model_id)
        # Clear cache to ensure the new active model is loaded
        if model_id in _model_cache:
            del _model_cache[model_id]
        return True
    except Exception as e:
        logger.error(f"Failed to set active model: {e}")
        return False


def get_active_model_id() -> Optional[str]:
    """Get the currently active model ID."""
    return model_registry.get_active_model_id()