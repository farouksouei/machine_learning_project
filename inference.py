from save_load import load_model
import pandas as pd
import os

# Try to load model, create a simple one if it doesn't exist
try:
    if os.path.exists("churn_model.pkl"):
        model = load_model()
    else:
        # Create a simple model for testing
        print("No pre-trained model found. Creating a simple model for testing...")
        from train import train
        model = train()
        from save_load import save_model
        save_model(model, "churn_model.pkl")
        print("Simple model created and saved.")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

def predict(input_data: dict):
    """
    Predict customer churn.
    
    Args:
        input_data: Dictionary containing customer features
        
    Returns:
        Dictionary with churn prediction
    """
    if model is None:
        # Fallback prediction
        return {"churn": False, "confidence": "low", "message": "No model available"}
    
    try:
        df = pd.DataFrame([input_data])
        
        # Remove customerID if present (it's not used for prediction)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        prediction = model.predict(df)[0]
        
        # Try to get probability if available
        try:
            proba = model.predict_proba(df)[0, 1]
            confidence = "high" if proba > 0.7 or proba < 0.3 else "medium" if proba > 0.6 or proba < 0.4 else "low"
        except:
            proba = 0.5
            confidence = "medium"
        
        return {
            "churn": bool(prediction),
            "probability": float(proba),
            "confidence": confidence
        }
    except Exception as e:
        return {"churn": False, "confidence": "low", "error": str(e)}
