import os
import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train first: python src\\train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model
