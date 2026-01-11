from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.model import get_model
import pandas as pd

app = FastAPI(title="Churn Prediction API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    X = pd.DataFrame([req.features])

    prob = float(model.predict_proba(X)[:, 1][0])
    return PredictResponse(
        churn_probability=prob,
        will_churn=prob >= 0.5
    )
