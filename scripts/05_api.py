from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os
import json

app = FastAPI(title="Cold Chain Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = None
scaler = None

if os.path.exists("models/xgboost_model.pkl"):
    model = joblib.load("models/xgboost_model.pkl")
    print("✅ XGBoost model loaded successfully")
else:
    print("⚠️ Model not found - run 03_train_xgboost.py first")

if os.path.exists("models/scaler.pkl"):
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Scaler loaded successfully")

class SensorData(BaseModel):
    temperature_celsius: float
    humidity_percent: float
    battery_percent: float
    door_open: int
    hour: int
    day_of_week: int
    temp_rate_change: float = 0.0
    temp_rolling_mean: float = 0.0
    temp_rolling_std: float = 0.0
    door_open_count: int = 0

@app.get("/")
def root():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
def get_metrics():
    """Endpoint for dashboard to fetch model performance metrics"""
    xgb_metrics = {}
    lstm_metrics = {}
    
    try:
        if os.path.exists("models/xgboost_metrics.json"):
            with open("models/xgboost_metrics.json", "r") as f:
                xgb_metrics = json.load(f)
    except:
        pass
    
    try:
        if os.path.exists("models/lstm_forecast_metrics.json"):
            with open("models/lstm_forecast_metrics.json", "r") as f:
                lstm_metrics = json.load(f)
    except:
        pass
    
    return {"xgboost": xgb_metrics, "lstm": lstm_metrics}

@app.post("/predict")
def predict(data: SensorData):
    if model is None:
        return {"failure_probability": 0.5, "risk_level": "UNKNOWN", "recommendation": "Model not loaded"}
    
    features = np.array([[
        data.temperature_celsius,
        data.humidity_percent,
        data.battery_percent,
        data.temp_rate_change,
        data.temp_rolling_mean,
        data.temp_rolling_std,
        data.door_open_count,
        data.hour,
        data.day_of_week
    ]])
    
    if scaler is not None:
        features = scaler.transform(features)
    
    prob = float(model.predict_proba(features)[0, 1])
    
    if prob > 0.7:
        risk = "HIGH"
        rec = "Immediate action recommended. Inspect equipment within 24 hours."
    elif prob > 0.3:
        risk = "MEDIUM"
        rec = "Schedule preventive maintenance within 48 hours."
    else:
        risk = "LOW"
        rec = "Normal operation. Continue monitoring."
    
    return {"failure_probability": prob, "risk_level": risk, "recommendation": rec}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)