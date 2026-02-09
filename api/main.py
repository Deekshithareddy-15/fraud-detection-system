
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import os
from .schemas import Transaction
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.drift_detection import detect_drift

app = FastAPI(title="Fraud Detection API", version="1.0")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve frontend
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Load Model and Scaler at startup
MODEL_PATH = "models/saved_models/best_fraud_model.pkl"
SCALER_PATH = "models/saved_models/scaler.pkl"

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        # Check if files exist relative to current working directory or absolute
        # Assuming running from root: uvicorn api.main:app
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model or Scaler file not found. Please train model first.")
            
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, maybe fail startup or use a dummy model?
        # For now, we'll let it run but endpoints might fail.

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/drift")
async def check_drift():
    drift_result = detect_drift("data/raw/new_batch.csv") # Assuming new data constantly lands here or passed as arg
    return drift_result

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to DataFrame
    # Order must match training: V1...V28, Time, Amount
    data_dict = transaction.dict()
    
    # Create DataFrame with specific column order
    # Note: Pydantic dict order is insertion order (python 3.7+), but let's be explicit
    columns = [f"V{i+1}" for i in range(28)] + ["Time", "Amount"]
    
    df = pd.DataFrame([data_dict], columns=columns)
    
    # Scale Time and Amount
    try:
        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during preprocessing: {str(e)}")
    
    # Predict
    try:
        # Models like XGBoost/RF accept DataFrame
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_fraud": bool(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
