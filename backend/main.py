from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import predict_phishing
import joblib
import numpy as np
import os

app = FastAPI()

# Allow Next.js frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load behavioral model
behavior_model = joblib.load("model/behavior_model.pkl")

# ✅ Define request schema
class RequestData(BaseModel):
    text: str                  # The URL to check
    login_frequency: float     # e.g. 0.0 to 1.0
    device_type: int           # 0 = Mobile, 1 = Desktop

# ✅ Main prediction route
@app.post("/predict")
async def predict(data: RequestData):
    slm_score = float(predict_phishing(data.text))
    behavior_score = float(behavior_model.predict_proba([[data.login_frequency, data.device_type]])[0][1])
    final_score = round(0.7 * slm_score + 0.3 * behavior_score, 4)

    return {
        "phishing_probability": final_score,
        "slm_score": slm_score,
        "behavior_score": behavior_score,
        "label": "phishing" if final_score > 0.5 else "safe"
    }

# ✅ Basic health check
@app.get("/")
def root():
    return {"message": "Phishing detection backend is running!"}
