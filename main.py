from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Add CORS middleware
# app.add_middleware(
#     allow_origins=["*"], allow_credentials=True,allow_methods=["*"],allow_headers=["*"],
# )


class Features(BaseModel):
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float


# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.post("predict/financial_fraud_status")
async def predict(features: Features):
    feature_array = np.array([[features.type, features.amount, features.oldbalanceOrg, features.newbalanceOrig]])
    scaled_features = scaler.transform(feature_array)
    prediction = model.predict(scaled_features)
    return {"prediction": "Fraud" if prediction[0] == 1 else "No Fraud"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
