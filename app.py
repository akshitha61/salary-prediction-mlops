import os
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.sklearn.load_model(
    "models:/SalaryPredictionModel@champion"
)

app = FastAPI(
    title="Salary Prediction API",
    version="1.0.0"
)

REQUESTS = Counter(
    "salary_api_requests_total",
    "Number of salary prediction requests"
)

LATENCY = Histogram(
    "salary_api_request_latency_seconds",
    "Time spent processing prediction requests"
)


class EmployeeInput(BaseModel):
    age: float = Field(gt=0, le=100)
    education_level: str
    job_title: str
    years_of_experience: float = Field(ge=0, le=80)
    gender: Optional[str] = None


@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(item: EmployeeInput):
    REQUESTS.inc()

    row = pd.DataFrame([{
        "Age": item.age,
        "Education Level": item.education_level,
        "Job Title": item.job_title,
        "Years of Experience": item.years_of_experience,
    }])

    with LATENCY.time():
        prediction = float(model.predict(row)[0])

    return {"predicted_salary": prediction}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )