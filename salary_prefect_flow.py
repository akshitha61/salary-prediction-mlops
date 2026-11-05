
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Salary Data.xlsx"

FEATURES = [
    "Age",
    "Education Level",
    "Job Title",
    "Years of Experience"
]
TARGET = "Salary"


@task
def load_data():
    return pd.read_excel(DATA_FILE)


@task
def clean_data(df):
    keep = FEATURES + [TARGET]
    df = df[keep].copy()

    for column in ["Age", "Years of Experience", TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=keep)
    return df


@task
def train_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), ["Age", "Years of Experience"]),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), ["Education Level", "Job Title"])
    ])

    model = Pipeline([
        ("preprocessor", prep),
        ("model", Ridge(alpha=1.0))
    ])

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Employee_Salary_Prediction")

    with mlflow.start_run(run_name="prefect-ridge"):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

        mlflow.log_metric(
            "mae", mean_absolute_error(y_test, pred)
        )
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2_score(y_test, pred))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="SalaryPredictionModel",
            serialization_format="cloudpickle"
        )

    return rmse


@task
def register_approved(rmse):
    if not np.isfinite(rmse):
        raise ValueError("Model failed validation: RMSE is not finite")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(
        tracking_uri="http://127.0.0.1:5000"
    )

    versions = client.search_model_versions(
        "name='SalaryPredictionModel'"
    )

    latest = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(
        "SalaryPredictionModel",
        "champion",
        latest.version
    )

    return str(latest.version)


@flow(name="salary-training-pipeline")
def salary_training_pipeline():
    raw = load_data()
    clean = clean_data(raw)
    score = train_evaluate(clean)
    return register_approved(score)


if __name__ == "__main__":
    salary_training_pipeline()
