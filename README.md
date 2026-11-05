\# Salary Prediction MLOps Project



\## Project Overview

This project develops a machine learning pipeline to predict employee salaries based on age, education level, job title, and years of experience.



\## Technologies Used

\- Python

\- Pandas and NumPy

\- Scikit-learn

\- MLflow

\- Prefect

\- FastAPI

\- Docker

\- Prometheus



\## Project Features

\- Data preprocessing and cleaning

\- Machine learning model training and evaluation

\- MLflow experiment tracking

\- Model registration and champion model selection

\- Prefect workflow orchestration

\- REST API for salary predictions

\- API monitoring using Prometheus metrics

\- Docker containerization

\- Data drift and fairness analysis



\## Project Files

\- `app.py` – FastAPI application

\- `salary\_prefect\_flow.py` – Prefect training workflow

\- `salary\_prefect\_serve.py` – Prefect deployment

\- `Dockerfile` – Docker configuration

\- `requirements.txt` – Project dependencies

\- `requirements-api.txt` – API dependencies



\## Running the API

```bash

uvicorn app:app --host 0.0.0.0 --port 8000

