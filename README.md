# Churn Prediction System — FastAPI + Streamlit

An end-to-end machine learning project that predicts customer churn and exposes the model through a production-style REST API with an interactive web interface.

This project demonstrates how to move from raw data to a deployable ML system, following real-world engineering and data science best practices.

---

## Project Overview

Customer churn prediction is a common and high-impact business problem in subscription-based companies (telecom, fintech, SaaS, e-commerce).

This system:
- Trains a churn prediction model using historical customer data
- Serves predictions via a REST API built with FastAPI
- Provides a user-friendly interactive GUI built with Streamlit
- Is fully reproducible, testable, and containerized

---

## Architecture

```text
Raw Dataset
↓
Data Cleaning
↓
Data Preprocessing + Feature Engineering
↓
ML Training Pipeline (scikit-learn)
↓
Saved Model Artifact
↓
FastAPI Inference Service
↓
Streamlit Interactive Web UI
```
---

## Tech Stack

### Programming & Data
- Python
- Pandas, NumPy

### Machine Learning
- Scikit-learn
- Pipelines and ColumnTransformer
- Logistic Regression (baseline, interpretable)
- Binary classification
- ROC-AUC evaluation

### Backend / APIs
- FastAPI
- REST API design
- Pydantic schemas
- Model inference endpoints
- Swagger documentation

### Frontend
- Streamlit
- Interactive forms and widgets
- Real-time ML predictions

### Software Engineering
- Modular project structure
- Unit tests with pytest
- Virtual environments
- Git & GitHub best practices

### DevOps
- Docker (containerized API)
- Reproducible setup
- Environment isolation

---

## Project Structure

```text
churn-api-ml/
│
├── app/ # FastAPI application
│ ├── main.py # API endpoints
│ ├── model.py # Model loading logic
│ └── schemas.py # Request/response schemas
│
├── src/ # Training pipeline
│ └── train.py
│
├── gui/ # Streamlit GUI
│ └── app.py
│
├── tests/ # Automated tests
│ └── test_api.py
│
├── models/ # Trained model artifacts
│
├── requirements.txt
├── Dockerfile
└── README.md

```
## Dataset

- **Telco Customer Churn Dataset (IBM / Kaggle)**
- ~7,000 customer records
- Demographic, service usage, and billing features
- Target variable: `Churn` (Yes / No)

The dataset is not included in the repository. You can download it from Kaggle.

---

## Setup (Windows)

### Create virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt 
```
### Training the model
```powershell
python src\train.py
```

### Run the FastAPI Service 
```powershell
uvicorn app.main:app --reload
```

Available endpoints:

Health check: GET /health

Prediction: POST /predict

API docs: 
``` html
http://127.0.0.1:8000/docs
```
---
## Run the Streamlit GUI

Open a new terminal:

```powershell
venv\Scripts\activate
streamlit run gui\app.py
```
#### Run Tests
Includes:
1. API health check test
2. Prediction response validation
```sh
pytest
```
---

## Docker Deployment

Build the Docker image:

```powershell
docker build -t churn-api .
```

Run the container:
```powerhell
docker run -p 8000:8000 churn-api
```

---

## Key Takeaways

1. End-to-end machine learning workflow

2. API-based ML inference

3. Interactive business-facing interface

4. Clean, modular codebase

5. Testing and containerization

6. Real-world churn prediction use case


---

## Future Improvements

1. Model monitoring and logging

2. Input feature validation

3. Authentication for API

4. Model versioning (MLflow)

5. Cloud deployment (AWS / GCP / Azure)

---
___Thank You___


