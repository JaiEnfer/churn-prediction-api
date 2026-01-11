import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📉 Customer Churn Predictor")
st.write("This UI sends customer features to your FastAPI model and returns churn probability.")

API_URL = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")

st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    phone = st.selectbox("PhoneService", ["Yes", "No"])
    multiple = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

with col2:
    online_sec = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

st.subheader("Billing")

col3, col4 = st.columns(2)
with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
    pay_method = st.selectbox(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
with col4:
    monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=500.0, value=89.1)
    total = st.number_input("TotalCharges", min_value=0.0, max_value=50000.0, value=1069.2)

payload = {
    "features": {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": pay_method,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
}

if st.button("Predict churn"):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            out = r.json()
            prob = float(out["churn_probability"])
            will = bool(out["will_churn"])

            st.metric("Churn probability", f"{prob:.2%}")
            st.write("Prediction:", "🚨 Will churn" if will else "✅ Will not churn")

            st.progress(min(max(prob, 0.0), 1.0))
            with st.expander("Show sent payload"):
                st.json(payload)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API: {e}")
        st.info("Make sure FastAPI is running: uvicorn app.main:app --reload")
