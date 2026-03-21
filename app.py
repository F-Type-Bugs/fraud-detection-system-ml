import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# Load Model + Features
# -------------------------
model = joblib.load("fraud_model.pkl")
feature_columns = joblib.load("fraud_feature_columns.pkl")

# -------------------------
# Title
# -------------------------
st.title("Fraud Detection System")
st.write("Predict whether a transaction is legitimate or fraudulent.")

# -------------------------
# Helper Function
# -------------------------
def prepare_input():
    input_data = {col: 0.0 for col in feature_columns}

    input_data["Time"] = time_val
    input_data["Amount"] = amount_val

    for i in range(1, 29):
        input_data[f"V{i}"] = v_inputs[f"V{i}"]

    return pd.DataFrame([input_data])

# -------------------------
# Layout
# -------------------------
left_col, right_col = st.columns([1.3, 0.7], gap="large")

# -------------------------
# Inputs
# -------------------------
with left_col:
    st.subheader("Transaction Inputs")

    time_val = st.slider("Time", 0.0, 200000.0, 50000.0, 1.0)
    amount_val = st.slider("Amount", 0.0, 25000.0, 100.0, 1.0)

    st.markdown("### PCA Features")

    v_inputs = {}
    col1, col2 = st.columns(2)

    for i in range(1, 15):
        with col1:
            v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

    for i in range(15, 29):
        with col2:
            v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

    predict_btn = st.button("Predict Fraud", use_container_width=True)

# -------------------------
# Prediction Result
# -------------------------
with right_col:
    st.subheader("Prediction Result")

    if predict_btn:
        input_df = prepare_input()

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{probability * 100:.2f}%")
        st.progress(float(probability))

        if probability >= 0.8:
            st.warning("High fraud risk detected. This transaction should be reviewed immediately.")
        elif probability >= 0.4:
            st.info("Moderate fraud risk detected. Manual review may be needed.")
        else:
            st.success("Low fraud risk based on current model prediction.")

        with st.expander("Input Data Used for Prediction"):
            st.dataframe(input_df, use_container_width=True)