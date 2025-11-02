import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# -----------------------------
# CONFIGURATION
# -----------------------------
SEQ_LEN = 20
FEATURES = ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt']
TARGET = 'SOH'

st.set_page_config(page_title="Battery Health Prediction (LSTM)", layout="wide")

st.title("🔋 Lithium-ion Battery Health Prediction using LSTM")
st.markdown("Upload your last 20 cycle data (CSV or Excel) to predict **State of Health (SOH)** using a trained LSTM model.")

# -----------------------------
# LOAD MODEL AND SCALER
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model('battery_lstm_model.h5', custom_objects={'mse': MeanSquaredError()})
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    st.success("✅ Model and scaler loaded successfully.")
else:
    st.stop()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file containing the last 20 cycles", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read the uploaded file
        if uploaded_file.name.lower().endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith('.xlsx'):
            df_in = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()

        st.write("### Preview of Uploaded Data")
        st.dataframe(df_in.head())

        # Validate columns
        if not all(feat in df_in.columns for feat in FEATURES):
            st.error(f"Uploaded file must contain columns: {FEATURES}")
            st.stop()

        # Ensure enough rows
        if len(df_in) < SEQ_LEN:
            st.error(f"File must contain at least {SEQ_LEN} rows.")
            st.stop()

        # Use last SEQ_LEN rows
        df_in = df_in.tail(SEQ_LEN).copy()
        st.info(f"Using the last {SEQ_LEN} cycles for prediction.")

        # -----------------------------
        # DATA PREPARATION
        # -----------------------------
        df_in[TARGET] = 0  # Placeholder for target
        df_in = df_in[FEATURES + [TARGET]]

        # Scale data
        inp_scaled = scaler.transform(df_in)

        # Prepare input for model
        X_user = np.expand_dims(inp_scaled[:, :-1], axis=0)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        pred_scaled = model.predict(X_user)
        dummy_inverse = np.zeros((1, df_in.shape[1]))
        dummy_inverse[:, -1] = pred_scaled[0][0]
        predicted_soh = scaler.inverse_transform(dummy_inverse)[:, -1][0]

        # -----------------------------
        # DISPLAY RESULT
        # -----------------------------
        st.success("✅ Prediction Complete")
        st.metric(label="🔍 Predicted Battery State of Health (SOH)", value=f"{predicted_soh:.4f}")

        # Optional visualization
        st.write("### 📊 Last 20 Cycles Data Used for Prediction")
        st.dataframe(df_in)

        st.line_chart(df_in[FEATURES])

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV or Excel file to begin.")
