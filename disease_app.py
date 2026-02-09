import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease Detection", page_icon="🩸", layout="wide")

# ── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = "blood_model.pkl"
DISEASE_LABELS = {
    0: "Anemia",
    1: "Diabetes",
    2: "Healthy",
    3: "Heart Disease",
    4: "Thalassemia",
    5: "Thrombocytosis",
}

FEATURE_NAMES = [
    "Glucose",
    "Cholesterol",
    "Hemoglobin",
    "Platelets",
    "White Blood Cells",
    "Red Blood Cells",
    "Hematocrit",
    "Mean Corpuscular Volume",
    "Mean Corpuscular Hemoglobin",
    "Mean Corpuscular Hemoglobin Concentration",
    "Insulin",
    "BMI",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Triglycerides",
    "HbA1c",
    "LDL Cholesterol",
    "HDL Cholesterol",
    "ALT",
    "AST",
    "Heart Rate",
    "Creatinine",
    "Troponin",
    "C-reactive Protein",
]

# Typical real-world ranges for display (used as slider defaults)
FEATURE_RANGES = {
    "Glucose":          (50.0, 300.0, 100.0),
    "Cholesterol":      (100.0, 400.0, 200.0),
    "Hemoglobin":       (5.0, 20.0, 14.0),
    "Platelets":        (50000.0, 500000.0, 250000.0),
    "White Blood Cells": (2000.0, 20000.0, 7000.0),
    "Red Blood Cells":  (2.0, 8.0, 5.0),
    "Hematocrit":       (20.0, 60.0, 42.0),
    "Mean Corpuscular Volume": (60.0, 110.0, 85.0),
    "Mean Corpuscular Hemoglobin": (20.0, 40.0, 29.0),
    "Mean Corpuscular Hemoglobin Concentration": (28.0, 38.0, 33.0),
    "Insulin":          (2.0, 300.0, 15.0),
    "BMI":              (10.0, 50.0, 25.0),
    "Systolic Blood Pressure": (80.0, 200.0, 120.0),
    "Diastolic Blood Pressure": (40.0, 130.0, 80.0),
    "Triglycerides":    (30.0, 500.0, 150.0),
    "HbA1c":            (3.0, 15.0, 5.5),
    "LDL Cholesterol":  (30.0, 250.0, 100.0),
    "HDL Cholesterol":  (15.0, 100.0, 55.0),
    "ALT":              (5.0, 200.0, 30.0),
    "AST":              (5.0, 200.0, 30.0),
    "Heart Rate":       (40.0, 150.0, 75.0),
    "Creatinine":       (0.3, 5.0, 1.0),
    "Troponin":         (0.0, 2.0, 0.02),
    "C-reactive Protein": (0.0, 50.0, 3.0),
}

DISEASE_COLORS = {
    "Anemia": "#e74c3c",
    "Diabetes": "#f39c12",
    "Healthy": "#2ecc71",
    "Heart Disease": "#9b59b6",
    "Thalassemia": "#3498db",
    "Thrombocytosis": "#e67e22",
}

DISEASE_DESCRIPTIONS = {
    "Anemia": "A condition where you lack enough healthy red blood cells to carry adequate oxygen to your tissues.",
    "Diabetes": "A group of metabolic diseases characterized by high blood sugar levels over a prolonged period.",
    "Healthy": "No significant disease detected based on the provided blood sample values.",
    "Heart Disease": "A range of conditions that affect the heart, including coronary artery disease and heart failure.",
    "Thalassemia": "An inherited blood disorder causing the body to have less hemoglobin than normal.",
    "Thrombocytosis": "A condition of high platelet count in the blood, which can cause clotting issues.",
}


@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file **{MODEL_PATH}** not found. Make sure the model is in the same directory as this app.")
        st.stop()


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🩸 Blood Sample Disease Detection")
st.markdown(
    "Enter your blood test values below and click **Predict** to detect potential diseases. "
    "The model was trained on a blood samples dataset using a Stacking Classifier pipeline."
)

model = load_model()

# ── Sidebar – Input Mode ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Input Mode")
input_mode = st.sidebar.radio("Choose input method:", ["Sliders", "Manual Entry", "CSV Upload"])

features = {}

if input_mode == "Sliders":
    st.subheader("📊 Adjust Blood Parameters")
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i, feat in enumerate(FEATURE_NAMES):
        lo, hi, default = FEATURE_RANGES[feat]
        with columns[i % 3]:
            features[feat] = st.slider(feat, min_value=lo, max_value=hi, value=default, step=(hi - lo) / 200)

elif input_mode == "Manual Entry":
    st.subheader("✏️ Enter Blood Parameters")
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i, feat in enumerate(FEATURE_NAMES):
        _, _, default = FEATURE_RANGES[feat]
        with columns[i % 3]:
            features[feat] = st.number_input(feat, value=default, format="%.4f")

elif input_mode == "CSV Upload":
    st.subheader("📁 Upload a CSV File")
    st.markdown(f"The CSV must have these columns: `{', '.join(FEATURE_NAMES)}`")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            missing = [c for c in FEATURE_NAMES if c not in upload_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.dataframe(upload_df[FEATURE_NAMES].head(20), use_container_width=True)
                if st.button("🔍 Predict All Rows"):
                    X_upload = upload_df[FEATURE_NAMES].values
                    preds = model.predict(X_upload)
                    upload_df["Predicted Disease"] = [DISEASE_LABELS.get(p, "Unknown") for p in preds]
                    st.success("Predictions complete!")
                    st.dataframe(upload_df, use_container_width=True)

                    csv = upload_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ── Single Prediction ────────────────────────────────────────────────────────
if input_mode in ["Sliders", "Manual Entry"]:
    st.divider()
    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        input_array = np.array([[features[f] for f in FEATURE_NAMES]])
        prediction = model.predict(input_array)[0]
        disease = DISEASE_LABELS.get(prediction, "Unknown")
        color = DISEASE_COLORS.get(disease, "#555")

        st.divider()

        # Result Card
        col_result, col_info = st.columns([1, 2])

        with col_result:
            st.markdown(
                f"""
                <div style="background:{color}22; border-left: 5px solid {color};
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color:{color}; margin:0;">🏥 {disease}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_info:
            st.info(DISEASE_DESCRIPTIONS.get(disease, ""))

        # Show input summary
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame(
                {"Feature": FEATURE_NAMES, "Value": [features[f] for f in FEATURE_NAMES]}
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ This tool is for educational purposes only and should not replace professional medical advice.")
