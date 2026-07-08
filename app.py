import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Stroke Prediction System")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "stroke_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ stroke_model.pkl not found!")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model:\n{e}")
    st.stop()

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Female", "Male", "Other"])

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

ever_married = st.selectbox(
    "Ever Married",
    ["No", "Yes"]
)

work_type = st.selectbox(
    "Work Type",
    [
        "Govt_job",
        "Never_worked",
        "Private",
        "Self-employed",
        "children"
    ]
)

Residence_type = st.selectbox(
    "Residence Type",
    ["Rural", "Urban"]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    value=100.0
)

bmi = st.number_input(
    "BMI",
    value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    [
        "Unknown",
        "formerly smoked",
        "never smoked",
        "smokes"
    ]
)

# -----------------------------
# Manual Encoding
# -----------------------------
gender = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}[gender]

ever_married = {
    "No": 0,
    "Yes": 1
}[ever_married]

work_type = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}[work_type]

Residence_type = {
    "Rural": 0,
    "Urban": 1
}[Residence_type]

smoking_status = {
    "Unknown": 0,
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3
}[smoking_status]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Stroke"):

    data = pd.DataFrame({
        "gender":[gender],
        "age":[age],
        "hypertension":[hypertension],
        "heart_disease":[heart_disease],
        "ever_married":[ever_married],
        "work_type":[work_type],
        "Residence_type":[Residence_type],
        "avg_glucose_level":[avg_glucose_level],
        "bmi":[bmi],
        "smoking_status":[smoking_status]
    })

    try:
        prediction = model.predict(data)

        probability = model.predict_proba(data)[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

        st.write(f"Stroke Probability: **{probability*100:.2f}%**")

    except Exception as e:
        st.error(e)
