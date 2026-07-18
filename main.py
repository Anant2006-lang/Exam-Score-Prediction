import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

# -------------------------------
# Load the saved model
# -------------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("best_xgboost_model.json")
    return model


# -------------------------------
# Load the label encoders
# -------------------------------
@st.cache_resource
def load_encoders():
    with open("label_encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    return encoders


# Load resources
model = load_model()
label_encoders = load_encoders()

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🎓 Exam Score Prediction App")
st.write("Enter the student details to predict the exam score.")

st.header("Student Information")

study_hours = st.slider(
    "Study Hours (per day)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
)

class_attendance = st.slider(
    "Class Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=0.5,
)

sleep_hours = st.slider(
    "Sleep Hours (per night)",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.1,
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    list(label_encoders["sleep_quality"].classes_)
)

study_method = st.selectbox(
    "Study Method",
    list(label_encoders["study_method"].classes_)
)

facility_rating = st.selectbox(
    "Facility Rating",
    list(label_encoders["facility_rating"].classes_)
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Exam Score"):

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "class_attendance": [class_attendance],
        "sleep_hours": [sleep_hours],
        "sleep_quality": [sleep_quality],
        "study_method": [study_method],
        "facility_rating": [facility_rating]
    })

    # Encode categorical columns
    for feature in label_encoders:
        input_data[feature] = label_encoders[feature].transform(input_data[feature])

    # Ensure correct feature order
    input_data = input_data[
        [
            "study_hours",
            "class_attendance",
            "sleep_hours",
            "sleep_quality",
            "study_method",
            "facility_rating",
        ]
    ]

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Exam Score: **{prediction:.2f}**")
    st.balloons()