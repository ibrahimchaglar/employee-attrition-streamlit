import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Model ve scaler yükle
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🔍")

st.title("🔍 Employee Attrition Prediction App")
st.write("Fill out the form below to see if the employee is likely to leave the company.")

# Kullanıcıdan veri al
satisfaction_level = st.number_input("Satisfaction Level (0–1)", min_value=0.0, max_value=1.0, step=0.01)
average_monthly_hours = st.number_input("Average Monthly Hours", min_value=96, max_value=310, step=1)

work_accident = st.selectbox("Work Accident?", ["No", "Yes"])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])

department = st.selectbox("Department", [
    "sales", "accounting", "hr", "management", "marketing",
    "product_mng", "support", "technical", "RandD"
])

salary = st.selectbox("Salary Level", ["low", "medium", "high"])

if st.button("Predict"):
    # input_dict oluştur
    input_dict = {feature: 0 for feature in feature_names}
    input_dict["satisfaction_level"] = satisfaction_level
    input_dict["average_monthly_hours"] = average_monthly_hours
    input_dict["Work_accident"] = 1 if work_accident == "Yes" else 0
    input_dict["promotion_last_5years"] = 1 if promotion_last_5years == "Yes" else 0

    dept_col = f"Department_{department}"
    if dept_col in input_dict:
        input_dict[dept_col] = 1

    if salary == "low":
        input_dict["salary_low"] = 1
    elif salary == "medium":
        input_dict["salary_medium"] = 1

    df = pd.DataFrame([input_dict], columns=feature_names)
    scaled_input = scaler.transform(df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.warning("⚠️ This employee is likely to leave the company. Consider proactive engagement.")
    else:
        st.success("✅ This employee is likely to stay with the company.")