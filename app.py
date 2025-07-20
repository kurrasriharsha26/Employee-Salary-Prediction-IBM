import streamlit as st
import pickle
import numpy as np
import pandas as pd  # ✅ Added missing import
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load model and related objects
model_data = joblib.load("salary_predictor.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

# Streamlit UI Setup
st.set_page_config(page_title="AI Salary Predictor", layout="wide", page_icon="💰")

st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    .title-style {
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-style'>💼 Employee Salary Predictor</div>", unsafe_allow_html=True)

left, center, right = st.columns([1.2, 2.5, 1.5])

with center:
    with st.form("salary_form"):
        st.subheader("👤 Employee Information")
        name = st.text_input("Employee Name")
        age = st.slider("Age", 18, 65, 30)
        gender_input = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["10th", "12th", "Bachelors", "Masters", "PhD"])
        occupation = st.selectbox("Occupation", ["Clerical", "Technical", "Managerial", "Sales", "Other"])
        hours = st.slider("Hours/Week", 10, 80, 40)
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
        native_country = st.selectbox("Native Country", ["India", "USA", "Canada", "Germany", "Other"])
        submitted = st.form_submit_button("🔎 Predict Salary")

        if submitted:
            # Feature dictionaries
            gender_dict = {"Male": 1, "Female": 0}
            education_dict = {"10th": 6, "12th": 8, "Bachelors": 13, "Masters": 14, "PhD": 16}
            occupation_dict = {"Clerical": 2, "Technical": 1, "Managerial": 4, "Sales": 3, "Other": 0}
            country_dict = {"India": 39, "USA": 0, "Canada": 1, "Germany": 2, "Other": 3}

            # Add static or default values
            marital_status = 2
            relationship = 1
            race = 1
            extra_feature = 1
            workclass = 4
            fnlwgt = 200000

            # Construct the input feature list
            features = np.array([[
                age, workclass, fnlwgt, education_dict[education], marital_status,
                occupation_dict[occupation], relationship, race, gender_dict[gender_input],
                capital_gain, capital_loss, hours, country_dict[native_country], extra_feature
            ]])

            # Create DataFrame with correct column names
            input_df = pd.DataFrame(features, columns=feature_names)

            # Scale & Predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            # Output
            label = ">50K" if prediction == 1 else "<=50K"
            st.success(f"💡 {name}'s Predicted Income Class: **{label}**")

            monthly_salary = 60000 if prediction == 1 else 25000
            st.info(f"💰 Estimated Monthly Salary: ₹{monthly_salary:,}")
            st.info(f"📅 Estimated Annual Salary: ₹{monthly_salary * 12:,}")

            # Graphs
            salary_trend = [monthly_salary + np.random.randint(-2000, 2000) for _ in range(6)]
            st.markdown("#### 📈 6-Month Salary Projection")
            st.line_chart(salary_trend)

            st.markdown("#### 💼 Avg. Monthly Salary by Role")
            st.bar_chart({
                "Clerical": 22000,
                "Technical": 35000,
                "Managerial": 65000,
                "Sales": 30000,
                "Other": 28000
            })

# Caption
st.markdown("---")
st.caption("🚀 Created with ❤️ using Streamlit • Powered by Machine Learning")
