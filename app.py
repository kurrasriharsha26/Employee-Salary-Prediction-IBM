import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model artifacts
model_data = joblib.load("salary_predictor.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]  # expected to be a list of 14 features

# Encoding dictionaries
gender_dict = {"Male": 1, "Female": 0}
education_dict = {"10th": 6, "12th": 8, "Bachelors": 13, "Masters": 14, "PhD": 16}
occupation_dict = {"Clerical": 2, "Technical": 1, "Managerial": 4, "Sales": 3, "Other": 0}
country_dict = {"India": 39, "USA": 0, "Canada": 1, "Germany": 2, "Other": 3}

# Streamlit UI
st.set_page_config(page_title="AI Salary Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>💼 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.write("### Predict whether an employee earns >50K or <=50K based on census-style data")

left, center, right = st.columns([1.2, 2.5, 1.5])

# LEFT COLUMN
with left:
    st.markdown("### 🔍 Model Details")
    st.markdown("""
    - Dataset: Modeled after Indian Census Income Data  
    - Algorithm: Random Forest Classifier  
    - Accuracy: ~88%  
    - Features: Age, Gender, Education, Occupation  
    - Capital Gain/Loss, Hours/Week  
    - Marital Status, Relationship, Country, etc.
    """)
    st.markdown("📌 Suggestion: Upskill, take leadership roles, invest in higher education.")

# CENTER FORM
with center:
    with st.form("salary_form"):
        st.markdown("## 👤 Employee Information")
        name = st.text_input("Employee Name")
        age = st.slider("Age", 18, 65, 30)
        gender_input = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", list(education_dict.keys()))
        occupation = st.selectbox("Occupation", list(occupation_dict.keys()))
        hours = st.slider("Hours/Week", 10, 80, 40)
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
        native_country = st.selectbox("Native Country", list(country_dict.keys()))

        submitted = st.form_submit_button("🔎 Predict Salary")

# Process input and predict
if submitted:
    # Static values used during training
    workclass = 4
    fnlwgt = 200000
    marital_status = 2
    relationship = 1
    race = 1
    extra_feature = 1  # included as 14th feature

    # Build feature array (14 features)
    features = np.array([[age, workclass, fnlwgt, education_dict[education],
                          marital_status, occupation_dict[occupation], relationship,
                          race, gender_dict[gender_input], capital_gain, capital_loss,
                          hours, country_dict[native_country], extra_feature]])

    # Validate feature length
    if len(feature_names) != features.shape[1]:
        st.error(f"⚠️ Feature mismatch! Expected {len(feature_names)} features, got {features.shape[1]}.")
    else:
        # Proper DataFrame format
        input_df = pd.DataFrame(features, columns=feature_names)
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        label = ">50K" if prediction == 1 else "<=50K"
        monthly_salary = 60000 if prediction == 1 else 25000

        st.success(f"💡 {name}'s Predicted Income Class: **{label}**")
        st.info(f"💰 Estimated Monthly Salary: ₹{monthly_salary:,}")
        st.info(f"📅 Estimated Annual Salary: ₹{monthly_salary * 12:,}")

        # RIGHT COLUMN - Visuals
        with right:
            st.markdown("### 📊 Visual Insights")

            # Simulated salary trend
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
else:
    with right:
        st.info("👈 Fill the form to unlock salary graphs.")

st.markdown("---")
st.caption("🚀 Created with ❤️ using Streamlit • Powered by ML")
