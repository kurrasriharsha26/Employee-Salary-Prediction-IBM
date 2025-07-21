import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 30% 30%, #F0F8FF 60%, #e6e6fa 100%);
        animation: circulateBG 15s linear infinite;
        background-size: 150% 150%;
    }
    @keyframes circulateBG {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }    
    /* Optional: Add floating particles using pseudo-elements */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        background: url('https://assets.codepen.io/1462889/floating-particles.svg');
        opacity: 0.4;
        animation: floatParticles 30s linear infinite;
        z-index: 0;
    }
    @keyframes floatParticles {
        0% {background-position: 0 0;}
        100% {background-position: 300px 600px;}
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Load the trained model and components
model_data = joblib.load("salary_predictor.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

# Define encoding mappings
gender_dict = {"Male": 1, "Female": 0}
education_dict = {"10th": 6, "12th": 8, "Bachelors": 13, "Masters": 14, "PhD": 16}
occupation_dict = {"Clerical": 2, "Technical": 1, "Managerial": 4, "Sales": 3, "Other": 0}

# Page setup
st.set_page_config(page_title="AI Salary Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>💵 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.write("### Predict whether an employee earns >50K or <=50K")

left, center, right = st.columns([1.2, 2.5, 1.5])

with left:
    st.image(
        "https://thumbs.dreamstime.com/b/cloud-computing-technology-internet-storage-network-concept-114143142.jpg",
        caption="Technology",
        use_container_width=True
    )
    st.markdown("### 🔍 Model Details")
    st.markdown("""
    - Dataset: Custom
    - Algorithm: ML Classifier
    - **Features Used:**  
        - Age, Gender, Education, Occupation, Hours/Week  
    """)

with center:
    with st.form("salary_form"):
        st.markdown("## 👤 Employee Information")
        name = st.text_input("Employee Name")
        age = st.slider("Age", 18, 65, 30)
        gender_input = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", list(education_dict.keys()))
        occupation = st.selectbox("Occupation", list(occupation_dict.keys()))
        hours = st.slider("Hours/Week", 10, 80, 40)
        submitted = st.form_submit_button("🔎 Predict Salary")

if submitted:
    # Build feature vector exactly matching model training
    features = np.array([[age,
                          education_dict[education],
                          occupation_dict[occupation],
                          hours,
                          gender_dict[gender_input]]])
    # Build DataFrame with correct names/order
    input_df = pd.DataFrame(features, columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    label = ">50K" if prediction == 1 else "<=50K"
    monthly_salary = 60000 if prediction == 1 else 25000

    st.success(f"💡 {name}'s Predicted Income Class: **{label}**")
    st.info(f"💰 Estimated Monthly Salary: ₹{monthly_salary:,}")
    st.info(f"📅 Estimated Annual Salary: ₹{monthly_salary * 12:,}")

    with right:
        st.markdown("### 📊 Visual Insights")
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
