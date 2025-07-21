import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie JSON
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URL
cash_lottie_url = "https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_cash = load_lottieurl(cash_lottie_url)

# Create two columns: left for animation, right for instructions
col1, col2 = st.columns([1, 2])  # You can adjust the ratio

with col1:
    st_lottie(lottie_cash, width=120, height=150, speed=1, loop=True)

with col2:
    st.markdown("""
    <div style="background: #f0f8ff;
                border-left: 5px solid #2a5298;
                padding: 16px 24px;
                border-radius: 8px;">
        <h4>🧭 How This Works</h4>
        <ol>
            <li>Fill out the employee information</li>
            <li>Click “🔎 Predict Salary”</li>
            <li>View the salary class and estimated income 💰</li>
        </ol>
        <p style="font-size:15px;color:#6497b1;">
            This prediction is based on machine learning trained on salary data.
        </p>
    </div>
    """, unsafe_allow_html=True)



# Inject animated circulating background using CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298, #f0f8ff, #6497b1, #3a5683, #1e3c72);
        background-size: 400% 400%;
        animation: gradientBG 20s ease-in-out infinite;
        /* Ensure the background covers the whole app window */
        min-height: 100vh;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
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
