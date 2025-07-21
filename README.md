📊 Employee Salary Predictor using AI/ML (Streamlit App)


A machine learning-based web app to predict whether an employee earns more than ₹50K or less than ₹50K annually, based on demographic and professional features — inspired by the Indian Census dataset.

🔗 Live Demo
👉 Click here to try the app ("https://employee-salary-prediction-ibm-oujclshorpofbckc8jcrct.streamlit.app/")

📌 Features
✅ Predict salary class (>50K or <=50K) using ML

✅ Streamlit web interface with form-based input

✅ Visual insights: 6-month salary trend, role-wise salary bar chart

✅ Interactive UI with custom styling

🧠 Machine Learning Model

Algorithm: Random Forest Classifier
Accuracy: ~88%
Trained using: Scikit-learn
Scaler: StandardScaler
Model File: salary_predictor.pkl

🧾 Input Features

Feature	Type	Example

Age	Integer	30

Gender	Categorical (Male/Female)	Female

Education	Categorical (10th, 12th, Bachelors, Masters, PhD)	

Occupation	Categorical (Clerical, Technical, Managerial, Sales, Other)

Hours/Week	Integer	40

Capital Gain	Integer	0

Capital Loss	Integer	0

Native Country	Categorical (India, USA, Canada, etc.)	

🛠️ How to Run Locally

✅ Prerequisites
Python 3.8+
pip installed

🔧 Installation
bash
Copy
Edit

# Clone the repo
git clone https://github.com/kurrasriharsha26/employee-salary-prediction.git
cd employee-salary-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
📁 Project Structure
bash
Copy
Edit
employee-salary-prediction/
│
├── app.py                  # Streamlit app
├── salary_predictor.pkl    
├── requirements.txt        
├── assets/                 
└── README.md               
🖼️ Sample Screenshots
Form Input Prediction + Graphs
<img width="2240" height="1400" alt="image" src="https://github.com/user-attachments/assets/22d169f6-a3fc-407a-a03d-af45605eba1a" />
<img width="2240" height="1400" alt="image" src="https://github.com/user-attachments/assets/d068a719-9dce-4808-b302-bcb8ab44cd59" />


🚀 Deployment Options
✅ Streamlit Cloud
✅ Render
✅ ngrok (for local demo sharing)
✅ Docker (advanced users)
🙋‍♂️ Author
Developed by [Kurra Sriharsha]
🔗 LinkedIn:
www.linkedin.com/in/kurra-sri-harsha26
📧 Email: kurrasriharsha49@gmail.com
⭐️ Show Your Support

If you like this project, consider:
Giving a ⭐️ on GitHub
Sharing it with friends
Forking it and building your own improvements!

📄 License
This project is for educational and internship purposes. All rights reserved by the author.
