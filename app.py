import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for shadow effects and styling cards
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN regressor model
@st.cache_resource
def load_model():
    with open("knn.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header Section
st.title("🎓 Student Performance & Outcome Predictor")
st.markdown("Provide the student details below to generate a prediction using your KNeighborsRegressor model[cite: 1].")

# Input Form inside a styled container card
with st.form("prediction_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Student Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course / Program", ["Science", "Arts", "Engineering", "Business", "Other"])
        study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=24.0, value=4.5, step=0.5)
        class_attendance = st.number_input("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])

    with col2:
        sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])
        study_method = st.selectbox("Study Method", ["Self-Study", "Group", "Online", "Hybrid"])
        facility_rating = st.slider("Facility Rating (1-10)", min_value=1, max_value=10, value=7)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])

    st.markdown('</div>', unsafe_allow_html=True)
    
    submit_button = st.form_submit_button(label="Predict Performance")

# Prediction Logic
if submit_button:
    # Construct input dataframe matching the feature names expected by the model
    # Note: If your model requires encoded numerical values for categorical inputs, 
    # make sure to apply the same label encoding or mapping used during training.
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty]
    })

    try:
        # If your model pipeline includes preprocessors/encoders, predict directly:
        prediction = model.predict(input_data)
        
        st.success("### Prediction Result")
        st.metric(label="Estimated Performance Metric / Score", value=f"{prediction[0]:.2f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Tip: If your `knn.pkl` expects numerical columns only, ensure categorical variables are mapped/encoded to numbers matching your training pipeline.")
