import streamlit as st
import pickle
import numpy as np

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open('knn.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Student Performance Prediction App")
st.write("Enter the student details below to generate a prediction using your K-Nearest Neighbors Regressor model[cite: 1].")

# Input widgets for each feature[cite: 1]
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=5, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    course = st.selectbox("Course", ["Science", "Arts", "Commerce", "Engineering", "Other"])
    study_hours = st.slider("Study Hours per Day", 0.0, 16.0, 4.0)
    class_attendance = st.slider("Class Attendance (%)", 0.0, 100.0, 85.0)
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])

with col2:
    sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
    sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
    study_method = st.selectbox("Study Method", ["Self-Study", "Group Study", "Online Courses", "Coaching"])
    facility_rating = st.slider("Facility Rating (1-5)", 1, 5, 3)
    exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])

# Note: Adjust these mappings if your training script used a different encoding method (e.g., One-Hot Encoding)
gender_encoded = 1 if gender == "Male" else (2 if gender == "Other" else 0)
internet_encoded = 1 if internet_access == "Yes" else 0

course_map = {"Science": 0, "Arts": 1, "Commerce": 2, "Engineering": 3, "Other": 4}
sleep_quality_map = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
study_method_map = {"Self-Study": 0, "Group Study": 1, "Online Courses": 2, "Coaching": 3}
exam_difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}

if st.button("Predict"):
    # Construct feature array matching the model's expected input order[cite: 1]
    features = np.array([[
        age,
        gender_encoded,
        course_map.get(course, 0),
        study_hours,
        class_attendance,
        internet_encoded,
        sleep_hours,
        sleep_quality_map.get(sleep_quality, 2),
        study_method_map.get(study_method, 0),
        facility_rating,
        exam_difficulty_map.get(exam_difficulty, 2)
    ]])
    
    # Make prediction
    prediction = model.predict(features)
    st.success(f"Predicted Output: {prediction[0]:.2f}")
