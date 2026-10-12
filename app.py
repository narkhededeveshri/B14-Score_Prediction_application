import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model and scaler (Ensure these files are in your directory)
# model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Machine Learning Prediction App")
st.write("Please enter the required details below:")

# --- 1. USER INPUTS ---
# Using selectbox for categorical features
gender_input = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)

# --- 2. FIX: CONVERT CATEGORICAL STRINGS TO NUMBERS ---
# Map 'Male' and 'Female' to numbers (adjust 1/0 based on how your model was trained)
gender_encoded = 1 if gender_input == "Male" else 0

# --- 3. PREDICTION LOGIC ---
if st.button("Predict"):
    try:
        # Create a DataFrame or array with the exact feature order used during training
        # Example features: [Gender, Age, Income]
        input_data = pd.DataFrame([[gender_encoded, age, income]], 
                                  columns=['Gender', 'Age', 'Income'])
        
        # If you used a scaler during training, scale the data here:
        # scaled_data = scaler.transform(input_data)
        # prediction = model.predict(scaled_data)
        
        # Temporary simulation for demonstration (Replace with your model.predict)
        st.success("Prediction generated successfully!")
        st.write(f"**Processed Input Data:**")
        st.dataframe(input_data)
        
        # st.write(f"Result: {prediction[0]}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
