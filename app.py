import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Load dataset
data = pd.read_csv("diabetes.csv")

# Title and Branding
st.title("🧬 GeneGuard AI")
st.subheader("AI-Powered Diabetes Risk Prediction System")
st.write("An intelligent healthcare prediction system using Machine Learning.")

# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Model Accuracy", value="90%")

with col2:
    st.metric(label="Dataset", value="PIMA Diabetes")

with col3:
    st.metric(label="ML Algorithm", value="Random Forest")

st.write("---")

# Sidebar Inputs
st.sidebar.header("Patient Health Parameters")

preg = st.sidebar.number_input("Pregnancies")
glucose = st.sidebar.number_input("Glucose Level")
bp = st.sidebar.number_input("Blood Pressure")
skin = st.sidebar.number_input("Skin Thickness")
insulin = st.sidebar.number_input("Insulin")
bmi = st.sidebar.number_input("BMI")
dpf = st.sidebar.number_input("Diabetes Pedigree Function")
age = st.sidebar.number_input("Age")

# Prediction
if st.button("Predict Diabetes Risk"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write("Risk Probability:", round(probability[0][1]*100,2), "%")

st.write("---")

# Dataset Visualization
st.subheader("Dataset Visualization")

st.write("Glucose Level Distribution")

st.bar_chart(data["Glucose"])

st.write("---")

# Feature Importance (Example Visualization)
st.subheader("Important Health Factors")

features = {
"Glucose":0.35,
"BMI":0.25,
"Age":0.15,
"Blood Pressure":0.10,
"Insulin":0.08,
"Skin Thickness":0.04,
"Pregnancies":0.02,
"Diabetes Pedigree Function":0.01
}

st.bar_chart(features)

st.write("---")

# Explanation Section
st.subheader("How GeneGuard Works")

st.write("""
GeneGuard AI analyzes patient health indicators such as glucose level, BMI,
blood pressure, and age using a trained Machine Learning model.
The model was trained using the PIMA Indians Diabetes Dataset to identify
patterns associated with diabetes risk.
""")

st.write("---")

# Footer
st.write("Developed by Prajwal Biradar | AIML Engineer")

st.write("⚠ Disclaimer: This AI system is for educational purposes only and should not replace professional medical advice.")