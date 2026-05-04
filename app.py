import streamlit as st
import joblib

# load model
model = joblib.load("model.pkl")

st.title("Salary Prediction App")

years = st.number_input("Enter Years of Experience", 0.0, 10.0, 1.0)

if st.button("Predict"):
    result = model.predict([[years]])
    st.success(f"Predicted Salary: ₹ {result[0]:,.2f}")