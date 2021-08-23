import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction")

st.image('photo.jpg', use_column_width='always')

st.header("Please Enter Data below")
st.subheader("Warning : It is an academic project, please don't treat it as medical advice")
st.write(' ')

def data():
    
    preg = st.slider("How many time pregnant")
    plasma = st.number_input("Plasma glucose concentration in oral glucose tolerance test (0-199)")
    bp = st.number_input("Diastolic blood pressure in (mm Hg)")
    skin_thickness = st.number_input("Triceps skin fold thickness (mm) (0-99)")
    insulin = st.number_input("2-Hour serum insulin (mu U/ml) (0-846)")
    bmi = st.number_input("Body mass index (BMI)")
    dpf = st.number_input("Diabetes Pedigree Function value (0.08 - 2.42)")
    age = st.slider("Age")
    
    X_new = np.array([[preg,plasma,bp,skin_thickness,insulin,bmi,dpf,age]])
    
    return X_new
    

pred_data = data()
st.header("Your Diabetes Data is")
st.write(pred_data)
pred = model.predict(pred_data)


def prediction(pred):
    
    if pred[0] == 1:
        st.write("You have diabetes")
    else:
        st.write("You don't have diabetes")
        
if st.button("Submit"):
    prediction(pred)

