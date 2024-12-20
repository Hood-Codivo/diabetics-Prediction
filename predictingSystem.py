"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import streamlit as st

# Open the file in binary read mode and load the model
with open(r'C:\Users\GODWIN\Downloads\trained_model.sav', 'rb') as file:
    loaded_data = pickle.load(file)


# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Ensure data is numeric

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_data.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # giving a title
    st.title('Diabetes Prediction Web App for Women')

    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', key='pregnancies')
    Glucose = st.text_input('Glucose Level', key='glucose')
    BloodPressure = st.text_input('Blood Pressure', key='blood_pressure')
    SkinThickness = st.text_input('Skin Thickness', key='skin_thickness')
    Insulin = st.text_input('Insulin Level', key='insulin')
    BMI = st.text_input('BMI level', key='bmi')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', key='dpf')
    Age = st.text_input('Age', key='age')

    # code for Prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to floats
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]
            # Make prediction
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numeric values for all inputs.'

    st.success(diagnosis)


if __name__ == '__main__':
    main()
