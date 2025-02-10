import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('lrmodel_sustainable.pkl')

# Set up the Streamlit app
st.title('Sustainability Prediction Model')

# Get user input for features
st.sidebar.header('Input Parameters')
carbon_emissions = st.sidebar.slider('Carbon Emissions', 50, 400, 200)
energy_output = st.sidebar.slider('Energy Output', 100, 1000, 500)
renewability_index = st.sidebar.slider('Renewability Index', 0.0, 1.0, 0.5)
cost_efficiency = st.sidebar.slider('Cost Efficiency', 0.5, 5.0, 2.5)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'carbon_emissions': [carbon_emissions],
    'energy_output': [energy_output],
    'renewability_index': [renewability_index],
    'cost_efficiency': [cost_efficiency]
})

# Make predictions
prediction = model.predict(input_data)

# Display the result
if prediction == 1:
    st.success('The system is likely to be Sustainable!')
else:
    st.error('The system is likely to be Not Sustainable.')

