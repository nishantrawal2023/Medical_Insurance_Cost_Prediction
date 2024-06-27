import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# Load data
medical_df = pd.read_csv('insurance.csv')


# Data preprocessing
medical_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)


# Split data into features and target
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']


# Train the model
gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm.fit(X, y)


# Define Streamlit app
def main():
    st.title("Medical Insurance Cost Prediction")
    age = st.number_input("Enter age", min_value=0, step=1)
    sex = st.radio("Select sex", options=['Male', 'Female'])
    sex_encoded = 0 if sex == 'Male' else 1
    bmi = st.number_input("Enter BMI", min_value=0.0, step=0.1)
    children = st.number_input("Enter number of children", min_value=0, step=1)
    smoker = st.radio("Smoker?", options=['Yes', 'No'])
    smoker_encoded = 0 if smoker == 'Yes' else 1
    region = st.selectbox("Select region", options=['Southeast', 'Southwest', 'Northwest', 'Northeast'])
    region_encoded = ['Southeast', 'Southwest', 'Northwest', 'Northeast'].index(region)

    if st.button("Predict"):
        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        prediction = gbm.predict(input_data)
        st.write("Medical Insurance for this person is (USD):", prediction[0])


# Run Streamlit app
if __name__ == "__main__":
    main()
