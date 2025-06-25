import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = tf.keras.models.load_model('model.keras', compile=False, safe_mode=False)

# Load the encoders and scaler
with open('le.pkl', 'rb') as file:
    le = pickle.load(file)
with open('onehot_encoder_geo', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", le.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Account Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (Years with bank)", 0, 10)
num_of_products = st.selectbox("Number of Products", list(range(1, 5)))
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# Convert to DataFrame
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine one-hot with input
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'🔮 **Churn Probability:** `{prediction_prob:.2f}`')

if prediction_prob > 0.5:
    st.error("❌ The customer is **likely to churn**.")
else:
    st.success("✅ The customer is **not likely to churn**.")
