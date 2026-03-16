import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("csat_model.pkl", "rb"))

st.title("DeepCSAT - Customer Satisfaction Predictor")

st.write("Enter support details to predict CSAT score")

channel_name = st.number_input("Channel Name")
category = st.number_input("Category")
sub_category = st.number_input("Sub Category")
product_category = st.number_input("Product Category")
customer_city = st.number_input("Customer City")
item_price = st.number_input("Item Price")
connected_handling_time = st.number_input("Connected Handling Time")
response_time = st.number_input("Response Time")
agent_shift = st.number_input("Agent Shift")
tenure_bucket = st.number_input("Tenure Bucket")

if st.button("Predict CSAT"):

    features = np.array([[channel_name, category, sub_category,
                          product_category, customer_city,
                          item_price, connected_handling_time,
                          response_time, agent_shift, tenure_bucket]])

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {round(prediction[0],2)}")