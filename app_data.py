import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load Model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏠HOUSE PRICE PREDICTION")

st.write(
    "Predict the land price per unit area using machine learning."
)

st.sidebar.header("House Details")

# Transaction Year
year = st.sidebar.selectbox(
    "Transaction Year",
    options=list(range(2000, 2027)),
    index=13 
)

# Transaction Month
month = st.sidebar.selectbox(
    "Transaction Month",
    [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]
)

# Convert Year & Month to decimal format used by the model
month_decimal = {
    "January": 0.000,
    "February": 0.083,
    "March": 0.167,
    "April": 0.250,
    "May": 0.333,
    "June": 0.417,
    "July": 0.500,
    "August": 0.583,
    "September": 0.667,
    "October": 0.750,
    "November": 0.833,
    "December": 0.917
}

transaction_date = year + month_decimal[month]

house_age = st.sidebar.number_input(
    "House Age in years",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

distance = st.sidebar.number_input(
    "Distance to BUS Station in Meters",
    min_value=0.0,
    value=500.0
)

stores = st.sidebar.number_input(
    "Number of Convenience Stores",
    min_value=0,
    max_value=20,
    value=5
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=24.97,
    format="%.5f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=121.54,
    format="%.5f"
)

if st.sidebar.button("Predict Price"):

    data = pd.DataFrame({
        "X1 transaction date":[transaction_date],
        "X2 house age":[house_age],
        "X3 distance to the nearest MRT station":[distance],
        "X4 number of convenience stores":[stores],
        "X5 latitude":[latitude],
        "X6 longitude":[longitude]
    })

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    st.success(
        f"🏠 Predicted House Price Per Unit Area: {prediction[0]:.2f}"
    )