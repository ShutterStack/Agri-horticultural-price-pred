import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from datetime import datetime

# Load models
rf_model = joblib.load("rf_price_model.pkl")
xgb_model = xgb.Booster()  # Use Booster instead of XGBRegressor
xgb_model.load_model("xgb_price_model.json")  # Load the .json file directly

# Streamlit app
st.title("Agri-Horticultural Price Prediction")
st.write("Predict commodity prices using Random Forest or XGBoost models.")

# User inputs (matching Colab features)
commodity = st.selectbox("Commodity", ["onion", "gram"])
date = st.date_input("Date", value=datetime(2024, 1, 1))
rainfall = st.slider("Rainfall (mm/day)", 0.0, 50.0, 10.0)
production = st.slider("Production (tons)", 500.0, 1500.0, 1000.0)
lag_price_7d = st.number_input("Price 7 Days Ago (INR/kg)", min_value=10.0, value=30.0)
model_choice = st.selectbox("Model", ["Random Forest", "XGBoost"])

# Prediction function
def predict_price(model, day_of_year, rainfall, production, lag_price_7d):
    # Prepare features as a DataFrame
    features = pd.DataFrame({
        "day_of_year": [day_of_year],
        "rainfall": [rainfall],
        "production": [production],
        "lag_price_7d": [lag_price_7d]
    })
    
    if model_choice == "XGBoost":
        try:
            # Convert to numpy array and create DMatrix
            feature_array = features.to_numpy()
            st.write(f"Features for XGBoost: {feature_array}")  # Debug
            dmatrix = xgb.DMatrix(feature_array, feature_names=features.columns.tolist())
            st.write(f"DMatrix created: {dmatrix.num_row()} rows, {dmatrix.num_col()} cols")  # Debug
            price = model.predict(dmatrix)[0]
            st.write(f"Raw XGBoost prediction: {price}")  # Debug
        except Exception as e:
            st.error(f"XGBoost prediction failed: {str(e)}")
            raise
    else:  # Random Forest
        price = model.predict(features)[0]
    
    return price

# Predict button
if st.button("Predict Price"):
    try:
        day_of_year = pd.Timestamp(date).dayofyear
        selected_model = xgb_model if model_choice == "XGBoost" else rf_model
        predicted_price = predict_price(selected_model, day_of_year, rainfall, production, lag_price_7d)
        st.success(f"Predicted Price for {commodity} on {date} using {model_choice}: **{predicted_price:.2f} INR/kg**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Additional info
st.write("### About")
st.write("This app uses models trained on synthetic data to predict prices, incorporating seasonality, weather, and production. XGBoost and Random Forest overcome ARIMA's limitations.")