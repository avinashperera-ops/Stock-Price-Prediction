# src/model.py

import streamlit as st  # Add this import
import pandas as pd
from prophet import Prophet

def train_model(df, model_choice):
    # Placeholder for model training
    pass

def predict_stock(df, forecast_days, model_choice):
    try:
        # Prepare DataFrame for Prophet
        df_prophet = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        # Double-check that 'ds' is timezone-naive
        if df_prophet["ds"].dt.tz is not None:
            df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
        
        # Initialize and fit the Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def generate_signals(forecast):
    signals = pd.DataFrame(index=forecast.index)
    signals["ds"] = forecast["ds"]
    signals["yhat"] = forecast["yhat"]
    signals["SMA20"] = signals["yhat"].rolling(window=20).mean()
    signals["SMA50"] = signals["yhat"].rolling(window=50).mean()
    signals["Signal"] = 0
    signals.loc[signals["SMA20"] > signals["SMA50"], "Signal"] = 1  # Buy
    signals.loc[signals["SMA20"] < signals["SMA50"], "Signal"] = -1  # Sell
    return signals
