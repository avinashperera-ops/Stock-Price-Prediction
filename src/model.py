# src/model.py

import pandas as pd
from prophet import Prophet

def predict_stock(df, forecast_days):
    try:
        df_prophet = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        if df_prophet["ds"].dt.tz is not None:
            df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
        
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        return forecast
    except Exception:
        return None

def generate_signals(forecast):
    signals = forecast[["ds", "yhat"]].copy()
    signals = signals.rename(columns={"ds": "Date", "yhat": "Predicted_Price"})
    signals["SMA_20"] = signals["Predicted_Price"].rolling(window=20).mean()
    signals["SMA_50"] = signals["Predicted_Price"].rolling(window=50).mean()
    signals["Signal"] = 0
    signals.loc[signals["SMA_20"] > signals["SMA_50"], "Signal"] = 1
    signals.loc[signals["SMA_20"] < signals["SMA_50"], "Signal"] = -1
    return signals