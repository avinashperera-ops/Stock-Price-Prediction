# visualize.py

import streamlit as st
import plotly.graph_objs as go

def plot_historical_data(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Historical Price"))
    fig.update_layout(
        title=f"Historical Prices for {symbol if symbol else 'Uploaded Data'}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

def plot_forecast(df, forecast, symbol):
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Historical"))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(dash="dash")))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(color="rgba(255,0,0,0.2)")))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(color="rgba(0,255,0,0.2)"), fill="tonexty"))
    fig_forecast.update_layout(
        title=f"Price Forecast for {symbol if symbol else 'Uploaded Data'}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig_forecast)

def display_signals(signals):
    st.subheader("Buy/Sell Signals")
    st.dataframe(signals[["ds", "Signal"]].tail())
