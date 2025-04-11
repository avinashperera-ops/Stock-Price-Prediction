# main.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Debug import to ensure data_fetch is accessible
try:
    from data_fetch import fetch_stock_data, fetch_sentiment_data
except ImportError as e:
    st.error(f"Failed to import fetch_stock_data from data_fetch: {e}")
    st.stop()

# Debug to confirm fetch_stock_data is defined
if 'fetch_stock_data' not in globals():
    st.error("fetch_stock_data is not defined after import. Please check data_fetch.py.")
    st.stop()

from model import train_model, predict_stock, generate_signals
from visualize import plot_historical_data, plot_forecast, display_signals

# NLTK downloads (for sentiment analysis, if used)
import nltk
nltk.download('vader_lexicon')

# Custom CSS for a dark theme
st.markdown("""
    <style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-color: #1e1e1e !important;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    /* Style the button */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    /* Style selectbox, slider, and text input */
    .stSelectbox, .stSlider, .stTextInput, .stDateInput {
        background-color: #2e2e2e;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    /* Style the title */
    h1 {
        color: white;
        text-align: center;
    }
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #2e2e2e !important;
        color: white;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Style alerts */
    .stAlert {
        border-radius: 5px;
    }
    /* Style dataframe */
    .stDataFrame {
        background-color: #2e2e2e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("Stock Price Prediction with Buy/Sell Signals")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Ticker input
st.sidebar.subheader("Ticker")
ticker = st.sidebar.text_input("Global Stock (e.g., AAPL, HNB)", "AAPL", key="global_stock_input")
is_sri_lankan = st.sidebar.checkbox("Sri Lankan Stock (e.g., HNB, COMB)", False, key="sri_lankan_checkbox")

# Date range
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start", datetime(2024, 4, 11), key="start_date_input")
end_date = st.sidebar.date_input("End", datetime(2025, 4, 11), key="end_date_input")

# Model selection
st.sidebar.subheader("Model")
model_choice = st.sidebar.selectbox("Choose Model", ["Prophet", "LSTM"], index=0, key="model_selectbox")

# Signal type
st.sidebar.subheader("Signal Type")
signal_type = st.sidebar.selectbox("Signal Type", ["Moving Average"], index=0, key="signal_selectbox")

# Option to upload a CSV file for Sri Lankan stocks
uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV) for Sri Lankan Stocks", type="csv", key="csv_uploader")
if uploaded_file is not None:
    st.sidebar.info("Using uploaded CSV data. Map the columns below if needed.")

# CSV column mapping
if uploaded_file is not None:
    df_temp = pd.read_csv(uploaded_file)
    columns = df_temp.columns.tolist()
    
    st.sidebar.subheader("Map CSV Columns")
    date_col = st.sidebar.selectbox("Select the Date column", columns, index=columns.index("Date") if "Date" in columns else 0, key="date_col_selectbox")
    close_col = st.sidebar.selectbox("Select the Close column", columns, index=columns.index("Close") if "Close" in columns else 0, key="close_col_selectbox")
    volume_col = st.sidebar.selectbox("Select the Volume column (optional)", ["None"] + columns, index=columns.index("Volume") if "Volume" in columns else 0, key="volume_col_selectbox")

# Main app logic
if st.button("Run Prediction", key="run_prediction_button"):
    with st.spinner("Fetching data and generating predictions..."):
        # Determine which symbol to use
        symbol = ticker
        if is_sri_lankan:
            symbol = f"{symbol}.N0000.CSE"  # Add suffix for EODHD

        # Calculate forecast days based on date range
        forecast_days = (end_date - start_date).days

        # Fetch data
        df = fetch_stock_data(symbol, start_date, end_date, uploaded_file, date_col, close_col, volume_col)
        if df is not None:
            # Plot historical data
            plot_historical_data(df, symbol)

            # Generate predictions
            forecast = predict_stock(df, forecast_days, model_choice)
            if forecast is not None:
                # Plot forecast
                plot_forecast(df, forecast, symbol)

                # Generate and display buy/sell signals
                signals = generate_signals(forecast)
                display_signals(signals)

                # Download prediction as CSV
                forecast_csv = forecast.to_csv(index=False)
                st.download_button(
                    label="Download Prediction as CSV",
                    data=forecast_csv,
                    file_name="prediction.csv",
                    mime="text/csv",
                    key="download_prediction_csv"
                )

                # Download historical data as CSV
                historical_csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Historical Data as CSV",
                    data=historical_csv,
                    file_name="historical_data.csv",
                    mime="text/csv",
                    key="download_historical_csv"
                )
