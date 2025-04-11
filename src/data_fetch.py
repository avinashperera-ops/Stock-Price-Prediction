# data_fetch.py

import streamlit as st
import yfinance as yf
import requests
import pandas as pd

# EODHD API key
EODHD_API_KEY = "your_eodhd_api_key_here"

@st.cache_data
def fetch_stock_data(symbol, start, end, uploaded_file=None, date_col=None, close_col=None, volume_col=None):
    try:
        # If a CSV file is uploaded, use it
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            required_columns = [date_col, close_col]
            if not all(col in df.columns for col in required_columns):
                st.error(f"The CSV must contain the selected columns: {required_columns}")
                return None
            
            # Rename columns to standard names
            df = df.rename(columns={date_col: "Date", close_col: "Close"})
            if volume_col != "None":
                df = df.rename(columns={volume_col: "Volume"})
            else:
                df["Volume"] = 0  # Add a dummy Volume column if not provided
            
            df["Date"] = pd.to_datetime(df["Date"])
            df["Close"] = df["Close"].astype(float)
            df["Volume"] = df["Volume"].astype(int)
            df = df[["Date", "Close", "Volume"]]
            df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
            return df

        # Use EODHD for Sri Lankan stocks (ending with .N0000.CSE)
        if symbol.endswith(".N0000.CSE"):
            url = f"https://eodhistoricaldata.com/api/eod/{symbol}?api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                st.error(f"No data found for {symbol} on EODHD. Please check the symbol or upload a CSV file.")
                return None
            
            # Convert EODHD data to DataFrame
            df = pd.DataFrame(data)
            df["Date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"close": "Close", "volume": "Volume"})
            df = df[["Date", "Close", "Volume"]]
            df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
            return df

        # Use Yahoo Finance for global stocks
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end)
        if df.empty:
            st.error(f"No data found for {symbol} on Yahoo Finance. Please check the symbol (e.g., AAPL for Apple) or upload a CSV file.")
            return None
        df.reset_index(inplace=True)
        # Ensure the Date column is timezone-naive
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        return df[["Date", "Close", "Volume"]]
    
    except Exception as e:
        st.error(f"Error fetching data: {e}. Please try another symbol, upload a CSV file, or try again later.")
        return None

def fetch_sentiment_data():
    # Placeholder for sentiment data fetching (if needed)
    pass
