# src/data_fetch.py

import streamlit as st
import yfinance as yf
import requests
import pandas as pd

EODHD_API_KEY = "67ed03b7d053d7.37731358"

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period_years=1):
    try:
        # Handle Sri Lankan CSE Tickers
        if ".CSE" in symbol.upper() or symbol.endswith(".N0000"):
            if not symbol.endswith(".CSE"):
                symbol = f"{symbol}.CSE"
            url = f"https://eodhistoricaldata.com/api/eod/{symbol}?api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data or isinstance(data, dict) and 'error' in data:
                return None
            
            df = pd.DataFrame(data)
            df["Date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"close": "Close", "volume": "Volume"})
            return df[["Date", "Close", "Volume"]].sort_values("Date").dropna()

        # Handle Global Tickers via Yahoo Finance
        ticker_symbol = symbol.strip().upper()
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=f"{period_years}y")
        
        if df.empty:
            return None
            
        df.reset_index(inplace=True)
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
            
        return df[["Date", "Close", "Volume"]].sort_values("Date").dropna()
    except Exception:
        return None