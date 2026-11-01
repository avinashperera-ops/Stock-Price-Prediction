# src/data_fetch.py

import streamlit as st
import yfinance as yf
import requests
import pandas as pd

EODHD_API_KEY = "67ed03b7d053d7.37731358"

@st.cache_data(ttl=1800)
def fetch_stock_data(symbol, period_years=1):
    try:
        clean_symbol = symbol.strip().upper()

        # Handle Sri Lankan CSE Tickers via EODHD (.LK exchange suffix)
        if ".CSE" in clean_symbol or ".LK" in clean_symbol or clean_symbol.endswith(".N0000") or (len(clean_symbol) <= 6 and clean_symbol not in ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"]):
            base_ticker = clean_symbol.split(".")[0]
            eodhd_symbol = f"{base_ticker}.LK"
            
            url = f"https://eodhistoricaldata.com/api/eod/{eodhd_symbol}?api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df["Date"] = pd.to_datetime(df["date"])
                    df = df.rename(columns={"close": "Close", "volume": "Volume"})
                    return df[["Date", "Close", "Volume"]].sort_values("Date").dropna()

        # Handle Global Tickers via Yahoo Finance
        stock = yf.Ticker(clean_symbol)
        df = stock.history(period=f"{period_years}y")
        
        if df.empty:
            return None
            
        df.reset_index(inplace=True)
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
            
        return df[["Date", "Close", "Volume"]].sort_values("Date").dropna()
    except Exception as e:
        return None

@st.cache_data(ttl=1800)
def fetch_live_watchlist(symbols=["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]):
    watchlist_data = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if not hist.empty:
                latest = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else latest
                change = ((latest - prev) / prev) * 100
                watchlist_data.append({
                    "Symbol": sym,
                    "Price": latest,
                    "Change": change
                })
        except Exception:
            continue
    return pd.DataFrame(watchlist_data)