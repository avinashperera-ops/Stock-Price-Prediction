import yfinance as yf
import pandas as pd
import requests
import os
import streamlit as st

# API keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Map tickers to company names for better news search
TICKER_TO_COMPANY = {
    "HNB": "Hatton National Bank",
    "COMB": "Commercial Bank of Ceylon",
    "SAMP": "Sampath Bank",
    # Add more as needed
}

def fetch_stock_data(ticker, start_date, end_date, is_sri_lankan=False):
    try:
        # If the ticker is Sri Lankan, append the .CO suffix for yfinance
        original_ticker = ticker
        if is_sri_lankan:
            ticker = f"{ticker}.CO"
        
        # Try yfinance first
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                raise ValueError(f"yfinance returned no data for {ticker}")
        except Exception as yf_error:
            st.warning(f"yfinance failed for {ticker}: {str(yf_error)}")
            data = None

        # If yfinance fails, try MarketStack for Sri Lankan tickers
        if (data is None or (isinstance(data, pd.DataFrame) and data.empty)) and is_sri_lankan:
            if MARKETSTACK_API_KEY:
                try:
                    # Use the .N0000 suffix for MarketStack (CSE format)
                    marketstack_ticker = f"{original_ticker}.N0000"
                    days = (end_date - start_date).days
                    params = {
                        "access_key": MARKETSTACK_API_KEY,
                        "symbols": marketstack_ticker,
                        "exchange": "CSE",
                        "date_from": start_date.strftime("%Y-%m-%d"),
                        "date_to": end_date.strftime("%Y-%m-%d"),
                        "limit": 1000
                    }
                    response = requests.get("http://api.marketstack.com/v1/eod", params=params)
                    response.raise_for_status()
                    market_data = response.json()

                    if "data" not in market_data or not market_data["data"]:
                        raise ValueError(f"No data found for {marketstack_ticker} on MarketStack")

                    df = pd.DataFrame(market_data["data"])
                    df["Date"] = pd.to_datetime(df["date"])
                    df.set_index("Date", inplace=True)
                    df = df[["open", "high", "low", "close", "volume"]]
                    df.columns = ["Open", "High", "Low", "Close", "Volume"]
                    return df.sort_index()
                except Exception as ms_error:
                    st.warning(f"MarketStack failed for {marketstack_ticker}: {str(ms_error)}")

            # If MarketStack fails, try Alpha Vantage
            if ALPHA_VANTAGE_API_KEY:
                try:
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": f"{original_ticker}.CSE",
                        "outputsize": "full",
                        "apikey": ALPHA_VANTAGE_API_KEY
                    }
                    response = requests.get("https://www.alphavantage.co/query", params=params)
                    response.raise_for_status()
                    av_data = response.json()

                    if "Time Series (Daily)" not in av_data:
                        raise ValueError(f"No data found for {original_ticker} on Alpha Vantage")

                    time_series = av_data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(time_series, orient="index")
                    df.index = pd.to_datetime(df.index)
                    df = df[["1. open", "2. high", "3. low", "4. close", "5. volume"]]
                    df.columns = ["Open", "High", "Low", "Close", "Volume"]
                    df = df.astype(float)
                    df = df.loc[start_date:end_date]
                    if df.empty:
                        raise ValueError(f"No data in the specified date range for {original_ticker} on Alpha Vantage")
                    return df.sort_index()
                except Exception as av_error:
                    raise ValueError(f"Alpha Vantage failed for {original_ticker}: {str(av_error)}")
            else:
                raise ValueError("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in environment variables.")

        # If yfinance succeeded, return the data
        if data is not None and not data.empty:
            return data
        else:
            raise ValueError(f"No data found for {ticker}. If this is a Sri Lankan stock, ensure the ticker is correct (e.g., HNB for Hatton National Bank).")
    except Exception as e:
        raise Exception(f"Error fetching {ticker}: {str(e)}")

def fetch_sentiment_data(ticker, dates, is_sri_lankan=False):
    try:
        if not NEWSAPI_KEY:
            raise ValueError("NewsAPI key not found. Please set NEWSAPI_KEY in environment variables.")
        
        # Use the company name for Sri Lankan stocks, if available
        query = TICKER_TO_COMPANY.get(ticker.replace(".CO", ""), ticker)
        if is_sri_lankan:
            query = f"{query} Sri Lanka"
        
        # Increment request count using session state
        if "newsapi_request_count" not in st.session_state:
            st.session_state.newsapi_request_count = 0
        st.session_state.newsapi_request_count += 1

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get("articles", [])
        if not articles:
            raise ValueError(f"No news articles found for {query}")

        # Simplified sentiment calculation (placeholder)
        sentiment_scores = [0.1 * len(articles)] * len(dates)
        return pd.DataFrame({"Sentiment": sentiment_scores}, index=dates)
    except Exception as e:
        raise Exception(f"Error fetching news for {ticker}: {str(e)}")
