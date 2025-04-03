# src/data_fetch.py
import yfinance as yf
import pandas as pd
import requests
import os
import streamlit as st
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLTK VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# API keys (using python-dotenv)
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
EOD_API_KEY = os.getenv("EOD_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Map tickers to company names for better news search
TICKER_TO_COMPANY = {
    "HNB": "Hatton National Bank",
    "COMB": "Commercial Bank of Ceylon",
    "SAMP": "Sampath Bank",
    "AAPL": "Apple Inc.",
    # Add more as needed
}

def fetch_stock_data(ticker, start_date, end_date, is_sri_lankan=False):
    """
    Fetch historical stock data from multiple sources.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., AAPL, HNB).
        start_date (datetime.date): Start date for historical data.
        end_date (datetime.date): End date for historical data.
        is_sri_lankan (bool): Whether the stock is from the Colombo Stock Exchange (CSE).
    
    Returns:
        pd.DataFrame: Historical stock data with columns: Open, High, Low, Close, Volume.
    
    Raises:
        ValueError: If no data is found from any source.
    """
    try:
        # Validate date range
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")
        if end_date > pd.Timestamp.today().date():
            raise ValueError("End date cannot be in the future.")

        # If the ticker is Sri Lankan, append the .CO suffix for yfinance
        original_ticker = ticker
        yf_ticker = f"{ticker}.CO" if is_sri_lankan else ticker
        
        # Try yfinance first for all tickers
        try:
            stock = yf.Ticker(yf_ticker)
            data = stock.history(start=start_date, end=end_date)
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                raise ValueError(f"yfinance returned no data for {yf_ticker}")
            return data
        except Exception as yf_error:
            st.warning(f"yfinance failed for {yf_ticker}: {str(yf_error)}")
            data = None

        # If yfinance fails, try Alpha Vantage for all tickers
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            if ALPHA_VANTAGE_API_KEY:
                try:
                    av_ticker = f"{original_ticker}.CSE" if is_sri_lankan else original_ticker
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": av_ticker,
                        "outputsize": "full",
                        "apikey": ALPHA_VANTAGE_API_KEY
                    }
                    response = requests.get("https://www.alphavantage.co/query", params=params)
                    response.raise_for_status()
                    av_data = response.json()

                    if "Time Series (Daily)" not in av_data:
                        raise ValueError(f"No data found for {av_ticker} on Alpha Vantage: {av_data.get('Note', 'No data returned')}")

                    time_series = av_data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(time_series, orient="index")
                    df.index = pd.to_datetime(df.index)
                    df = df[["1. open", "2. high", "3. low", "4. close", "5. volume"]]
                    df.columns = ["Open", "High", "Low", "Close", "Volume"]
                    df = df.astype(float)
                    df = df.loc[start_date:end_date]
                    if df.empty:
                        raise ValueError(f"No data in the specified date range for {av_ticker} on Alpha Vantage")
                    return df.sort_index()
                except Exception as av_error:
                    st.warning(f"Alpha Vantage failed for {av_ticker}: {str(av_error)}")

        # If Alpha Vantage fails, try Financial Modeling Prep for all tickers
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            if FMP_API_KEY:
                try:
                    fmp_ticker = f"{original_ticker}.N0000" if is_sri_lankan else original_ticker
                    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{fmp_ticker}?apikey={FMP_API_KEY}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}"
                    response = requests.get(url)
                    response.raise_for_status()
                    fmp_data = response.json()

                    if not fmp_data or "historical" not in fmp_data:
                        raise ValueError(f"No data found for {fmp_ticker} on Financial Modeling Prep")

                    historical_data = fmp_data["historical"]
                    if not historical_data:
                        raise ValueError(f"No historical data available for {fmp_ticker} on Financial Modeling Prep")

                    df = pd.DataFrame(historical_data)
                    df["Date"] = pd.to_datetime(df["date"])
                    df.set_index("Date", inplace=True)
                    df = df[["open", "high", "low", "close", "volume"]]
                    df.columns = ["Open", "High", "Low", "Close", "Volume"]
                    if df.empty:
                        raise ValueError(f"No data in the specified date range for {fmp_ticker} on Financial Modeling Prep")
                    return df.sort_index()
                except Exception as fmp_error:
                    st.warning(f"Financial Modeling Prep failed for {fmp_ticker}: {str(fmp_error)}")

        # If FMP fails, try EOD Historical Data for all tickers
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            if EOD_API_KEY:
                try:
                    eod_ticker = f"{original_ticker}.N0000.CSE" if is_sri_lankan else original_ticker
                    url = f"https://eodhistoricaldata.com/api/eod/{eod_ticker}?api_token={EOD_API_KEY}&fmt=json&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}"
                    response = requests.get(url)
                    response.raise_for_status()
                    eod_data = response.json()

                    if not eod_data:
                        raise ValueError(f"No data found for {eod_ticker} on EOD Historical Data")

                    df = pd.DataFrame(eod_data)
                    df["Date"] = pd.to_datetime(df["date"])
                    df.set_index("Date", inplace=True)
                    df = df[["open", "high", "low", "close", "volume"]]
                    df.columns = ["Open", "High", "Low", "Close", "Volume"]
                    if df.empty:
                        raise ValueError(f"No data in the specified date range for {eod_ticker} on EOD Historical Data")
                    return df.sort_index()
                except Exception as eod_error:
                    st.warning(f"EOD Historical Data failed for {eod_ticker}: {str(eod_error)}")

        # If all previous methods fail and the stock is Sri Lankan, scrape from CSE website
        if (data is None or (isinstance(data, pd.DataFrame) and data.empty)) and is_sri_lankan:
            try:
                url = f"https://www.cse.lk/api/companyDetails?symbol={original_ticker}.N0000"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                cse_data = response.json()

                if not cse_data or "historicalData" not in cse_data:
                    raise ValueError(f"No historical data found for {original_ticker} on CSE website")

                historical_data = cse_data["historicalData"]
                if not historical_data:
                    raise ValueError(f"No historical data available for {original_ticker} on CSE website")

                # Convert to DataFrame
                df = pd.DataFrame(historical_data)
                df["Date"] = pd.to_datetime(df["date"])
                df.set_index("Date", inplace=True)
                df = df[["open", "high", "low", "close", "volume"]]
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                df = df.astype(float)
                df = df.loc[start_date:end_date]
                if df.empty:
                    raise ValueError(f"No data in the specified date range for {original_ticker} on CSE website")
                return df.sort_index()
            except Exception as scrape_error:
                st.warning(f"CSE scraping failed for {original_ticker}: {str(scrape_error)}")

        # If all methods fail, raise an error
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            error_msg = f"No data found for {ticker} in the specified date range ({start_date} to {end_date})."
            if is_sri_lankan:
                error_msg += " Please ensure the ticker is correct (e.g., HNB for Hatton National Bank) or upload a CSV file with historical data."
            else:
                error_msg += " Please check the ticker or date range, or upload a CSV file with historical data."
            raise ValueError(error_msg)

        return data

    except Exception as e:
        raise Exception(f"Error fetching {ticker}: {str(e)}")

def fetch_sentiment_data(ticker, dates, is_sri_lankan=False):
    """
    Fetch sentiment data based on news articles using NLTK VADER.
    
    Args:
        ticker (str): Stock ticker symbol.
        dates (pd.Index): Dates for which to fetch sentiment data.
        is_sri_lankan (bool): Whether the stock is from the Colombo Stock Exchange (CSE).
    
    Returns:
        pd.DataFrame: Sentiment scores for the given dates.
    
    Raises:
        ValueError: If no news articles are found.
    """
    try:
        if not NEWSAPI_KEY:
            raise ValueError("NewsAPI key not found. Please set NEWSAPI_KEY in environment variables.")
        
        # Use the company name for better news search
        query = TICKER_TO_COMPANY.get(ticker.replace(".CO", ""), ticker)
        if is_sri_lankan:
            query = f"{query} Sri Lanka"
        
        # Increment request count using session state
        if "newsapi_request_count" not in st.session_state:
            st.session_state.newsapi_request_count = 0
        st.session_state.newsapi_request_count += 1

        # Fetch news articles
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get("articles", [])
        if not articles:
            raise ValueError(f"No news articles found for {query}")

        # Analyze sentiment of each article
        sentiment_scores = []
        for article in articles:
            text = article.get("title", "") + " " + article.get("description", "")
            if text:
                scores = sia.polarity_scores(text)
                sentiment_scores.append(scores["compound"])  # Use compound score
            else:
                sentiment_scores.append(0.0)  # Neutral if no text

        # Average sentiment score
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        # Create a DataFrame with the same sentiment score for all dates
        sentiment_df = pd.DataFrame({"Sentiment": [avg_sentiment] * len(dates)}, index=dates)
        return sentiment_df
    except Exception as e:
        raise Exception(f"Error fetching news for {ticker}: {str(e)}")
