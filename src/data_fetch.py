import yfinance as yf
import pandas as pd
import requests
import os

# API keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY")

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
        if is_sri_lankan:
            ticker = f"{ticker}.CO"
        
        # Try yfinance first
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            # If yfinance fails and it's a Sri Lankan ticker, try MarketStack
            if is_sri_lankan and MARKETSTACK_API_KEY:
                # Calculate date range in days
                days = (end_date - start_date).days
                params = {
                    "access_key": MARKETSTACK_API_KEY,
                    "symbols": ticker.replace(".CO", ""),  # MarketStack uses the base ticker (e.g., HNB)
                    "exchange": "CSE",  # Colombo Stock Exchange
                    "date_from": start_date.strftime("%Y-%m-%d"),
                    "date_to": end_date.strftime("%Y-%m-%d"),
                    "limit": 1000
                }
                response = requests.get("http://api.marketstack.com/v1/eod", params=params)
                response.raise_for_status()
                market_data = response.json()

                if "data" not in market_data or not market_data["data"]:
                    raise ValueError(f"No data found for {ticker} on MarketStack")

                # Convert MarketStack data to a pandas DataFrame
                df = pd.DataFrame(market_data["data"])
                df["Date"] = pd.to_datetime(df["date"])
                df.set_index("Date", inplace=True)
                df = df[["open", "high", "low", "close", "volume"]]
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                return df.sort_index()
            else:
                raise ValueError(f"No data found for {ticker}. If this is a Sri Lankan stock, ensure the ticker is correct (e.g., HNB for Hatton National Bank).")
        return data
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
        
        # Increment request count
        request_file = "request_count.txt"
        if os.path.exists(request_file):
            with open(request_file, 'r') as f:
                count = int(f.read().strip())
        else:
            count = 0
        count += 1
        with open(request_file, 'w') as f:
            f.write(str(count))

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get("articles", [])
        if not articles:
            raise ValueError(f"No news articles found for {query}")

        # Simplified sentiment calculation (placeholder)
        # In a real app, you'd use a sentiment analysis library like NLTK or TextBlob
        sentiment_scores = [0.1 * len(articles)] * len(dates)
        return pd.DataFrame({"Sentiment": sentiment_scores}, index=dates)
    except Exception as e:
        raise Exception(f"Error fetching news for {ticker}: {str(e)}")
