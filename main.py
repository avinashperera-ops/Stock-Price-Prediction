import yfinance as yf
import pandas as pd
import requests
import os

# MarketStack API key (optional, for fallback)
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY")

def fetch_stock_data(ticker, period="1y", is_sri_lankan=False):
    try:
        # If the ticker is Sri Lankan, append the .CO suffix for yfinance
        if is_sri_lankan:
            ticker = f"{ticker}.CO"
        
        # Try yfinance first
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            # If yfinance fails and it's a Sri Lankan ticker, try MarketStack
            if is_sri_lankan and MARKETSTACK_API_KEY:
                # Map period to MarketStack date range
                period_mapping = {
                    "1mo": 30,
                    "3mo": 90,
                    "6mo": 180,
                    "1y": 365,
                    "2y": 730
                }
                days = period_mapping.get(period, 365)

                # MarketStack API request
                params = {
                    "access_key": MARKETSTACK_API_KEY,
                    "symbols": ticker.replace(".CO", ""),  # MarketStack uses the base ticker (e.g., HNB)
                    "exchange": "CSE",  # Colombo Stock Exchange
                    "date_from": (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                    "date_to": pd.Timestamp.today().strftime("%Y-%m-%d"),
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
