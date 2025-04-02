import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data with custom date range."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        raise ValueError(f"Error fetching {ticker}: {str(e)}")

def fetch_sentiment_data(ticker, dates):
    """Fetch news headlines from NewsAPI and calculate sentiment."""
    api_key = os.getenv("NEWSAPI_KEY")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Track API requests
    request_file = "request_count.txt"
    if os.path.exists(request_file):
        with open(request_file, 'r') as f:
            count = int(f.read().strip())
    else:
        count = 0
    
    if count >= 100:
        raise ValueError("NewsAPI request limit (100/day) reached.")
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={start_date}&to={end_date}&language=en&apiKey={api_key}"
    response = requests.get(url)
    with open(request_file, 'w') as f:
        f.write(str(count + 1))
    
    news_data = response.json()
    if news_data['status'] != 'ok':
        print("Error fetching news:", news_data.get('message', 'Unknown error'))
        return pd.DataFrame({'Date': dates, 'Sentiment': [0] * len(dates)})

    articles = news_data.get('articles', [])
    sentiment_data = []
    for article in articles:
        date = pd.to_datetime(article['publishedAt']).date()
        title = article.get('title', '')
        sentiment_data.append({'Date': date, 'Text': title})

    sentiment_df = pd.DataFrame(sentiment_data)
    if sentiment_df.empty:
        return pd.DataFrame({'Date': dates, 'Sentiment': [0] * len(dates)})

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()
    
    sentiment_df['Sentiment'] = sentiment_df['Text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    sentiment_df = sentiment_df.groupby('Date')['Sentiment'].mean().reset_index()
    sentiment_df.set_index('Date', inplace=True)
    return sentiment_df