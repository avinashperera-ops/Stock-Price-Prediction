import pandas as pd
from datetime import datetime

def save_predictions(ticker, future_dates, predictions, buy_signals, sell_signals):
    """Save predictions and signals to CSV."""
    df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions.flatten(),
        'Signal': ['None'] * len(future_dates)
    })
    for idx in buy_signals:
        df.loc[idx, 'Signal'] = 'Buy'
    for idx in sell_signals:
        df.loc[idx, 'Signal'] = 'Sell'
    df.to_csv(f"{ticker}_predictions.csv", index=False)
    return f"{ticker}_predictions.csv"

def load_predictions(file_path):
    """Load predictions from CSV."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df