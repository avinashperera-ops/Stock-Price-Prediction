import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(stock_data, sentiment_data, seq_length=10):
    """Preprocess stock and sentiment data into sequences."""
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    stock_data.index = stock_data.index.date
    combined_data = stock_data.join(sentiment_data['Sentiment'], how='left').fillna(0)
    sentiment_values = combined_data['Sentiment'].values

    X, y = [], []
    for i in range(len(scaled_close) - seq_length):
        X.append(np.hstack((scaled_close[i:i + seq_length], 
                           sentiment_values[i:i + seq_length].reshape(-1, 1))))
        y.append(scaled_close[i + seq_length])
    
    return np.array(X), np.array(y), scaler