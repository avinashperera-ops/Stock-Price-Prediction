# src/model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error

def train_model(stock_data, model_type="LSTM", epochs=20, batch_size=32):
    """
    Build and train a model based on user selection.
    
    Args:
        stock_data (pd.DataFrame): Historical stock data with columns: Open, High, Low, Close, Volume.
        model_type (str): Type of model ("LSTM", "GRU", or "Moving Average").
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
    
    Returns:
        tuple: (model, X_test, y_test, scaler) where model is the trained model (or None for Moving Average),
               X_test and y_test are the test data, and scaler is the MinMaxScaler used.
    """
    # Use the 'Close' price for prediction
    data = stock_data[["Close"]].values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data (use 60 days to predict the next day)
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM/GRU [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into training and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if model_type == "LSTM" or model_type == "GRU":
        model = Sequential()
        layer = LSTM if model_type == "LSTM" else GRU
        model.add(layer(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(layer(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  validation_data=(X_test, y_test), verbose=1)
    elif model_type == "Moving Average":
        # Simple moving average baseline
        y_pred = pd.Series(y_train).rolling(window=10).mean().fillna(y_train[0]).values
        model = None  # No trainable model
        return model, X_test, y_test, scaler

    return model, X_test, y_test, scaler

def predict_stock(model, stock_data, scaler, model_type="LSTM"):
    """
    Predict future stock prices using the trained model.
    
    Args:
        model: Trained Keras model (or None for Moving Average).
        stock_data (pd.DataFrame): Historical stock data with columns: Open, High, Low, Close, Volume.
        scaler: MinMaxScaler used to scale the data.
        model_type (str): Type of model ("LSTM", "GRU", or "Moving Average").
    
    Returns:
        pd.DataFrame: Predicted stock prices with column "Predicted_Close".
    """
    # Use the 'Close' price for prediction
    data = stock_data[["Close"]].values
    scaled_data = scaler.transform(data)

    # Prepare input for prediction (last 60 days)
    look_back = 60
    last_sequence = scaled_data[-look_back:]
    last_sequence = np.reshape(last_sequence, (look_back, 1))

    if model_type == "Moving Average":
        last_values = scaler.inverse_transform(last_sequence)
        ma = pd.Series(last_values.flatten()).rolling(window=10).mean().iloc[-1]
        prediction_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=7)
        predictions = pd.DataFrame(np.full((7, 1), ma), index=prediction_dates, columns=["Predicted_Close"])
        return predictions
    
    # Predict 7 days into the future
    future_days = 7
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)
        predictions.append(pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred[0, 0]

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create a DataFrame for predictions
    prediction_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    predictions_df = pd.DataFrame(predictions, index=prediction_dates, columns=["Predicted_Close"])
    
    return predictions_df

def generate_signals(stock_data, predictions, signal_type="MinMax"):
    """
    Generate buy/sell signals based on predictions.
    
    Args:
        stock_data (pd.DataFrame): Historical stock data with columns: Open, High, Low, Close, Volume.
        predictions (pd.DataFrame): Predicted stock prices with column "Predicted_Close".
        signal_type (str): Type of signal to generate ("Moving Average" or "MinMax").
    
    Returns:
        pd.DataFrame: DataFrame with a "Signal" column (1 for Buy, -1 for Sell, 0 for Hold).
    """
    # Combine historical and predicted data
    combined_data = pd.DataFrame(index=stock_data.index.append(predictions.index))
    combined_data["Close"] = stock_data["Close"]
    combined_data["Predicted_Close"] = predictions["Predicted_Close"]
    combined_data["Close"] = combined_data["Close"].fillna(combined_data["Predicted_Close"])
    
    signals = pd.DataFrame(index=combined_data.index, columns=["Signal"])
    signals["Signal"] = 0  # Default to Hold

    if signal_type == "MinMax":
        for i in range(1, len(combined_data) - 1):
            if (combined_data["Predicted_Close"].iloc[i] < combined_data["Predicted_Close"].iloc[i-1] and
                combined_data["Predicted_Close"].iloc[i] < combined_data["Predicted_Close"].iloc[i+1]):
                signals["Signal"].iloc[i] = 1  # Buy
            elif (combined_data["Predicted_Close"].iloc[i] > combined_data["Predicted_Close"].iloc[i-1] and
                  combined_data["Predicted_Close"].iloc[i] > combined_data["Predicted_Close"].iloc[i+1]):
                signals["Signal"].iloc[i] = -1  # Sell
    elif signal_type == "Moving Average":
        short_ma = combined_data["Predicted_Close"].rolling(window=3).mean().fillna(combined_data["Predicted_Close"].iloc[0])
        long_ma = combined_data["Predicted_Close"].rolling(window=10).mean().fillna(combined_data["Predicted_Close"].iloc[0])
        signals["Signal"] = 0
        signals.loc[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)), "Signal"] = 1  # Buy
        signals.loc[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)), "Signal"] = -1  # Sell
    
    return signals
