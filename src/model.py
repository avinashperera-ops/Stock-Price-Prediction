import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error
import pandas as pd

def build_and_train_model(X, y, model_type="LSTM", epochs=20, batch_size=32):
    """Build and train a model based on user selection."""
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if model_type == "LSTM" or model_type == "GRU":
        model = Sequential()
        layer = LSTM if model_type == "LSTM" else GRU
        model.add(layer(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
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
        return model, X_test, y_test, y_pred[-len(X_test):]

    test_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, test_pred)
    return model, X_test, y_test, test_pred, mse

def predict_future(model, last_sequence, scaler, future_days=7, model_type="LSTM"):
    """Predict future prices with confidence intervals."""
    if model_type == "Moving Average":
        last_values = scaler.inverse_transform(last_sequence[:, 0].reshape(-1, 1))
        ma = pd.Series(last_values.flatten()).rolling(window=10).mean().iloc[-1]
        return np.full((future_days, 1), ma), np.zeros(future_days)  # No confidence for MA
    
    predictions = []
    variances = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)
        predictions.append(pred[0, 0])
        variances.append(0.05)  # Simplified variance (could use Monte Carlo)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred[0, 0]
        current_sequence[-1, 1] = 0

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    variances = np.array(variances) * predictions.std()  # Rough confidence
    return predictions, variances

def generate_signals(predictions, signal_type="MinMax"):
    """Generate buy/sell signals based on user-selected method."""
    if signal_type == "MinMax":
        buy_signals, sell_signals = [], []
        for i in range(1, len(predictions) - 1):
            if predictions[i] < predictions[i-1] and predictions[i] < predictions[i+1]:
                buy_signals.append(i)
            elif predictions[i] > predictions[i-1] and predictions[i] > predictions[i+1]:
                sell_signals.append(i)
    elif signal_type == "Moving Average":
        short_ma = pd.Series(predictions.flatten()).rolling(window=3).mean().fillna(predictions[0])
        long_ma = pd.Series(predictions.flatten()).rolling(window=10).mean().fillna(predictions[0])
        buy_signals = np.where((short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)))[0].tolist()
        sell_signals = np.where((short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)))[0].tolist()
    
    return buy_signals, sell_signals