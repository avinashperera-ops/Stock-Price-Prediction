import streamlit as st
import plotly.graph_objects as go
from src.data_fetch import fetch_stock_data, fetch_sentiment_data
from src.preprocess import preprocess_data
from src.model import build_and_train_model, predict_future, generate_signals
from src.utils import save_predictions, load_predictions
from datetime import datetime, timedelta
import pandas as pd
import os

# Sidebar Help Section
st.sidebar.title("Help")
st.sidebar.markdown("""
- **Ticker**: Enter stock symbols (e.g., "AAPL, TSLA").
- **Date Range**: Select historical data period.
- **Model**: Choose prediction method (LSTM, GRU, Moving Average).
- **Signals**: Pick buy/sell logic (MinMax or Moving Average).
- **Save/Load**: Export or import predictions.
- **Refresh**: Update data manually.
""")

# Main UI
st.title("Stock Price Prediction with Buy/Sell Signals")
st.write("Predict stock prices and get buy/sell recommendations with sentiment analysis.")

# User Inputs
tickers = st.text_input("Stock Tickers (comma-separated, e.g., AAPL, TSLA)", "AAPL")
start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.now())
model_type = st.selectbox("Prediction Model", ["LSTM", "GRU", "Moving Average"], index=0)
epochs = st.slider("Training Epochs", 5, 50, 20)
future_days = st.slider("Days to Predict", 1, 30, 7)
signal_type = st.selectbox("Signal Type", ["MinMax", "Moving Average"], index=0)
run_button = st.button("Run Prediction")
refresh_button = st.button("Refresh Data")
load_file = st.file_uploader("Load Predictions (CSV)", type="csv")

# API Request Warning
request_file = "request_count.txt"
if os.path.exists(request_file):
    with open(request_file, 'r') as f:
        count = int(f.read().strip())
    st.write(f"NewsAPI Requests Used Today: {count}/100")

# Process Predictions
if run_button or refresh_button:
    tickers_list = [t.strip() for t in tickers.split(",")]
    for ticker in tickers_list:
        with st.spinner(f"Processing {ticker}..."):
            try:
                # Fetch and preprocess data
                stock_data = fetch_stock_data(ticker, start_date, end_date)
                sentiment_data = fetch_sentiment_data(ticker, stock_data.index)
                X, y, scaler = preprocess_data(stock_data, sentiment_data, seq_length=10)

                # Train and evaluate model
                model, X_test, y_test, test_pred, mse = build_and_train_model(X, y, model_type, epochs)
                st.write(f"{ticker} - Test MSE: {mse:.2f}")

                # Future predictions
                last_sequence = X[-1]
                future_predictions, variances = predict_future(model, last_sequence, scaler, future_days, model_type)
                buy_signals, sell_signals = generate_signals(future_predictions, signal_type)

                # Plot predictions
                future_dates = [datetime.now().date() + timedelta(days=i) for i in range(future_days)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), 
                                       mode='lines', name='Predicted Price'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten() + variances, 
                                       mode='lines', line=dict(dash='dash'), name='Upper Bound', opacity=0.3))
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten() - variances, 
                                       mode='lines', line=dict(dash='dash'), name='Lower Bound', opacity=0.3))
                fig.add_trace(go.Scatter(x=[future_dates[i] for i in buy_signals], 
                                       y=[future_predictions[i][0] for i in buy_signals], 
                                       mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), 
                                       name='Buy'))
                fig.add_trace(go.Scatter(x=[future_dates[i] for i in sell_signals], 
                                       y=[future_predictions[i][0] for i in sell_signals], 
                                       mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), 
                                       name='Sell'))
                fig.update_layout(title=f'{ticker} Price Prediction', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

                # Sentiment plot
                if st.checkbox(f"Show Sentiment Data for {ticker}"):
                    fig_sent = go.Figure()
                    fig_sent.add_trace(go.Scatter(x=sentiment_data.index, y=sentiment_data['Sentiment'], 
                                                mode='lines+markers', name='Sentiment'))
                    fig_sent.update_layout(title=f'{ticker} Sentiment Trend', xaxis_title='Date', yaxis_title='Sentiment')
                    st.plotly_chart(fig_sent)

                # Buy/Sell Recommendations with Exact Dates
                st.subheader(f"{ticker} Buy/Sell Recommendations")
                recommendations = []
                total_profit = 0
                trades = 0
                
                if buy_signals:
                    st.write("**Buy Signals:**")
                    for buy_idx in buy_signals:
                        buy_date = future_dates[buy_idx]
                        buy_price = future_predictions[buy_idx][0]
                        st.write(f"- Buy on {buy_date} at ${buy_price:.2f}")
                        recommendations.append(f"Buy on {buy_date} at ${buy_price:.2f}")
                
                if sell_signals:
                    st.write("**Sell Signals:**")
                    for sell_idx in sell_signals:
                        sell_date = future_dates[sell_idx]
                        sell_price = future_predictions[sell_idx][0]
                        st.write(f"- Sell on {sell_date} at ${sell_price:.2f}")
                        recommendations.append(f"Sell on {sell_date} at ${sell_price:.2f}")
                
                if buy_signals and sell_signals:
                    st.write("**Profit/Loss Estimates:**")
                    for buy_idx in buy_signals:
                        next_sell = next((s for s in sell_signals if s > buy_idx), None)
                        if next_sell:
                            buy_date = future_dates[buy_idx]
                            sell_date = future_dates[next_sell]
                            buy_price = future_predictions[buy_idx][0]
                            sell_price = future_predictions[next_sell][0]
                            profit = sell_price - buy_price
                            total_profit += profit
                            trades += 1
                            st.write(f"- Buy on {buy_date} (${buy_price:.2f}), Sell on {sell_date} (${sell_price:.2f}), Profit: ${profit:.2f}")
                            recommendations.append(f"Trade: Buy {buy_date} (${buy_price:.2f}), Sell {sell_date} (${sell_price:.2f}), Profit: ${profit:.2f}")
                
                if trades > 0:
                    avg_profit = total_profit / trades
                    st.write(f"**Summary:** Total Profit: ${total_profit:.2f}, Average Profit/Trade: ${avg_profit:.2f}")
                else:
                    st.write("No complete buy/sell pairs for profit calculation.")

                # Save predictions
                if st.button(f"Save {ticker} Predictions"):
                    file_path = save_predictions(ticker, future_dates, future_predictions, buy_signals, sell_signals)
                    st.write(f"Saved to {file_path}")

            except Exception as e:
                st.error(f"Error with {ticker}: {str(e)}")

# Load Predictions
if load_file:
    df = load_predictions(load_file)
    st.write("Loaded Predictions:")
    st.dataframe(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_Price'], mode='lines', name='Predicted Price'))
    fig.add_trace(go.Scatter(x=df[df['Signal'] == 'Buy']['Date'], y=df[df['Signal'] == 'Buy']['Predicted_Price'], 
                           mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
    fig.add_trace(go.Scatter(x=df[df['Signal'] == 'Sell']['Date'], y=df[df['Signal'] == 'Sell']['Predicted_Price'], 
                           mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))
    fig.update_layout(title='Loaded Prediction', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)