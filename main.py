# main.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import nltk

# Download NLTK data for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from src.data_fetch import fetch_stock_data, fetch_sentiment_data
from src.model import train_model, predict_stock, generate_signals

# Rest of the code remains the same...
st.title("Stock Price Prediction with Buy/Sell Signals")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Ticker input
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, HNB)", "AAPL").strip().upper()
is_sri_lankan = st.sidebar.checkbox("Sri Lankan Stock (e.g., HNB, COMB)", False)

# Date range
st.sidebar.subheader("Date Range")
end_date = datetime.today().date()
start_date = end_date - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", start_date, max_value=end_date - timedelta(days=1))
end_date = st.sidebar.date_input("End Date", end_date, max_value=datetime.today().date())

# Model selection
st.sidebar.subheader("Model")
model_type = st.sidebar.selectbox("Choose Model", ["LSTM", "GRU", "Moving Average"])
signal_type = st.sidebar.selectbox("Signal Type", ["Moving Average", "MinMax"])

# Refresh data button
refresh_data = st.sidebar.button("Refresh Data")

# File upload for manual data
st.sidebar.subheader("Load Predictions (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Drag and drop file here",
    type=["csv"],
    help="Limit 200MB per file • CSV. Required columns: Date, Open, High, Low, Close, Volume."
)

# Main content
stock_data = None
if refresh_data or uploaded_file is None:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Fetch stock data
            stock_data = fetch_stock_data(ticker, start_date, end_date, is_sri_lankan)
        except Exception as e:
            st.error(f"Error with {ticker}: {str(e)}")
            if is_sri_lankan:
                st.warning(
                    "Automated data fetching failed for Sri Lankan stock. Please upload historical data manually in CSV format. "
                    "The CSV should have columns: Date, Open, High, Low, Close, Volume. "
                    "You can download historical data for your ticker from the CSE website: "
                    f"https://www.cse.lk/pages/company-profile/company-profile.component.html?symbol={ticker}.N0000"
                )
            else:
                st.warning(
                    "Automated data fetching failed. Please check the ticker or date range, or upload historical data manually in CSV format. "
                    "The CSV should have columns: Date, Open, High, Low, Close, Volume."
                )

# Load data from uploaded file if provided
if uploaded_file is not None:
    with st.spinner("Loading uploaded file..."):
        try:
            stock_data = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in stock_data.columns for col in required_columns):
                st.error("Uploaded CSV must contain columns: Date, Open, High, Low, Close, Volume")
                stock_data = None
            else:
                stock_data = stock_data.loc[start_date:end_date]
                if stock_data.empty:
                    st.error("No data in the specified date range in the uploaded CSV")
                    stock_data = None
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            stock_data = None

# Run prediction if data is available
if stock_data is not None and not stock_data.empty:
    try:
        # Fetch sentiment data
        with st.spinner("Fetching sentiment data..."):
            sentiment_data = fetch_sentiment_data(ticker, stock_data.index, is_sri_lankan)

        # Train model and predict
        with st.spinner("Training model and generating predictions..."):
            model, X_test, y_test, scaler = train_model(stock_data, model_type)
            predictions = predict_stock(model, stock_data, scaler, model_type)
        
        # Generate buy/sell signals
        with st.spinner("Generating buy/sell signals..."):
            signals = generate_signals(stock_data, predictions, signal_type)

        # Display results
        st.subheader(f"Stock Data for {ticker}")
        st.write(stock_data.tail())

        st.subheader("Predictions")
        st.line_chart(predictions)

        st.subheader("Buy/Sell Signals")
        st.write(signals.tail())

        # Download predictions
        csv = predictions.to_csv()
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
else:
    if stock_data is None and not refresh_data and uploaded_file is None:
        st.info("Please click 'Refresh Data' to fetch stock data or upload a CSV file.")

# Run prediction button
if st.button("Run Prediction"):
    if stock_data is not None and not stock_data.empty:
        try:
            # Fetch sentiment data
            with st.spinner("Fetching sentiment data..."):
                sentiment_data = fetch_sentiment_data(ticker, stock_data.index, is_sri_lankan)

            # Train model and predict
            with st.spinner("Training model and generating predictions..."):
                model, X_test, y_test, scaler = train_model(stock_data, model_type)
                predictions = predict_stock(model, stock_data, scaler, model_type)
            
            # Generate buy/sell signals
            with st.spinner("Generating buy/sell signals..."):
                signals = generate_signals(stock_data, predictions, signal_type)

            # Display results
            st.subheader(f"Stock Data for {ticker}")
            st.write(stock_data.tail())

            st.subheader("Predictions")
            st.line_chart(predictions)

            st.subheader("Buy/Sell Signals")
            st.write(signals.tail())

            # Download predictions
            csv = predictions.to_csv()
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("No data available to run predictions. Please fetch data or upload a CSV file.")
