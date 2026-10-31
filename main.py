# main.py

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="TradUp - Quantitative Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Glassmorphism Styling Matching the UI Reference
st.markdown("""
    <style>
    .stApp { background-color: #08060F; color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    [data-testid="stSidebar"] { background-color: #0E0A1F; border-right: 1px solid rgba(255,255,255,0.05); }
    .card { background: #130F26; border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 16px; margin-bottom: 16px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #FFFFFF; margin-top: 5px; }
    .metric-title { font-size: 13px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        color: white; font-weight: 600; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #6D28D9 0%, #4338CA 100%); }
    </style>
""", unsafe_allow_html=True)

try:
    from src.data_fetch import fetch_stock_data
    from src.model import predict_stock, generate_signals
    from src.visualize import plot_dashboard_chart
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# Top Navigation Bar Simulation
col_top1, col_top2 = st.columns([3, 1])
with col_top1:
    st.markdown("## Dashboard")
with col_top2:
    st.markdown("<div style='text-align: right; color: #94A3B8; padding-top: 10px;'>Samuel Jr | @samjrw</div>", unsafe_allow_html=True)

st.markdown("<hr style='border: 0.5px solid rgba(255,255,255,0.05); margin-top: 0px;'>", unsafe_allow_html=True)

# Sidebar Menu Controls
st.sidebar.markdown("### TradUp")
st.sidebar.markdown("MENU")
nav_selection = st.sidebar.radio("Navigation", ["Dashboard", "Assets", "Market", "Portfolio", "Stacking Provider"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### Asset Query")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="AAPL").strip().upper()
is_sri_lankan = st.sidebar.checkbox("Sri Lankan CSE Stock", value=False)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)
run_analysis = st.sidebar.button("Run Live Analysis")

# Main Content Architecture
if run_analysis or ticker_input:
    symbol = f"{ticker_input}.N0000.CSE" if is_sri_lankan and not ticker_input.endswith(".CSE") else ticker_input
    
    with st.spinner(f"Querying live feeds and executing models for {symbol}..."):
        df = fetch_stock_data(symbol, period_years=1)
        
        if df is not None and not df.empty:
            forecast = predict_stock(df, forecast_horizon)
            signals = generate_signals(forecast)
            
            latest_price = df["Close"].iloc[-1]
            prev_price = df["Close"].iloc[-2] if len(df) > 1 else latest_price
            change_pct = ((latest_price - prev_price) / prev_price) * 100
            
            # Top 3 Stat Cards Row matching the UI reference
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Total Value</div>
                        <div class="metric-value">${latest_price:,.2f}</div>
                        <div style="color: {'#10B981' if change_pct >= 0 else '#EF4444'}; font-size: 12px; margin-top: 4px;">{change_pct:+.2f}% past session</div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Market Volume</div>
                        <div class="metric-value">{int(df['Volume'].iloc[-1]):,}</div>
                        <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">Latest Traded Volume</div>
                    </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-title">Model Horizon</div>
                        <div class="metric-value">{forecast_horizon} Days</div>
                        <div style="color: #38BDF8; font-size: 12px; margin-top: 4px;">Prophet Time-Series Engine</div>
                    </div>
                """, unsafe_allow_html=True)

            # Central Chart Layout Matching the Interface
            st.markdown('<div class="card">', unsafe_allow_html=True)
            plot_dashboard_chart(df, forecast, symbol)
            st.markdown('</div>', unsafe_allow_html=True)

            # Bottom Analytics Table
            st.markdown("### Automated Technical Signals")
            display_signals = signals.tail(5).copy()
            display_signals["Signal"] = display_signals["Signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
            st.dataframe(display_signals.rename(columns={"Date": "Timestamp", "Predicted_Price": "Projected Price"}), use_container_width=True)

            # File Downloads
            st.markdown("### Data Export Options")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("Download Forecast CSV", forecast.to_csv(index=False), "forecast.csv", "text/csv")
            with col_d2:
                st.download_button("Download Historical CSV", df.to_csv(index=False), "historical.csv", "text/csv")
        else:
            st.warning("Invalid ticker symbol or data source unavailable. Please verify your input.")
else:
    st.info("Configure your target ticker symbol in the sidebar and click **Run Live Analysis** to render the dashboard.")