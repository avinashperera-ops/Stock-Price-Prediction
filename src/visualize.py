# src/visualize.py

import streamlit as st
import plotly.graph_objs as go

def plot_dashboard_chart(df, forecast, symbol):
    fig = go.Figure()

    # Historical line (Neon Purple/Blue Theme matching the UI reference)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], 
        mode="lines", name="Historical",
        line=dict(color="#8B5CF6", width=2.5)
    ))

    # AI Forecast line
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], 
        mode="lines", name="AI Forecast",
        line=dict(color="#38BDF8", width=2.5, dash="dot")
    ))

    # Upper/Lower Bounds confidence area
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", name="Upper Band",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        mode="lines", name="Lower Band",
        fill="tonexty", fillcolor="rgba(139, 92, 246, 0.1)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"
    ))

    fig.update_layout(
        title=dict(text=f"Market Intelligence: {symbol}", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(showgrid=False, color="#94A3B8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94A3B8"),
        template="plotly_dark",
        paper_bgcolor="#0A0814",
        plot_bgcolor="#0A0814",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)