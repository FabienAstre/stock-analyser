import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# App Title
st.title("📈 Smart Stock Analyzer with Signals")

# User input for ticker
ticker = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA, NVDA)", value="AAPL")

# Load data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    return df

# Trigger signal logic
def trigger_signal(df):
    df = df.dropna(subset=["SMA20", "SMA50"])
    if df.empty:
        return "⚠️ Not enough data for trigger signal."

    last_row = df.iloc[-1]
    if last_row["SMA20"] > last_row["SMA50"]:
        return "📈 BUY (SMA20 > SMA50)"
    elif last_row["SMA20"] < last_row["SMA50"]:
        return "📉 SELL (SMA20 < SMA50)"
    else:
        return "🤔 HOLD"

# Main execution
if ticker:
    data = load_data(ticker)

    if data["SMA50"].isna().all():
        st.warning("⚠️ Not enough data to compute SMA50. Try a longer period.")
    else:
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode='lines', name='SMA20'))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode='lines', name='SMA50'))

        fig.update_layout(
            title=f"{ticker.upper()} Price & SMA Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=600,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Action
        signal = trigger_signal(data)
        st.subheader(f"💡 Suggested Action: {signal}")

        # Show raw data
        with st.expander("🔍 Show raw data"):
            st.dataframe(data.tail(50))
