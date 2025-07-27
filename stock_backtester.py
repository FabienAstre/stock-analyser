import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# ---------------------- Config ----------------------
st.set_page_config(page_title="📊 Simple Stock Analysis", layout="wide")
st.title("📊 Simple Stock Analysis")
st.caption("A basic stock analysis tool with SMA and RSI indicators.")

# ---------------------- Sidebar ----------------------
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())

# ---------------------- Helper Function ----------------------

# Download stock data
@st.cache_data
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None, "No data returned"
        return df, None
    except Exception as e:
        return None, f"Download error: {e}"

# ---------------------- Main ----------------------

if ticker:
    df, error = download_data(ticker, start_date, end_date)

    if error:
        st.error(error)
    else:
        # Calculate SMA (20 and 50 periods)
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        # Calculate RSI (14 periods)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ---------------------- Chart: Candlestick with SMAs ----------------------
        fig = go.Figure()

        # Candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick"
        ))

        # SMA20 trace
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='blue'))
        )

        # SMA50 trace
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange'))
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------- RSI Chart ----------------------
        fig_rsi = go.Figure()

        # RSI trace
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], name="RSI(14)", line=dict(color='purple'))
        )

        # Overbought and Oversold lines
        fig_rsi.add_hline(y=70, line_dash='dot', line_color='red', annotation_text='Overbought (70)')
        fig_rsi.add_hline(y=30, line_dash='dot', line_color='green', annotation_text='Oversold (30)')

        st.plotly_chart(fig_rsi, use_container_width=True)
