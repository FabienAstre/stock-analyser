import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
import pandas as pd

# ---------------------- Sidebar ----------------------
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())

# ---------------------- Technical Indicator Functions ----------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain_ema = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss_ema = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain_ema / loss_ema.replace(0, pd.NA)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def sma(series, window=20):
    return series.rolling(window=window).mean()

# ---------------------- Stock Data Fetching ----------------------
@st.cache_data
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        return df, None
    except Exception as e:
        return None, f"Error downloading {ticker}: {e}"

# ---------------------- Main ----------------------
if ticker:
    # Fetch Stock Data
    df, error = download_data(ticker, start_date, end_date)
    
    if error:
        st.error(error)
    else:
        # Calculate Indicators
        df['SMA20'] = sma(df['Close'], 20)
        df['SMA50'] = sma(df['Close'], 50)
        df['RSI'] = rsi(df['Close'])
        df['MACD'], df['MACD_signal'] = macd(df['Close'])

        # Plot Stock Data
        st.subheader(f"{ticker} Stock Analysis")

        # Candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick"
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='red')))
        st.plotly_chart(fig, use_container_width=True)

        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI(14)', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash='dot', line_color='red', annotation_text='Overbought (70)')
        fig_rsi.add_hline(y=30, line_dash='dot', line_color='green', annotation_text='Oversold (30)')
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD Chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal Line', line=dict(color='orange')))
        st.plotly_chart(fig_macd, use_container_width=True)
