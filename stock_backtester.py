import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# ---------------------- Config ----------------------
st.set_page_config(page_title="📊 Stock Analysis Tool", layout="wide")
st.title("📊 Stock Analysis Tool")
st.caption("Analyze stock data with key indicators and signals.")

# ---------------------- Sidebar ----------------------
st.sidebar.header("Your Portfolio / Tickers")
portfolio_text = st.sidebar.text_area(
    "Enter your portfolio tickers separated by commas",
    value="AAPL, MSFT, AMZN"
).upper()

portfolio_tickers = [t.strip() for t in portfolio_text.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())
sr_window = st.sidebar.slider("Support/Resistance window (days)", 10, 60, 20)

# ---------------------- Helpers ----------------------

@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return None, "No data returned"
        return df, None
    except Exception as e:
        return None, f"Download error: {e}"

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def support_resistance(df, window=20):
    support = df['Low'].rolling(window).min()
    resistance = df['High'].rolling(window).max()
    return support, resistance

def generate_signal(row):
    price = row['Close']
    sma20 = row['SMA20']
    sma50 = row['SMA50']
    rsi_val = row['RSI']
    macd_val = row['MACD']
    macd_signal = row['MACD_signal']
    support = row['Support']
    resistance = row['Resistance']

    if price > sma20 and sma20 > sma50 and macd_val > macd_signal and rsi_val < 70:
        return "BUY"
    elif price < sma20 and sma20 < sma50 and macd_val < macd_signal and rsi_val > 30:
        return "SELL"
    elif price > resistance:
        return "BREAKOUT"
    elif price < support:
        return "BREAKDOWN"
    return "HOLD"

# ---------------------- Main analysis function ----------------------

def analyze_ticker(ticker):
    df, error = download_data(ticker, start_date, end_date)
    if error:
        return None, f"Error downloading {ticker}: {error}"

    if len(df) < 60:
        return None, f"Not enough data to analyze {ticker} (need at least 60 rows)."

    # Calculate indicators
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'])
    df['MACD'], df['MACD_signal'] = macd(df['Close'])
    df['Support'], df['Resistance'] = support_resistance(df, window=sr_window)

    # Drop NaN values
    df = df.dropna(subset=['SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'Support', 'Resistance'])
    
    if df.empty:
        return None, f"Not enough data after cleaning for {ticker}"

    # Get the latest data and generate signal
    latest = df.iloc[-1].copy()
    latest['Signal'] = generate_signal(latest)

    return df, latest

# ---------------------- UI ----------------------

if portfolio_tickers:
    ticker_to_analyze = st.selectbox("Choose a stock to analyze", portfolio_tickers)
else:
    ticker_to_analyze = None
    st.info("Please enter tickers in the sidebar.")

if ticker_to_analyze:
    df, latest = analyze_ticker(ticker_to_analyze)

    if latest is None:
        st.error("Error: " + latest)
    else:
        st.markdown(f"### Latest data for {ticker_to_analyze}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Last Close", f"{latest['Close']:.2f}")
        col2.metric("RSI(14)", f"{latest['RSI']:.2f}")
        col3.metric("SMA20 / SMA50", f"{latest['SMA20']:.2f} / {latest['SMA50']:.2f}")
        col4.metric("MACD", f"{latest['MACD']:.2f}")
        col5.metric("Signal", latest['Signal'])

        st.markdown(f"### Support/Resistance: {latest['Support']} / {latest['Resistance']}")

        # Price + SMA + S/R Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='Support', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], name='Resistance', line=dict(color='red', dash='dot')))
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
