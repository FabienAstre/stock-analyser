import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# ---------------------- Config ----------------------
st.set_page_config(page_title="📊 Comprehensive Stock Analysis", layout="wide")

st.title("📊 Comprehensive Stock Analysis Tool")
st.caption("Analyze any stock symbol with multiple indicators and signals")

# ---------------------- Sidebar: Portfolio & Settings ----------------------
st.sidebar.header("Your Portfolio / Tickers")
portfolio_text = st.sidebar.text_area(
    "Enter your portfolio tickers separated by commas",
    value="AAPL, MSFT, AMZN"
).upper()

portfolio_tickers = [t.strip() for t in portfolio_text.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())
sr_window = st.sidebar.slider("Support/Resistance window (days)", 10, 60, 20)

# ---------------------- Indicators ----------------------

@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=True, threads=False)
    if df.empty:
        return None
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    gain_ema = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss_ema = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = gain_ema / loss_ema.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.clip(0, 100)

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    return atr_val

def support_resistance(df, window=20):
    support = df['Low'].rolling(window).min()
    resistance = df['High'].rolling(window).max()
    return support, resistance

# ---------------------- Signal Logic ----------------------

def generate_signal(row):
    price = row['Close']
    sma20 = row['SMA20']
    sma50 = row['SMA50']
    rsi_val = row['RSI']
    macd_val = row['MACD']
    macd_signal = row['MACD_signal']
    support = row['Support']
    resistance = row['Resistance']

    trend_bull = (price > sma20) and (sma20 > sma50)
    trend_bear = (price < sma20) and (sma20 < sma50)
    macd_bull = macd_val > macd_signal
    macd_bear = macd_val < macd_signal
    broke_up = pd.notna(resistance) and price > resistance
    broke_down = pd.notna(support) and price < support

    if broke_up and trend_bull and macd_bull and rsi_val < 70:
        return "STRONG BUY"
    if broke_down and trend_bear and macd_bear and rsi_val > 30:
        return "STRONG SELL"
    if trend_bull and macd_bull and rsi_val < 70:
        return "BUY"
    if trend_bear and macd_bear and rsi_val > 30:
        return "SELL"
    return "HOLD"

def tips(row):
    price = row['Close']
    support = row['Support']
    resistance = row['Resistance']
    atr_val = row['ATR']
    signal = row['Signal']

    tips = []

    if pd.notna(support):
        tips.append(f"Distance to Support: {(price - support) / price * 100:.2f}%")
    else:
        tips.append("Support: N/A")

    if pd.notna(resistance):
        tips.append(f"Distance to Resistance: {(resistance - price) / price * 100:.2f}%")
    else:
        tips.append("Resistance: N/A")

    if pd.notna(atr_val):
        tips.append(f"ATR (14): {atr_val:.2f}")

    if signal in ("BUY", "STRONG BUY"):
        sl = price - atr_val if pd.notna(atr_val) else price * 0.97
        tp = price + 2 * atr_val if pd.notna(atr_val) else price * 1.06
        tips.append(f"Suggested Stop-Loss ~ {sl:.2f}")
        tips.append(f"Suggested Take-Profit ~ {tp:.2f}")

    elif signal in ("SELL", "STRONG SELL"):
        sl = price + atr_val if pd.notna(atr_val) else price * 1.03
        tp = price - 2 * atr_val if pd.notna(atr_val) else price * 0.94
        tips.append(f"Suggested Buy-to-Cover (TP) ~ {tp:.2f}")
        tips.append(f"Suggested Stop-Loss ~ {sl:.2f}")
    else:
        tips.append("No strong actionable signal.")

    return " | ".join(tips)

# ---------------------- Data Processing ----------------------

def analyze_ticker(ticker):
    df = download_data(ticker, start_date, end_date)
    if df is None or df.empty:
        return None, None, f"No data for {ticker}"

    # Indicators
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'])
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'])
    df['ATR'] = atr(df)
    df['Support'], df['Resistance'] = support_resistance(df, window=sr_window)

required_cols = ['SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'Support', 'Resistance']
existing_cols = [col for col in required_cols if col in df.columns]

if len(existing_cols) < len(required_cols):
    missing = set(required_cols) - set(existing_cols)
    return None, None, f"Missing indicator columns: {', '.join(missing)}"

df.dropna(subset=existing_cols, inplace=True)
    if df.empty:
        return None, None, f"Not enough data to calculate indicators for {ticker}"

    latest = df.iloc[-1].copy()
    latest['Signal'] = generate_signal(latest)
    latest['Tips'] = tips(latest)

    return df, latest, None

# ---------------------- UI ----------------------

ticker_to_analyze = st.selectbox("Choose a stock to analyze", options=portfolio_tickers, index=0 if portfolio_tickers else None)

if ticker_to_analyze:
    result = analyze_ticker(ticker_to_analyze)

    if result is None or len(result) != 3:
        st.error("Unexpected error: analyze_ticker() did not return valid data.")
    else:
        df, latest, error = result
        if error:
            st.error(error)
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Last Close", f"{latest['Close']:.2f}")
            col2.metric("RSI(14)", f"{latest['RSI']:.2f}")
            col3.metric("SMA20 / SMA50", f"{latest['SMA20']:.2f} / {latest['SMA50']:.2f}")
            col4.metric("MACD", f"{latest['MACD']:.2f}")
            col5.metric("Signal", latest['Signal'])

            st.markdown(f"**Tips:** {latest['Tips']}")

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

else:
    st.info("Please enter tickers in the sidebar to analyze.")
