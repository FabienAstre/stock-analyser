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

# ---------------------- Buy/Sell Signal Generation ----------------------
def generate_signals(df):
    # Buy signal: When RSI < 30 (oversold), MACD crosses above the Signal Line, and price above SMA
    buy_signal = (df['RSI'] < 30) & (df['MACD'] > df['MACD_signal']) & (df['Close'] > df['SMA20'])
    
    # Sell signal: When RSI > 70 (overbought), MACD crosses below the Signal Line, and price below SMA
    sell_signal = (df['RSI'] > 70) & (df['MACD'] < df['MACD_signal']) & (df['Close'] < df['SMA20'])
    
    return buy_signal, sell_signal

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

        # Remove rows with NaN values in SMA, RSI, MACD columns
        df = df.dropna(subset=['SMA20', 'RSI', 'MACD', 'MACD_signal'])

        # Generate Buy/Sell Signals
        buy_signal, sell_signal = generate_signals(df)
        df['Buy_Signal'] = buy_signal
        df['Sell_Signal'] = sell_signal
        
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

        # Buy and Sell Signal Highlight
        st.subheader("Buy and Sell Signals")

        # Plot Buy Signals
        fig_buy = go.Figure()
        fig_buy.add_trace(go.Scatter(x=df.index[buy_signal], y=df['Close'][buy_signal],
                                    mode='markers', name='Buy Signal', marker=dict(color='green', size=12)))
        fig_buy.add_trace(go.Scatter(x=df.index[sell_signal], y=df['Close'][sell_signal],
                                    mode='markers', name='Sell Signal', marker=dict(color='red', size=12)))
        fig_buy.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                        name="Price"))
        st.plotly_chart(fig_buy, use_container_width=True)

        # Indicator Legend & Explanation
        st.subheader("Indicator Legend and Explanation")

        st.write("""
            **1. Simple Moving Average (SMA):**
            - **SMA20**: 20-day moving average. Shows the average closing price over the last 20 days.
            - **SMA50**: 50-day moving average. Shows the average closing price over the last 50 days.
            - **Buy Signal**: When the price is above the SMA, it suggests an uptrend.
            - **Sell Signal**: When the price is below the SMA, it suggests a downtrend.

            **2. Relative Strength Index (RSI):**
            - **RSI(14)**: Measures the strength of a stock’s recent price action.
            - **Overbought**: RSI > 70 (potential sell signal).
            - **Oversold**: RSI < 30 (potential buy signal).
            - **Buy Signal**: When RSI is below 30, the stock is considered oversold.
            - **Sell Signal**: When RSI is above 70, the stock is considered overbought.

            **3. Moving Average Convergence Divergence (MACD):**
            - **MACD Line**: Difference between a short-term (12-day) and long-term (26-day) exponential moving average.
            - **Signal Line**: 9-day EMA of the MACD line.
            - **Buy Signal**: When MACD crosses above the Signal Line, it suggests a bullish trend.
            - **Sell Signal**: When MACD crosses below the Signal Line, it suggests a bearish trend.

            **Buy and Sell Signals:**
            - **Buy Signal**: When RSI < 30, MACD crosses above the Signal Line, and the price is above SMA20.
            - **Sell Signal**: When RSI > 70, MACD crosses below the Signal Line, and the price is below SMA20.
        """)
