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

# ---------------------- Helpers ----------------------

# Download data
@st.cache_data
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None, "No data returned"
        return df, None
    except Exception as e:
        return None, f"Download error: {e}"

# Calculate RSI
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

# ---------------------- Main analysis function ----------------------

def analyze_ticker(ticker):
    df, error = download_data(ticker, start_date, end_date)
    if error:
        return None, f"Error downloading {ticker}: {error}"

    if len(df) < 60:
        return None, f"Not enough data to analyze {ticker} (need at least 60 rows)."

    # Calculate SMA (20 and 50 periods)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # Calculate RSI (14 period)
    df['RSI'] = rsi(df['Close'])

    return df, None

# ---------------------- UI ----------------------

if ticker:
    df, error = analyze_ticker(ticker)

    if error:
        st.error(error)
    else:
        # Check if the DataFrame is empty before trying to access the last row
        if not df.empty:
            # Show Metrics
            col1, col2 = st.columns(2)
            col1.metric("Last Close", f"{df['Close'].iloc[-1]:.2f}")
            col2.metric("RSI(14)", f"{df['RSI'].iloc[-1]:.2f}")

            # Plot Candlestick chart with SMA
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Candlestick"
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='blue'))
            )
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange'))
            )
            st.plotly_chart(fig, use_container_width=True)

            # Plot RSI chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index, y=df['RSI'], name="RSI(14)", line=dict(color='purple'))
            )
            fig_rsi.add_hline(y=70, line_dash='dot', line_color='red', annotation_text='Overbought (70)')
            fig_rsi.add_hline(y=30, line_dash='dot', line_color='green', annotation_text='Oversold (30)')
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.error("No data available to display.")
