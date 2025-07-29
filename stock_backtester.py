import yfinance as yf
import plotly.graph_objs as go
import streamlit as st
import pandas as pd

# --- Fetch data ---
def get_stock_data(ticker, period="3mo", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    return data

# --- Plot with Plotly ---
def plot_stock(data, ticker):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA20'],
        line=dict(color='blue', width=1),
        name='SMA 20'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA50'],
        line=dict(color='orange', width=1),
        name='SMA 50'
    ))

    # Layout
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI ---
st.title("📊 Stock Market Visualizer")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", value="AAPL")
period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    df = get_stock_data(ticker, period=period)
    if not df.empty:
        plot_stock(df, ticker)
    else:
        st.error("No data found.")
