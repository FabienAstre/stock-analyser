import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from transformers import pipeline

# Load sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_model()

# --- Fetch Data ---
def get_stock_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period, interval="1d")
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- Plot Chart ---
def plot_chart(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', line=dict(color='orange')))

    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=800,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- News Sentiment ---
def analyze_sentiment(text):
    if text:
        result = sentiment_analyzer(text[:500])[0]
        return f"{result['label']} ({round(result['score'] * 100, 1)}%)"
    return "No input"

# --- Buy/Sell Trigger ---
def trigger_signal(df):
    last_row = df.iloc[-1]
    if last_row['SMA20'] > last_row['SMA50']:
        return "📈 BUY (SMA20 > SMA50)"
    elif last_row['SMA20'] < last_row['SMA50']:
        return "📉 SELL (SMA20 < SMA50)"
    else:
        return "🤔 HOLD"

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📊 AI-Powered Stock Market Dashboard")

col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Enter stock ticker:", "AAPL")
    period = st.selectbox("Select time range:", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    if ticker:
        df = get_stock_data(ticker, period)
        if not df.empty:
            plot_chart(df, ticker)
            st.subheader(f"💡 Suggested Action: {trigger_signal(df)}")
        else:
            st.error("No data found for that ticker.")

with col2:
    st.subheader("🧠 News Sentiment Analysis")
    news_text = st.text_area("Paste relevant stock news/article:", height=200)
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(news_text)
        st.success(f"📰 Sentiment: {sentiment}")
