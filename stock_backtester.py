import streamlit as st
import yfinance as yf
import requests
from textblob import TextBlob
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# ---------------------- Sidebar ----------------------
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())

# ---------------------- News Fetching Function ----------------------
def get_stock_news(ticker):
    api_key = 'YOUR_NEWSAPI_KEY'  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles']

# ---------------------- Sentiment Analysis ----------------------
def analyze_sentiment(news_articles):
    sentiments = []
    for article in news_articles:
        text = article['title'] + ' ' + article['description']
        sentiment = TextBlob(text).sentiment.polarity  # Get sentiment polarity
        sentiments.append(sentiment)
    return sum(sentiments) / len(sentiments) if sentiments else 0

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
        # Get News Data
        news = get_stock_news(ticker)
        sentiment_score = analyze_sentiment(news)
        
        # Show News Sentiment
        st.write(f"**News Sentiment for {ticker}:** {'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'} (Sentiment Score: {sentiment_score:.2f})")
        
        # Plot Stock Data
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick"
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # AI-based Analysis (Simple Example)
        if sentiment_score > 0:
            st.write("AI-based Prediction: **Bullish** - Stock might go up based on positive sentiment!")
        else:
            st.write("AI-based Prediction: **Bearish** - Stock might go down based on negative sentiment.")
