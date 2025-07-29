import streamlit as st
import yfinance as yf
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import nltk

nltk.download('vader_lexicon')

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo")  # 1 month daily data
    return df

def fetch_stock_news(ticker, api_key, page_size=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": api_key
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return [article["title"] for article in data.get("articles", [])]

def analyze_sentiment(news_list):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for news in news_list:
        vs = analyzer.polarity_scores(news)
        score = (vs['compound'] + 1) / 2 * 100
        scores.append(score)
    return sum(scores) / len(scores) if scores else None

st.title("📈 Stock Market AI Analyzer")

ticker = st.text_input("Enter stock ticker (e.g., AAPL)")

if ticker:
    df = fetch_stock_data(ticker)
    st.line_chart(df["Close"])

    news = fetch_stock_news(ticker, "YOUR_NEWSAPI_KEY_HERE")
    st.write("Recent news headlines:")
    for n in news:
        st.write("-", n)

    sentiment_score = analyze_sentiment(news)
    st.write(f"News Sentiment Score: {sentiment_score:.1f}%" if sentiment_score else "No sentiment data")
