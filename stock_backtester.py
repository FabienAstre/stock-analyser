import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="📈 Stock Pattern & Backtesting App", layout="wide")

st.title("📈 Stock Pattern & Backtesting App")
st.write("Analyze historical stock patterns and backtest trading strategies.")

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- Fetch Data ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

st.subheader(f"Stock Data for {ticker}")
st.dataframe(data.tail(10))

# --- Plot Price Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

# --- Strategy: SMA Crossover ---
st.sidebar.header("Strategy Parameters")
short_window = st.sidebar.slider("Short SMA Window", 5, 50, 20)
long_window = st.sidebar.slider("Long SMA Window", 20, 200, 50)

data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

# Generate buy/sell signals with correct .loc assignment
data['Signal'] = 0
mask = data.index[short_window:]
data.loc[mask, 'Signal'] = np.where(
    data.loc[mask, 'SMA_Short'] > data.loc[mask, 'SMA_Long'], 1, 0
)
data['Position'] = data['Signal'].diff()

# Plot SMA Crossover
fig_sma = go.Figure()
fig_sma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
fig_sma.add_trace(go.Scatter(x=data.index, y=data['SMA_Short'], mode='lines', name=f'SMA {short_window}'))
fig_sma.add_trace(go.Scatter(x=data.index, y=data['SMA_Long'], mode='lines', name=f'SMA {long_window}'))

# Add buy/sell markers
buy_signals = data[data['Position'] == 1]
sell_signals = data[data['Position'] == -1]
fig_sma.add_trace(go.Scatter(
    x=buy_signals.index, y=buy_signals['Close'],
    mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Buy Signal'
))
fig_sma.add_trace(go.Scatter(
    x=sell_signals.index, y=sell_signals['Close'],
    mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Sell Signal'
))

fig_sma.update_layout(title=f"SMA Crossover Strategy for {ticker}", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_sma, use_container_width=True)

# --- Backtest ---
initial_capital = 10000
positions = pd.DataFrame(index=data.index)
positions['Position'] = data['Signal'].fillna(0) * (initial_capital / data['Close'].iloc[0])
positions['Position'] = positions['Position'].fillna(0)

portfolio = pd.DataFrame(index=data.index)
portfolio['Holdings'] = positions['Position'].reindex(data.index).fillna(0) * data['Close'].reindex(data.index).fillna(0)

trade_values = positions['Position'].diff().fillna(0) * data['Close'].fillna(0)
portfolio['Cash'] = initial_capital - trade_values.cumsum()

portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']

# --- Plot Portfolio Value ---
st.subheader("Backtest Portfolio Value")
fig_port = go.Figure()
fig_port.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Total'], mode='lines', name='Portfolio Value'))
fig_port.update_layout(title="Portfolio Performance", xaxis_title="Date", yaxis_title="Value ($)")
st.plotly_chart(fig_port, use_container_width=True)

final_value = portfolio['Total'].iloc[-1]
profit = final_value - initial_capital
st.metric("Final Portfolio Value", f"${final_value:,.2f}", f"{profit:+,.2f}")
