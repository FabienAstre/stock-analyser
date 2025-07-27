import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="📈 Stock Portfolio Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Portfolio Analyzer")
st.write("View buy/sell signals, analyze resistance levels, and track portfolio performance.")

# ---------------- PORTFOLIO ----------------
portfolio_data = [
    {"Symbol": "AAPL", "Company": "Apple CDR (CAD Hedged)", "Total Value": 631.96},
    {"Symbol": "ABCL", "Company": "AbCellera Biologics Inc", "Total Value": 339.21},
    {"Symbol": "AEHR", "Company": "Aehr Test Systems", "Total Value": 434.80},
    {"Symbol": "AI", "Company": "C3.ai Inc (Class A)", "Total Value": 391.35},
    {"Symbol": "AMZN", "Company": "Amazon.com CDR (CAD Hedged)", "Total Value": 1629.00},
    {"Symbol": "BAM", "Company": "Brookfield Asset Management Ltd", "Total Value": 439.48},
    {"Symbol": "BRK", "Company": "Berkshire Hathaway CDR", "Total Value": 924.76},
    {"Symbol": "CGL", "Company": "iShares Gold Bullion ETF CAD-Hedged", "Total Value": 1323.57},
    {"Symbol": "CRCL", "Company": "Circle Internet Group Inc.", "Total Value": 961.50},
    {"Symbol": "CRWV", "Company": "CoreWeave Inc.", "Total Value": 230.48},
    {"Symbol": "CU", "Company": "Canadian Utilities Ltd. (Class A)", "Total Value": 1261.35},
    {"Symbol": "DCBO", "Company": "Docebo Inc", "Total Value": 301.84},
    {"Symbol": "DRUG", "Company": "Bright Minds Biosciences Inc.", "Total Value": 481.00},
    {"Symbol": "ENB", "Company": "Enbridge Inc", "Total Value": 666.92},
    {"Symbol": "GSI", "Company": "Gatekeeper Systems Inc.", "Total Value": 405.00},
    {"Symbol": "ISRG", "Company": "Intuitive Surgical CDR", "Total Value": 356.07},
    {"Symbol": "JOBY", "Company": "Joby Aviation Inc", "Total Value": 800.10},
    {"Symbol": "MSFT", "Company": "Microsoft CDR", "Total Value": 1848.99},
    {"Symbol": "NVDA", "Company": "Nvidia CDR", "Total Value": 2989.06},
    {"Symbol": "META", "Company": "Meta CDR", "Total Value": 1055.04},
    {"Symbol": "NVTS", "Company": "Navitas Semiconductor Corp", "Total Value": 614.59},
    {"Symbol": "QBTS", "Company": "D-Wave Quantum Inc.", "Total Value": 464.65},
    {"Symbol": "QSR", "Company": "Restaurant Brands International Inc", "Total Value": 484.17},
    {"Symbol": "RAIL", "Company": "Railtown A.I. Technologies Inc.", "Total Value": 380.00},
    {"Symbol": "RDDT", "Company": "Reddit Inc.", "Total Value": 751.00},
    {"Symbol": "RXRX", "Company": "Recursion Pharmaceuticals Inc", "Total Value": 324.53},
    {"Symbol": "SMCI", "Company": "Supermicro CDR", "Total Value": 235.32},
    {"Symbol": "SOUN", "Company": "SoundHound AI Inc", "Total Value": 939.19},
    {"Symbol": "SRFM", "Company": "Surf Air Mobility Inc", "Total Value": 287.40},
    {"Symbol": "STN", "Company": "Stantec Inc", "Total Value": 300.69},
    {"Symbol": "TEM", "Company": "Tempus A.I. Inc.", "Total Value": 451.01},
    {"Symbol": "TMC", "Company": "TMC the metals company Inc", "Total Value": 389.00},
    {"Symbol": "VEE", "Company": "Vanguard FTSE Emerging Markets ETF", "Total Value": 838.97},
]

portfolio_df = pd.DataFrame(portfolio_data)
total_value = portfolio_df["Total Value"].sum()

st.metric("Total Portfolio Value", f"${total_value:,.2f} CAD")

# ---------------- STOCK SELECTION ----------------
ticker = st.selectbox("Select a Stock", [item["Symbol"] for item in portfolio_data])

# ---------------- FETCH DATA ----------------
start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
end_date = st.date_input("End Date", datetime.date.today())

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error(f"No data found for {ticker}.")
else:
    # Calculate Indicators
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).apply(lambda x: (x[x > 0].mean() / abs(x[x < 0].mean())) if abs(x[x < 0].mean()) > 0 else 0)))
    
    # Resistance / Buy-Sell Signals
    data['Signal'] = 0
    data.loc[(data['Close'] > data['SMA20']) & (data['SMA20'] > data['SMA50']), 'Signal'] = 1  # Buy
    data.loc[(data['Close'] < data['SMA20']) & (data['SMA20'] < data['SMA50']), 'Signal'] = -1 # Sell

    # Display Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candles'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='blue', width=1.5), name='SMA20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='orange', width=1.5), name='SMA50'))
    
    # Plot Buy/Sell markers
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=8), name='Buy'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=8), name='Sell'))
    
    fig.update_layout(title=f"{ticker} Price with SMA & Signals", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    st.subheader("RSI Indicator")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple', width=1.5), name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Display last signals
    last_signal = data['Signal'].iloc[-1]
    if last_signal == 1:
        st.success(f"**Latest Signal for {ticker}: BUY**")
    elif last_signal == -1:
        st.error(f"**Latest Signal for {ticker}: SELL**")
    else:
        st.info(f"**Latest Signal for {ticker}: HOLD**")
