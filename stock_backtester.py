import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# ---------------------------- CONFIG ----------------------------
st.set_page_config(page_title="📈 Stock Signals (Support/Resistance + Tips)", page_icon="📈", layout="wide")
st.title("📈 Stock Signals – Support/Resistance + Tips")
st.caption("Your tickers, your rules. Signals based on SMA20/SMA50, RSI(14), and S/R levels.")

# ---------------------------- YOUR TICKERS ----------------------------
TICKERS = [
    "AAPL", "ABCL", "AEHR", "AI", "AMZN", "BAM", "BRK",
    "CGL", "CRCL", "CRWV", "CU", "DCBO", "DRUG", "ENB",
    "GSI", "IRD", "ISRG", "JOBY", "LUNR", "MCLD.H", "MDA",
    "META", "MSFT", "NVDA", "NVTS", "NXT", "PSA", "QBTS",
    "QSR", "RAIL", "RDDT", "RGTI", "RXRX", "SMCI", "SOUN",
    "SRFM", "STN", "TEM", "TMC", "VEE",
]

YF_MAP = {sym: sym for sym in TICKERS}

# ---------------------------- HELPERS ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Force 1D Series
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.Series(series).astype(float)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    gain_ema = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss_ema = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = gain_ema / loss_ema.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.clip(0, 100)

def get_support_resistance(df: pd.DataFrame, window: int = 20):
    support = df['Low'].rolling(window).min()
    resistance = df['High'].rolling(window).max()
    return support, resistance

def generate_signal(row):
    price = row['Close']
    sma20 = row['SMA20']
    sma50 = row['SMA50']
    rsi_val = row['RSI']
    support = row['Support']
    resistance = row['Resistance']

    trend_bull = (price > sma20) and (sma20 > sma50)
    trend_bear = (price < sma20) and (sma20 < sma50)
    broke_up = (pd.notna(resistance) and price > resistance)
    broke_down = (pd.notna(support) and price < support)

    if broke_up and trend_bull and rsi_val < 70:
        return "BUY"
    if broke_down and trend_bear and rsi_val > 30:
        return "SELL"
    if trend_bull and rsi_val < 70 and price > sma20:
        return "BUY"
    if trend_bear and rsi_val > 30 and price < sma20:
        return "SELL"
    return "HOLD"

def tips_from_row(row, sl_pct=0.03, tp_pct=0.06):
    price = row['Close']
    support = row['Support']
    resistance = row['Resistance']
    signal = row['Signal']

    tips = []
    if pd.notna(support):
        dist_to_support = (price - support) / price * 100
        tips.append(f"Distance to support: {dist_to_support:.2f}%")
    else:
        tips.append("Support: N/A")

    if pd.notna(resistance):
        dist_to_res = (resistance - price) / price * 100
        tips.append(f"Distance to resistance: {dist_to_res:.2f}%")
    else:
        tips.append("Resistance: N/A")

    if signal == "BUY":
        stop_loss = price * (1 - sl_pct)
        if pd.notna(support):
            stop_loss = min(stop_loss, support * 0.995)
        take_profit = price * (1 + tp_pct)
        if pd.notna(resistance) and resistance > price:
            take_profit = max(take_profit, resistance * 1.01)
        tips.append(f"Suggested stop-loss: {stop_loss:.2f}")
        tips.append(f"Suggested take-profit: {take_profit:.2f}")

    elif signal == "SELL":
        stop_loss = price * (1 + sl_pct)
        if pd.notna(resistance):
            stop_loss = max(stop_loss, resistance * 1.005)
        take_profit = price * (1 - tp_pct)
        if pd.notna(support) and support < price:
            take_profit = min(take_profit, support * 0.99)
        tips.append(f"Suggested buy-to-cover (TP): {take_profit:.2f}")
        tips.append(f"Suggested stop-loss: {stop_loss:.2f}")

    else:
        tips.append("No strong signal; consider waiting for a break of support/resistance or a trend confirmation.")

    return " | ".join(tips)

@st.cache_data(ttl=3600)
def cached_download(ticker, start, end):
    return yf.download(
        ticker, start=start, end=end + timedelta(days=1),
        progress=False, auto_adjust=True, threads=False
    )

def fetch_and_analyze(ticker, start, end, window_sr=20):
    yf_ticker = YF_MAP.get(ticker, ticker)
    try:
        df = cached_download(yf_ticker, start, end)
    except Exception as e:
        return None, f"Download failed for {ticker} ({yf_ticker}): {e}"

    if df.empty:
        return None, f"No data for {ticker} (mapped to {yf_ticker})"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    close = df.get('Close')
    if close is None:
        return None, f"'Close' column missing for {ticker} (mapped to {yf_ticker})"

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    df['Close'] = close.astype(float)

    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'], 14)
    df['Support'], df['Resistance'] = get_support_resistance(df, window=window_sr)

    df = df.dropna(subset=['SMA20', 'SMA50', 'RSI', 'Support', 'Resistance'])
    if df.empty:
        return None, f"Not enough data to compute indicators for {ticker}"

    latest = df.iloc[-1].copy()
    latest['Signal'] = generate_signal(latest)
    tip = tips_from_row(latest)

    return (df, latest, tip), None

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name="SMA20", line=dict(width=1.3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50", line=dict(width=1.3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name="Support", line=dict(color="green", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], name="Resistance", line=dict(color="red", width=1, dash="dot")))
    fig.update_layout(title=f"{ticker} – Price, SMAs, Support/Resistance", xaxis_rangeslider_visible=False)
    return fig

def plot_rsi(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI(14)", line=dict(color="purple", width=1.3)))
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(title=f"{ticker} – RSI(14)")
    return fig

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End", date.today())
sr_window = st.sidebar.slider("Support/Resistance window (days)", 10, 60, 20)
scan_all = st.sidebar.checkbox("Scan all tickers", value=False)

# ---------------------------- SCAN ALL ----------------------------
if scan_all:
    rows = []
    errors = []
    progress = st.progress(0)
    for i, t in enumerate(TICKERS, start=1):
        res, err = fetch_and_analyze(t, start_date, end_date, sr_window)
        if err:
            errors.append(err)
        else:
            df, last, tip = res
            rows.append({
                "Ticker": t,
                "Close": last["Close"],
                "Signal": last["Signal"],
                "RSI": last["RSI"],
                "SMA20": last["SMA20"],
                "SMA50": last["SMA50"],
                "Support": last["Support"],
                "Resistance": last["Resistance"],
                "Tip": tip
            })
        progress.progress(i / len(TICKERS))

    if rows:
        scan_df = pd.DataFrame(rows).set_index("Ticker").sort_values(by="Signal", ascending=False)
        st.subheader("🔍 Scan Results")
        st.dataframe(scan_df, use_container_width=True)
    if errors:
        with st.expander("Symbols with no/invalid data"):
            for e in errors:
                st.write("•", e)
    st.stop()

# ---------------------------- SINGLE TICKER MODE ----------------------------
ticker = st.selectbox("Select a ticker", TICKERS, index=0)
res, err = fetch_and_analyze(ticker, start_date, end_date, sr_window)

if err:
    st.error(err)
    st.info("Try another ticker or update your Yahoo symbol mapping.")
else:
    df, latest, tip = res

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Close", f"{latest['Close']:.2f}")
    col2.metric("RSI(14)", f"{latest['RSI']:.2f}")
    col3.metric("SMA20 / SMA50", f"{latest['SMA20']:.2f} / {latest['SMA50']:.2f}")
    col4.metric("Signal", latest["Signal"])

    st.markdown(f"**Tips:** {tip}")

    price_fig = plot_chart(df, ticker)
    st.plotly_chart(price_fig, use_container_width=True)

    rsi_fig = plot_rsi(df, ticker)
    st.plotly_chart(rsi_fig, use_container_width=True)
