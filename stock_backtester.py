import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# ---------------------------- CONFIG ----------------------------
st.set_page_config(page_title="📈 Stock Signals (Support/Resistance + Tips)", page_icon="📈", layout="wide")
st.title("📈 Stock Signals – Support/Resistance + Tips")
st.caption("Your tickers, your rules. No lotto. Signals based on SMA20/SMA50, RSI(14), and S/R levels.")

# ---------------------------- YOUR TICKERS ----------------------------
# If you want to map your display symbols to Yahoo tickers, do it here (left=display, right=yf symbol).
# For now, we try as-is; those that fail will be skipped.
TICKERS = [
    "AAPL",      # Apple CDR (CAD Hedged) - likely wrong for Yahoo, but we keep as user-provided
    "ABCL",
    "AEHR",
    "AI",
    "AMZN",
    "BAM",
    "BRK",       # Berkshire CDR (may not work)
    "CGL",
    "CRCL",      # might fail
    "CRWV",      # might fail
    "CU",
    "DCBO",
    "DRUG",
    "ENB",
    "GSI",       # might fail
    "IRD",       # (Opus Genetics) missing in earlier lists but present in your text, might fail
    "ISRG",
    "JOBY",
    "LUNR",
    "MCLD.H",    # might fail
    "MDA",
    "META",
    "MSFT",
    "NVDA",
    "NVTS",
    "NXT",
    "PSA",
    "QBTS",
    "QSR",
    "RAIL",
    "RDDT",
    "RGTI",
    "RXRX",
    "SMCI",
    "SOUN",
    "SRFM",
    "STN",
    "TEM",
    "TMC",
    "VEE",
]

# If you do know the correct Yahoo symbols, set a mapping like:
# YF_MAP = {"AAPL": "AAPL.NE", "MSFT": "MSFT.NE", ...}
YF_MAP = {sym: sym for sym in TICKERS}

# ---------------------------- HELPERS ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, min_periods=period).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain_ema / loss_ema.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def get_support_resistance(df: pd.DataFrame, window: int = 20):
    """
    Simple rolling support/resistance:
      - Support: rolling min of Low over last `window`
      - Resistance: rolling max of High over last `window`
    """
    support = df['Low'].rolling(window).min()
    resistance = df['High'].rolling(window).max()
    return support, resistance

def generate_signal(row):
    """
    Generate a BUY/SELL/HOLD suggestion given the latest row of indicators.
    Priority:
      - If price breaks ABOVE resistance with bull trend -> BUY
      - If price breaks BELOW support with bear trend -> SELL
      - Else HOLD
    """
    price = row['Close']
    sma20 = row['SMA20']
    sma50 = row['SMA50']
    rsi_val = row['RSI']
    support = row['Support']
    resistance = row['Resistance']

    trend_bull = (price > sma20) and (sma20 > sma50)
    trend_bear = (price < sma20) and (sma20 < sma50)
    broke_up = (price > resistance) if not np.isnan(resistance) else False
    broke_down = (price < support) if not np.isnan(support) else False

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
    """
    Provide a simple tip:
      - If BUY: stop-loss slightly below support or -3%, TP slightly above resistance or +6%
      - If SELL: the inverse
      - HOLD: show distances to S/R
    """
    price = row['Close']
    support = row['Support']
    resistance = row['Resistance']
    signal = row['Signal']

    tips = []

    # Distances
    if not np.isnan(support):
        dist_to_support = (price - support) / price * 100
        tips.append(f"Distance to support: {dist_to_support:.2f}%")
    else:
        tips.append("Support: N/A")

    if not np.isnan(resistance):
        dist_to_res = (resistance - price) / price * 100
        tips.append(f"Distance to resistance: {dist_to_res:.2f}%")
    else:
        tips.append("Resistance: N/A")

    if signal == "BUY":
        # choose SL: max(support, price*(1-sl_pct)) if support exists
        stop_loss = price * (1 - sl_pct)
        if not np.isnan(support):
            stop_loss = min(stop_loss, support * 0.995)  # a hair below support
        take_profit = price * (1 + tp_pct)
        if not np.isnan(resistance) and resistance > price:
            take_profit = max(take_profit, resistance * 1.01)  # slightly above resistance
        tips.append(f"Suggested stop-loss: {stop_loss:.2f}")
        tips.append(f"Suggested take-profit: {take_profit:.2f}")

    elif signal == "SELL":
        # if shorting, invert logic
        stop_loss = price * (1 + sl_pct)
        if not np.isnan(resistance):
            stop_loss = max(stop_loss, resistance * 1.005)
        take_profit = price * (1 - tp_pct)
        if not np.isnan(support) and support < price:
            take_profit = min(take_profit, support * 0.99)
        tips.append(f"Suggested buy-to-cover (TP): {take_profit:.2f}")
        tips.append(f"Suggested stop-loss: {stop_loss:.2f}")

    else:
        tips.append("No strong signal; consider waiting for a break of support/resistance or a trend confirmation.")

    return " | ".join(tips)

def fetch_and_analyze(ticker, start, end, window_sr=20):
    yf_ticker = YF_MAP.get(ticker, ticker)
    df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None, f"No data for {ticker} (mapped to {yf_ticker})"

    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'], 14)
    df['Support'], df['Resistance'] = get_support_resistance(df, window=window_sr)

    # remove rows without indicators yet
    df = df.dropna(subset=['SMA20', 'SMA50', 'RSI', 'Support', 'Resistance'])

    if df.empty:
        return None, f"Not enough data to compute indicators for {ticker}"

    # Latest signal/tips
    latest = df.iloc[-1].copy()
    signal = generate_signal(latest)
    latest['Signal'] = signal
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

    # Charts
    price_fig = plot_chart(df, ticker)
    st.plotly_chart(price_fig, use_container_width=True)

    rsi_fig = plot_rsi(df, ticker)
    st.plotly_chart(rsi_fig, use_container_width=True)
