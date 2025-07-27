def analyze_ticker(ticker):
    df = download_data(ticker, start_date, end_date)
    if df is None or df.empty:
        return None, None, f"No data for {ticker}"

    # Indicators
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'])
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'])
    df['ATR'] = atr(df)
    df['Support'], df['Resistance'] = support_resistance(df, window=sr_window)

    df.dropna(subset=['SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'Support', 'Resistance'], inplace=True)
    if df.empty:
        return None, None, f"Not enough data to calculate indicators for {ticker}"

    latest = df.iloc[-1].copy()
    latest['Signal'] = generate_signal(latest)
    latest['Tips'] = tips(latest)

    return df, latest, None

# Main app part

ticker_to_analyze = st.selectbox("Choose a stock to analyze", options=portfolio_tickers, index=0 if portfolio_tickers else None)

if ticker_to_analyze:
    result = analyze_ticker(ticker_to_analyze)

    if result is None or len(result) != 3:
        st.error("Unexpected error: analyze_ticker() did not return valid data.")
    else:
        df, latest, error = result
        if error:
            st.error(error)
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Last Close", f"{latest['Close']:.2f}")
            col2.metric("RSI(14)", f"{latest['RSI']:.2f}")
            col3.metric("SMA20 / SMA50", f"{latest['SMA20']:.2f} / {latest['SMA50']:.2f}")
            col4.metric("MACD", f"{latest['MACD']:.2f}")
            col5.metric("Signal", latest['Signal'])

            st.markdown(f"**Tips:** {latest['Tips']}")

            # Plot price + indicators
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='Support', line=dict(color='green', dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], name='Resistance', line=dict(color='red', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

            # RSI chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI(14)', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash='dot', line_color='red', annotation_text='Overbought (70)')
            fig_rsi.add_hline(y=30, line_dash='dot', line_color='green', annotation_text='Oversold (30)')
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal Line', line=dict(color='orange')))
            st.plotly_chart(fig_macd, use_container_width=True)

else:
    st.info("Please enter tickers in the sidebar to analyze.")
