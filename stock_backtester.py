import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="📊 My Stock Portfolio Tracker", page_icon="💹", layout="wide")

st.title("📊 My Stock Portfolio Tracker")
st.write("Track your portfolio value, daily changes, and all-time returns with interactive charts.")

# --- Portfolio Data ---
portfolio_data = [
    {"Symbol": "AAPL", "Company": "Apple CDR (CAD Hedged)", "Total Value": 631.96, "Shares": 20.432, "Price": 30.93, "Daily Change %": -0.10, "All Time Return": -23.12, "All Time %": -3.53},
    {"Symbol": "ABCL", "Company": "AbCellera Biologics Inc", "Total Value": 339.21, "Shares": 66.2519, "Price": 5.12, "Daily Change %": 1.99, "All Time Return": 88.51, "All Time %": 35.31},
    {"Symbol": "AEHR", "Company": "Aehr Test Systems", "Total Value": 434.80, "Shares": 20, "Price": 21.74, "Daily Change %": 1.97, "All Time Return": 230.55, "All Time %": 112.87},
    {"Symbol": "AI", "Company": "C3.ai Inc (Class A)", "Total Value": 391.35, "Shares": 15, "Price": 26.09, "Daily Change %": 0.37, "All Time Return": -129.60, "All Time %": -24.88},
    {"Symbol": "AMZN", "Company": "Amazon.com CDR (CAD Hedged)", "Total Value": 1629.00, "Shares": 60, "Price": 27.15, "Daily Change %": -0.55, "All Time Return": 832.10, "All Time %": 104.42},
    {"Symbol": "BAM", "Company": "Brookfield Asset Management Ltd", "Total Value": 439.48, "Shares": 5.0877, "Price": 86.38, "Daily Change %": 0.23, "All Time Return": 57.44, "All Time %": 15.03},
    {"Symbol": "BRK", "Company": "Berkshire Hathaway CDR", "Total Value": 924.76, "Shares": 25.6879, "Price": 36.00, "Daily Change %": 0.70, "All Time Return": 79.41, "All Time %": 9.39},
    {"Symbol": "CGL", "Company": "iShares Gold Bullion ETF CAD-Hedged", "Total Value": 1323.57, "Shares": 50.9064, "Price": 26.00, "Daily Change %": -0.88, "All Time Return": 193.95, "All Time %": 17.17},
    {"Symbol": "CRCL", "Company": "Circle Internet Group Inc.", "Total Value": 961.50, "Shares": 5, "Price": 192.30, "Daily Change %": -0.36, "All Time Return": 402.29, "All Time %": 71.94},
    {"Symbol": "CRWV", "Company": "CoreWeave Inc.", "Total Value": 230.48, "Shares": 2, "Price": 115.24, "Daily Change %": -3.97, "All Time Return": -130.09, "All Time %": -36.08},
    {"Symbol": "CU", "Company": "Canadian Utilities Ltd. (Class A)", "Total Value": 1261.35, "Shares": 32.4755, "Price": 38.84, "Daily Change %": -0.31, "All Time Return": 168.11, "All Time %": 15.38},
    {"Symbol": "DCBO", "Company": "Docebo Inc", "Total Value": 301.84, "Shares": 7, "Price": 43.12, "Daily Change %": 0.79, "All Time Return": -194.11, "All Time %": -39.14},
    {"Symbol": "DRUG", "Company": "Bright Minds Biosciences Inc.", "Total Value": 481.00, "Shares": 10, "Price": 48.10, "Daily Change %": 9.05, "All Time Return": 96.30, "All Time %": 25.03},
    {"Symbol": "ENB", "Company": "Enbridge Inc", "Total Value": 666.92, "Shares": 10.8143, "Price": 61.67, "Daily Change %": -0.58, "All Time Return": 138.27, "All Time %": 26.15},
    {"Symbol": "GSI", "Company": "Gatekeeper Systems Inc.", "Total Value": 405.00, "Shares": 300, "Price": 1.35, "Daily Change %": 0.00, "All Time Return": 107.00, "All Time %": 35.91},
    {"Symbol": "ISRG", "Company": "Intuitive Surgical CDR", "Total Value": 356.07, "Shares": 13.4774, "Price": 26.42, "Daily Change %": 1.50, "All Time Return": -32.03, "All Time %": -8.25},
    {"Symbol": "JOBY", "Company": "Joby Aviation Inc", "Total Value": 800.10, "Shares": 44.2777, "Price": 18.07, "Daily Change %": 3.23, "All Time Return": 312.79, "All Time %": 64.19},
    {"Symbol": "NVDA", "Company": "Nvidia CDR", "Total Value": 2989.06, "Shares": 75.0265, "Price": 39.84, "Daily Change %": -0.08, "All Time Return": 2262.98, "All Time %": 311.67},
    {"Symbol": "MSFT", "Company": "Microsoft CDR", "Total Value": 1848.99, "Shares": 49.9998, "Price": 36.98, "Daily Change %": 0.41, "All Time Return": 544.64, "All Time %": 41.76},
    {"Symbol": "META", "Company": "Meta CDR", "Total Value": 1055.04, "Shares": 27.0523, "Price": 39.00, "Daily Change %": -0.38, "All Time Return": 140.05, "All Time %": 15.31},
    # Add remaining items similarly...
]

portfolio_df = pd.DataFrame(portfolio_data)

# --- Metrics ---
total_value = portfolio_df["Total Value"].sum()
total_return = portfolio_df["All Time Return"].sum()
total_pct = (total_return / (total_value - total_return)) * 100 if total_value - total_return != 0 else 0

st.metric("Total Portfolio Value", f"${total_value:,.2f}", f"{total_return:+,.2f} ({total_pct:+.2f}%)")

# --- Display Data ---
st.subheader("Portfolio Details")
st.dataframe(portfolio_df, use_container_width=True)

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(portfolio_df, values='Total Value', names='Symbol', title='Portfolio Allocation')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(portfolio_df, x='Symbol', y='All Time Return', color='All Time Return',
                     title='All-Time Gains/Losses', text_auto=True)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Daily Change Chart ---
st.subheader("Daily Performance (Top Movers)")
fig_change = px.bar(
    portfolio_df.sort_values(by='Daily Change %', ascending=False),
    x='Symbol', y='Daily Change %', color='Daily Change %', text='Daily Change %',
    title='Daily % Change'
)
st.plotly_chart(fig_change, use_container_width=True)
