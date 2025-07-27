import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Initialize the Web Agent (DuckDuckGo for searching the web)
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Initialize the Finance Agent (Yahoo Finance for financial data)
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Streamlit Interface for User Input and Display
st.title("Web and Finance Agent App")
st.markdown("""
    This app allows you to search the web and analyze financial data using AI agents.
    You can ask questions or get stock market data.
""")

# User Input: Question or Stock Symbol
user_input = st.text_input("Ask a question or enter a stock symbol (e.g., AAPL):")

# Check if the user input is a stock symbol or a general query
if user_input:
    if user_input.isalpha():  # If it's a stock symbol (assuming alphabetic symbols for simplicity)
        # Run the Finance Agent to get stock info
        with st.spinner("Fetching stock data..."):
            try:
                finance_response = finance_agent.run(user_input)
                st.write(finance_response)
            except Exception as e:
                st.error(f"An error occurred while fetching stock data: {e}")
    else:
        # Run the Web Agent to get general information from the web
        with st.spinner("Searching the web..."):
            try:
                web_response = web_agent.run(user_input)
                st.write(web_response)
            except Exception as e:
                st.error(f"An error occurred while searching the web: {e}")

# Display the agents' outputs for specific buttons
if st.button("Run Web Agent for Stock News"):
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    if stock_symbol:
        with st.spinner(f"Fetching {stock_symbol} news..."):
            try:
                response = web_agent.run(f"{stock_symbol} stock news")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while fetching news: {e}")

if st.button("Run Finance Agent for Stock Information"):
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    if stock_symbol:
        with st.spinner(f"Fetching {stock_symbol} information..."):
            try:
                response = finance_agent.run(stock_symbol)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while fetching stock info: {e}")

# Optional: Add a markdown section explaining what each agent does
st.markdown("""
### Web Agent (DuckDuckGo)
The Web Agent searches the web for information based on the query you provide. You can use it to ask about general topics, companies, or any specific information.

### Finance Agent (Yahoo Finance)
The Finance Agent retrieves detailed stock market data such as stock prices, company info, recommendations, and the latest news. You can ask for any publicly traded company's stock data by providing the stock symbol (e.g., `AAPL` for Apple).
""")

