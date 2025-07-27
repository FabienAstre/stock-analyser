import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Setup Web Agent (DuckDuckGo for web searches)
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    storage=SqliteAgentStorage(table_name="web_agent_data", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Setup Finance Agent (Yahoo Finance for financial data)
finance_agent = Agent(
    name="Finance Agent",
    role="Retrieve financial data for stock analysis",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display stock data clearly."],
    storage=SqliteAgentStorage(table_name="finance_agent_data", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Streamlit App Interface
st.title("Web and Finance Agent Dashboard")
st.markdown("""
    **Welcome to the Web and Finance Agent Dashboard**!  
    Here, you can use two AI agents to interact with both **financial data** and **web searches**:
    - **Web Agent**: Search the web for any information you need.
    - **Finance Agent**: Get real-time stock market data like stock prices, company info, and news.
""")

# User Input: Ask a question or provide a stock symbol
user_input = st.text_input("Enter your query or a stock symbol (e.g., AAPL for Apple):")

# Handling User Input: Search Web or Fetch Stock Data
if user_input:
    # If it's a stock symbol, fetch financial data
    if user_input.isalpha():  # Check if it's likely a stock symbol (alphabetic)
        with st.spinner(f"Fetching stock data for {user_input}..."):
            try:
                finance_response = finance_agent.run(user_input)
                st.write(finance_response)
            except Exception as e:
                st.error(f"Failed to retrieve stock data: {e}")
    else:
        # Otherwise, search the web for the query
        with st.spinner(f"Searching the web for '{user_input}'..."):
            try:
                web_response = web_agent.run(user_input)
                st.write(web_response)
            except Exception as e:
                st.error(f"Error searching the web: {e}")

# Buttons for fetching specific data or news
if st.button("Fetch Stock News"):
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    if stock_symbol:
        with st.spinner(f"Fetching latest news for {stock_symbol}..."):
            try:
                news_response = web_agent.run(f"{stock_symbol} stock news")
                st.write(news_response)
            except Exception as e:
                st.error(f"Error fetching stock news: {e}")

if st.button("Get Stock Information"):
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    if stock_symbol:
        with st.spinner(f"Fetching stock details for {stock_symbol}..."):
            try:
                stock_response = finance_agent.run(stock_symbol)
                st.write(stock_response)
            except Exception as e:
                st.error(f"Error fetching stock info: {e}")

# Informational Section on Agents
st.markdown("""
### How It Works:
- **Web Agent**: Uses DuckDuckGo to search for information based on your queries. It's perfect for general knowledge, product info, or even company details.
- **Finance Agent**: Fetches financial data from Yahoo Finance. You can enter stock symbols (like `AAPL`) to get real-time stock prices, market news, and analyst recommendations.

### Example Queries:
- "What is the latest news on Tesla?"
- "Stock price of Apple (AAPL)"
- "How is Microsoft doing in the market?"

""")
