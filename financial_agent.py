from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


from phi.tools.duckduckgo import DuckDuckGo


# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",  
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Ensure this model ID is valid for Groq
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
)

# Financial agent
finance_agent = Agent(
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Use tables to display the data."],
)

# Multi-agent application
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Ensure this model ID is valid
    instructions=[
        "Use the web search agent to find recent company news.",
        "Use the financial agent to fetch stock prices and analyst recommendations.",
        "Display data in a table format."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Generate response
multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA",
    stream=True
)
