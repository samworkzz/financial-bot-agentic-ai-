from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()


phi.api=os.getenv("PHI_API_KEY")

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

app=Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)