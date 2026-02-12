from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

def get_web_tools():
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        raise ValueError("❌ Missing TAVILY_API_KEY in .env file")

    search_tool = TavilySearchResults(
        max_results=3,
        api_key=tavily_key
    )
    return [search_tool]
