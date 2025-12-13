from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for relevant jobs."""
    return f"Results for: {query}"