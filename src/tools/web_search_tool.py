from tavily import TavilyClient

from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str, max_results: int = 5):
    """
    Search public information about a company for evaluation purposes.
    """

    response = tavily_client.search(query=query, max_results=max_results,
                                         include_images=False, topic="general", search_depth="basic")
    return response['results']

# if __name__ == "__main__":
#     test_query = "Firegroup employee reviews company culture work environment"

#     result = web_search.invoke(
#         {
#             "query": test_query,
#             "max_results": 5
#         }
#     )

#     print("=== Tavily search result ===")
#     print(result)