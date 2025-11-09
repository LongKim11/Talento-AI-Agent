from tavily import AsyncTavilyClient

async def web_search(query: str, max_results: int = 5):
    client = AsyncTavilyClient()
    search_results = await client.search(query=query, max_results=max_results,
                                         include_images=True, topic="general", search_depth="basic")
    return [result.url for result in search_results]

