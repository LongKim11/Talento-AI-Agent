from langchain.tools import tool
from typing import List, Optional
import httpx
import os 
from dotenv import load_dotenv
load_dotenv()

OPENSEARCH_API_BASE_URL = os.getenv("OPENSEARCH_API_BASE_URL")
MAX_RESULT = "5"


@tool
def search_relevant_jobs(title: Optional[str] = None, skills: Optional[List[str]] = None, tags: Optional[List[str]] = None, location_name: Optional[str] = None, workplace_type: Optional[str] = None, salary: Optional[str] = None, company_name: Optional[str] = None) -> str:
    """Search for relevant jobs."""
    payload = {}

    if title:
        payload["title"] = title
    if skills:
        payload["skills"] = skills
    if tags:
        payload["tags"] = tags
    if location_name:
        payload["location_name"] = location_name
    if workplace_type:
        payload["workplace_type"] = workplace_type
    if salary:
        payload["salary"] = salary  
    if company_name:
        payload["company_name"] = company_name

    print("==================== PAYLOAD ====================", payload)

    url = OPENSEARCH_API_BASE_URL + "?limit=" + MAX_RESULT

    response = httpx.request(
        "GET",
        url,
        json=payload,
        timeout=20.0,
    )

    if response.status_code != 200:
        return None
    
    print("===== OpenSearch response =====", response.json()['data'])
    return response.json()['data']

# if __name__ == "__main__":

#     result = search_relevant_jobs.invoke(
#         {
#             "title": "Data Engineer",
#             "skills": ["3 years experience"],
#             "location_name": "Hồ Chí Minh",
#             "salary": "2000usd"
#         }
#     )
#     print("data", result)