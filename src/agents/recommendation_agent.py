from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import json

from src.tools.job_retriever_tool import search_relevant_jobs
from src.tools.web_search_tool import web_search

from concurrent.futures import ThreadPoolExecutor, as_completed

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Data model
class ScoreRelevantJob(BaseModel):
    """Relevance score between user question and a job."""

    score: int = Field(
        ge=1,
        le=10,
        description="Overall relevance score from 1 (not relevant) to 10 (highly relevant)."
    )

    score_description: str = Field(
        description="Brief explanation of how well the job matches the user question."
    )


class EvaluateCompany(BaseModel):
    """Evaluation of how suitable and attractive a company is as a workplace."""

    company_rating: int = Field(
        ge=1,
        le=10,
        description=(
            "Overall company rating from 1 (poor workplace) to 10 (excellent workplace), based on factors such as company culture, employee review and reputation."
        )
    )

    company_rating_description: str = Field(
        description=(
            "Brief explanation justifying the company rating, highlighting positive aspects of the company as well as any notable concerns."
        )
    )

structured_llm_retrieval_grader = llm.with_structured_output(ScoreRelevantJob)

# LLM scores relevant jobs to question
job_score_prompt = """
You are an expert career advisor assessing whether a job posting matches or is better-than-expected relative to a user's question.

Your task is to assign a relevance score and provide a short, natural recommendation explaining the overall fit.

Scoring rules (1-10)
- 10: Strong match with the user's intent, role, skills, and seniority
- 7-9: Matches most requirements
- 4-6: Partial match with some relevant skills or role overlap
- 1-3: Weak match with only indirect or minimal relevance

Evaluation criteria:
- Job title alignment with user intent
- Required skills alignment with mentioned skills
- Seniority and experience level match
- Domain or industry relevance

Explanation rules:
- Write in a natural, human-like advisory tone, as if recommending the job to a candidate
- Do NOT assume information that is not explicitly in the job
- Keep the explanation concise (2-3 sentences)

Output language (STRICT): Vietnamese only
"""

job_score_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", job_score_prompt),
        ("human", "Relevant job: \n\n {job} \n\n User question: {question}"),
    ]
)

retrieval_grader = job_score_prompt_template | structured_llm_retrieval_grader


### Nodes

def retrieve(state):
    """
    Retrieve relevant jobs based on user question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with relevant jobs
    """

    question = state["question"]

    system_prompt = """
    You are a job retrieval agent.

    Your task:
    - Analyze the user's question to identify key criteria for relevant job postings.
    - If location is mentioned, only use the city-level location (e.g. TP. Hồ Chí Minh, Hà Nội,..)
    - Use the search_relevant_jobs tool to find job postings that match these criteria.
    - Do NOT fabricate job data

    Output rules:
    - Return ONLY the tool result
    - Do not add explanations or comments
    
    Output language (STRICT): Vietnamese
    """

    agent = create_agent(llm, tools=[search_relevant_jobs], system_prompt=system_prompt)

    response = agent.invoke({"messages": [{"role": "user", "content": question}]})    

    content = response["messages"][-1].content
   
    relevant_jobs = json.loads(content) if content else None

    print("===== Retrieved relevant jobs =====", relevant_jobs)
    return {"relevant_jobs": relevant_jobs, "question": question}


def grade_relevant_jobs(state):
    """
    Score each relevant job against the user question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with scored jobs
    """

    question = state["question"]
    relevant_jobs = state["relevant_jobs"]

    def grade_single_job(job):
        result = retrieval_grader.invoke(
            {"question": question, "job": job}
        )
        job["score"] = result.score
        job["score_description"] = result.score_description
        return job

    with ThreadPoolExecutor(max_workers=min(10, len(relevant_jobs))) as executor:
        graded_jobs = list(executor.map(grade_single_job, relevant_jobs))

    return {"relevant_jobs": graded_jobs, "question": question}


def grade_companies(state):
    """
    Rate the companies of relevant jobs.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates relavent jobs with company rating
    """

    relevant_jobs = state["relevant_jobs"]

    prompt = """
    You are an expert in workplace evaluation, assessing how attractive and suitable a company is for employees.

    Your task:
    - Use the web search tool to gather information about a company.
    - Evaluate how suitable and attractive the company is as a workplace

    Evaluation criteria:
    - Company culture and values
    - Employee reviews and overall reputation
    - Work-life balance and management quality (if available)

    Rating rules (1-10): 
    - 9-10: Excellent workplace with a strong culture and consistently positive reviews
    - 6-8: Generally good workplace with minor concerns
    - 4-5: Mixed or average reputation
    - 1-3: Poor workplace or significant concerns

    Output requirements (STRICT):
    - Do not assume or infer missing information
    - If no reliable company information is found, assign a rating of 5 and clearly state that available information is insufficient
    - Use a neutral, professional recommendation tone
    - Keep the explanation concise (2-3 sentences)

    Output language (STRICT): Vietnamese only
    """

    agent = create_agent(llm, tools=[web_search], system_prompt=prompt, response_format=ToolStrategy(EvaluateCompany))

    def grade_single_company(job):
        company_name = job["company_name"]

        response = agent.invoke({"messages": [{"role": "user", "content": f"Company name: {company_name}"}]})
        
        result = response["structured_response"]

        job["company_rating"] = result.company_rating
        job["company_rating_description"] = result.company_rating_description
        
        return job

    with ThreadPoolExecutor(max_workers=min(10, len(relevant_jobs))) as executor:
        graded_jobs = list(executor.map(grade_single_company, relevant_jobs))

    return {"relevant_jobs": graded_jobs, "question": state["question"]}


def decide_to_grade(state):
    """
    Determines whether to grade the relevant jobs or companies.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    relevant_jobs = state["relevant_jobs"]

    if not relevant_jobs or len(relevant_jobs) == 0:
        return "end"
    else:
        return "grade"


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        relevant_jobs: list of relevants job
    """

    question: str
    relevant_jobs: List[str]


workflow = StateGraph(state_schema=GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_relevant_jobs", grade_relevant_jobs)  # grade relevant jobs
workflow.add_node("grade_companies", grade_companies)  # grade companies

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    decide_to_grade,
    {
        "end": END,
        "grade": "grade_relevant_jobs",
    },
)
workflow.add_edge("grade_relevant_jobs", "grade_companies")
workflow.add_edge("grade_companies", END)

# Compile
graph = workflow.compile(name="recommendation_agent")