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
You are an AI assistant that evaluates job relevance and explains it in a conversational, recommendation-style tone.

Your task is to score how well a job matches the user's question and briefly explain the key matching points and any missing aspects.

Scoring rules (VERY IMPORTANT):
- Score ranges from 1 to 10
- 10: Job strongly matches the user's intent, role, skills, and seniority
- 7-9: Job matches most requirements but misses minor aspects
- 4-6: Partial match (some relevant skills or role overlap)
- 1-3: Weak match (only vague or indirect relevance)

Evaluation criteria:
- Job title vs user intent
- Required skills vs mentioned skills
- Seniority / experience level
- Domain or industry relevance

Explanation rules:
- Write from the perspective of a chatbot giving a recommendation
- Do NOT assume information that is not explicitly in the job
- Keep the explanation concise (2-3 sentences)
- Use neutral, natural language
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
    - Analyze the user's question
    - Use the search_relevant_jobs tool to retrieve relevant job postings
    - Do NOT fabricate job data

    Output rules:
    - Return ONLY the tool result
    - Do not add explanations or comments
    - If no relevant jobs are found, return None
    """

    agent = create_agent(llm, tools=[search_relevant_jobs], system_prompt=system_prompt)

    response = agent.invoke({"messages": [{"role": "user", "content": question}]})    

    relevant_jobs = response["messages"][-1].content

    relevant_jobs = json.loads(relevant_jobs) if relevant_jobs is not None else None
   
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

    for job in relevant_jobs:
        result = retrieval_grader.invoke(
            {"question": question, "job": job}
        )

        job["score"] = result.score 
        job["score_description"] = result.score_description

    return {"relevant_jobs": relevant_jobs, "question": question}


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
    You are an AI assistant that evaluates how suitable and attractive a company is as a workplace.

    Your task:
    - Use web search to gather publicly available information about a company
    - Evaluate how suitable and attractive the company is as a workplace

    Evaluation criteria:
    - Company culture and values
    - Employee reviews and overall reputation
    - Work-life balance and management quality (if available)

    Rating rules:
    - Score ranges from 1 to 10
    - 9-10: Excellent workplace with strong culture and positive reviews
    - 6-8: Generally good workplace with minor concerns
    - 4-5: Mixed or average reputation
    - 1-3: Poor workplace or significant concerns

    Output requirements (STRICT):
    - Do NOT assume missing information
    - If no company information is found, assign a rating of 5 and state that information is insufficient
    - Use a neutral, chatbot-style recommendation tone
    - Be concise and factual (2-3 sentences)
    """

    agent = create_agent(llm, tools=[web_search], system_prompt=prompt, response_format=ToolStrategy(EvaluateCompany))

    for job in relevant_jobs:
        company_name = job["company_name"]

        response = agent.invoke({"messages": [{"role": "user", "content": f"Company name: {company_name}"}]})
        
        result = response["structured_response"]

        job["company_rating"] = result.company_rating
        job["company_rating_description"] = result.company_rating_description

    return {"relevant_jobs": relevant_jobs, "question": state["question"]}


def decide_to_grade(state):
    """
    Determines whether to grade the relevant jobs or companies.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    relevant_jobs = state["relevant_jobs"]

    if not relevant_jobs:
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
workflow.add_edge("retrieve", "grade_relevant_jobs")
workflow.add_edge("grade_relevant_jobs", "grade_companies")
workflow.add_edge("grade_companies", END)

# Compile
graph = workflow.compile(name="recommendation_agent")