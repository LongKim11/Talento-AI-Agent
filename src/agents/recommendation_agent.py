from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.tools.job_retriever_tool import search

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Data model
class ScoreRelevantJob(BaseModel):
    """Relevance score between user question and a job."""

    score: int = Field(
        ge=0,
        le=10,
        description="Overall relevance score from 0 (not relevant) to 10 (highly relevant)."
    )

    score_description: str = Field(
        description="A brief description explaining which aspects of the job match the user question and which aspects do not."
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
Job score prompt
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
        state (dict): New key added to state, documents, that contains relevent jobs
    """

    question = state["question"]

    prompt = ""

    agent = create_agent(llm, tools=[search], prompt=prompt)

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})    

    return {"relevant_jobs": result, "question": question}


def grade_relevant_jobs(state):
    """
    Score the relevant jobs to the user question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates relavent jobs with scores
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

    prompt = ""

    agent = create_agent(llm, tools=[search], prompt=prompt, response_format=ToolStrategy(EvaluateCompany))

    for job in relevant_jobs:
        result = agent.invoke({"messages": [{"role": "user", "content": ""}]})
        job["company_rating"] = result.company_rating
        job["company_rating_description"] = result.company_rating_description

    return {"relevant_jobs": relevant_jobs, "question": state["question"]}


def decide_to_grade(state):
    """
    Determines whether to grade the relevant jobs or companies.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
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