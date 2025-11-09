from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI

from src.agents.support_agent import graph as support_agent
from src.utils.state import SupervisorState

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

supervisor_prompt = """You are a chatbot of a job search platform that helps users with their inquiries. \n
    For company products and services, use support_agent. \n
    For job recommendations, use recommendation_agent. \n
    If the user query lacks sufficient information to determine which agent to use, ask clarifying questions to gather more details. \n
    Important: Just return the final answer from the selected agent without additional commentary. \n
"""

workflow = create_supervisor(
    [support_agent],
    model=model,
    supervisor_name="supervisor",
    prompt=supervisor_prompt,
    state_schema=SupervisorState,
    add_handoff_back_messages=False,
)

graph = workflow.compile(name="supervisor")
