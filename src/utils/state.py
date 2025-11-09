from langchain.agents import AgentState
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages.utils import trim_messages
from langchain_core.messages import BaseMessage

def messages_reducer(existing, new):
    existing = trim_messages(
        existing,
        max_tokens=100,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=True
    )

    messages = add_messages(existing, new)

    return messages


class SupervisorState(AgentState):
    messages: Annotated[Sequence[BaseMessage], messages_reducer]
    remaining_steps: int = 0