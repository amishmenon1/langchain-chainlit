"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Any

from langchain_core.messages import AnyMessage
from langchain.schema import Document
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from langgraph.prebuilt.chat_agent_executor import AgentState
import operator
from pydantic import BaseModel, Field


@dataclass
class State(AgentState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    # is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    # Additional attributes can be added here as needed.
    attached_files: List[Document] = field(default_factory=list)
    ### DOC CHUNKS ###
    # extracted_documents: List[Document] = field(default_factory=list)
    ### MARKDOWN DOCS ###
    extracted_documents: Annotated[List[Document], {
        "description": "Extracted documents in the form of Markdown chunks"}] = field(default_factory=list)
    document_context: Annotated[str, {
        "description": "Contextual information from documents"}] = field(default="")

    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
