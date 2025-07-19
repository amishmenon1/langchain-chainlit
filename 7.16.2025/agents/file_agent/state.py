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
class MedicalAnalysis(BaseModel):
    """
    Structured output model for the Analyzer Agent in a medical assistant chatbot system.

    This model encapsulates the agent's clinical reasoning and analysis of patient context,
    including symptom evaluation, risk assessment, and educational insight. It is designed 
    to help caregivers and patients understand medical concerns, identify red flags, and 
    determine when professional consultation or urgent care is needed.
    """

    medical_analysis: str = Field(
        description="Comprehensive analysis of the patient's medical condition and overall situation."
    )
    key_findings: str = Field(
        description="Important observations, symptoms, or patterns identified in the medical data."
    )
    risk_assessment: str = Field(
        description="Evaluation of the urgency, severity, and potential risks associated with the current medical condition."
    )
    immediate_action_needed: str = Field(
        description="Indicates whether immediate medical intervention or emergency care is recommended."
    )
    educational_insights: str = Field(
        description="Relevant medical explanations and insights intended to educate the caregiver or patient."
    )
    professional_consultation_recommended: str = Field(
        description="Highlights specific areas where a specialist or healthcare professional should be consulted."
    )
    safety_warnings: str = Field(
        description="Critical safety notes, contraindications, or activities/treatments to avoid for patient safety."
    )


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


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
    document_context: Annotated[str, {
        "description": "Contextual information from documents"}] = field(default_factory=list)
    analysis: MedicalAnalysis = field(default_factory=MedicalAnalysis)

    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
