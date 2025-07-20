"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Any, Literal

from langchain_core.messages import AnyMessage
from langchain.schema import Document
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import MessagesState
import operator
from pydantic import BaseModel, Field


@dataclass
class RewrittenMessage(BaseModel):
    """
    Represents a rewritten message with its original content and the reason for rewriting.
    This is used to track changes made to messages during processing.
    """
    rewritten_message: str = Field(
        description="The contextualized standalone message.")
    context_added: str = Field(
        description="Summary of what context was incorporated.")
    original_intent_preserved: bool = Field(
        description="Whether the original intent of the message was preserved.")


@dataclass
class Classification(BaseModel):
    classification: Literal['MEDICAL', 'GENERAL', 'MISCELLANEOUS'] = Field(
        description="Classification of the user's query, used to determine the route of the graph flow"
    )
    reasoning: str = Field(description="Reason for classification")


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
class ParentGraphState(MessagesState):
    """State for the parent graph, containing messages and additional context."""

# OFFLINE SOLUTION ONLY - OLLAMA
    # classification: Literal['MEDICAL', 'FILE', 'GENERAL', 'MISCELLANEOUS'] = Field(
    #     description="Classification of the user's query, used to determine the route of the graph flow"
    # )
    # rewritten_message: str = Field(
    #     description="Rewritten message with context and intent preservation")

# ONLINE SOLUTION - OPENAI
    rewritten_message: RewrittenMessage = Field(
        description="Rewritten message with context and intent preservation")
    classification: Classification = Field(
        description="Classification of the user's query, used to determine the route of the graph flow"
    )
    has_files: bool = Field(
        default=False,
        description="Indicates whether the user has uploaded files with their message."
    )
    analysis: MedicalAnalysis = Field(
        description="Analysis results"
    )
    attached_files: Annotated[List[Document], {
        "description": "List of files attached by the user"}] = field(default_factory=list)
    retrieved_documents: Annotated[List[Document], {
        "description": "List of retrieved documents"}] = field(default_factory=list)
    document_context: Annotated[str, {
        "description": "Contextual information from documents"}] = ""
    research_context: Annotated[str, {
        "description": "Contextual information from research"}] = ""
