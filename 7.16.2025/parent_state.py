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
