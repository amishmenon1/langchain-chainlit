from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import os
from str_utils import pretty_print_messages

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, AIMessageChunk, SystemMessage, BaseMessage
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast, List, Optional
import rag_workflow
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from templates.agent_prompts.analysis import ANALYSIS_PROMPT
from templates.agent_prompts.generation import GENERATION_PROMPT
from templates.agent_prompts.research import RESEARCH_PROMPT
import chainlit as cl
import asyncio
from langgraph.prebuilt.chat_agent_executor import AgentState


memory = InMemorySaver()

# from IPython.display import display, Image

load_dotenv()


class QueryClassification(BaseModel):
    """Classify query using a binary score to determine if all documents should be retrieved."""
    retrieve_all: bool = Field(
        description="'True' if all documents should be retrieved, 'False' if only specific documents are needed based on the query."
    )


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


class CustomState(AgentState):
    document_context: str


class AnalysisState(AgentState):
    structured_response: Optional[MedicalAnalysis]
    document_context: str


def research_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    document_context = config["configurable"].get("document_context")
    system_msg = RESEARCH_PROMPT.format(document_context=document_context)
    return [{"role": "system", "content": system_msg}] + state["messages"]


def generation_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    document_context = config["configurable"].get("document_context")
    system_msg = GENERATION_PROMPT.format(document_context=document_context)
    return [{"role": "system", "content": system_msg}] + state["messages"]


def analysis_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    document_context = config["configurable"].get("document_context")
    # print(f"\n\nDocument context: {document_context}\n\n")
    system_msg = ANALYSIS_PROMPT.format(document_context=document_context)
    # print(f"\n\nSystem message: {system_msg}\n\n")
    return [{"role": "system", "content": system_msg}] + state["messages"]


@cl.step(name="🔎 Web Search")
async def web_search(query: str) -> List[BaseMessage]:
    """Search the web for a given query and return the results."""
    # initialize Tavily search tool
    response = TavilySearch(max_results=3).invoke(query)
    print(f"Web search results for query '{query}': {response['results']}")
    return response["results"]

# create custom math tools


@cl.step(name="🧠 Addition")
async def add(a: float, b: float):
    """Add two numbers."""
    return a + b


@cl.step(name="🧠 Multiplication")
async def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


@cl.step(name="🧠 Division")
async def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


analysis_agent = create_react_agent(
    model=default_llm,
    tools=[],
    state_schema=AnalysisState,
    prompt=analysis_prompt,
    # prompt=ANALYSIS_PROMPT,
    name="analysis_agent",
    response_format=MedicalAnalysis,
    checkpointer=memory,
)

math_agent = create_react_agent(
    model=default_llm,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
    checkpointer=memory,
)

research_agent = create_react_agent(
    model=default_llm,
    tools=[web_search],
    state_schema=CustomState,
    prompt=research_prompt,
    name="research_agent",
    checkpointer=memory,
)


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat with a welcome message."""
    ###### AGENTS ######
    document_context = cl.user_session.get("document_context", None)
    # print(f"\n\nDocument context: {document_context}\n\n")

    # Create supervisor workflow
    supervisor = create_supervisor(
        model=default_llm,
        agents=[research_agent,
                math_agent,
                analysis_agent
                ],
        state_schema=CustomState,
        prompt=generation_prompt,
        # prompt=GENERATION_PROMPT,

        add_handoff_back_messages=True,
        output_mode="full_history",

    ).compile(checkpointer=memory)

    await cl.Message(
        content="Welcome to the team chat! How can I assist you today?"
    ).send()

    cl.user_session.set("app", supervisor)


@cl.on_message
async def on_message(message: cl.Message):
    print(f"\n\nChat context:\n{cl.chat_context.to_openai()}\n\n")
    # Check if files are attached to this message
    attached_files = None
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    # If files are attached, process them first
    if attached_files:
        cl.user_session.set('has_attachments', True)
        existing_filenames = cl.user_session.get('filenames', [])
        new_files = [
            f for f in attached_files if f.name not in existing_filenames]
        if len(new_files) > 0:
            document_context, filenames = await rag_workflow.run_workflow(messages=[HumanMessage(content=message.content)], files=attached_files)
            print(f"\n\nExtracted docs: {document_context}\n\n")
            cl.user_session.set("document_context", document_context)
            cl.user_session.set('filenames', filenames)

    else:
        cl.user_session.set('has_attachments', False)
        cl.user_session.set('latest_attached_docs', None)

    app = cast(Runnable, cl.user_session.get("app"))
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id,
                         "document_context": cl.user_session.get("document_context", None)},
    }

    answer = cl.Message(content="")

    # tool_msg = cl.Message(content="")

##### streaming messages #####

    async for msg, metadata in app.astream(
        {"messages":  [HumanMessage(content=message.content)],
         #  "document_context": cl.user_session.get("document_context")
         },
        config,
        stream_mode="messages",
    ):
        print(f"\n\nMessage: {msg}\nMetadata: {metadata}\n\n")
        if (
            msg.content
            # and isinstance(msg, AIMessage)
            and metadata["langgraph_node"] == "supervisor"
        ):
            await answer.stream_token(msg.content)

    await answer.update()

    # answer = cl.Message(content="")
    # final_response_started = False

    # async for chunk in app.astream(
    #     {"messages": [HumanMessage(content=message.content)]},
    #     config,
    #     stream_mode="updates",  # switch from "messages"
    # ):
    #     # print(f"chunk: {chunk}")
    #     for node, update in chunk.items():
    #         print(f"Update: {update}")
    #         for msg in update.get("messages", []):
    #             # ✅ If streaming from supervisor
    #             if msg.name == "supervisor" and isinstance(msg, (AIMessage, AIMessageChunk)):
    #                 if isinstance(msg, AIMessageChunk):
    #                     final_response_started = True
    #                     await answer.stream_token(msg.content or "")
    #                 elif isinstance(msg, AIMessage):
    #                     final_response_started = True
    #                     await answer.stream_token(msg.content)

    # if final_response_started:
    #     await answer.update()
    # else:
    #     await cl.Message(content="Response complete, but nothing was streamed.").send()
