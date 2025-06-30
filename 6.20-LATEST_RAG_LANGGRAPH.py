from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain.schema.runnable import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import StrOutputParser, Document
from langchain.schema.runnable.config import RunnableConfig
from typing import cast, Annotated, List, Optional, Any

import chainlit as cl

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from message import update_message, new_message
from langchain import hub as prompts
from prompt_templates import MSG_CLASSIFIER_PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import operator
from typing_extensions import TypedDict
from templates.system.retriever import RETRIEVER_SYSTEM_PROMPT, RETRIEVER_STRUCTURED_OUTPUT
from templates.system.analyzer import ANALYZER_SYSTEM_PROMPT
from templates.system.formatter import FORMATTER_PROMPT_TEMPLATE, CONVERSATIONAL_FORMATTER_PROMPT
from utils.file import load_files_into_db,  retrieve_chunks
from pydantic import BaseModel, Field
import pprint
import os


#### RETRIEVER STRUCTURED OUTPUTS ####


class LabResult(BaseModel):
    test_name: str
    value: str
    reference_range: Optional[str]
    # interpretation: Optional[str]


class ImagingResult(BaseModel):
    modality: str
    date: Optional[str]
    findings: str


class MedicalDocumentSummary(BaseModel):
    document_title: str
    date: Optional[str]
    lab_results: List[LabResult]
    imaging_results: List[ImagingResult]
    other_notes: Optional[List[str]]

#### ANALYSIS STRUCTURED OUTPUTS ####


class MedicalAnalysis(BaseModel):
    key_findings: str = Field(description="Concise summary of what's found.")
    clinical_interpretation: str = Field(
        description="Medical meaning of the findings.")
    medical_breakdown: str = Field(
        description="Detailed reasoning, logic, and science.")
    recommendations: str = Field(description="Next steps, monitoring advice.")


class UserMessageState(MessagesState):
    document_context: Any
    retrieved_docs: List[Any]
    analysis: MedicalAnalysis
    previous_user_message: str


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
CHROMA_PATH = 'chroma'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "6.25-Chainlit-Langgraph-Assistant"

chat_history = []

default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
msg_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_llm = ChatOpenAI(model="gpt-4o-mini")
research_llm = ChatOpenAI(model="gpt-4o-mini")
analysis_llm = ChatOpenAI(model="gpt-4o-mini")
formatter_llm = ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def on_chat_start():

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # node 1 - classifier node
    def classify_message(state: UserMessageState):
        print(f"\n-----classify_message()-----\n")
        message = state["messages"][-1]

        prompt = ChatPromptTemplate.from_template(
            MSG_CLASSIFIER_PROMPT_TEMPLATE)

        chain = prompt | msg_classifier_llm
        result = chain.invoke({"message": message.content}
                              ).content.strip().lower()

        print(f"\n\nclassify output:\n {result}\n\n")
        return result

        # if result == "file":
        #     return True  # Proceed to RAG + Analyzer
        # elif result == "medical":
        #     return True  # Proceed to RAG + Analyzer
        # else:
        #     return False  # Go directly to generate()

    # node 1a - RAG node

    async def retrieve_docs(state: UserMessageState):
        print(f"\n-----retrieve_docs()-----\n")
        message = state["messages"][-1]

        print(f"message: {message.content}")

        retrieved_text, unique_sources = await retrieve_chunks(
            message_content=message.content)

        prompt = ChatPromptTemplate.from_template(RETRIEVER_SYSTEM_PROMPT)

        # chain = prompt | rag_llm | StrOutputParser()
        chain = prompt | rag_llm.with_structured_output(MedicalDocumentSummary)

        response = chain.invoke(
            {"message": message.content, "context": retrieved_text})

        pprint.pprint(
            f"\n\nretrieve_docs() output:\n {response.model_dump()}\n\n")

        if not response.lab_results and not response.imaging_results and not response.other_notes:
            print("⚠️ Empty document content. Skipping downstream analysis.")
            return {
                "document_context": None,
                "retrieved_docs": [],
            }

        return {
            "document_context": response,
            "retrieved_docs": retrieved_text,
        }

    # node 2a - researcher node

    def web_search(state: UserMessageState):
        print(f"web_search()")
        # TODO implement method using pub med and web search tools
        return state

    # node 3a - analyzer node

    def analyze(state: UserMessageState):
        print(f"\n-----analyze()-----\n")
        message = state["messages"][-1]
        document_context = state.get("document_context", None)

        prompt = ChatPromptTemplate.from_template(ANALYZER_SYSTEM_PROMPT)

        chain = prompt | analysis_llm.with_structured_output(MedicalAnalysis)
        response = chain.invoke(
            {"message": message.content, "context": document_context})

        pprint.pprint(f"\n\nanalyze() output:\n{response.model_dump()}\n\n")

        return {"analysis": response}

    # node 1b - default generator node

    def generate(state: UserMessageState):
        print("\n-----generate()-----\n")
        message = state["messages"][-1]
        analysis = state.get("analysis", None)

        # Retrieve the full chat history as a text block.
        chat_history_text = "\n".join(
            f"{m.__class__.__name__}: {m.content}" for m in state["messages"])

        # Extract the last user message (prior to current) if needed.
        previous_user_message = next(
            (m.content for m in reversed(
                state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        # Decide which prompt to use based on analysis content.
        # For example: if the analysis exists and the current message is a clear clinical inquiry, use structured.
        # Otherwise, use the conversational format.
        # You can refine this logic depending on your use case.
        use_conversational = True
        # Alternatively, if analysis is missing or empty, treat as conversational.
        # if analysis is None or not any(getattr(analysis, field, "").strip() for field in ["key_findings", "clinical_interpretation", "medical_breakdown", "recommendations"]):
        #     use_conversational = True

        if use_conversational:
            print("→ Using conversational formatter")
            prompt = ChatPromptTemplate.from_messages([
                ("system", CONVERSATIONAL_FORMATTER_PROMPT),
                ("human", "{message}")
            ])
            inputs = {
                "previous_user_message": previous_user_message,
                "chat_history": chat_history_text,
                "message": message.content,
                # can be None or partial; template will handle if not present.
                "analysis": analysis
            }
        else:
            print("→ Using clinical formatter")
            prompt = ChatPromptTemplate.from_messages([
                ("system", FORMATTER_PROMPT_TEMPLATE),
                ("human", "{message}")
            ])
            inputs = {
                "analysis": analysis,
                "message": message.content,
                # if your clinical prompt uses it.
                "chat_history": chat_history_text,
                "previous_user_message": previous_user_message
            }

        llm = prompt | formatter_llm | StrOutputParser()
        response = llm.invoke(inputs)
        return {"messages": [AIMessage(content=response)]}

    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("web_search", web_search)
    workflow.add_node("analyze", analyze)
    workflow.add_node("generate", generate)

    workflow.add_conditional_edges(START, classify_message, {
        # If medical question, use RAG and medical analysis
        "file": "retrieve_docs",
        # Otherwise use general model completion
        "medical": "analyze",
        "general": "generate"
    })

    workflow.add_edge("retrieve_docs", "web_search")
    workflow.add_edge("web_search", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    cl.user_session.set("app", app)


@cl.on_message
async def on_message(message: cl.Message):
    print(cl.chat_context.to_openai())
    # Check if files are attached to this message
    attached_files = None
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    cl.user_session.set("latest_attached_files", attached_files)
    # If files are attached, process them first
    if attached_files:
        chat_history.append(SystemMessage(
            content=f"Attachments: {len(attached_files)}"))
        sys_msg_1 = await new_message(
            content="📎 **Files detected!**")
        await load_files_into_db(attached_files)
    else:
        chat_history.append(SystemMessage(
            content=f"Attachments: {0}"))

    chat_history.append(HumanMessage(content=message.content))

    app = cast(Runnable, cl.user_session.get("app"))

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    answer = cl.Message(content="")

    async for msg, metadata in app.astream(
        # {"messages": chat_history},
        {"messages": [HumanMessage(content=message.content)]},

        config,
        stream_mode="messages",
    ):
        if (
            msg.content
            and isinstance(msg, AIMessageChunk)
            and metadata["langgraph_node"] == "generate"
        ):

            await answer.stream_token(msg.content)

    await answer.update()
