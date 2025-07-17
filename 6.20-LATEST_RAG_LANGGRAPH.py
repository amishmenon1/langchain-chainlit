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
# from prompt_templates import MSG_CLASSIFIER_PROMPT_TEMPLATE, RETRIEVER_SYSTEM_PROMPT, ANALYZER_SYSTEM_PROMPT, CONVERSATIONAL_ANALYSIS_SYSTEM_PROMPT, CONVERSATIONAL_FORMATTER_PROMPT, MSG_REWRITER_SYSTEM_PROMPT, RETRIEVAL_GRADER_PROMPT_TEMPLATE2
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import operator
from typing_extensions import TypedDict
# from utils.file import load_files_into_db,  retrieve_chunks
# extract_docs_from_files_parallel
from utils.file import create_vectordb, add_files_to_db, get_all_patient_docs, extract_docs_from_files
from pydantic import BaseModel, Field
import pprint
import os
import asyncio
from templates.system.generated_rewriter import MSG_REWRITER_SYSTEM_PROMPT
from templates.system.generated_classifier import MSG_CLASSIFIER_PROMPT_TEMPLATE
from templates.system.generated_retriever import RETRIEVER_SYSTEM_PROMPT
from templates.system.generated_research import RESEARCH_SYSTEM_PROMPT
from templates.system.generated_analysis import CONVERSATIONAL_ANALYSIS_SYSTEM_PROMPT
from templates.system.generated_generate import CONVERSATIONAL_FORMATTER_PROMPT


#### RETRIEVER STRUCTURED OUTPUTS ####


class LabResult(BaseModel):
    test_name: str
    value: str
    reference_range: Optional[str]
    date: str
    # interpretation: Optional[str]


class ImagingResult(BaseModel):
    modality: str
    date: Optional[str]
    findings: str


# class MedicalDocumentSummary(BaseModel):
#     document_title: str
#     date: Optional[str]
#     lab_results: List[LabResult]
#     imaging_results: List[ImagingResult]
#     other_notes: Optional[List[str]]

class MedicalDocumentSummary(BaseModel):
    document_title: str
    report_date: str
    # lab_test_date: str
    lab_results: List[LabResult]
    # imaging_results: List[ImagingResult]
    other_notes: Optional[List[str]]

#### ANALYSIS STRUCTURED OUTPUTS ####


# class MedicalAnalysis(BaseModel):
#     key_findings: str = Field(description="Concise summary of what's found.")
#     clinical_interpretation: str = Field(
#         description="Medical meaning of the findings.")
#     medical_breakdown: str = Field(
#         description="Detailed reasoning, logic, and science.")
#     recommendations: str = Field(description="Next steps, monitoring advice.")

class MedicalAnalysis(BaseModel):
    key_findings: str = Field(
        description="Important observations and patterns identified")
    medical_analysis: str = Field(
        description="Comprehensive analysis of the medical situation")
    risk_assessment: str = Field(
        description="Assessment of urgency and risk factors")
    immediate_action_needed: str = Field(
        description="Whether immediate medical care is required")
    educational_insights: str = Field(
        description="Relevant medical education for the caregiver")
    professional_consultation_recommended: str = Field(
        description="Specific areas requiring professional input")
    safety_warnings: str = Field(
        description="Important safety considerations and contraindications")

#### CLASSIFICATION STRUCTURE ####
# class GradeQuestion(BaseModel):
#     score: str = Field(description="Is the question asking a clinical-related question, file-related question, or general question")


class GradeDocument(BaseModel):
    score: str = Field(
        description="Is the document relevant to the question? If yes -> 'Yes'. If not -> 'No'.")


# class RephrasedMessage(BaseModel):
#     rephrased_message: str = Field("")
#     asking_about_attachments: bool = Field("")

class RephrasedMessage(BaseModel):
    rewritten_message: str = Field(
        description="The contextualized standalone message")
    context_added: str = Field(
        description="Summary of what context was incorporated")
    original_intent_preserved: bool = Field(description="True | False")


class Classification(BaseModel):
    #     classification: str = Field("""
    #     - MEDICAL_COMPLEX: Medical questions, patient-specific queries, health concerns, symptoms, treatments, diagnoses
    #     - SIMPLE_GENERAL: Greetings, general conversation, non-medical questions, system queries
    #     - FILE_RELATED: Questions about uploaded files, requests to analyze documents, references to specific files
    # """)
    classification: str = Field(
        description="Classification of the user's message")
    has_files: bool = Field(
        description="Does the user's message contain attachments?")
    reasoning: str = Field(description="Reason for classification")

#### GRAPH STATE ####


class UserMessageState(MessagesState):
    # document_context: Annotated[List[Document], operator.add]  # Any
    document_context: str
    retrieved_docs: Annotated[List[Document], operator.add]
    analysis: MedicalAnalysis
    previous_user_message: str
    rephrased_question: str
    rephrase_count: int
    proceed: bool
    asking_about_attachments: bool
    latest_attachments: List[Document]  # override with latest attachments


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
CHROMA_PATH = 'chroma'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "6.25-Chainlit-Langgraph-Assistant"

default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
msg_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
research_llm = ChatOpenAI(model="gpt-4o-mini")
analysis_llm = ChatOpenAI(model="gpt-4o-mini")
formatter_llm = ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def on_chat_start():

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    def question_rewriter(state: UserMessageState):
        print(f"\n----question_rewriter()----\n")
        # state["retrieved_docs"] = []
        state["rephrased_question"] = ""
        state["proceed"] = False
        # # Extract the last user message (prior to current) if needed.
        previous_user_message = next(
            (m.content for m in reversed(
                state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        # if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["messages"][-1]

        latest_docs = cl.user_session.get("latest_attached_docs", [])
        latest_context = cl.user_session.get("document_context", [])
        document_context = state.get("document_context", None)

        rephrase_prompt = ChatPromptTemplate.from_template(
            MSG_REWRITER_SYSTEM_PROMPT)

        chain = rephrase_prompt | default_llm.with_structured_output(
            RephrasedMessage)

        response = chain.invoke(
            {"history": conversation, "context": document_context, "message": current_question, "previous_user_message": previous_user_message})
        better_question = response.rewritten_message.strip()
        print(f"\n\nrewriter response:\n{response}\n\n")
        return {
            "rephrased_question": better_question,
            "retrieved_docs": latest_docs,
            "document_context": latest_context
            # "asking_about_attachments": response.asking_about_attachments
            # "asking_about_attachments": response.asking_about_attachments
        }

    # node 1 - classifier node
    def classify_message(state: UserMessageState):
        print(f"\n-----classify_message()-----\n")
        print(
            f"\nRephrased question: {state.get("rephrased_question", None)}\n")

        # # Extract the last user message (prior to current) if needed.
        previous_user_message = next(
            (m.content for m in reversed(
                state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        # if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["messages"][-1]
        document_context = state.get("document_context", None)
        rephrased_message = state.get(
            "rephrased_question", current_question.content)

        print(
            f"\n\ncurrent question: {current_question}\ncontext:{document_context}\nmessage: {rephrased_message}\nprevious msg: {previous_user_message}\nhistory: {conversation[:200]}")

        prompt = ChatPromptTemplate.from_template(
            MSG_CLASSIFIER_PROMPT_TEMPLATE)

        chain = prompt | msg_classifier_llm.with_structured_output(
            Classification)
        # result = chain.invoke({"history": conversation, "context": document_context, "message": rephrased_message, "previous_user_message": previous_user_message}
        #                       )

        result = chain.invoke({"message": rephrased_message, "previous_user_message": previous_user_message}
                              )

        print(f"\n\nclassify output:\n {result}\n\n")
        return result.classification

    # node 1a - RAG node

    async def retrieve_docs(state: UserMessageState):
        print(f"\n-----retrieve_docs()-----\n")
        # add all med docs to context
        # else continue to analyze node
        message = state["messages"][-1]
        rephrased_message = state.get("rephrased_question", message.content)
        updated_document_context = []

        # if state.get('asking_about_attachments', False):
        print(f"\n\nuser is asking about attachments\n\n")
        if cl.user_session.get('has_attachments', False):
            print(f"\n\nFound attached files\n\n")
            updated_document_context = cl.user_session.get(
                'latest_attached_docs', []) + state.get('retrieved_docs', [])
        else:
            print(f"\n\nNo files attached. Fetching all...\n\n")
            updated_document_context = await get_all_patient_docs()
            # TODO return "please upload files that you have questions about"
        return {
            "document_context": updated_document_context,
            "latest_attachments": cl.user_session.get(
                'latest_attached_docs', [])
        }

    # node 2a - researcher node

    def web_search(state: UserMessageState):
        print(f"\n----web_search()----\n")
        # TODO implement method using pub med and web search tools
        return state

    # node 3a - analyzer node
    @cl.step(name="🧠 Medical analysis")
    async def analyze(state: UserMessageState):
        print(f"\n-----analyze()-----\n")
        message = state["messages"][-1]
        messages = state["messages"]
        document_context = state.get("document_context", None)
        # # Extract the last user message (prior to current) if needed.
        previous_user_message = next(
            (m.content for m in reversed(
                state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        print(f"\n\nuser message: {message.content}\n\n")

        # prompt = ChatPromptTemplate.from_template(ANALYZER_SYSTEM_PROMPT)
        prompt = ChatPromptTemplate.from_template(
            CONVERSATIONAL_ANALYSIS_SYSTEM_PROMPT)

        chain = prompt | analysis_llm.with_structured_output(MedicalAnalysis)
        # chain = prompt | analysis_llm | StrOutputParser()
        response = chain.invoke(
            {"message": message.content, "context": document_context, "history": messages, "previous_user_message": previous_user_message})

        pprint.pprint(f"\n\nanalyze() output:\n{response}\n\n")

        return {"analysis": response}
        # return {"messages": [AIMessage(content=response)]}

    # node 1b - default generator node

    def generate(state: UserMessageState):
        print("\n-----generate()-----\n")
        conversation = state["messages"][:-1]
        message = state["messages"][-1]
        analysis = state.get("analysis", None)
        document_context = state.get("document_context", None)

        # Retrieve the full chat history as a text block.
        # chat_history_text = "\n".join(
        #     f"{m.__class__.__name__}: {m.content}" for m in state["messages"])

        # # Extract the last user message (prior to current) if needed.
        previous_user_message = next(
            (m.content for m in reversed(
                state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        print(f"previous message: {previous_user_message}")
        prompt = ChatPromptTemplate.from_template(
            CONVERSATIONAL_FORMATTER_PROMPT)
        inputs = {
            "previous_user_message": previous_user_message,
            "history": conversation,
            "message": message.content,
            "context": document_context,
            # can be None or partial; template will handle if not present.
            "analysis": analysis
        }

        llm = prompt | formatter_llm | StrOutputParser()
        response = llm.invoke(inputs)
        return {"messages": [AIMessage(content=response)]}

    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("retrieve_docs", retrieve_docs)
    # workflow.add_node("retrieval_grader", retrieval_grader)
    # workflow.add_node("refine_question", refine_question)
    # workflow.add_node("proceed_router", proceed_router)
    # workflow.add_node("perform_rag_qa", perform_rag_qa)
    workflow.add_node("web_search", web_search)
    workflow.add_node("analyze", analyze)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "question_rewriter")
    workflow.add_conditional_edges("question_rewriter", classify_message, {
        # If medical question, use RAG and medical analysis
        "FILE_RELATED": "retrieve_docs",
        # "FILE_RELATED": "retrieve_docs",
        # Otherwise use general model completion
        "MEDICAL_COMPLEX": "analyze",
        "SIMPLE_GENERAL": "generate"
    })

    workflow.add_edge("retrieve_docs", "analyze")
    workflow.add_edge("analyze", "generate")
    # workflow.add_edge("analyze", END)
    workflow.add_edge("generate", END)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    cl.user_session.set("app", app)


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
        # extracted_docs, extracted_filenames = await extract_docs_from_files(attached_files)
        # extracted_filenames = []
        extracted_docs, formatted_context = await extract_docs_from_files(
            attached_files)
        cl.user_session.set("latest_attached_docs", extracted_docs)
        cl.user_session.set("document_context", formatted_context)
        asyncio.create_task(add_files_to_db(
            extracted_docs))
        # await add_files_to_db(attached_files)
    else:
        cl.user_session.set('has_attachments', False)
        cl.user_session.set('latest_attached_docs', None)

    app = cast(Runnable, cl.user_session.get("app"))

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    answer = cl.Message(content="")

    async for msg, metadata in app.astream(
        {"messages": [HumanMessage(content=message.content)]},

        config,
        stream_mode="messages",
    ):
        if (
            msg.content
            and isinstance(msg, AIMessageChunk)
            and (metadata["langgraph_node"] in ["generate",
                                                #  "analyze"
                                                ])
        ):

            await answer.stream_token(msg.content)

    await answer.update()
