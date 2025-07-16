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
from typing import cast, Annotated, List, Optional, Any, Literal

import chainlit as cl

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from message import update_message, new_message
from langchain import hub as prompts
from prompt_templates import MSG_CLASSIFIER_PROMPT_TEMPLATE, RAG_CLASSIFIER_PROMPT_TEMPLATE, MEDICAL_ANALYSIS_PROMPT_TEMPLATE, FORMATTER_PROMPT_TEMPLATE, RETRIEVER_SYSTEM_TEMPLATE
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import operator
from typing_extensions import TypedDict
from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE_2
from utils.file import load_files_into_db,  retrieve_chunks

from retriever_tool import retrieve_patient_documents
from langgraph.prebuilt import ToolNode


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
CHROMA_PATH = 'chroma'

chat_history = []

default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
msg_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_llm = ChatOpenAI(model="gpt-4o-mini")
research_llm = ChatOpenAI(model="gpt-4o-mini")
analysis_llm = ChatOpenAI(model="gpt-4o-mini")
formatter_llm = ChatOpenAI(model="gpt-4o-mini")


class UserMessageState(MessagesState):
    document_context: Any
    retrieved_docs: List[Any]
    analysis: SystemMessage
    is_medical_question: bool


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


async def load_files_into_vectordb(files=[]):
    print(f"Creating RAG chain...")
    # Process the attached files

    all_pdf_pages = []
    processed_filenames = []
    all_doc_chunks = []
    if len(files) > 0:
        for file in files:
            print(f"Loading attached file: {file.name}")
            try:
                loader = PyPDFLoader(file.path)
                pdf_pages = []
                for page in loader.lazy_load():
                    page.metadata['source_file'] = file.name
                    pdf_pages.append(page)

                all_pdf_pages.extend(pdf_pages)
                processed_filenames.append(file.name)
                print(
                    f"Successfully processed: {file.name} ({len(pdf_pages)} pages)")

            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")

    if all_pdf_pages:
        # Split the text into chunks
        all_doc_chunks = text_splitter.split_documents(all_pdf_pages)
        print(f"Length of all text splits: {len(all_doc_chunks)}")

    # Create vector store with all documents
    vector_store = Chroma.from_documents(
        documents=all_doc_chunks, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    existing_chunks = cl.user_session.get("extracted_data", [])
    cl.user_session.set("last_attached_docs", all_doc_chunks)
    cl.user_session.set("extracted_data", existing_chunks + all_doc_chunks)
    cl.user_session.set("vector_store", vector_store)
    print(f"Successfully saved vector store to session")
    return vector_store


@cl.on_chat_start
async def on_chat_start():
    retrieval_tools = [retrieve_patient_documents]
    tool_node = ToolNode(tools=retrieval_tools)

    rag_llm_with_tools = rag_llm.bind_tools(retrieval_tools)
    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # node 1 - classifier node
    def classify_message(state: UserMessageState):
        print(f"classify_message()")
        message = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    MSG_CLASSIFIER_PROMPT_TEMPLATE,
                ),
                ("human", "{message}"),
            ]
        )
        llm = prompt | msg_classifier_llm
        response = llm.invoke({"message": message})
        is_med_question = response.content

        if is_med_question:
            print(f"User is asking a medical question")
            return True

        else:
            print(f"User is asking a general question")
            return False

    # node 1a - RAG node

    def call_rag_or_tool(state: UserMessageState):
        print("🧠 call_rag_or_tool")
        messages = state["messages"]
        response = rag_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def extract_retrieval_result(state: UserMessageState):
        print("🧪 extract_retrieval_result")
        last_message = state["messages"][-1]
        tool_output = last_message.content
        stored_chunks = cl.user_session.get("extracted_data", [])
        # Do not append another message here — just return doc context and docs
        return {
            "document_context": tool_output,
            "retrieved_docs": stored_chunks[:5],  # optional if needed
        }

    # node 2a - researcher node

    def web_search(state: UserMessageState):
        print(f"web_search()")
        # TODO implement method using pub med and web search tools
        return state

    # node 3a - analyzer node

    def analyze(state: UserMessageState):
        print(f"analyze()")
        message = state["messages"][-1]
        document_context = state.get("document_context", [])
        retrieved_docs = state.get("retrieved_docs", [])
        print(f"document context: {document_context}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    MEDICAL_ANALYSIS_PROMPT_TEMPLATE,
                ),
                ("human", "{message}"),
            ]
        )
        llm = prompt | analysis_llm
        response = llm.invoke(
            # {"message": message, "context": format_docs(retrieved_docs), }
            {"message": message, "context": document_context, }
        )

        return {
            "analysis": SystemMessage(content=response.content),
            "messages": [SystemMessage(content=response.content)]
        }

    # node 1b - default generator node

    def generate(state: UserMessageState):
        print(f"generate()")
        analysis = state.get("analysis", None)
        # analysis = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    FORMATTER_PROMPT_TEMPLATE,
                ),
                # ("human", "{message}"),
            ]
        )
        llm = prompt | formatter_llm | StrOutputParser()
        response = llm.invoke({"message": analysis.content}
                              )

        return {"messages": [AIMessage(content=response)]}

    def check_if_tool_called(state: UserMessageState) -> Literal["tool_node", "analyze"]:
        last = state["messages"][-1]
        return "tool_node" if getattr(last, "tool_calls", None) else "analyze"

    workflow.add_node("web_search", web_search)
    workflow.add_node("analyze", analyze)
    workflow.add_node("generate", generate)

    workflow.add_node("retrieve_or_tool", call_rag_or_tool)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("parse_tool_output", extract_retrieval_result)

    workflow.add_conditional_edges("retrieve_or_tool", check_if_tool_called, {
        "tool_node": "tool_node",
        "analyze": "analyze"
    })

    workflow.add_edge("tool_node", "parse_tool_output")
    workflow.add_edge("parse_tool_output", "analyze")

    workflow.add_conditional_edges(START, classify_message, {
        # If medical question, use RAG and medical analysis
        True: "retrieve_or_tool",
        # Otherwise use general model completion
        False: "generate",
    })

    # workflow.add_edge("retrieve_docs", "web_search")
    workflow.add_edge("web_search", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    cl.user_session.set("app", app)


@cl.on_message
async def on_message(message: cl.Message):
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

    answer = cl.Message(content="")
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    async for msg, metadata in app.astream(
        {"messages": chat_history},
        config,
        stream_mode="messages",
    ):
        if (
            msg.content
            and isinstance(msg, AIMessage)
            and metadata["langgraph_node"] == "generate"
        ):
            answer.content += msg.content  # type: ignore
            await answer.stream_token(msg.content)

    await answer.update()
