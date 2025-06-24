import os
import uuid
import logging

from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain.schema.runnable import Runnable
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import chainlit as cl
from dotenv import load_dotenv
from langsmith import Client
import langsmith
from typing import cast
from chainlit import make_async

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHROMA_PATH = "./chroma_data"

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Chainlit-RAG-Assistant"

# Langchain models and tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

### VECTOR STORE LOADER ###


async def load_files_into_vectordb(files=[]):
    logger.info("Starting to load files into vector DB")
    all_pdf_pages = []
    processed_filenames = []

    for file in files:
        try:
            logger.info(f"Loading: {file.name}")
            loader = PyPDFLoader(file.path)
            pdf_pages = []
            for page in loader.lazy_load():
                page.metadata['source_file'] = file.name
                pdf_pages.append(page)

            all_pdf_pages.extend(pdf_pages)
            processed_filenames.append(file.name)
            logger.info(
                f"Successfully processed {file.name} ({len(pdf_pages)} pages)")

        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")

    if not all_pdf_pages:
        raise ValueError("No PDF pages to embed")

    chunks = text_splitter.split_documents(all_pdf_pages)
    logger.info(f"Split into {len(chunks)} chunks")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    logger.info("Vector DB successfully created and persisted")
    return vectordb

### DYNAMIC RETRIEVER TOOL ###


def get_pdf_retriever_tool():
    vectordb = cl.user_session.get("vectordb")
    if not vectordb:
        logger.warning("No vector DB found in session")
        return None

    retriever = vectordb.as_retriever()
    return create_retriever_tool(
        retriever,
        name="pdf_retriever",
        description="Retrieve information from uploaded PDF documents."
    )

### GRAPH NODES ###


def generate_query_or_respond(state: State):
    logger.info("Generating query or response with tools")
    retriever_tool = get_pdf_retriever_tool()
    logger.info(f"retriever tool: {retriever_tool}")
    tools = [retriever_tool] if retriever_tool else []

    response = llm.bind_tools(tools).invoke(state["messages"])
    logger.info(f"LLM response generated: {response.content}")
    return {"messages": [response]}


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' otherwise")


GRADE_PROMPT = """You are a grader assessing document relevance.\nDocument:\n{context}\n\nUser question:\n{question}\n\nIs this document relevant? Respond only with 'yes' or 'no'.\n"""


def grade_documents(state: State) -> Literal["generate_answer", "chatbot"]:
    logger.info("Grading document relevance")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)

    response = llm.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    logger.info(f"Grading result: {response.binary_score}")
    return "generate_answer" if response.binary_score == "yes" else "chatbot"


ANSWER_PROMPT = """Use the following context to answer the question.\nIf the answer is not contained, say you don't know.\n\nQuestion: {question}\nContext: {context}\nAnswer:"""


def generate_answer(state: State):
    logger.info("Generating final answer")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = ANSWER_PROMPT.format(question=question, context=context)

    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def chatbot(state: State):
    logger.info("Running fallback chatbot node")
    return {"messages": [llm.invoke(state["messages"])]}

### REBUILDABLE GRAPH ###


def rebuild_graph():
    logger.info("Rebuilding graph dynamically")
    graph_builder = StateGraph(State)
    graph_builder.add_node("generate_query_or_respond",
                           generate_query_or_respond)
    graph_builder.add_node("generate_answer", generate_answer)
    graph_builder.add_node("chatbot", chatbot)

    retriever_tool = get_pdf_retriever_tool()
    if retriever_tool:
        logger.info("Including PDF retriever node")
        graph_builder.add_node("pdf_retrieve", ToolNode([retriever_tool]))

    graph_builder.add_edge(START, "generate_query_or_respond")
    graph_builder.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "pdf_retrieve" if retriever_tool else "chatbot",
            END: END,
        },
    )

    if retriever_tool:
        graph_builder.add_conditional_edges(
            "pdf_retrieve",
            grade_documents,
            {
                "generate_answer": "generate_answer",
                "chatbot": "chatbot",
            },
        )
        graph_builder.add_edge("generate_answer", END)
        graph_builder.add_edge("chatbot", END)
    else:
        graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile(checkpointer=memory)
    cl.user_session.set("graph", graph)
    logger.info("Graph compiled and saved to session")

### CHAINLIT HOOKS ###


@cl.on_chat_start
async def start():
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "metadata": {
                "conversation_id": thread_id,
                "client_type": "chainlit"
            }
        }
    }
    checkpoint = memory.get(config) or {}
    cl.user_session.set("messages", checkpoint.get("messages", []))
    rebuild_graph()


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {
        "configurable": {
            "thread_id": thread_id,
            "metadata": {
                "conversation_id": thread_id,
                "message_id": str(uuid.uuid4()),
                "client_type": "chainlit"
            }
        }
    }

    pdf_files = [
        elem for elem in message.elements or []
        if hasattr(elem, "path") and elem.path.endswith(".pdf")
    ]

    # Prepare the message content
    user_message_content = message.content

    if pdf_files:
        cl.user_session.set("latest_attached_files", pdf_files)
        await cl.Message(content=f"📎 **Detected {len(pdf_files)} PDF file(s). Loading...**").send()

        try:
            vectordb = await load_files_into_vectordb(pdf_files)
            cl.user_session.set("vectordb", vectordb)
            await cl.Message(content="✅ **PDFs successfully embedded. Answering your question...**").send()
            await make_async(rebuild_graph)()

            # Modify the message to include context about uploaded files
            file_names = [f.name for f in pdf_files]
            user_message_content = f"I have uploaded the following PDF file(s): {', '.join(file_names)}. Please use the pdf_retriever tool to answer my question: {message.content}"

        except Exception as e:
            await cl.Message(content=f"❌ Failed to embed PDF: {str(e)}").send()
            return

    # Get graph and run full message flow
    graph = cast(Runnable, cl.user_session.get("graph"))
    existing_messages = cast(list, cl.user_session.get("messages", []))
    existing_messages.append(HumanMessage(content=user_message_content))

    answer = cl.Message(content="")
    await answer.send()

    for msg, _ in graph.stream(
        {"messages": existing_messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content
            await answer.update()

    cl.user_session.set("messages", existing_messages)
