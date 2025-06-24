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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, SystemMessage, ToolMessage
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
from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE
from langchain_core.tools import tool
from langchain.schema import StrOutputParser

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

    # retrieval chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")
    #  retriever_chain = retriever_prompt | llm | StrOutputParser()
    from langchain.chains.combine_documents import create_stuff_documents_chain
    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                RETRIEVER_SYSTEM_TEMPLATE,
            ),

        ]
    )
    document_chain = create_stuff_documents_chain(llm, retriever_prompt)
    from langchain.chains.retrieval import create_retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    cl.user_session.set("rag_chain", retrieval_chain)
    cl.user_session.set("last_attached_docs", chunks)
    # cl.user_session.set("extracted_data", existing_chunks + all_doc_chunks)
    cl.user_session.set("vector_store", vectordb)
    return vectordb


async def retrieve_chunks(message_content: str):
    vector_store = cast(Chroma, cl.user_session.get("vector_store", None))
    metadatas = cl.user_session.get("metadatas", [])
    stored_texts = cl.user_session.get("extracted_data", [])
    if vector_store:
        print("🔍 Retrieving documents from vector store")
        # await update_message(msg=sys_msg_2, content="🔍 Retrieving relevant documents...")
        # First, get documents using similarity search
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 15, })  # Increased to get more coverage

        similarity_docs = await retriever.ainvoke(message_content)

        # Second, ensure we have content from ALL uploaded files
        print("🔍 Step 1b: Ensuring all files are represented...")
        all_file_sources = set()
        for metadata in metadatas:
            all_file_sources.add(metadata['source_file'])

        print(f"📋 Total files referenced: {len(all_file_sources)}")
        print(f"📋 Files: {list(all_file_sources)}")

        # Check which files are represented in similarity search
        similarity_sources = set()
        for doc in similarity_docs:
            for i, text_chunk in enumerate(stored_texts):
                if text_chunk.page_content == doc.page_content:
                    source_file = metadatas[i].get(
                        'source_file', 'Unknown')
                    similarity_sources.add(source_file)
                    break

        print(
            f"📋 Files found in similarity search: {len(similarity_sources)}")
        print(f"📋 Missing files: {all_file_sources - similarity_sources}")

        # Add documents from missing files to ensure comprehensive coverage
        comprehensive_docs = list(similarity_docs)
        missing_files = all_file_sources - similarity_sources

        if missing_files:
            print(
                f"🔍 Step 1c: Adding content from {len(missing_files)} missing files...")
            for missing_file in missing_files:
                # Find representative chunks from missing files
                file_docs = []
                for i, text_chunk in enumerate(stored_texts):
                    if metadatas[i].get('source_file') == missing_file:
                        file_docs.append(text_chunk)

                # Add the first few chunks from each missing file
                # Add up to 3 chunks per missing file
                comprehensive_docs.extend(file_docs[:3])

        print(f"📋 Total documents for analysis: {len(comprehensive_docs)}")

        # Combine all retrieved documents with source identification
        retrieved_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([
            f"{doc.page_content}"
            for doc in comprehensive_docs
        ])

        return retrieved_text


def chatbot(state: State):
    logger.info("Running fallback chatbot node")
    return {"messages": [llm.invoke(state["messages"])]}


### CHAINLIT HOOKS ###


@cl.on_chat_start
async def start():
    pass
    # thread_id = str(uuid.uuid4())
    # cl.user_session.set("thread_id", thread_id)
    # config = {
    #     "configurable": {
    #         "thread_id": thread_id,
    #         "metadata": {
    #             "conversation_id": thread_id,
    #             "client_type": "chainlit"
    #         }
    #     }
    # }
    # checkpoint = memory.get(config) or {}
    # cl.user_session.set("messages", checkpoint.get("messages", []))
    # rebuild_graph()


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id", "123")
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
            # await make_async(rebuild_graph)()

        except Exception as e:
            await cl.Message(content=f"❌ Failed to embed PDF: {str(e)}").send()
            return

        if message.content:
            try:

                # Step 1: Retrieve relevant documents with comprehensive approach
                retrieved_text = await retrieve_chunks(
                    message_content=message.content)

            except Exception as e:
                error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
                print(f"❌ Full error details: {e}")

    # document_context = cl.user_session.get("last_attached_docs", [])

    rag_chain = cast(Runnable, cl.user_session.get("rag_chain"))
    answer = cl.Message(content="")
    await answer.send()

    async for chunk in rag_chain.astream({"input": message.content, "context": retrieved_text}):
        print(f"chunk: {chunk}")
        # answer.content += chunk
        # await answer.update()
