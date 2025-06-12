from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable import Runnable
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
from prompt_templates import MSG_CLASSIFIER_PROMPT_TEMPLATE, RAG_CLASSIFIER_PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate
import logging
import operator
from typing_extensions import TypedDict


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
CHROMA_PATH = 'chroma'
# Define the function that calls the model


class UserMessageState(MessagesState):
    document_context: Any
    # clusters: Annotated[List[List[dict]], operator.add]


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
                async for page in loader.alazy_load():
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
    default_llm = ChatOpenAI(model="gpt-4o-mini")
    msg_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
    rag_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
    rag_llm = ChatOpenAI(model="gpt-4o-mini")
    research_llm = ChatOpenAI(model="gpt-4o-mini")
    analysis_llm = ChatOpenAI(model="gpt-4o-mini")
    formatter_llm = ChatOpenAI(model="gpt-4o-mini")

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # node 1 - classifier node
    def classify_message(state: UserMessageState):
        print(f"classify_message()")
        message = state["messages"][-1]
        # sys_prompt = MSG_CLASSIFIER_PROMPT_TEMPLATE.format(message=message)
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
        # print(f"classifier response: {response}")
        if "is_medical_question" in response.content:
            print(f"User is asking a medical question")
            return "is_medical_question"
        else:
            print(f"User is asking a general question")
            return "is_general_question"

    # node 1a - RAG node
    def retrieve_docs(state: UserMessageState):
        # def format_docs(docs: List[Document]):
        #     return "\n\n".join(doc.page_content for doc in docs)

        print(f"retrieve_docs()")
        message = state["messages"][-1]
        rag_chain = cl.user_session.get("rag_chain", None)
        if rag_chain:
            rag_chain = cast(Runnable, rag_chain)
            document_context = cl.user_session.get("extracted_data", [])
            # print(f"retrieved docs: {document_context[:200]}")
            rag_results = rag_chain.invoke(
                {"question": message, "context": document_context},
                config=RunnableConfig(
                    callbacks=[cl.LangchainCallbackHandler()]),
            )
            print(f"rag results\n\n: {rag_results[:200]}\n\n")
        return {
            "document_context": rag_results
        }

    # node 2a - researcher node

    def web_search(state: UserMessageState):
        print(f"web_search()")
        # print(f"state: {state}")
        document_context = state["document_context"]
        # print(f"document_context: {document_context[:500]}")
        # TODO implement method using pub med and web search tools
        return state

    # node 3a - analyzer node
    def analyze(state: MessagesState):
        print(f"analyze()")
        # TODO implement method using analysis_llm + memory + docs + tool results
        return state
        # response = analysis_llm.invoke(state["messages"])
        # return {"messages": response}

    # node 1b - default generator node
    def generate(state: MessagesState):
        print(f"generate()")
        # TODO implement method using formatter_llm + latest messages
        return state
        # response = formatter_llm.invoke(state["messages"])
        # return {"messages": response}

    # node 4 - formatter node
    def format_and_respond(state: MessagesState):
        print(f"format_and_respond()")
        response = formatter_llm.invoke(state["messages"])
        return {"messages": response}

    # Define the (single) node in the graph
    # workflow.add_node("classify_message", classify_message)
    # workflow.add_node("rag_router", rag_router)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("web_search", web_search)
    workflow.add_node("analyze", analyze)
    workflow.add_node("generate", generate)
    # workflow.add_node("format_and_respond", format_and_respond)

    # workflow.add_edge(START, "route_message")
    workflow.add_conditional_edges(START, classify_message, {
        # If medical question, use RAG and medical analysis
        # "is_medical_question": "rag_router",
        "is_medical_question": "retrieve_docs",
        # Otherwise use general model completion
        "is_general_question": "generate",
    })

    workflow.add_edge("retrieve_docs", "web_search")
    workflow.add_edge("web_search", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    cl.user_session.set("app", app)

    # setup RAG chain
    rag_prompt = prompts.pull("rlm/rag-prompt")
    rag_chain: Runnable = rag_prompt | rag_llm | StrOutputParser()
    cl.user_session.set("rag_chain", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    input_messages = [HumanMessage(message.content)]

    # Check if files are attached to this message
    attached_files = None
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    # If files are attached, process them first
    if attached_files:
        sys_msg_1 = await new_message(
            content="📎 **Files detected!**")
        await load_files_into_vectordb(attached_files)

    config = {"configurable": {"thread_id": "abc123"}}
    app = cast(Runnable, cl.user_session.get("app"))
    msg = cl.Message(content="")
    for chunk in app.stream(
        {"messages": input_messages},
        config,
        # highlight-next-line
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            # print(chunk.content, end="|")
            await msg.stream_token(chunk.content)

    await msg.send()
