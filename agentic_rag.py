from IPython.display import Image, display
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from typing import cast, List, Literal
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState
from langchain_core.messages import convert_to_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# CONSTANTS
EXPORT_TYPE = ExportType.DOC_CHUNKS
CHROMA_PATH = "./chroma"

# MODELS
default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
msg_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
research_llm = ChatOpenAI(model="gpt-4o-mini")
analysis_llm = ChatOpenAI(model="gpt-4o-mini")
formatter_llm = ChatOpenAI(model="gpt-4o-mini")

# IMPROVED PROMPTS
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains ANY keywords, values, or semantic meaning related to the user question, grade it as relevant. \n"
    "Be LIBERAL in your assessment - if there's even a small chance the document could help answer the question, mark it as relevant.\n"
    "For medical/lab reports, if the document contains any test names, values, or results mentioned in the question, it should be considered relevant.\n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question that would help find specific values, results, or detailed information:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Extract ALL relevant information from the context that answers the question. "
    "If you find partial information, state what you found and what might be missing. "
    "Be thorough and include all relevant details from the context.\n"
    "Question: {question} \n"
    "Context: {context}"
)

CLASSIFY_QUERY_PROMPT = """
You are a query classifier that determines whether or not to retrieve all documents from the vector store. 
If the query mentions any specific lab test names or lab-related terms, return `False`. 
Give a binary response: 'True' or 'False'.

Query: {query} 
"""

# Extended state to track all retrieved documents


class ExtendedMessagesState(MessagesState):
    all_retrieved_docs: List[str] = Field(default_factory=list)
    retrieval_attempts: int = Field(default=0)

# CLASSES


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class QueryClassification(BaseModel):
    """Classify query using a binary score to determine if all documents should be retrieved."""
    retrieve_all: bool = Field(
        description="'True' if all documents should be retrieved, 'False' if only specific documents are needed based on the query."
    )

# UTILS


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}


def extract_docs_from_files(paths: List[str]) -> List[Document]:
    print(f"extract_docs_from_files: {paths}")
    extracted_docs = []

    for path in paths:
        try:
            loader = DoclingLoader(file_path=path, export_type=EXPORT_TYPE)
            docs = []
            print(f"extracting {path}...")
            chunks = loader.load()
            print(f"Extracted {len(chunks)} chunks.")

            for chunk in chunks:
                chunk.metadata = clean_metadata(chunk.metadata)
                docs.append(chunk)

            extracted_docs.extend(docs)

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")

    # IMPROVED CHUNKING STRATEGY
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,  # Increased chunk size
        chunk_overlap=200,  # Increased overlap to preserve context
        # Better separators for medical docs
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    doc_splits = text_splitter.split_documents(extracted_docs)

    return doc_splits


def create_vector_store_from_docs(extracted_chunks: List[Document]) -> Chroma:
    """Create a vector store from the extracted documents."""
    vector_store = Chroma.from_documents(
        documents=extracted_chunks,
        collection_name='patient_files',
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    return vector_store


extracted_chunks = extract_docs_from_files(
    paths=["pdf/12_10_2024_Hemoglobin_and_Hematocrit.pdf", "pdf/12_7_2024_Urinalysis.pdf"])

vector_store = create_vector_store_from_docs(extracted_chunks=extracted_chunks)


def get_all_docs_tool(vector_store: Chroma, query: str = ""):
    """Create a tool that retrieves ALL documents from the vector store."""
    def retrieve_all_docs(query: str = "") -> str:
        """Retrieve all documents from the vector store regardless of query."""
        try:
            print(f"RETRIEVING ALL DOCUMENTS FROM VECTOR STORE")
            # Get all documents from the collection
            collection = vector_store._collection
            all_docs = collection.get()

            # Combine all document contents
            all_content = []
            for doc in all_docs['documents']:
                all_content.append(doc)

            combined_content = "\n\n---DOCUMENT SEPARATOR---\n\n".join(
                all_content)
            print(
                f"Retrieved ALL {len(all_content)} document chunks from vector store")
            return combined_content

        except Exception as e:
            print(f"Error retrieving all documents: {e}")
            return "Error retrieving documents"

    from langchain.tools import tool

    @tool
    def all_docs_retriever_tool(query: str) -> str:
        """Retrieves ALL documents from the vector store."""
        return retrieve_all_docs(query)

    return all_docs_retriever_tool


# IMPROVED NODES
def generate_answer(state: ExtendedMessagesState):
    """Generate an answer using ALL retrieved documents."""
    question = state["messages"][0].content

    # Use all retrieved documents for comprehensive context
    if state.get("all_retrieved_docs") and state["all_retrieved_docs"]:
        # All content is in single entry
        context = state["all_retrieved_docs"][0]
    else:
        context = state["messages"][-1].content if len(
            state["messages"]) > 1 else ""

    print(f"\n\\generate_answer()\n")
    # print(f"\nCONTEXT LENGTH: {len(context)} characters")
    # print(f"\nFIRST 300 CHARS: {context[:300]}...")

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = default_llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def rewrite_question(state: ExtendedMessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    # print(f"\n\nrewrite_question()")
    # print(f"\nquestion: {question}\n")

    prompt = REWRITE_PROMPT.format(question=question)
    response = default_llm.invoke([{"role": "user", "content": prompt}])

    # Reset retrieval attempts when rewriting
    return {"messages": [response], "retrieval_attempts": 0}


def grade_documents(state: ExtendedMessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Since we always get ALL documents, always proceed to generate answer."""
    print(f"\n\ngrade_documents() - Always proceeding to answer generation since we have all docs")

    # Always proceed to answer generation when we retrieve all docs
    return "generate_answer"


def generate_query_or_respond(state: ExtendedMessagesState):
    """Call the model to generate a response based on the current state."""
    print(f"\n\ngenerate_query_or_respond()\n")
    # print(f"\nmessages: {len(state['messages'])} messages\n")

    # Increment retrieval attempts
    current_attempts = state.get("retrieval_attempts", 0)

    response = (
        default_llm
        .bind_tools([get_all_docs_tool(vector_store=vector_store)])
        .invoke(state["messages"])
    )

    return {
        "messages": [response],
        "retrieval_attempts": current_attempts + 1
    }


def custom_retrieve_node(state: ExtendedMessagesState):
    """Custom retrieve node that gets ALL documents and saves to state."""
    # Get the tool call from the last message
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, 'tool_calls', [])

    if not tool_calls:
        return state

    tool_call = tool_calls[0]
    query = tool_call['args'].get('query', 'retrieve all')

    all_docs_tool = get_all_docs_tool(vector_store, query)

    # (query doesn't matter right now since we get all docs)
    all_content = all_docs_tool.invoke({"query": query})

    # Store ALL content in state
    # print(f"Storing all content in state (length: {len(all_content)} chars)")

    # Create tool message
    from langchain_core.messages import ToolMessage
    tool_message = ToolMessage(
        content=all_content,
        tool_call_id=tool_call['id'],
        name="all_docs_retriever_tool"
    )

    return {
        "messages": [tool_message],
        "all_retrieved_docs": [all_content]  # Single entry with all content
    }

# Test what's actually in the vector store


# Add this before the workflow definition
# debug_vector_store()

workflow = StateGraph(ExtendedMessagesState)

# Define the nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
# Simplified - no ToolNode needed
workflow.add_node("retrieve", custom_retrieve_node)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `retrieve` node is called
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # this function returns the "generate_answer" as the next node - no need for explicit edge here
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What were the hemoglobin and hematocrit levels in the report?",
                # "content": "What were the WBC urine levels in the report?",
                # "content": "Please give me all the lab tests and results in an organized format.",
            }
        ],
        "all_retrieved_docs": [],
        "retrieval_attempts": 0
    }
):
    for node, update in chunk.items():
        print(f"Update from node {node}")
        if "messages" in update and update["messages"] and node == "generate_answer":
            update["messages"][-1].pretty_print()
        print("\n" + "="*50 + "\n")


#################

# def debug_vector_store():
#     """Debug function to see what chunks are actually stored."""
#     print("\n=== DEBUGGING VECTOR STORE CONTENTS ===")

#     # Get all documents from the collection
#     collection = vector_store._collection
#     all_docs = collection.get()

#     print(f"Total documents in vector store: {len(all_docs['documents'])}")

#     for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
#         print(f"\n--- CHUNK {i+1} ---")
#         print(f"Content: {doc[:200]}...")
#         if len(doc) > 200:
#             print(f"... (total length: {len(doc)} chars)")
#         print(f"Metadata: {metadata}")

#     # Test specific queries
#     test_queries = [
#         "hemoglobin result",
#         "hematocrit result",
#         "RESULT =",
#         "8.3 g/dL",
#         "25.6%",
#         "lab results values"
#     ]

#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})

#     for query in test_queries:
#         print(f"\n=== TESTING QUERY: '{query}' ===")
#         results = retriever.get_relevant_documents(query)
#         for j, result in enumerate(results):
#             print(f"Result {j+1}: {result.page_content[:150]}...")


# WIP - determine whether to retrieve all docs or not based on query
# def get_all_docs_tool(vector_store: Chroma, query: str = ""):
#     """Create a tool that retrieves ALL documents from the vector store."""
#     def retrieve_all_docs(query: str = "") -> str:
#         """Retrieve relevant documents from the vector store based on the query.
#         - If the query asks about a specific lab test, data point, or section of a report, retrieve only the relevant documents.
#         - In all other cases when the user asks about uploaded patient health data, retrieve ALL documents from the vectore store.
#         """
#         try:
#             # print(f"RETRIEVING ALL DOCUMENTS FROM VECTOR STORE")
#             # Get all documents from the collection
#             # collection = vector_store._collection
#             # all_docs = collection.get()
#             prompt = ChatPromptTemplate.from_template(
#                 CLASSIFY_QUERY_PROMPT)
#             query_classifier_chain = prompt | rag_classifier_llm.with_structured_output(
#                 QueryClassification)

#             query_classifier_response = query_classifier_chain.invoke(
#                 {"query": query}
#             )
#             print(
#                 f"Query classifier response: {query_classifier_response.retrieve_all}")
#             all_docs = []
#             if query_classifier_response.retrieve_all:
#                 print(f"Retrieving ALL docs")
#                 collection = vector_store._collection
#                 all_docs = collection.get()["documents"]
#             else:
#                 print(f"Retrieving only relevant docs")
#                 retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#                 all_docs = retriever.get_relevant_documents(query)

#             # Combine all document contents
#             all_content = []

#             for chunk in all_docs:
#                 if hasattr(chunk, "page_content"):
#                     print(
#                         f"Chunk has page_content: {chunk.page_content[:100]}...")
#                     all_content.append(chunk.page_content)
#                 else:
#                     print(f"Chunk has content: {chunk[:100]}...")
#                     all_content.append(chunk)

#             combined_content = "\n\n---DOCUMENT SEPARATOR---\n\n".join(
#                 all_content)
#             print(
#                 f"Retrieved ALL {len(all_content)} document chunks from vector store")
#             return combined_content

#         except Exception as e:
#             print(f"Error retrieving all documents: {e}")
#             return "Error retrieving documents"

#     from langchain.tools import tool

#     @tool
#     def all_docs_retriever_tool(query: str) -> str:
#         """Retrieves ALL documents from the vector store."""
#         return retrieve_all_docs(query)

#     return all_docs_retriever_tool
