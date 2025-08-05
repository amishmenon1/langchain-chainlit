from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_docling.loader import ExportType
import asyncio

# Import tools from the tools module
from agents.file_agent.tools import TOOLS, load_and_process_pdf, get_vector_store
from agents.file_agent.configuration import Configuration
from agents.file_agent.state import State
# from tools import TOOLS, load_and_process_pdf
# from configuration import Configuration
# from state import State
import chainlit as cl


load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration constants
PDF_PATH = "pdf/12_7_2024_Urinalysis.pdf"
EXPORT_TYPE = ExportType.MARKDOWN

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o", temperature=0)

# Use tools from the tools module
llm_with_tools = llm.bind_tools(TOOLS)


# Use helper function to load and process PDF
# vectorstore = load_and_process_pdf(PDF_PATH, EXPORT_TYPE)


def should_continue(state: State):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


# Get system prompt from configuration
configuration = Configuration.from_context()
system_prompt = configuration.system_prompt


# Creating a dictionary of our tools
tools_dict = {our_tool.name: our_tool for our_tool in TOOLS}

# LLM Agent


async def call_llm(state: State) -> State:
    """Async function to call the LLM with the current state."""
    cl.context.current_step.name = "Processing files..."
    await cl.context.current_step.update()

    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    attached_files = state.get('attached_files', [])
    all_processed_filenames = state.get('processed_filenames', [])
    processed_filenames = []
    new_files = [f for f in attached_files if getattr(
        f, "name", None) not in all_processed_filenames]
    print(f"New files to process: {len(new_files)}")
    doc_splits = []
    formatted_docs = ""
    message = AIMessage(content="No files were found.")
    # with cl.Step(name="Processing files...") as step:
    if len(new_files) > 0:

        # cl.context.current_step.name = "Processing files..."
        # await cl.context.current_step.update()
        # If load_and_process_pdf is not async, run in thread
        doc_splits, formatted_docs, processed_filenames = await asyncio.to_thread(
            load_and_process_pdf, new_files, EXPORT_TYPE)
        print(f"Loaded {len(doc_splits)} documents from attached files.")
        # If llm_with_tools.invoke is not async, run in thread
    message = await asyncio.to_thread(llm_with_tools.invoke, messages)
    cl.context.current_step.output = f"Successfully loaded {len(processed_filenames)} files."
    await cl.context.current_step.update()
    # cl.context.current_step.name = "Analyzing query..."
    # await cl.context.current_step.update()
    # with cl.Step(name="Analyzing file data...") as step:
    return {'has_files': False, 'processed_filenames': processed_filenames,
            'extracted_documents': doc_splits, 'document_context': formatted_docs,
            'messages': [message]}


# Retriever Agent
async def take_action(state: State) -> State:
    """Async function to execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            # If tool is not async, run in thread
            result = await asyncio.to_thread(tools_dict[t['name']].invoke, t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(ToolMessage(
            tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


# def running_agent():
#     print("\n=== RAG AGENT===")

#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break

#         # converts back to a HumanMessage type
#         messages = [HumanMessage(content=user_input)]

#         result = rag_agent.invoke({"messages": messages})

#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)


# running_agent()
