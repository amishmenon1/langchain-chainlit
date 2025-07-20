

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.analysis_agent.graph_offline import react_graph
from langchain_community.llms.ollama import Ollama

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from IPython.display import Image, display
from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph import MessagesState
import operator
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain.schema.runnable.config import RunnableConfig

from langgraph.checkpoint.memory import InMemorySaver
from parent_prompts_offline import CLASSIFY_MSG_PROMPT, SYSTEM_PROMPT, MSG_REWRITER_SYSTEM_PROMPT
from parent_state import ParentGraphState, Classification, RewrittenMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import operator
from typing import Literal, cast
import chainlit as cl
from langchain.chat_models import init_chat_model

# llm = ChatOpenAI(model="gpt-4o")
# llm = Ollama(model="llama3.1:8b")
llm = init_chat_model(model="llama3.1:8b", model_provider="ollama")

memory = InMemorySaver()

graph = StateGraph(ParentGraphState)


def rewrite_message(state: ParentGraphState):
    print(f"\nrewriting message...\n")
    # state["retrieved_docs"] = []
    state["rewritten_message"] = ""
    # # Extract the last user message (prior to current) if needed.
    previous_user_message = next(
        (m.content for m in reversed(
            state["messages"][:-1]) if isinstance(m, HumanMessage)),
        ""
    )

    conversation = state["messages"][:-1]
    current_question = state["messages"][-1]
    document_context = state.get("document_context", None)

    rephrase_prompt = ChatPromptTemplate.from_template(
        MSG_REWRITER_SYSTEM_PROMPT)

    # chain = rephrase_prompt | llm.with_structured_output(
    #     RewrittenMessage)

    chain = rephrase_prompt | llm

    response = chain.invoke(
        {"history": conversation, "context": document_context, "message": current_question, "previous_user_message": previous_user_message})
    better_question = response.content.strip()
    print(f"\n\nrewriter response:\n{response}\n\n")
    return {
        "rewritten_message": better_question,
        "document_context": document_context
    }


def classify_query(state: ParentGraphState):
    """Classify the user's message to determine the next step in the graph."""
    user_message = state["messages"][-1]
    rewritten_message = state["rewritten_message"]
    previous_user_message = next(
        (m.content for m in reversed(
            state["messages"][:-1]) if isinstance(m, HumanMessage)),
        ""
    )
    has_files = len(state["attached_files"]) > 0
    prompt = ChatPromptTemplate.from_template(CLASSIFY_MSG_PROMPT)
    # chain = prompt | llm.with_structured_output(Classification)
    chain = prompt | llm

    response = chain.invoke({"message": rewritten_message,
                            "previous_user_message": previous_user_message,
                             "has_files": has_files})
    print(f"\n\nClassification response: {response}\n\n")
    # classification = 'GENERAL'
    # if hasattr(response.content, 'classification'):
    #     classification = response.classification
    return {
        "classification": response.content
    }


def route_query(state: ParentGraphState) -> Literal["analysis_agent", "generate_answer_offline"]:
    """Route the query based on classification."""
    classification = state["classification"]
    has_files = len(state["attached_files"]) > 0
    if has_files:
        print("\n\nRouting to file agent...\n\n")
        pass
        # return "file_agent"
    if classification == "GENERAL":
        print("\n\nRouting to generate_answer...\n\n")
        return "generate_answer_offline"
    elif classification == "MEDICAL":
        print("\n\nRouting to analysis_agent...\n\n")
        return "analysis_agent"
    else:
        print("\n\nUnknown classification, routing to generate_answer...\n\n")
        return "generate_answer_offline"
# TODO update prompt to ask user if they want a deeper analysis


def generate_answer(state: ParentGraphState):
    """Generate the final answer based on the user's message and chat context."""
    print("\n\nNode - Generate answer...\n\n")
    messages = state["messages"]
    message = state["rewritten_message"]

    analysis = state.get("analysis", None)

    system_message = SystemMessage(
        content=SYSTEM_PROMPT.format(analysis=analysis))

    print(f"last message: {state['messages'][-1]}\n\n")
    print(f"last message rewritten: {message}\n\n")
    answer = llm.invoke([system_message,
                        HumanMessage(content=message), *messages])
    print(f"\n\n{answer}\n\n")
    # Return messages properly for MessagesState
    return {"messages": [answer]}


def generate_answer_offline(state: ParentGraphState):
    """Generate the final answer based on the user's message and chat context."""
    print("\n\nNode - Generate answer...\n\n")
    message = state["messages"][-1]
    messages = [message.content, *state["messages"]]

    analysis = state.get("analysis", None)

    system_message = SystemMessage(SYSTEM_PROMPT.format(analysis=analysis))

    # answer = llm.invoke([system_message,
    #                     HumanMessage(content=message), *messages])
    answer = llm.invoke([system_message, *messages])
    # print(f"\n\n{answer}\n\n")
    # Return messages properly for MessagesState
    return {"messages": [answer]}


def build_graph():
    graph.add_node("analysis_agent", react_graph)

    # TODO UNCOMMENT WHEN BACK ONLINE
    graph.add_node("generate_answer_offline", generate_answer_offline)
    graph.add_node("rewrite_message", rewrite_message)
    graph.add_node("classify_query", classify_query)
    graph.add_edge(START, "rewrite_message")
    graph.add_edge("rewrite_message", "classify_query")
    graph.add_conditional_edges(
        "classify_query",
        route_query,
    )
    graph.add_edge("analysis_agent", "generate_answer_offline")
    graph.add_edge("generate_answer_offline", END)

    return graph


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat with a system message."""
    graph = build_graph()
    compiled_graph: CompiledStateGraph = graph.compile(checkpointer=memory)
    cl.user_session.set("app", compiled_graph)

    await cl.Message(
        content="Welcome to the team chat! How can I assist you today?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    attached_files = []
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    app = cast(CompiledStateGraph, cl.user_session.get("app"))
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id},
    }
    document_context = cl.user_session.get("document_context", [])
    answer = cl.Message(content="")

    # OFFLINE SOLUTION

    for chunk in app.stream(
        {"messages": [HumanMessage(message.content)],
         "attached_files": attached_files,
         "document_context": document_context},
        config=config,
            stream_mode="messages"):
        message, metadata = chunk
        if (metadata["langgraph_node"] == "generate_answer_offline"):
            await answer.stream_token(message.content)
    await answer.update()


### TEST GRAPH DIRECTLY ###
# MEDICAL_QUESTION = "I have a patient with high cholesterol in his 60s. Please advise on how to manage his condition."
# GENERAL_QUESTION = "What is cholesterol?"

# question = GENERAL_QUESTION

# ### WORKING INVOKE ###
# graph = build_graph()
# compiled_graph: CompiledStateGraph = graph.compile(checkpointer=memory)
# config: RunnableConfig = {
#     "configurable": {"thread_id": "123"},
# }

# compiled_graph.invoke(
#     {"messages": [SystemMessage(content=SYSTEM_PROMPT),
#                   HumanMessage(content=question)], },
#     config=config,)
