from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.analysis_agent.graph import react_graph as analysis_agent_graph
from agents.file_agent.graph import rag_agent as file_agent_graph
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
from parent_prompts import CLASSIFY_MSG_PROMPT, SYSTEM_PROMPT, MSG_REWRITER_SYSTEM_PROMPT
from parent_state import ParentGraphState, Classification, RewrittenMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import operator
from typing import Literal, cast
import chainlit as cl

llm = ChatOpenAI(model="gpt-4o")
# llm = Ollama(model="llama3.1:8b")
memory = InMemorySaver()

graph = StateGraph(ParentGraphState)


async def rewrite_message(state: ParentGraphState):
    # with cl.Step(name="Thinking...", type="run") as step:
    print(f"\nrewriting message...\n")
    state["rewritten_message"] = ""
    # Extract the last user message (prior to current) if needed.
    previous_user_message = next(
        (m.content for m in reversed(
            state["messages"][:-1]) if isinstance(m, HumanMessage)),
        ""
    )
    conversation = state["messages"][:-1]
    current_question = state["messages"][-1]
    document_context = state.get("document_context", None)
    response = None
    better_question = ""
    rephrase_prompt = ChatPromptTemplate.from_template(
        MSG_REWRITER_SYSTEM_PROMPT)
    chain = rephrase_prompt | llm.with_structured_output(
        RewrittenMessage)
    response = await chain.ainvoke(
        {"history": conversation, "context": document_context, "message": current_question, "previous_user_message": previous_user_message})
    better_question = response.rewritten_message.strip()
    print(f"\n\nrewriter response:\n{response}\n\n")
    return {
        "rewritten_message": better_question,
        "document_context": document_context,
    }


async def classify_query(state: ParentGraphState):
    """Classify the user's message to determine the next step in the graph."""
    # with cl.Step(name="Interpreting message...", type="run") as step:
    rewritten_message = state["rewritten_message"]
    previous_user_message = next(
        (m.content for m in reversed(
            state["messages"][:-1]) if isinstance(m, HumanMessage)),
        ""
    )
    has_files = len(state["attached_files"]) > 0
    print(f"has_files: {has_files}\n\n")
    prompt = ChatPromptTemplate.from_template(CLASSIFY_MSG_PROMPT)
    chain = prompt | llm.with_structured_output(Classification)
    response = await chain.ainvoke({"message": rewritten_message,
                                    "previous_user_message": previous_user_message,
                                    "has_files": has_files})
    print(f"\n\nClassification response: {response}\n\n")

    return {
        "classification": response,
        "has_files": has_files,
    }


async def route_query(state: ParentGraphState) -> Literal["analysis_agent", "file_agent", "generate_answer"]:
    """Route the query based on classification."""
    # cl.context.current_step.__exit__(None, None, None)
    # await cl.Step(name="Routing query...").send()
    # await cl.context.current_step.update()
    with cl.Step(name="Routing query...") as step:
        classificationState = state.get("classification", None)
        if classificationState:
            classification = classificationState.classification
        else:
            classification = "GENERAL"
        has_files = state["has_files"]
        print(
            f"routing - has files: {has_files}, classification: {classification}\n\n")
        if has_files:
            print("\n\nRouting to file agent...\n\n")
            return "file_agent"
        if classification == "GENERAL":
            print("\n\nRouting to generate_answer...\n\n")
            return "generate_answer"
        elif classification == "MEDICAL":
            print("\n\nRouting to analysis_agent...\n\n")
            return "analysis_agent"
        else:
            print("\n\nUnknown classification, routing to generate_answer...\n\n")
            return "generate_answer"


async def generate_answer(state: ParentGraphState):
    print("\n\nNode - Generate answer...\n\n")
    messages = state["messages"]
    message = state["rewritten_message"]
    document_context = state.get("document_context", "")
    analysis = state.get("analysis", None)
    system_message = SystemMessage(
        # TODO update prompt to always ask user if they want a deeper analysis
        content=SYSTEM_PROMPT.format(message=message, analysis=analysis, document_context=document_context))
    answer = await llm.ainvoke([system_message,
                                HumanMessage(content=message), *messages])
    print(f"\n\nFINAL ANSWER:\n{answer}\n\n")
    # Return messages properly for MessagesState
    return {"messages": [answer]}


def build_graph():
    graph.add_node("file_agent", file_agent_graph)
    graph.add_node("analysis_agent", analysis_agent_graph)

    graph.add_node("generate_answer", generate_answer)
    graph.add_node("rewrite_message", rewrite_message)
    graph.add_node("classify_query", classify_query)
    graph.add_edge(START, "rewrite_message")
    graph.add_edge("rewrite_message", "classify_query")
    graph.add_conditional_edges("classify_query", route_query)
    graph.add_conditional_edges("file_agent", route_query)
    graph.add_edge("analysis_agent", "generate_answer")
    graph.add_edge("generate_answer", END)

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

    # Step: User message received

    answer = cl.Message(content="")
    step = None
    step_active = False  # Track if the step is currently active

    async for chunk in app.astream(
        {
            "messages": [HumanMessage(content=message.content)],
            "attached_files": attached_files,
            "document_context": document_context
        },
        config=config,
        stream_mode="messages"
    ):
        message, metadata = chunk
        node = metadata.get("langgraph_node")
        in_step_nodes = node in ("rewrite_message", "classify_query")

        if in_step_nodes and cl.context.current_step.name != "Thinking...":
            step = cl.Step(name="Thinking...", type="run")
            step.__enter__()

        if node == "generate_answer":
            await answer.stream_token(message.content)

    await answer.update()

### WORKING STREAM ###
# for chunk in compiled_graph.stream(
#         {"messages": [HumanMessage(content="what is cholesterol")]},
#         config=config,
#         stream_mode="messages"):
#     # chunk is a tuple: (node_name, message_data)
#     message, metadata = chunk
#     # print(f"Node: {metadata["langgraph_node"]}")
#     print(f"{message.content}")
#     # print(f"Message type: {type(message)}")
#     # print("---")


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
