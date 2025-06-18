import os
import uuid

from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import chainlit as cl
from dotenv import load_dotenv
from langsmith import Client
import langsmith
from typing import cast
from langchain.schema.runnable import Runnable

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Chainlit-RAG-Assistant"


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

### TOOLS SETUP ###
tool = TavilySearchResults(max_results=10)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
memory = MemorySaver()
graph_builder.add_node("chatbot", chatbot)

### ADD TOOL NODE WITH TOOL CONDITION ###

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)


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
    checkpoint = memory.get(config)
    if checkpoint and "messages" in checkpoint:
        cl.user_session.set("messages", checkpoint["messages"])
    else:
        cl.user_session.set("messages", [])
    cl.user_session.set("graph", graph)


@cl.on_message
async def main(message: cl.Message):
    graph = cast(Runnable, cl.user_session.get("graph"))
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

    # Retrieve the existing messages from the session
    existing_messages = cast(list, cl.user_session.get("messages", []))

    # Append the new user message
    existing_messages.append(HumanMessage(content=message.content))

    answer = cl.Message(content="")
    await answer.send()
    for msg, _ in graph.stream(
        {"messages": existing_messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()
