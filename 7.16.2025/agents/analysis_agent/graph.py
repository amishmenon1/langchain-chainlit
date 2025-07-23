"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""


from IPython.display import Image, display
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, END, StateGraph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.analysis_agent.tools import TOOLS
from agents.analysis_agent.state import MedicalAnalysis, State
from agents.analysis_agent.prompts import ANALYSIS_SYSTEM_PROMPT
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.prompts import ChatPromptTemplate
from typing import cast, Literal
from langchain_community.llms.ollama import Ollama
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()
simple_llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = simple_llm.bind_tools(TOOLS)

# simple_llm = Ollama(model="llama3.1:8b")
# simple_llm = init_chat_model(model="llama3.1:8b", model_provider="ollama")


memory = InMemorySaver()


# def optimize_request(state: State):
#     """Optimize the request for the analysis agent."""
#     print("\n\nNode - Optimize request...\n\n")
#     # TODO return {"optimized_query":optimized_query} (str - should exist on internal graph state)
#     pass


async def assistant(state: State):
    """Run the analysis assistant with the provided state."""
    cl.context.current_step.name = "Medical Analysis Agent"
    await cl.context.current_step.update()
    print("\n\nNode - analysis assistant...\n\n")
    messages = state["messages"]
    user_message = state["messages"][-1]
    document_context = state.get("document_context", "")
    response = ""
    # with cl.Step(name="Medical Analysis Agent...") as step:
    system_msg = ANALYSIS_SYSTEM_PROMPT.format(
        document_context=document_context, user_message=user_message.content)

    response = cast(
        AIMessage,
        llm_with_tools.invoke(
            [SystemMessage(content=system_msg), *messages]
        ),
    )
    cl.context.current_step.output = "Analysis complete! Returning to chat..."
    await cl.context.current_step.update()
    print(f"AGENT response: {response}\n\n")

    # TODO return MedicalAnalysis object (should exist on parent state)
    return {"messages": [response]}

# Node 2 - Tool Router node


async def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """

    last_message = state["messages"][-1]
    # print(f"\n\nAGENT - Last message: {last_message}\n\n")
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        print("\n\nAGENT - No tool calls in last message, ending...\n\n")
        return END
    # Otherwise we execute the requested actions
    print("\n\nAGENT - Tool calls found, routing to tools...\n\n")
    return "tools"


### DEFINE AGENT GRAPH ###


# Graph
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(TOOLS))

# Define edges: these determine how the control flow moves
# TODO implmement optimize_request to pass to assistant node
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,


)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)
