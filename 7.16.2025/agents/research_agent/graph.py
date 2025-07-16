"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from agents.research_agent.configuration import Configuration
from agents.research_agent.state import State
from agents.research_agent.tools import TOOLS
from agents.research_agent.utils import load_chat_model
from dotenv import load_dotenv
import pprint
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from agents.research_agent.state import MedicalAnalysis
from langchain_openai import ChatOpenAI
import chainlit as cl

load_dotenv()

default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Define the function that calls the model


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # print(f"state: {state}")
    # print(f"state keys: {list(state.keys())}")
    messages = state.get("messages", [])
    document_context = state.get("document_context", [])

    configuration = Configuration.from_context()
    #### Format the system prompt. Customize this to change the agent's behavior. ####

    system_message = configuration.system_prompt.format(
        document_context=state.get("document_context", []))

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(
        configuration.model).bind_tools(TOOLS)

    #### Get the model's response ####
    # async with cl.Step(name="🤖 Research Agent", type="run") as step:
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *messages]
        ),
    )

    #### Handle the case when it's the last step and the model still wants to use a tool ####
    if state.get("is_last_step", False) and response["tool_calls"]:
        return {
            "messages": [
                AIMessage(
                    id=response["id"],
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    #### Return the model's response as a list to be added to existing messages ####
    return {"messages": [response]}
    # return {"analysis": response}


# Define a new graph

builder = StateGraph(State, input_schema=State,
                     config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
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
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="research_agent")


# async def run():
#     res = await graph.ainvoke(
#         {"messages": [("user", "Who is the founder of LangChain?")],
#          "attached_files": []},
#         {"configurable": {"system_prompt": "You are a helpful AI assistant."}},
#     )

#     print("Response:", res)


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(run())
