from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, AIMessageChunk, SystemMessage, BaseMessage, ToolMessage
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast, List, Optional

from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from templates.agent_prompts.analysis import ANALYSIS_PROMPT
from templates.agent_prompts.generation import GENERATION_PROMPT, SUPERVISOR_PROMPT, SUPERVISOR_PROMPT2
from templates.agent_prompts.research import RESEARCH_PROMPT
import chainlit as cl
import asyncio
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.research_agent.state import State
from agents.research_agent.graph import graph as research_agent
# from rag_agent import graph as rag_agent
import rag_workflow

import logging
from langgraph.prebuilt import tools_condition

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


memory = InMemorySaver()


load_dotenv()

default_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)


def generation_prompt(state: State, config: RunnableConfig) -> list[AnyMessage]:
    # document_context = config["configurable"].get("document_context")
    document_context = state.get("document_context", [])
    analysis = state.get("analysis", None)
    # Clean up any messages with invalid names before passing to OpenAI
    messages = state.get("messages", [])

    system_msg = SUPERVISOR_PROMPT2.format(
        document_context=document_context, analysis=analysis)

    return [{"role": "system", "content": system_msg}, *messages]


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat with a welcome message."""
    ###### AGENTS ######

    supervisor = create_supervisor(
        model=default_llm,
        agents=[research_agent],
        state_schema=State,
        prompt=generation_prompt,
        # prompt="You are a helpful AI assistant.",
        add_handoff_back_messages=True,
        output_mode="last_message",

    )

    # Define the nodes
    # supervisor.add_node("call_model", call_model)

    # supervisor.add_edge(START, "call_model")
    # supervisor.add_edge("call_model", END)

    # supervisor.compile(checkpointer=memory)

    await cl.Message(
        content="Welcome to the team chat! How can I assist you today?"
    ).send()

    cl.user_session.set("app", supervisor.compile(checkpointer=memory))


@cl.on_message
async def on_message(message: cl.Message):
    # Check if files are attached to this message
    attached_files = []
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    # If files are attached, process them first
    if attached_files:
        cl.user_session.set('has_attachments', True)
        existing_document_context = cl.user_session.get("document_context", [])
        existing_filenames = cl.user_session.get('filenames', [])
        new_files = [
            f for f in attached_files if f.name not in existing_filenames]
        # print(f"\n\nNew files to process: {new_files}\n\n")
        # print(f"\n\nExisting files: {existing_filenames}\n\n")

        if len(new_files) > 0:
            document_context, filenames = await rag_workflow.run_workflow(messages=[HumanMessage(content=message.content)], files=new_files)
            # print(f"\n\nExtracted docs: {document_context[:200]}\n\n")
            # logger.debug(f"\n\nExtracted docs: {document_context[:200]}\n\n")
            existing_document_context.extend(document_context)
            existing_filenames.extend(filenames)
            # print(f"\n\nProcessed filenames:\n {existing_filenames}\n\n")
            # logger.debug(
            #     f"\n\nProcessed filenames:\n {existing_filenames}\n\n")
            cl.user_session.set("document_context", existing_document_context)
            cl.user_session.set('filenames', existing_filenames)

    else:
        cl.user_session.set('has_attachments', False)
        cl.user_session.set('latest_attached_docs', None)

    app = cast(CompiledStateGraph, cl.user_session.get("app"))
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id},
    }
    document_context = cl.user_session.get("document_context", [])
    # Ensure document_context is never None
    if document_context is None:
        document_context = []

    async def stream_formatted():
        """Stream the response with better formatting."""
        print("🤖 Starting Research Agent...")
        print("=" * 50)

        answer = cl.Message(content="")
        response_started = False
        # Track messages we've already processed
        seen_message_ids = cast(
            set, cl.user_session.get("seen_message_ids", set()))

        async for chunk in app.astream(
            {"messages": [{"role": "user", "content": message.content}],
             "document_context": document_context,
             },
            config,
        ):
            print(f"\n\nChunk received: {chunk}\n\n")
            for node_name, node_output in chunk.items():
                print(f"\n🔄 Node: {node_name}")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if msg.id in seen_message_ids:
                            continue  # Skip already processed messages

                        seen_message_ids.add(msg.id)
                        cl.user_session.set(
                            "seen_message_ids", seen_message_ids)

                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"🔧 Tool Call: {msg.tool_calls[0]['name']}")
                            print(f"📝 Arguments: {msg.tool_calls[0]['args']}")
                        elif hasattr(msg, 'content') and msg.content:
                            print(f"💬 Response: {msg.content}")
                            # print(f"💬 Message: {msg}")

                            # Check if this is a supervisor response that should be streamed
                            if (node_name == "supervisor" and
                                isinstance(msg, AIMessage) and
                                hasattr(msg, 'name') and
                                    msg.name == "supervisor"):

                                if not response_started:
                                    response_started = True
                                    answer = cl.Message(content="")

                                await answer.stream_token(msg.content)

                print("-" * 30)

        if response_started:
            await answer.update()

        else:
            await cl.Message(content="No response was generated.").send()

    async def stream_messages():

        print("=" * 50)

        answer = cl.Message(content="")
        response_started = False
        async for token, metadata in app.astream(
            {"messages": [{"role": "user", "content": message.content}],
             "document_context": document_context,
             },
            config,
            stream_mode="messages"
        ):
            print("Token", token)
            print("Metadata", metadata)
            print("\n")
            await answer.stream_token(token.content)
        await answer.update()

    async def stream_updates():
        """Stream each update from the graph execution to the user."""
        print("🤖 Starting stream with updates mode...")
        print("=" * 50)

        answer = cl.Message(content="")
        response_started = False
        # Track messages we've already processed
        seen_message_ids = cast(
            set, cl.user_session.get("seen_message_ids", set()))
        active_tool_steps = {}  # Track active tool steps by tool call ID

        async for update in app.astream(
            {"messages": [{"role": "user", "content": message.content}],
             "document_context": document_context,
             },
            config,
            stream_mode="updates"
        ):
            print(f"\n📦 Update received: {update}")

            # Process each node update
            for node_name, node_data in update.items():
                print(f"\n🔄 Processing node: {node_name}")

                # Create a step for each node
                step_name = node_name.replace("_", " ").title()
                if node_name == "supervisor":
                    step_name = "🎯 Supervisor Analysis"
                elif node_name == "research_agent":
                    step_name = "🔍 Research Agent"
                elif "tool" in node_name.lower():
                    step_name = f"🛠️ {step_name}"
                else:
                    step_name = f"⚙️ {step_name}"

                # Check if there are messages in this update
                if "messages" in node_data and node_data["messages"]:
                    # Only process the NEW messages (typically the last one in the list)
                    messages = node_data["messages"]

                    # Process only new messages we haven't seen before
                    for msg in messages:
                        print(f"📨 New message type: {type(msg).__name__}")
                        if msg.id in seen_message_ids:
                            continue  # Skip already processed messages

                        seen_message_ids.add(msg.id)
                        # Handle tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_call = msg.tool_calls[0]
                            tool_name = tool_call.get('name', 'Unknown')
                            tool_args = tool_call.get('args', {})
                            tool_call_id = tool_call.get(
                                'id', f"{tool_name}_{len(active_tool_steps)}")
                            print(f"🔧 Tool Call: {tool_name}")

                            # Create a step for tool usage
                            step = cl.Step(name=f"🔧 {tool_name}")
                            await step.__aenter__()

                            step.output = f"Executing tool: **{tool_name}**"
                            if tool_args:
                                # Format tool arguments nicely
                                args_str = ", ".join([f"{k}: {str(v)[:50]}..." if len(str(v)) > 50 else f"{k}: {v}"
                                                      for k, v in tool_args.items()])
                                step.output += f"\n\nArguments: {args_str}"

                            # Store the step reference to update it later with results
                            active_tool_steps[tool_call_id] = step

                        # Handle AI message content - stream it properly
                        elif (hasattr(msg, 'content') and msg.content):
                            if isinstance(msg, (AIMessage, AIMessageChunk)):
                                print(f"💬 AI Content: {msg.content[:100]}...")

                                # Check if this is a final response from supervisor
                                if (node_name == "supervisor" and
                                        isinstance(msg, AIMessage)):

                                    cl.user_session.set(
                                        "seen_message_ids", seen_message_ids)

                                    if not response_started:
                                        response_started = True
                                        answer = cl.Message(content="")

                                    # MANUAL STREAMING the content character by character for smooth streaming
                                    content = msg.content
                                    chunk_size = 5  # Stream in small chunks for smooth effect

                                    for i in range(0, len(content), chunk_size):
                                        chunk = content[i:i + chunk_size]
                                        await answer.stream_token(chunk)
                                        # Small delay for smoother streaming effect
                                        await asyncio.sleep(0.01)

                            if isinstance(msg, (HumanMessage)):
                                print(
                                    f"💬 Human Content: {msg.content[:100]}...")

                            # Handle tool message results
                            elif isinstance(msg, ToolMessage):
                                print(
                                    f"🛠️ Tool Result: {str(msg.content)[:100]}...")

                                # Find the corresponding tool step and update it with results
                                tool_call_id = getattr(
                                    msg, 'tool_call_id', None)
                                if tool_call_id and tool_call_id in active_tool_steps:
                                    step = active_tool_steps[tool_call_id]
                                    result_content = str(msg.content)

                                    # Update the existing step with the result
                                    if len(result_content) > 300:
                                        step.output += f"\n\n**Result**: {result_content[:300]}..."
                                    else:
                                        step.output += f"\n\n**Result**: {result_content}"

                                    # Close the step and remove from active steps
                                    await step.__aexit__(None, None, None)
                                    del active_tool_steps[tool_call_id]
                                else:
                                    # Fallback: create a separate step if we can't find the matching tool call
                                    async with cl.Step(name="📊 Tool Results") as step:
                                        if len(result_content) > 300:
                                            step.output = f"**Result**: {result_content[:300]}..."
                                        else:
                                            step.output = f"**Result**: {result_content}"

                print("-" * 30)

        # Finalize the response
        if response_started:
            await answer.update()
        else:
            await cl.Message(content="No response was generated.").send()

    # await stream_formatted()
    # await stream_messages()
    await stream_updates()

    async def stream_llm():

        ### THIS DOES NOT STREAM ###
        # answer = cl.Message(content="")
        # # msg = {"role": "user", "content": message.content}
        # async for update, metadata in app.astream(
        #     {"messages": [{"role": "user", "content": message.content}],
        #      "document_context": document_context,
        #      },
        #     config,
        #     stream_mode="messages"
        # ):
        #     print(f"\n📦 Update: {update}\n\n")
        #     print(f"\n📦 Metadata: {metadata}\n\n")
        #     response_content = update.content[0]["text"]
        #     print(f"\n📦 Update content: {update.content[0]["text"]}")
        #     await answer.stream_token(response_content)
        # await answer.update()

        ### THIS DOES STREAM ###

        answer = cl.Message(content="")
        msg = {"role": "user", "content": message.content}

        async for chunk in default_llm.astream([msg]):
            print(f"chunk: {chunk}")
            await answer.stream_token(chunk.text())
        await answer.update()

    await stream_llm()
