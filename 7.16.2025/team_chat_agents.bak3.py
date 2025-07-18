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
from typing import AsyncGenerator

from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from torch import chunk
from templates.agent_prompts.analysis import ANALYSIS_PROMPT
from templates.agent_prompts.generation import ENHANCED_GENERATE, SUPERVISOR_PROMPT, SUPERVISOR_PROMPT2
from templates.agent_prompts.research import RESEARCH_PROMPT
import chainlit as cl
import asyncio
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import Runnable
from agents.research_agent.state import State
from agents.research_agent.graph import graph as research_agent
# from rag_agent import graph as rag_agent
import rag_workflow
from langchain.prompts import ChatPromptTemplate

import logging
from langgraph.prebuilt import tools_condition
from langchain_anthropic import ChatAnthropic
from langchain.schema import StrOutputParser


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


memory = InMemorySaver()


load_dotenv()

openai_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
gemini_llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# claude_llm = ChatAnthropic(
#     model="claude-sonnet-4-20250514",
#     temperature=0,
#     max_tokens=1024,
#     timeout=None,
#     max_retries=2,
#     streaming=True
# )

default_llm = openai_llm


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

    # -- Node 1: Stream from OpenAI --
    def stream_from_openai() -> Runnable:
        async def node(state: State) -> AsyncGenerator:
            answer = cl.Message(content="")
            print("🤖 Starting stream from OpenAI...")
            # user_input = state["messages"][-1].content
            # messages = [{"role": "user", "content": user_input}]

            document_context = state.get("document_context", [])
            analysis = state.get("analysis", None)
            # Clean up any messages with invalid names before passing to OpenAI
            message_history = state.get("messages", [])
            sys_msg = SUPERVISOR_PROMPT2.format(
                document_context=document_context, analysis=analysis)
            messages = [
                {"role": "system", "content": sys_msg}, *message_history]
            async for chunk in openai_llm.astream(messages):
                await answer.stream_token(chunk.text())
            await answer.update()
        return cast(Runnable, node)

    # -- Node 2: Supervisor subgraph --
    med_team = create_supervisor(
        model=default_llm,
        supervisor_name="team",
        agents=[research_agent],
        state_schema=State,
        prompt=generation_prompt,
        add_handoff_back_messages=True,
        output_mode="last_message",
    ).compile(checkpointer=memory)

    # -- Node 3: Stream from Gemini --
    def stream_from_gemini() -> Runnable:
        async def node(state: State) -> AsyncGenerator:
            answer = cl.Message(content="")
            print("🤖 Starting stream from Gemini...")
            message = state["messages"][-1].content
            # print(f"User input: {user_input}")
            # messages = [{"role": "user", "content": user_input}]
            # document_context = state.get("document_context", [])
            # analysis = state.get("analysis", None)
            # # Clean up any messages with invalid names before passing to OpenAI
            # message_history = state.get("messages", [])
            # last_message = state["messages"][-1]
            # sys_msg = ENHANCED_GENERATE.format(
            #     document_context=document_context, analysis=analysis)
            # messages = [
            #     {"role": "system", "content": sys_msg}, *message_history]

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        ENHANCED_GENERATE.format(document_context=state.get(
                            "document_context", []), analysis=state.get("analysis", None))
                    ),
                    ("human", "{message}"),
                ]
            )
            llm = prompt | gemini_llm | StrOutputParser()
            async for chunk in llm.astream({"message": message}):
                # yield {"messages": [AIMessageChunk(content=chunk.text())]}
                print(f"Gemini chunk: {chunk}")
                await answer.stream_token(chunk)
            await answer.update()
        return cast(Runnable, node)

    # -- Classifier Edge Logic --
    def classify_message(state: State) -> str:
        user_input = state["messages"][-1].content.lower()
        # if any(term in user_input for term in ["health", "lab", "blood", "scan", "diagnosis", "medical", "doctor", "symptoms"]):
        return "analyze"
        # return "generate"

    # -- Assemble Graph --
    graph = StateGraph(State)
    graph.add_node("generate", stream_from_openai())
    graph.add_node("team", med_team)
    graph.add_node("generate_enhanced", stream_from_gemini())

    graph.add_conditional_edges(START, classify_message, {
        "generate": "generate",
        "analyze": "team"
    })
    graph.add_edge("generate", END)
    # graph.add_edge("team", "generate_enhanced")
    graph.add_edge("team", END)
    # graph.add_edge("generate_enhanced", END)

    compiled_graph: CompiledStateGraph = graph.compile(checkpointer=memory)

    await cl.Message(
        content="Welcome to the team chat! How can I assist you today?"
    ).send()

    cl.user_session.set("app", compiled_graph)


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
        existing_document_context = cl.user_session.get("document_context", [])
        existing_filenames = cl.user_session.get('filenames', [])
        new_files = [
            f for f in attached_files if f.name not in existing_filenames]

        if len(new_files) > 0:
            document_context, filenames = await rag_workflow.run_workflow(messages=[HumanMessage(content=message.content)], files=new_files)
            existing_document_context.extend(document_context)
            existing_filenames.extend(filenames)
            cl.user_session.set("document_context", existing_document_context)
            cl.user_session.set('filenames', existing_filenames)

    app = cast(CompiledStateGraph, cl.user_session.get("app"))
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id},
    }
    document_context = cl.user_session.get("document_context", [])

    async def stream_updates():
        """Stream each update from the graph execution to the user."""
        print("🤖 Starting stream with updates mode...")
        print("=" * 50)

        # Track messages we've already processed
        seen_message_ids = cast(
            set, cl.user_session.get("seen_message_ids", set()))
        active_tool_steps = {}  # Track active tool steps by tool call ID
        response_started = False
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
                print(f"\n🔄 Processing node: {node_name}\n\n")
                print(f"\n🔄  node data: {node_data}")

                # Create a step for each node
                step_name = node_name.replace("_", " ").title()
                if node_name == "team":
                    step_name = "🎯 Analysis Team"
                elif node_name == "research_agent":
                    step_name = "🔍 Research Agent"
                elif "tool" in node_name.lower():
                    step_name = f"🛠️ {step_name}"
                else:
                    step_name = f"⚙️ {step_name}"

                # Check if there are messages in this update
                if node_data and "messages" in node_data and node_data["messages"]:
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

                                # Check if this is a final response from team
                                if (node_name == "team" and
                                        isinstance(msg, AIMessage)):

                                    cl.user_session.set(
                                        "seen_message_ids", seen_message_ids)

                                    if not response_started:
                                        response_started = True
                                        answer = cl.Message(content="")

                                    # STREAM DOESNT WORK
                                    await answer.stream_token(msg.content)

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

    await stream_updates()
