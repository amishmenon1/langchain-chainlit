from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import os
from str_utils import pretty_print_messages

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AnyMessage
from typing import cast, List
import rag_workflow
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver

memory = InMemorySaver()

# from IPython.display import display, Image

load_dotenv()


class QueryClassification(BaseModel):
    """Classify query using a binary score to determine if all documents should be retrieved."""
    retrieve_all: bool = Field(
        description="'True' if all documents should be retrieved, 'False' if only specific documents are needed based on the query."
    )


def web_search(query: str) -> List[BaseMessage]:
    """Search the web for a given query and return the results."""
    # initialize Tavily search tool
    response = TavilySearch(max_results=3).invoke(query)
    print(f"Web search results for query '{query}': {response['results']}")
    return response["results"]

# create custom math tools


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


# def perform_rag(messages: List[AnyMessage]):
#     rag_workflow.run_workflow(messages)

    # Create specialized agents
math_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
    checkpointer=memory,
)


research_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
    checkpointer=memory,
)

# Create supervisor workflow
supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),
    agents=[research_agent,
            math_agent
            ],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),

    add_handoff_back_messages=True,
    output_mode="full_history",

).compile(checkpointer=memory)

# # Compile and run
# app = workflow.compile()
# result = app.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": "what's the combined headcount of the FAANG companies in 2024?"
#         }
#     ]
# })

##### Run tools directly #####

# Run web search tool directly
# web_search_results = web_search.invoke("who is the mayor of NYC?")
# print(web_search_results["results"][0]["content"])

##### Run the agents directly #####

# Research Agent

# for chunk in research_agent.stream(
#     {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
# ):
#     pretty_print_messages(chunk)


# Math Agent

# for chunk in math_agent.stream(
#     {"messages": [{"role": "user", "content": "what's (3 + 5) x 7"}]}
# ):
#     pretty_print_messages(chunk)

config = {"configurable": {"thread_id": "1"}}

# Supervisor Agent
for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                # "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
                "content": "What is the weather like in New York City today?",
            }
        ]
    }, config=config
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]


##### Display graph #####

# display(Image(supervisor.get_graph().draw_mermaid_png()))
