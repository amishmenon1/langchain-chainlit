

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.analysis_agent.graph import react_graph

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
from parent_prompts import CLASSIFY_MSG_PROMPT, SYSTEM_PROMPT
from parent_state import ParentGraphState, Classification
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import operator
from typing import Literal

llm = ChatOpenAI(model="gpt-4o-mini")
memory = InMemorySaver()


def query_classification(state: ParentGraphState):
    """Classify the user's message to determine the next step in the graph."""
    user_message = state["messages"][-1]
    previous_user_message = next(
        (m.content for m in reversed(
            state["messages"][:-1]) if isinstance(m, HumanMessage)),
        ""
    )
    has_files = state.get("has_files", False)
    prompt = ChatPromptTemplate.from_template(CLASSIFY_MSG_PROMPT)
    chain = prompt | llm.with_structured_output(Classification)

    response = chain.invoke({"message": user_message.content,
                             "previous_user_message": previous_user_message,
                             "has_files": has_files})
    print(f"\n\nClassification response: {response}\n\n")

    if has_files:
        # run wag workflow
        pass

    if response.classification == "GENERAL":
        return "generate_answer"
    # elif response.classification == "FILE":
    #     return "rag_agent"
    else:
        return "analysis_agent"


# TODO update prompt to ask user if they want a deeper analysis
def generate_answer(state: ParentGraphState):
    """Generate the final answer based on the user's message and chat context."""
    print("\n\nNode - Generate answer...\n\n")
    analysis = state.get("analysis", None)

    system_message = SystemMessage(
        content=SYSTEM_PROMPT.format(analysis=analysis))

    # print(f"message last: {state['messages'][-1]}")
    answer = llm.invoke([system_message,
                        HumanMessage(content=state["messages"][-1].content)])
    print(f"\n\n{answer}\n\n")
    # Return messages properly for MessagesState
    return {"messages": [answer]}


# Use MessagesState instead of State to match what react_graph expects
graph = StateGraph(ParentGraphState)
graph.add_node("analysis_agent", react_graph)
# graph.add_node("rag_agent", file_graph)

graph.add_node("generate_answer", generate_answer)


graph.add_conditional_edges(
    START,
    query_classification,
)
graph.add_edge("analysis_agent", "generate_answer")
graph.add_edge("generate_answer", END)

compiled_graph: CompiledStateGraph = graph.compile(checkpointer=memory)
config: RunnableConfig = {
    "configurable": {"thread_id": "123"},
}

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

MEDICAL_QUESTION = "I have a patient with high cholesterol in his 60s. Please advise on how to manage his condition."
GENERAL_QUESTION = "What is cholesterol?"

question = GENERAL_QUESTION

### WORKING INVOKE ###
compiled_graph.invoke(
    {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                  HumanMessage(content=question)], },
    config=config,)
