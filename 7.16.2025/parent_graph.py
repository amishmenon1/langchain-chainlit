

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.analysis_agent.graph import react_graph

from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph import MessagesState
import operator
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain.schema.runnable.config import RunnableConfig

from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
import operator

llm = ChatOpenAI(model="gpt-4o-mini")
memory = InMemorySaver()

# TODO create ParentGraphState(MessagesState) for analysis, docs (list), document context (str), research context, answer
# TODO implement classification node to classify the question


def generate_answer(state: MessagesState):
    # """Generate an answer based on the question and classification."""
    print("\n\nNode - Generate answer...\n\n")
    # System message
    system_message = SystemMessage(content=("""
        You are an AI assistant that answers questions based on general knowledge.  

        ### **Guidelines:**  
            - Provide **direct, concise, and accurate** answers. 
                                             """))
    print(f"message last: {state['messages'][-1]}")
    answer = llm.invoke([system_message,
                        HumanMessage(content=state["messages"][-1].content)])

    # Return messages properly for MessagesState
    return {"messages": [answer]}


# Use MessagesState instead of State to match what react_graph expects
graph = StateGraph(MessagesState)
graph.add_node("analysis_agent", react_graph)
graph.add_node("generate_answer", generate_answer)

# TODO add conditional edge based on message classification
# graph.add_edge(START, "generate_answer")
graph.add_edge(START, "analysis_agent")
graph.add_edge("analysis_agent", "generate_answer")
graph.add_edge("generate_answer", END)
# graph.add_edge("team", "generate_enhanced")

# graph.add_edge("generate_enhanced", END)

compiled_graph: CompiledStateGraph = graph.compile(checkpointer=memory)
config: RunnableConfig = {
    "configurable": {"thread_id": "123"},
}

# WORKING STREAM
for chunk in compiled_graph.stream(
        {"messages": [HumanMessage(content="what is day trading?")]},
        config=config,
        stream_mode="messages"):
    # chunk is a tuple: (node_name, message_data)
    message, metadata = chunk
    print(f"Node: {metadata["langgraph_node"]}")
    print(f"Message content: {message.content}")
    print(f"Message type: {type(message)}")
    print("---")
