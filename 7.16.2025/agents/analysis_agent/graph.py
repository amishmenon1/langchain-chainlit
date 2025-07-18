"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""


from IPython.display import Image, display
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, StateGraph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from agents.analysis_agent.tools import TOOLS
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

### DEFINE AGENT TOOLS ###
# define more custom functions here and use Tool.from_function to create tools

simple_llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = simple_llm.bind_tools(TOOLS)
memory = InMemorySaver()

### DEFINE AGENT NODE ###


# System message
assistant_system_message = SystemMessage(content=("""
You are a professional financial assistant specializing in stock market analysis and investment strategies. 
Your role is to analyze stock data and provide **clear, decisive recommendations** that users can act on, 
whether they already hold the stock or are considering investing.

You have access to a set of tools that can provide the data you need to analyze stocks effectively. 
Use these tools to gather relevant information such as stock symbols, current prices, historical trends, 
and key financial indicators. Your goal is to leverage these resources efficiently to generate accurate, 
actionable insights for the user.

Your responses should be:
- **Concise and direct**, summarizing only the most critical insights.
- **Actionable**, offering clear guidance on whether to buy, sell, hold, or wait for better opportunities.
- **Context-aware**, considering both current holders and potential investors.
- **Free of speculation**, relying solely on factual data and trends.

### Response Format:
1. **Recommendation:** Buy, Sell, Hold, or Wait.
2. **Key Insights:** Highlight critical trends and market factors that influence the decision.
3. **Suggested Next Steps:** What the user should do based on their current position.

If the user does not specify whether they own the stock, provide recommendations for both potential buyers and current holders. Ensure your advice considers valuation, trends, and market sentiment.

Your goal is to help users make informed financial decisions quickly and confidently.
"""))

# Node 1


def optimize_request(state: MessagesState):
    """Optimize the request for the analysis agent."""
    print("\n\nNode - Optimize request...\n\n")
    # TODO return optimized_query (str - should exist on internal graph state)
    pass

# TODO create internal custom state AnalysisAgentState - optimized_query (str)
# TODO ONLY return structured output MedicalAnalysis to parent graph's generate_answer


def assistant(state: MessagesState):
    print("\n\nNode - analysis assistant...\n\n")
    response = llm_with_tools.invoke(
        [assistant_system_message] + state["messages"])

    # TODO return MedicalAnalysis object (should exist on parent state)
    return {"messages": [response]}

# Node 2 - Tool Node -- defined in the graph edges

### DEFINE AGENT GRAPH ###


# Graph
builder = StateGraph(MessagesState)

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
