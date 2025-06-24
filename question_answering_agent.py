from google.adk.agents import Agent
from templates.system.analyzer import ANALYZER_SYSTEM_TEMPLATE

# Create the root agent
question_answering_agent = Agent(
    name="question_answering_agent",
    model="gemini-2.0-flash",
    description="Question answering agent",
    # instruction=ANALYZER_SYSTEM_TEMPLATE,
    instruction="""
    You are a helpful assistant that answers questions given the user's context and attached files.

    Here is some information about the user:
    Context:
    {user_context}
    Relevant files:
    {extracted_data}
    """,
)
