# Import necessary modules and define env variables
import os
import chainlit as cl

import uuid

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from question_answering_agent import question_answering_agent
from templates.welcome import WELCOME_MSG
from utils.file import load_files_into_db, prompt_file_upload, get_source_file, generate_sources, retrieve_chunks
from utils.message import update_message, new_message
from orchestrator import MultiModelOrchestrator
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Create a new session service to store state
session_service_stateful = InMemorySessionService()


@cl.on_chat_start
async def on_chat_start():
    print(f"TRIGGERED: on_chat_start()")

    initial_state = {
        "user_name": "Brandon Hancock",
        "user_preferences": """
            I like to play Pickleball, Disc Golf, and Tennis.
            My favorite food is Mexican.
            My favorite TV show is Game of Thrones.
            Loves it when people like and subscribe to his YouTube channel.
        """,
    }

    # Create a NEW session
    APP_NAME = "Brandon Bot"
    USER_ID = cl.user_session.get("id")
    SESSION_ID = str(uuid.uuid4())
    stateful_session = await session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )
    print("CREATED NEW SESSION:")
    print(f"\tSession ID: {SESSION_ID}")

    runner = Runner(
        agent=question_answering_agent,
        app_name=APP_NAME,
        session_service=session_service_stateful,
    )

    cl.user_session.set("adk_runner", runner)
    cl.user_session.set("adk_session_id", SESSION_ID)
    cl.user_session.set("adk_user_id", USER_ID)
    cl.user_session.set("adk_app_name", APP_NAME)


@cl.on_message
async def main(message: cl.Message):
    print(f"TRIGGERED: on_message() with: {message.content}")

    new_message = types.Content(
        role="user", parts=[types.Part(text=message.content)]
    )
    runner = cl.user_session.get("adk_runner")
    SESSION_ID = cl.user_session.get("adk_session_id")
    USER_ID = cl.user_session.get("adk_user_id")
    APP_NAME = cl.user_session.get("adk_app_name")

    # Create a Chainlit message container for streaming the response
    msg = cl.Message(content="")
    await msg.send()

    final_response_content = ""

    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                # print(f"Final Response: {event.content.parts[0].text}")
                part_text = event.content.parts[0].text
                # Avoid duplicates if streaming content already covered final response
                if part_text not in final_response_content:
                    await msg.stream_token(part_text)
                # Ensure final_response_content has the complete final response
                final_response_content = part_text

    await msg.update()

    print("==== Session Event Exploration ====")
    session = await session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    # Log final Session state
    print("=== Final Session State ===")
    for key, value in session.state.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
