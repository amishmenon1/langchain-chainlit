# Import necessary modules and define env variables
import os
import chainlit as cl

from dotenv import load_dotenv
from templates.welcome import WELCOME_MSG
from utils.file import load_files_into_db, prompt_file_upload, get_source_file, generate_sources, retrieve_chunks
from utils.message import update_message, new_message
from orchestrator import MultiModelOrchestrator
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from templates.context.medical_history import MEDICAL_HISTORY
from langchain.schema.runnable import Runnable
from typing import cast
from langchain.prompts import ChatPromptTemplate
from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE_2
from langchain.schema import StrOutputParser

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@cl.on_chat_start
async def on_chat_start():
    print(f"TRIGGERED: on_chat_start()")

    elements = [
        # cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    initial_state = {
        "user_name": "Amish Menon",
        "user_question": "",
        "user_context": MEDICAL_HISTORY,
        "extracted_data": ""
    }

    await cl.Message(content=WELCOME_MSG, elements=elements).send()

    # Set initial state - no documents loaded
    cl.user_session.set("documents_loaded", False)
    cl.user_session.set("vector_store", None)
    cl.user_session.set("metadatas", [])
    cl.user_session.set("texts", [])
    cl.user_session.set("extracted_data", [])
    cl.user_session.set("state", initial_state)

    print("✅ Chat started - user can upload files or ask questions directly")

    # Testing OpenBioLLM-70B
    # response = await orchestrator.analyzer_model.ainvoke([
    #     HumanMessage(
    #         content="Explain the medical implications of low hemoglobin and high WBC.")
    # ])
    # print(response.content)


@cl.on_message
async def main(message: cl.Message):
    print(f"TRIGGERED: on_message() with: {message.content}")

    # Check if files are attached to this message
    attached_files = None
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    # If files are attached, process them first
    if attached_files:
        sys_msg_1 = await new_message(
            content="📎 **Files detected!**")

        await load_files_into_db(attached_files)
        # Handle document-based query
        print("📚 Fetching all docs")
        # TODO - based on user query, only fetch relevant docs
        if message.content:
            try:
                # sys_msg_2 = await new_message(content="🔄 Analyzing your question...")

                # Step 1: Retrieve relevant documents with comprehensive approach
                retrieved_text, unique_sources = await retrieve_chunks(
                    message_content=message.content)

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            RETRIEVER_SYSTEM_TEMPLATE_2,
                        ),

                    ]
                )

                # prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                #     <context>
                #     {context}
                #     </context>
                #     Question: {input}""")
                chain = prompt | llm | StrOutputParser()
                answer = cl.Message(content="")
                final_answer = ""

                async for chunk in chain.astream({"input": message.content, "context": retrieved_text}):
                    print(f"chunk: {chunk}")
                    token = chunk
                    if final_answer and token:
                        final_answer += token
                    await answer.stream_token(token)
                await answer.update()

            except Exception as e:
                error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
                print(f"❌ Full error details: {e}")
                await new_message(content=error_msg)


if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
