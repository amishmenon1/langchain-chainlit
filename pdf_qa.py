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

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the orchestrator
orchestrator = MultiModelOrchestrator()


@cl.on_chat_start
async def on_chat_start():
    print(f"TRIGGERED: on_chat_start()")

    elements = [
        # cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(content=WELCOME_MSG, elements=elements).send()

    # Set initial state - no documents loaded
    cl.user_session.set("documents_loaded", False)
    cl.user_session.set("vector_store", None)
    cl.user_session.set("metadatas", [])
    cl.user_session.set("texts", [])

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

                # Step 2: Use the orchestrator to process the query
                await orchestrator.process_query(message.content, retrieved_text, unique_sources)

            except Exception as e:
                error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
                print(f"❌ Full error details: {e}")
                await new_message(content=error_msg)

    # Check if user wants to upload files via command
    elif message.content.lower().strip() in ['upload', 'upload files', 'add files', 'load documents']:
        await prompt_file_upload()
        return

    # Handle queries without documents using intelligent team orchestration
    else:
        print("💬 No documents loaded - using intelligent team orchestration")

        try:
            # Step 1: Team orchestration - determine approach
            routing_decision = await orchestrator.route_query(message.content, has_documents=False)

            # Step 2: Execute team coordination based on orchestration decision
            if routing_decision["route"] == "general":
                print("📝 Team Orchestration: Routing to conversational interface")
                response = await orchestrator.handle_general_conversation(message.content)

            elif routing_decision["route"] == "medical":
                print("🧬 Team Orchestration: Consulting Medical Expert")
                med_msg = await new_message("🧬 **Medical models analyzing your question...**")
                # For pure medical questions, coordinate Medical Expert consultation
                response = await orchestrator.handle_mixed_query(message.content)

            else:  # document_analysis or mixed route
                print("🔀 Team Orchestration: Full team coordination needed")
                response = await orchestrator.handle_mixed_query(message.content)

            # Add team-aware guidance for medical questions
            if routing_decision.get("needs_medical_expert", False):
                response += f"\n\n---\n💡 **Team Tip**: Our Medical AI Team can provide detailed analysis of your medical documents. Attach files to your message or type 'upload' for document analysis!"

            # await update_message(msg=msg, content="✅ **Team Response Ready** - Streaming...")

            # # Create streaming response
            # response_msg = cl.Message(content="")
            # await response_msg.send()

            # for i, char in enumerate(response):
            #     await response_msg.stream_token(char)
            #     # response_msg.content += char
            #     # if i % 25 == 0 or i == len(response) - 1:
            #     #     await response_msg.update()
            #     #     await asyncio.sleep(0.02)
            # await response_msg.update()

        except Exception as e:
            error_msg = f"❌ **Medical AI Team Error**: {str(e)}\n\nOur team is having trouble processing your request. Please ensure all AI services are running."
            await cl.Message(content=error_msg).send()
            print(f"Error in team orchestration: {e}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
