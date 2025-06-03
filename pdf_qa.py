# Import necessary modules and define env variables
import os
import chainlit as cl

from dotenv import load_dotenv
from templates.welcome import WELCOME_MSG
from utils.file import load_files_into_db, prompt_file_upload, get_source_file, generate_sources
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

        sys_msg_2 = await new_message(
            content=f"🔄 Processing {len(attached_files)} attached file(s)")

        processed_files = await load_files_into_db(attached_files)

        if len(processed_files) > 0:
            await update_message(msg=sys_msg_2, content=f"✅ **Processing Complete!** \nLoaded **{len(processed_files)}** attached file(s): {processed_files}")
            sys_msg_3 = await new_message(content="4. 🔄 Now analyzing your question...")
        else:
            await cl.Message(content="❌ No files were successfully processed from attachments.").send()

    # Check if user wants to upload files via command
    elif message.content.lower().strip() in ['upload', 'upload files', 'add files', 'load documents']:
        await prompt_file_upload()
        return

    # Get current state
    documents_loaded = cl.user_session.get("documents_loaded", False)
    vector_store = cl.user_session.get("vector_store")
    metadatas = cl.user_session.get("metadatas", [])
    texts = cl.user_session.get("texts", [])

    # Handle document-based queries OR general queries with intelligent routing
    if documents_loaded and vector_store:
        # We have documents - use full document analysis pipeline
        print("📚 Documents available - using document analysis mode")

        try:
            # Step 1: Retrieve relevant documents with comprehensive approach
            print("🔍 Step 1a: Comprehensive document retrieval...")
            await update_message(msg=sys_msg_3, content="🔍 Retrieving relevant documents...")

            # First, get documents using similarity search
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 15, })  # Increased to get more coverage
            similarity_docs = await retriever.ainvoke(message.content)

            # Second, ensure we have content from ALL uploaded files
            print("🔍 Step 1b: Ensuring all files are represented...")
            all_file_sources = set()
            for metadata in metadatas:
                all_file_sources.add(metadata['source_file'])

            print(f"📋 Total files referenced: {len(all_file_sources)}")
            print(f"📋 Files: {list(all_file_sources)}")

            # Check which files are represented in similarity search
            similarity_sources = set()
            for doc in similarity_docs:
                for i, text_chunk in enumerate(texts):
                    if text_chunk.page_content == doc.page_content:
                        source_file = metadatas[i].get(
                            'source_file', 'Unknown')
                        similarity_sources.add(source_file)
                        break

            print(
                f"📋 Files found in similarity search: {len(similarity_sources)}")
            print(f"📋 Missing files: {all_file_sources - similarity_sources}")

            # Add documents from missing files to ensure comprehensive coverage
            comprehensive_docs = list(similarity_docs)
            missing_files = all_file_sources - similarity_sources

            if missing_files:
                print(
                    f"🔍 Step 1c: Adding content from {len(missing_files)} missing files...")
                for missing_file in missing_files:
                    # Find representative chunks from missing files
                    file_docs = []
                    for i, text_chunk in enumerate(texts):
                        if metadatas[i].get('source_file') == missing_file:
                            file_docs.append(text_chunk)

                    # Add the first few chunks from each missing file
                    # Add up to 3 chunks per missing file
                    comprehensive_docs.extend(file_docs[:3])

            print(f"📋 Total documents for analysis: {len(comprehensive_docs)}")

            # Combine all retrieved documents with source identification
            retrieved_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([
                f"SOURCE FILE: {get_source_file(doc, texts, metadatas)}\n{doc.page_content}"
                for doc in comprehensive_docs
            ])

            sys_msg_4 = await new_message(content=f"✅ Documents retrieved from {len(all_file_sources)} files")

            unique_sources = generate_sources(
                comprehensive_docs, texts, metadatas)
            # Step 2: Use the orchestrator to process the query
            # final_response = await orchestrator.process_query(message.content, retrieved_text, unique_sources)
            final_response = await orchestrator.process_query(message.content, retrieved_text, unique_sources, sys_msg_3)

            print(f"✅ Final response length: {len(final_response)} characters")

            # Update processing message to final response (NO source elements)
            print(f"📤 Sending final response WITHOUT source elements to avoid raw text")
            final_response_msg = await cl.Message(content=final_response).send()
            final_response_msg.elements = []
            # processor_msg.content = final_response
            # processor_msg.elements = []  # Empty elements to avoid raw text display
            await final_response_msg.update()

        except Exception as e:
            error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
            print(f"❌ Full error details: {e}")
            await cl.Message(content=error_msg).send()

    # Handle queries without documents using intelligent team orchestration
    else:
        print("💬 No documents loaded - using intelligent team orchestration")

        try:
            # Step 1: Team orchestration - determine approach
            msg = cl.Message(
                content="🧭 **Medical AI Team analyzing your question...**")
            await msg.send()

            routing_decision = await orchestrator.route_query(message.content, has_documents=False)

            # Step 2: Execute team coordination based on orchestration decision
            if routing_decision["route"] == "general":
                print("📝 Team Orchestration: Routing to conversational interface")

                msg.content = "💬 **Team preparing response...**"
                await msg.update()

                response = await orchestrator.handle_general_conversation(message.content)

            elif routing_decision["route"] == "medical":
                print("🧬 Team Orchestration: Consulting Medical Expert")
                msg.content = "🧬 **Medical models analyzing your question...**"
                await msg.update()

                # For pure medical questions, coordinate Medical Expert consultation
                response = await orchestrator.handle_mixed_query(message.content)

            else:  # document_analysis or mixed route
                print("🔀 Team Orchestration: Full team coordination needed")

                msg.content = "🤖 **Full Medical AI Team coordination...**"
                await msg.update()

                response = await orchestrator.handle_mixed_query(message.content)

            # Add team-aware guidance for medical questions
            if routing_decision.get("needs_medical_expert", False):
                response += f"\n\n---\n💡 **Team Tip**: Our Medical AI Team can provide detailed analysis of your medical documents. Attach files to your message or type 'upload' for document analysis!"

            # Stream the team-coordinated response
            msg.content = "✅ **Team Response Ready** - Streaming..."
            await msg.update()

            # Create streaming response
            response_msg = cl.Message(content="")
            await response_msg.send()

            for i, char in enumerate(response):
                await response_msg.stream_token(char)
                # response_msg.content += char
                # if i % 25 == 0 or i == len(response) - 1:
                #     await response_msg.update()
                #     await asyncio.sleep(0.02)
            await response_msg.update()

        except Exception as e:
            error_msg = f"❌ **Medical AI Team Error**: {str(e)}\n\nOur team is having trouble processing your request. Please ensure all AI services are running."
            await cl.Message(content=error_msg).send()
            print(f"Error in team orchestration: {e}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
