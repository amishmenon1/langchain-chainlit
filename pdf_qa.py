# Import necessary modules and define env variables
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# Use this instead of community version
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import json
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Different system templates for different models
RETRIEVER_SYSTEM_TEMPLATE = """You are a document retrieval specialist. Extract lab information from the medical documents.

Please organize the information like this:

NORMAL LAB RESULTS:
- Test Name: Value Unit (Location)
- Test Name: Value Unit (Location)

ABNORMAL LAB RESULTS:
- Test Name: Value Unit - Status (Location)
- Test Name: Value Unit - Status (Location)

TRENDS AND OBSERVATIONS:
- Notable trend 1
- Notable trend 2

ADDITIONAL INFORMATION:
- Key finding 1
- Key finding 2

Extract specific lab values, their units, normal/abnormal status, and any trends mentioned.

Context:
{summaries}"""

ANALYZER_SYSTEM_TEMPLATE = """You are a medical data analysis expert specializing in biomedical insights. 

Analyze the following extracted lab data and provide clinical insights:
- Identify patterns in the lab results
- Explain what abnormal values might indicate
- Suggest potential correlations between different parameters
- Provide medical context for the findings

Be specific and use your biomedical knowledge to provide meaningful analysis.

Data to analyze:
{extracted_data}"""

FORMATTER_SYSTEM_TEMPLATE = """You are a medical communication specialist. Format the provided analysis into a clear, well-structured markdown response.

CRITICAL FORMATTING RULES:
1. Start with "## Lab Results Summary"
2. Use proper markdown formatting throughout
3. Create clear sections: Normal Results, Abnormal Results, Clinical Analysis
4. Use bullet points and bold text for readability
5. DO NOT include a Sources section - this will be added separately

Take this raw analysis and format it beautifully:
{raw_analysis}

Original user question: {user_question}"""


class MultiModelOrchestrator:
    def __init__(self):
        # Model 1: OpenAI for formatting and communication (good at following instructions)
        self.formatter_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=2000
        )

        # Model 2: OpenAI for document retrieval (consistent and reliable)
        self.retriever_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=1500
        )

        # Model 3: LM Studio BioLLM for medical analysis (domain expertise)
        self.analyzer_model = ChatOpenAI(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model="openbiollm-llama3-8b",
            temperature=0.2,
            max_tokens=2000
        )

    async def process_query(self, user_question: str, retrieved_docs: str) -> str:
        """Orchestrate the three-model pipeline"""

        # Step 1: Use retriever model to extract structured data
        print("🔍 Step 1: Extracting data with retriever model...")
        print("📝 Using simplified structured format")
        retriever_prompt = RETRIEVER_SYSTEM_TEMPLATE.format(
            summaries=retrieved_docs)

        retriever_messages = [
            SystemMessage(content=retriever_prompt),
            HumanMessage(
                content=f"Extract information relevant to: {user_question}")
        ]

        try:
            extracted_response = await self.retriever_model.ainvoke(retriever_messages)
            raw_extracted = extracted_response.content
            print(f"✅ Raw extracted data: {raw_extracted[:200]}...")

            # Use raw extracted content directly - no JSON parsing at all
            print(f"🔄 Using raw extracted content directly")
            extracted_data = f"Extracted Information:\n{raw_extracted}"

        except Exception as e:
            print(f"❌ Retriever model error: {e}")
            extracted_data = f"Raw context: {retrieved_docs}"

        # Step 2: Use BioLLM for medical analysis
        print("🧬 Step 2: Analyzing with BioLLM...")
        analyzer_prompt = ANALYZER_SYSTEM_TEMPLATE.format(
            extracted_data=extracted_data)

        analyzer_messages = [
            SystemMessage(content=analyzer_prompt),
            HumanMessage(
                content=f"Provide biomedical analysis for: {user_question}")
        ]

        try:
            analysis_response = await self.analyzer_model.ainvoke(analyzer_messages)
            raw_analysis = analysis_response.content
            print(f"✅ Analysis complete: {raw_analysis[:200]}...")
        except Exception as e:
            print(f"❌ Analyzer model error: {e}")
            # Create a fallback analysis if BioLLM fails
            try:
                # Try to create a basic analysis from the extracted data
                if "lab_results" in extracted_data or "NORMAL LAB" in extracted_data:
                    raw_analysis = f"Based on the extracted lab data:\n{extracted_data}\n\nBasic analysis: Multiple lab parameters were measured with some values outside normal ranges."
                else:
                    raw_analysis = f"Analysis unavailable. Raw data: {extracted_data}"
            except:
                raw_analysis = f"Error in analysis. Raw data: {extracted_data}"

        # Step 3: Use formatter model for final presentation
        print("✨ Step 3: Formatting with OpenAI...")
        formatter_prompt = FORMATTER_SYSTEM_TEMPLATE.format(
            raw_analysis=raw_analysis,
            user_question=user_question
        )

        formatter_messages = [
            SystemMessage(content=formatter_prompt),
            HumanMessage(
                content="Format this into a beautiful, well-structured response.")
        ]

        try:
            formatted_response = await self.formatter_model.ainvoke(formatter_messages)
            final_answer = formatted_response.content
            print("✅ Formatting complete!")

            # Ensure the response has proper structure
            if not final_answer.strip().startswith('##'):
                final_answer = f"## Lab Results Summary\n\n{final_answer}"

            # Clean up any extra whitespace or formatting issues
            final_answer = final_answer.strip()
            print(f"📝 Final formatted answer length: {len(final_answer)}")

        except Exception as e:
            print(f"❌ Formatter model error: {e}")
            # Create a basic formatted response as fallback
            final_answer = f"## Lab Results Summary\n\n{raw_analysis}"

        return final_answer


async def process_uploaded_files():
    """Handle file upload and processing"""
    # Ask for file upload
    files = await cl.AskFileMessage(
        content="Please upload one or more PDF files! 📄",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
        max_files=10
    ).send()

    if not files:
        await cl.Message(content="No files uploaded. You can try again anytime!").send()
        return False

    # Processing message
    file_names = [f.name for f in files]
    msg = cl.Message(
        content=f"🔄 Processing {len(files)} file(s): {', '.join(file_names)}...")
    await msg.send()

    # Process all uploaded files
    all_pages = []
    processed_files = []

    for file in files:
        print(f"Loading file: {file.name}")
        try:
            loader = PyPDFLoader(file.path)
            pages = []
            async for page in loader.alazy_load():
                page.metadata['source_file'] = file.name
                pages.append(page)

            all_pages.extend(pages)
            processed_files.append(file.name)
            print(f"Successfully processed: {file.name} ({len(pages)} pages)")

        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            await cl.Message(content=f"❌ Error processing {file.name}: {str(e)}").send()

    if not all_pages:
        await cl.Message(content="❌ No files were successfully processed. Please try again.").send()
        return False

    # Split the text into chunks
    all_splits = text_splitter.split_documents(all_pages)
    print(f"Length of all text splits: {len(all_splits)}")

    # Create vector store with all documents
    vector_store = Chroma.from_documents(all_splits, OpenAIEmbeddings())

    # Update metadata to include source file information
    metadatas = []
    for i, split in enumerate(all_splits):
        source_file = split.metadata.get('source_file', 'unknown')
        page_num = split.metadata.get('page', 0)
        metadatas.append({
            "source": f"{i}-pl",
            "source_file": source_file,
            "page": page_num
        })

    # Store everything in user session
    cl.user_session.set("vector_store", vector_store)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", all_splits)
    cl.user_session.set("documents_loaded", True)

    # Success message
    file_list = ", ".join(processed_files)
    msg.content = f"✅ **Processing Complete!** \n\nLoaded **{len(processed_files)}** file(s): {file_list}\n\n🚀 **Multi-Model Pipeline Ready:**\n- 📊 Vector database created\n- 🤖 Three AI models standing by\n- 💬 Ready for your questions!"
    await msg.update()

    return True

# Initialize the orchestrator
orchestrator = MultiModelOrchestrator()


@cl.on_chat_start
async def on_chat_start():
    print(f"TRIGGERED: on_chat_start()")

    # Welcome message with options
    welcome_msg = """Hello! Welcome to your **Multi-Model AI Medical Assistant**! 🤖🧬

I use three specialized AI models:
- 📄 **Document Retriever**: Extracts data from your PDFs
- 🧬 **BioMedical Analyzer**: Provides domain expertise  
- ✨ **Smart Formatter**: Creates beautiful, readable responses

**Choose how you'd like to start:**
- 📄 **Upload medical documents** for detailed analysis
- 💬 **Ask me anything** about medical topics (general knowledge)

You can upload documents later by typing 'upload' or using the paperclip icon! 📎"""

    elements = [
        # cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(content=welcome_msg, elements=elements).send()

    # Set initial state - no documents loaded
    cl.user_session.set("documents_loaded", False)
    cl.user_session.set("vector_store", None)
    cl.user_session.set("metadatas", [])
    cl.user_session.set("texts", [])

    print("✅ Chat started - user can upload files or ask questions directly")


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
        await cl.Message(content="📎 **Files detected!** Processing your attached documents first...").send()

        # Process the attached files
        file_names = [f.name for f in attached_files]
        msg = cl.Message(
            content=f"🔄 Processing {len(attached_files)} attached file(s): {', '.join(file_names)}...")
        await msg.send()

        # Process all attached files
        all_pages = []
        processed_files = []

        for file in attached_files:
            print(f"Loading attached file: {file.name}")
            try:
                loader = PyPDFLoader(file.path)
                pages = []
                async for page in loader.alazy_load():
                    page.metadata['source_file'] = file.name
                    pages.append(page)

                all_pages.extend(pages)
                processed_files.append(file.name)
                print(
                    f"Successfully processed: {file.name} ({len(pages)} pages)")

            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
                await cl.Message(content=f"❌ Error processing {file.name}: {str(e)}").send()

        if all_pages:
            # Split the text into chunks
            all_splits = text_splitter.split_documents(all_pages)
            print(f"Length of all text splits: {len(all_splits)}")

            # Create vector store with all documents
            vector_store = Chroma.from_documents(
                all_splits, OpenAIEmbeddings())

            # Update metadata to include source file information
            metadatas = []
            for i, split in enumerate(all_splits):
                source_file = split.metadata.get('source_file', 'unknown')
                page_num = split.metadata.get('page', 0)
                metadatas.append({
                    "source": f"{i}-pl",
                    "source_file": source_file,
                    "page": page_num
                })

            # Store everything in user session (merge with existing if any)
            existing_texts = cl.user_session.get("texts", [])
            existing_metadatas = cl.user_session.get("metadatas", [])

            cl.user_session.set("vector_store", vector_store)
            cl.user_session.set("metadatas", existing_metadatas + metadatas)
            cl.user_session.set("texts", existing_texts + all_splits)
            cl.user_session.set("documents_loaded", True)

            # Success message
            file_list = ", ".join(processed_files)
            msg.content = f"✅ **Processing Complete!** \n\nLoaded **{len(processed_files)}** attached file(s): {file_list}\n\n🔄 Now analyzing your question..."
            await msg.update()
        else:
            await cl.Message(content="❌ No files were successfully processed from attachments.").send()
            return

    # Check if user wants to upload files via command
    elif message.content.lower().strip() in ['upload', 'upload files', 'add files', 'load documents']:
        await process_uploaded_files()
        return

    # Get current state
    documents_loaded = cl.user_session.get("documents_loaded", False)
    vector_store = cl.user_session.get("vector_store")
    metadatas = cl.user_session.get("metadatas", [])
    texts = cl.user_session.get("texts", [])

    # Handle document-based queries
    if documents_loaded and vector_store:
        # Show processing message for document analysis
        processing_msg = cl.Message(
            content="🤖 **Multi-Model Processing Pipeline Started...**\n\n🔍 Retrieving relevant documents...")
        await processing_msg.send()

        try:
            # Step 1: Retrieve relevant documents
            retriever = vector_store.as_retriever(search_kwargs={"k": 6})
            docs = await retriever.ainvoke(message.content)

            # Combine retrieved documents
            retrieved_text = "\n\n".join([doc.page_content for doc in docs])

            # Update processing message
            processing_msg.content = "🤖 **Multi-Model Processing Pipeline Started...**\n\n✅ Documents retrieved\n🧬 Analyzing with specialized models..."
            await processing_msg.update()

            # Step 2: Use the orchestrator to process the query
            final_response = await orchestrator.process_query(message.content, retrieved_text)

            # Step 3: Create source list (no source elements to avoid raw text display)
            doc_sources = []
            seen_sources = set()  # Track unique source files

            for doc in docs:
                # Find the metadata for this document
                for i, text_chunk in enumerate(texts):
                    if text_chunk.page_content == doc.page_content:
                        metadata = metadatas[i]
                        source_file = metadata.get('source_file', 'Unknown')

                        # Create a unique identifier for source files
                        source_key = f"{source_file}"

                        # Only add if we haven't seen this file before
                        if source_key not in seen_sources:
                            doc_sources.append(f"**{source_file}**")
                            seen_sources.add(source_key)
                        break

            # Add properly formatted sources to the response (unique files only)
            if doc_sources:
                # Sort sources alphabetically
                unique_sources = sorted(list(set(doc_sources)))
                final_response += f"\n\n## Sources\n" + \
                    "\n".join([f"- {source}" for source in unique_sources])

            print(f"✅ Final response length: {len(final_response)} characters")
            print(f"✅ Response ends with: ...{final_response[-100:]}")

            # Update processing message to final response (NO source elements)
            print(f"📤 Sending final response WITHOUT source elements to avoid raw text")
            processing_msg.content = final_response
            processing_msg.elements = []  # Empty elements to avoid raw text display
            await processing_msg.update()

        except Exception as e:
            error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
            print(f"❌ Full error details: {e}")
            await cl.Message(content=error_msg).send()

    # Handle general medical questions (no documents)
    else:
        try:
            # Show that we're using general knowledge mode
            processing_msg = cl.Message(
                content="🧬 **Consulting BioMedical AI...**")
            await processing_msg.send()

            # Use only the BioLLM for general medical questions
            general_prompt = f"""You are a helpful medical AI assistant. Answer the following question using your biomedical knowledge:

Question: {message.content}

Provide a clear, informative response with proper medical context. If this is a specific medical question that would benefit from document analysis, suggest that the user upload relevant medical documents.

Format your response with proper markdown including headers, bullet points, and bold text for readability."""

            general_messages = [
                SystemMessage(content="You are a knowledgeable medical AI assistant. Provide helpful, accurate medical information while reminding users to consult healthcare professionals for personalized advice."),
                HumanMessage(content=general_prompt)
            ]

            # Use the BioLLM for medical expertise
            response = await orchestrator.analyzer_model.ainvoke(general_messages)
            raw_answer = response.content

            # Format the response nicely
            if not raw_answer.strip().startswith('##'):
                formatted_answer = f"## Medical Information\n\n{raw_answer}"
            else:
                formatted_answer = raw_answer

            # Add helpful note about document upload
            formatted_answer += f"\n\n---\n💡 **Tip**: For analysis of specific lab results or medical documents, attach files to your message or type 'upload'!"

            # Update the processing message
            processing_msg.content = formatted_answer
            await processing_msg.update()

        except Exception as e:
            error_msg = f"❌ **Error**: {str(e)}\n\nI'm having trouble connecting to the medical AI. Please make sure LM Studio is running."
            await cl.Message(content=error_msg).send()
            print(f"Error in general query: {e}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
