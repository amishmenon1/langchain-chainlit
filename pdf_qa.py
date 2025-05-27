# Import necessary modules and define env variables
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
import os
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import json
from typing import Dict, Any
from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE
from templates.system.formatter import FORMATTER_SYSTEM_TEMPLATE
from templates.system.analyzer import ANALYZER_SYSTEM_TEMPLATE
from templates.welcome import WELCOME_MSG
from templates.human.prompts import generate_routing_prompt, generate_general_conversation_prompt, generate_mixed_conversation_prompt, generate_medical_prompt, generate_analyzer_prompt, generate_eval_prompt, generate_alt_analysis_prompt, generate_alt_enhancement_prompt

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Different system templates for different models


def get_source_file(doc, texts, metadatas):
    """Helper function to get source file name for a document"""
    for i, text_chunk in enumerate(texts):
        if text_chunk.page_content == doc.page_content:
            return metadatas[i].get('source_file', 'Unknown')
    return 'Unknown'


class MultiModelOrchestrator:
    def __init__(self):
        # Model 1: OpenAI for formatting and communication (good at following instructions)
        self.formatter_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4000
        )

        # Model 2: OpenAI for document retrieval (consistent and reliable)
        self.retriever_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=300
        )

        # Model 3: LM Studio BioLLM for medical analysis (domain expertise)
        self.analyzer_model = ChatOpenAI(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model="openbiollm-llama3-8b",
            temperature=0.1,  # Lower temperature for more focused medical analysis
            max_tokens=4000   # Increased for comprehensive analysis
        )

        # Model 4: OpenAI as conversation router/manager
        self.router_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4000
        )

    async def route_query(self, user_question: str, has_documents: bool = False) -> dict:
        """Determine how to handle the user's query with orchestration awareness"""

        routing_prompt = generate_routing_prompt(user_question, has_documents)
        try:
            routing_messages = [
                SystemMessage(
                    content="You are a Medical AI Team Orchestrator. Always respond with valid JSON and coordinate team resources efficiently."),
                HumanMessage(content=routing_prompt)
            ]

            response = await self.router_model.ainvoke(routing_messages)

            # Try to parse the JSON response
            import json
            routing_decision = json.loads(response.content.strip())
            print(f"🧭 Team Orchestration: {routing_decision}")
            return routing_decision

        except Exception as e:
            print(f"❌ Orchestration routing error: {e}")
            # Default to medical route if routing fails
            return {
                "route": "medical",
                "reasoning": "orchestration routing failed, defaulting to medical expert consultation",
                "needs_medical_expert": True,
                "needs_document_retrieval": has_documents,
                "response_type": "clinical_analysis",
                "orchestration_notes": "fallback to direct medical expert consultation"
            }

    async def handle_general_conversation(self, user_question: str) -> str:
        """Handle non-medical general conversation with orchestration awareness"""

        conversation_prompt = generate_general_conversation_prompt(
            user_question)
        try:
            conv_messages = [
                SystemMessage(
                    content="You are the friendly communication interface for a Medical AI Team. Be helpful, warm, and represent the team's collaborative capabilities."),
                HumanMessage(content=conversation_prompt)
            ]

            response = await self.router_model.ainvoke(conv_messages)
            return response.content

        except Exception as e:
            print(f"❌ Team conversation error: {e}")
            return "Hello! I'm part of a Medical AI Team here to help with medical questions and document analysis. What can our team assist you with today?"

    async def handle_mixed_query(self, user_question: str, retrieved_docs: str = None) -> str:
        """Handle queries that need both conversational flow and medical expertise"""

        # First, get medical insights if we have documents or medical questions
        medical_content = ""
        if retrieved_docs:
            # Use document analysis pipeline
            medical_content = await self.process_query(user_question, retrieved_docs)
        else:
            # Consult BioLLM for medical knowledge
            medical_content = await self.get_medical_insights(user_question)

        # Then, have OpenAI create a conversational response incorporating the medical info
        conversation_prompt = generate_mixed_conversation_prompt(
            user_question, medical_content)
        try:
            conv_messages = [
                SystemMessage(
                    content="You are a conversational medical AI that combines technical knowledge with friendly communication."),
                HumanMessage(content=conversation_prompt)
            ]

            response = await self.router_model.ainvoke(conv_messages)
            return response.content

        except Exception as e:
            print(f"❌ Mixed query error: {e}")
            return medical_content  # Fallback to just medical content

    async def get_medical_insights(self, user_question: str) -> str:
        """Get medical insights from BioLLM for general medical questions with clinical expertise"""

        medical_prompt = generate_medical_prompt(user_question)

        try:
            medical_messages = [
                SystemMessage(content="You are a Medical Expert who provides systematic, evidence-based medical analysis and education. Always demonstrate clinical reasoning and reference medical knowledge appropriately."),
                HumanMessage(content=medical_prompt)
            ]

            response = await self.analyzer_model.ainvoke(medical_messages)
            return response.content

        except Exception as e:
            print(f"❌ Medical insights error: {e}")
            return f"I encountered an issue accessing medical information. Please ensure the BioLLM is running."

    async def process_query(self, user_question: str, retrieved_docs: str) -> str:
        """Original document analysis pipeline (unchanged)"""

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

        # Step 2: Use BioLLM for medical analysis with clinical expertise
        print("🧬 Step 2: Analyzing with Medical Expert (BioLLM)...")
        analyzer_prompt = generate_analyzer_prompt(
            user_question, extracted_data)
        analyzer_messages = [
            SystemMessage(content="You are a Medical Education Expert who provides comprehensive educational analysis of laboratory data for learning purposes. Your role is to demonstrate clinical reasoning, lab interpretation, and medical decision-making processes using provided lab values as educational examples. Always provide detailed educational analysis to help learners understand medical concepts."),
            HumanMessage(content=analyzer_prompt)
        ]

        try:
            analysis_response = await self.analyzer_model.ainvoke(analyzer_messages)
            raw_analysis = analysis_response.content
            print(f"✅ Initial analysis complete: {raw_analysis[:200]}...")

            # Check if the BioLLM refused to analyze
            if "cannot generate" in raw_analysis.lower() or "unable to" in raw_analysis.lower() or "sorry" in raw_analysis.lower():
                print("⚠️ BioLLM refused analysis - using alternative approach")
                raw_analysis = await self.create_alternative_analysis(extracted_data, user_question)
            else:
                # Step 2b: Self-evaluation and enhancement
                print("🔍 Step 2b: Self-evaluating and enhancing analysis...")
                raw_analysis = await self.self_evaluate_and_enhance(raw_analysis, extracted_data, user_question)

        except Exception as e:
            print(f"❌ Analyzer model error: {e}")
            # Create alternative analysis if BioLLM fails
            raw_analysis = await self.create_alternative_analysis(extracted_data, user_question)

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
                final_answer = f"## Medical Analysis\n\n{final_answer}"

            # Clean up any extra whitespace or formatting issues
            final_answer = final_answer.strip()
            print(f"📝 Final formatted answer length: {len(final_answer)}")

        except Exception as e:
            print(f"❌ Formatter model error: {e}")
            # Create a basic formatted response as fallback
            final_answer = f"## Medical Analysis\n\n{raw_analysis}"

        return final_answer

    async def self_evaluate_and_enhance(self, initial_analysis: str, extracted_data: str, user_question: str) -> str:
        """Self-evaluate the initial analysis and enhance it for comprehensiveness"""

        evaluation_prompt = generate_eval_prompt(
            user_question, extracted_data, initial_analysis)
        try:
            evaluation_messages = [
                SystemMessage(content="You are a Medical Expert performing rigorous quality assurance and enhancement of medical analysis. Always provide comprehensive, evidence-based medical analysis that exceeds clinical standards."),
                HumanMessage(content=evaluation_prompt)
            ]

            enhanced_response = await self.analyzer_model.ainvoke(evaluation_messages)
            enhanced_analysis = enhanced_response.content
            print(
                f"✅ Enhanced analysis complete: {enhanced_analysis[:200]}...")

            # Check if enhancement was successful (longer and more detailed)
            if len(enhanced_analysis) > len(initial_analysis) * 1.2:  # At least 20% longer
                return enhanced_analysis
            else:
                print("⚠️ Enhancement insufficient - using alternative enhancement")
                return await self.alternative_enhancement(initial_analysis, extracted_data, user_question)

        except Exception as e:
            print(f"❌ Self-evaluation error: {e}")
            return await self.alternative_enhancement(initial_analysis, extracted_data, user_question)

    async def alternative_enhancement(self, initial_analysis: str, extracted_data: str, user_question: str) -> str:
        """Use OpenAI to enhance the analysis if BioLLM enhancement fails"""

        enhancement_prompt = generate_alt_enhancement_prompt(
            user_question, extracted_data, initial_analysis)

        try:
            enhancement_messages = [
                SystemMessage(
                    content="You are a Medical Analysis Expert who enhances medical analyses to meet the highest clinical standards. Be thorough, specific, and comprehensive."),
                HumanMessage(content=enhancement_prompt)
            ]

            # Use OpenAI for enhancement
            response = await self.formatter_model.ainvoke(enhancement_messages)
            return response.content

        except Exception as e:
            print(f"❌ Alternative enhancement error: {e}")
            return initial_analysis  # Return original if all enhancement fails

    async def create_alternative_analysis(self, extracted_data: str, user_question: str) -> str:
        """Create comprehensive analysis using OpenAI when BioLLM refuses"""

        alternative_prompt = generate_alt_analysis_prompt(
            user_question, extracted_data)

        try:
            alternative_messages = [
                SystemMessage(
                    content="You are a Medical Analysis Assistant providing educational lab interpretation. Be comprehensive and specific in your analysis."),
                HumanMessage(content=alternative_prompt)
            ]

            # Use OpenAI formatter model
            response = await self.formatter_model.ainvoke(alternative_messages)
            return response.content

        except Exception as e:
            print(f"❌ Alternative analysis error: {e}")
            return f"Comprehensive analysis of laboratory data:\n{extracted_data}\n\nDetailed medical interpretation and recommendations would be provided here based on the specific lab values found in the uploaded documents."


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

    # Handle document-based queries OR general queries with intelligent routing
    if documents_loaded and vector_store:
        # We have documents - use full document analysis pipeline
        print("📚 Documents available - using document analysis mode")

        # Show processing message for document analysis
        # processing_msg = cl.Message(
        #     content="🤖 **Multi-Model Processing Pipeline Started...**\n\n🔍 Retrieving relevant documents...")
        # await processing_msg.send()

        # try yielding
        msg.content = "🤖 **Multi-Model Processing Pipeline Started...**\n\n🔍 Retrieving relevant documents..."
        await msg.update()

        try:
            # Step 1: Retrieve relevant documents with comprehensive approach
            print("🔍 Step 1a: Comprehensive document retrieval...")

            # First, get documents using similarity search
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 15})  # Increased to get more coverage
            similarity_docs = await retriever.ainvoke(message.content)

            # Second, ensure we have content from ALL uploaded files
            print("🔍 Step 1b: Ensuring all files are represented...")
            all_file_sources = set()
            for metadata in metadatas:
                all_file_sources.add(metadata['source_file'])

            print(f"📋 Total files uploaded: {len(all_file_sources)}")
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

            # # Update processing message
            # processing_msg.content = f"🤖 **Multi-Model Processing Pipeline Started...**\n\n✅ Documents retrieved from {len(all_file_sources)} files\n🧬 Analyzing with specialized models..."
            # await processing_msg.update()

            # Update processing message
            msg.content = f"🤖 **Multi-Model Processing Pipeline Started...**\n\n✅ Documents retrieved from {len(all_file_sources)} files\n🧬 Analyzing with specialized models..."
            await msg.update()

            # Step 2: Use the orchestrator to process the query
            final_response = await orchestrator.process_query(message.content, retrieved_text)

            # Step 3: Create source list (no source elements to avoid raw text display)
            doc_sources = []
            seen_sources = set()  # Track unique source files

            for doc in comprehensive_docs:
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

            # Update processing message to final response (NO source elements)
            print(f"📤 Sending final response WITHOUT source elements to avoid raw text")
            # processing_msg.content = final_response
            # processing_msg.elements = []  # Empty elements to avoid raw text display
            # await processing_msg.update()

            msg.content = final_response
            msg.elements = []  # Empty elements to avoid raw text display
            await msg.update()

        except Exception as e:
            error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
            print(f"❌ Full error details: {e}")
            await cl.Message(content=error_msg).send()

    # Handle queries without documents using intelligent team orchestration
    else:
        print("💬 No documents loaded - using intelligent team orchestration")

        try:
            # # Step 1: Team orchestration - determine approach
            # processing_msg = cl.Message(
            #     content="🧭 **Medical AI Team analyzing your question...**")
            # await processing_msg.send()

            # Step 1: Team orchestration - determine approach
            msg = cl.Message(
                content="🧭 **Medical AI Team analyzing your question...**")
            await msg.send()

            routing_decision = await orchestrator.route_query(message.content, has_documents=False)

            # Step 2: Execute team coordination based on orchestration decision
            if routing_decision["route"] == "general":
                print("📝 Team Orchestration: Routing to conversational interface")
                # processing_msg.content = "💬 **Team preparing response...**"
                # await processing_msg.update()

                msg.content = "💬 **Team preparing response...**"
                await msg.update()

                response = await orchestrator.handle_general_conversation(message.content)

            elif routing_decision["route"] == "medical":
                print("🧬 Team Orchestration: Consulting Medical Expert")
                # processing_msg.content = "🧬 **Medical Expert analyzing your question...**"
                # await processing_msg.update()

                msg.content = "🧬 **Medical Expert analyzing your question...**"
                await msg.update()

                # For pure medical questions, coordinate Medical Expert consultation
                response = await orchestrator.handle_mixed_query(message.content)

            else:  # document_analysis or mixed route
                print("🔀 Team Orchestration: Full team coordination needed")
                # processing_msg.content = "🤖 **Full Medical AI Team coordination...**"
                # await processing_msg.update()

                msg.content = "🤖 **Full Medical AI Team coordination...**"
                await msg.update()

                response = await orchestrator.handle_mixed_query(message.content)

            # Add team-aware guidance for medical questions
            if routing_decision.get("needs_medical_expert", False):
                response += f"\n\n---\n💡 **Team Tip**: Our Medical AI Team can provide detailed analysis of your medical documents. Attach files to your message or type 'upload' for document analysis!"

            # Stream the team-coordinated response
            # processing_msg.content = "✅ **Team Response Ready** - Streaming..."
            # await processing_msg.update()

            msg.content = "✅ **Team Response Ready** - Streaming..."
            await msg.update()

            # Create streaming response
            response_msg = cl.Message(content="")
            await response_msg.send()

            import asyncio
            for i, char in enumerate(response):
                response_msg.content += char
                if i % 25 == 0 or i == len(response) - 1:
                    await response_msg.update()
                    await asyncio.sleep(0.02)

        except Exception as e:
            error_msg = f"❌ **Medical AI Team Error**: {str(e)}\n\nOur team is having trouble processing your request. Please ensure all AI services are running."
            await cl.Message(content=error_msg).send()
            print(f"Error in team orchestration: {e}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
