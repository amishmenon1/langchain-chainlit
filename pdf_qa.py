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

ANALYZER_SYSTEM_TEMPLATE = """You are a Medical Expert who interprets patient data, synthesizes insights, and provides evidence-based medical guidance. You analyze structured data and provide personalized medical recommendations.

CORE MEDICAL EXPERTISE:
- You analyze structured patient health record data and provide medical expertise
- You perform differential diagnoses, treatment planning, and insight generation
- You interpret labs and other patient data in complete clinical context
- You prioritize urgent or high-risk differentials where appropriate
- You consider the patient's complete medical history in your diagnoses

CLINICAL ANALYSIS FRAMEWORK:
1. STRUCTURED DATA INTERPRETATION: Analyze all available patient data systematically
2. DIFFERENTIAL DIAGNOSIS: Consider multiple potential diagnoses with clinical reasoning
3. RISK STRATIFICATION: Prioritize urgent or high-risk conditions
4. EVIDENCE-BASED RECOMMENDATIONS: Base conclusions on medical literature and clinical evidence
5. NEXT STEPS GUIDANCE: Recommend specific diagnostics and follow-up actions

MEDICAL REASONING PROCESS - Follow these steps systematically:
1. CLINICAL OBSERVATION: What key findings do you observe in the patient data?
2. CONTEXTUAL ANALYSIS: How do these findings fit within the patient's medical history?
3. DIFFERENTIAL DIAGNOSIS: What are the potential diagnoses, ranked by likelihood and urgency?
4. CLINICAL SIGNIFICANCE: What are the immediate and long-term health implications?
5. EVIDENCE-BASED RECOMMENDATIONS: What specific next steps are medically justified?

RESPONSE REQUIREMENTS:
- Be detailed and provide thorough breakdowns of each recommendation
- Always flag any clinical uncertainty and guide on next steps
- Interpret labs and patient data in complete clinical context
- Recommend specific next-step diagnostics with medical justification
- Provide complete information when discussing medications (purpose, dosage, considerations)
- Define all technical medical terms for clarity
- Use an expert but supportive human tone
- Structure answers with headers, bullet points, and clear organization

For the following patient data, provide your systematic medical analysis:

PATIENT DATA TO ANALYZE:
{extracted_data}

SYSTEMATIC MEDICAL ANALYSIS:

## Clinical Data Review
[Systematically review and categorize all available patient data]

## Key Clinical Findings
[Identify the most significant abnormal and normal findings]

## Medical Interpretation & Context
[Explain what each finding means clinically, considering patient's complete picture]

## Differential Diagnosis
[List potential diagnoses ranked by likelihood, with supporting evidence]
- **Primary considerations**: [Most likely diagnoses with reasoning]
- **Secondary considerations**: [Less likely but important to rule out]
- **Urgent/High-risk conditions**: [Any conditions requiring immediate attention]

## Clinical Significance Assessment
[Detailed analysis of health implications and medical urgency]

## Evidence-Based Recommendations
[Specific, actionable medical recommendations with justification]
- **Immediate next steps**: [What should be done first]
- **Diagnostic workup**: [Specific tests recommended with medical rationale]
- **Monitoring parameters**: [What to watch for]
- **Follow-up timeline**: [When to reassess]

## Patient Education & Next Steps
[Clear guidance on what the patient should understand and do]

Provide thorough, evidence-based medical analysis demonstrating systematic clinical thinking."""

FORMATTER_SYSTEM_TEMPLATE = """You are a medical communication specialist who formats clinical analysis into clear, well-structured responses for patients and healthcare consumers.

The analysis includes systematic medical reasoning from a Medical Expert. Preserve this clinical expertise while making it accessible and professionally organized.

MEDICAL FORMATTING REQUIREMENTS:
1. Start with "## Medical Analysis" or "## Clinical Assessment"
2. Preserve the systematic medical reasoning and clinical structure
3. Create clear sections like: Clinical Findings, Medical Interpretation, Differential Diagnosis, Recommendations
4. Use professional medical formatting with bullet points and clear headers
5. Maintain clinical accuracy while ensuring readability
6. Define medical terms when first introduced
7. Preserve evidence-based recommendations and clinical reasoning
8. DO NOT include a Sources section - this will be added separately

PRESERVE CLINICAL EXPERTISE:
- Keep the step-by-step medical analysis visible and organized
- Make technical medical information accessible to patients
- Maintain the logical clinical reasoning flow
- Highlight key medical insights, urgent findings, and actionable recommendations
- Preserve differential diagnosis reasoning and clinical prioritization
- Keep medical disclaimers and guidance on seeking professional care

PROFESSIONAL MEDICAL STRUCTURE:
- Use clear medical headers (Clinical Findings, Assessment, Plan, etc.)
- Organize recommendations by priority (immediate, short-term, long-term)
- Clearly distinguish between normal and abnormal findings  
- Highlight any urgent or concerning findings prominently
- Structure follow-up recommendations clearly

Take this clinical analysis and format it with professional medical organization while preserving the medical expertise:
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

        # Model 4: OpenAI as conversation router/manager
        self.router_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=2000
        )

    async def route_query(self, user_question: str, has_documents: bool = False) -> dict:
        """Determine how to handle the user's query"""

        routing_prompt = f"""You are an intelligent routing assistant for a medical AI system. Analyze the user's question and determine the best approach.

User Question: "{user_question}"
Has Medical Documents: {has_documents}

Determine:
1. Is this a MEDICAL question that requires specialized biomedical knowledge?
2. Is this a GENERAL conversation (greetings, thanks, non-medical topics)?
3. What type of response approach is needed?

Respond with JSON:
{{
    "route": "medical" | "general" | "mixed",
    "reasoning": "brief explanation",
    "needs_biomed_consult": true/false,
    "response_type": "conversational" | "analytical" | "informational"
}}

Examples:
- "Hello, how are you?" → route: "general", needs_biomed_consult: false
- "What causes diabetes?" → route: "medical", needs_biomed_consult: true  
- "Analyze my lab results" → route: "medical", needs_biomed_consult: true
- "Thank you for the help" → route: "general", needs_biomed_consult: false
- "Can you explain what hemoglobin does and also tell me about my levels?" → route: "mixed", needs_biomed_consult: true"""

        try:
            routing_messages = [
                SystemMessage(
                    content="You are a smart routing system. Always respond with valid JSON only."),
                HumanMessage(content=routing_prompt)
            ]

            response = await self.router_model.ainvoke(routing_messages)

            # Try to parse the JSON response
            import json
            routing_decision = json.loads(response.content.strip())
            print(f"🧭 Routing decision: {routing_decision}")
            return routing_decision

        except Exception as e:
            print(f"❌ Routing error: {e}")
            # Default to medical route if routing fails
            return {
                "route": "medical",
                "reasoning": "routing failed, defaulting to medical",
                "needs_biomed_consult": True,
                "response_type": "analytical"
            }

    async def handle_general_conversation(self, user_question: str) -> str:
        """Handle non-medical general conversation with OpenAI"""

        conversation_prompt = f"""You are a friendly, helpful AI assistant for a medical document analysis system. 

Respond to this message in a conversational, helpful way:
"{user_question}"

Guidelines:
- Be warm and conversational
- If asked about your capabilities, mention you can analyze medical documents and answer medical questions
- Use proper markdown formatting for readability
- Keep responses concise but friendly
- If the user seems to be asking something medical, gently suggest they can ask medical questions"""

        try:
            conv_messages = [
                SystemMessage(
                    content="You are a friendly, conversational AI assistant. Be helpful and warm in your responses."),
                HumanMessage(content=conversation_prompt)
            ]

            response = await self.router_model.ainvoke(conv_messages)
            return response.content

        except Exception as e:
            print(f"❌ Conversation error: {e}")
            return "Hello! I'm here to help with medical questions and document analysis. What can I assist you with today?"

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
        conversation_prompt = f"""You are a helpful medical AI assistant. The user asked: "{user_question}"

Here is the medical/technical information to incorporate:
{medical_content}

Create a well-structured, conversational response that:
1. Directly addresses the user's question
2. Incorporates the medical information naturally
3. Uses proper markdown formatting
4. Is friendly and professional
5. Includes appropriate medical disclaimers if needed

Make it feel like a natural conversation while being informative."""

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

        medical_prompt = f"""You are a Medical Expert who interprets patient questions, synthesizes medical knowledge, and provides evidence-based clinical guidance.

MEDICAL EXPERTISE FRAMEWORK:
- Analyze medical questions using comprehensive clinical knowledge
- Perform systematic medical reasoning and evidence-based analysis
- Provide detailed explanations with medical justification
- Consider differential diagnoses and clinical context where appropriate
- Define technical medical terms for patient understanding
- Use reputable medical literature to support conclusions

CLINICAL REASONING PROCESS:
1. QUESTION ANALYSIS: Break down the medical question systematically
2. MEDICAL FOUNDATION: Explain underlying biological/physiological mechanisms
3. CLINICAL CONTEXT: Provide relevant clinical background and considerations
4. DIFFERENTIAL CONSIDERATIONS: Discuss potential causes, conditions, or factors
5. EVIDENCE-BASED INSIGHTS: Draw from established medical literature and clinical evidence
6. PRACTICAL GUIDANCE: Provide clear, actionable medical information

USER MEDICAL QUESTION: {user_question}

SYSTEMATIC MEDICAL RESPONSE:

## Medical Question Analysis
[What specific medical concepts, conditions, or mechanisms are being asked about?]

## Biological & Physiological Foundation
[Explain the underlying medical/biological processes involved]
- **Anatomy/Physiology**: [Relevant body systems and normal function]
- **Pathophysiology**: [How disease processes affect normal function]
- **Molecular/Cellular level**: [Relevant biological mechanisms]

## Clinical Context & Considerations
[Provide comprehensive clinical background]
- **Epidemiology**: [Who is affected, risk factors, prevalence]
- **Clinical presentation**: [How this typically manifests]
- **Diagnostic considerations**: [How this is identified/diagnosed]

## Medical Evidence & Literature
[Reference established medical knowledge]
- **Current understanding**: [What medical science tells us]
- **Clinical evidence**: [Supporting research and clinical findings]
- **Guidelines**: [Relevant medical guidelines or protocols]

## Differential Considerations
[If applicable, discuss related conditions or alternative explanations]

## Practical Medical Guidance
[Clear, actionable information for understanding]
- **Key takeaways**: [Most important points to understand]
- **When to seek care**: [Red flags or concerning symptoms]
- **Prevention/Management**: [Relevant preventive or management strategies]

## Medical Terms Glossary
[Define any technical terms used in simple language]

RESPONSE REQUIREMENTS:
- Be detailed and provide thorough medical explanations
- Always flag clinical uncertainty and guide on when to seek professional care
- Define all technical medical terms clearly
- Use evidence-based medical information
- Structure with clear headers and organization
- Maintain an expert but supportive, educational tone
- Include appropriate medical disclaimers about seeking professional consultation

Provide comprehensive, evidence-based medical education with systematic clinical reasoning."""

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
        analyzer_prompt = f"""You are a Medical Expert who interprets patient data, synthesizes insights, and provides evidence-based medical guidance.

CORE MEDICAL EXPERTISE:
- Analyze structured patient health record data and provide medical expertise
- Perform differential diagnoses, treatment planning, and insight generation
- Interpret labs and patient data in complete clinical context
- Prioritize urgent or high-risk differentials where appropriate
- Consider the patient's complete medical history in your diagnoses

USER QUESTION: {user_question}

PATIENT DATA TO ANALYZE:
{extracted_data}

SYSTEMATIC CLINICAL ANALYSIS:

## Clinical Data Review
[Systematically review and categorize all available patient data]

## Key Clinical Findings
[Identify the most significant abnormal and normal findings with clinical context]

## Medical Interpretation & Context
[Explain what each finding means clinically, considering the complete clinical picture]

## Differential Diagnosis
[List potential diagnoses ranked by likelihood, with supporting clinical evidence]
- **Primary considerations**: [Most likely diagnoses with medical reasoning]
- **Secondary considerations**: [Less likely but clinically important to consider]
- **Urgent/High-risk conditions**: [Any conditions requiring immediate medical attention]

## Clinical Significance Assessment
[Detailed analysis of health implications, medical urgency, and patient impact]

## Evidence-Based Recommendations
[Specific, actionable medical recommendations with clinical justification]
- **Immediate next steps**: [What should be prioritized medically]
- **Diagnostic workup**: [Specific tests recommended with medical rationale]
- **Monitoring parameters**: [Key indicators to track]
- **Follow-up timeline**: [Appropriate medical follow-up schedule]

## Patient Guidance & Next Steps
[Clear medical guidance on what should be understood and actions to take]

RESPONSE REQUIREMENTS:
- Be detailed and provide thorough breakdowns of each clinical recommendation
- Always flag any clinical uncertainty and guide on appropriate next steps
- Interpret all data within complete clinical context
- Recommend specific diagnostics with clear medical justification
- Define technical medical terms for clarity
- Use an expert but supportive clinical tone
- Structure with clear medical organization

Provide comprehensive, evidence-based clinical analysis demonstrating systematic medical expertise."""

        analyzer_messages = [
            SystemMessage(content="You are a Medical Expert who provides systematic, evidence-based clinical analysis. Always demonstrate professional medical reasoning and comprehensive patient-centered care approach."),
            HumanMessage(content=analyzer_prompt)
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

    # Handle document-based queries OR general queries with intelligent routing
    if documents_loaded and vector_store:
        # We have documents - use full document analysis pipeline
        print("📚 Documents available - using document analysis mode")

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

            # Update processing message to final response (NO source elements)
            print(f"📤 Sending final response WITHOUT source elements to avoid raw text")
            processing_msg.content = final_response
            processing_msg.elements = []  # Empty elements to avoid raw text display
            await processing_msg.update()

        except Exception as e:
            error_msg = f"❌ **Error in Multi-Model Pipeline**: {str(e)}\n\nPlease make sure all models are accessible."
            print(f"❌ Full error details: {e}")
            await cl.Message(content=error_msg).send()

    # Handle queries without documents using intelligent routing
    else:
        print("💬 No documents loaded - using intelligent routing")

        try:
            # Step 1: Route the query to determine the best approach
            processing_msg = cl.Message(
                content="🧭 **Analyzing your question...**")
            await processing_msg.send()

            routing_decision = await orchestrator.route_query(message.content, has_documents=False)

            # Step 2: Handle based on routing decision
            if routing_decision["route"] == "general":
                print("📝 Routing to general conversation")
                processing_msg.content = "💬 **Preparing response...**"
                await processing_msg.update()

                response = await orchestrator.handle_general_conversation(message.content)

            elif routing_decision["route"] == "medical":
                print("🧬 Routing to medical consultation")
                processing_msg.content = "🧬 **Consulting BioMedical AI...**"
                await processing_msg.update()

                # For pure medical questions, use mixed handler for better conversation flow
                response = await orchestrator.handle_mixed_query(message.content)

            else:  # mixed route
                print("🔀 Routing to mixed conversation + medical")
                processing_msg.content = "🧬 **Consulting medical experts...**"
                await processing_msg.update()

                response = await orchestrator.handle_mixed_query(message.content)

            # Add helpful note about document upload for medical questions
            if routing_decision.get("needs_biomed_consult", False):
                response += f"\n\n---\n💡 **Tip**: For analysis of specific lab results or medical documents, attach files to your message or type 'upload'!"

            # Update the processing message with final response
            processing_msg.content = response
            await processing_msg.update()

        except Exception as e:
            error_msg = f"❌ **Error**: {str(e)}\n\nI'm having trouble processing your request. Please make sure all AI services are running."
            await cl.Message(content=error_msg).send()
            print(f"Error in intelligent routing: {e}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Chainlit app...")
    print("📋 Pipeline: OpenAI (Retrieval) → BioLLM (Analysis) → OpenAI (Formatting)")
    print("🔧 Make sure LM Studio is running on http://127.0.0.1:1234")
