from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import chainlit as cl
import json
from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE
from templates.system.formatter import FORMATTER_SYSTEM_TEMPLATE
from templates.human.prompts import generate_routing_prompt, generate_general_conversation_prompt, generate_mixed_conversation_prompt, generate_medical_prompt, generate_analyzer_prompt, generate_eval_prompt, generate_analysis_prompt, generate_alt_enhancement_prompt
from utils.message import update_message, new_message


class MultiModelOrchestrator:
    def __init__(self):
        # Model 1: OpenAI for formatting and communication (good at following instructions)
        self.formatter_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4000,
            streaming=True
        )

        # Model 2: OpenAI for document retrieval (consistent and reliable)
        self.retriever_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=3000
        )

        # # Model 3: LM Studio BioLLM for medical analysis (domain expertise)
        # self.analyzer_model = ChatOpenAI(
        #     base_url="http://127.0.0.1:1234/v1",
        #     api_key="lm-studio",
        #     model="openbiollm-llama3-8b",
        #     temperature=0.1,  # Lower temperature for more focused medical analysis
        #     max_tokens=4000   # Increased for comprehensive analysis
        # )

        # # Model 3: OpenBioLLM-70B
        # self.analyzer_model = ChatOpenAI(
        #     # Replace with your actual RunPod endpoint
        #     base_url="https://smu0tfnzayidsv-8000.proxy.runpod.net/v1",
        #     api_key="runpod-70b",  # Fake key; required by LangChain for compatibility
        #     model="aaditya/Llama3-OpenBioLLM-70B",
        #     temperature=0.1,
        #     max_tokens=4096,  # Adjust according to RunPod’s limit
        #     # streaming=True
        # )

        # # Model 3: OpenAI as medical analyzer (temporary)
        self.analyzer_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4000,
            streaming=True
        )

        # Model 4: OpenAI as conversation router/manager
        self.router_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=4000,
            streaming=True
        )

    async def route_query(self, user_question: str, has_documents: bool = False) -> dict:
        """Determine how to handle the user's query with orchestration awareness"""

        routing_prompt = generate_routing_prompt(user_question, has_documents)
        print(f"routing prompt: {routing_prompt}")
        try:
            routing_messages = [
                SystemMessage(
                    content="You are a Medical AI Team Orchestrator. Always respond with valid JSON and coordinate team resources efficiently."),
                HumanMessage(content=routing_prompt)
            ]

            response = await self.router_model.ainvoke(routing_messages)
            print(
                f"invoked router model - routing messages: {routing_messages}")
            # Try to parse the JSON response
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
        print(f"🧭 Orchestrator - handling general conversation")
        conversation_prompt = generate_general_conversation_prompt(
            user_question)
        try:
            conv_messages = [
                SystemMessage(
                    content="You are the friendly communication interface for a Medical AI Team. Be helpful, warm, and represent the team's collaborative capabilities."),
                HumanMessage(content=conversation_prompt)
            ]

            # response = await self.router_model.ainvoke(conv_messages)
            # return response.content
            chainlit_message = await new_message(content="")

            final_answer = ""

            async for chunk in self.router_model.astream(conv_messages):
                token = chunk.content or ""
                if final_answer and token:
                    final_answer += token
                await chainlit_message.stream_token(token)

            await chainlit_message.update()

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

            # response = await self.router_model.ainvoke(conv_messages)
            # return response.content
            # Stream the team-coordinated response

            chainlit_message = await new_message(content="")

            final_answer = ""

            async for chunk in self.formatter_model.astream(conv_messages):
                token = chunk.content or ""
                if final_answer and token:
                    final_answer += token
                await chainlit_message.stream_token(token)

            await chainlit_message.update()

        except Exception as e:
            print(f"❌ Mixed query error: {e}")
            return medical_content  # Fallback to just medical content

    # @cl.step(name="Medical Insights", show_input=False)
    async def get_medical_insights(self, user_question: str) -> str:
        """Get medical insights from OpenBioLLM for general medical questions with clinical expertise"""

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

    async def format_and_stream(self, user_question: str, analysis: str, sources: list = []):
        formatter_prompt = FORMATTER_SYSTEM_TEMPLATE.format(
            raw_analysis=analysis,
            user_question=user_question
        )

        formatter_messages = [
            SystemMessage(
                content="You are a medical formatter. Structure this analysis with rich formatting, tables, sections, and patient-facing clarity."),
            HumanMessage(
                content=formatter_prompt)
        ]

        try:
            chainlit_message = await new_message(content="")

            final_answer = ""

            async for chunk in self.formatter_model.astream(formatter_messages):
                token = chunk.content or ""
                if final_answer and token:
                    final_answer += token
                await chainlit_message.stream_token(token)

            await chainlit_message.update()

            # Add properly formatted sources to the response (unique files only)

            if sources:
                sources_section = f"\n\n## Sources\n" + \
                    "\n".join([f"- {source}" for source in sources])
                for source in sources_section:
                    await chainlit_message.stream_token(source)
            await chainlit_message.update()

            print(
                f"✅ Final formatted output ready (length={len(final_answer)})")

        except Exception as e:
            print(f"❌ Formatter error: {e}")

        return final_answer

    async def process_query(self, user_question: str, retrieved_docs: str, unique_sources: list) -> str:
        """Improved document analysis pipeline"""

        # Step 1: Use retriever model to extract structured lab data
        # print("🔍 Step 1: Extracting structured data from documents...")
        # sys_msg_5 = await new_message(content="🔍 Checking database for relevant documents...")

        # retriever_prompt = RETRIEVER_SYSTEM_TEMPLATE.format(
        #     summaries=retrieved_docs)

        # retriever_messages = [
        #     SystemMessage(content=retriever_prompt),
        #     HumanMessage(
        #         content=f"Extract information relevant to the user's question: {user_question}")
        # ]

        # try:
        #     extracted_response = await self.retriever_model.ainvoke(retriever_messages)
        #     raw_extracted = extracted_response.content.strip()
        #     print(f"✅ Extracted lab data (length={len(raw_extracted)})")
        #     extracted_data = f"Patient Lab Data Extracted:\n{raw_extracted}"

        #     # Save for future access or follow-ups
        #     cl.user_session.set("structured_lab_data", raw_extracted)
        #     await update_message(msg=sys_msg_5, content="✅ I've extracted all the patient data.")

        # except Exception as e:
        #     print(f"❌ Retriever model error: {e}")
        #     extracted_data = f"Raw fallback document text:\n\n{retrieved_docs}"

        # Step 2: Use alt_analysis prompt to force detailed full-document analysis
        print("🧬 Step 2: Running BioLLM medical analysis...")
        sys_msg_6 = await new_message(content="🧬 Performing medical analysis...")

        analyzer_prompt = generate_analysis_prompt(
            user_question, retrieved_docs)

        analyzer_messages = [
            SystemMessage(content="You are a Medical Analyst. DO NOT skip any lab parameters. Analyze all lab values across ALL provided files. Include personalized insight and organized tables."),
            HumanMessage(content=analyzer_prompt)
        ]

        # Add this to double-check actual prompt length (token-wise if needed)
        print(
            f"🧠 Final analyzer prompt length: {len(analyzer_prompt)} characters")
        print(analyzer_prompt[:2000])  # Preview safely

        try:
            analysis_response = await self.analyzer_model.ainvoke(analyzer_messages)
            raw_analysis = analysis_response.content.strip()

            await update_message(msg=sys_msg_6, content="✅ Initial medical analysis complete. Evaluating response...")
            print(f"✅ Initial BioLLM analysis (length={len(raw_analysis)})")

            # Quality filter - check for insufficient or hallucinated output
            if (
                len(raw_analysis) < 800
                or "i cannot" in raw_analysis.lower()
                or "not enough information" in raw_analysis.lower()
                or "no lab results" in raw_analysis.lower()
                or "Hemoglobin" not in raw_analysis  # heuristic
            ):
                print(
                    "⚠️ Weak or incomplete analysis detected — falling back to OpenAI alt analysis")
                raw_analysis = await self.create_alternative_analysis(retrieved_docs, user_question)
            else:
                # Attempt enhancement ONLY if initial is decent
                print("🔍 Enhancing analysis...")

                await update_message(msg=sys_msg_6, content="🔍 Evaluating response...")

                enhanced_analysis = await self.self_evaluate_and_enhance(
                    raw_analysis, retrieved_docs, user_question
                )

                # Use enhanced only if meaningfully longer
                if len(enhanced_analysis) > len(raw_analysis) * 1.2:
                    raw_analysis = enhanced_analysis
                    print("✅ Enhancement accepted")
                else:
                    print("⚠️ Enhancement not significantly better — skipping")

                await update_message(msg=sys_msg_6, content="✅ Evaluation complete!")

        except Exception as e:
            print(f"❌ BioLLM error: {e}")
            raw_analysis = await self.create_alternative_analysis(retrieved_docs, user_question)

        # Step 3: Format final response
        print("✨ Step 3: Formatting with OpenAI formatter...")

        await update_message(msg=sys_msg_6, content="✅ Analysis complete!")
        return await self.format_and_stream(user_question=user_question, analysis=raw_analysis, sources=unique_sources)

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

        alternative_prompt = generate_analysis_prompt(
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
