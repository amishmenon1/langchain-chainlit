from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain.schema.runnable import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import StrOutputParser, Document
from langchain.schema.runnable.config import RunnableConfig
from typing import cast, Annotated, List, Optional, Any

import chainlit as cl

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from message import update_message, new_message
from langchain import hub as prompts
from prompt_templates import MSG_CLASSIFIER_PROMPT_TEMPLATE, RETRIEVER_SYSTEM_PROMPT, ANALYZER_SYSTEM_PROMPT, CONVERSATIONAL_FORMATTER_PROMPT, RAG_QA_PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import operator
from typing_extensions import TypedDict
# from templates.system.retriever import RETRIEVER_SYSTEM_PROMPT, RETRIEVER_STRUCTURED_OUTPUT
# from templates.system.analyzer import ANALYZER_SYSTEM_PROMPT
# from templates.system.formatter import FORMATTER_PROMPT_TEMPLATE, CONVERSATIONAL_FORMATTER_PROMPT
from utils.file import load_files_into_db,  retrieve_chunks
from pydantic import BaseModel, Field
import pprint
import os


#### RETRIEVER STRUCTURED OUTPUTS ####


class LabResult(BaseModel):
    test_name: str
    value: str
    reference_range: Optional[str]
    # interpretation: Optional[str]


class ImagingResult(BaseModel):
    modality: str
    date: Optional[str]
    findings: str


class MedicalDocumentSummary(BaseModel):
    document_title: str
    date: Optional[str]
    lab_results: List[LabResult]
    imaging_results: List[ImagingResult]
    other_notes: Optional[List[str]]

#### ANALYSIS STRUCTURED OUTPUTS ####


class MedicalAnalysis(BaseModel):
    key_findings: str = Field(description="Concise summary of what's found.")
    clinical_interpretation: str = Field(
        description="Medical meaning of the findings.")
    medical_breakdown: str = Field(
        description="Detailed reasoning, logic, and science.")
    recommendations: str = Field(description="Next steps, monitoring advice.")

#### CLASSIFICATION STRUCTURE ####
# class GradeQuestion(BaseModel):
#     score: str = Field(description="Is the question asking a clinical-related question, file-related question, or general question")


class GradeDocument(BaseModel):
    score: str = Field(
        description="Is the document relevant to the question? If yes -> 'Yes'. If not -> 'No'.")


#### GRAPH STATE ####


class UserMessageState(MessagesState):
    document_context: Any
    retrieved_docs: Annotated[List[Document], operator.add]
    analysis: MedicalAnalysis
    previous_user_message: str
    rephrased_question: str
    rephrase_count: int
    proceed: bool


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
CHROMA_PATH = 'chroma'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "6.25-Chainlit-Langgraph-Assistant"

chat_history = []

default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
msg_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_classifier_llm = ChatOpenAI(model="gpt-4o-mini")
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
research_llm = ChatOpenAI(model="gpt-4o-mini")
analysis_llm = ChatOpenAI(model="gpt-4o-mini")
formatter_llm = ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def on_chat_start():

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    def question_rewriter(state: UserMessageState):
        print(f"\n----question_rewriter()----\n")
        # state["retrieved_docs"] = []
        state["rephrased_question"] = ""
        state["proceed"] = False

        if len(state["messages"]) > 1:
            conversation = state["messages"][:-1]
            current_question = state["messages"][-1]
            messages = [
                SystemMessage(
                    content="You are a helpful assistent that rephrases the user's question to be a standalone question optimized for retrieval.")
            ]
            messages.extend(conversation)
            messages.append(HumanMessage(content=current_question.content))
            rephrase_prompt = ChatPromptTemplate.from_messages(messages)
            prompt = rephrase_prompt.format()
            response = default_llm.invoke(prompt)
            better_question = response.content.strip()
            print(f"\n\nRephrased question:\n{better_question}\n\n")
            return {
                "rephrased_question": better_question
            }

    # node 1 - classifier node
    def classify_message(state: UserMessageState):
        print(f"\n-----classify_message()-----\n")
        print(
            f"\nRephrased question: {state.get("rephrased_question", None)}\n")
        message = state["messages"][-1]

        prompt = ChatPromptTemplate.from_template(
            MSG_CLASSIFIER_PROMPT_TEMPLATE)

        chain = prompt | msg_classifier_llm
        result = chain.invoke({"message": message.content}
                              ).content.strip().lower()

        print(f"\n\nclassify output:\n {result}\n\n")
        return result

    # node 1a - RAG node

    async def retrieve_docs(state: UserMessageState):
        print(f"\n-----retrieve_docs()-----\n")
        message = state["messages"][-1]
        rephrased_message = state.get("rephrased_question", message.content)

        print(f"rephrased message: {rephrased_message}")

        docs = []
        vector_store = cast(Chroma, cl.user_session.get("vector_store", None))
        if vector_store:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 5, })  # Increased to get more coverage
            similarity_docs = retriever.invoke(rephrased_message)
            docs.extend(similarity_docs)

            print(f"Documents retrieved: {len(similarity_docs)}")

        return {
            "retrieved_docs": docs
        }

    def retrieval_grader(state: UserMessageState):
        print(f"\n----retrieval_grader()----\n")
        print(
            f"\n\nRephrased question: {state.get("rephrased_question", None)}\n\n")
        message = state["messages"][-1]
        system_msg = SystemMessage(content="""
            You are a grader assessing the relevance of a retrieved document to a user question. Only answer with 'Yes' or 'No'. If the document contains information relevant to the user's query, respond with 'Yes'. Otherwise, respond with 'No'.
        """)

        structured_llm = rag_llm.with_structured_output(GradeDocument)
        relevant_docs = []
        for doc in state.get("retrieved_docs", []):
            human_msg = HumanMessage(
                content=f"User question:{state.get("rephrased_question", message.content)}\n\nRetrieved document:\n{doc.page_content}")
            grade_prompt = ChatPromptTemplate.from_messages(
                [system_msg, human_msg])
            grader_llm = grade_prompt | structured_llm
            result = grader_llm.invoke({})
            print(
                f"\n\nGrading document: {doc.page_content[:30]}... \n\nResult: {result.score.strip()}")

            if result.score.strip().lower() == "yes":
                relevant_docs.append(doc)
            print(
                f"proceed: {len(relevant_docs) > 0}")
        # return {
        #     "retrieved_docs": relevant_docs,
        #     "proceed": len(relevant_docs) > 0
        # }
        return {
            "retrieved_docs": relevant_docs,
            "proceed": True
        }

    def proceed_router(state: UserMessageState):
        print(f"\n----proceed_router()----\n")
        rephrase_count = state.get("rephrase_count", 0)
        if state.get("proceed", False):
            print(f"\n\nRouting to generate_answer\n\n")
            return "proceed"
        elif rephrase_count >= 2:
            print(f"Max rephrase attempts reached. Cannot find relevant documents.")
            return "generate"
        else:
            print("Routing to refine_question")
            return "refine_question"

    def refine_question(state: UserMessageState):
        print(f"\n----refine_question()----\n")
        rephrase_count = state.get("rephrase_count", 0)
        if rephrase_count >= 2:
            print(f"Maximum rephrase attempts reached")
            return state
        question_to_refine = state.get("rephrased_question", None)
        system_msg = SystemMessage(
            content="You are a helpful assistant that slightly refines the user's question to improve retrieval results. Provide a slightly adjusted version of the question.")
        human_msg = HumanMessage(
            content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question.")
        refine_prompt = ChatPromptTemplate.from_messages(
            [system_msg, human_msg])
        prompt = refine_prompt.format()
        response = rag_llm.invoke(prompt)
        refined_question = response.content.strip()
        print(f"\n\nRefined question: {refined_question}\n\n")
        return {
            "rephrased_question": refined_question,
            "rephrase_count": rephrase_count + 1
        }

    def perform_rag_qa(state: UserMessageState):
        print(f"\n----perform_rag_qa()----\n")
        history = state["messages"][:-1]
        message = state["messages"][-1]
        documents = state.get("retrieved_docs", [])
        rephrased_question = state.get("rephrased_question", message.content)
        messages = state["messages"]
        print(f"\n\nmessage: {message}")

        # prompt = ChatPromptTemplate.from_template(RAG_QA_PROMPT_TEMPLATE)
        prompt = ChatPromptTemplate.from_template(RETRIEVER_SYSTEM_PROMPT)
        chain = prompt | rag_llm.with_structured_output(MedicalDocumentSummary)

        response = chain.invoke(
            {"message": rephrased_question, "context": documents, "history": messages})
        print(f"\n\nresponse: {response}")
        # generation = response.content.strip()
        # messages.append(AIMessage(content=response))

        return {
            "document_context": response
        }

    # node 2a - researcher node

    def web_search(state: UserMessageState):
        print(f"\n----web_search()----\n")
        # TODO implement method using pub med and web search tools
        return state

    # node 3a - analyzer node

    def analyze(state: UserMessageState):
        print(f"\n-----analyze()-----\n")
        message = state["messages"][-1]
        messages = state["messages"]
        document_context = state.get("document_context", None)
        print(f"user message: {message.content}")
        prompt = ChatPromptTemplate.from_template(ANALYZER_SYSTEM_PROMPT)

        chain = prompt | analysis_llm.with_structured_output(MedicalAnalysis)
        response = chain.invoke(
            {"message": message.content, "context": document_context, "history": messages})

        pprint.pprint(f"\n\nanalyze() output:\n{response.model_dump()}\n\n")

        print("→ Using conversational formatter")
        prompt = ChatPromptTemplate.from_template(
            CONVERSATIONAL_FORMATTER_PROMPT)
        inputs = {
            "previous_user_message": message.content,
            "chat_history": messages,
            "message": message.content,
            # can be None or partial; template will handle if not present.
            "analysis": response
        }
        return {"analysis": response}

    # node 1b - default generator node

    def generate(state: UserMessageState):
        print("\n-----generate()-----\n")
        message = state["messages"][-1]
        analysis = state.get("analysis", None)

        # Retrieve the full chat history as a text block.
        chat_history_text = "\n".join(
            f"{m.__class__.__name__}: {m.content}" for m in state["messages"])

        # Extract the last user message (prior to current) if needed.
        previous_user_message = next(
            (m.content for m in reversed(
                state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        # Decide which prompt to use based on analysis content.
        # For example: if the analysis exists and the current message is a clear clinical inquiry, use structured.
        # Otherwise, use the conversational format.
        # You can refine this logic depending on your use case.
        use_conversational = True
        # Alternatively, if analysis is missing or empty, treat as conversational.
        # if analysis is None or not any(getattr(analysis, field, "").strip() for field in ["key_findings", "clinical_interpretation", "medical_breakdown", "recommendations"]):
        #     use_conversational = True

        if use_conversational:
            print("→ Using conversational formatter")
            prompt = ChatPromptTemplate.from_template(
                CONVERSATIONAL_FORMATTER_PROMPT)
            inputs = {
                "previous_user_message": previous_user_message,
                "chat_history": chat_history_text,
                "message": message.content,
                # can be None or partial; template will handle if not present.
                "analysis": analysis
            }
        # else:
        #     print("→ Using clinical formatter")
        #     prompt = ChatPromptTemplate.from_messages([
        #         ("system", FORMATTER_PROMPT_TEMPLATE),
        #         ("human", "{message}")
        #     ])
        #     inputs = {
        #         "analysis": analysis,
        #         "message": message.content,
        #         # if your clinical prompt uses it.
        #         "chat_history": chat_history_text,
        #         "previous_user_message": previous_user_message
        #     }

        llm = prompt | formatter_llm | StrOutputParser()
        response = llm.invoke(inputs)
        return {"messages": [AIMessage(content=response)]}

    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("proceed_router", proceed_router)
    workflow.add_node("perform_rag_qa", perform_rag_qa)
    workflow.add_node("web_search", web_search)
    workflow.add_node("analyze", analyze)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "question_rewriter")
    workflow.add_conditional_edges("question_rewriter", classify_message, {
        # If medical question, use RAG and medical analysis
        "file": "retrieve_docs",
        # Otherwise use general model completion
        "medical": "analyze",
        "general": "generate"
    })
    workflow.add_edge("retrieve_docs", "retrieval_grader")
    workflow.add_conditional_edges("retrieval_grader", proceed_router, {
        "proceed": "web_search",
        "refine_question": "refine_question",
        "generate": "generate"
    })
    workflow.add_edge("refine_question", "retrieve_docs")
    workflow.add_edge("web_search", "perform_rag_qa")
    workflow.add_edge("perform_rag_qa", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    cl.user_session.set("app", app)


@cl.on_message
async def on_message(message: cl.Message):
    print(f"\n\nChat context:\n{cl.chat_context.to_openai()}\n\n")
    # Check if files are attached to this message
    attached_files = None
    if hasattr(message, 'elements') and message.elements:
        # Look for file elements in the message
        attached_files = [elem for elem in message.elements if hasattr(
            elem, 'path') and elem.path.endswith('.pdf')]
        print(f"📎 Found {len(attached_files)} attached files")

    cl.user_session.set("latest_attached_files", attached_files)
    # If files are attached, process them first
    if attached_files:
        chat_history.append(SystemMessage(
            content=f"Attachments: {len(attached_files)}"))
        sys_msg_1 = await new_message(
            content="📎 **Files detected!**")
        await load_files_into_db(attached_files)
    else:
        chat_history.append(SystemMessage(
            content=f"Attachments: {0}"))

    chat_history.append(HumanMessage(content=message.content))

    app = cast(Runnable, cl.user_session.get("app"))

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    answer = cl.Message(content="")

    async for msg, metadata in app.astream(
        # {"messages": chat_history},
        {"messages": [HumanMessage(content=message.content)]},

        config,
        stream_mode="messages",
    ):
        if (
            msg.content
            and isinstance(msg, AIMessageChunk)
            and metadata["langgraph_node"] == "generate"
        ):

            await answer.stream_token(msg.content)

    await answer.update()
