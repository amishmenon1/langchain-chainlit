# tools/retriever_tool.py
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from utils.file import retrieve_chunks
import chainlit as cl


@tool
async def retrieve_patient_documents(query: str) -> str:
    """Retrieve relevant patient document chunks based on a medical query."""
    retrieved_text, _ = await retrieve_chunks(query)
    return retrieved_text
