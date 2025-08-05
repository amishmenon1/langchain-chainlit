"""This module provides tools for patient file retrieval and analysis.

It includes a retriever tool for searching through patient PDF documents
stored in a ChromaDB vector database.

These tools are specifically designed for medical document analysis and
patient data retrieval from processed PDF files.
"""

import os
from typing import Any, List, Optional, cast
from uuid import uuid4

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter
import chromadb

from agents.file_agent.configuration import Configuration
# from configuration import Configuration
from dotenv import load_dotenv

load_dotenv()

# Vector database configuration
CHROMA_PATH = "./chroma"
COLLECTION_NAME = "patient_files"

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def clear_document_store():
    # Delete all documents in the collection
    collection.delete(where={"id": {"$ne": ""}})


def get_vector_store() -> Optional[Chroma]:
    """Get or create the ChromaDB vector store instance.

    Returns:
        Chroma vector store instance or None if not available
    """
    try:
        if not os.path.exists(CHROMA_PATH):
            print(f"ChromaDB path {CHROMA_PATH} does not exist")
            return None

        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        return vector_store
    except Exception as e:
        print(f"Error accessing ChromaDB vector store: {str(e)}")
        return None


def get_retriever(k: int = 5) -> Optional[VectorStoreRetriever]:
    """Create a retriever instance for document search.

    Args:
        k: Number of documents to retrieve (default: 5)

    Returns:
        VectorStoreRetriever instance or None if vector store unavailable
    """
    vector_store = get_vector_store()
    if vector_store is None:
        return None

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def format_retrieval_results(docs) -> str:
    """Format retrieved documents into a readable string.

    Args:
        docs: List of retrieved documents

    Returns:
        Formatted string containing document content
    """
    if not docs:
        return "I found no relevant information in the patient files."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


def clean_metadata(metadata: dict) -> dict:
    """Clean metadata dictionary to contain only primitive types.

    Args:
        metadata: Raw metadata dictionary from document loading

    Returns:
        Cleaned metadata with only primitive types
    """
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}


def load_pdf_documents(files: List[Document], export_type: ExportType = ExportType.MARKDOWN) -> List[Document]:
    """Load and process documents from a PDF file.

    Args:
        file_path: Path to the PDF file
        export_type: Type of export format (MARKDOWN or DOC_CHUNKS)

    Returns:
        List of processed Document objects

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If there's an error loading the PDF
    """
    for file in files:
        if not os.path.exists(file.path):
            raise FileNotFoundError(f"PDF file not found: {file.path}")

    extracted_docs = []
    processed_filenames = []

    for file in files:
        print(f"Loading attached file: {file.name}")
        try:
            loader = DoclingLoader(file_path=file.path,
                                   export_type=export_type
                                   )
            docs = []
            for page in loader.lazy_load():
                # page.metadata['source_file'] = file.name
                page.metadata = clean_metadata(page.metadata)
                # print(f"\n\npage: {page}\n\n")
                docs.append(page)
            # docs = loader.load()

            extracted_docs.extend(docs)
            processed_filenames.append(file.name)

            print(
                f"Successfully processed: {file.name} ({len(docs)} pages)")

        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")

    return extracted_docs, processed_filenames
    # try:
    #     # Initialize DoclingLoader
    #     pdf_loader = DoclingLoader(
    #         file_path=file_path,
    #         export_type=export_type,
    #     )

    #     # Extract documents and clean metadata
    #     docs = []
    #     for page in pdf_loader.lazy_load():
    #         page.metadata['source_file'] = os.path.basename(file_path)
    #         page.metadata = clean_metadata(page.metadata)
    #         docs.append(page)

    #     print(
    #         f"✅ Successfully loaded {len(docs)} pages from {os.path.basename(file_path)}")
    #     return docs

    # except Exception as e:
    #     print(f"❌ Error loading {file_path}: {str(e)}")
    #     raise


def process_documents_for_chunking(docs: List[Document], export_type: ExportType = ExportType.MARKDOWN) -> List[Document]:
    """Process documents based on export type for optimal chunking.

    Args:
        docs: List of raw documents from PDF loading
        export_type: Type of export format used

    Returns:
        List of processed/chunked documents ready for embedding

    Raises:
        ValueError: If export_type is not supported
    """
    print(f"🔧 Processing {len(docs)} documents for chunking...")

    if export_type == ExportType.DOC_CHUNKS:
        # Documents are already properly chunked
        splits = docs
    elif export_type == ExportType.MARKDOWN:
        # Use markdown-specific chunking
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = []
        for doc in docs:
            try:
                doc_splits = splitter.split_text(doc.page_content)
                splits.extend(doc_splits)
            except Exception as e:
                print(
                    f"Warning: Could not split document, adding as-is: {str(e)}")
                splits.append(doc)
    else:
        raise ValueError(f"Unexpected export type: {export_type}")

    print(f"✅ Processed into {len(splits)} chunks")

    # Display sample content for debugging
    for i, d in enumerate(splits[:3]):
        print(f"- Chunk {i+1}: {d.page_content[:100]}...")
    if len(splits) > 3:
        print("...")

    def format_docs(docs: List[Document]):
        return "\n\n".join(doc.page_content for doc in docs)

    formatted_docs = format_docs(docs)

    return splits, formatted_docs


def add_documents_to_vector_store(documents: List[Document], vector_store: Optional[Chroma] = None) -> Chroma:
    """Add documents to the ChromaDB vector store.

    Args:
        documents: List of processed Document objects to add
        vector_store: Existing vector store instance (optional)

    Returns:
        ChromaDB vector store instance with added documents

    Raises:
        Exception: If there's an error adding documents to the store
    """
    if not documents:
        print("⚠️  No documents to add to vector store")
        return vector_store or get_vector_store()

    print(f"💾 Adding {len(documents)} documents to vector store...")

    # Get or create vector store
    if vector_store is None:
        vector_store = get_vector_store()
        if vector_store is None:
            # Create new vector store if none exists
            print("🆕 Creating new ChromaDB vector store...")
            persist_directory = CHROMA_PATH
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)

            vector_store = Chroma.from_documents(
                documents=documents[:1],  # Initialize with first document
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=COLLECTION_NAME
            )
            # Add remaining documents if any
            if len(documents) > 1:
                uuids = [str(uuid4()) for _ in range(len(documents[1:]))]
                vector_store.add_documents(documents=documents[1:], ids=uuids)
        else:
            # Add to existing vector store
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)
    else:
        # Use provided vector store
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

    print(f"✅ Successfully added {len(documents)} documents to vector store")
    return vector_store


def load_and_process_pdf(files: List[Document], export_type: ExportType = ExportType.MARKDOWN) -> List[Any]:
    """Process multiple PDF files and add them to the vector store.

    This function processes multiple PDFs in sequence and combines them
    into a single vector store, useful for batch processing.

    Args:
        files: List of PDF files to process
        export_type: Type of export format to use

    Returns:
        doc_splits: All processed documents
        formatted_docs: Formatted string containing all document content

    Raises:
        FileNotFoundError: If any PDF file doesn't exist
        Exception: If there's an error in any step of the process
    """
    try:
        # Step 1: Load PDF documents
        docs, processed_filenames = load_pdf_documents(files, export_type)

        # Step 2: Process documents for chunking
        doc_splits, formatted_docs = process_documents_for_chunking(
            docs, export_type)

        # Step 3: Add to vector store
        vector_store = add_documents_to_vector_store(doc_splits)

        # print(f"🎉 Successfully processed PDF: {os.path.basename(file_path)}")
        print(f"doc splits: {doc_splits}")
        return doc_splits, formatted_docs, processed_filenames

    except Exception as e:
        print(f"❌ Error in PDF processing pipeline: {str(e)}")
        raise


def load_and_process_multiple_pdfs(file_paths: List[str], export_type: ExportType = ExportType.MARKDOWN) -> Chroma:
    """Process multiple PDF files and add them to the vector store.

    This function processes multiple PDFs in sequence and combines them
    into a single vector store, useful for batch processing.

    Args:
        file_paths: List of paths to PDF files to process
        export_type: Type of export format to use

    Returns:
        ChromaDB vector store instance with all processed documents

    Raises:
        FileNotFoundError: If any PDF file doesn't exist
        Exception: If there's an error in any step of the process
    """
    print(f"📚 Processing {len(file_paths)} PDF files...")

    all_documents = []
    processed_files = []

    for file_path in file_paths:
        try:
            # Load and process each file
            docs = load_pdf_documents(file_path, export_type)
            processed_docs = process_documents_for_chunking(docs, export_type)
            all_documents.extend(processed_docs)
            processed_files.append(os.path.basename(file_path))

        except Exception as e:
            print(f"⚠️  Skipping {file_path} due to error: {str(e)}")
            continue

    if not all_documents:
        raise Exception(
            "No documents were successfully processed from the provided files")

    # Add all documents to vector store
    vector_store = add_documents_to_vector_store(all_documents)

    print(
        f"🎉 Successfully processed {len(processed_files)} files: {', '.join(processed_files)}")
    print(f"📊 Total documents added: {len(all_documents)}")

    return vector_store


@tool
def retriever_tool(query: str) -> str:
    """Search and retrieve information from patient PDF files.

    This tool searches through the processed patient documents stored in the
    vector database and returns relevant information based on the query.
    If the query is empty or asks for all information, it will attempt to
    retrieve the most relevant documents.

    Args:
        query: The search query to find relevant information in patient files

    Returns:
        Formatted string containing relevant document excerpts
    """
    print(f"🔍 Searching patient files with query: {query}")

    # Get configuration for search parameters
    configuration = Configuration.from_context()

    # Create retriever with appropriate number of results
    retriever = get_retriever(k=min(configuration.max_search_results, 10))

    if retriever is None:
        return "Error: Unable to access patient file database. Please ensure files have been uploaded and processed."

    try:
        # Perform the search
        docs = retriever.invoke(query)

        # Format and return results
        return format_retrieval_results(docs)

    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return f"Error occurred while searching patient files: {str(e)}"


TOOLS = [retriever_tool]

# Export the main helper functions for easy access
__all__ = [
    'get_vector_store',
    'get_retriever',
    'format_retrieval_results',
    'clean_metadata',
    'load_pdf_documents',
    'process_documents_for_chunking',
    'add_documents_to_vector_store',
    'load_and_process_pdf',
    'load_and_process_multiple_pdfs',
    'retriever_tool',
    'TOOLS'
]
