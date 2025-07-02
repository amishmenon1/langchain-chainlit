import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from utils.message import new_message, update_message
from typing import cast, List
from langchain.prompts import ChatPromptTemplate
from uuid import uuid4
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
import os
import chromadb
# from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_PATH = 'chroma'
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("patient_files")

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# DOCLING
EXPORT_TYPE = ExportType.DOC_CHUNKS


def clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        else:
            print(f"Removed non-primitive metadata key: {k} → {type(v)}")
    return cleaned


async def create_vectordb() -> Chroma:

    # Create vector store
    vector_store = Chroma(
        collection_name='patient_files', embedding_function=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    vector_store_from_client = Chroma(
        client=client,
        collection_name="patient_files",
        embedding_function=OpenAIEmbeddings(),
    )
    cl.user_session.set("vector_store", vector_store)

    # return vector_store
    return vector_store_from_client


@cl.step(name="🔍 Scanning patient files")
async def extract_docs_from_files(files) -> List[Document]:

    # Process all attached files
    extracted_docs = []
    processed_filenames = []

    for file in files:
        print(f"Loading attached file: {file.name}")
        try:
            loader = DoclingLoader(file_path=file.path,
                                   export_type=EXPORT_TYPE)
            docs = []
            async for page in loader.alazy_load():
                page.metadata['source_file'] = file.name
                page.metadata = clean_metadata(page.metadata)
                print(f"\n\npage: {page}\n\n")
                docs.append(page)

            extracted_docs.extend(docs)
            processed_filenames.append(file.name)
            print(
                f"Successfully processed: {file.name} ({len(docs)} pages)")

        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            # await new_message(content=f"❌ Error processing {file.name}: {str(e)}")

    # TODO rename to 'chunked_attachments'?
    cl.user_session.set("latest_attached_docs", extracted_docs)

    # session_docs = cl.user_session.get("session_docs", [])
    # cl.user_session.set("session_docs", session_docs + extracted_docs)
    return extracted_docs, processed_filenames


# TODO - make this a @cl.step
# @cl.step(name="💾 Analyzing files")
async def add_files_to_db(extracted_docs: List[Document], extracted_filenames: List[str]) -> Chroma:
    # # Process the attached files
    # file_names = [f.name for f in files]
    # file_upload_msg = await new_message(content=f"🔄 Processing {len(files)} attached file(s): {', '.join(file_names)}...")

    # extracted_docs, extracted_filenames = await extract_docs_from_files(files)
    vector_store = cast(Chroma, cl.user_session.get("vector_store", None))

    if vector_store is None:
        print(f"No vector store found. Creating new...")
        vector_store = await create_vectordb()

    if len(extracted_docs) > 0:
        # for d in extracted_docs[:3]:
        #     print(f"- {d}")
        # print("...")
        async with cl.Step(name="Saving files to memory") as step:
            uuids = [str(uuid4()) for _ in range(len(extracted_docs))]
            # add docs to vector store

            vector_store.add_documents(documents=extracted_docs, ids=uuids)

            existing_chunks = cl.user_session.get("stored_documents", [])
            processed_filenames = cl.user_session.get(
                "processed_filenames", [])

            cl.user_session.set("vector_store", vector_store)
            cl.user_session.set("stored_documents",
                                existing_chunks + extracted_docs)
            cl.user_session.set("processed_filenames",
                                processed_filenames + extracted_filenames)
            cl.user_session.set("documents_loaded", True)

            num_stored_chunks = len(existing_chunks) + len(extracted_docs)
            cl.user_session.set("num_stored_chunks", num_stored_chunks)
            print(f"\n\nsuccessfully stored documents\n\n")
        # Success message
        # processed_files_list = ", ".join(processed_filenames)
        # await update_message(msg=file_upload_msg, content=f"✅ **Processing Complete!** \nLoaded **{len(processed_filenames)}** attached file(s): {processed_files_list}")

    # else:
        # await update_message(msg=file_upload_msg, content="❌ No files were successfully processed from attachments.")

    return vector_store


# TODO - make this a @cl.step
@cl.step(name="📎 Fetching patient files")
async def get_all_patient_docs():
    vector_store = cast(Chroma, cl.user_session.get("vector_store", None))
    if vector_store is None:
        return []

    metadatas = cl.user_session.get("metadatas", [])
    stored_texts = cl.user_session.get("stored_documents", [])
    print("🔍 Retrieving documents from vector store")
    # retrieve_msg = await new_message(content="🔍 Retrieving patient files...")

    # patient_docs = vector_store.get('patient_files')
    patient_docs = collection.get()
    # print(f"collection results: {patient_docs}")

    # for i, doc in enumerate(patient_docs["documents"]):
    #     print(f"\n\nDocument {i+1}:\n {doc}\n\n")
    #     print(f"\n\nMetadata:\n {patient_docs['metadatas'][i]}\n\n")

    # await update_message(msg=retrieve_msg, content=f"✅ {len(patient_docs)} documents retrieved.")

    return patient_docs["documents"]
