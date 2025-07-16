import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document, StrOutputParser

from utils.message import new_message, update_message
from typing import cast, List
from langchain.prompts import ChatPromptTemplate
from uuid import uuid4
from langchain_docling import DoclingLoader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_docling.loader import ExportType
import os
import chromadb
from templates.system.generated_pdf_processor import PDF_PROCESSOR_SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor
import asyncio
# from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
document_converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={
                                       InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

formatter_llm = ChatOpenAI(model="gpt-4o-mini")
CHROMA_PATH = 'chroma'
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("patient_files")

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# DOCLING
EXPORT_TYPE = ExportType.DOC_CHUNKS


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


# def clean_metadata(metadata: dict) -> dict:
#     cleaned = {}
#     for k, v in metadata.items():
#         if isinstance(v, (str, int, float, bool)) or v is None:
#             cleaned[k] = v
#         else:
#             print(f"Removed non-primitive metadata key: {k} → {type(v)}")
#     return cleaned


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}


@cl.step(name="🔍 Scanning patient files")
async def extract_docs_from_files(files) -> List[Document]:

    # Process all attached files
    extracted_docs = []
    processed_filenames = []

    for file in files:
        print(f"Loading attached file: {file.name}")
        try:
            loader = DoclingLoader(file_path=file.path,
                                   export_type=EXPORT_TYPE
                                   )
            docs = []
            async for page in loader.alazy_load():
                page.metadata['source_file'] = file.name
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

    prompt = ChatPromptTemplate.from_template(
        PDF_PROCESSOR_SYSTEM_PROMPT)
    chain = prompt | formatter_llm | StrOutputParser()

    formatted_context = chain.invoke({"context": extracted_docs})
    print(f"extracted context: {formatted_context}")
    return extracted_docs, formatted_context


# # ✅ This must be a sync function for ThreadPoolExecutor
# def extract_docs_from_single_file(file_path: str, file_name: str):
#     print(f"⏳ Loading {file_name}...")
#     try:
#         loader = DoclingLoader(file_path=file_path, export_type=EXPORT_TYPE)
#         docs = list(loader.lazy_load())
#         for page in docs:
#             page.metadata["source_file"] = file_name
#             page.metadata = clean_metadata(page.metadata)
#         print(f"✅ Loaded {file_name} ({len(docs)} pages)")
#         return docs
#     except Exception as e:
#         print(f"❌ Error loading {file_name}: {str(e)}")
#         return []


# # ✅ Async wrapper to run sync file processing in parallel
# async def extract_docs_from_files_parallel(files) -> tuple[list[Document], list[str]]:
#     loop = asyncio.get_running_loop()
#     extracted_docs = []
#     processed_filenames = []

#     with ThreadPoolExecutor() as executor:
#         tasks = [
#             loop.run_in_executor(
#                 executor, extract_docs_from_single_file, file.path, file.name
#             )
#             for file in files
#         ]
#         results = await asyncio.gather(*tasks)

#     for file, docs in zip(files, results):
#         if docs:
#             extracted_docs.extend(docs)
#             processed_filenames.append(file.name)

#     return extracted_docs, processed_filenames


# TODO - make this a @cl.step
# @cl.step(name="💾 Analyzing files")
async def add_files_to_db(extracted_docs: List[Document]) -> Chroma:
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
            # processed_filenames = cl.user_session.get(
            #     "processed_filenames", [])

            cl.user_session.set("vector_store", vector_store)
            cl.user_session.set("stored_documents",
                                existing_chunks + extracted_docs)
            # cl.user_session.set("processed_filenames",
            #                     processed_filenames + extracted_filenames)
            cl.user_session.set("documents_loaded", True)

            num_stored_chunks = len(existing_chunks) + len(extracted_docs)
            cl.user_session.set("num_stored_chunks", num_stored_chunks)
            print(f"\n\nsuccessfully stored documents\n\n")

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
