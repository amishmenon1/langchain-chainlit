import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.message import new_message, update_message
from typing import cast
from langchain.prompts import ChatPromptTemplate
from templates.system.retriever import RETRIEVER_SYSTEM_TEMPLATE

CHROMA_PATH = 'chroma'

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


async def load_files_into_db(files) -> Chroma:
    # Process the attached files
    file_names = [f.name for f in files]

    file_upload_msg = await new_message(content=f"🔄 Processing {len(files)} attached file(s): {', '.join(file_names)}...")

    # Process all attached files
    all_pdf_pages = []
    processed_filenames = []

    for file in files:
        print(f"Loading attached file: {file.name}")
        try:
            loader = PyPDFLoader(file.path)
            pdf_pages = []
            async for page in loader.alazy_load():
                page.metadata['source_file'] = file.name
                pdf_pages.append(page)

            all_pdf_pages.extend(pdf_pages)
            processed_filenames.append(file.name)
            print(
                f"Successfully processed: {file.name} ({len(pdf_pages)} pages)")

        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            await new_message(content=f"❌ Error processing {file.name}: {str(e)}")

    if all_pdf_pages:
        # Split the text into chunks
        all_doc_chunks = text_splitter.split_documents(all_pdf_pages)
        print(f"Length of all text splits: {len(all_doc_chunks)}")

        # Create vector store with all documents
        vector_store = Chroma.from_documents(
            documents=all_doc_chunks, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

        # Update metadata to include source file information
        metadatas = []
        for i, chunk in enumerate(all_doc_chunks):
            source_file = chunk.metadata.get('source_file', 'unknown')
            page_num = chunk.metadata.get('page', 0)
            metadatas.append({
                "source": f"{i}-pl",
                "source_file": source_file,
                "page": page_num
            })

        # Store everything in user session (merge with existing if any)
        existing_chunks = cl.user_session.get("extracted_data", [])
        existing_metadatas = cl.user_session.get("metadatas", [])

        cl.user_session.set("vector_store", vector_store)
        cl.user_session.set("metadatas", existing_metadatas + metadatas)
        # cl.user_session.set("texts", existing_chunks + all_doc_chunks)
        cl.user_session.set("extracted_data", existing_chunks + all_doc_chunks)
        cl.user_session.set("documents_loaded", True)
        # print(f"all doc chunks: {all_doc_chunks}")
        # Success message
        processed_files_list = ", ".join(processed_filenames)
        await update_message(msg=file_upload_msg, content=f"✅ **Processing Complete!** \nLoaded **{len(processed_filenames)}** attached file(s): {processed_files_list}")

    else:
        await update_message(msg=file_upload_msg, content="❌ No files were successfully processed from attachments.")

    return vector_store


async def retrieve_chunks(message_content: str):
    vector_store = cast(Chroma, cl.user_session.get("vector_store", None))
    metadatas = cl.user_session.get("metadatas", [])
    stored_texts = cl.user_session.get("extracted_data", [])
    if vector_store:
        print("🔍 Retrieving documents from vector store")
        # await update_message(msg=sys_msg_2, content="🔍 Retrieving relevant documents...")
        print(f"NEW retrieve msg")
        retrieve_msg = await new_message(content="🔍 Retrieving relevant documents...")
        # First, get documents using similarity search
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5, })  # Increased to get more coverage

        similarity_docs = await retriever.ainvoke(message_content)

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
            for i, text_chunk in enumerate(stored_texts):
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
                for i, text_chunk in enumerate(stored_texts):
                    if metadatas[i].get('source_file') == missing_file:
                        file_docs.append(text_chunk)

                # Add the first few chunks from each missing file
                # Add up to 3 chunks per missing file
                comprehensive_docs.extend(file_docs[:3])

        print(f"📋 Total documents for analysis: {len(comprehensive_docs)}")

        # Combine all retrieved documents with source identification
        retrieved_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([
            f"SOURCE FILE: {get_source_file(doc, stored_texts, metadatas)}\n{doc.page_content}"
            for doc in comprehensive_docs
        ])
        print(f"updating retrieve msg")
        await update_message(msg=retrieve_msg, content=f"✅ Documents retrieved from {len(all_file_sources)} files")

        unique_sources = generate_sources(
            comprehensive_docs, stored_texts, metadatas)

        return retrieved_text, unique_sources


async def prompt_file_upload():
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

    await load_files_into_db(files)


def generate_sources(comprehensive_docs, texts, metadatas):
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
        print(f"unique sources: {unique_sources}")
        return unique_sources


def get_source_file(doc, texts, metadatas):
    """Helper function to get source file name for a document"""
    for i, text_chunk in enumerate(texts):
        if text_chunk.page_content == doc.page_content:
            return metadatas[i].get('source_file', 'Unknown')
    return 'Unknown'
