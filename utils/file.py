import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CHROMA_PATH = 'chroma'

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


async def load_files_into_db(files) -> Chroma:
    # Process the attached files
    file_names = [f.name for f in files]
    # msg = cl.Message(
    #     content=f"🔄 Processing {len(files)} attached file(s): {', '.join(file_names)}...")
    # await msg.send()

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
            await cl.Message(content=f"❌ Error processing {file.name}: {str(e)}").send()

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
        existing_texts = cl.user_session.get("texts", [])
        existing_metadatas = cl.user_session.get("metadatas", [])

        cl.user_session.set("vector_store", vector_store)
        cl.user_session.set("metadatas", existing_metadatas + metadatas)
        cl.user_session.set("texts", existing_texts + all_doc_chunks)
        cl.user_session.set("documents_loaded", True)

        # Success message
        processed_files_list = ", ".join(processed_filenames)
        # msg.content = f"✅ **Processing Complete!** \nLoaded **{len(processed_filenames)}** attached file(s): {processed_files_list}\n🔄 Now analyzing your question..."
        # await msg.update()
    else:
        processed_files_list = []
        # await cl.Message(content="❌ No files were successfully processed from attachments.").send()

    return processed_files_list


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
