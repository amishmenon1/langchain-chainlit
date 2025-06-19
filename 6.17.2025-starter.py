from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig

from dotenv import load_dotenv

load_dotenv()

###### SETUP ######
workflow = StateGraph(state_schema=MessagesState)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
CHROMA_PATH = 'chroma'

###### FUNCTIONS ######


def load_files_into_vectordb(files=[]):
    # Process the attached files

    all_doc_chunks = []
    if len(files) > 0:
        for file in files:
            print(f"Loading attached file: {file.name}")
            loader = PyPDFLoader(file.path)
            documents = loader.load()
            all_doc_chunks += text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=all_doc_chunks, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    return vector_store


###### NODE FUNCTIONS ######

def generate(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


###### NODES ######
workflow.add_node("generate", generate)

###### EDGES ######

workflow.add_edge(START, "generate")

###### MEMORY ######

memory = MemorySaver()


###### COMPILE WORKFLOW ######
app = workflow.compile(checkpointer=memory)


@cl.on_chat_start
async def on_chat_start():
    pass
    # template = """Answer the question based only on the following context:

    # {context}

    # Question: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)

    # def format_docs(docs):
    #     return "\n\n".join([d.page_content for d in docs])

    # retriever = vector_store.as_retriever()

    # runnable = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )

    # cl.user_session.set("runnable", runnable)


@cl.on_message
async def main(message: cl.Message):
    answer = cl.Message(content="")
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    for msg, _ in app.stream(
        {"messages": [HumanMessage(content=message.content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()
