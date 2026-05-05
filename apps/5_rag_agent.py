from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    # load the document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()


    # split in chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = splitter.split_documents(documents=docs)


    # embedding the vector db
    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vector_db = InMemoryVectorStore.from_documents(
        documents=splitted_docs,
        embedding=embedding
    )


    # create agent - tool, llm, prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")


    @tool
    def retrieve_context(query: str):
        """Retrive documents relevant to query from the knowledge base."""

        context = ""

        docs = vector_db.similarity_search(query=query, k=3)
        for doc in docs:
            context += doc.page_content + "\n\n"

        return context


    system_prompt = """You are a helpful assistant that answers questoins using retrieved context.
        My Knowledge base consists of the details from the uploaded document.
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True



# upload UI
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload PDF", type=["pdf"], accept_multiple_files=True)

    if uploaded:
        with st.spinner("Processing..."):
            path = "./apps/doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()

# chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})



