from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

loader = PyPDFLoader("./data/medical_report.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = splitter.split_documents(docs)

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = InMemoryVectorStore.from_documents(
    documents=splitted_docs, embedding=embedding
)


@tool
def retriever_tool(query: str):
    """
    This tool can help you to retrieve the relevant data of the PDF Documents, and these pdf documents
    have the details about medical reports.
    """
    docs = vector_store.similarity_search(query=query, k=4)

    print("TOOL CALLED: ", query)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    return context


model = ChatGroq(model="openai/gpt-oss-20b")

system_prompt = """
    You are a helpful assistant that answers questions using retrieved context.
    ALWAYS use the 'retriever_tool' tool for questions requiring external knowledge.
"""


agent = create_agent(model=model, tools=[retriever_tool], system_prompt=system_prompt)

query = "What is the name of patient, and what is the name of doctors"

response = agent.invoke({"messages": [{"role": "user", "content": query}]})
result = response["messages"][-1].content

print(result)
