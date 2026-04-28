from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# Step 1: Load the Document
loader = PyPDFLoader("./data/data_science_syllabus.pdf")
docs = loader.load()

# Step 2: Split the Document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_data = splitter.split_documents(docs)

# Step 3: Create Embeddings
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Step 4: Store the data in Vector DB
vector_store = Chroma.from_documents(
    documents=splitted_data,
    embedding=embedding
)

# query = "Machine learning and data science"
# data = vector_store.similarity_search(query=query)

# context = ""
# for doc in data:
#     context += doc.page_content + "\n"


# Step 5: Call the LLM with Context
llm = ChatGroq(model="openai/gpt-oss-20b")

# res = llm.invoke(f"Can you provide me the answer based on the provided context for my question. Context: {context} Question: {query}")
# print(res.content)




def get_context(query: str):
    data = vector_store.similarity_search(query=query)
    
    context = ""
    for doc in data:
        context += doc.page_content + "\n"

    return {
        "context": context,
        "question": query
    }

prompt = PromptTemplate.from_template("""
    You are a helpful assistant and provide answers based on the context for the user question.
    If you don't know the answer then you can say "I Don't Know That"
    Context: {context}
    Question: {question}
""")

rag_chain = get_context | prompt | llm

res = rag_chain.invoke("What is the duration of module 3")
print(res.content)


