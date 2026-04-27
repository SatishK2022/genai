from dotenv import load_dotenv
load_dotenv()

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Python is a programming language popular for data science",
    "India won the cricket world cup in 2011",
    "AI models can understand and generate human language"
]

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Create Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# vector = embeddings.embed_documents(documents)
# print(len(vector))


# Store the Vector Embedding in Vector Store
from langchain_chroma import Chroma

vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="./vector_db"
)

query = "Data Science"
result = vector_store.similarity_search(query=query, k=2)
print(result)
