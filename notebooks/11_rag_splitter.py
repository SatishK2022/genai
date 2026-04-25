from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# text = "Today, I want to speak from the heart about three pillars that define the future of our nation our youth, our education system, and our farmers."

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
# chunks = text_splitter.split_text(text=text)
# print(chunks)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text_loader = TextLoader("./data/speech.txt", encoding="utf-8")
texts = text_loader.load()
text_chunks = splitter.split_documents(texts)
print(text_chunks)

pdf_loader = PyPDFLoader("./data/medical_report.pdf")
pdf_texts = pdf_loader.load()
pdf_chunks = splitter.split_documents(pdf_texts)
print(pdf_chunks)


