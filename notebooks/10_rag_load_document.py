# Text Document Loader
# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("./data/speech.txt", encoding="utf-8")
# data = loader.load()
# print(data[0].page_content)



# PDF Document Loader
# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("./data/medical_report.pdf")
# data = loader.load()
# print(len(data))



# CSV Document Loader
# from langchain_community.document_loaders import CSVLoader
# loader = CSVLoader("./data/employee_records.csv")
# docs = loader.load()
# print(len(docs))
# print(docs[:5])



# WebBase Document Loader
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader(web_path="https://www.morajenterprises.com/")
# data = loader.load()
# print(data)



# Wikipedia Document Loader
from langchain_community.document_loaders import WikipediaLoader
loader = WikipediaLoader(query="GenAI", load_max_docs=2)
data = loader.load()

for doc in data:
    print(doc.page_content)