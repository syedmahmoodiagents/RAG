import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b'))

loader = TextLoader("paracetamol.txt") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

split_docs = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(split_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "What are the key side effects mentioned?"
retrieved_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in retrieved_docs])
response = llm.invoke(f"Context: {context} \n\nQuestion: What are the key side effects mentioned?")
print(response.content)

# print(context)

