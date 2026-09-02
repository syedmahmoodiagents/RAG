from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

folder = Path("../Data_RAG")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


document_stores = {}
for file_path in folder.glob("*.txt"):
    document = Document(
        page_content=file_path.read_text(encoding="utf-8"),
        metadata={"source": file_path.name}
    )

    chunks = splitter.split_documents([document])
    vector_store = InMemoryVectorStore(embedding = embeddings)

    vector_store.add_documents(chunks)
    document_stores[file_path.name] = vector_store


print("\n================ DOCUMENT STORES ================\n")
print(document_stores.keys())

print("Number of vector stores:", len(document_stores))


question = """
Give me a complete analysis of cancer diagnosis and treatment.
"""

all_retrieved_docs = []
for filename, vector_store in document_stores.items():
    print("SEARCHING:", filename)

    docs = vector_store.similarity_search(question, k=2)

    for doc in docs:
        print("\nSOURCE:", doc.metadata["source"])
        print(doc.page_content)

        all_retrieved_docs.append(doc)


context = "\n\n".join(
    f"""
    SOURCE: {doc.metadata["source"]}

    {doc.page_content}
    """
    for doc in all_retrieved_docs
)

llm = ChatOllama(model="llama3.2:1b", temperature=0)

prompt = f"""
You are a research assistant.

Answer the question using ONLY the supplied context.

Question:
{question}

Context:
{context}

Instructions:
- Use information from all relevant documents.
- Do not invent information.
- Mention the source document for important claims.
- Combine information across documents into one answer.
"""

response = llm.invoke(prompt)

print("\n================ FINAL ANSWER ================\n")

print(response.content)