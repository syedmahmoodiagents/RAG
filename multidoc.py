from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings, ChatOllama

folder = Path("../Data_RAG")

documents = []
for file_path in folder.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    documents.append(
        Document(page_content=text, metadata={"source": file_path.name}
        )
    )

print("Documents loaded:", len(documents))

for doc in documents:
    print(doc.metadata["source"])


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("\nTotal chunks:", len(chunks))

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embedding=embeddings)

vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

question = """
Give me a complete analysis of cancer diagnosis and treatment.
"""

retrieved_docs = retriever.invoke(question)


print("\n================ RETRIEVED CHUNKS ================\n")

for doc in retrieved_docs:

    print("SOURCE:", doc.metadata["source"])
    print(doc.page_content)
    print("-" * 50)


context = "\n\n".join(
    f"""
        SOURCE: {doc.metadata["source"]}

        {doc.page_content}
    """
    for doc in retrieved_docs
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
- Do not invent information.
- Use information from the relevant documents.
- Mention the source document for important claims.
"""

response = llm.invoke(prompt)

print("\n================ FINAL ANSWER ================\n")

print(response.content)