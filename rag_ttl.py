import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

import streamlit as st
import time

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage


CACHE_TTL = 30

if "cache_store" not in st.session_state:
    st.session_state.cache_store = {}

def normalize_query(query):
    return query.strip().lower()


def get_cache(query):

    query = normalize_query(query)

    cache = st.session_state.cache_store

    if query in cache:

        answer, timestamp = cache[query]

        if time.time() - timestamp < CACHE_TTL:
            return answer

        else:
            del cache[query]

    return None


def set_cache(query, answer):

    query = normalize_query(query)

    st.session_state.cache_store[query] = (answer, time.time())



@st.cache_resource
def load_models():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = [
        "LangGraph is a framework for building stateful AI agents.",
        "CrewAI is used for orchestrating multi-agent systems.",
        "RAG stands for Retrieval Augmented Generation.",
        "Python caching stores results temporarily in memory.",
        "Agentic AI systems combine reasoning, tools, and memory."
    ]

    vectorstore = FAISS.from_texts(docs, embedding_model)

    retriever = vectorstore.as_retriever()

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.3,
    )

    llm = ChatHuggingFace(llm=llm_endpoint)

    return retriever, llm


retriever, llm = load_models()



def rag_with_cache(query):

    cached_answer = get_cache(query)

    if cached_answer:
        return cached_answer, True

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    answer = response.content

    set_cache(query, answer)

    return answer, False




st.title("RAG with Python TTL Cache + GPT-OSS")

query = st.text_input("Ask a question")

if st.button("Ask"):
    if query:

        start = time.time()

        answer, cached = rag_with_cache(query)

        end = time.time()

        if cached:
            st.success("Returned from Cache")
        else:
            st.info("Generated from RAG")

        st.write("### Answer")
        st.write(answer)

        st.write("Response time:", round(end - start, 2), "seconds")




with st.expander("View Cache"):
    st.write(st.session_state.cache_store)
