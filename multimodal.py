
from pathlib import Path
import base64
import requests

# LangChain
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data_dir")

TEXT_FILE = DATA_DIR/"document.txt"
IMAGE_FILE = DATA_DIR/"diagram.jpeg"

TEXT_LLM = "llama3.2:1b"
VISION_LLM = "llava-phi3"

llm = ChatOllama(model=TEXT_LLM, temperature=0)
vision_llm = ChatOllama(model=VISION_LLM, temperature=0)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
text = TEXT_FILE.read_text(encoding="utf-8")

text_document = Document(
    page_content=text, metadata={"source": TEXT_FILE.name, "type": "text"}
)


print("=" * 70)
print("TEXT DOCUMENT LOADED")
print("=" * 70)

print(text_document.page_content)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


text_chunks = splitter.split_documents([text_document])


print("\n" + "=" * 70)
print("TEXT CHUNKS")
print("=" * 70)

for i, chunk in enumerate(text_chunks):

    print(f"\nTEXT CHUNK {i}")
    print(chunk.page_content)



# We don't simply throw the raw image into a normal text
# embedding model.
#
# First we ask a vision LLM to understand the image.
# diagram.png -> Vision LLM 
#      ↓
# "The diagram contains three layers:
#  ingestion -> processing -> storage"
#
# That description can then be embedded.

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_base64 = image_to_base64(IMAGE_FILE)
vision_prompt = """
    Analyze this image for use in a RAG system.

    Extract:

    1. All important visible text.
    2. Important entities.
    3. Relationships between entities.
    4. Arrows and data flow.
    5. Tables, numbers and labels.
    6. Diagram structure.
    7. Any semantic information that OCR alone might miss.

    Return a detailed textual description of the image.
    """


vision_message = HumanMessage(
    content=[
        {"type": "text", "text": vision_prompt},

        {
            "type": "image_url",
            "image_url": {"url": (f"data:image/png;base64," f"{image_base64}")}
        }
    ]
)

vision_response = vision_llm.invoke([vision_message])
image_description = vision_response.content

print("\n" + "=" * 70)
print("VISION LLM IMAGE DESCRIPTION")
print("=" * 70)

print(image_description)

# We preserve the fact that this information originated
# from an IMAGE.

image_document = Document(
    page_content=image_description,
    metadata={"source": IMAGE_FILE.name, "type": "image"}
)

# CREATE IMAGE CHUNKS
image_chunks = splitter.split_documents([image_document])

# CREATE ONE COMMON VECTOR STORE
# We now have:
# TEXT CHUNKS + IMAGE-DERIVED TEXT CHUNKS
# Both can be stored in the same vector store.
#
# IMPORTANT:
#
# We are NOT merging the raw image and raw text into one
# embedding.
#
# Instead:
#
# Text:
#       text -> embedding
#
# Image:
#       image -> vision LLM -> description -> embedding


all_documents = (text_chunks + image_chunks)

vector_store = Chroma.from_documents(
    documents=all_documents,
    embedding=embedding_model,
    collection_name="multimodal_rag"
)


print("\n" + "=" * 70)
print("VECTOR STORE CREATED")
print("=" * 70)

print(f"Total indexed chunks: {len(all_documents)}")



# CREATE TWO RETRIEVAL PATHS
#
# We deliberately create:
#
#       TEXT RETRIEVAL
#
# and
#
#       IMAGE RETRIEVAL
#
# Then we fuse their results.


def retrieve_text_candidates(question, k=5):
    results = vector_store.similarity_search(question, k=k)

    return [doc for doc in results if doc.metadata.get("type") == "text"]


def retrieve_image_candidates(question, k=5):
    results = vector_store.similarity_search(question, k=k)

    return [doc for doc in results if doc.metadata.get("type") == "image"]



# Because both image descriptions and text are in the same
# vector store, another production approach would be:
#
#     similarity_search(question, k=10)
#
# and then let a reranker decide which modality is useful.
#
# Here we explicitly separate them so that we can demonstrate
# multimodal fusion.



question = input("\nEnter your question: ")

# Example:
#
# How does data flow through the three layers?
#
# or:
#
# What are the components shown in the architecture?


text_candidates = retrieve_text_candidates(question, k=5)


print("\n" + "=" * 70)
print("TEXT RETRIEVAL")
print("=" * 70)

for rank, doc in enumerate(text_candidates, start=1):
    print(f"\nRank {rank}")
    print("Source:",doc.metadata["source"])
    print(doc.page_content)


image_candidates = retrieve_image_candidates(question,k=5)


print("\n" + "=" * 70)
print("IMAGE RETRIEVAL")
print("=" * 70)

for rank, doc in enumerate(image_candidates,start=1):
    print(f"\nRank {rank}")
    print("Source:", doc.metadata["source"])
    print(doc.page_content)


# RECIPROCAL RANK FUSION


# We now have two ranked lists:
#
# TEXT:
#
# T1
# T2
# T3
#
#
# IMAGE:
#
# I1
# I2
#
#
# RRF combines the rankings.
#
# Formula:
# RRF(d) = SUM 1 / (k + rank(d))
# Usually k = 60.
# ============================================================


def reciprocal_rank_fusion(result_lists,rrf_k=60):
    scores = {}
    documents_by_id = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            # Create a unique ID for each candidate
            doc_id = (doc.metadata.get("source", "") + "::" + doc.page_content)
            score = 1 / (rrf_k + rank) # RRF score
            scores[doc_id] = (scores.get(doc_id, 0) + score)
            documents_by_id[doc_id] = doc


    # Sort highest RRF score first

    ranked_ids = sorted(scores,key=scores.get,reverse=True)


    fused_results = []

    for doc_id in ranked_ids:
        doc = documents_by_id[doc_id]
        fused_results.append((doc, scores[doc_id]))

    return fused_results

# FUSE TEXT + IMAGE RESULTS
fused_results = reciprocal_rank_fusion([text_candidates, image_candidates])


print("\n" + "=" * 70)
print("RRF FUSION RESULTS")
print("=" * 70)


for rank, (doc,score) in enumerate(fused_results,start=1):

    print(f"\nRank {rank}")
    print("Source:", doc.metadata["source"])
    print("Type:", doc.metadata["type"])

    print("RRF Score:", score)

    print(doc.page_content)


# Now we have:
#
#       TEXT candidates
#              +
#       IMAGE candidates
#              ↓
#            RRF
#              ↓
#       FUSED candidates

# We now want to determine:
#
# "Which candidate is actually most relevant to the
#  user's question?"
#
# A sophisticated implementation would use a dedicated
# cross-encoder or multimodal reranker.
#
# For a simple Ollama implementation, we can ask an LLM
# to score each candidate.
#
# IMPORTANT:
#
# This is an LLM-based reranking demonstration, not a
# specialized production cross-encoder.


def rerank_with_llm(question,fused_results,top_n=5):

    reranked = []
    for doc, rrf_score in fused_results:
        candidate_type = (doc.metadata.get("type","unknown"))

        rerank_prompt = f"""
            You are a retrieval reranker.

            User question:
            {question}

            Candidate type:
            {candidate_type}

            Candidate content:
            {doc.page_content}

            Determine how relevant this candidate is
            to answering the user's question.

            Return ONLY a number between 0 and 100.

            100 = directly answers the question.
            75  = highly relevant.
            50  = somewhat relevant.
            25  = weakly relevant.
            0   = irrelevant.
            """

        response = llm.invoke(rerank_prompt)
        raw_score = response.content.strip()
        # Try to extract numeric score

        import re
        match = re.search(r"\d+(\.\d+)?", raw_score)

        if match:
            rerank_score = float(match.group())

        else:
            rerank_score = 0

        reranked.append({
                "document": doc,
                "rrf_score": rrf_score,
                "rerank_score": rerank_score
            })


    # Sort by reranker score

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]

reranked_results = rerank_with_llm(question,fused_results,top_n=5)


print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)


for rank, item in enumerate(reranked_results, start=1):

    doc = item["document"]
    print(f"\nRank {rank}")
    print("Source:",doc.metadata["source"])
    print("Type:",doc.metadata["type"])
    print("RRF Score:",item["rrf_score"])
    print("Reranker Score:",item["rerank_score"])
    print("Content:", doc.page_content)


# Only the best reranked candidates are passed to the final
# generation model.

final_context_parts = []

for rank, item in enumerate(reranked_results, start=1):

    doc = item["document"]

    final_context_parts.append(f"""
        ===== EVIDENCE {rank} =====

        SOURCE:
        {doc.metadata["source"]}

        TYPE:
        {doc.metadata["type"]}

        CONTENT:
        {doc.page_content}
        """
    )


final_context = "\n".join(final_context_parts)

final_prompt = f"""
    You are a multimodal RAG assistant.

    Answer the user's question using ONLY
    the evidence supplied below.

    USER QUESTION:
    {question}


    RETRIEVED EVIDENCE:

    {final_context}


    INSTRUCTIONS:

    1. Combine information from both text and image-derived
    evidence when relevant.

    2. Do not invent facts.

    3. If the image contains information that complements
    the text, explain the relationship.

    4. Identify the source of important information.

    5. If the evidence is insufficient, say so.

    6. Give a clear and concise answer.
    """


final_response = llm.invoke(final_prompt)

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(final_response.content)