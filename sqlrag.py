
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder

engine = create_engine("sqlite:///company.db")

# CONNECT LANGCHAIN TO EXISTING SQLITE DATABASE

db = SQLDatabase(engine)

print("EXISTING SQLITE TABLES")

tables = db.get_usable_table_names()

print(tables)

print("DATABASE SCHEMA")

schema = db.get_table_info()

print(schema)

llm = ChatOllama(model="llama3.2:1b",temperature=0)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Chroma is NOT replacing SQLite.
# It is simply a semantic index over selected text from SQLite.

vector_store = Chroma(
    collection_name="sqlite_semantic_data",
    embedding_function=embeddings,
    persist_directory="./chroma_sql_rag"
)


def build_vector_index():


    print("BUILDING VECTOR INDEX")
    
    
    with engine.connect() as conn:

        rows = conn.execute(text("""
            SELECT customer_id,name,department,location,order_value,complaint
            FROM customers
        """)).fetchall()


    print(f"Number of SQLite rows read: {len(rows)}")


    # --------------------------------------------------------
    # Convert SQLite rows into LangChain Documents
    # --------------------------------------------------------

    documents = []

    ids = []


    for row in rows:

        document_text = f"""
            Customer ID: {row.customer_id}
            Name: {row.name}
            Department: {row.department}
            Location: {row.location}
            Order Value: {row.order_value}
            Complaint: {row.complaint}
            """


        document = Document(
            page_content=document_text,
            metadata={
                "customer_id": row.customer_id,
                "name": row.name,
                "location": row.location,
                "order_value": row.order_value
            }
        )

        documents.append(document)
        ids.append(str(row.customer_id))

    try:

        existing = vector_store.get()
        if existing["ids"]:
            vector_store.delete(ids=existing["ids"]) # Remove previous vector index content

    except Exception as e:

        print("Could not clear previous Chroma data:",e)


    # --------------------------------------------------------
    # Add documents to Chroma
    # --------------------------------------------------------

    if documents:
        vector_store.add_documents(documents=documents,ids=ids)

    print(f"Vector documents indexed: "f"{len(documents)}")



# SQL RETRIEVAL

# The LLM converts:
# Natural language -> SQL -> SQLite

def sql_retrieval(question):

    sql_prompt = f"""
        You are an expert SQLite SQL developer.

        DATABASE SCHEMA:

        {schema}

        USER QUESTION:

        {question}

        Generate a SQLite SELECT query that retrieves the
        information necessary to answer the user's question.

        Rules:

        1. Only use tables and columns present in the schema.
        2. Only generate SELECT statements.
        3. Do not generate INSERT.
        4. Do not generate UPDATE.
        5. Do not generate DELETE.
        6. Do not generate DROP.
        7. Do not generate ALTER.
        8. Return ONLY the SQL query.
        9. Do not use markdown code fences.
        """


    response = llm.invoke(sql_prompt)
    sql_query = response.content.strip()

    # --------------------------------------------------------
    # Remove markdown fences if the LLM accidentally produces
    # them.
    # --------------------------------------------------------

    sql_query = sql_query.replace("```sql","")
    sql_query = sql_query.replace("```","")
    sql_query = sql_query.strip()


    print(sql_query)

    forbidden = ["INSERT","UPDATE","DELETE","DROP","ALTER","TRUNCATE"]
    upper_sql = sql_query.upper()

    for word in forbidden:
        if word in upper_sql:
            raise ValueError(f"Unsafe SQL operation detected: {word}")


    result = db.run(sql_query)

    return result, sql_query


# ============================================================
# VECTOR RETRIEVAL
# ============================================================
#
# This searches Chroma semantically.
#
# Example:
#
# Question:
#
#     "customers who had delivery problems"
#
# It can retrieve:
#
#     "My package arrived three days late"
#
# even if the exact phrase "delivery problems" does not
# exist in the database.
#
# ============================================================

def vector_retrieval(question,k=5):

    results = vector_store.similarity_search_with_score(question,k=k)

    vector_results = []

    for document, score in results:

        vector_results.append({
            "document": document,
            "vector_score": float(score)
        })


    return vector_results


# ============================================================
# FUSION
# ============================================================
#
# Combine SQL retrieval + vector retrieval.
#
# For the semantic candidates we use Reciprocal Rank Fusion.
#
# RRF:
#
#       score = 1 / (60 + rank)
#
# ============================================================

def reciprocal_rank_fusion(vector_results):

    fused = {}
    # --------------------------------------------------------
    # Add vector results
    # --------------------------------------------------------

    for rank, item in enumerate(vector_results,start=1):

        document = item["document"]

        customer_id = document.metadata["customer_id"]

        fused[customer_id] = {

            "customer_id": customer_id,

            "document": document,

            "fusion_score": 1.0 / (60 + rank),

            "source": "vector"

        }


    # --------------------------------------------------------
    # Sort by fusion score
    # --------------------------------------------------------

    fused_results = sorted(fused.values(),key=lambda x: x["fusion_score"],reverse=True)

    return fused_results



# GET COMPLETE SQLITE ROW

# After vector retrieval gives us customer IDs, we go back
# to SQLite and obtain the authoritative structured data.
#
# This is an important hybrid-RAG pattern.

def get_sqlite_rows(customer_ids):

    if not customer_ids:
        return []


    placeholders = ",".join(["?"] * len(customer_ids))

    query = f"""
        SELECT customer_id,name,department,location,order_value,complaint
        FROM customers WHERE customer_id IN ({placeholders})
    """

    with engine.connect() as conn:

        result = conn.exec_driver_sql(query,tuple(customer_ids))
        rows = result.fetchall()

    return rows


# CROSS ENCODER RERANKING

# Vector search:
#
#       Fast but approximate
#
# CrossEncoder:
#
#       Slower but more accurate
#
#
# It receives:
#
#       Query + Candidate
#
# and calculates:
#
#       Relevance Score

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def rerank(question,candidates,top_k=3):

    if not candidates:
        return []

    pairs = []
    for candidate in candidates:

        candidate_text = f"""
            Customer ID: {candidate['customer_id']}
            Name: {candidate['name']}
            Department: {candidate['department']}
            Location: {candidate['location']}
            Order Value: {candidate['order_value']}
            Complaint: {candidate['complaint']}
            """


        pairs.append((question,candidate_text))


    # --------------------------------------------------------
    # CrossEncoder scoring
    # --------------------------------------------------------

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates,scores):
        candidate["rerank_score"] = float(score)


    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]

def generate_final_answer(question,sql_result,ranked_candidates):

    context_parts = []
    for candidate in ranked_candidates:

        context_parts.append(
            f"""
            Customer ID: {candidate['customer_id']}
            Name: {candidate['name']}
            Department: {candidate['department']}
            Location: {candidate['location']}
            Order Value: {candidate['order_value']}
            Complaint: {candidate['complaint']}
            Source: {candidate['source']}
            Reranker Score: {candidate['rerank_score']}
            """
        )


    vector_context = ("\n-------------------------\n".join(context_parts))

    prompt = f"""
        You are a data assistant.

        Answer the user's question using ONLY the retrieved
        information.

        USER QUESTION:

        {question}


        SQL DATABASE RESULT:

        {sql_result}


        SEMANTICALLY RETRIEVED AND RERANKED DATA:

        {vector_context}


        RULES:

        1. Do not invent facts.
        2. Do not assume information not present in the data.
        3. Prefer exact SQL results for numerical calculations.
        4. Use semantic results for textual/meaning-based questions.
        5. If the available information is insufficient, say so.
        6. Give a concise and clear answer.
        """


    response = llm.invoke(prompt)


    return response.content



# COMPLETE HYBRID RAG PIPELINE

def hybrid_rag(question):

    print("\nUSER QUESTION:")
    print(question)

    print("STEP 1 - SQL RETRIEVAL")
    
    try:

        sql_result, generated_sql = sql_retrieval(question)

    except Exception as e:
        print("\nSQL retrieval failed:",e)
        sql_result = ""
        generated_sql = ""


    print("\nSQL RESULT:")
    print(sql_result)

    print("STEP 2 - VECTOR RETRIEVAL")
    

    vector_results = vector_retrieval(question,k=5)


    for rank, item in enumerate(vector_results,start=1):

        document = item["document"]
        print(f"\nVector Rank: {rank}")
        print("Customer ID:", document.metadata["customer_id"])
        print("Name:",document.metadata["name"])

        print("Vector Score:",item["vector_score"])


    
    # STEP 3 - FUSION
    
    print("STEP 3 - FUSION")
    
    fused_results = reciprocal_rank_fusion(vector_results)


    print("Fused candidates:",len(fused_results))

    # STEP 4 - GET AUTHORITATIVE DATA FROM SQLITE
    
    customer_ids = []
    for item in fused_results:
        customer_ids.append(item["customer_id"])

    sqlite_rows = get_sqlite_rows(customer_ids)


    # --------------------------------------------------------
    # Convert SQLite rows into candidates
    # --------------------------------------------------------

    candidates = []
    for row in sqlite_rows:

        candidates.append({
            "customer_id": row.customer_id,
            "name": row.name,
            "department": row.department,
            "location": row.location,
            "order_value": row.order_value,
            "complaint": row.complaint,
            "source": "vector"
        })


    
    # STEP 5 - CROSS ENCODER RERANKING
    
    print("STEP 5 - CROSS ENCODER RERANKING")
    
    ranked_candidates = rerank(question,candidates,top_k=3)


    for rank, candidate in enumerate(ranked_candidates,start=1):
        print(f"\nRerank {rank}")
        print("Customer ID:",candidate["customer_id"])
        print("Name:",candidate["name"])
        print("Complaint:",candidate["complaint"])
        print("CrossEncoder Score:",candidate["rerank_score"])



    # STEP 6 - FINAL LLM

    print("STEP 6 - FINAL LLM ANSWER")
    answer = generate_final_answer(question,sql_result,ranked_candidates)


    print("\nFINAL ANSWER:")
    print(answer)


    return answer



# BUILD VECTOR INDEX

# IMPORTANT:
#
# Run this when you initially create the vector index.
#
# It READS your existing SQLite data.
#
# It DOES NOT INSERT OR MODIFY ANY SQLite data.

build_vector_index()

question = """
Find the customers who had delivery issues and tell me
their order values.
""" 

hybrid_rag(question)

# OTHER EXAMPLE QUESTIONS


# hybrid_rag(
#     "Which customers complained about delayed delivery?"
# )


# hybrid_rag(
#     "What is the total order value of customers
#      who had delivery problems?"
# )


# hybrid_rag(
#     "Which Bangalore customers had delivery issues?"
# )


# hybrid_rag(
#     "Which department has the highest total order value?"
# )