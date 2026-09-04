
from typing import TypedDict, List, Dict, Any
from sqlalchemy import create_engine, text

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

from langgraph.graph import (StateGraph,START,END)

DATABASE_URL = (
    "sqlite:///company.db"
)

engine = create_engine(
    DATABASE_URL
)

======================================================

llm = ChatOllama(model="llama3.2:1b",temperature=0)

# OLLAMA EMBEDDINGS

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# CHROMA VECTOR STORE

vector_store = Chroma(
    collection_name="sqlite_semantic_data",
    embedding_function=embeddings,
    persist_directory="./chroma_sql_rag"
)

# CROSS ENCODER

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# GET DATABASE SCHEMA

def get_database_schema():
    with engine.connect() as conn:
        tables = conn.execute(
            text("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
        ).fetchall()


        schema_parts = []


        for table in tables:

            table_name = table[0]
            columns = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()

            schema_parts.append(f"\nTABLE: {table_name}")

            for column in columns:

                column_name = column[1]
                column_type = column[2]
                schema_parts.append(f"{column_name} {column_type}")


        return "\n".join(schema_parts)


SCHEMA = get_database_schema()

print("DATABASE SCHEMA")

print(SCHEMA)


# ============================================================
# 7. LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):

    # Original user question
    question: str

    # Agent's retrieval decision
    route: str

    # Generated SQL
    sql_query: str

    # SQL result
    sql_result: str

    # Vector results
    vector_results: List[Dict[str, Any]]

    # Fused results
    fused_results: List[Dict[str, Any]]

    # Reranked results
    reranked_results: List[Dict[str, Any]]

    # Evidence evaluation
    sufficient: bool

    # Number of retrieval attempts
    attempts: int

    # Final answer
    answer: str

    # Error information
    error: str


# QUERY ANALYZER / AGENT

# The LLM decides:
#
# SQL
# VECTOR
# BOTH
#
# This is the first agentic decision.


def analyze_query(state: RAGState):

    question = state["question"]


    prompt = f"""
You are the routing agent in an Agentic RAG system.

The system has two retrieval tools.

TOOL 1: SQL

Use SQL when the question requires:

- numerical calculations
- SUM
- COUNT
- AVG
- MIN
- MAX
- filtering
- sorting
- grouping
- dates
- exact structured values
- joins

TOOL 2: VECTOR

Use VECTOR when the question requires:

- semantic search
- meaning-based search
- complaints
- descriptions
- comments
- reviews
- natural language concepts

You can use BOTH when the question requires
both semantic and structured information.

DATABASE SCHEMA:

{SCHEMA}

USER QUESTION:

{question}

Return ONLY one of:

SQL
VECTOR
BOTH
"""

    response = llm.invoke(prompt)

    route = response.content.strip().upper()
    if "BOTH" in route:
        route = "BOTH"

    elif "VECTOR" in route:
        route = "VECTOR"

    else:

        route = "SQL"


    
    print("AGENT ROUTING DECISION")
    
    print(route)


    state["route"] = route
    state["attempts"] = state.get("attempts",0)


    return state


# ============================================================
# 9. ROUTER
# ============================================================

def route_after_analysis(state: RAGState):

    route = state["route"]

    if route == "SQL":
        return "sql"


    if route == "VECTOR":
        return "vector"


    return "both"


# ============================================================
# 10. SQL RETRIEVAL
# ============================================================

def sql_retrieval(state: RAGState):

    question = state["question"]


    prompt = f"""
        You are an expert SQLite SQL developer.

        DATABASE SCHEMA:

        {SCHEMA}

        USER QUESTION:

        {question}

        Generate a SQLite SELECT query.

        Rules:

        1. Use only tables and columns in the schema.
        2. Only generate SELECT statements.
        3. Do not generate INSERT.
        4. Do not generate UPDATE.
        5. Do not generate DELETE.
        6. Do not generate DROP.
        7. Do not generate ALTER.
        8. Return ONLY SQL.
        9. Do not use markdown fences.
        """


    response = llm.invoke(prompt)
    sql_query = response.content.strip()
    sql_query = sql_query.replace("```sql","")
    sql_query = sql_query.replace("```","")
    sql_query = sql_query.strip()


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    forbidden = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE"
    ]


    upper_sql = sql_query.upper()


    for word in forbidden:
        if word in upper_sql:
            state["error"] = (f"Unsafe SQL detected: {word}")
            state["sql_result"] = ""
            return state


    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()

        state["sql_query"] = sql_query
        state["sql_result"] = str(rows)


    except Exception as e:

        state["sql_query"] = sql_query
        state["sql_result"] = ""
        state["error"] = str(e)


    print("SQL RETRIEVAL")
    
    print("\nSQL:")
    print(state.get("sql_query"))

    print("\nRESULT:")
    print(state.get("sql_result"))


    return state



# VECTOR RETRIEVAL

def vector_retrieval(state: RAGState):

    question = state["question"]
    # --------------------------------------------------------
    # Increase K if this is a retry
    # --------------------------------------------------------

    attempts = state.get("attempts",0)
    k = 5


    if attempts >= 1:
        k = 10


    if attempts >= 2:
        k = 20


    results = vector_store.similarity_search_with_score(question,k=k)

    vector_results = []
    for rank, (document,distance) in enumerate(results,start=1):

        vector_results.append({
            "rank": rank,
            "text":document.page_content,
            "metadata":document.metadata,
            "vector_score":float(distance)

        })


    state["vector_results"] = (vector_results)

    print("VECTOR RETRIEVAL")
    
    print("Retrieved:",len(vector_results))

    for item in vector_results:

        print("\nRank:",item["rank"])

        print("Customer:",item["metadata"].get("customer_id"))

        print("Vector score:",item["vector_score"])

    return state


def fusion(state: RAGState):
    vector_results = state.get("vector_results",[])
    fused = {}
    # --------------------------------------------------------
    # Vector candidates
    # --------------------------------------------------------

    for item in vector_results:
        metadata = item["metadata"]
        customer_id = metadata.get("customer_id")

        if customer_id is None:
            customer_id = (metadata.get("id"))

        if customer_id is None:
            continue

        if customer_id not in fused:

            fused[customer_id] = {
                "customer_id":customer_id,
                "text":item["text"],
                "metadata":metadata,
                "fusion_score":0.0,
                "source":"vector"}

        rank = item["rank"]
        fused[customer_id]["fusion_score"] += 1.0 / (60 + rank)


   
    # SQL result is kept as a separate candidate because
    # aggregate SQL results may not map directly to individual
    # vector documents.
    
    sql_result = state.get("sql_result","")

    if sql_result:

        fused["SQL_RESULT"] = {
            "customer_id":"SQL_RESULT",
            "text": sql_result, 
            "metadata": {}, 
            "fusion_score": 1.0,
            "source":"sql"
        }


    fused_results = sorted(fused.values(), key=lambda x: x["fusion_score"],reverse=True)
    state["fused_results"] = (fused_results)

    print("FUSION")
    
    print("Candidates:",len(fused_results))


    return state



# RERANKING

def rerank(state: RAGState):

    question = state["question"]


    candidates = state.get(
        "fused_results",
        []
    )


    if not candidates:

        state["reranked_results"] = []

        return state


    pairs = []


    for candidate in candidates:

        pairs.append((question,candidate["text"]))

    # --------------------------------------------------------
    # CrossEncoder
    # --------------------------------------------------------

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates,scores):

        candidate["rerank_score"] = (float(score))


    ranked = sorted(candidates,key=lambda x: x["rerank_score"],reverse=True)


    state["reranked_results"] = (ranked[:5])


    print("CROSS ENCODER RERANKING")
   

    for rank, candidate in enumerate(state["reranked_results"],start=1):
        print("\nRank:",rank)

        print("Customer:",candidate["customer_id"])
        print("Score:",candidate["rerank_score"])

    return state


# ============================================================
# EVIDENCE CHECKER
# ============================================================
#
# This is another agentic decision.
#
# The LLM decides whether the retrieved information is enough
# to answer the question.
# ============================================================

def evidence_checker(state: RAGState):
    question = state["question"]
    sql_result = state.get("sql_result","")

    ranked = state.get("reranked_results",[])
    context = "\n\n".join(item["text"] for item in ranked)
    prompt = f"""
        You are an evidence evaluation agent.

        Determine whether the retrieved information is sufficient
        to answer the user's question accurately.

        QUESTION:

        {question}

        SQL RESULT:

        {sql_result}

        RERANKED SEMANTIC RESULTS:

        {context}

        Return ONLY:

        YES

        or

        NO

        Return NO if important information is missing.
        """

    response = llm.invoke(prompt)
    decision = response.content.strip().upper()

    sufficient = (decision.startswith("YES"))
    state["sufficient"] = (sufficient)
    print("EVIDENCE CHECK")
    print("Sufficient:",sufficient)

    return state

# DECIDE WHETHER TO ANSWER OR RETRY
def after_evidence_check(state: RAGState):
    if state.get("sufficient",False):
        return "answer"
    attempts = state.get("attempts",0)
    # Prevent infinite loops
    if attempts >= 2:
        return "answer"
    return "retry"


# RETRY NODE

# If evidence is insufficient, the agent increases the
# retrieval depth and performs another retrieval cycle.

def retry_search(state: RAGState):
    attempts = state.get("attempts",0)
    state["attempts"] = (attempts + 1)
    print("AGENT RETRY")
    print("Retrieval attempt:",state["attempts"])

    return state

# FINAL ANSWER

def generate_answer(state: RAGState):
    question = state["question"]
    sql_result = state.get("sql_result","")
    ranked = state.get("reranked_results",[])

    semantic_context = "\n\n".join(item["text"] for item in ranked)

    prompt = f"""
        You are the final answer generator in an Agentic RAG system.

        USER QUESTION:

        {question}

        STRUCTURED SQL RESULT:

        {sql_result}

        SEMANTICALLY RETRIEVED AND RERANKED CONTEXT:

        {semantic_context}

        Instructions:

        1. Answer using only the retrieved information.
        2. Do not invent facts.
        3. Prefer SQLite results for exact numerical calculations.
        4. Use semantic context for textual/meaning-based information.
        5. If the evidence is insufficient, clearly say so.
        6. Give a concise natural-language answer.
        """
    response = llm.invoke(prompt)
    state["answer"] = (response.content)
    print("FINAL ANSWER")
    print(state["answer"])

    return state


graph = StateGraph(RAGState)

# ADD NODES
graph.add_node("analyze_query",analyze_query)
graph.add_node("sql_retrieval",sql_retrieval)
graph.add_node("vector_retrieval",vector_retrieval)
graph.add_node("fusion",fusion)
graph.add_node("rerank",rerank)
graph.add_node("evidence_checker",evidence_checker)
graph.add_node("retry_search",retry_search)
graph.add_node("generate_answer",generate_answer)
# START → QUERY ANALYZER
graph.add_edge(START,"analyze_query")
# QUERY ANALYZER → ROUTER
graph.add_conditional_edges(
    "analyze_query",route_after_analysis,
    {"sql":"sql_retrieval","vector":"vector_retrieval","both":"sql_retrieval"}
)

graph.add_edge("sql_retrieval","fusion")
graph.add_edge("vector_retrieval","fusion")
# Then we explicitly perform vector retrieval.
def sql_to_vector_or_fusion(state: RAGState):
    if state["route"] == "BOTH":
        return "vector"
    return "fusion"


# Replace SQL → FUSION with conditional routing

# StateGraph edges added above can be replaced conceptually
# by conditional routing. For simplicity we use a dedicated
# conditional edge here.

graph = StateGraph(RAGState)

graph.add_node("analyze_query",analyze_query)
graph.add_node("sql_retrieval",sql_retrieval)
graph.add_node("vector_retrieval",vector_retrieval)
graph.add_node("fusion",fusion)
graph.add_node("rerank",rerank)
graph.add_node("evidence_checker",evidence_checker)
graph.add_node("retry_search",retry_search)
graph.add_node("generate_answer",generate_answer)

graph.add_edge(START,"analyze_query")

graph.add_conditional_edges(
    "analyze_query",route_after_analysis,
    {"sql":"sql_retrieval","vector":"vector_retrieval","both":"sql_retrieval"}
)

# If route == BOTH, go to vector retrieval.
# Otherwise go to fusion.
def after_sql(state: RAGState):
    if state["route"] == "BOTH":
        return "vector"
    return "fusion"

graph.add_conditional_edges(
    "sql_retrieval",after_sql,
    {"vector":"vector_retrieval", "fusion":"fusion"}
)
# Vector → Fusion
graph.add_edge("vector_retrieval","fusion")
# Fusion → Rerank
graph.add_edge("fusion","rerank")
graph.add_edge("rerank","evidence_checker")
graph.add_conditional_edges(
    "evidence_checker",after_evidence_check,
    {"answer": "generate_answer", "retry":"retry_search"}
)
# We retry semantic retrieval with a larger K.
graph.add_edge("retry_search","vector_retrieval")
graph.add_edge("generate_answer",END)
app = graph.compile()

question = """
Find customers who had problems with delayed delivery
and tell me their order values.
"""

result = app.invoke({"question":question, "attempts":0})

print("#                    FINAL RESULT                           #")

print(result.get("answer", "No answer generated."))