1.User Query (Natural Language)  
Example: “Show me last week’s chicken sales in Coimbatore.”

2.LLM 1 (Ollama NL → SQL)

Prompted with schema + examples.

Converts NL → SQL query.
3.Database Execution (Oracle DB)

Use oracledb Python driver (thin mode for production).

Query runs securely, result returned in tabular format.

4.RAG Layer (Hybrid Retrieval)

If SQL alone doesn’t answer (e.g., policy docs, FAQs), fallback to FAISS/Chroma vector store.

Retrieve relevant text chunks.

Combine DB results + RAG context.

LLM 2 (Ollama SQL → NL)

Takes structured result + RAG context.

Generates human‑readable answer.

Example: “Chicken sales in Coimbatore last week totaled ₹8,75,000.”

5.Final Output

User sees a natural sentence, not SQL or raw tables.
------------------------------------------------------------------------------------------------------------------
---config.py
DB_CONFIG = {
    "user": "apps",
    "password": "sugapps",
    "dsn": "10.4.1.43:1601/TESTPRE",  # e.g. "localhost:1521/ORCLPDB1"
}

OLLAMA_BASE_URL = "http://localhost:11434"
NL_TO_SQL_MODEL = "llama3"   # model for NL → SQL
SQL_TO_NL_MODEL = "llama3"   # model for SQL result → NL

# DB schema description injected into NL→SQL prompt
DB_SCHEMA = """
Tables:
- SALES(sale_id, product_name, category, city, sale_date, quantity, amount)
"""

# Few-shot examples for NL→SQL
FEW_SHOT_EXAMPLES = """
"""

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model

-------------------------------------------------------------------------
---db.py
import oracledb
from config import DB_CONFIG

oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_23_6\instantclient_23_0")

def run_query(sql: str) -> list[dict]:
    """Execute SQL on Oracle DB and return rows as list of dicts."""
    with oracledb.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

--------------------------------------------------------------------------
-------llm.py
import re
import requests
from config import OLLAMA_BASE_URL, NL_TO_SQL_MODEL, SQL_TO_NL_MODEL, DB_SCHEMA, FEW_SHOT_EXAMPLES

def _chat(model: str, prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

def nl_to_sql(user_query: str) -> str:
    prompt = f"""You are an Oracle SQL expert. Given the schema and examples, output ONLY the SQL query with no explanation.

Schema:
{DB_SCHEMA}

Examples:
{FEW_SHOT_EXAMPLES}

Q: {user_query}
SQL:"""
    raw = _chat(NL_TO_SQL_MODEL, prompt)
    # Extract first SQL statement
    match = re.search(r"(SELECT[\s\S]+?);?$", raw, re.IGNORECASE)
    return match.group(1).strip() if match else raw

def result_to_nl(user_query: str, db_rows: list[dict], rag_context: list[str]) -> str:
    context_block = "\n".join(rag_context) if rag_context else "None"
    rows_block = "\n".join(str(r) for r in db_rows) if db_rows else "No data found."
    prompt = f"""You are a helpful business assistant. Answer the user's question in one or two natural sentences using the data and context below.

User question: {user_query}

Database result:
{rows_block}

Additional context:
{context_block}

Answer:"""
    return _chat(SQL_TO_NL_MODEL, prompt)

------------------------------------------------------------------------------------
---main.py
from pipeline import run_pipeline
from rag import build_index

# Optional: load your policy/FAQ docs into RAG index once
# docs = open("docs/policy.txt").read().split("\n\n")
# build_index(docs)

if __name__ == "__main__":
    print("Sales Bot ready. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer = run_pipeline(query)
        print(f"Bot: {answer}\n")

--------------------------------------------------------------------
-------pipeline.py
import re
from llm import nl_to_sql, result_to_nl
from db import run_query
from rag import retrieve

def _clean_sql(sql: str) -> str:
    """Strip markdown fences and trailing semicolons LLMs sometimes add."""
    sql = re.sub(r"```[\w]*\n?", "", sql).strip()
    return sql.rstrip(";")

def run_pipeline(user_query: str) -> str:
    # Step 1: NL → SQL
    sql = _clean_sql(nl_to_sql(user_query))
    print(f"[SQL] {sql}\n")

    # Step 2: Execute on Oracle DB
    try:
        rows = run_query(sql)
    except Exception as e:
        rows = []
        print(f"[DB Error] {e}")

    # Step 3: RAG fallback — only when DB returns nothing
    rag_chunks = retrieve(user_query) if not rows else []

    # Step 4: SQL result + RAG context → NL answer
    return result_to_nl(user_query, rows, rag_chunks)

---------------------------------------------------------------------------
--rag.py
import os
import pickle
import numpy as np
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _embed(texts: list[str]) -> np.ndarray:
    return _get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def build_index(docs: list[str]):
    """Build and persist FAISS index from a list of text chunks."""
    import faiss
    vectors = _embed(docs)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, f"{FAISS_INDEX_PATH}/index.bin")
    with open(f"{FAISS_INDEX_PATH}/docs.pkl", "wb") as f:
        pickle.dump(docs, f)

def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Return top_k relevant chunks for a query. Returns [] if no index exists."""
    import faiss
    index_file = f"{FAISS_INDEX_PATH}/index.bin"
    if not os.path.exists(index_file):
        return []
    index = faiss.read_index(index_file)
    with open(f"{FAISS_INDEX_PATH}/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    vec = _embed([query])
    _, indices = index.search(vec, top_k)
    return [docs[i] for i in indices[0] if i < len(docs)]

-------------------------------------------------------------------
----requirements.txt
oracledb>=2.0.0
requests>=2.31.0
faiss-cpu>=1.7.4
sentence-transformers>=2.7.0
numpy>=1.26.0

------------------------------------------------------------------------
---------test_pipeline.py
"""
Quick end-to-end test:
  1. DB connection
  2. NL → SQL via Ollama
  3. Full pipeline answer
"""
from db import run_query
from llm import nl_to_sql
from pipeline import run_pipeline

def test_db():
    print("=== DB Connection ===")
    rows = run_query("SELECT 1 AS ping FROM DUAL")
    assert rows == [{"PING": 1}], f"Unexpected: {rows}"
    print("OK — Oracle connected\n")

def test_nl_to_sql():
    print("=== NL → SQL ===")
    sql = nl_to_sql("Show total sales this month")
    print(f"Generated SQL:\n{sql}\n")
    assert "SELECT" in sql.upper(), "LLM did not return a SELECT statement"
    print("OK\n")

def test_pipeline():
    print("=== Full Pipeline ===")
    answer = run_pipeline("What are the total sales this month?")
    print(f"Answer: {answer}\n")
    assert len(answer) > 5, "Answer too short"
    print("OK\n")

if __name__ == "__main__":
    test_db()
    test_nl_to_sql()
    test_pipeline()
    print("All tests passed.")
