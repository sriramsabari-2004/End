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
