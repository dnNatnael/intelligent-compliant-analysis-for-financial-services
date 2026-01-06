# RAG Qualitative Evaluation

The evaluation runner `src/evaluate_rag.py` populates this table after loading the
pre-built FAISS vector store (`vector_store/faiss_index`). If the vector store is
missing, run the indexing pipeline or place the provided pre-built store at that
location and re-run the script.

| Question | Generated Answer | Retrieved Sources (2) | Quality Score (1-5) | Comments |
| --- | --- | --- | --- | --- |
| Pending | Run `python -m src.evaluate_rag` after the vector store is available | Pending | TBD | Pending manual review |

