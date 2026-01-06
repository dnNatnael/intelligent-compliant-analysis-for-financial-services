"""
Qualitative evaluation runner for the RAG pipeline.

Loads the pre-built vector store, asks a set of representative questions,
and writes a Markdown table with answers and retrieved sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from src.config import PROJECT_ROOT
from src.llm import format_docs_for_context
from src.rag_pipeline import RAGPipeline

# Representative questions to probe coverage across products/issues
DEFAULT_QUESTIONS: List[str] = [
    "What are the most common issues with credit cards?",
    "Summarize frequent complaints about personal loans.",
    "How do customers describe problems with money transfers?",
    "What savings account issues do customers report?",
    "Are there trends about fraudulent transactions?",
    "Do customers mention poor customer service or response delays?",
    "How often are billing disputes raised in credit products?",
    "What resolutions are customers seeking most often?",
    "Are there concerns about interest rates or fees?",
    "Do any complaints mention mobile app or website problems?",
]


def _summarize_sources(docs, limit: int = 2) -> str:
    """Return a compact preview of the first N sources."""
    previews = []
    for doc in docs[:limit]:
        meta = doc.metadata or {}
        tag = meta.get("complaint_id") or meta.get("source_index") or "N/A"
        snippet = doc.page_content.strip().replace("\n", " ")
        previews.append(f"{tag}: {snippet[:220]}{'...' if len(snippet) > 220 else ''}")
    return "<br>".join(previews) if previews else "No sources retrieved"


def run_evaluation(
    questions: Sequence[str] = DEFAULT_QUESTIONS,
    output_path: Path = PROJECT_ROOT / "notebook" / "rag_evaluation.md",
    retrieval_k: int = 5,
) -> List[dict]:
    """
    Run qualitative evaluation and write a Markdown table.

    Returns:
        List of result dicts with question, answer, sources, and placeholders
        for quality score/comments to be filled manually.
    """
    try:
        rag = RAGPipeline(retrieval_k=retrieval_k)
    except FileNotFoundError as exc:
        output_path.write_text(
            "# RAG Qualitative Evaluation\n\n"
            "Vector store not found. Please place the pre-built FAISS index at "
            "`vector_store/faiss_index` and re-run this script.\n\n"
            f"Details: {exc}\n"
        )
        raise

    results = []
    rows = [
        "# RAG Qualitative Evaluation",
        "",
        "| Question | Generated Answer | Retrieved Sources (2) | Quality Score (1-5) | Comments |",
        "| --- | --- | --- | --- | --- |",
    ]

    for question in questions:
        answer_bundle = rag.answer(question)
        answer_text = answer_bundle["answer"]
        docs = answer_bundle["sources"]
        source_preview = _summarize_sources(docs, limit=2)

        row = {
            "question": question,
            "answer": answer_text,
            "retrieved_sources": source_preview,
            "quality_score": "TBD",
            "comments": "Add manual assessment.",
        }
        results.append(row)

        def _escape(value: str) -> str:
            return value.replace("|", "\\|").replace("\n", "<br>")

        rows.append(
            f"| {_escape(question)} | {_escape(answer_text)} | "
            f"{_escape(source_preview)} | {row['quality_score']} | {row['comments']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows))
    return results


if __name__ == "__main__":
    run_evaluation()

