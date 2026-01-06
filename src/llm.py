"""
LLM utilities: prompt template, context formatting, and generator loader.
"""
from __future__ import annotations

from typing import List, Optional

try:
    # Newer LangChain versions
    from langchain.schema import Document
except Exception:  # pragma: no cover - fallback for older versions
    from langchain.docstore.document import Document

from transformers import pipeline

from src.config import LLM_MODEL_NAME, setup_hf_cache

# Prompt template guiding the LLM to stay grounded in retrieved context.
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust.
Use ONLY the information in the provided context to answer the user's question.
If the context does not contain the answer, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:"""


def format_docs_for_context(docs: List[Document], max_chars: Optional[int] = None) -> str:
    """
    Convert retrieved documents into a readable context string for the prompt.

    Args:
        docs: Retrieved LangChain Document objects.
        max_chars: Optional character cap to avoid overly long prompts.

    Returns:
        Concatenated string of source-tagged snippets.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source_bits = []
        for key in ("complaint_id", "product", "chunk_index"):
            if key in meta:
                source_bits.append(f"{key}={meta[key]}")
        source_tag = f"[Source {i}"
        if source_bits:
            source_tag += " | " + ", ".join(source_bits)
        source_tag += "]"
        snippet = doc.page_content.replace("\n", " ").strip()
        formatted_chunks.append(f"{source_tag} {snippet}")

    context = "\n".join(formatted_chunks) if formatted_chunks else "No relevant context retrieved."
    if max_chars and len(context) > max_chars:
        context = context[:max_chars] + "..."
    return context


def build_prompt(question: str, context: str) -> str:
    """Fill the RAG prompt template."""
    safe_context = context or "No relevant context provided."
    return PROMPT_TEMPLATE.format(context=safe_context, question=question.strip())


def load_text_generator(model_name: str = LLM_MODEL_NAME, device: int = -1):
    """
    Load a lightweight text generator pipeline (defaults to FLAN-T5).

    Args:
        model_name: Hugging Face model id.
        device: Device index (-1 for CPU).

    Returns:
        Hugging Face pipeline for text2text generation.
    """
    setup_hf_cache()
    return pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )

