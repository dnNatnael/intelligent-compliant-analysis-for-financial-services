"""
Core Retrieval-Augmented Generation (RAG) pipeline.

Loads the pre-built FAISS vector store, retrieves top-k chunks using the
all-MiniLM-L6-v2 embedding model, builds a grounded prompt, and generates an
answer with FLAN-T5 (or any compatible HF text2text model).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    # Prefer the new LangChain community namespace
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older versions
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain.schema import Document
except Exception:  # pragma: no cover
    from langchain.docstore.document import Document

from src import config
from src.config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    RETRIEVAL_K,
    VECTOR_STORE_DIR,
    setup_hf_cache,
)
from src.llm import build_prompt, format_docs_for_context, load_text_generator


class RAGPipeline:
    """End-to-end retrieval + generation pipeline."""

    def __init__(
        self,
        vector_store_dir: Path | str = VECTOR_STORE_DIR,
        retrieval_k: int = RETRIEVAL_K,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
        llm_device: int = -1,
        max_context_chars: int = 2000,
    ):
        """
        Initialize RAG pipeline components.

        Args:
            vector_store_dir: Directory containing the saved FAISS index.
            retrieval_k: Number of chunks to retrieve.
            embedding_model_name: Sentence transformer model to embed queries.
            llm_model_name: Hugging Face text2text model for generation.
            llm_device: Device index for HF pipeline (-1 = CPU).
            max_context_chars: Safety cap for prompt context length.
        """
        setup_hf_cache()
        self.vector_store_dir = Path(vector_store_dir)
        self.retrieval_k = retrieval_k
        self.max_context_chars = max_context_chars

        # Embeddings for both retrieval and vector store loading
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},  # set to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = self._load_vector_store(self.vector_store_dir)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k},
        )

        # Text generator (FLAN-T5 by default)
        self.generator = load_text_generator(model_name=llm_model_name, device=llm_device)

    def _load_vector_store(self, vector_store_dir: Path) -> FAISS:
        """Load FAISS index from disk."""
        index_path = vector_store_dir / "faiss_index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Ensure the pre-built vector store is placed at this location."
            )
        return FAISS.load_local(
            str(index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, question: str) -> List[Document]:
        """Return top-k relevant chunks for the question."""
        return self.retriever.get_relevant_documents(question)

    def generate(self, question: str, context: str, **gen_kwargs) -> str:
        """Generate an answer given the question and retrieved context."""
        prompt = build_prompt(question, context)
        outputs = self.generator(
            prompt,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            num_beams=2,
            **gen_kwargs,
        )
        return outputs[0]["generated_text"].strip()

    def answer(self, question: str):
        """
        Full RAG call: retrieve, format context, and generate an answer.

        Returns:
            Dict with 'answer' text and 'sources' (retrieved Documents).
        """
        sources = self.retrieve(question)
        context = format_docs_for_context(sources, max_chars=self.max_context_chars)
        answer_text = self.generate(question, context)
        return {"answer": answer_text, "sources": sources}


def load_default_pipeline(retrieval_k: Optional[int] = None) -> RAGPipeline:
    """
    Convenience loader using config defaults.

    Args:
        retrieval_k: Optional override for number of retrieved chunks.

    Returns:
        Initialized RAGPipeline instance.
    """
    return RAGPipeline(
        vector_store_dir=VECTOR_STORE_DIR,
        retrieval_k=retrieval_k or config.RETRIEVAL_K,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        llm_model_name=config.LLM_MODEL_NAME,
    )

