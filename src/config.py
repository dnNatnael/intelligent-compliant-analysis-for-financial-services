"""
Configuration file for the intelligent complaint analysis project.
Contains paths, model names, and other constants.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FILTERED_DATA_PATH = DATA_DIR / "filtered_complaints.csv"

# Vector store path
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Model names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

# RAG settings
RETRIEVAL_K = 3  # Number of documents to retrieve

# Chunking settings
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Sampling settings
MIN_SAMPLE_SIZE = 10000
MAX_SAMPLE_SIZE = 15000


def setup_hf_cache():
    """
    Set up HuggingFace cache directory within the project.
    This ensures models are downloaded to a predictable location.
    """
    import os
    hf_cache_dir = PROJECT_ROOT / "models" / "hf"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir)

