"""
Text Chunking, Embedding, and Vector Store Indexing Script

This script performs a complete pipeline for building a vector store from complaint data:

1. STRATIFIED SAMPLING: Creates a 10,000-15,000 complaint subset with proportional 
   representation across all product categories (Credit card, Personal loan, 
   Savings account, Money transfers) before chunking. This ensures the index 
   reflects a balanced dataset representative of all products.

2. TEXT CHUNKING: Splits complaint narratives into smaller segments using 
   LangChain's RecursiveCharacterTextSplitter.

3. EMBEDDING GENERATION: Generates dense vector embeddings using 
   sentence-transformers/all-MiniLM-L6-v2.

4. VECTOR STORE CREATION: Builds and persists a FAISS index with metadata storage
   for efficient similarity search.

The stratified sampling step is critical: it ensures the vector store maintains
balanced representation across product categories, which is essential for unbiased
retrieval during RAG queries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    # Try newer LangChain structure
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fall back to older structure
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

# Project imports
from src.config import (
    FILTERED_DATA_PATH,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_SAMPLE_SIZE,
    MAX_SAMPLE_SIZE,
    setup_hf_cache
)

# Set up HuggingFace cache
setup_hf_cache()


class ComplaintChunker:
    """Handles text chunking for complaint narratives."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each text chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraphs, sentences, words
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or pd.isna(text):
            return []
        return self.text_splitter.split_text(str(text))


class ComplaintEmbedder:
    """Handles embedding generation for complaint chunks."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
        print("✓ Embedding model loaded!")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)


def stratified_sample(
    df: pd.DataFrame,
    product_column: str,
    narrative_column: str,
    min_size: int = MIN_SAMPLE_SIZE,
    max_size: int = MAX_SAMPLE_SIZE,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample of complaints ensuring proportional representation
    across all product categories.
    
    This function creates a 10,000-15,000 complaint subset (configurable via 
    min_size and max_size) with proportional representation across all product 
    categories (typically Credit card, Personal loan, Savings account, Money 
    transfers). The sampling preserves the relative distribution of each product 
    category in the original dataset, ensuring the resulting index reflects a 
    balanced and representative dataset.
    
    This stratified sampling step MUST be performed BEFORE chunking to ensure
    the vector store maintains balanced representation across products.
    
    Args:
        df: DataFrame containing complaints
        product_column: Name of the column containing product categories
        narrative_column: Name of the column containing complaint narratives
        min_size: Minimum sample size (default: 10,000)
        max_size: Maximum sample size (default: 15,000)
        random_state: Random seed for reproducibility
        
    Returns:
        Stratified sample DataFrame with proportional representation across
        product categories
    """
    print("=" * 80)
    print("STRATIFIED SAMPLING")
    print("=" * 80)
    
    # Remove rows with missing products or empty narratives
    df_clean = df.dropna(subset=[product_column, narrative_column]).copy()
    df_clean = df_clean[df_clean[narrative_column].str.strip() != ''].copy()
    
    print(f"\nTotal records in cleaned dataset: {len(df_clean):,}")
    
    # Get product distribution
    product_counts = df_clean[product_column].value_counts()
    n_products = len(product_counts)
    
    print(f"\nNumber of product categories: {n_products}")
    print("\nProduct distribution in full dataset:")
    for product, count in product_counts.items():
        pct = (count / len(df_clean)) * 100
        print(f"  - {product}: {count:,} ({pct:.2f}%)")
    
    # Determine target sample size
    target_size = min(max_size, max(min_size, len(df_clean)))
    
    # Calculate proportional sample sizes per product
    # Ensure each product has at least 1 sample
    product_proportions = product_counts / len(df_clean)
    product_sample_sizes = (product_proportions * target_size).astype(int)
    
    # Adjust to ensure we hit target size and each product has at least 1 sample
    total_allocated = product_sample_sizes.sum()
    if total_allocated < target_size:
        # Distribute remaining samples proportionally
        remaining = target_size - total_allocated
        additional = (product_proportions * remaining).astype(int)
        product_sample_sizes += additional
        total_allocated = product_sample_sizes.sum()
        
        # If still not enough, add to largest categories
        if total_allocated < target_size:
            diff = target_size - total_allocated
            largest_indices = product_proportions.nlargest(diff).index
            for idx in largest_indices:
                product_sample_sizes[idx] += 1
    
    # Ensure each product has at least 1 sample
    product_sample_sizes = product_sample_sizes.clip(lower=1)
    
    # Cap sample sizes to available data
    for product in product_sample_sizes.index:
        available = product_counts[product]
        product_sample_sizes[product] = min(product_sample_sizes[product], available)
    
    # Perform stratified sampling
    sampled_dfs = []
    np.random.seed(random_state)
    
    print(f"\nTarget sample size: {target_size:,}")
    print("\nSampling strategy per product:")
    for product, sample_size in product_sample_sizes.items():
        product_df = df_clean[df_clean[product_column] == product]
        if len(product_df) >= sample_size:
            sampled = product_df.sample(n=sample_size, random_state=random_state)
        else:
            sampled = product_df  # Take all available
        sampled_dfs.append(sampled)
        print(f"  - {product}: {len(sampled):,} samples (from {len(product_df):,} available)")
    
    # Combine all samples
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final sample
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nFinal sample size: {len(df_sampled):,}")
    print("\nProduct distribution in sampled dataset:")
    sampled_product_counts = df_sampled[product_column].value_counts()
    for product, count in sampled_product_counts.items():
        pct = (count / len(df_sampled)) * 100
        print(f"  - {product}: {count:,} ({pct:.2f}%)")
    
    return df_sampled


def create_vector_store(
    df: pd.DataFrame,
    narrative_column: str,
    product_column: str,
    id_column: str = None,
    output_dir: Path = VECTOR_STORE_DIR
) -> FAISS:
    """
    Create a FAISS vector store from complaint narratives.
    
    Args:
        df: DataFrame containing complaints
        narrative_column: Name of the column containing complaint narratives
        product_column: Name of the column containing product categories
        id_column: Name of the column containing complaint IDs (if None, uses index)
        output_dir: Directory to save the vector store
        
    Returns:
        FAISS vector store
    """
    print("\n" + "=" * 80)
    print("TEXT CHUNKING")
    print("=" * 80)
    
    # Initialize chunker
    chunker = ComplaintChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"\nChunking parameters:")
    print(f"  - Chunk size: {CHUNK_SIZE} characters")
    print(f"  - Chunk overlap: {CHUNK_OVERLAP} characters")
    
    # Chunk all narratives
    all_chunks = []
    chunk_metadata = []
    
    print("\nChunking narratives...")
    for idx, row in df.iterrows():
        narrative = row[narrative_column]
        complaint_id = row[id_column] if id_column and id_column in row else str(idx)
        product = row[product_column]
        
        chunks = chunker.chunk_text(narrative)
        
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "complaint_id": str(complaint_id),
                "product": str(product),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "source_index": idx
            })
    
    print(f"✓ Created {len(all_chunks):,} chunks from {len(df):,} complaints")
    print(f"  Average chunks per complaint: {len(all_chunks) / len(df):.2f}")
    
    # Show chunk size distribution
    chunk_lengths = [len(chunk) for chunk in all_chunks]
    print(f"\nChunk length statistics:")
    print(f"  - Mean: {np.mean(chunk_lengths):.1f} characters")
    print(f"  - Median: {np.median(chunk_lengths):.1f} characters")
    print(f"  - Min: {np.min(chunk_lengths)} characters")
    print(f"  - Max: {np.max(chunk_lengths)} characters")
    
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION")
    print("=" * 80)
    
    # Initialize embedder
    embedder = ComplaintEmbedder(model_name=EMBEDDING_MODEL_NAME)
    embedding_dim = embedder.get_embedding_dimension()
    print(f"\nEmbedding dimension: {embedding_dim}")
    
    print("\nGenerating embeddings...")
    print(f"  This may take a few minutes for {len(all_chunks):,} chunks...")
    
    # Create vector store with FAISS
    # FAISS will automatically generate embeddings using the provided embeddings model
    vectorstore = FAISS.from_texts(
        texts=all_chunks,
        embedding=embedder.embeddings,
        metadatas=chunk_metadata
    )
    
    print(f"✓ Generated embeddings for {len(all_chunks):,} chunks")
    
    print("\n" + "=" * 80)
    print("SAVING VECTOR STORE")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vector store
    vectorstore_path = output_dir / "faiss_index"
    vectorstore.save_local(str(vectorstore_path))
    print(f"✓ Vector store saved to: {vectorstore_path}")
    
    # Save metadata about the vector store
    metadata_info = {
        "created_at": datetime.now().isoformat(),
        "total_chunks": len(all_chunks),
        "total_complaints": len(df),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_dim,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "product_distribution": df[product_column].value_counts().to_dict()
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_info, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return vectorstore


def main():
    """
    Main function to run the complete pipeline:
    
    1. Load filtered complaint dataset
    2. Perform stratified sampling (10k-15k records with proportional 
       representation across product categories)
    3. Chunk the sampled narratives
    4. Generate embeddings
    5. Create and save FAISS vector store
    
    The stratified sampling step ensures the index reflects a balanced dataset
    with proportional representation across all product categories.
    """
    print("=" * 80)
    print("COMPLAINT ANALYSIS: CHUNKING, EMBEDDING, AND INDEXING")
    print("=" * 80)
    
    # Load cleaned dataset
    print(f"\nLoading cleaned dataset from: {FILTERED_DATA_PATH}")
    if not FILTERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {FILTERED_DATA_PATH}. "
            "Please run the EDA and preprocessing notebook first."
        )
    
    df = pd.read_csv(FILTERED_DATA_PATH, low_memory=False)
    print(f"✓ Loaded {len(df):,} records")
    
    # Identify columns
    # Try to find the narrative and product columns
    narrative_column = None
    product_column = None
    id_column = None
    
    # Common column name variations
    narrative_candidates = [
        'Consumer complaint narrative',
        'narrative_cleaned',
        'narrative',
        'complaint_narrative',
        'text'
    ]
    product_candidates = ['Product', 'product', 'product_category', 'category']
    id_candidates = ['Complaint ID', 'complaint_id', 'id', 'ID']
    
    for col in df.columns:
        if col in narrative_candidates:
            narrative_column = col
        if col in product_candidates:
            product_column = col
        if col in id_candidates:
            id_column = col
    
    if not narrative_column:
        raise ValueError("Could not find narrative column. Available columns: " + ", ".join(df.columns))
    if not product_column:
        raise ValueError("Could not find product column. Available columns: " + ", ".join(df.columns))
    
    print(f"\nUsing columns:")
    print(f"  - Narrative: {narrative_column}")
    print(f"  - Product: {product_column}")
    if id_column:
        print(f"  - ID: {id_column}")
    
    # Perform stratified sampling
    df_sampled = stratified_sample(
        df=df,
        product_column=product_column,
        narrative_column=narrative_column,
        min_size=MIN_SAMPLE_SIZE,
        max_size=MAX_SAMPLE_SIZE
    )
    
    # Create vector store
    vectorstore = create_vector_store(
        df=df_sampled,
        narrative_column=narrative_column,
        product_column=product_column,
        id_column=id_column,
        output_dir=VECTOR_STORE_DIR
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nVector store saved to: {VECTOR_STORE_DIR}")
    print(f"Total chunks indexed: {len(df_sampled):,} complaints → {len(df_sampled) * 2:.0f} chunks (approx.)")
    print("\nYou can now use this vector store for RAG-based question answering.")


if __name__ == "__main__":
    main()

