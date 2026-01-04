# Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) system for analyzing and querying financial service complaints using natural language. This project leverages LangChain, FAISS, and transformer models to enable interactive question-answering over consumer complaint data from the Consumer Financial Protection Bureau (CFPB).

## Features

- **Interactive Chat Interface**: Gradio-based web UI for asking questions about financial complaints
- **RAG Pipeline**: Combines semantic search (FAISS) with language generation (FLAN-T5) for accurate, context-aware responses
- **Stratified Sampling**: Creates a 10,000-15,000 complaint subset with proportional representation across all four product categories (Credit card, Personal loan, Savings account, Money transfers) before chunking, ensuring the index reflects a balanced dataset
- **Text Preprocessing**: Comprehensive cleaning and normalization of complaint narratives
- **Vector Store**: Efficient semantic search using FAISS with sentence transformer embeddings
- **Configurable Settings**: Easy-to-modify configuration for models, chunking, and retrieval parameters

## Technologies Used

- **LangChain**: Framework for building LLM applications and RAG pipelines
- **FAISS**: Facebook AI Similarity Search for efficient vector similarity search
- **Sentence Transformers**: `all-MiniLM-L6-v2` for generating text embeddings
- **FLAN-T5**: Google's FLAN-T5-base model for text generation
- **Gradio**: Web UI framework for interactive chatbot interface
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization (used in notebooks)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd intelligent-compliant-analysis-for-financial-services
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data directory**:
   Create a `data/raw/` directory and place your CFPB complaint dataset as `data/raw/complaints.csv`

## Project Structure

```
intelligent-compliant-analysis-for-financial-services/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.py                       # Alternative app entry point
├── gradio_app.py                # Main Gradio application
├── src/                         # Source code directory
│   ├── __init__.py
│   ├── config.py                # Configuration settings (paths, models, etc.)
│   └── chunk_embed_index.py     # Text chunking, embedding, and vector store creation
├── notebook/                    # Jupyter notebooks
│   ├── eda_and_preprocessing.ipynb  # Exploratory data analysis and data cleaning
│   ├── prompt_engineering.ipynb     # Prompt engineering experiments
│   └── demo.ipynb                   # Demonstration notebook
├── data/                        # Data directory (created during setup)
│   ├── raw/                     # Raw data files
│   └── filtered_complaints.csv  # Cleaned and filtered dataset (generated)
├── vector_store/                # FAISS vector store (generated)
│   ├── faiss_index/             # FAISS index files
│   └── metadata.json            # Vector store metadata
├── models/                      # Model cache directory (created automatically)
│   └── hf/                      # HuggingFace model cache
└── tests/                       # Test directory
    └── __init__.py
```

## Usage

### Step 1: Data Preprocessing

Run the EDA and preprocessing notebook to clean and filter the complaint data:

```bash
jupyter notebook notebook/eda_and_preprocessing.ipynb
```

This notebook will:
- Load and explore the raw complaint dataset
- Filter complaints for specific product categories (Credit card, Personal loan, Savings account, Money transfers)
- Clean and normalize complaint narratives
- Save the processed data to `data/filtered_complaints.csv`

### Step 2: Create Vector Store

Run the chunking, embedding, and indexing script to create the FAISS vector store:

```bash
python src/chunk_embed_index.py
```

This script performs the following pipeline **in order**:

1. **Stratified Sampling (REQUIRED STEP)**: Creates a 10,000-15,000 complaint subset 
   with **proportional representation** across all four product categories 
   (Credit card, Personal loan, Savings account, Money transfers). This sampling 
   step ensures the vector store reflects a balanced dataset where each product 
   category is represented proportionally to its frequency in the original dataset.
   The stratified sample is created **before chunking** to maintain balanced 
   representation in the final index.

2. **Text Chunking**: Splits the sampled complaint narratives into smaller segments 
   using LangChain's RecursiveCharacterTextSplitter (512 characters per chunk with 
   50 character overlap).

3. **Embedding Generation**: Generates dense vector embeddings for each chunk using 
   the sentence-transformers/all-MiniLM-L6-v2 model (384-dimensional embeddings).

4. **Vector Store Creation**: Builds and saves a FAISS index with metadata storage 
   for efficient similarity search.

**Note**: This step may take several minutes depending on your dataset size and hardware. 
The stratified sampling step is critical for ensuring the vector store maintains 
balanced representation across product categories.

### Step 3: Launch the Chatbot

Start the Gradio web interface:

```bash
python gradio_app.py
```

The application will:
- Load the RAG pipeline (vector store + embedding model + LLM)
- Start a local web server (typically at `http://127.0.0.1:7860`)
- Display configuration information in the terminal

Open your browser and navigate to the displayed URL to start asking questions about the complaint data.

### Example Questions

- "What are the most common issues with credit cards?"
- "Tell me about problems with money transfers"
- "What complaints exist about personal loans?"
- "Describe issues with savings accounts"

## Configuration

Key configuration settings can be modified in `src/config.py`:

```python
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
```

## Notebooks

### `eda_and_preprocessing.ipynb`
- Exploratory data analysis of the CFPB complaint dataset
- Data quality assessment
- Text cleaning and normalization
- Product filtering
- Dataset summary and statistics

### `prompt_engineering.ipynb`
- Experiments with different prompt templates
- Text preprocessing demonstrations
- NLP techniques for financial text

### `demo.ipynb`
- Demonstration of text processing techniques
- NLTK usage examples

## Data Requirements

The project expects a CSV file with the following columns (CFPB complaint dataset format):
- `Product`: Product category (e.g., "Credit card", "Personal loan")
- `Consumer complaint narrative`: The complaint text
- `Complaint ID`: Unique identifier for each complaint

The script will attempt to match common column name variations automatically.

## Model Information

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional embeddings
  - Fast and efficient for semantic search
  - Trained on diverse text data

- **LLM Model**: `google/flan-t5-base`
  - Text generation model
  - Optimized for instruction following
  - Lightweight and suitable for local deployment

Both models are automatically downloaded from HuggingFace on first use and cached in `models/hf/`.

## Troubleshooting

### Vector Store Not Found
If you encounter an error about the vector store not being found, ensure you've run:
1. The preprocessing notebook (`eda_and_preprocessing.ipynb`)
2. The indexing script (`python src/chunk_embed_index.py`)

### Memory Issues
If you encounter memory issues:
- Reduce `MAX_SAMPLE_SIZE` in `config.py`
- Use a smaller chunk size
- Close other applications to free up RAM

### Model Download Issues
If models fail to download:
- Check your internet connection
- Verify HuggingFace access (some models may require authentication)
- Check available disk space in `models/hf/`

## Future Enhancements

- Support for additional product categories
- Advanced filtering and query capabilities
- Integration with other LLM models (GPT, Claude, etc.)
- Batch processing capabilities
- Enhanced visualization and analytics
- Export functionality for query results

## License

[Add your license information here]

## Contributing

[Add contributing guidelines if applicable]

## Contact

[Add contact information if applicable]
