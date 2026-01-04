# =============================================================================
# gradio_app.py - Gradio Chatbot UI for RAG Pipeline
# =============================================================================
"""
A simple Gradio chatbot for interactive RAG question answering.

Run with: python gradio_app.py

Features:
- Chat interface for asking questions
- Reuses existing RAGPipeline (FAISS + FLAN-T5)
"""

# sys is used to modify Python's import path at runtime
import sys
# Path helps us build file paths in a cross-platform way
from pathlib import Path

# Add project root to the Python import path so we can import `src.*`
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up HuggingFace cache BEFORE importing other modules
# This ensures model downloads go into our project folder (models/hf)
from src.config import setup_hf_cache
setup_hf_cache()

# Gradio provides the web UI
import gradio as gr
# RAGPipeline is our existing retrieval + generation logic
from src.rag_pipeline import RAGPipeline
# config holds constants like model names and vector store path
from src import config


# Safety cap: limit how much context we feed to the small FLAN-T5 model
# (FLAN-T5 has a short max input length; long prompts can degrade answers)
MAX_CONTEXT_CHARS = 1500


# =============================================================================
# GLOBAL: Load RAG pipeline once
# =============================================================================

rag_pipeline = None  # Will be loaded on first use


def load_rag_pipeline():
    """
    Load the RAG pipeline (cached globally).
    """
    # Use the global variable so we only load models once (fast after first call)
    global rag_pipeline
    # Create the pipeline the first time someone asks a question
    if rag_pipeline is None:
        print("Loading RAG pipeline... (this may take a minute on first run)")
        # This loads:
        # - FAISS vector store from disk
        # - embedding model
        # - FLAN-T5 model
        rag_pipeline = RAGPipeline(retrieval_k=config.RETRIEVAL_K)
        print("✓ RAG pipeline loaded!")
    # Return the cached pipeline
    return rag_pipeline


# =============================================================================
# CHAT FUNCTION
# =============================================================================

def chat(message: str, history: list, k: int):
    """
    Process a user message and return the assistant response.
    
    Args:
        message: The user's question
        history: List of (user_msg, assistant_msg) tuples (chat history)
        k: Number of sources to retrieve
        
    Returns:
        Answer text
    """
    # 1) Basic input validation
    if not message.strip():
        return "Please enter a question."
    
    # 2) Load the pipeline (cached so we don't reload models every message)
    try:
        rag = load_rag_pipeline()
    except FileNotFoundError as e:
        # This usually means the FAISS index folder doesn't exist yet
        error_msg = (
            "**Vector store not found!**\n\n"
            "Please run the notebooks first:\n"
            "1. `00_eda_and_cleaning.ipynb` - Preprocess the data\n"
            "2. `01_chunk_embed_index.ipynb` - Create the FAISS index\n\n"
            f"Error: {e}"
        )
        return error_msg
    except Exception as e:
        # Catch-all for unexpected loading errors
        return f"Error loading RAG pipeline: {e}"
    
    # 3) Update retriever settings (k = number of retrieved chunks)
    # In this simplified UI we pass a fixed k, but the function still supports changing it.
    if rag.retrieval_k != k:
        rag.retrieval_k = k
        # Re-create the retriever with the new k
        rag.retriever = rag.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    # 4) Run the RAG steps:
    # - retrieve relevant documents
    # - format them into a context string
    # - truncate context to reduce prompt length
    # - generate final answer
    try:
        # Retrieve top-k relevant chunks from FAISS
        sources = rag.retrieve(message)

        # Format the retrieved chunks into a single context string for the prompt
        from src.llm import format_docs_for_context

        formatted_context = format_docs_for_context(sources)
        # Truncate by characters to reduce risk of exceeding model max length
        if len(formatted_context) > MAX_CONTEXT_CHARS:
            formatted_context = formatted_context[:MAX_CONTEXT_CHARS] + "..."

        # Generate an answer using the LLM
        answer = rag.generate(message, formatted_context)

        # Minimal response object (kept from earlier versions; not strictly required now)
        class _Resp:
            def __init__(self, answer, sources):
                self.answer = answer
                self.sources = sources

        response = _Resp(answer=answer, sources=sources)
    except Exception as e:
        return f"Error generating answer: {e}"
    
    return response.answer


# =============================================================================
# BUILD GRADIO UI
# =============================================================================

def build_app():
    """
    Build and return the Gradio Blocks app.
    """
    # Blocks is Gradio's layout container
    with gr.Blocks(title="Support Ticket RAG Chatbot") as app:
        # Simple title
        gr.Markdown("# Support Ticket RAG Chatbot")

        # Chatbot displays the conversation
        chatbot = gr.Chatbot(label="Chat", height=500)

        # Row: user input + send button
        with gr.Row():
            # Textbox for the user's question
            msg_input = gr.Textbox(
                placeholder="Ask a question about the support tickets...",
                show_label=False,
                scale=4,
            )
            # Button to submit the question
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # Button to clear the chat
        clear_btn = gr.Button("Clear")
        
        # Footer
        gr.Markdown(
            """
            ---
            <center>
            Built with LangChain, FAISS, and Gradio | RAG Pipeline Demo
            </center>
            """,
        )
        
        # =================================================================
        # EVENT HANDLERS
        # =================================================================
        
        def respond(message, chat_history):
            """Handle user message and update chat."""
            # Call our RAG chat function to get an answer
            answer = chat(message, chat_history, k=config.RETRIEVAL_K)
            if chat_history is None:
                # If it's the first message, start an empty history list
                chat_history = []
            # Gradio Chatbot expects ChatMessage objects
            chat_history = chat_history + [
                gr.ChatMessage(role="user", content=message),
                gr.ChatMessage(role="assistant", content=answer),
            ]
            # Return empty textbox + updated chat history
            return "", chat_history

        def clear_chat():
            """Clear chat history."""
            # Returning an empty list resets the chatbot
            return []

        # Wire up events
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot],
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Print helpful info when starting the app
    print("=" * 60)
    print("🎫 Starting Support Ticket RAG Chatbot (Gradio)")
    print("=" * 60)
    # Show key configuration so beginners know what is being used
    print(f"Vector store: {config.VECTOR_STORE_DIR}")
    print(f"Embedding model: {config.EMBEDDING_MODEL_NAME}")
    print(f"LLM model: {config.LLM_MODEL_NAME}")
    print("=" * 60)
    
    # Build the UI
    app = build_app()
    # Launch starts the local web server
    app.launch(theme=gr.themes.Soft())
