# =============================================================================
# gradio_app.py - Enhanced Gradio Chatbot UI for RAG Pipeline
# =============================================================================
"""
An enhanced Gradio chatbot for interactive RAG question answering with source display.

Run with: python gradio_app.py

Features:
- Chat interface for asking questions
- Displays source text chunks below answers for transparency
- Streaming response generation (token-by-token)
- Clear button to reset conversation
- Reuses existing RAGPipeline (FAISS + FLAN-T5)
"""

# sys is used to modify Python's import path at runtime
import sys
# Path helps us build file paths in a cross-platform way
from pathlib import Path
import time

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
from src.llm import format_docs_for_context


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
# HELPER FUNCTIONS
# =============================================================================

def format_sources_for_display(sources):
    """
    Format retrieved source documents for display in the UI.
    
    Args:
        sources: List of Document objects from retrieval
        
    Returns:
        Formatted HTML string displaying sources
    """
    if not sources:
        return "**No sources retrieved.**"
    
    html_parts = ["<div style='margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>"]
    html_parts.append("<h3 style='margin-top: 0; color: #333;'>📚 Sources Used:</h3>")
    
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        source_info = []
        for key in ("complaint_id", "product", "chunk_index"):
            if key in meta:
                source_info.append(f"<strong>{key}</strong>: {meta[key]}")
        
        source_header = f"<strong>Source {i}</strong>"
        if source_info:
            source_header += f" ({', '.join(source_info)})"
        
        # Truncate long content for display
        content = doc.page_content.replace("\n", " ").strip()
        if len(content) > 300:
            content = content[:300] + "..."
        
        html_parts.append(f"""
        <div style='margin-bottom: 15px; padding: 10px; background-color: white; border-left: 3px solid #4CAF50; border-radius: 4px;'>
            <div style='color: #666; font-size: 0.9em; margin-bottom: 5px;'>{source_header}</div>
            <div style='color: #333; line-height: 1.5;'>{content}</div>
        </div>
        """)
    
    html_parts.append("</div>")
    return "".join(html_parts)


# =============================================================================
# CHAT FUNCTION WITH STREAMING
# =============================================================================

def chat_with_sources(message: str, history: list, k: int, enable_streaming: bool = True):
    """
    Process a user message and return the assistant response with sources.
    
    Args:
        message: The user's question
        history: List of (user_msg, assistant_msg) tuples (chat history)
        k: Number of sources to retrieve
        enable_streaming: Whether to stream the response token-by-token
        
    Yields:
        Tuple of (answer_text, sources_html) for streaming updates
    """
    # 1) Basic input validation
    if not message.strip():
        yield "Please enter a question.", ""
        return
    
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
        yield error_msg, ""
        return
    except Exception as e:
        # Catch-all for unexpected loading errors
        yield f"Error loading RAG pipeline: {e}", ""
        return
    
    # 3) Update retriever settings (k = number of retrieved chunks)
    if rag.retrieval_k != k:
        rag.retrieval_k = k
        # Re-create the retriever with the new k
        rag.retriever = rag.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    # 4) Run the RAG steps:
    try:
        # Retrieve top-k relevant chunks from FAISS
        sources = rag.retrieve(message)

        # Format the retrieved chunks into a single context string for the prompt
        formatted_context = format_docs_for_context(sources)
        # Truncate by characters to reduce risk of exceeding model max length
        if len(formatted_context) > MAX_CONTEXT_CHARS:
            formatted_context = formatted_context[:MAX_CONTEXT_CHARS] + "..."

        # Format sources for display (we'll show this at the end)
        sources_html = format_sources_for_display(sources)
        
        # Generate an answer using the LLM
        if enable_streaming:
            # Simulate streaming by generating and yielding token-by-token
            # Note: FLAN-T5 doesn't support true streaming, so we simulate it
            answer = rag.generate(message, formatted_context)
            
            # Simulate streaming by yielding partial answers
            accumulated = ""
            words = answer.split()
            for i, word in enumerate(words):
                accumulated += word + " "
                # Yield partial answer with sources placeholder
                yield accumulated.strip(), ""
                # Small delay to simulate streaming
                time.sleep(0.05)
            
            # Final yield with complete answer and sources
            yield answer, sources_html
        else:
            # Non-streaming: generate and return immediately
            answer = rag.generate(message, formatted_context)
            yield answer, sources_html
            
    except Exception as e:
        yield f"Error generating answer: {e}", ""


# =============================================================================
# BUILD GRADIO UI
# =============================================================================

def build_app():
    """
    Build and return the enhanced Gradio Blocks app with source display.
    """
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .source-display {
        margin-top: 20px;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    """
    
    # Blocks is Gradio's layout container
    with gr.Blocks(title="Financial Services RAG Chatbot", theme=gr.themes.Soft(), css=custom_css) as app:
        # Header with title and description
        gr.Markdown("""
        # 🏦 Intelligent Complaint Analysis Chatbot
        
        Ask questions about financial service complaints and get AI-powered answers with source citations.
        """)
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400,
            show_copy_button=True,
            avatar_images=(None, "🤖")
        )
        
        # Sources display area (hidden initially, shown after answer)
        sources_display = gr.HTML(
            label="Sources",
            visible=True,
            elem_classes=["source-display"]
        )
        
        # Input section
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your question here... (e.g., 'What are common issues with credit cards?')",
                show_label=False,
                scale=5,
                container=False,
            )
            send_btn = gr.Button("Ask", variant="primary", scale=1, size="lg")
        
        # Controls row
        with gr.Row():
            clear_btn = gr.Button("Clear Conversation", variant="secondary")
            streaming_toggle = gr.Checkbox(
                label="Enable Streaming",
                value=True,
                info="Show response token-by-token"
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=config.RETRIEVAL_K,
                step=1,
                label="Number of Sources (k)",
                info="How many document chunks to retrieve"
            )
        
        # Info section
        with gr.Accordion("ℹ️ About this Chatbot", open=False):
            gr.Markdown("""
            **How it works:**
            1. Your question is converted to a vector embedding
            2. The system searches a database of complaint documents for relevant chunks
            3. The AI generates an answer based on the retrieved context
            4. Source documents are displayed below for verification
            
            **Features:**
            - ✅ Retrieval-Augmented Generation (RAG) for accurate answers
            - ✅ Source citations for transparency and trust
            - ✅ Streaming responses for better user experience
            - ✅ Adjustable retrieval parameters
            """)
        
        # Footer
        gr.Markdown(
            """
            ---
            <center>
            <small>Built with LangChain, FAISS, and Gradio | RAG Pipeline for Financial Services</small>
            </center>
            """,
        )
        
        # =================================================================
        # EVENT HANDLERS
        # =================================================================
        
        def respond(message, chat_history, k, enable_streaming):
            """Handle user message and update chat with streaming support."""
            if not message.strip():
                return chat_history, ""
            
            # Initialize chat history if needed
            if chat_history is None:
                chat_history = []
            
            # Add user message to history
            chat_history.append(gr.ChatMessage(role="user", content=message))
            
            # Stream the response
            sources_html = ""
            for answer_text, sources in chat_with_sources(message, chat_history, k, enable_streaming):
                sources_html = sources if sources else sources_html
                # Update chat history with current answer
                if chat_history and chat_history[-1].role == "assistant":
                    chat_history[-1] = gr.ChatMessage(role="assistant", content=answer_text)
                else:
                    chat_history.append(gr.ChatMessage(role="assistant", content=answer_text))
                yield chat_history, sources_html

        def clear_chat():
            """Clear chat history and sources."""
            return [], ""

        # Wire up events
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, k_slider, streaming_toggle],
            outputs=[chatbot, sources_display],
        ).then(
            lambda: "",  # Clear input after sending
            outputs=[msg_input]
        )
        
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, k_slider, streaming_toggle],
            outputs=[chatbot, sources_display],
        ).then(
            lambda: "",  # Clear input after sending
            outputs=[msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, sources_display],
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Print helpful info when starting the app
    print("=" * 60)
    print("🏦 Starting Intelligent Complaint Analysis Chatbot (Gradio)")
    print("=" * 60)
    # Show key configuration so beginners know what is being used
    print(f"Vector store: {config.VECTOR_STORE_DIR}")
    print(f"Embedding model: {config.EMBEDDING_MODEL_NAME}")
    print(f"LLM model: {config.LLM_MODEL_NAME}")
    print(f"Retrieval k: {config.RETRIEVAL_K}")
    print("=" * 60)
    print("🌐 Opening web interface...")
    print("=" * 60)
    
    # Build the UI
    app = build_app()
    # Launch starts the local web server
    app.launch(
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,       # Default Gradio port
        share=False            # Set to True to create a public link
    )
