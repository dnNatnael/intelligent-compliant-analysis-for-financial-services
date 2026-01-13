#!/usr/bin/env python3
"""
Main entry point for the Gradio RAG Chatbot application.

Run with: python app.py
"""

from gradio_app import build_app

if __name__ == "__main__":
    # Build and launch the Gradio application
    app = build_app()
    app.launch(
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,      # Default Gradio port
        share=False            # Set to True to create a public link
    )