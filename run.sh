#!/bin/bash
# Quick start script for Multi-Modal RAG System

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set in .env file"
    exit 1
fi

# Run Streamlit app
streamlit run app/streamlit_app.py

