"""Example usage of the Multi-Modal RAG System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.pipeline import MultiModalRAGPipeline

def main():
    """Example usage of the RAG pipeline."""
    
    print("Initializing Multi-Modal RAG Pipeline...")
    pipeline = MultiModalRAGPipeline()
    
    # Example: Process a document (uncomment when you have a PDF)
    # pdf_path = "data/sample_docs/your_document.pdf"
    # if Path(pdf_path).exists():
    #     print(f"\nProcessing document: {pdf_path}")
    #     pipeline.process_document(pdf_path)
    #     print("Document processed successfully!")
    #     
    #     # Example queries
    #     questions = [
    #         "What is the GDP growth forecast?",
    #         "What are the main fiscal policy recommendations?",
    #         "What does the inflation table show?"
    #     ]
    #     
    #     for question in questions:
    #         print(f"\n{'='*60}")
    #         print(f"Question: {question}")
    #         print(f"{'='*60}")
    #         
    #         answer = pipeline.query(question, top_k=5)
    #         
    #         print(f"\nAnswer: {answer['answer']}")
    #         print(f"\nConfidence: {answer.get('confidence', 0):.3f}")
    #         print(f"\nSources ({len(answer.get('sources', []))}):")
    #         for i, source in enumerate(answer.get('sources', [])[:3], 1):
    #             print(f"  {i}. {source.get('source', 'Unknown')} (Page {source.get('page', '?')})")
    # else:
    #     print(f"Document not found: {pdf_path}")
    #     print("Please add a PDF file to data/sample_docs/")
    
    print("\n✅ Pipeline initialized successfully!")
    print("To use the system:")
    print("1. Add a PDF to data/sample_docs/")
    print("2. Uncomment the code above")
    print("3. Or use the Streamlit app: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()

