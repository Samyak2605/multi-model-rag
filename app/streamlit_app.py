"""Streamlit demo application for Multi-Modal RAG System."""

import streamlit as st
import os
import sys
from pathlib import Path
import time
import pandas as pd

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import MultiModalRAGPipeline
from src.evaluation.benchmark import Evaluator

# Page config
st.set_page_config(
    page_title="Multi-Modal RAG System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "evaluator" not in st.session_state:
    st.session_state.evaluator = Evaluator()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    try:
        if st.session_state.pipeline is None:
            with st.spinner("Initializing pipeline..."):
                st.session_state.pipeline = MultiModalRAGPipeline()
        return True
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return False


def main():
    st.title("📊 Multi-Modal RAG System for Financial Documents")
    st.markdown("**IMF Report Analysis with Text, Tables, and Images**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys check
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY not set")
        else:
            st.success("✅ OpenAI API key configured")
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload an IMF report or financial document PDF"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            data_dir = Path("data/sample_docs")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = data_dir / uploaded_file.name
            
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Saved: {uploaded_file.name}")
            
            # Process document
            if st.button("🔄 Process Document", type="primary"):
                if initialize_pipeline():
                    # Check if using local embeddings (safe access)
                    try:
                        embedder = st.session_state.pipeline.embedder
                        using_local = getattr(embedder, 'use_local', False) or \
                                     (hasattr(embedder, 'local_model') and embedder.local_model is not None)
                    except (AttributeError, Exception):
                        using_local = False
                    
                    if using_local:
                        st.info("ℹ️ Using free local embeddings (no API quota needed)")
                    
                    with st.spinner("Processing document (this may take a minute)..."):
                        try:
                            st.session_state.pipeline.process_document(str(file_path))
                            st.session_state.documents_loaded = True
                            embedding_type = "local embeddings" if using_local else "OpenAI embeddings"
                            st.success(f"✅ Document processed and indexed using {embedding_type}!")
                            st.rerun()
                        except FileNotFoundError as e:
                            st.error(f"File not found: {str(e)}")
                        except ValueError as e:
                            st.error(f"Configuration error: {str(e)}")
                        except Exception as e:
                            error_msg = str(e)
                            if "poppler" in error_msg.lower() or "path" in error_msg.lower():
                                st.warning(
                                    f"⚠️ Poppler not found. Install with: `brew install poppler`\n"
                                    f"Note: System will work without images, but image OCR will be skipped."
                                )
                                # Continue anyway - images are optional
                                try:
                                    st.session_state.pipeline.process_document(str(file_path))
                                    st.session_state.documents_loaded = True
                                    st.success("✅ Document processed (without images)!")
                                    st.rerun()
                                except:
                                    pass
                            elif "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower() or "429" in error_msg.lower():
                                # This shouldn't happen if use_local is True, but handle it anyway
                                st.warning("⚠️ OpenAI quota issue detected. Ensuring local embeddings are used...")
                                try:
                                    import yaml
                                    config_path = "config/config.yaml"
                                    with open(config_path, 'r') as f:
                                        config = yaml.safe_load(f)
                                    config["embeddings"]["use_local"] = True
                                    with open(config_path, 'w') as f:
                                        yaml.dump(config, f, default_flow_style=False)
                                    
                                    # Reinitialize and retry
                                    st.session_state.pipeline = None
                                    if initialize_pipeline():
                                        with st.spinner("Retrying with local embeddings..."):
                                            st.session_state.pipeline.process_document(str(file_path))
                                            st.session_state.documents_loaded = True
                                            st.success("✅ Document processed using local embeddings!")
                                            st.rerun()
                                except Exception as e2:
                                    st.error(f"Error: {str(e2)}")
                                    st.info("💡 Install: `pip install sentence-transformers`")
                            else:
                                st.error(f"Error processing document: {error_msg}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
        
        st.divider()
        
        # Clear index
        if st.button("🗑️ Clear Index"):
            if st.session_state.pipeline:
                st.session_state.pipeline.clear_index()
                st.session_state.documents_loaded = False
                st.success("Index cleared")
                st.rerun()
        
        st.divider()
        
        # Evaluation metrics
        if st.session_state.query_history:
            st.subheader("📈 Recent Metrics")
            recent_queries = st.session_state.query_history[-5:]
            avg_confidence = sum(q.get("confidence", 0) for q in recent_queries) / len(recent_queries)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            st.metric("Total Queries", len(st.session_state.query_history))
    
    # Main content
    if not initialize_pipeline():
        st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 QA Chatbot", "📊 Evaluation Dashboard", "ℹ️ About"])
    
    with tab1:
        st.header("Question Answering")
        
        if not st.session_state.documents_loaded:
            st.info("👆 Please upload and process a document in the sidebar first.")
        else:
            # Query input
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input(
                    "Ask a question about the document:",
                    placeholder="e.g., What is the GDP growth forecast?",
                    key="query_input"
                )
            with col2:
                use_multi_hop = st.checkbox("Multi-hop", help="Use multi-hop reasoning")
            
            if query:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button("🔍 Search", type="primary"):
                        with st.spinner("Retrieving and generating answer..."):
                            try:
                                start_time = time.time()
                                
                                # Query pipeline
                                answer = st.session_state.pipeline.query(
                                    question=query,
                                    top_k=5,
                                    use_multi_hop=use_multi_hop
                                )
                                
                                latency = (time.time() - start_time) * 1000
                                
                                # Store in history
                                query_data = {
                                    "query": query,
                                    "answer": answer.get("answer", ""),
                                    "confidence": answer.get("confidence", 0.0),
                                    "num_sources": len(answer.get("sources", [])),
                                    "latency_ms": latency,
                                    "timestamp": time.time()
                                }
                                st.session_state.query_history.append(query_data)
                                
                                # Display answer
                                st.subheader("Answer")
                                answer_text = answer.get("answer", "No answer generated.")
                                if not answer_text or answer_text.startswith("Error"):
                                    st.warning("⚠️ Could not generate answer. Please check if documents are properly indexed.")
                                st.write(answer_text)
                                
                                # Confidence score
                                confidence = answer.get("confidence", 0.0)
                                st.metric("Confidence Score", f"{confidence:.3f}")
                                st.metric("Latency", f"{latency:.0f} ms")
                                
                                # Sources
                                sources = answer.get("sources", [])
                                if sources:
                                    st.subheader("📚 Sources")
                                    for i, source in enumerate(sources, 1):
                                        with st.expander(f"Source {i}: {source.get('source', 'Unknown')} (Confidence: {source.get('confidence', 0):.3f})"):
                                            st.write(f"**Page:** {source.get('page', 'Unknown')}")
                                            st.write(f"**Content:** {source.get('content', '')}")
                                else:
                                    st.info("No sources retrieved. The document may not be indexed yet.")
                                
                                # Reasoning steps (if multi-hop)
                                if "reasoning_steps" in answer:
                                    st.subheader("🧠 Reasoning Steps")
                                    for step in answer.get("reasoning_steps", []):
                                        st.write(f"**Step {step['step']}:** Retrieved {step['chunks_retrieved']} chunks")
                            
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
    
    with tab2:
        st.header("Evaluation Dashboard")
        
        if not st.session_state.query_history:
            st.info("No queries yet. Start asking questions in the QA Chatbot tab.")
        else:
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", len(st.session_state.query_history))
            
            with col2:
                avg_conf = sum(q.get("confidence", 0) for q in st.session_state.query_history) / len(st.session_state.query_history)
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            
            with col3:
                avg_latency = sum(q.get("latency_ms", 0) for q in st.session_state.query_history) / len(st.session_state.query_history)
                st.metric("Avg Latency", f"{avg_latency:.0f} ms")
            
            with col4:
                avg_sources = sum(q.get("num_sources", 0) for q in st.session_state.query_history) / len(st.session_state.query_history)
                st.metric("Avg Sources", f"{avg_sources:.1f}")
            
            st.divider()
            
            # Query history table
            st.subheader("Query History")
            df = pd.DataFrame(st.session_state.query_history)
            if not df.empty:
                # Format columns
                display_df = df[["query", "confidence", "num_sources", "latency_ms"]].copy()
                display_df.columns = ["Query", "Confidence", "Sources", "Latency (ms)"]
                display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.3f}")
                display_df["Latency (ms)"] = display_df["Latency (ms)"].apply(lambda x: f"{x:.0f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confidence Over Time")
                if len(df) > 0:
                    st.line_chart(df.set_index("timestamp")["confidence"])
            
            with col2:
                st.subheader("Latency Over Time")
                if len(df) > 0:
                    st.line_chart(df.set_index("timestamp")["latency_ms"])
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### Multi-Modal RAG System
        
        This system processes financial documents (IMF reports) with:
        
        - **Text Extraction**: Full text parsing with structure preservation
        - **Table Extraction**: Structured table data extraction using pdfplumber
        - **Image Processing**: OCR for charts and figures
        - **Semantic Chunking**: Intelligent chunking preserving document hierarchy
        - **Hybrid Retrieval**: Combines dense embeddings (OpenAI) with BM25 using RRF fusion
        - **Source Attribution**: Page-level citations for all answers
        - **Multi-Hop Reasoning**: Iterative retrieval for complex questions
        
        ### Architecture
        
        - **Ingestion**: PDF parsing, table extraction, image OCR
        - **Chunking**: Semantic chunking with metadata preservation
        - **Retrieval**: Hybrid search (dense + BM25) with ChromaDB
        - **Generation**: GPT-4o-mini for answer generation with citations
        
        ### Key Features
        
        ✅ Multi-modal content processing  
        ✅ Hybrid search with confidence scores  
        ✅ Source attribution at chunk level  
        ✅ Evaluation dashboard with metrics  
        ✅ Multi-hop reasoning capability
        """)


if __name__ == "__main__":
    main()

