"""Main pipeline orchestrating ingestion, chunking, retrieval, and generation."""

import yaml
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set directly

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.image_processor import ImageProcessor
from src.chunking.semantic_chunker import SemanticChunker
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import HybridRetriever
from src.generation.qa_generator import QAGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalRAGPipeline:
    """Main pipeline for multi-modal RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Ingestion
        self.pdf_parser = PDFParser()
        self.table_extractor = TableExtractor(
            min_rows=self.config["table_extraction"]["min_rows"],
            min_cols=self.config["table_extraction"]["min_cols"]
        )
        self.image_processor = ImageProcessor(
            ocr_enabled=self.config["image_processing"]["ocr_enabled"],
            ocr_language=self.config["image_processing"]["ocr_language"]
        )
        
        # Chunking
        chunk_config = self.config["chunking"]
        self.chunker = SemanticChunker(
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            min_chunk_size=chunk_config["min_chunk_size"],
            max_chunk_size=chunk_config["max_chunk_size"]
        )
        
        # Embedding
        embed_config = self.config["embeddings"]
        use_local = embed_config.get("use_local", False)
        # Force local if explicitly set in config
        if use_local:
            logger.info("Using local embeddings as configured")
        self.embedder = Embedder(
            model=embed_config["model"],
            api_key=os.getenv("OPENAI_API_KEY"),
            use_local=use_local
        )
        
        # Retrieval
        ret_config = self.config["retrieval"]
        vec_config = self.config["vector_store"]
        self.retriever = HybridRetriever(
            persist_directory=vec_config["persist_directory"],
            collection_name=vec_config["collection_name"],
            dense_weight=ret_config["hybrid_search"]["dense_weight"],
            bm25_weight=ret_config["hybrid_search"]["bm25_weight"],
            rrf_k=ret_config["hybrid_search"]["rrf_k"]
        )
        
        # Generation
        gen_config = self.config["generation"]
        self.qa_generator = QAGenerator(
            model=gen_config["model"],
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"]
        )
    
    def ingest_document(self, pdf_path: str) -> Dict:
        """
        Ingest a PDF document and extract all modalities.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content
        """
        logger.info(f"Ingesting document: {pdf_path}")
        
        # Parse text
        pages = self.pdf_parser.parse(pdf_path)["pages"]
        logger.info(f"Extracted {len(pages)} pages")
        
        # Extract tables
        try:
            tables = self.table_extractor.extract_tables(pdf_path)
            logger.info(f"Extracted {len(tables)} tables")
        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}. Continuing without tables.")
            tables = []
        
        # Extract images (with OCR) - optional, won't fail if poppler missing
        try:
            images = self.image_processor.extract_images(pdf_path)
            logger.info(f"Extracted {len(images)} images")
        except Exception as e:
            logger.warning(f"Image extraction failed: {str(e)}. Continuing without images.")
            images = []
        
        return {
            "pages": pages,
            "tables": tables,
            "images": images,
            "source_file": pdf_path
        }
    
    def process_document(self, pdf_path: str) -> None:
        """
        Process document: ingest, chunk, embed, and index.
        
        Args:
            pdf_path: Path to PDF file
        """
        # Ingest
        content = self.ingest_document(pdf_path)
        
        # Chunk
        chunks = self.chunker.chunk_document(
            pages=content["pages"],
            tables=content["tables"],
            images=content["images"]
        )
        logger.info(f"Created {len(chunks)} chunks")
        
        # Embed
        try:
            chunks = self.embedder.embed_chunks(chunks)
            logger.info("Generated embeddings")
        except Exception as e:
            error_str = str(e).lower()
            # If quota error and not using local, try to switch
            if ("quota" in error_str or "insufficient_quota" in error_str or "429" in error_str) and not self.embedder.use_local:
                logger.warning("OpenAI quota exceeded, switching to local embeddings")
                # Reinitialize embedder with local mode
                embed_config = self.config["embeddings"]
                self.embedder = Embedder(
                    model="all-MiniLM-L6-v2",
                    use_local=True
                )
                # Retry embedding
                chunks = self.embedder.embed_chunks(chunks)
                logger.info("Generated embeddings using local model")
            else:
                raise
        
        # Index
        self.retriever.add_documents(chunks)
        logger.info("Indexed documents")
    
    def query(self, question: str, top_k: int = 5, use_multi_hop: bool = False) -> Dict:
        """
        Query the system and generate answer.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_multi_hop: Whether to use multi-hop reasoning
            
        Returns:
            Dictionary with answer and metadata
        """
        if use_multi_hop:
            answer = self.qa_generator.generate_multi_hop(
                question=question,
                retriever=self.retriever,
                embedder=self.embedder,
                max_hops=2
            )
        else:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(question)
            
            # Retrieve
            chunks = self.retriever.retrieve(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Generate
            answer = self.qa_generator.generate(question, chunks, include_sources=True)
        
        return answer
    
    def clear_index(self):
        """Clear the document index."""
        self.retriever.clear()
        logger.info("Cleared document index")

