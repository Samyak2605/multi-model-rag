"""Quick test to verify all imports work."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Testing imports...")
    
    from src.ingestion.pdf_parser import PDFParser
    print("✅ PDFParser imported")
    
    from src.ingestion.table_extractor import TableExtractor
    print("✅ TableExtractor imported")
    
    from src.ingestion.image_processor import ImageProcessor
    print("✅ ImageProcessor imported")
    
    from src.chunking.semantic_chunker import SemanticChunker
    print("✅ SemanticChunker imported")
    
    from src.retrieval.embedder import Embedder
    print("✅ Embedder imported")
    
    from src.retrieval.retriever import HybridRetriever
    print("✅ HybridRetriever imported")
    
    from src.generation.qa_generator import QAGenerator
    print("✅ QAGenerator imported")
    
    from src.evaluation.benchmark import Evaluator
    print("✅ Evaluator imported")
    
    from src.pipeline import MultiModalRAGPipeline
    print("✅ MultiModalRAGPipeline imported")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

