# Multi-Modal RAG System for Financial Documents

A comprehensive Retrieval-Augmented Generation (RAG) system designed for processing IMF reports and financial documents with support for text, tables, images, and charts.

## Features

- **Multi-Modal Ingestion**: Extracts text, tables, and images (with OCR) from PDF documents
- **Semantic Chunking**: Intelligent chunking preserving document hierarchy and metadata
- **Hybrid Retrieval**: Combines dense embeddings (OpenAI) with BM25 using Reciprocal Rank Fusion (RRF)
- **Source Attribution**: Page-level citations for all generated answers
- **Multi-Hop Reasoning**: Iterative retrieval for complex questions requiring multiple document sections
- **Evaluation Dashboard**: Real-time metrics visualization (confidence scores, latency, precision@k)

## Architecture

```
┌─────────────┐
│   Ingestion │  → PDF Parser, Table Extractor, Image Processor (OCR)
└──────┬──────┘
       │
┌──────▼──────┐
│   Chunking  │  → Semantic Chunker with Metadata Preservation
└──────┬──────┘
       │
┌──────▼──────┐
│  Embedding  │  → OpenAI Embeddings (text-embedding-3-small)
└──────┬──────┘
       │
┌──────▼──────┐
│  Retrieval  │  → Hybrid Search (Dense + BM25 with RRF)
└──────┬──────┘
       │
┌──────▼──────┐
│  Generation │  → GPT-4o-mini with Source Citations
└─────────────┘
```

## Installation

1. **Clone the repository**:
```bash
cd "Iai lab"
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

Or export them:
```bash
export OPENAI_API_KEY="sk-proj-..."
export GROQ_API_KEY="gsk_..."
```

5. **Install system dependencies** (for OCR):
```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Download and install from:
# - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# - Poppler: https://github.com/oschwartz10612/poppler-windows/releases
```

## Usage

### Streamlit Application

Run the interactive demo:
```bash
streamlit run app/streamlit_app.py
```

The application provides:
- Document upload and processing
- Interactive QA chatbot
- Evaluation dashboard with metrics
- Source attribution visualization

### Programmatic Usage

```python
from src.pipeline import MultiModalRAGPipeline

# Initialize pipeline
pipeline = MultiModalRAGPipeline()

# Process a document
pipeline.process_document("data/sample_docs/imf_report.pdf")

# Query the system
answer = pipeline.query("What is the GDP growth forecast?")
print(answer["answer"])
print(f"Sources: {answer['sources']}")
```

## Project Structure

```
.
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py     # PDF text extraction
│   │   ├── table_extractor.py # Table extraction (pdfplumber)
│   │   └── image_processor.py # Image OCR
│   ├── chunking/
│   │   └── semantic_chunker.py # Semantic chunking
│   ├── retrieval/
│   │   ├── embedder.py       # OpenAI embeddings
│   │   └── retriever.py      # Hybrid retrieval
│   ├── generation/
│   │   └── qa_generator.py   # Answer generation
│   ├── evaluation/
│   │   └── benchmark.py      # Evaluation metrics
│   └── pipeline.py           # Main pipeline
├── app/
│   └── streamlit_app.py      # Streamlit demo
├── data/
│   ├── sample_docs/          # Upload PDFs here
│   └── vector_store/         # ChromaDB storage
├── requirements.txt
├── README.md
└── setup.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Embedding model
- Chunk size and overlap
- Retrieval parameters (top_k, hybrid weights)
- Generation settings (model, temperature)

## Key Design Decisions

1. **Table Extraction**: Chose `pdfplumber` over `camelot-py` for better handling of merged cells and multi-line headers
2. **Hybrid Search**: RRF fusion combines dense embeddings (semantic) with BM25 (keyword) for better retrieval
3. **Chunking Strategy**: Semantic chunking preserves document structure while maintaining LLM-friendly sizes
4. **Source Attribution**: Metadata preserved at chunk level enables accurate page-level citations

## Evaluation Metrics

The system tracks:
- **Precision@K**: Relevance of top-K retrieved chunks
- **Confidence Scores**: Retrieval confidence from hybrid search
- **Latency**: End-to-end query processing time
- **Source Coverage**: Number of unique pages referenced

## Limitations

- OCR quality depends on image resolution
- Table extraction may struggle with complex layouts
- Requires OpenAI API access for embeddings and generation
- Processing large documents can be slow

## Future Improvements

- Fine-tune embeddings on financial domain
- Add cross-modal reranking
- Implement retrieval fine-tuning
- Support for more document formats (Word, HTML)
- Batch processing for multiple documents

## License

MIT License

## Contact

For questions or issues, please open an issue on the repository.

