"""PDF Parser for extracting text and structure from financial documents."""

import pdfplumber
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Parses PDF documents and extracts text with structure preservation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, pdf_path: str) -> Dict:
        """
        Parse PDF and extract text with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with pages, text, and metadata
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        result = {
            "pages": [],
            "total_pages": 0,
            "metadata": {}
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result["total_pages"] = len(pdf.pages)
                result["metadata"] = {
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", "")
                }
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    
                    page_data = {
                        "page_number": page_num,
                        "text": page_text,
                        "bbox": page.bbox,
                        "width": page.width,
                        "height": page.height
                    }
                    
                    result["pages"].append(page_data)
                    
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise
        
        return result
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """Extract text content page by page."""
        result = self.parse(pdf_path)
        return [page["text"] for page in result["pages"]]

