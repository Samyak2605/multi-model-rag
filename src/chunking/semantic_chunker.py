"""Semantic chunking strategy for financial documents."""

from typing import List, Dict, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunks documents semantically while preserving structure and metadata."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def chunk_document(
        self,
        pages: List[Dict],
        tables: Optional[List[Dict]] = None,
        images: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Chunk document preserving hierarchy and metadata.
        
        Args:
            pages: List of page dictionaries with text
            tables: List of extracted tables
            images: List of extracted images/figures
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        # Create lookup for tables and images by page
        tables_by_page = {}
        if tables:
            for table in tables:
                page_num = table["page_number"]
                if page_num not in tables_by_page:
                    tables_by_page[page_num] = []
                tables_by_page[page_num].append(table)
        
        images_by_page = {}
        if images:
            for img in images:
                page_num = img["page_number"]
                if page_num not in images_by_page:
                    images_by_page[page_num] = []
                images_by_page[page_num].append(img)
        
        # Process each page
        for page in pages:
            page_num = page["page_number"]
            page_text = page.get("text", "")
            
            if not page_text.strip():
                continue
            
            # Extract section titles (simple heuristic)
            section_title = self._extract_section_title(page_text)
            
            # Add related tables to page text
            if page_num in tables_by_page:
                for table in tables_by_page[page_num]:
                    table_text = table.get("text_representation", "")
                    if table_text:
                        page_text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]"
            
            # Add related image captions
            if page_num in images_by_page:
                for img in images_by_page[page_num]:
                    if img.get("ocr_text"):
                        page_text += f"\n\n[FIGURE CAPTION]\n{img['ocr_text']}\n[/FIGURE CAPTION]"
            
            # Split page into chunks
            text_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Skip very small chunks
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue
                
                # Truncate if too large
                if len(chunk_text) > self.max_chunk_size:
                    chunk_text = chunk_text[:self.max_chunk_size]
                
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "section_title": section_title,
                        "chunk_type": self._identify_chunk_type(chunk_text),
                        "source": f"Page {page_num}"
                    }
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def _extract_section_title(self, text: str) -> str:
        """Extract section title from text (simple heuristic)."""
        lines = text.split("\n")[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            # Look for lines that might be titles (short, uppercase, or numbered)
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^\d+\.?\s+[A-Z]', line) or
                 re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line))):
                return line
        return ""
    
    def _identify_chunk_type(self, text: str) -> str:
        """Identify type of chunk content."""
        if "[TABLE]" in text:
            return "table"
        elif "[FIGURE" in text:
            return "image"
        elif any(keyword in text.lower() for keyword in ["summary", "conclusion", "overview"]):
            return "summary"
        else:
            return "text"
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk plain text with optional metadata."""
        chunks = self.text_splitter.split_text(text)
        
        result = []
        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            chunk = {
                "content": chunk_text,
                "metadata": metadata or {}
            }
            chunk["metadata"]["chunk_index"] = idx
            result.append(chunk)
        
        return result

