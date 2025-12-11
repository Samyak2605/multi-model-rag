"""Table extraction from PDF documents using pdfplumber."""

import pdfplumber
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extracts tables from PDF documents preserving structure."""
    
    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        """
        Initialize table extractor.
        
        Args:
            min_rows: Minimum rows to consider as table
            min_cols: Minimum columns to consider as table
        """
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.logger = logging.getLogger(__name__)
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extract all tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of table dictionaries with data and metadata
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    
                    if not page_tables:
                        continue
                    
                    for table_idx, table in enumerate(page_tables):
                        if not table or len(table) < self.min_rows:
                            continue
                        
                        # Convert to DataFrame for easier handling
                        df = self._table_to_dataframe(table)
                        
                        if df.shape[1] < self.min_cols:
                            continue
                        
                        # Handle merged cells and multi-line headers
                        df = self._clean_dataframe(df)
                        
                        table_data = {
                            "page_number": page_num,
                            "table_index": table_idx,
                            "dataframe": df,
                            "data": df.to_dict("records"),
                            "text_representation": self._dataframe_to_text(df),
                            "shape": df.shape
                        }
                        
                        tables.append(table_data)
                        
        except Exception as e:
            self.logger.error(f"Error extracting tables from {pdf_path}: {str(e)}")
            raise
        
        return tables
    
    def _table_to_dataframe(self, table: List[List]) -> pd.DataFrame:
        """Convert raw table data to DataFrame."""
        if not table:
            return pd.DataFrame()
        
        # Use first row as header if it looks like headers
        headers = table[0] if table else []
        data = table[1:] if len(table) > 1 else []
        
        # Clean headers
        headers = [str(h).strip() if h else f"Column_{i}" 
                  for i, h in enumerate(headers)]
        
        # Ensure we have enough columns
        max_cols = max(len(row) for row in table) if table else 0
        headers = headers + [f"Column_{i}" for i in range(len(headers), max_cols)]
        
        # Pad rows to match header length
        padded_data = []
        for row in data:
            padded_row = row + [None] * (max_cols - len(row))
            padded_data.append(padded_row[:max_cols])
        
        df = pd.DataFrame(padded_data, columns=headers[:max_cols])
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame: handle merged cells, remove empty rows/cols."""
        # Forward fill for merged cells (common pattern)
        df = df.ffill(axis=0).fillna("")
        
        # Remove completely empty rows
        df = df[~df.isna().all(axis=1)]
        
        # Remove completely empty columns
        df = df.loc[:, ~df.isna().all(axis=0)]
        
        return df
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text format."""
        if df.empty:
            return ""
        
        lines = []
        # Header
        header = " | ".join(str(col) for col in df.columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for _, row in df.iterrows():
            row_str = " | ".join(str(val) if pd.notna(val) else "" for val in row)
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def extract_table_by_page(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Extract tables from specific page."""
        all_tables = self.extract_tables(pdf_path)
        return [t for t in all_tables if t["page_number"] == page_num]

