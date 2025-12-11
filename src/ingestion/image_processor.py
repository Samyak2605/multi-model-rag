"""Image processing and OCR for PDF documents."""

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from PIL import Image
from typing import List, Dict, Optional
from pathlib import Path
import logging
import io

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Processes images from PDFs and performs OCR."""
    
    def __init__(self, ocr_enabled: bool = True, ocr_language: str = "eng"):
        """
        Initialize image processor.
        
        Args:
            ocr_enabled: Whether to perform OCR on images
            ocr_language: OCR language code
        """
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.logger = logging.getLogger(__name__)
    
    def extract_images(self, pdf_path: str, dpi: int = 200) -> List[Dict]:
        """
        Extract images from PDF pages.
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion
            
        Returns:
            List of image dictionaries with OCR text and metadata
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        images = []
        
        # Check if pdf2image is available
        if not PDF2IMAGE_AVAILABLE:
            self.logger.warning("pdf2image not available. Skipping image extraction.")
            return images
        
        try:
            # Convert PDF pages to images
            try:
                pdf_images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            except Exception as e:
                error_msg = str(e).lower()
                if "poppler" in error_msg or "path" in error_msg:
                    self.logger.warning(
                        f"Poppler not found. Image extraction skipped. "
                        f"Install poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)"
                    )
                    return images
                else:
                    raise
            
            for page_num, img in enumerate(pdf_images, start=1):
                # Get image metadata
                width, height = img.size
                
                image_data = {
                    "page_number": page_num,
                    "image": img,
                    "width": width,
                    "height": height,
                    "ocr_text": "",
                    "has_text": False
                }
                
                # Perform OCR if enabled
                if self.ocr_enabled and TESSERACT_AVAILABLE:
                    try:
                        ocr_text = pytesseract.image_to_string(img, lang=self.ocr_language)
                        image_data["ocr_text"] = ocr_text.strip()
                        image_data["has_text"] = len(ocr_text.strip()) > 0
                    except Exception as e:
                        self.logger.warning(f"OCR failed for page {page_num}: {str(e)}")
                elif self.ocr_enabled:
                    self.logger.warning("Tesseract not available. OCR skipped.")
                
                images.append(image_data)
                
        except Exception as e:
            self.logger.warning(f"Error extracting images from {pdf_path}: {str(e)}. Continuing without images.")
            return images
        
        return images
    
    def extract_figures(self, pdf_path: str) -> List[Dict]:
        """
        Extract figures/charts from PDF (simplified - focuses on pages with images).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of figure metadata
        """
        images = self.extract_images(pdf_path)
        
        figures = []
        for img_data in images:
            if img_data["has_text"] or img_data["width"] > 400:  # Likely a figure
                figures.append({
                    "page_number": img_data["page_number"],
                    "ocr_text": img_data["ocr_text"],
                    "caption": self._extract_caption(img_data["ocr_text"])
                })
        
        return figures
    
    def _extract_caption(self, ocr_text: str) -> str:
        """Extract potential caption from OCR text."""
        lines = ocr_text.split("\n")
        # Simple heuristic: last non-empty line might be caption
        for line in reversed(lines):
            if line.strip() and len(line.strip()) < 200:
                return line.strip()
        return ""

