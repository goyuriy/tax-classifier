import pdfplumber
from pathlib import Path
from typing import List
from dataclasses import dataclass
from src.utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class PDFPage:
    page_number: int
    text: str
    width: float
    height: float
    has_images: bool
    is_scanned: bool = False

class PDFLoader:
    """Loads PDFs and flags pages that look like scans (images but little extractable text)."""

    DEFAULT_SCAN_THRESHOLD_CHARS = 50

    def __init__(self, scan_threshold_chars: int = DEFAULT_SCAN_THRESHOLD_CHARS):
        self.scan_threshold_chars = scan_threshold_chars

    def load(self, file_path: str) -> List[PDFPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        pages_data = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    cleaned_text = text.strip()
                    
                    # Check for raster images
                    has_images = len(page.images) > 0
                    
                    # if images present but text is missing/low
                    is_scanned = has_images and (len(cleaned_text) < self.scan_threshold_chars)
                    
                    if is_scanned:
                        logger.warning(
                            f"Page {i+1} in {path.name} has images but little text ({len(cleaned_text)} chars). "
                            "It is likely a scanned document requiring OCR."
                        )

                    pages_data.append(PDFPage(
                        page_number=i + 1,
                        text=cleaned_text,
                        width=float(page.width),
                        height=float(page.height),
                        has_images=has_images,
                        is_scanned=is_scanned
                    ))
                    
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

        return pages_data
