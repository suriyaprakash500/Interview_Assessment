import os
import logging
import fitz  # PyMuPDF
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    Handles PDF operations including page text extraction and rendering pages as images.
    """
    def __init__(self, pdf_path: str, dpi: int = 150):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.doc = None
        self._open_pdf()

    def _open_pdf(self):
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"Successfully opened PDF: {self.pdf_path} (Pages: {len(self.doc)})")
        except Exception as e:
            logger.error(f"Failed to open PDF at {self.pdf_path}: {e}")
            raise e

    def _ensure_open(self):
        is_closed = True
        if self.doc is not None:
            try:
                # PyMuPDF triggers __len__ which raises ValueError if closed
                _ = len(self.doc)
                is_closed = False
            except ValueError:
                is_closed = True
        if is_closed:
            self._open_pdf()

    def get_page_count(self) -> int:
        self._ensure_open()
        if self.doc:
            return len(self.doc)
        return 0

    def extract_text(self, page_num: int) -> str:
        """
        Extracts raw text from a page (1-indexed).
        """
        self._ensure_open()
        try:
            # fitz is 0-indexed internally
            page = self.doc[page_num - 1]
            text = page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num}: {e}")
            return ""

    def render_page_to_image(self, page_num: int, output_path: str) -> Optional[str]:
        """
        Renders a page (1-indexed) as a PNG image and saves it to output_path.
        Returns the output path if successful, otherwise None.
        """
        self._ensure_open()
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # fitz is 0-indexed internally
            page = self.doc[page_num - 1]
            
            # Use matrix to scale by DPI (default is 72 dpi, so zoom = dpi / 72)
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(output_path)
            
            logger.debug(f"Rendered page {page_num} to image at {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error rendering page {page_num} to image: {e}")
            return None

    def close(self):
        if self.doc and not self.doc.is_closed:
            self.doc.close()
            logger.info("Closed PDF document")
