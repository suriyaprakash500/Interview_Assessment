import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from src.ingestion.extractor import PDFExtractor
from src.ingestion.captioner import QwenVLDecoder

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Coordinates PDF text extraction, page image rendering, visual caption generation,
    and manages cache storage.
    """
    def __init__(self, config: Dict[str, Any], mock_mode: bool = False):
        self.config = config
        self.pdf_path = config.get("pdf_path", "input/financial_report.pdf")
        self.output_dir = config.get("output_dir", "outputs")
        self.images_dir = config.get("images_dir", "outputs/images")
        self.dpi = config.get("pdf_rendering_dpi", 150)
        self.cache_path = os.path.join(self.output_dir, "ingested_pages.json")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Initialize Extractor
        self.extractor = PDFExtractor(self.pdf_path, dpi=self.dpi)
        
        # Initialize Decoder
        self.decoder = QwenVLDecoder(mock_mode=mock_mode)
        
        # Load Cache
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    logger.info(f"Loaded existing ingestion cache with {len(cache_data.get('pages', {}))} pages.")
                    return cache_data
            except Exception as e:
                logger.error(f"Error loading cache at {self.cache_path}: {e}. Initializing clean cache.")
        
        return {
            "document_name": os.path.basename(self.pdf_path),
            "total_pages": self.extractor.get_page_count(),
            "pages": {}
        }

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved ingestion cache with {len(self.cache['pages'])} pages.")
        except Exception as e:
            logger.error(f"Error saving cache to {self.cache_path}: {e}")

    def run(self, start_page: Optional[int] = None, end_page: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the ingestion pipeline for the specified page range (1-indexed).
        """
        total_pdf_pages = self.extractor.get_page_count()
        
        # Resolve page range
        s_page = start_page or self.config.get("ingest_range", {}).get("start_page") or 1
        e_page = end_page or self.config.get("ingest_range", {}).get("end_page") or total_pdf_pages
        
        # Clip ranges
        s_page = max(1, min(s_page, total_pdf_pages))
        e_page = max(s_page, min(e_page, total_pdf_pages))
        
        logger.info(f"Running ingestion for pages {s_page} to {e_page} (Total pages: {total_pdf_pages})")
        
        # Check if total pages changed (e.g. document changed)
        if self.cache.get("total_pages") != total_pdf_pages:
            logger.warning("PDF page count mismatch with cache. Re-initializing cache metadata.")
            self.cache["total_pages"] = total_pdf_pages
            self.cache["document_name"] = os.path.basename(self.pdf_path)

        for page_num in range(s_page, e_page + 1):
            page_str = str(page_num)
            
            # Check cache
            if page_str in self.cache["pages"]:
                # Check if image and text exist, if so, skip
                page_data = self.cache["pages"][page_str]
                img_path = page_data.get("page_image_path")
                if img_path and os.path.exists(img_path) and page_data.get("extracted_text") is not None:
                    logger.info(f"Page {page_num} already ingested. Skipping.")
                    continue
            
            logger.info(f"Processing Page {page_num}...")
            
            # Step 1: Extract Text
            text = self.extractor.extract_text(page_num)
            
            # Step 2: Render Page as Image
            image_name = f"page_{page_num}.png"
            image_path = os.path.join(self.images_dir, image_name)
            rendered_path = self.extractor.render_page_to_image(page_num, image_path)
            
            if not rendered_path:
                logger.error(f"Failed to render Page {page_num} image. Skipping.")
                continue
                
            # Step 3: Send to Qwen-VL (real or mock)
            visual_caption = self.decoder.generate_caption(rendered_path, page_num, text)
            
            # Step 4: Save metadata in cache
            self.cache["pages"][page_str] = {
                "page_number": page_num,
                "page_image_path": os.path.abspath(rendered_path),
                "extracted_text": text,
                "visual_caption": visual_caption,
                "visual_caption_model": self.decoder.model_name if not self.decoder.mock_mode else "mock-vision-descriptor",
                "ingested_at": datetime.now().isoformat()
            }
            
            # Save progress incrementally
            self._save_cache()
            
        self.extractor.close()
        logger.info(f"Ingestion pipeline run completed. Total ingested pages: {len(self.cache['pages'])}")
        return self.cache
