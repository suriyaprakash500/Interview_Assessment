import os
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import faiss
from src.embeddings.embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

class FAISSIndexer:
    """
    Manages building, saving, loading, and searching FAISS indexes.
    Handles two indexes: Multimodal Index and Text-only Baseline.
    """
    def __init__(self, config: Dict[str, Any], embedder: DocumentEmbedder):
        self.config = config
        self.embedder = embedder
        self.indexes_dir = config.get("indexes_dir", "indexes")
        
        # Paths
        self.mm_index_path = config.get("multimodal_index_path", "indexes/multimodal_index.faiss")
        self.text_index_path = config.get("text_baseline_index_path", "indexes/text_baseline_index.faiss")
        self.mm_meta_path = self.mm_index_path.replace(".faiss", "_metadata.json")
        self.text_meta_path = self.text_index_path.replace(".faiss", "_metadata.json")
        
        # State
        self.mm_index = None
        self.mm_metadata = []
        self.text_index = None
        self.text_metadata = []
        
        os.makedirs(self.indexes_dir, exist_ok=True)

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits text into chunks of specified size and overlap.
        """
        chunks = []
        if not text:
            return chunks
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # Prevent infinite loop if overlap is larger than chunk size
            step = chunk_size - chunk_overlap
            if step <= 0:
                step = chunk_size // 2
            start += step
        return chunks

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalizes vectors to unit length so that Inner Product (FAISS IndexFlatIP)
        is equivalent to Cosine Similarity.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_indexes(self, ingested_data: Dict[str, Any]):
        """
        Builds both the Multimodal Index and Text-only Baseline Index from ingested page data.
        """
        pages = ingested_data.get("pages", {})
        doc_name = ingested_data.get("document_name", "financial_report.pdf")
        
        if not pages:
            logger.warning("No ingested pages found to build indexes.")
            return

        logger.info(f"Building FAISS Indexes from {len(pages)} ingested pages...")

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # 1. Build Index 1: Multimodal (Hybrid Chunked: Text + Visual Caption)
        # -------------------------------------------------------------
        mm_texts = []
        mm_meta_mappings = []
        chunk_size = self.config.get("chunk_size", 800)
        chunk_overlap = self.config.get("chunk_overlap", 100)
        
        for page_num_str, page_data in pages.items():
            page_num = int(page_num_str)
            text = page_data.get("extracted_text", "")
            visual_caption = page_data.get("visual_caption", "")
            img_path = page_data.get("page_image_path", "")
            
            if text.strip() or visual_caption.strip():
                # Split page text into chunks
                chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                if not chunks and visual_caption.strip():
                    # If page has no text, but has a visual caption, create a single visual-only chunk
                    chunks = [""]
                
                for chunk in chunks:
                    # Enrich text chunk with VLM visual caption context (Option 3)
                    parts = []
                    if chunk.strip():
                        parts.append(chunk)
                    if visual_caption.strip():
                        parts.append(f"[Visual Image Context]: {visual_caption}")
                    
                    hybrid_text = "\n\n".join(parts)
                    
                    mm_texts.append(hybrid_text)
                    mm_meta_mappings.append({
                        "document_name": doc_name,
                        "page_number": page_num,
                        "page_image_path": img_path,
                        "type": "hybrid",
                        "snippet": chunk if chunk.strip() else visual_caption[:500] + ("..." if len(visual_caption) > 500 else "")
                    })
                    
        if mm_texts:
            logger.info(f"Generating embeddings for Multimodal Index (Total hybrid vectors: {len(mm_texts)})...")
            mm_embeddings = self.embedder.embed_documents(mm_texts)
            mm_embeddings = self._normalize_vectors(mm_embeddings)
            
            dim = mm_embeddings.shape[1]
            self.mm_index = faiss.IndexFlatIP(dim)
            self.mm_index.add(mm_embeddings)
            self.mm_metadata = mm_meta_mappings
            logger.info(f"Multimodal Index built with {self.mm_index.ntotal} vectors of dimension {dim}.")
        
        # -------------------------------------------------------------
        # 2. Build Index 2: Text-only Baseline (Chunked Text Only)
        # -------------------------------------------------------------
        text_chunks = []
        text_meta_mappings = []
        chunk_size = self.config.get("chunk_size", 800)
        chunk_overlap = self.config.get("chunk_overlap", 100)
        
        for page_num_str, page_data in pages.items():
            page_num = int(page_num_str)
            text = page_data.get("extracted_text", "")
            img_path = page_data.get("page_image_path", "")
            
            if text.strip():
                chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                for chunk in chunks:
                    text_chunks.append(chunk)
                    text_meta_mappings.append({
                        "document_name": doc_name,
                        "page_number": page_num,
                        "page_image_path": img_path,
                        "type": "chunk",
                        "snippet": chunk
                    })
                    
        if text_chunks:
            logger.info(f"Generating embeddings for Text-only Baseline Index (Total chunks: {len(text_chunks)})...")
            text_embeddings = self.embedder.embed_documents(text_chunks)
            text_embeddings = self._normalize_vectors(text_embeddings)
            
            dim = text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatIP(dim)
            self.text_index.add(text_embeddings)
            self.text_metadata = text_meta_mappings
            logger.info(f"Text-only Baseline Index built with {self.text_index.ntotal} vectors of dimension {dim}.")

    def save_indexes(self):
        """
        Saves both indexes and metadata files to disk.
        """
        # Save Multimodal
        if self.mm_index is not None:
            try:
                faiss.write_index(self.mm_index, self.mm_index_path)
                with open(self.mm_meta_path, "w", encoding="utf-8") as f:
                    json.dump(self.mm_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved Multimodal Index to {self.mm_index_path} and metadata.")
            except Exception as e:
                logger.error(f"Error saving Multimodal Index: {e}")
                
        # Save Text Baseline
        if self.text_index is not None:
            try:
                faiss.write_index(self.text_index, self.text_index_path)
                with open(self.text_meta_path, "w", encoding="utf-8") as f:
                    json.dump(self.text_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved Text-only Baseline Index to {self.text_index_path} and metadata.")
            except Exception as e:
                logger.error(f"Error saving Text-only Baseline Index: {e}")

    def load_indexes(self) -> bool:
        """
        Loads indexes and metadata files from disk. Returns True if successful.
        """
        success = True
        
        # Load Multimodal
        if os.path.exists(self.mm_index_path) and os.path.exists(self.mm_meta_path):
            try:
                self.mm_index = faiss.read_index(self.mm_index_path)
                with open(self.mm_meta_path, "r", encoding="utf-8") as f:
                    self.mm_metadata = json.load(f)
                logger.info(f"Loaded Multimodal Index from {self.mm_index_path} (Vectors: {self.mm_index.ntotal})")
            except Exception as e:
                logger.error(f"Error loading Multimodal Index: {e}")
                success = False
        else:
            logger.warning("Multimodal Index or metadata file not found.")
            success = False
            
        # Load Text Baseline
        if os.path.exists(self.text_index_path) and os.path.exists(self.text_meta_path):
            try:
                self.text_index = faiss.read_index(self.text_index_path)
                with open(self.text_meta_path, "r", encoding="utf-8") as f:
                    self.text_metadata = json.load(f)
                logger.info(f"Loaded Text-only Baseline Index from {self.text_index_path} (Vectors: {self.text_index.ntotal})")
            except Exception as e:
                logger.error(f"Error loading Text-only Baseline Index: {e}")
                success = False
        else:
            logger.warning("Text-only Baseline Index or metadata file not found.")
            success = False
            
        return success
