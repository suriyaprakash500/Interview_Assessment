import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from src.embeddings.embedder import DocumentEmbedder
from src.embeddings.indexer import FAISSIndexer
from src.ingestion.captioner import QwenVLDecoder

logger = logging.getLogger(__name__)

class SearchCoordinator:
    """
    Coordinates retrieval queries (text or image) against both the 
    Multimodal Index and the Text-only Baseline Index.
    """
    def __init__(self, config: Dict[str, Any], embedder: DocumentEmbedder, indexer: FAISSIndexer):
        self.config = config
        self.embedder = embedder
        self.indexer = indexer
        self.top_k = config.get("top_k", 5)
        
        # Initialize a Qwen-VL/Groq decoder for query image captioning
        self.decoder = QwenVLDecoder()

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _search_index(self, index, metadata: List[Dict[str, Any]], query_emb: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Executes search on a single FAISS index and returns matching page details.
        """
        if index is None or not metadata:
            logger.warning("Attempted search on an uninitialized index.")
            return []
            
        # Ensure query embedding is 2D float32
        query_emb_2d = query_emb.reshape(1, -1).astype(np.float32)
        query_emb_2d = self._normalize_vector(query_emb_2d)
        
        # Search a larger number of candidates to allow for page-level deduplication
        search_k = min(index.ntotal, top_k * 4)
        if search_k <= 0:
            return []
            
        distances, indices = index.search(query_emb_2d, search_k)
        
        results = []
        seen_pages = set()
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
                
            meta = metadata[idx]
            page_num = meta["page_number"]
            
            # Deduplicate by page number so we return Top K unique pages
            if page_num in seen_pages:
                continue
                
            seen_pages.add(page_num)
            
            results.append({
                "document_name": meta["document_name"],
                "page_number": page_num,
                "similarity_score": float(dist),
                "matching_snippet": meta["snippet"],
                "page_image_path": meta["page_image_path"],
                "match_type": meta.get("type", "unknown")
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def search_by_text(self, query_text: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float]:
        """
        Searches both indexes using a text query.
        Returns:
            - Multimodal results (List)
            - Text-only baseline results (List)
            - Text index retrieval latency in seconds (float)
            - Multimodal index retrieval latency in seconds (float)
        """
        k = top_k or self.top_k
        logger.info(f"Executing search for text query: '{query_text}' (K={k})")
        
        # 1. Embed query
        query_emb = self.embedder.embed_queries([query_text])[0]
        
        # 2. Search Multimodal Index & time it
        t_mm_start = time.time()
        mm_results = self._search_index(self.indexer.mm_index, self.indexer.mm_metadata, query_emb, k)
        mm_latency = time.time() - t_mm_start
        
        # 3. Search Text Baseline Index & time it
        t_text_start = time.time()
        text_results = self._search_index(self.indexer.text_index, self.indexer.text_metadata, query_emb, k)
        text_latency = time.time() - t_text_start
        
        logger.info(f"Search completed. MM Latency: {mm_latency:.4f}s, Text Latency: {text_latency:.4f}s")
        
        return mm_results, text_results, text_latency, mm_latency

    def search_by_image(self, image_path: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float, float, str]:
        """
        Searches both indexes using an uploaded query image.
        The image is captioned using the Groq API provider, and the caption is used to perform text-based retrieval.
        Returns:
            - Multimodal results (List)
            - Text-only baseline results (List)
            - Text index retrieval latency in seconds (float)
            - Multimodal index retrieval latency in seconds (float)
            - End-to-end latency in seconds (float)
            - Generated query image caption (str)
        """
        k = top_k or self.top_k
        logger.info(f"Executing search for image query: '{image_path}' (K={k})")
        
        start_time = time.time()
        
        # 1. Generate query image caption via Groq Vision API
        query_prompt = (
            "Analyze this query image. Describe the key items, charts, graphs, labels, titles, "
            "numbers, or textual information shown in the image in detail. "
            "Your description will be used as a query to find matching pages in a document database."
        )
        
        try:
            base64_image = self.decoder._encode_image(image_path)
            response = self.decoder.provider.client.chat.completions.create(
                model=self.decoder.provider.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            query_caption = response.choices[0].message.content.strip()
            logger.info(f"Generated vision caption for query image successfully: '{query_caption[:100]}...'")
        except Exception as e:
            logger.error(f"Failed to caption query image via hosted vision API: {e}")
            raise e

        # 2. Run retrieval using query caption
        mm_results, text_results, text_latency, mm_latency = self.search_by_text(query_caption, top_k=k)
        
        total_latency = time.time() - start_time
        return mm_results, text_results, text_latency, mm_latency, total_latency, query_caption
