import os
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Wraps text embedding models to generate vector embeddings.
    Supports local SentenceTransformers (offline, free) and OpenAI Embeddings.
    """
    def __init__(self, provider: str = "local", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.provider = provider.lower()
        self.model_name = model_name
        
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment. Falling back to local Sentence-Transformers.")
                self.provider = "local"
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self._init_local()
            else:
                logger.info(f"Initializing OpenAI Embeddings client (Model: {self.model_name})")
                self.client = OpenAI(api_key=api_key)
        else:
            self._init_local()

    def _init_local(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Initializing local SentenceTransformers model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}. Retrying with default 'all-MiniLM-L6-v2'.")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
            
        # Clean inputs
        cleaned_texts = [text.strip() if text else "empty" for text in texts]
        
        if self.provider == "openai":
            try:
                response = self.client.embeddings.create(
                    input=cleaned_texts,
                    model=self.model_name
                )
                embeddings = [data.embedding for data in response.data]
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"OpenAI embedding generation failed: {e}. Falling back to mock embeddings.")
                # Return random vector as fallback to avoid crashing
                return np.random.randn(len(texts), 1536).astype(np.float32)
        else:
            try:
                embeddings = self.model.encode(
                    cleaned_texts, 
                    batch_size=32, 
                    show_progress_bar=False, 
                    convert_to_numpy=True
                )
                # Ensure they are float32
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"Local embedding generation failed: {e}")
                raise e

    def get_embedding_dim(self) -> int:
        """
        Returns the embedding dimension of the model.
        """
        if self.provider == "openai":
            # text-embedding-3-small is 1536, text-embedding-ada-002 is 1536
            return 1536
        else:
            return self.model.get_sentence_embedding_dimension()
