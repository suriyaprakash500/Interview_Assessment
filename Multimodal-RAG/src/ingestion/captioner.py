import os
import logging
from typing import Optional
from src.ingestion.providers import GroqVisionProvider

logger = logging.getLogger(__name__)

class QwenVLDecoder:
    """
    Captioner class that uses GroqVisionProvider to caption page images.
    Aligns with API-only hosted vision requirements.
    """
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, 
                 model_name: Optional[str] = None, mock_mode: bool = False):
        # We ignore mock_mode parameter and strictly use GroqVisionProvider
        self.provider = GroqVisionProvider(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name
        )
        self.model_name = self.provider.get_model_name()
        self.mock_mode = False  # Mock mode is completely disabled

    def _encode_image(self, image_path: str) -> str:
        return self.provider._encode_image(image_path)

    def generate_caption(self, image_path: str, page_num: int, page_text: str) -> str:
        """
        Delegates caption generation to the Groq vision provider.
        """
        return self.provider.generate_caption(image_path, page_num, page_text)
