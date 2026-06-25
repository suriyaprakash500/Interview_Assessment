import os
import time
import base64
import logging
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class VisionProvider(ABC):
    """
    Abstract Base Class for visual page captioning providers.
    """
    @abstractmethod
    def generate_caption(self, image_path: str, page_num: int, page_text: str) -> str:
        """
        Generates visual summary description for a rendered PDF page image.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Returns the model name used by the provider.
        """
        pass


class GroqVisionProvider(VisionProvider):
    """
    Connects to Groq's Vision API using an OpenAI-compatible completion client
    and generates visual summary captions using hosted vision-language models.
    """
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.api_base = api_base or os.getenv("QWEN_API_BASE", "https://api.groq.com/openai/v1")
        self.model_name = model_name or os.getenv("QWEN_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        
        if not self.api_key:
            logger.error("Groq API Key (QWEN_API_KEY) is missing from environment variables.")
            raise ValueError("API Key (QWEN_API_KEY) is required to run the Groq vision pipeline. Please add it to your .env file.")
            
        logger.info(f"Initializing Groq Vision Provider (Endpoint: {self.api_base}, Model: {self.model_name})")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_caption(self, image_path: str, page_num: int, page_text: str) -> str:
        """
        Generates visual details, tables, and chart descriptions from the page image using Groq API.
        """
        try:
            base64_image = self._encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to read/encode image at {image_path}: {e}")
            raise e

        prompt = (
            "You are analyzing a page from TVS Motor Company's Annual Report. "
            "Identify and describe the layout and visual content of this page. "
            "If the page contains any charts, graphs, tables, or infographics, provide:\n"
            "1. A detailed visual description.\n"
            "2. A summary of the charts/graphs, including exact data values, trends, titles, and legends.\n"
            "3. A structured summary of any tables, listing rows, columns, and important figures (such as revenue, profit, percentages).\n"
            "If it is a text-heavy page, describe the visual styling, layout formatting, any callout boxes, and key highlighted figures.\n"
            "Be extremely precise with numbers, labels, and text. Provide a comprehensive summary."
        )

        max_retries = 5
        backoff = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                caption = response.choices[0].message.content.strip()
                logger.info(f"Generated vision caption for page {page_num} successfully using {self.model_name}.")
                return caption
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for page {page_num}: {e}")
                
                # Check for rate limit / 429
                error_msg = str(e).lower()
                is_rate_limit = "rate limit" in error_msg or "429" in error_msg
                wait_time = backoff
                
                if is_rate_limit:
                    import re
                    # Match "try again in 6m49.536s" or "try again in 10m" or "try again in 15s"
                    match = re.search(r"try again in\s+(?:(\d+)m)?(?:([\d.]+)s)?", error_msg)
                    if match:
                        minutes = float(match.group(1)) if match.group(1) else 0.0
                        seconds = float(match.group(2)) if match.group(2) else 0.0
                        wait_time = int(minutes * 60 + seconds) + 5  # Add a 5-second buffer
                        logger.info(f"Parsed rate limit wait time of {wait_time} seconds from error message.")
                    else:
                        wait_time = 60  # Default to 60 seconds if parsing fails
                    
                    logger.info(f"Rate limit hit. Sleeping for {wait_time} seconds before retrying...")
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    if not is_rate_limit:
                        backoff *= 2
                else:
                    logger.error(f"All Groq Vision API retries failed for page {page_num}.")
                    raise e

    def get_model_name(self) -> str:
        return self.model_name
