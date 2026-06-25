"""Groq LLM wrapper for the agent."""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def chat(messages: list[dict]) -> str:
    """Send messages to Groq and return the assistant's response text."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content
