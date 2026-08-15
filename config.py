"""
config.py — Environment variable loading and validation.

Loads required configuration from environment variables (with optional
.env file support via python-dotenv for local development).
"""

import os

from dotenv import load_dotenv


def load_config() -> dict[str, str]:
    """
    Read required environment variables.
    Returns a dict with key 'GROQ_API_KEY'.
    Raises EnvironmentError with the variable name if any are missing.
    """
    # Load .env file if present (no-op if the file doesn't exist)
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY")

    return {"GROQ_API_KEY": api_key}
