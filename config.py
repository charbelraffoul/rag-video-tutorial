# config.py
#
# Loads configuration from a .env file (never commit .env to source control).
# Copy .env.example -> .env and fill in your values.

from dotenv import load_dotenv
import os

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Weaviate
# ──────────────────────────────────────────────────────────────
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_CLASS   = os.getenv("WEAVIATE_CLASS", "Scene")

# ──────────────────────────────────────────────────────────────
# Local data
# ──────────────────────────────────────────────────────────────
TRANSCRIPT_PATH  = os.getenv("TRANSCRIPT_PATH", "transcript.txt")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))

# ──────────────────────────────────────────────────────────────
# Gemini (Google Generative AI) — only needed for 2_embed_upload.py
# ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-pro")
