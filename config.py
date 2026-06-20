import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Generative AI (new SDK: google-genai) ─────────────────────────────
GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_EMBED_MODEL    = os.getenv("GOOGLE_EMBED_MODEL", "gemini-embedding-001")
GOOGLE_CLOUD_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# ── Qdrant Cloud ─────────────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

QDRANT_CONFIGS = {
    "slumber": {
        "url":     QDRANT_URL,
        "api_key": QDRANT_API_KEY,
    }
}

# ── Vertex AI (Gemini 2.5 Flash via service account) ─────────────────────────
VERTEX_SA_PROJECT_ID     = os.getenv("VERTEX_SA_PROJECT_ID", "")
VERTEX_SA_PRIVATE_KEY_ID = os.getenv("VERTEX_SA_PRIVATE_KEY_ID", "")
VERTEX_SA_PRIVATE_KEY    = os.getenv("VERTEX_SA_PRIVATE_KEY", "")
VERTEX_SA_CLIENT_EMAIL   = os.getenv("VERTEX_SA_CLIENT_EMAIL", "")
VERTEX_SA_CLIENT_ID      = os.getenv("VERTEX_SA_CLIENT_ID", "")

# ── Chunking / ingestion ──────────────────────────────────────────────────────
CHUNK_SIZE      = 512
CHUNK_SIZES     = [512]          # list for compatibility
SOURCE_FILE     = "muralichunks.txt"
COLLECTION_NAME = "murali_slumber_512"
EMBED_DIM       = 3072

VERTEX_LOCATION = "asia-south1"
VERTEX_MODEL_ID = "gemini-2.5-flash"
