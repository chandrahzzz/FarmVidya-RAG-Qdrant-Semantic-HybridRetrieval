"""
rag_utils.py — vector retrieval helpers using google-genai SDK (v2+)
"""

from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient

from config import (
    QDRANT_CONFIGS, GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
    COLLECTION_NAME,
)

_genai_client: genai.Client | None = None
_qdrant_client: QdrantClient | None = None


def _get_genai() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _genai_client


def init_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        cfg = QDRANT_CONFIGS["slumber"]
        _qdrant_client = QdrantClient(url=cfg["url"], api_key=cfg["api_key"], timeout=60)
    return _qdrant_client


def embed_query(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    client = _get_genai()
    result = client.models.embed_content(
        model=GOOGLE_EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=3072,
        ),
    )
    return result.embeddings[0].values


def retrieve(query: str, top_k: int = 6) -> list[dict]:
    """Dense vector search. Returns list of {text, cosine_similarity}."""
    from qdrant_client.models import Query
    qdrant = init_qdrant()
    vector = embed_query(query)
    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": p.payload.get("text", ""), "cosine_similarity": p.score}
        for p in result.points
    ]


def scroll_all_chunks() -> list[str]:
    """Fetch all chunk texts from Qdrant (used to build BM25 index)."""
    qdrant  = init_qdrant()
    chunks: list[str] = []
    offset  = None
    while True:
        records, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            if r.payload and "text" in r.payload:
                chunks.append(r.payload["text"])
        if offset is None:
            break
    return chunks
