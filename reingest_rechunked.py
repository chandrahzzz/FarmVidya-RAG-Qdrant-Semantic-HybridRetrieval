"""
reingest_rechunked.py
======================
Reads slumber_cache/rechunked.json (output of rechunk.py),
resets Qdrant collection murali_slumber_512, and re-uploads all chunks.
"""

import json, time
from google import genai
from google.genai import types as gt
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from config import (
    QDRANT_CONFIGS, GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
    COLLECTION_NAME, EMBED_DIM,
)

RECHUNKED = "slumber_cache/rechunked.json"

qdrant_cfg   = QDRANT_CONFIGS["slumber"]
qdrant       = QdrantClient(url=qdrant_cfg["url"], api_key=qdrant_cfg["api_key"], timeout=120)
genai_client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_one(text: str) -> list[float]:
    result = genai_client.models.embed_content(
        model=GOOGLE_EMBED_MODEL,
        contents=text,
        config=gt.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def reset_collection():
    try:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"  Deleted '{COLLECTION_NAME}'")
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"  Created '{COLLECTION_NAME}' ({EMBED_DIM}-dim COSINE)")


def upload(chunks: list[str], batch_size: int = 50):
    total  = len(chunks)
    points = []
    for idx, text in enumerate(chunks):
        try:
            vec = embed_one(text)
            points.append(PointStruct(id=idx, vector=vec, payload={"text": text}))
        except Exception as e:
            print(f"  [ERR] embed idx={idx}: {e}", flush=True)
            time.sleep(5)
            try:
                vec = embed_one(text)
                points.append(PointStruct(id=idx, vector=vec, payload={"text": text}))
            except Exception as e2:
                print(f"  [SKIP] idx={idx}: {e2}", flush=True)

        if len(points) >= batch_size or (idx == total - 1 and points):
            try:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"  [OK] {idx + 1}/{total}", flush=True)
                points = []
            except Exception as e:
                print(f"  [ERR] upsert idx={idx}: {e}", flush=True)
                time.sleep(5)


def main():
    with open(RECHUNKED, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {RECHUNKED}")

    print("Resetting Qdrant collection...")
    reset_collection()

    print(f"Uploading {len(chunks)} chunks...")
    upload(chunks)

    count = qdrant.get_collection(COLLECTION_NAME).points_count
    print(f"\n[DONE] {count} points in '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
