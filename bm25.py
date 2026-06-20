"""
Thin wrapper around rank_bm25.BM25Okapi so the rest of the code
can do `from bm25 import BM25Okapi`.
"""
from rank_bm25 import BM25Okapi  # noqa: F401
