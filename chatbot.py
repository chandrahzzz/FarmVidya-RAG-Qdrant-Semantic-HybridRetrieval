"""
chatbot.py  —  FarmVaidya Chatbot
==================================
Pipeline:
  User Query
    → BM25 sparse retrieval  (local index on Qdrant chunks)
    → Dense vector retrieval  (Qdrant Cloud, gemini-embedding-2)
    → RRF merge
    → Answer generation  (Vertex AI Gemini 2.5 Flash / fallback)

Run:
    streamlit run chatbot.py
"""

import os
import re
import sys
import time
import logging
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import rag_utils
from bm25 import BM25Okapi
from config import (
    GOOGLE_API_KEY,
    VERTEX_SA_PROJECT_ID, VERTEX_SA_PRIVATE_KEY_ID,
    VERTEX_SA_PRIVATE_KEY, VERTEX_SA_CLIENT_EMAIL, VERTEX_SA_CLIENT_ID,
    VERTEX_LOCATION, VERTEX_MODEL_ID,
    SARVAM_API_KEY,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, f"farmvaidya_{datetime.now().strftime('%Y-%m-%d')}.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("farmvaidya")

# ── Vertex AI auth ────────────────────────────────────────────────────────────
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GRequest


@st.cache_resource
def _vertex_creds():
    pk = VERTEX_SA_PRIVATE_KEY.replace("\\n", "\n")
    info = {
        "type": "service_account",
        "project_id": VERTEX_SA_PROJECT_ID,
        "private_key_id": VERTEX_SA_PRIVATE_KEY_ID,
        "private_key": pk,
        "client_email": VERTEX_SA_CLIENT_EMAIL,
        "client_id": VERTEX_SA_CLIENT_ID,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": (
            f"https://www.googleapis.com/robot/v1/metadata/x509/{VERTEX_SA_CLIENT_EMAIL}"
        ),
        "universe_domain": "googleapis.com",
    }
    return service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


def call_vertex_gemini(prompt: str, system_instruction: str | None = None) -> tuple[str, str]:
    """Returns (answer_text, model_used). model_used is 'vertex' or 'fallback'."""
    try:
        creds = _vertex_creds()
        if not creds.valid:
            creds.refresh(GRequest())

        url = (
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1"
            f"/projects/{VERTEX_SA_PROJECT_ID}/locations/{VERTEX_LOCATION}"
            f"/publishers/google/models/{VERTEX_MODEL_ID}:generateContent"
        )
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            return text, "vertex"
        raise Exception(f"Vertex {resp.status_code}: {resp.text[:300]}")

    except Exception as vertex_err:
        logger.warning(f"Vertex AI failed: {vertex_err} — switching to fallback")
        try:
            from google import genai
            from google.genai import types as gt

            client = genai.Client(api_key=GOOGLE_API_KEY)
            cfg    = gt.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
                system_instruction=system_instruction,
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=cfg,
            )
            return resp.text.strip(), "fallback"
        except Exception as fb_err:
            logger.error(f"Fallback also failed: {fb_err}")
            return f"[Error] Vertex: {vertex_err} | Fallback: {fb_err}", "error"


# ── BM25 index (built once per session from Qdrant chunks) ────────────────────
@st.cache_resource(show_spinner="Building keyword index…")
def get_bm25_index():
    logger.info("Building BM25 index from Qdrant chunks...")
    t0     = time.perf_counter()
    chunks = rag_utils.scroll_all_chunks()
    if not chunks:
        logger.warning("Qdrant scroll returned no chunks — falling back to muralichunks.txt")
        src = os.path.join(os.path.dirname(__file__), "muralichunks.txt")
        with open(src, encoding="utf-8") as f:
            raw = f.read()
        chunks = [ln.strip() for ln in raw.split("\n") if len(ln.strip()) > 30]
    tokenized = [c.lower().split() for c in chunks]
    elapsed   = time.perf_counter() - t0
    logger.info(f"BM25 index built: {len(chunks)} chunks in {elapsed:.2f}s")
    return BM25Okapi(tokenized), chunks


# ── LLM-based Query Normalisation ────────────────────────────────────────────
# Detects English variety codes (e.g. BPT 2782, RGL 7034) and rewrites them
# to Telugu script form so BM25 and dense retrieval can match stored chunks.
_ABBR_RE = re.compile(
    r'\b[A-Za-z]{2,5}[\s.\-]?\d{3,6}\b'       # English: BPT 2782, MTU-1010
    r'|[ఀ-౿]{3,8}\s+\d{3,6}'         # Telugu no-dots: బిపిటి 2782, యంటియు 1318
)

_NORMALISE_SYSTEM = """You are a query normalizer for a Telugu farming chatbot about rice varieties.
The knowledge base stores variety codes in Telugu script with dots between letters.
Farmers type codes in three ways — all must be normalized to the canonical dot form:

  English     Telugu (no dots)   → Canonical (with dots)       Telugu name
  BPT 2782    బిపిటి 2782        → బి.పి.టి. 2782              భవతి
  BPT 2841    బిపిటి 2841        → బి.పి.టి. 2841
  BPT 2846    బిపిటి 2846        → బి.పి.టి. 2846
  MTU 1010    యంటియు 1010        → యం.టి.యు. రైస్ 1010
  MTU 1212    యంటియు 1212        → యం.టి.యు. రైస్ 1212
  MTU 1232    యంటియు 1232        → యం.టి.యు. రైస్ 1232
  MTU 1271    యంటియు 1271        → యం.టి.యు. రైస్ 1271
  MTU 1280    యంటియు 1280        → యం.టి.యు. రైస్ 1280
  MTU 1281    యంటియు 1281        → యం.టి.యు. రైస్ 1281
  MTU 1293    యంటియు 1293        → యం.టి.యు. రైస్ 1293
  MTU 1310    యంటియు 1310        → యం.టి.యు. రైస్ 1310
  MTU 1318    యంటియు 1318        → యం.టి.యు. రైస్ 1318
  MTU 1321    యంటియు 1321        → యం.టి.యు. రైస్ 1321
  RGL 7034    ఆర్జిఎల్ 7034      → ఆర్. జి. ఎల్. 7034
  NLR 3238    ఎన్ఎల్ఆర్ 3238     → ఎన్.ఎల్.ఆర్. రైస్ 3238
  NLR 3354    ఎన్ఎల్ఆర్ 3354     → ఎన్.ఎల్.ఆర్. 3354         నెల్లూరు ధాన్యరాశి
  NLR 4001    ఎన్ఎల్ఆర్ 4001     → ఎన్.ఎల్.ఆర్. 4001         నెల్లూరు సిరి
  NLR 40054   ఎన్ఎల్ఆర్ 40054    → ఎన్.ఎల్.ఆర్. 40054        నెల్లూరు సుగంధ
  MCM 103     యంసియం 103         → యం.సి.యం. రైస్ 103

Task: Rewrite the query replacing any variety code (English or Telugu-without-dots) with the canonical dot form.
Include the Telugu variety name alongside where known.
Keep all other Telugu words exactly unchanged.
Return ONLY the rewritten query — no explanation."""


def normalize_query(raw_query: str) -> str:
    """Expand English variety codes (BPT 2782, RGL 7034 …) to Telugu script via LLM."""
    if not _ABBR_RE.search(raw_query):
        return raw_query
    normalized, _ = call_vertex_gemini(
        f"Normalize: {raw_query}", system_instruction=_NORMALISE_SYSTEM
    )
    if normalized and not normalized.startswith("[Error]"):
        logger.info(f"QUERY_NORM | '{raw_query}' → '{normalized}'")
        return normalized
    logger.warning(f"QUERY_NORM_FAIL | keeping original: '{raw_query}'")
    return raw_query


# ── pipeline helpers ──────────────────────────────────────────────────────────
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def hybrid_retrieve(query: str, top_k: int = 5) -> tuple[list[dict], dict]:
    timings: dict[str, float] = {}

    # Normalize: expand English variety codes → Telugu script before retrieval
    t_qn = time.perf_counter()
    retrieval_query = normalize_query(query)
    timings["query_norm_ms"] = round((time.perf_counter() - t_qn) * 1000, 1)

    bm25_index, raw_chunks = get_bm25_index()
    query_tokens = retrieval_query.lower().split()

    # ── Dense + BM25 in parallel, each self-timed ─────────────────────────────
    def _dense():
        t = time.perf_counter()
        hits = rag_utils.retrieve(retrieval_query, top_k + 2)
        return hits, round(time.perf_counter() - t, 4)

    def _bm25():
        t = time.perf_counter()
        scores = bm25_index.get_scores(query_tokens)
        return scores, round(time.perf_counter() - t, 4)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_dense = ex.submit(_dense)
        f_bm25  = ex.submit(_bm25)
        vector_hits,  timings["dense_s"] = f_dense.result()
        bm25_scores,  timings["bm25_s"]  = f_bm25.result()

    logger.info(
        f"RETRIEVAL_STEP | dense={timings['dense_s']}s "
        f"bm25_score={timings['bm25_s']}s | dense_hits={len(vector_hits)} "
        f"bm25_corpus={len(bm25_scores)}"
    )

    # ── BM25 normalisation: numpy vectorised min-max → [0, 1] ─────────────────
    # Chosen method: numpy vectorised min-max — single pass, no Python loop,
    # operates on the full float32 array in microseconds regardless of corpus size.
    t_norm = time.perf_counter()
    bm25_arr  = np.asarray(bm25_scores, dtype=np.float32)
    bm25_min, bm25_max = float(bm25_arr.min()), float(bm25_arr.max())
    bm25_norm = (bm25_arr - bm25_min) / (bm25_max - bm25_min + 1e-9)
    timings["norm_ms"] = round((time.perf_counter() - t_norm) * 1000, 4)

    logger.info(
        f"NORM | method=numpy_minmax | corpus={len(bm25_arr)} chunks | "
        f"raw_range=[{bm25_min:.4f}, {bm25_max:.4f}] → normalised=[0, 1] | "
        f"took {timings['norm_ms']:.4f}ms"
    )

    # ── Select top BM25 hits by normalised score ───────────────────────────────
    top_bm25_idx  = np.argsort(bm25_norm)[::-1][: top_k + 2]
    bm25_results  = [
        {"text": raw_chunks[i], "bm25_norm": float(bm25_norm[i])}
        for i in top_bm25_idx
    ]
    bm25_score_map = {raw_chunks[i]: float(bm25_norm[i]) for i in top_bm25_idx}

    # ── RRF fusion (rank-based, unaffected by normalisation) ──────────────────
    t_rrf = time.perf_counter()
    rrf_scores: dict[str, float] = {}
    for rank, hit in enumerate(bm25_results):
        txt = hit["text"]
        rrf_scores[txt] = rrf_scores.get(txt, 0) + rrf_score(rank + 1)
    for rank, hit in enumerate(vector_hits):
        txt = hit["text"]
        rrf_scores[txt] = rrf_scores.get(txt, 0) + rrf_score(rank + 1)
    timings["rrf_ms"] = round((time.perf_counter() - t_rrf) * 1000, 4)

    logger.info(
        f"RRF | candidates={len(rrf_scores)} | took {timings['rrf_ms']:.4f}ms"
    )

    merged           = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    vector_score_map = {h["text"]: h["cosine_similarity"] for h in vector_hits}

    results = []
    for text, rrf in merged[:top_k]:
        cosine  = vector_score_map.get(text)          # None if BM25-only hit
        bm25_n  = bm25_score_map.get(text, 0.0)
        # Fix: BM25-only hits no longer get a fake min_cos — they show their
        # actual normalised BM25 score so the 0.3 threshold remains meaningful.
        results.append({
            "text":             text,
            "cosine_similarity": cosine if cosine is not None else bm25_n,
            "bm25_norm":        bm25_n,
            "rrf_score":        rrf,
            "source":           "both" if cosine is not None else "bm25_only",
        })

    logger.info(
        f"HYBRID_RETRIEVE | top_k={top_k} | "
        f"both={sum(1 for r in results if r['source']=='both')} "
        f"bm25_only={sum(1 for r in results if r['source']=='bm25_only')} | "
        f"norm={timings['norm_ms']:.4f}ms rrf={timings['rrf_ms']:.4f}ms"
    )
    return results, timings


SYSTEM_PROMPT = """You are FarmVaidya, an expert farming assistant for Telugu-speaking farmers.

RULES:
1. Answer in approximately 100–150 words
2. Be specific: mention crop varieties, quantities, schedules where relevant
3. Use Telugu words for numbers (ఒకటి, రెండు కిలోలు, etc.)
4. Natural, friendly conversational Telugu — NOT formal or bookish
5. No English words unless it is a genuinely untranslatable technical term
6. No markdown headings, no bullet points — one flowing paragraph
7. No greetings or preamble — get straight to the answer
"""


# ── Sarvam AI Speech-to-Text ─────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """Send audio to Sarvam AI and return Telugu transcript."""
    try:
        ext = "webm" if "webm" in mime_type else mime_type.split("/")[-1]
        logger.info(f"STT_REQUEST | mime={mime_type} ext={ext} size={len(audio_bytes)}B")
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": (f"audio.{ext}", audio_bytes, mime_type)},
            data={"language_code": "te-IN", "model": "saarika:v2.5"},
            timeout=30,
        )
        logger.info(f"STT_RESPONSE | status={resp.status_code} | {resp.text[:300]}")
        if resp.status_code == 200:
            transcript = resp.json().get("transcript", "").strip()
            logger.info(f"STT | transcript='{transcript}'")
            return transcript
        logger.warning(f"STT_FAIL | status={resp.status_code} | {resp.text[:300]}")
    except Exception as e:
        logger.error(f"STT_ERROR | {e}")
    return ""


def generate_answer(raw_query: str, chunks: list[dict]) -> tuple[str, float, str]:
    """Returns (answer, generation_time_s, model_used)."""
    context = "\n\n".join(
        f"సమాచారం {i+1}:\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = f"ప్రశ్న: {raw_query}\n\nసందర్భం:\n{context}"
    t0 = time.perf_counter()
    answer, model_used = call_vertex_gemini(prompt, system_instruction=SYSTEM_PROMPT)
    gen_time = round(time.perf_counter() - t0, 3)
    return answer, gen_time, model_used


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmVaidya — వ్యవసాయ సహాయకుడు",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 FarmVaidya")
st.caption("వ్యవసాయ సహాయకుడు — మీ ప్రశ్నలకు తెలుగులో సమాధానాలు")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k       = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=5)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    st.divider()
    st.markdown("""
**Pipeline**
- `gemini-embedding-2` (3072-dim)
- Qdrant Cloud — us-east-1
- BM25 (min-max norm) + Dense → RRF merge
- Vertex AI Gemini 2.5 Flash (Mumbai)
    """)
    st.divider()
    st.markdown(f"📋 **Logs** → `logs/`")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("caption"):
            st.caption(msg["caption"])
        if show_chunks and msg.get("chunks"):
            with st.expander("📚 Retrieved chunks", expanded=False):
                for i, c in enumerate(msg["chunks"]):
                    src_badge = "🔵 both" if c.get("source") == "both" else "🟡 bm25-only"
                    st.markdown(
                        f"**Chunk {i+1}** {src_badge} | "
                        f"Cosine: `{c['cosine_similarity']:.4f}` | "
                        f"BM25-norm: `{c['bm25_norm']:.4f}` | "
                        f"RRF: `{c['rrf_score']:.4f}`"
                    )
                    st.text_area("", c["text"], height=80, key=f"h_{msg['ts']}_{i}")
                    st.divider()

# ── voice input ───────────────────────────────────────────────────────────────
audio = st.audio_input("🎤 మాట్లాడండి — నొక్కి మీ ప్రశ్న చెప్పండి")
audio_query = None
if audio:
    audio_bytes = audio.read()
    mime_type   = getattr(audio, "type", "audio/webm")
    audio_hash  = hash(audio_bytes)
    if st.session_state.get("_last_audio_hash") != audio_hash:
        st.session_state["_last_audio_hash"] = audio_hash
        with st.spinner("మీ మాటలు అర్థం చేసుకుంటున్నాను…"):
            audio_query = transcribe_audio(audio_bytes, mime_type)
        if audio_query:
            st.info(f"🎤 మీరు అన్నది: **{audio_query}**")
        else:
            st.warning("మాటలు అర్థం కాలేదు — దయచేసి మళ్ళీ ప్రయత్నించండి.")

# ── chat input ────────────────────────────────────────────────────────────────
text_query = st.chat_input("మీ వ్యవసాయ ప్రశ్న ఇక్కడ టైప్ చేయండి…")
query = audio_query or text_query
if query:
    ts = str(time.time())
    logger.info(f"QUERY | {query}")

    with st.spinner("సమాధానం తయారు చేస్తున్నాను…"):
        # Run the pipeline before touching session state
        t_total = time.perf_counter()
        chunks, timings = hybrid_retrieve(query, top_k=top_k)
        logger.info(
            f"RETRIEVAL | chunks={len(chunks)} | "
            f"dense={timings['dense_s']}s bm25={timings['bm25_s']}s "
            f"norm={timings['norm_ms']:.4f}ms rrf={timings['rrf_ms']:.4f}ms"
        )

        answer, gen_time, model_used = generate_answer(query, chunks)
        total_elapsed = round(time.perf_counter() - t_total, 2)
        logger.info(f"GENERATION | model={model_used} gen={gen_time}s total={total_elapsed}s")

        if not chunks or all(c["cosine_similarity"] < 0.3 for c in chunks):
            logger.warning(f"LOW_RELEVANCE | query={query} | top_cos={chunks[0]['cosine_similarity'] if chunks else 'N/A'}")

        caption = (
            f"⏱️ {total_elapsed}s | qnorm: {timings['query_norm_ms']:.1f}ms | "
            f"dense: {timings['dense_s']}s | "
            f"bm25: {timings['bm25_s']}s | norm: {timings['norm_ms']:.3f}ms | "
            f"rrf: {timings['rrf_ms']:.3f}ms | gen: {gen_time}s | "
            f"model: {model_used} | {len(chunks)} chunks"
        )

        # Persist to session state — history loop above will render on next run
        st.session_state.messages.append({"role": "user", "content": query, "ts": ts})
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "chunks": chunks, "ts": ts, "caption": caption,
        })
    st.rerun()
