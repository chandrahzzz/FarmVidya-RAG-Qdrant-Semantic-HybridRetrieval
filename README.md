# FarmVaidya — వ్యవసాయ సహాయకుడు

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers Telugu farmers' questions about rice cultivation in natural, conversational Telugu. Built with a hybrid retrieval pipeline, LLM-based query normalization, and Gemini 2.5 Flash for generation.

---

## What I Built

FarmVaidya is a domain-specific RAG chatbot designed for Telugu-speaking farmers in Andhra Pradesh. Farmers can ask questions in Telugu — including using English variety codes like `BPT 2782` or spoken-form numbers like `ఏడు సున్నా మూడు నాలుగు` — and get accurate, friendly answers about rice varieties, crop seasons, seed treatment, nutrient management, and more.

The core challenge was bridging the vocabulary gap between how farmers naturally type queries (mixed Telugu/English, abbreviations without dots, word-form numbers) and how the knowledge base stores data (Telugu script with dot-separated abbreviations like `బి.పి.టి. 2782`). I solved this with an LLM-based query normalization step before retrieval.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM (Generation)** | Vertex AI Gemini 2.5 Flash (primary), Gemini 2.5 Flash API (fallback) |
| **Embeddings** | Google `gemini-embedding-001` (3072-dim) |
| **Vector Database** | Qdrant Cloud (us-east-1) |
| **Keyword Search** | BM25 (Okapi BM25, min-max normalised) |
| **Retrieval Strategy** | Hybrid — Dense + BM25 → RRF merge |
| **Query Normalization** | LLM-based (Gemini) |
| **Chunking Strategy** | Semantic Q&A pair chunking |
| **Auth** | Google Service Account (Vertex AI), API Key (Gemini, Qdrant) |
| **Language** | Python 3.11 |

---

## Architecture

```
User Query (Telugu / English / Mixed)
        │
        ▼
┌─────────────────────┐
│  Query Normalizer   │  ← LLM rewrites English codes (BPT 2782)
│  (Gemini LLM)       │    to Telugu script (బి.పి.టి. 2782 భవతి)
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│           Hybrid Retrieval               │
│                                          │
│  ┌─────────────────┐  ┌───────────────┐  │
│  │  Dense Search   │  │  BM25 Search  │  │
│  │ (Qdrant Cloud)  │  │ (local index) │  │
│  │ gemini-embed    │  │ min-max norm  │  │
│  └────────┬────────┘  └──────┬────────┘  │
│           └────────┬─────────┘           │
│                    ▼                     │
│             RRF Fusion                   │
│         (Reciprocal Rank Fusion)         │
└──────────────────────────────────────────┘
        │
        ▼  Top-5 chunks
┌─────────────────────┐
│  Answer Generation  │  ← Vertex AI Gemini 2.5 Flash
│  (Telugu response)  │    with farming system prompt
└─────────────────────┘
```

---

## Methods & Design Decisions

### 1. Semantic Q&A Chunking
Instead of fixed word-count chunking (128/256/512 words), I chunk the knowledge base by Q&A pairs. Each chunk = one question + its complete answer. This ensures every chunk is a self-contained fact about a single topic (one rice variety, one practice), giving the LLM clean, focused context.

### 2. Hybrid Retrieval (Dense + BM25 → RRF)
- **Dense retrieval** — `gemini-embedding-001` embeds the query and searches Qdrant by cosine similarity. Handles semantic understanding.
- **BM25** — keyword-based sparse retrieval on the same chunks, with min-max normalisation. Handles exact term matches.
- **RRF (Reciprocal Rank Fusion)** — combines ranks from both retrievers without needing score normalisation across different scales.

### 3. LLM-Based Query Normalisation
Farmers type variety codes in three different ways:
- English: `BPT 2782`
- Telugu without dots: `బిపిటి 2782`
- Telugu word-numbers: `బిపిటి ఏడు సున్నా మూడు నాలుగు`

The knowledge base stores them as: `బి.పి.టి. 2782`

A regex detects English or Telugu abbreviation + number patterns in the query. If detected, Gemini rewrites the query to the canonical dot-separated Telugu form before retrieval runs. Queries without codes skip this step entirely (zero overhead).

### 4. Qdrant Cloud as Vector Store
All 124 Q&A chunks are embedded (3072-dim) and stored in Qdrant Cloud. The BM25 index is built at startup by scrolling all chunks from Qdrant — so there is a single source of truth.

---

## Project Structure

```
FARMvaidyaCHATBOT/
├── chatbot.py              # Main app — pipeline + Streamlit UI
├── rag_utils.py            # Qdrant retrieval + Gemini embeddings
├── config.py               # All config loaded from .env
├── bm25.py                 # BM25 Okapi implementation
├── semantic512.py          # Chunking script → slumber_cache/rechunked.json
├── reingest_rechunked.py   # Embeds chunks → uploads to Qdrant
├── muralichunks.txt        # Raw knowledge base (rice cultivation Q&A)
├── requirements.txt
├── .env.example            # Template for required environment variables
├── .gitignore
└── slumber_cache/
    └── rechunked.json      # 124 semantic Q&A chunks (committed to repo)
```

---

## Setup — Step by Step

### Prerequisites
- Python 3.11+
- A Qdrant Cloud account (free tier works)
- A Google Cloud project with Vertex AI enabled
- A Google AI Studio API key (for embeddings + fallback LLM)

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/your-username/FARMvaidyaCHATBOT.git
cd FARMvaidyaCHATBOT
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
GOOGLE_EMBED_MODEL=gemini-embedding-001

QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

VERTEX_SA_PROJECT_ID=your_gcp_project_id
VERTEX_SA_PRIVATE_KEY_ID=your_key_id
VERTEX_SA_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
VERTEX_SA_CLIENT_EMAIL=your-sa@your-project.iam.gserviceaccount.com
VERTEX_SA_CLIENT_ID=your_client_id
```

### Step 5 — Upload chunks to Qdrant

The `slumber_cache/rechunked.json` file is already in the repo. Just run:

```bash
python reingest_rechunked.py
```

This embeds all 124 Q&A chunks using `gemini-embedding-001` and uploads them to your Qdrant collection `murali_slumber_512`. Takes ~2-3 minutes.

### Step 6 — Run the app

```bash
streamlit run chatbot.py
```

Open **http://localhost:8501** in your browser.

---

## Re-chunking (if knowledge base changes)

If you update `muralichunks.txt` with new Q&A content:

```bash
# Step 1: Re-chunk
python semantic512.py

# Step 2: Re-ingest into Qdrant
python reingest_rechunked.py
```

---

## Example Queries

```
బిపిటి 2782 వరి రకం గురించి చెప్తారా?
ఆర్జిఎల్ 7034 రకం ఏ దశలో ఉంది?
సార్వా కాలంలో కృష్ణా మండలానికి అనువైన రకాలు ఏవి?
వరి విత్తన శుద్ధి ఎలా చేయాలి?
MTU 1318 రకం దిగుబడి ఎంత?
```

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio key — used for embeddings and fallback LLM |
| `GOOGLE_EMBED_MODEL` | Embedding model (default: `gemini-embedding-001`) |
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `VERTEX_SA_PROJECT_ID` | GCP project ID for Vertex AI |
| `VERTEX_SA_PRIVATE_KEY` | Service account private key (PEM format) |
| `VERTEX_SA_CLIENT_EMAIL` | Service account email |
| `VERTEX_SA_PRIVATE_KEY_ID` | Key ID from service account JSON |
| `VERTEX_SA_CLIENT_ID` | Client ID from service account JSON |

---

## Known Limitations

- **124 Q&A chunks only** — answers are limited to what's in `muralichunks.txt`. Questions outside this scope will get a "don't know" response.
- **Telugu word-form numbers** (e.g., `ఏడు సున్నా మూడు నాలుగు` for 7034) rely on dense retrieval since BM25 can't match them lexically — works but cosine scores are lower.
- **Vertex AI auth** requires a valid service account PEM key. If misconfigured, the app automatically falls back to the Gemini API key.
