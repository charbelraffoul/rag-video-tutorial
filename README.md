# Video RAG — Scene-aware retrieval for tutorial videos

Ask questions about a video tutorial and get answers with **exact timestamps** and **frame previews**, powered by a fully local stack (Whisper + Ollama + Weaviate).

---

## How it works

```
Video file
    │
    ▼
1_transcribe.py              — Whisper transcribes audio → transcript_segments.jsonl
    │
    ▼
2_scenes.py                  — OpenCV detects scene changes (SSIM) → scenes.json
                               Saves a representative frame per scene → scene_frames/
    │
    ▼
3_align_and_upload.py        — Merges Whisper segments into each scene's time window
                               Uploads Scene objects to Weaviate
    │
    ▼
4_embed.py                   — Embeds each scene's text with nomic-embed-text (Ollama)
                               Uploads SceneVec objects to Weaviate (HNSW cosine index)
    │
    ▼
rag_server.py  /  rag_ask.py — Query: embed question → nearest-vector search → LLM answer
                                        + precise timestamp via sub-segment re-ranking
```

---

## Stack

| Component | Role |
|---|---|
| [Whisper](https://github.com/openai/whisper) | Speech-to-text with timestamps |
| OpenCV (SSIM) | Visual scene detection |
| [Ollama](https://ollama.com) | Local embeddings (`nomic-embed-text`) + LLM (`llama3.2`) |
| [Weaviate](https://weaviate.io) | Vector store with HNSW cosine index |
| FastAPI | REST server + web UI |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com) installed and running

Pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 2. Start Weaviate

```bash
docker compose up -d
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local Docker setup)
```

---

## Running the pipeline

```bash
# Step 1 — transcribe your video
python 1_transcribe_timestamps.py

# Step 2 — detect scenes and extract frames
python 2_scenes.py

# Step 3 — align transcript to scenes and upload to Weaviate
python 3_align_and_upload.py

# Step 4 — embed scenes and store vectors
python 4_embed.py
```

---

## Querying

### Web UI (recommended)

```bash
python rag_server.py
# Open http://localhost:8000
```

Paste a YouTube URL into the top bar to enable "Open at Ns" timestamp links.

### Command line

```bash
python rag_ask.py "How do I connect a column to a level?"

# Open the top 3 frame images and jump to the best video timestamp:
python rag_ask.py "How do I connect a column to a level?" --open 3 --open-video 1 --video-url "https://youtube.com/watch?v=YOUR_ID"
```

---

## Project structure

```
├── 1_transcribe.py              # Whisper transcription
├── 2_scenes.py                  # Scene detection + frame extraction
├── 3_align_and_upload.py        # Transcript → scene alignment + Weaviate upload
├── 4_embed.py                   # Embed scenes with Ollama → Weaviate
├── rag_ask.py                   # CLI query tool
├── rag_server.py                # FastAPI server
├── web/index.html               # Chat UI
├── config.py                    # Loads settings from .env
├── docker-compose.yml           # Weaviate + deps
├── requirements.txt
├── .env.example                 # Copy to .env and fill in secrets
└── scene_frames/                # Auto-generated frame images (git-ignored)
```

---

## Notes

- All processing is **local** — no data leaves your machine once Whisper runs.
- The Gemini API key in `.env` is only needed if you use the older `2_embed_upload.py` path.
- `scene_frames/`, `audio.wav`, and transcript files are git-ignored (large / generated).
