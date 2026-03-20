# Video RAG

> Ask questions about a video tutorial — get precise answers with **scene thumbnails** and **exact timestamps**, all running locally on your machine.

Built with Whisper, OpenCV, Ollama, and Weaviate. No cloud APIs required after setup.

---

## What it does

Most RAG systems split text into fixed-size chunks. This understands the **visual structure** of a video.

It detects every time the screen changes (a new tool opens, a panel switches, a step begins), treats each visual change as a meaningful scene, and aligns the spoken transcript to that scene. When you ask a question, it finds the most relevant scene and tells you exactly where in the video to look.

**Example:**

> _"How do I connect a column to a level?"_

The system returns:

- A natural language answer grounded in what was actually said in the video
- A thumbnail of the relevant screen
- A clickable link that opens the video **at the exact second** the answer is given

---

## How it works

```
 Your video
     │
     ▼
 1_transcribe.py
 ─────────────────────────────────────────────────────────────────
 Whisper listens to the audio and produces a transcript with
 word-level timestamps. Every sentence is tagged with start/end
 times so we know exactly when it was spoken.
     │
     ▼
 2_scenes.py
 ─────────────────────────────────────────────────────────────────
 OpenCV scans the video frame by frame and measures how different
 each frame is from the previous one (SSIM). When the difference
 crosses a threshold, a new scene begins. One representative
 frame is saved per scene as a thumbnail.
     │
     ▼
 3_align_and_upload.py
 ─────────────────────────────────────────────────────────────────
 Each Whisper segment is matched to the scene it falls inside
 (by timestamp). The full spoken text for each scene is assembled
 and uploaded to Weaviate as a Scene object.
     │
     ▼
 4_embed.py
 ─────────────────────────────────────────────────────────────────
 Each scene's text is sent to Ollama (nomic-embed-text) which
 converts it into a 768-dimensional vector — a point in space
 where semantically similar text lands close together.
 These vectors are stored in Weaviate's HNSW index.
     │
     ▼
 rag_server.py / rag_ask.py
 ─────────────────────────────────────────────────────────────────
 At query time, your question is embedded with the same model.
 Weaviate finds the closest scene vectors (cosine similarity).
 The top results are passed to a local LLM (llama3.2) which
 synthesizes a concise answer.
 For timestamps, each segment within the winning scene is
 re-embedded and the one most similar to your question is chosen
 — giving sub-scene precision.
```

---

## Tech stack

| Component                                         | What it does                                      |
| ------------------------------------------------- | ------------------------------------------------- |
| [Whisper](https://github.com/openai/whisper)      | Speech-to-text with word-level timestamps         |
| OpenCV (SSIM)                                     | Visual scene change detection                     |
| [Ollama](https://ollama.com) — `nomic-embed-text` | Converts text into semantic vectors               |
| [Ollama](https://ollama.com) — `llama3.2`         | Generates the final answer from retrieved context |
| [Weaviate](https://weaviate.io)                   | Vector database with HNSW cosine index            |
| FastAPI + Uvicorn                                 | REST API and web server                           |
| Vanilla JS + Tailwind                             | Chat UI in the browser                            |

Everything runs **locally**. Once the video is transcribed, no data leaves your machine.

---

## Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Weaviate)
- [Ollama](https://ollama.com) with the following models pulled:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/charbelraffoul/rag-video-tutorial.git
cd rag-video-tutorial
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

```bash
cp .env.example .env
# The defaults work for a local Docker setup — no changes needed
```

**4. Start Weaviate**

```bash
docker compose up -d
```

**5. Start Ollama**

```bash
ollama serve
```

---

## Running the pipeline

Run these scripts once per video, in order:

```bash
# Step 1 — Transcribe the video with Whisper
python 1_transcribe.py

# Step 2 — Detect scene changes and extract thumbnails
python 2_scenes.py

# Step 3 — Align transcript to scenes and upload to Weaviate
python 3_align_and_upload.py --video-id your-video-slug

# Step 4 — Embed scenes with Ollama and store vectors
python 4_embed.py --video-id your-video-slug
```

Each step builds on the previous one. After step 4, your data is in Weaviate and ready to query.

---

## Querying

### Web UI

```bash
python rag_server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

- Type your question and press **Ask**
- Each answer shows the relevant scene thumbnail and transcript snippet
- Paste your YouTube URL in the **Settings** panel and click **Save** — this enables the **"Open video @ time"** button which jumps to the exact second in the video

### Command line

```bash
python rag_ask.py "How do I connect a column to a level?"

# Retrieve top 5 results, open the best frame image, and jump to the video timestamp
python rag_ask.py "How do I connect a column to a level?" -k 5 --open 1 --open-video 1 --video-url "https://youtube.com/watch?v=YOUR_ID"
```

---

## Project structure

```
rag-video-tutorial/
│
├── 1_transcribe.py          # Whisper transcription with timestamps
├── 2_scenes.py              # Scene detection + frame extraction
├── 3_align_and_upload.py    # Align transcript → scenes, upload to Weaviate
├── 4_embed.py               # Embed scenes with Ollama, store vectors
│
├── rag_server.py            # FastAPI server (serves UI + /ask API)
├── rag_ask.py               # Command-line query tool
├── web/
│   └── index.html           # Chat UI
│
├── config.py                # Loads settings from .env
├── docker-compose.yml       # Weaviate container
├── requirements.txt
├── .env.example             # Copy to .env and fill in your values
└── .gitignore
```

---

## How the timestamp precision works

A scene can be 30–90 seconds long. Instead of linking to the start of the scene, the system finds the **exact sentence** that best answers the question:

1. Retrieve the top matching scene via vector search
2. Fetch all individual Whisper segments (5–10 second chunks) within that scene
3. Embed each segment with the same Ollama model
4. Compute cosine similarity between each segment and the query
5. Return the timestamp of the best-matching segment

This means the video link lands within a few seconds of the actual answer.
