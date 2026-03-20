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

Multiple videos are supported — index as many as you want and query across all of them at once.

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
 and uploaded to Weaviate as a Scene object, tagged with a
 video_id. The video's YouTube URL is saved here too, enabling
 timestamped links in the UI.
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
 rag_server.py
 ─────────────────────────────────────────────────────────────────
 At query time, your question is embedded with the same model.
 Weaviate finds the closest scene vectors (cosine similarity).
 Neighboring scenes are also fetched to capture procedural steps
 that follow problem-description scenes.
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
| Vanilla JS                                        | Chat UI in the browser                            |

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

Run these four scripts once per video, in order. Everything is namespaced by `--video-id` so multiple videos never conflict.

```bash
# Step 1 — Transcribe the video with Whisper
python 1_transcribe.py --video-path "path/to/video.mp4" --video-id your-video-slug

# Step 2 — Detect scene changes and extract thumbnails
python 2_scenes.py --video-path "path/to/video.mp4" --video-id your-video-slug

# Step 3 — Align transcript to scenes, upload to Weaviate, save YouTube URL
python 3_align_and_upload.py --video-id your-video-slug --video-url "https://youtube.com/watch?v=..."

# Step 4 — Embed scenes with Ollama and store vectors
python 4_embed.py --video-id your-video-slug
```

Steps 1 and 2 automatically name their output files after the `--video-id`, so you can safely run the pipeline for multiple videos without overwriting anything.

After step 4, your data is in Weaviate and ready to query.

---

## Querying

### Web UI

```bash
python rag_server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

- Type your question and press **Ask**
- Each result shows a scene thumbnail, the transcript excerpt, similarity score, and timestamp
- Click **Open video @ time** to jump to the exact second in the YouTube video
- Click **Open image** to view the full-resolution frame

The **Settings** panel (top right) lets you manage saved video URLs manually if needed.

### Command line

```bash
python rag_ask.py "How do I connect a column to a level?"

# Top 5 results, open the best frame, jump to timestamp
python rag_ask.py "How do I connect a column to a level?" -k 5 --open 1 --open-video 1 --video-url "https://youtube.com/watch?v=YOUR_ID"
```

---

## Adding more videos

The pipeline is fully multi-video. Just run all four steps with a new `--video-id` for each video:

```bash
python 1_transcribe.py --video-path "video2.mp4" --video-id second-video
python 2_scenes.py     --video-path "video2.mp4" --video-id second-video
python 3_align_and_upload.py --video-id second-video --video-url "https://youtube.com/watch?v=..."
python 4_embed.py      --video-id second-video
```

All videos share the same Weaviate index. Queries automatically search across all of them and each result card shows which video it came from.

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
├── docker-compose.yml       # Weaviate container
├── requirements.txt
├── .env.example             # Copy to .env and fill in your values
└── .gitignore
```

---

## How the timestamp precision works

A scene can be 30–90 seconds long. Instead of linking to the start of the scene, the system finds the **exact sentence** that best answers the question:

1. Retrieve the top matching scenes via vector search
2. Fetch neighboring scenes to capture procedural steps that follow problem descriptions
3. Pass all context to the LLM to synthesize an answer
4. For the timestamp, fetch all individual Whisper segments within the best scene
5. Embed each segment and compute cosine similarity against the query
6. Return the timestamp of the best-matching segment

This means the video link lands within a few seconds of the actual answer.
