#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_server.py

Tiny FastAPI server for your RAG demo:
- Serves the frontend UI (web/index.html)
- Static frames under /frames/<relpath>
- POST /ask            -> runs vector search + short LLM answer
- GET  /video-url      -> returns persisted video URL (Weaviate or env)
- POST /video-url      -> saves video URL in Weaviate (VideoMeta)
- GET  /health         -> basic liveness

Environment (optional)
- WEAVIATE_URL  (default: http://127.0.0.1:8080)
- OLLAMA_URL    (default: http://localhost:11434)
- EMBED_MODEL   (default: nomic-embed-text:latest)
- LLM_MODEL     (default: llama3.2:latest)
- ROOT_DIR      (default: cwd, used to resolve frame_path)
- VIDEO_URL     (fallback if not stored in Weaviate)
- SRC_CLASS     (default: Scene)
- DST_CLASS     (default: SceneVec)

Run:
  pip install fastapi uvicorn requests
  py -3.12 rag_server.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------
# Config (from env)
# ----------------------
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080").rstrip("/")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama3.2:latest")
ROOT_DIR     = Path(os.getenv("ROOT_DIR", os.getcwd()))
VIDEO_URL_ENV = os.getenv("VIDEO_URL")

SCENE_CLASS     = os.getenv("SRC_CLASS", "Scene")
SCENE_VEC_CLASS = os.getenv("DST_CLASS", "SceneVec")

WEB_DIR    = Path(__file__).parent / "web"
INDEX_HTML = WEB_DIR / "index.html"

VIDEO_META_CLASS = "VideoMeta"
VIDEO_META_ID    = "00000000-0000-0000-0000-000000000001"

# ----------------------
# HTTP helpers
# ----------------------
def _http_post(url: str, payload: dict, timeout: int = 60, headers: Optional[dict] = None) -> dict:
    r = requests.post(url, json=payload, timeout=timeout, headers=headers)
    if not r.ok:
        raise RuntimeError(f"POST {url} -> {r.status_code} {r.text}")
    return r.json()

def _http_get(url: str, timeout: int = 60, headers: Optional[dict] = None) -> dict:
    r = requests.get(url, timeout=timeout, headers=headers)
    if not r.ok:
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.text}")
    return r.json()

def _http_put(url: str, payload: dict, timeout: int = 60, headers: Optional[dict] = None) -> dict:
    r = requests.put(url, json=payload, timeout=timeout, headers=headers)
    if not r.ok:
        raise RuntimeError(f"PUT {url} -> {r.status_code} {r.text}")
    return r.json()

def _http_patch(url: str, payload: dict, timeout: int = 60, headers: Optional[dict] = None) -> dict:
    r = requests.patch(url, json=payload, timeout=timeout, headers=headers)
    if not r.ok:
        raise RuntimeError(f"PATCH {url} -> {r.status_code} {r.text}")
    return r.json()

# ----------------------
# Ollama helpers
# ----------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama embeddings. Handles:
      {"embeddings":[...]} OR {"embedding":[...]} (older/single response)
    Falls back to per-item if the server returns a single vector for a batch.
    """
    if not texts:
        return []
    url = f"{OLLAMA_URL}/api/embeddings"

    # batch attempt
    r = requests.post(url, json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]

    if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"], list):
        # fallback per-item
        out: List[List[float]] = []
        for t in texts:
            r2 = requests.post(url, json={"model": EMBED_MODEL, "input": t}, timeout=60)
            r2.raise_for_status()
            d2 = r2.json()
            if "embedding" not in d2:
                raise RuntimeError(f"Unexpected embeddings payload: {d2}")
            out.append(d2["embedding"])
        return out

    raise RuntimeError(f"Unexpected embeddings payload keys: {list(data.keys())}")

def chat_answer(context_snippets: List[str], question: str, max_tokens: int = 256) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer succinctly and ONLY from the provided context. "
        "If the context is insufficient, say you don't know."
    )
    context_block = "\n".join(f"- {s}" for s in context_snippets)
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer briefly:"},
        ],
        "stream": False,
    }
    url = f"{OLLAMA_URL}/api/chat"
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "message" in data:
        return (data["message"].get("content") or "").strip()
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        return (data["choices"][0]["message"]["content"] or "").strip()
    return "(no answer)"

# ----------------------
# Weaviate helpers
# ----------------------
def weaviate_graphql(query: str) -> dict:
    url = f"{WEAVIATE_URL}/v1/graphql"
    r = requests.post(url, json={"query": query}, timeout=60)
    if not r.ok:
        raise RuntimeError(f"GraphQL error {r.status_code}: {r.text}")
    out = r.json()
    if "errors" in out:
        raise RuntimeError(f"GraphQL errors: {out['errors']}")
    return out["data"]

def search_scenevec(query_vec: List[float], k: int = 5) -> List[dict]:
    vec = ",".join(str(x) for x in query_vec)
    gql = f"""
    {{
      Get {{
        {SCENE_VEC_CLASS}(nearVector:{{vector:[{vec}]}}, limit:{k}) {{
          scene_id
          scene_start
          scene_end
          duration
          frame_path
          text
          _additional {{ distance }}
        }}
      }}
    }}
    """
    data = weaviate_graphql(gql)
    return data.get("Get", {}).get(SCENE_VEC_CLASS, []) or []

def get_scene_segments(scene_id: int) -> Tuple[float, float, List[dict]]:
    gql = f"""
    {{
      Get {{
        {SCENE_CLASS}(
          where: {{ path:[\"scene_id\"], operator: Equal, valueInt: {scene_id} }}
          limit: 1
        ) {{
          scene_start
          scene_end
          segments {{ start end text }}
        }}
      }}
    }}
    """
    data = weaviate_graphql(gql)
    arr = data.get("Get", {}).get(SCENE_CLASS, []) or []
    if not arr:
        return 0.0, 0.0, []
    row = arr[0]
    return float(row.get("scene_start") or 0.0), float(row.get("scene_end") or 0.0), row.get("segments") or []

def ensure_video_meta_class() -> None:
    # See if it exists
    schema = _http_get(f"{WEAVIATE_URL}/v1/schema")
    classes = [c.get("class") for c in schema.get("classes", [])]
    if VIDEO_META_CLASS in classes:
        return

    body = {
        "class": VIDEO_META_CLASS,
        "description": "Holds a single video URL for the dataset.",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "url", "dataType": ["text"], "tokenization": "word", "indexSearchable": True, "indexFilterable": True}
        ],
    }

    # Try POST /v1/schema; if it 405s, fall back to PUT /v1/schema/<Class>
    try:
        _http_post(f"{WEAVIATE_URL}/v1/schema", body, timeout=30)
    except Exception:
        _http_put(f"{WEAVIATE_URL}/v1/schema/{VIDEO_META_CLASS}", body, timeout=30)

def upsert_video_meta(url_value: str) -> None:
    ensure_video_meta_class()
    payload = {"class": VIDEO_META_CLASS, "id": VIDEO_META_ID, "properties": {"url": url_value}}
    # Try PUT (create or replace)
    try:
        _http_put(f"{WEAVIATE_URL}/v1/objects/{VIDEO_META_ID}", payload, timeout=30)
        return
    except Exception:
        pass
    # Try POST (create with id)
    try:
        _http_post(f"{WEAVIATE_URL}/v1/objects", payload, timeout=30)
        return
    except Exception:
        pass
    # Try PATCH (update)
    _http_patch(f"{WEAVIATE_URL}/v1/objects/{VIDEO_META_ID}", {"properties": {"url": url_value}}, timeout=30)

def get_video_url() -> Optional[str]:
    # Read from Weaviate if available
    try:
        data = weaviate_graphql(f'{{ Get {{ {VIDEO_META_CLASS}(limit:1) {{ url }} }} }}')
        arr = data.get("Get", {}).get(VIDEO_META_CLASS, []) or []
        if arr and arr[0].get("url"):
            return arr[0]["url"]
    except Exception:
        pass
    # Fallback to env
    return VIDEO_URL_ENV

# ----------------------
# Scoring / utilities
# ----------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def best_timestamp_for_hit(hit: dict, query_vec: List[float]) -> int:
    """
    If Scene.segments exists, embed each segment.text and choose the segment with max cosine to query_vec.
    Otherwise, use scene_start. Returns seconds (int).
    """
    try:
        scene_id = hit.get("scene_id")
        scene_start, _, segments = get_scene_segments(scene_id)
        if segments:
            seg_vecs = embed_texts([s.get("text", "") for s in segments])
            best_i = 0
            best_sim = -1.0
            for i, v in enumerate(seg_vecs):
                sim = cosine_similarity(query_vec, v)
                if sim > best_sim:
                    best_sim = sim
                    best_i = i
            ts = segments[best_i].get("start", scene_start) or scene_start
            return int(round(float(ts)))
        return int(round(float(scene_start or 0)))
    except Exception:
        try:
            return int(round(float(hit.get("scene_start") or 0.0)))
        except Exception:
            return 0

def video_link_at(url: str, seconds: int) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        u = urlparse(url)
        # YouTube: prefer ?t=
        if "youtube.com" in u.netloc or "youtu.be" in u.netloc:
            qs = parse_qs(u.query)
            qs["t"] = [str(max(0, int(seconds)))]
            new_q = urlencode(qs, doseq=True)
            return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))
        # Fallback: fragment anchor
        return f"{url}#t={max(0, int(seconds))}"
    except Exception:
        return f"{url}#t={max(0, int(seconds))}"

def frame_url_for(rel_path: str) -> str:
    # Served by our /frames/<path> route
    parts = [p for p in rel_path.split("/") if p]
    return "/frames/" + "/".join([requests.utils.quote(p) for p in parts])

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="RAG Server", version="1.0.0")

# CORS (handy during local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Models
class AskRequest(BaseModel):
    query: str
    limit: int = 5

class VideoUrlPayload(BaseModel):
    url: str

# --------- Routes
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def index():
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text(encoding="utf-8")
    # minimal fallback
    return HTMLResponse("<h1>RAG Server</h1><p>UI missing. Put it at <code>web/index.html</code>.</p>", status_code=200)

@app.get("/frames/{relpath:path}")
def get_frame(relpath: str):
    # Resolve within ROOT_DIR; forbid traversal
    safe = Path(relpath.replace("\\", "/")).parts
    resolved = ROOT_DIR.joinpath(*safe).resolve()
    try:
        resolved.relative_to(ROOT_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(resolved))

@app.get("/video-url")
def read_video_url():
    url = get_video_url()
    return {"url": url} if url else {"url": None}

@app.post("/video-url")
def save_video_url(payload: VideoUrlPayload):
    try:
        upsert_video_meta(payload.url)
        return {"ok": True, "url": payload.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(req: AskRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    k = max(1, min(int(req.limit or 5), 20))

    # 1) Embed query
    try:
        q_vec = embed_texts([query])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 2) Vector search
    try:
        hits = search_scenevec(q_vec, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # 3) Build enriched hits (sim score, frame_url, timestamp, video link)
    video_url = get_video_url()
    enriched = []
    for h in hits:
        dist = h.get("_additional", {}).get("distance", None)
        sim = None if dist is None else (1.0 - float(dist))
        frame_path = h.get("frame_path") or ""
        ts = best_timestamp_for_hit(h, q_vec)  # seconds
        vlink = video_link_at(video_url, ts) if video_url else None
        enriched.append({
            "text": (h.get("text") or "").strip(),
            "frame_path": frame_path,
            "frame_url": frame_url_for(frame_path) if frame_path else None,
            "scene_id": h.get("scene_id"),
            "scene_start": h.get("scene_start"),
            "scene_end": h.get("scene_end"),
            "duration": h.get("duration"),
            "distance": dist,
            "sim": sim,
            "timestamp": ts,
            "video_link": vlink,
        })

    # 4) Short answer via LLM using top contexts
    contexts = [e["text"] for e in enriched[:min(5, len(enriched))] if e.get("text")]
    try:
        answer = chat_answer(contexts, query) if contexts else "(no context)"
    except Exception:
        # graceful fallback
        answer = (contexts[0] if contexts else "(no answer)") if contexts else "(no answer)"

    return JSONResponse({"answer": answer, "hits": enriched})

# ------------- Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_server:app", host="127.0.0.1", port=8000, reload=False)
