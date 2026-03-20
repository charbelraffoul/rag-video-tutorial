#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_server.py

FastAPI server for the Video RAG demo:
- Serves the frontend UI (web/index.html)
- Static frames under /frames/<relpath>
- POST /ask            -> vector search + LLM answer
- GET  /video-url      -> returns all saved {video_id, url} pairs
- POST /video-url      -> saves a {video_id, url} pair in Weaviate
- GET  /health         -> liveness check

Environment (optional):
  WEAVIATE_URL  (default: http://127.0.0.1:8080)
  OLLAMA_URL    (default: http://localhost:11434)
  EMBED_MODEL   (default: nomic-embed-text:latest)
  LLM_MODEL     (default: llama3.2:latest)
  ROOT_DIR      (default: cwd, used to resolve frame_path)
  VIDEO_URL     (fallback URL when no VideoMeta found for a video_id)

Run:
  pip install fastapi uvicorn requests
  python rag_server.py
"""

from __future__ import annotations

import math
import os
import uuid as _uuid
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

WEAVIATE_URL  = os.getenv("WEAVIATE_URL",  "http://127.0.0.1:8080").rstrip("/")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434").rstrip("/")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text:latest")
LLM_MODEL     = os.getenv("LLM_MODEL",     "llama3.2:latest")
ROOT_DIR      = Path(os.getenv("ROOT_DIR", os.getcwd()))
VIDEO_URL_ENV = os.getenv("VIDEO_URL")

SCENE_CLASS     = os.getenv("SRC_CLASS", "Scene")
SCENE_VEC_CLASS = os.getenv("DST_CLASS", "SceneVec")
VIDEO_META_CLASS = "VideoMeta"

WEB_DIR    = Path(__file__).parent / "web"
INDEX_HTML = WEB_DIR / "index.html"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(url, payload, timeout=60):
    r = requests.post(url, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"POST {url} -> {r.status_code} {r.text}")
    return r.json()

def _get(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.text}")
    return r.json()

def _put(url, payload, timeout=60):
    r = requests.put(url, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"PUT {url} -> {r.status_code} {r.text}")
    return r.json()

def _patch(url, payload, timeout=60):
    r = requests.patch(url, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"PATCH {url} -> {r.status_code} {r.text}")
    return r.json()


# ── Schema helpers ────────────────────────────────────────────────────────────

def ensure_property(class_name: str, prop_name: str, data_type: list) -> None:
    """Add a property to an existing class only if it is not already present."""
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/schema/{class_name}", timeout=10)
        if not r.ok:
            return
        existing = [p["name"] for p in r.json().get("properties", [])]
        if prop_name in existing:
            return
        body = {"name": prop_name, "dataType": data_type,
                "tokenization": "field", "indexSearchable": True, "indexFilterable": True}
        requests.post(f"{WEAVIATE_URL}/v1/schema/{class_name}/properties",
                      json=body, timeout=20)
    except Exception:
        pass  # non-fatal; class or Weaviate may not be ready yet


def ensure_video_meta_class() -> None:
    schema = _get(f"{WEAVIATE_URL}/v1/schema")
    classes = [c.get("class") for c in schema.get("classes", [])]
    if VIDEO_META_CLASS in classes:
        ensure_property(VIDEO_META_CLASS, "video_id", ["text"])
        return

    body = {
        "class": VIDEO_META_CLASS,
        "description": "One object per video, storing its id and YouTube URL.",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "video_id", "dataType": ["text"], "tokenization": "field",
             "indexSearchable": True, "indexFilterable": True},
            {"name": "url",      "dataType": ["text"], "tokenization": "word",
             "indexSearchable": True, "indexFilterable": True},
        ],
    }
    try:
        _post(f"{WEAVIATE_URL}/v1/schema", body, timeout=30)
    except Exception:
        _put(f"{WEAVIATE_URL}/v1/schema/{VIDEO_META_CLASS}", body, timeout=30)


# ── VideoMeta CRUD ────────────────────────────────────────────────────────────

def _video_meta_uuid(video_id: str) -> str:
    """Deterministic UUID per video_id so upserts are idempotent."""
    return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"video-meta-{video_id}"))


def upsert_video_meta(video_id: str, url_value: str) -> None:
    ensure_video_meta_class()
    obj_id = _video_meta_uuid(video_id)
    payload = {
        "class": VIDEO_META_CLASS,
        "id": obj_id,
        "properties": {"video_id": video_id, "url": url_value},
    }
    try:
        _put(f"{WEAVIATE_URL}/v1/objects/{obj_id}", payload, timeout=30)
        return
    except Exception:
        pass
    try:
        _post(f"{WEAVIATE_URL}/v1/objects", payload, timeout=30)
        return
    except Exception:
        pass
    _patch(f"{WEAVIATE_URL}/v1/objects/{obj_id}",
           {"properties": {"video_id": video_id, "url": url_value}}, timeout=30)


def get_all_videos() -> List[dict]:
    """Return all stored {video_id, url} pairs."""
    try:
        data = weaviate_graphql(
            f'{{ Get {{ {VIDEO_META_CLASS}(limit:100) {{ video_id url }} }} }}'
        )
        arr = data.get("Get", {}).get(VIDEO_META_CLASS, []) or []
        return [{"video_id": r.get("video_id"), "url": r.get("url")} for r in arr]
    except Exception:
        return []


def get_video_url_for(video_id: Optional[str]) -> Optional[str]:
    """Look up the URL for a specific video_id. Falls back to VIDEO_URL env var."""
    if not video_id:
        return VIDEO_URL_ENV
    try:
        safe = video_id.replace('"', '')
        data = weaviate_graphql(f"""
        {{
          Get {{
            {VIDEO_META_CLASS}(
              where: {{ path:["video_id"], operator: Equal, valueText: "{safe}" }}
              limit: 1
            ) {{ url }}
          }}
        }}
        """)
        arr = data.get("Get", {}).get(VIDEO_META_CLASS, []) or []
        if arr and arr[0].get("url"):
            return arr[0]["url"]
    except Exception:
        pass
    return VIDEO_URL_ENV


# ── Ollama helpers ────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    safe = [(t or " ").strip() for t in texts]
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": safe}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data and len(data["embeddings"]) == len(safe):
        return data["embeddings"]
    raise RuntimeError(f"Unexpected /api/embed response: {list(data.keys())}")


def chat_answer(context_snippets: List[str], question: str) -> str:
    system = (
        "You are an assistant that answers questions about a video tutorial. "
        "You are given transcript excerpts from the video. "
        "Synthesize a direct, confident answer from those excerpts. "
        "Present every relevant step or detail you find — do not omit steps that are mentioned. "
        "Do not add steps that are not in the context. "
        "Do not hedge or say the context is incomplete. Just answer with what is there."
    )
    context = "\n\n---\n\n".join(context_snippets)
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "message" in data:
        return (data["message"].get("content") or "").strip()
    if "choices" in data and data["choices"]:
        return (data["choices"][0]["message"]["content"] or "").strip()
    return "(no answer)"


# ── Weaviate helpers ──────────────────────────────────────────────────────────

def weaviate_graphql(query: str) -> dict:
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=60)
    if not r.ok:
        raise RuntimeError(f"GraphQL {r.status_code}: {r.text}")
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
          video_id
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


def fetch_neighboring_scenes(scene_ids: List[int], window: int = 2) -> List[dict]:
    """
    For each scene_id in the list, fetch the next `window` scenes by scene_id.
    Returns a deduplicated list of {scene_id, text} sorted by scene_id.
    """
    if not scene_ids:
        return []
    # Build a set of all IDs to fetch (originals + neighbors)
    all_ids = set()
    for sid in scene_ids:
        for offset in range(1, window + 1):
            all_ids.add(sid + offset)
    # Remove IDs already in the original hits
    all_ids -= set(scene_ids)
    if not all_ids:
        return []

    # Fetch all in one GraphQL query using a GreaterThan / LessThan range
    min_id = min(all_ids)
    max_id = max(all_ids)
    gql = f"""
    {{
      Get {{
        {SCENE_VEC_CLASS}(
          where: {{
            operator: And
            operands: [
              {{ path:["scene_id"], operator: GreaterThanEqual, valueInt: {min_id} }}
              {{ path:["scene_id"], operator: LessThanEqual,    valueInt: {max_id} }}
            ]
          }}
          limit: {len(all_ids) + 5}
        ) {{
          scene_id scene_start text video_id
        }}
      }}
    }}
    """
    try:
        data = weaviate_graphql(gql)
        rows = data.get("Get", {}).get(SCENE_VEC_CLASS, []) or []
        # Only keep IDs we actually want
        return [r for r in rows if r.get("scene_id") in all_ids]
    except Exception:
        return []


def get_scene_segments(scene_id: int):
    gql = f"""
    {{
      Get {{
        {SCENE_CLASS}(
          where: {{ path:["scene_id"], operator: Equal, valueInt: {scene_id} }}
          limit: 1
        ) {{
          scene_start scene_end
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


# ── Scoring / utilities ───────────────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y; na += x * x; nb += y * y
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (math.sqrt(na) * math.sqrt(nb))


def best_timestamp_for_hit(hit: dict, query_vec: List[float]) -> int:
    try:
        _, _, segments = get_scene_segments(hit.get("scene_id"))
        if segments:
            seg_vecs = embed_texts([s.get("text", "") for s in segments])
            best_i, best_sim = 0, -1.0
            for i, v in enumerate(seg_vecs):
                sim = cosine_similarity(query_vec, v)
                if sim > best_sim:
                    best_sim = sim; best_i = i
            return int(round(float(segments[best_i].get("start") or hit.get("scene_start") or 0)))
    except Exception:
        pass
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
        if "youtube.com" in u.netloc or "youtu.be" in u.netloc:
            qs = parse_qs(u.query)
            qs["t"] = [str(max(0, int(seconds)))]
            return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(qs, doseq=True), u.fragment))
        return f"{url}#t={max(0, int(seconds))}"
    except Exception:
        return f"{url}#t={max(0, int(seconds))}"


def frame_url_for(rel_path: str) -> str:
    parts = [p for p in rel_path.split("/") if p]
    return "/frames/" + "/".join([requests.utils.quote(p) for p in parts])


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Video RAG", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    query: str
    limit: int = 5


class VideoUrlPayload(BaseModel):
    video_id: str
    url: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Video RAG</h1><p>UI missing. Put it at <code>web/index.html</code>.</p>")


@app.get("/frames/{relpath:path}")
def get_frame(relpath: str):
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
def read_video_urls():
    videos = get_all_videos()
    return {"videos": videos}


@app.post("/video-url")
def save_video_url(payload: VideoUrlPayload):
    try:
        upsert_video_meta(payload.video_id, payload.url)
        return {"ok": True, "video_id": payload.video_id, "url": payload.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    k = max(1, min(int(req.limit or 5), 20))

    try:
        q_vec = embed_texts([query])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        hits = search_scenevec(q_vec, k=max(k, 10))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Cache video URLs to avoid one Weaviate lookup per hit
    url_cache: dict = {}

    enriched = []
    for h in hits:
        vid_id = h.get("video_id")
        if vid_id not in url_cache:
            url_cache[vid_id] = get_video_url_for(vid_id)
        video_url = url_cache[vid_id]

        dist = h.get("_additional", {}).get("distance")
        sim  = None if dist is None else (1.0 - float(dist))
        fp   = h.get("frame_path") or ""
        ts   = best_timestamp_for_hit(h, q_vec)

        enriched.append({
            "text":        (h.get("text") or "").strip(),
            "frame_path":  fp,
            "frame_url":   frame_url_for(fp) if fp else None,
            "video_id":    vid_id,
            "scene_id":    h.get("scene_id"),
            "scene_start": h.get("scene_start"),
            "scene_end":   h.get("scene_end"),
            "duration":    h.get("duration"),
            "distance":    dist,
            "sim":         sim,
            "timestamp":   ts,
            "video_link":  video_link_at(video_url, ts) if video_url else None,
        })

    # Fetch neighboring scenes for the top 3 hits to capture procedural steps
    top_scene_ids = [h.get("scene_id") for h in hits[:3] if h.get("scene_id") is not None]
    neighbors = fetch_neighboring_scenes(top_scene_ids, window=2)
    neighbor_texts = sorted(
        [n for n in neighbors if n.get("text")],
        key=lambda x: x.get("scene_id") or 0
    )

    # LLM context = top hits + neighbors sorted by scene order
    all_context_scenes = {e["scene_id"]: e["text"] for e in enriched if e.get("text")}
    for n in neighbor_texts:
        all_context_scenes[n["scene_id"]] = n["text"]
    contexts = [all_context_scenes[sid] for sid in sorted(all_context_scenes.keys())]
    try:
        answer = chat_answer(contexts, query) if contexts else "(no context)"
    except Exception:
        answer = contexts[0] if contexts else "(no answer)"

    # Deduplicate by text to avoid showing identical cards
    seen_texts: set = set()
    deduped = []
    for e in enriched:
        key = (e.get("text") or "")[:120]
        if key not in seen_texts:
            seen_texts.add(key)
            deduped.append(e)
    enriched = deduped

    return JSONResponse({"answer": answer, "hits": enriched[:k]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_server:app", host="127.0.0.1", port=8000, reload=False)
