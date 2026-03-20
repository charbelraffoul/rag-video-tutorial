#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_ask.py

Console RAG query tool for your Scene/SceneVec dataset.

Features
- Embeds the user's query with Ollama embeddings
- Vector search in Weaviate (SceneVec) using nearVector (cosine distance)
- Pretty-prints top matches with image paths and "file://" links
- Optional: open top-N frame images in your default viewer/browser (--open N)
- Optional: open top-N *video* links at the best timestamp (--open-video N)
  * Video URL is read from Weaviate (class VideoMeta) if present,
    or from --video-url / $VIDEO_URL.
  * You can store/update the URL in Weaviate via --set-video-url <URL>.
- Generates a short answer using your local Ollama LLM (chat) with the top-K contexts.

Environment (optional)
- WEAVIATE_URL  (default: http://127.0.0.1:8080)
- OLLAMA_URL    (default: http://localhost:11434)
- EMBED_MODEL   (default: nomic-embed-text:latest)
- LLM_MODEL     (default: llama3.2:latest)
- ROOT_DIR      (default: current working directory; used to resolve frame_path)

Schema assumptions (already true in your setup)
- Class SceneVec (vectorizer none, hnsw cosine) with properties:
  scene_id(int), scene_start(number), scene_end(number), duration(number),
  frame_path(text), text(text).
- Class Scene (source) may contain nested "segments" with {start,end,text}.
  We use it to refine timestamps per-hit if available.

Test:
  py -3.12 .\rag_ask.py "How do I handle columns not connected to levels?" --open 3 --open-video 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ----------------------
# Config (from env)
# ----------------------
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080").rstrip("/")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama3.2:latest")
ROOT_DIR     = Path(os.getenv("ROOT_DIR", os.getcwd()))
VIDEO_URL_ENV = os.getenv("VIDEO_URL")  # optional fallback if not stored in Weaviate

# Weaviate constants
SCENE_VEC_CLASS = os.getenv("DST_CLASS", "SceneVec")
SCENE_CLASS     = os.getenv("SRC_CLASS", "Scene")

# Single fixed UUID for VideoMeta row (deterministic)
VIDEO_META_CLASS = "VideoMeta"
VIDEO_META_ID = "00000000-0000-0000-0000-000000000001"


# ----------------------
# HTTP helpers
# ----------------------
def _http_post(url: str, payload: dict, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"POST {url} -> {r.status_code} {r.text}")
    return r.json()


def _http_get(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.text}")
    return r.json()


def _http_put(url: str, payload: dict, timeout: int = 60, params: Optional[dict] = None) -> dict:
    r = requests.put(url, json=payload, timeout=timeout, params=params)
    if not r.ok:
        raise RuntimeError(f"PUT {url} -> {r.status_code} {r.text}")
    return r.json()


# ----------------------
# Ollama helpers
# ----------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama embeddings.
    Handles servers that return either:
      {"embeddings": [[...],[...]]}
    or (older/single) {"embedding": [...]}
    Falls back to per-item calls if batching returns a single vector.
    """
    if not texts:
        return []

    # Try batch first
    batch_payload = {"model": EMBED_MODEL, "input": texts}
    url = f"{OLLAMA_URL}/api/embeddings"
    r = requests.post(url, json=batch_payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    if "embeddings" in data and isinstance(data["embeddings"], list):
        # Proper batch
        return data["embeddings"]

    # If server returned a single "embedding" for the whole batch (seen on some builds)
    if "embedding" in data and isinstance(data["embedding"], list):
        # Fallback to per-item
        sys.stderr.write("[warn] Ollama returned a single 'embedding' for a batch; falling back to per-item requests.\n")
        out: List[List[float]] = []
        for t in texts:
            r2 = requests.post(url, json={"model": EMBED_MODEL, "input": t}, timeout=60)
            r2.raise_for_status()
            d2 = r2.json()
            if "embedding" not in d2:
                raise RuntimeError(f"Unexpected per-item embedding payload: {d2}")
            out.append(d2["embedding"])
        return out

    raise RuntimeError(f"Unexpected embeddings payload keys: {list(data.keys())}")


def chat_answer(context_snippets: List[str], question: str, max_tokens: int = 256) -> str:
    """
    Generate a concise answer using Ollama chat API (OpenAI-like).
    Keeps dependencies minimal (no openai pkg).
    """
    system_prompt = (
        "You are a helpful assistant. Answer the user's question succinctly, "
        "based ONLY on the provided context. If the context is insufficient, say you don't know."
    )
    context_block = "\n".join(f"- {s}" for s in context_snippets)

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer briefly:",
            },
        ],
        "stream": False,
        # Some Ollama builds accept extra options; we keep it simple/compatible.
    }
    url = f"{OLLAMA_URL}/api/chat"
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama /api/chat returns {"message":{"role":"assistant","content":"..."}, ...}
    # or OpenAI-like {"choices":[{"message":{"content":"..."}}], ...} for /v1/chat/completions
    if "message" in data and isinstance(data["message"], dict):
        return data["message"].get("content", "").strip()
    # Fallback in case it's routed through OpenAI-compatible endpoint
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()
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
    """
    Returns top-k objects from SceneVec with distance and properties.
    """
    vec_str = ",".join(str(x) for x in query_vec)
    gql = f"""
    {{
      Get {{
        {SCENE_VEC_CLASS}(nearVector:{{vector:[{vec_str}]}}, limit:{k}) {{
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
    """
    Fetch scene_start, scene_end and segments array for a given scene_id from source class Scene.
    Returns (scene_start, scene_end, segments[])
    """
    gql = f"""
    {{
      Get {{
        {SCENE_CLASS}(
          where: {{
            path: ["scene_id"]
            operator: Equal
            valueInt: {scene_id}
          }}
          limit: 1
        ) {{
          scene_start
          scene_end
          segments {{
            start
            end
            text
          }}
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
    """
    Ensure a simple VideoMeta class exists to store a single URL (property 'url').
    Uses PUT /v1/schema/<class> to avoid 405s on some setups.
    """
    # Check schema
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
    _http_put(f"{WEAVIATE_URL}/v1/schema/{VIDEO_META_CLASS}", body, timeout=30)


def set_video_url_in_weaviate(url_value: str) -> None:
    ensure_video_meta_class()
    body = {"class": VIDEO_META_CLASS, "id": VIDEO_META_ID, "properties": {"url": url_value}}
    # Upsert with PUT /v1/objects/{id}
    _http_put(f"{WEAVIATE_URL}/v1/objects/{VIDEO_META_ID}", body, timeout=30)


def get_video_url_from_weaviate() -> Optional[str]:
    try:
        gql = f'{{ Get {{ {VIDEO_META_CLASS}(limit:1) {{ url }} }} }}'
        data = weaviate_graphql(gql)
        arr = data.get("Get", {}).get(VIDEO_META_CLASS, []) or []
        if arr and arr[0].get("url"):
            return arr[0]["url"]
    except Exception:
        pass
    return None


# ----------------------
# Scoring / utility
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


def pick_best_timestamp_for_hit(
    hit: dict, query_vec: List[float], embed_fn=embed_texts
) -> int:
    """
    For a given top hit, try to find the most relevant segment time.
    1) If Scene.segments exists for this scene_id, embed each segment.text and
       choose the one with highest cosine similarity to query_vec. Return its start (rounded).
    2) Fallback to the scene_start.
    """
    scene_id = hit.get("scene_id")
    try:
        scene_start, scene_end, segments = get_scene_segments(scene_id)
        if segments:
            seg_texts = [s.get("text") or "" for s in segments]
            seg_vecs = embed_fn(seg_texts)
            best_idx = 0
            best_sim = -1.0
            for i, v in enumerate(seg_vecs):
                sim = cosine_similarity(query_vec, v)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            best_start = segments[best_idx].get("start") or scene_start
            return int(round(float(best_start)))
        return int(round(float(scene_start)))
    except Exception:
        # Any error -> fallback
        try:
            return int(round(float(hit.get("scene_start") or 0.0)))
        except Exception:
            return 0


def to_file_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except Exception:
        # Windows very old python guard
        return "file:///" + str(path.resolve()).replace("\\", "/")


def format_match_line(i: int, hit: dict) -> str:
    text = (hit.get("text") or "").strip().replace("\n", " ")
    if len(text) > 200:
        text = text[:197] + "..."
    frame_rel = hit.get("frame_path") or ""
    dist = hit.get("_additional", {}).get("distance")
    sim = None if dist is None else (1.0 - float(dist))
    sim_str = "n/a" if sim is None else f"{sim:.3f}"
    return (
        f" {i}. {text}\n"
        f"    image: {frame_rel}\n"
        f"    sim:   {sim_str}\n"
    )


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ask the RAG index a question; open frames and/or video at best timestamps."
    )
    parser.add_argument("query", nargs="?", help="Your question to the RAG index.")
    parser.add_argument("-k", "--limit", type=int, default=5, help="How many results to retrieve (default: 5).")
    parser.add_argument("--open", dest="open_count", type=int, default=0, help="Open top-N frame images.")
    parser.add_argument("--open-video", dest="open_video_count", type=int, default=0, help="Open top-N video links at the best timestamp.")
    parser.add_argument("--video-url", dest="video_url", type=str, default=None, help="Override video URL for this run (does not save).")
    parser.add_argument("--set-video-url", dest="set_video_url", type=str, default=None, help="Persist a video URL into Weaviate (VideoMeta). Then exit.")
    parser.add_argument("--print-links", action="store_true", help="Also print file:// image links and video links.")
    args = parser.parse_args()

    # Just store a URL and exit?
    if args.set_video_url:
        set_video_url_in_weaviate(args.set_video_url)
        print(f"Saved video URL in Weaviate ({VIDEO_META_CLASS}): {args.set_video_url}")
        return

    if not args.query:
        print("No query provided. Example:")
        print('  py -3.12 .\\rag_ask.py "How do I handle columns not connected to levels?" --open 3 --open-video 1')
        sys.exit(1)

    # Read/pick video URL for this run
    video_url: Optional[str] = None
    # precedence: --video-url > VideoMeta in Weaviate > $VIDEO_URL
    if args.video_url:
        video_url = args.video_url
    else:
        video_url = get_video_url_from_weaviate() or VIDEO_URL_ENV

    # 1) Embed query
    q_vec = embed_texts([args.query])[0]

    # 2) Retrieve from SceneVec
    hits = search_scenevec(q_vec, k=args.limit)

    # 3) Print matches
    print("Top matches:")
    for i, h in enumerate(hits, 1):
        # add file path if available
        frame_rel = h.get("frame_path") or ""
        frame_path = (ROOT_DIR / frame_rel).resolve()
        link_line = ""
        if args.print_links and frame_rel:
            link_line = f"    path:  {str(frame_path)}\n    link:  {to_file_uri(frame_path)}\n"
        print(format_match_line(i, h) + link_line)

    # 4) Open images if requested
    if args.open_count > 0:
        openN = min(args.open_count, len(hits))
        print(f"Opening top {openN} image(s)...")
        for h in hits[:openN]:
            frame_rel = h.get("frame_path")
            if not frame_rel:
                continue
            frame_path = (ROOT_DIR / frame_rel).resolve()
            if frame_path.exists():
                webbrowser.open(to_file_uri(frame_path))

    # 5) Open video links at best timestamps if requested
    if args.open_video_count > 0:
        if not video_url:
            print("No video URL known. Use --video-url <URL> for this run or --set-video-url <URL> to persist.")
        else:
            openN = min(args.open_video_count, len(hits))
            print(f"Opening top {openN} video timestamp link(s)...")
            for idx, h in enumerate(hits[:openN], 1):
                ts = pick_best_timestamp_for_hit(h, q_vec, embed_fn=embed_texts)
                # build a generic "#t=SS" anchor; many players honor it; if not, the link still opens the video.
                video_link = f"{video_url}#t={int(ts)}"
                print(f"  {idx}. t={ts:>5d}s -> {video_link}")
                webbrowser.open(video_link)

    # 6) Build a concise LLM answer using the top contexts
    contexts = [ (h.get("text") or "").strip() for h in hits[:max(1, args.limit)] ]
    print("\n---")
    print("Answer:\n")
    try:
        ans = chat_answer(contexts[:5], args.query)
    except Exception as e:
        # Fall back to a tiny heuristic if chat is unavailable
        ans = (contexts[0] if contexts else "").strip()
        if not ans:
            ans = "(no answer)"
        ans = ans if len(ans) < 500 else ans[:497] + "..."
        ans = f"{ans}\n\n[note] LLM generation unavailable ({e}). Shown the top context instead."
    print(ans)


if __name__ == "__main__":
    main()
