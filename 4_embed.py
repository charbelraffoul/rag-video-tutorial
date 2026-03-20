#!/usr/bin/env python3
"""
4_embed.py

Reads Scene objects from Weaviate, embeds each scene's text with Ollama,
and stores the resulting vectors in a SceneVec collection.

Usage:
  python 4_embed.py --video-id my-video-slug   # embed only this video's scenes
  python 4_embed.py                             # embed all scenes

Env:
  WEAVIATE_URL   default http://localhost:8080
  SRC_CLASS      default Scene
  DST_CLASS      default SceneVec
  OLLAMA_URL     default http://localhost:11434
  OLLAMA_EMBED   default nomic-embed-text:latest
  BATCH_SIZE     default 64
"""

import argparse
import os
import requests

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
SRC_CLASS    = os.getenv("SRC_CLASS",  "Scene")
DST_CLASS    = os.getenv("DST_CLASS",  "SceneVec")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text:latest")
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "64"))


# ── Schema helpers ────────────────────────────────────────────────────────────

def ensure_property(class_name, prop_name, data_type):
    """Add a property to an existing class only if it is not already present."""
    r = requests.get(f"{WEAVIATE_URL}/v1/schema/{class_name}", timeout=10)
    if not r.ok:
        return
    existing = [p["name"] for p in r.json().get("properties", [])]
    if prop_name in existing:
        return
    body = {"name": prop_name, "dataType": data_type,
            "tokenization": "field", "indexSearchable": True, "indexFilterable": True}
    r2 = requests.post(f"{WEAVIATE_URL}/v1/schema/{class_name}/properties", json=body, timeout=20)
    if r2.status_code not in (200, 201, 409):
        raise SystemExit(f"Failed to add '{prop_name}' to {class_name}: {r2.status_code} {r2.text}")


def ensure_dst_class():
    schema = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=10).json()
    if any(c.get("class") == DST_CLASS for c in schema.get("classes", [])):
        ensure_property(DST_CLASS, "video_id", ["text"])
        return

    body = {
        "class": DST_CLASS,
        "description": f"Vectorized copy of {SRC_CLASS} using Ollama embeddings",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "video_id",    "dataType": ["text"], "tokenization": "field",
             "indexSearchable": True, "indexFilterable": True},
            {"name": "scene_id",    "dataType": ["int"]},
            {"name": "scene_start", "dataType": ["number"]},
            {"name": "scene_end",   "dataType": ["number"]},
            {"name": "duration",    "dataType": ["number"]},
            {"name": "frame_path",  "dataType": ["text"], "tokenization": "word"},
            {"name": "text",        "dataType": ["text"], "tokenization": "word"},
        ]
    }
    r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=body, timeout=20)
    if r.ok or r.status_code == 409:
        return
    raise SystemExit(f"Failed to create {DST_CLASS}: {r.status_code} {r.text}")


# ── Weaviate queries ──────────────────────────────────────────────────────────

def gql(query):
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=30)
    r.raise_for_status()
    return r.json()


def _where(video_id):
    if not video_id:
        return ""
    safe = video_id.replace('"', '')
    return f'where: {{ path:["video_id"], operator: Equal, valueText: "{safe}" }}'


def count_src(video_id=None):
    w = _where(video_id)
    q = f'{{ Aggregate {{ {SRC_CLASS}({w}) {{ meta {{ count }} }} }} }}'
    return gql(q)["data"]["Aggregate"][SRC_CLASS][0]["meta"]["count"]


def fetch_src_batch(limit=200, offset=0, video_id=None):
    w = _where(video_id)
    sep = ", " if w else ""
    q = f"""
    {{
      Get {{
        {SRC_CLASS}(limit:{limit}, offset:{offset}{sep}{w}) {{
          video_id scene_id scene_start scene_end duration frame_path text
          _additional {{ id }}
        }}
      }}
    }}
    """
    return gql(q)["data"]["Get"][SRC_CLASS]


# ── Ollama embeddings ─────────────────────────────────────────────────────────

def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]
    safe = [(t or " ").strip() for t in texts]
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": OLLAMA_EMBED, "input": safe}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data and len(data["embeddings"]) == len(safe):
        return data["embeddings"]
    raise SystemExit(f"Unexpected /api/embed response: {list(data.keys())}")


# ── Weaviate batch insert ─────────────────────────────────────────────────────

def batch_insert(props, vecs):
    url = f"{WEAVIATE_URL}/v1/batch/objects"
    batch = {"objects": [
        {"class": DST_CLASS, "properties": p, "vector": v}
        for p, v in zip(props, vecs)
    ]}
    r = requests.post(url, json=batch, timeout=120)
    if not r.ok:
        raise SystemExit(f"Batch insert error: {r.status_code} {r.text}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default=None,
                        help="Embed only scenes for this video_id. Omit to embed all.")
    args = parser.parse_args()
    video_id = args.video_id

    if not requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=5).ok:
        raise SystemExit(f"Weaviate not ready at {WEAVIATE_URL}")
    if not requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).ok:
        raise SystemExit(f"Ollama not reachable at {OLLAMA_URL}")

    ensure_dst_class()

    total = count_src(video_id=video_id)
    label = f"video_id='{video_id}'" if video_id else "all videos"
    print(f"Source {SRC_CLASS} count ({label}): {total}", flush=True)

    fetched = 0
    while fetched < total:
        rows = fetch_src_batch(limit=200, offset=fetched, video_id=video_id)
        if not rows:
            break

        props = [{
            "video_id":    r.get("video_id"),
            "scene_id":    r.get("scene_id"),
            "scene_start": r.get("scene_start"),
            "scene_end":   r.get("scene_end"),
            "duration":    r.get("duration"),
            "frame_path":  r.get("frame_path"),
            "text":        (r.get("text") or "").strip(),
        } for r in rows]

        for i in range(0, len(props), BATCH_SIZE):
            chunk_props = props[i:i + BATCH_SIZE]
            chunk_texts = [p["text"] for p in chunk_props]
            vecs = embed_texts(chunk_texts)
            if len(vecs) != len(chunk_props):
                raise SystemExit(f"Embedding count mismatch: got {len(vecs)} for {len(chunk_props)}")
            batch_insert(chunk_props, vecs)
            print(f"Inserted {fetched + i + len(chunk_props)}/{total}", flush=True)

        fetched += len(rows)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
