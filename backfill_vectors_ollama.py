#!/usr/bin/env python3
"""
backfill_vectors_ollama.py
- Reads all objects from a source Weaviate class (default: Scene)
- Uses Ollama embeddings (default model: nomic-embed-text:latest) to embed the 'text' field
- Writes new objects with attached vectors into a target class (default: SceneVec)
- Target class is created if it doesn't exist (vectorizer: none, cosine distance)

Env (optional):
  WEAVIATE_URL   default http://localhost:8080
  SRC_CLASS      default Scene
  DST_CLASS      default SceneVec
  OLLAMA_URL     default http://localhost:11434
  OLLAMA_EMBED   default nomic-embed-text:latest
  BATCH_SIZE     default 64
"""
import os, json, time, sys, requests

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
SRC_CLASS    = os.getenv("SRC_CLASS", "Scene")
DST_CLASS    = os.getenv("DST_CLASS", "SceneVec")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text:latest")
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "64"))

def ready() -> bool:
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
        return r.ok
    except Exception:
        return False

def get_schema():
    r = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=10)
    r.raise_for_status()
    return r.json()

def ensure_dst_class():
    schema = get_schema()
    if any(c.get("class") == DST_CLASS for c in schema.get("classes", [])):
        return

    body = {
        "class": DST_CLASS,
        "description": f"Vectorized copy of {SRC_CLASS} using Ollama embeddings",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "scene_id",    "dataType": ["int"]},
            {"name": "scene_start", "dataType": ["number"]},
            {"name": "scene_end",   "dataType": ["number"]},
            {"name": "duration",    "dataType": ["number"]},
            {"name": "frame_path",  "dataType": ["text"], "tokenization": "word"},
            {"name": "text",        "dataType": ["text"], "tokenization": "word"},
        ]
    }

    # Try modern endpoint first
    r = requests.post(f"{WEAVIATE_URL}/v1/schema/classes", json=body, timeout=20)
    if r.ok or r.status_code == 409:
        return

    # Some versions reject POST /v1/schema/classes with 405 – use PUT /v1/schema/{class}
    if r.status_code in (404, 405):
        r2 = requests.put(f"{WEAVIATE_URL}/v1/schema/{DST_CLASS}", json=body, timeout=20)
        if r2.ok or r2.status_code == 409:
            return
        raise SystemExit(f"Failed to create {DST_CLASS} (PUT): {r2.status_code} {r2.text}")

    raise SystemExit(f"Failed to create {DST_CLASS} (POST): {r.status_code} {r.text}")

def gql(query: str) -> dict:
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=30)
    r.raise_for_status()
    return r.json()

def count_src() -> int:
    q = f'{{ Aggregate {{ {SRC_CLASS} {{ meta {{ count }} }} }} }}'
    return gql(q)["data"]["Aggregate"][SRC_CLASS][0]["meta"]["count"]

def fetch_src_batch(limit=200, offset=0):
    q = f"""
    {{
      Get {{
        {SRC_CLASS}(limit:{limit}, offset:{offset}) {{
          scene_id scene_start scene_end duration frame_path text
          _additional {{ id }}
        }}
      }}
    }}
    """
    return gql(q)["data"]["Get"][SRC_CLASS]

def _embed_single(text: str):
    payload = {"model": OLLAMA_EMBED, "input": text if text else " "}
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "embedding" in data:
        return data["embedding"]
    if "embeddings" in data and len(data["embeddings"]) == 1:
        return data["embeddings"][0]
    raise RuntimeError(f"Unexpected single embedding payload keys: {list(data.keys())}")

def embed_texts(texts):
    """Return list of vectors (len == len(texts)).
       If Ollama returns a single 'embedding' for a batched input, fall back to per-item requests.
    """
    if isinstance(texts, str):
        texts = [texts]
    safe_inputs = [(t or " ").strip() for t in texts]

    # Try batched call
    try:
        payload = {"model": OLLAMA_EMBED, "input": safe_inputs}
        r = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise SystemExit(f"Error calling Ollama embeddings: {e}")

    if "embeddings" in data:
        vecs = data["embeddings"]
        if len(vecs) == len(safe_inputs):
            return vecs
        # Some builds return wrong length; fall back
        print(f"[warn] Batched embeddings length {len(vecs)} != inputs {len(safe_inputs)}; falling back to per-item.", flush=True)

    elif "embedding" in data:
        # Definitely a single vector — if batch size >1, we must loop
        if len(safe_inputs) == 1:
            return [data["embedding"]]
        print("[warn] Ollama returned a single 'embedding' for a batch; falling back to per-item requests.", flush=True)
    else:
        print(f"[warn] Unexpected batched payload keys {list(data.keys())}; falling back to per-item.", flush=True)

    # Per-item fallback (slower, but correct)
    vecs = []
    for t in safe_inputs:
        vecs.append(_embed_single(t))
    return vecs

def batch_insert(objs, vecs):
    url = f"{WEAVIATE_URL}/v1/batch/objects"
    batch = {"objects": []}
    for o, v in zip(objs, vecs):
        batch["objects"].append({
            "class": DST_CLASS,
            "properties": o,
            "vector": v,
        })
    r = requests.post(url, json=batch, timeout=120)
    if not r.ok:
        raise SystemExit(f"Batch insert error: {r.status_code} {r.text}")

def main():
    if not ready():
        raise SystemExit(f"Weaviate not ready at {WEAVIATE_URL}")

    # quick Ollama check
    try:
        t = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        t.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {OLLAMA_URL}: {e}")

    ensure_dst_class()

    total = count_src()
    print(f"Source {SRC_CLASS} count: {total}", flush=True)

    fetched = 0
    while fetched < total:
        rows = fetch_src_batch(limit=200, offset=fetched)
        if not rows:
            break

        props = []
        for r in rows:
            props.append({
                "scene_id": r.get("scene_id"),
                "scene_start": r.get("scene_start"),
                "scene_end": r.get("scene_end"),
                "duration": r.get("duration"),
                "frame_path": r.get("frame_path"),
                "text": (r.get("text") or "").strip(),
            })

        # embed in chunks
        for i in range(0, len(props), BATCH_SIZE):
            chunk_props = props[i:i+BATCH_SIZE]
            chunk_texts = [p["text"] for p in chunk_props]
            vecs = embed_texts(chunk_texts)

            if len(vecs) != len(chunk_props):
                raise SystemExit(f"Embedding count mismatch after fallback: got {len(vecs)} for {len(chunk_props)}")

            batch_insert(chunk_props, vecs)
            print(f"Inserted {fetched + i + len(chunk_props)}/{total}", flush=True)

        fetched += len(rows)

    print("Done.", flush=True)

if __name__ == "__main__":
    main()
