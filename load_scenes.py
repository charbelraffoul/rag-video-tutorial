#!/usr/bin/env python3
"""
Load scene JSON into Weaviate.

Usage:
  python load_scenes.py /path/to/scenes.jsonl --recreate
  python load_scenes.py /path/to/scenes.json --url http://localhost:8080

The input can be JSONL (one JSON object per line) or a JSON array.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Any, List

import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.exceptions import WeaviateBaseError


def connect(url: str):
    # Weaviate Python client v4
    client = weaviate.connect_to_local(host=url)
    return client


def ensure_schema(client, recreate: bool = False):
    collections = client.collections.list_all()
    have_scene = any(c.name == "Scene" for c in collections)
    if have_scene and recreate:
        client.collections.delete("Scene")

    if recreate or not have_scene:
        client.collections.create(
            name="Scene",
            vectorizer_config=Configure.Vectorizer.none(),  # no embeddings yet
            properties=[
                Property(name="scene_id",    data_type=DataType.INT),
                Property(name="scene_start", data_type=DataType.NUMBER),
                Property(name="scene_end",   data_type=DataType.NUMBER),
                Property(name="duration",    data_type=DataType.NUMBER),
                Property(name="frame_path",  data_type=DataType.TEXT),
                Property(name="text",        data_type=DataType.TEXT),
                Property(
                    name="segments",
                    data_type=DataType.OBJECT_ARRAY,
                    nested_properties=[
                        Property(name="start",           data_type=DataType.NUMBER),
                        Property(name="end",             data_type=DataType.NUMBER),
                        Property(name="text",            data_type=DataType.TEXT),
                        Property(name="overlap_seconds", data_type=DataType.NUMBER),
                    ],
                ),
            ],
        )


def coerce_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize types so Weaviate accepts them cleanly.
    """
    out = dict(obj)
    # ints / floats
    if "scene_id" in out:
        try:
            out["scene_id"] = int(out["scene_id"])
        except Exception:
            pass
    for k in ("scene_start", "scene_end", "duration"):
        if k in out:
            try:
                out[k] = float(out[k])
            except Exception:
                pass
    # normalize Windows backslashes to forward slashes for portability
    if "frame_path" in out and isinstance(out["frame_path"], str):
        out["frame_path"] = out["frame_path"].replace("\\\\", "/").replace("\\", "/")

    # segments normalization
    segs = out.get("segments")
    if isinstance(segs, list):
        cleaned = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            sc = dict(s)
            for k in ("start", "end", "overlap_seconds"):
                if k in sc:
                    try:
                        sc[k] = float(sc[k])
                    except Exception:
                        pass
            if "text" in sc and sc["text"] is None:
                sc["text"] = ""
            cleaned.append(sc)
        out["segments"] = cleaned

    return out


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield dicts from a JSONL file (one object per line) or a JSON array file.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        return
    # else: JSONL
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSON on line {i}: {e}") from e
        if isinstance(obj, dict):
            yield obj


def batch_ingest(client, recs: Iterator[Dict[str, Any]], batch_size: int = 200):
    scenes = client.collections.get("Scene")
    count = 0
    try:
        with scenes.batch.dynamic() as batch:
            for obj in recs:
                props = coerce_record(obj)
                batch.add_object(properties=props)
                count += 1
    except WeaviateBaseError as e:
        raise SystemExit(f"Weaviate error during batch ingest: {e}") from e
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Path to scenes.jsonl or scenes.json")
    ap.add_argument("--url", type=str, default=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                    help="Weaviate URL (default: http://localhost:8080)")
    ap.add_argument("--recreate", action="store_true", help="Drop and recreate the Scene collection")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    client = connect(args.url)
    try:
        ensure_schema(client, recreate=args.recreate)
        count = batch_ingest(client, iter_records(path))
        print(f"Ingested {count} Scene objects into Weaviate at {args.url}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
