#!/usr/bin/env python3
"""
3_align_and_upload.py

Aligns Whisper transcript segments to visual scenes, then uploads
the merged Scene objects to Weaviate tagged with a video_id.

Usage:
  python 3_align_and_upload.py --video-id my-video-slug

Args:
  --video-id     Short stable identifier for this video (e.g. revit-01). Required.
  --video-url    YouTube (or other) URL for this video. Saved to Weaviate so the UI can link to it.
  --scenes-json  Path to scenes.json (default: scenes_<video-id>.json or scenes.json)
  --segments     Path to transcript_segments.jsonl (default: transcript_segments_<video-id>.jsonl or transcript_segments.jsonl)
  --out          Path to write video_manifest.jsonl (default: video_manifest.jsonl)

Env:
  WEAVIATE_URL   default http://localhost:8080
"""

import argparse
import json
import os
import requests
from pathlib import Path

WEAVIATE_URL      = os.getenv("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
SCENE_CLASS       = "Scene"
VIDEO_META_CLASS  = "VideoMeta"

MIN_OVERLAP  = 0.20  # keep a segment if it overlaps a scene by >= 200 ms


# ── Alignment ────────────────────────────────────────────────────────────────

def load_scenes(path):
    with open(path, encoding="utf-8") as f:
        scenes = json.load(f)
    scenes.sort(key=lambda x: (x["start"], x["end"]))
    return scenes


def load_segments(path):
    segs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segs.append(json.loads(line))
    segs.sort(key=lambda x: (x["start"], x["end"]))
    return segs


def overlap(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def align(scenes, segs, video_id):
    records = []
    j = 0
    for sc in scenes:
        s, e = float(sc["start"]), float(sc["end"])
        while j < len(segs) and segs[j]["end"] <= s:
            j += 1
        k, attached = j, []
        while k < len(segs) and segs[k]["start"] < e:
            sg = segs[k]
            ov = overlap(sg["start"], sg["end"], s, e)
            if ov >= MIN_OVERLAP:
                attached.append({
                    "start": sg["start"], "end": sg["end"],
                    "text": sg["text"].strip(), "overlap_seconds": ov
                })
            k += 1
        attached.sort(key=lambda x: x["start"])
        records.append({
            "video_id":    video_id,
            "scene_id":    sc["scene_id"],
            "scene_start": s,
            "scene_end":   e,
            "duration":    round(e - s, 3),
            "frame_path":  sc["frame_path"].replace("\\", "/"),
            "text":        " ".join(x["text"] for x in attached).strip(),
            "segments":    attached,
        })
    return records


# ── Weaviate helpers ──────────────────────────────────────────────────────────

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


def ensure_scene_class():
    schema = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=10).json()
    if any(c.get("class") == SCENE_CLASS for c in schema.get("classes", [])):
        ensure_property(SCENE_CLASS, "video_id", ["text"])
        return
    body = {
        "class": SCENE_CLASS,
        "vectorizer": "none",
        "properties": [
            {"name": "video_id",    "dataType": ["text"],     "tokenization": "field",
             "indexSearchable": True, "indexFilterable": True},
            {"name": "scene_id",    "dataType": ["int"]},
            {"name": "scene_start", "dataType": ["number"]},
            {"name": "scene_end",   "dataType": ["number"]},
            {"name": "duration",    "dataType": ["number"]},
            {"name": "frame_path",  "dataType": ["text"],     "tokenization": "word"},
            {"name": "text",        "dataType": ["text"],     "tokenization": "word"},
            {"name": "segments",    "dataType": ["object[]"],
             "nestedProperties": [
                 {"name": "start",           "dataType": ["number"]},
                 {"name": "end",             "dataType": ["number"]},
                 {"name": "text",            "dataType": ["text"], "tokenization": "word"},
                 {"name": "overlap_seconds", "dataType": ["number"]},
             ]},
        ],
    }
    r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=body, timeout=20)
    if r.status_code not in (200, 201, 409):
        raise SystemExit(f"Failed to create Scene class: {r.status_code} {r.text}")


def upload(records):
    ensure_scene_class()
    batch = {"objects": []}
    for rec in records:
        batch["objects"].append({
            "class": SCENE_CLASS,
            "properties": {
                "video_id":    rec["video_id"],
                "scene_id":    rec["scene_id"],
                "scene_start": rec["scene_start"],
                "scene_end":   rec["scene_end"],
                "duration":    rec["duration"],
                "frame_path":  rec["frame_path"],
                "text":        rec["text"],
                "segments":    rec["segments"],
            }
        })
    r = requests.post(f"{WEAVIATE_URL}/v1/batch/objects", json=batch, timeout=120)
    if not r.ok:
        raise SystemExit(f"Batch upload failed: {r.status_code} {r.text}")
    results = r.json()
    errors = [x for x in results if x.get("result", {}).get("errors")]
    if errors:
        print(f"[warn] {len(errors)} objects had upload errors.")
    print(f"Uploaded {len(records) - len(errors)} Scene objects for video_id='{records[0]['video_id']}'.")


# ── VideoMeta ─────────────────────────────────────────────────────────────────

def save_video_url(video_id: str, url: str) -> None:
    """Upsert a VideoMeta object so the server can build timestamped links."""
    import uuid as _uuid
    # Ensure class exists
    schema = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=10).json()
    if not any(c.get("class") == VIDEO_META_CLASS for c in schema.get("classes", [])):
        body = {
            "class": VIDEO_META_CLASS,
            "vectorizer": "none",
            "properties": [
                {"name": "video_id", "dataType": ["text"], "tokenization": "field",
                 "indexSearchable": True, "indexFilterable": True},
                {"name": "url",      "dataType": ["text"], "tokenization": "word"},
            ],
        }
        requests.post(f"{WEAVIATE_URL}/v1/schema", json=body, timeout=20)

    obj_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"video-meta-{video_id}"))
    payload = {"class": VIDEO_META_CLASS, "id": obj_id,
               "properties": {"video_id": video_id, "url": url}}
    r = requests.put(f"{WEAVIATE_URL}/v1/objects/{obj_id}", json=payload, timeout=30)
    if not r.ok:
        requests.post(f"{WEAVIATE_URL}/v1/objects", json=payload, timeout=30)
    print(f"Saved video URL for '{video_id}'.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id",  required=True,
                        help="Short stable identifier for this video (e.g. revit-01)")
    parser.add_argument("--video-url", default=None,
                        help="YouTube or other URL for this video — enables timestamped links in the UI")
    parser.add_argument("--scenes-json", default=None)
    parser.add_argument("--segments",    default=None)
    parser.add_argument("--out",         default="video_manifest.jsonl")
    args = parser.parse_args()

    vid = args.video_id
    scenes_json = args.scenes_json or (f"scenes_{vid}.json"                  if Path(f"scenes_{vid}.json").exists()                  else "scenes.json")
    segments    = args.segments    or (f"transcript_segments_{vid}.jsonl"    if Path(f"transcript_segments_{vid}.jsonl").exists()    else "transcript_segments.jsonl")

    if not Path(segments).exists():
        raise SystemExit(f"Missing {segments}. Run 1_transcribe.py first.")
    if not Path(scenes_json).exists():
        raise SystemExit(f"Missing {scenes_json}. Run 2_scenes.py first.")

    if args.video_url:
        save_video_url(vid, args.video_url)

    scenes  = load_scenes(scenes_json)
    segs    = load_segments(segments)
    records = align(scenes, segs, vid)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out} with {len(records)} rows.")

    upload(records)


if __name__ == "__main__":
    main()
